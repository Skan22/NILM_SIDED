import torch
import torch.nn as nn
import math

# 1. Shared TCN Components

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalLayerNorm(nn.Module):
    def __init__(self, num_features):
        super(TemporalLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        # x: (batch, channels, time)
        x = x.transpose(1, 2) # (batch, time, channels)
        x = self.norm(x)
        return x.transpose(1, 2) # (batch, channels, time)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, causal=True):
        super(TemporalBlock, self).__init__()
        
        # Branch 1
        layers1 = []
        layers1.append(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        if causal:
            layers1.append(Chomp1d(padding))
        layers1.append(TemporalLayerNorm(n_outputs)) # Changed to TemporalLayerNorm
        layers1.append(nn.ReLU())
        layers1.append(nn.Dropout(dropout))
        self.net1 = nn.Sequential(*layers1)

        # Branch 2
        layers2 = []
        layers2.append(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        if causal:
            layers2.append(Chomp1d(padding))
        layers2.append(TemporalLayerNorm(n_outputs)) # Changed to TemporalLayerNorm
        layers2.append(nn.ReLU())
        layers2.append(nn.Dropout(dropout))
        self.net2 = nn.Sequential(*layers2)


        # Skip Connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net1(x)
        out = self.net2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNBackbone(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2, causal=True):
        super(TCNBackbone, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            if causal:
                # Causal padding: we pad (k-1)*d to the left, then chomp it off
                padding = (kernel_size - 1) * dilation_size
            else:
                # Non-causal padding: (k-1)*d / 2 on both sides (assuming odd kernel)
                # This keeps sequence length same without chomp
                padding = ((kernel_size - 1) * dilation_size) // 2
                
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=padding, dropout=dropout, causal=causal))
            
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 2. Models

class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels=[64]*6, kernel_size=3, dropout=0.33, output_size=5, causal=False):
        super(TCNModel, self).__init__()
        self.tcn = TCNBackbone(input_size, num_channels, kernel_size, dropout, causal=causal)
        self.fc = nn.Linear(num_channels[-1], output_size)
        self.causal = causal
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1) # -> (batch, features, seq_len)
        y = self.tcn(x)        # -> (batch, channels, seq_len)
        
        if self.causal:
            # Predict using last step
            out = y[:, :, -1]
        else:
            # Predict using middle step (for midpoint target)
            mid = y.size(2) // 2
            out = y[:, :, mid]
            
        return self.fc(out)

class ATCNModel(nn.Module):
    def __init__(self, input_size, num_channels=[64]*6, kernel_size=3, dropout=0.33, output_size=5, causal=False):
        super(ATCNModel, self).__init__()
        self.tcn = TCNBackbone(input_size, num_channels, kernel_size, dropout, causal=causal)
        
        # Multi-Head Attention
        # embed_dim must be divisible by num_heads
        self.embed_dim = num_channels[-1]
        self.num_heads = 4 # Default to 4 heads
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)
        
        self.fc = nn.Linear(self.embed_dim, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)      # -> (batch, features, seq_len)
        features = self.tcn(x)      # -> (batch, channels, seq_len)
        features = features.permute(0, 2, 1) # -> (batch, seq_len, channels)
        
        # Multi-Head Attention
        # query, key, value are all 'features' for self-attention
        # attn_output: (batch, seq_len, embed_dim)
        attn_output, _ = self.mha(features, features, features)
        
        # Global Average Pooling or Weighted Sum
        # In typical NILM attention papers, we sum weighted features. 
        # Here MHA returns contextualized features. We can pool them.
        # Alternatively, use the middle token if non-causal?
        # A common powerful method: Soft pooling / Attention weighted sum
        # But MHA outputs a sequence. Let's start with Mean Pooling for stability
        context = attn_output.mean(dim=1)
        
        return self.fc(context)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=5, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=0.2,
                           bidirectional=bidirectional)
        
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x) # (batch, seq_len, hidden_size * num_directions)
        
        if self.bidirectional:
            # Take the middle point for prediction if using midpoint target
            # However, typically people concatenate the last forward and first backward?
            # For "midpoint" NILM, the state at index t contains info from 0..t(fwd) and T..t(bwd)
            mid = lstm_out.size(1) // 2
            out = lstm_out[:, mid, :]
        else:
            # Causal: Last step
            out = lstm_out[:, -1, :]
            
        return self.fc(out)

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3, output_size=5, dropout=0.1, max_len=500):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Predict using middle token
        mid = x.size(1) // 2
        out = x[:, mid, :]
        return self.fc(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)