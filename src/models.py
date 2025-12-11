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
    """Temporal Convolutional Network Backbone
    
    Args:
        causal: If True, uses causal convolutions (only past context).
                If False, uses non-causal convolutions with symmetric padding,
                providing bidirectional context (past + future) like BiLSTM.
    """
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2, causal=True):
        super(TCNBackbone, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            if causal:
                # Causal padding: pad (k-1)*d to the left, then chomp it off
                # Only uses past context (unidirectional)
                padding = (kernel_size - 1) * dilation_size
            else:
                # Non-causal (bidirectional) padding: (k-1)*d / 2 on both sides
                # This provides symmetric context from past AND future, like BiLSTM
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
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Robust initialization for TCN"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1) # -> (batch, features, seq_len)
        y = self.tcn(x)        # -> (batch, channels, seq_len)
        
        if self.causal:
            # Causal: Use last timestep (only past context available)
            out = y[:, :, -1]
        else:
            # Non-causal (bidirectional): Use middle timestep
            # At this point, each position has seen both past AND future context
            # through the symmetric (non-causal) convolutions, similar to BiLSTM
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

class ImprovedATCNModel(nn.Module):
    """
    Attention-enhanced TCN with improved temporal aggregation
    Changes from original:
    1. Midpoint selection instead of mean pooling
    2. Layer normalization after attention
    3. Residual connection to preserve TCN features
    """
    def __init__(self, input_size, num_channels=[128]*6, kernel_size=3, 
                 dropout=0.33, output_size=1, causal=False, num_heads=2):
        super().__init__()
        self.tcn = TCNBackbone(input_size, num_channels, kernel_size, dropout, causal=causal)
        
        # Multi-Head Attention
        self.embed_dim = num_channels[-1]
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(self.embed_dim)
        
        self.fc = nn.Linear(self.embed_dim, output_size)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Robust initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        mid_idx = seq_len // 2
        
        # TCN features
        x = x.permute(0, 2, 1)      # -> (batch, features, seq_len)
        features = self.tcn(x)      # -> (batch, channels, seq_len)
        features = features.permute(0, 2, 1)  # -> (batch, seq_len, channels)
        
        # Multi-Head Self-Attention
        attn_output, attn_weights = self.mha(features, features, features)
        
        # Layer normalization
        attn_output = self.norm(attn_output)
        
        # IMPROVEMENT 1: Use midpoint instead of mean pooling
        # This aligns with the midpoint target position
        context = attn_output[:, mid_idx, :]
        
        # IMPROVEMENT 2: Optional residual connection
        # Preserves TCN features + adds attention refinement
        tcn_mid = features[:, mid_idx, :]
        context = context + tcn_mid  # Residual connection
        
        return self.fc(context)