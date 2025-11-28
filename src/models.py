import torch
import torch.nn as nn

# Temporal Convolutional Network (TCN) Components

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels=[128]*8, kernel_size=3, dropout=0.33, output_size=5):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                       stride=1, dilation=dilation_size,
                                       padding=padding, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        x = self.network(x)
        x = x.mean(dim=2)  # Global average pooling
        return self.fc(x)

# Attention Mechanism
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        attention_weights = self.attention(x)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted = x * attention_weights
        return weighted.sum(dim=1)  # (batch, hidden_size)

# Attention + TCN Model (ATCN)
class ATCNModel(nn.Module):
    def __init__(self, input_size, num_channels=[128]*8, kernel_size=3, dropout=0.33, output_size=5):
        super(ATCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                       stride=1, dilation=dilation_size,
                                       padding=padding, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        self.attention = AttentionLayer(num_channels[-1])
        self.fc = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        x = self.network(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        x = self.attention(x)  # Apply attention
        return self.fc(x)







class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=0.2)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, (_h_n, _c_n) = self.lstm(x)
        # Use the last hidden state
        out = self.fc(lstm_out[:, -1, :])
        return out
