import torch
import torch.nn as nn
import torch.nn.functional as F
import math # For PositionalEncoding

# --- 1. CNN Module (No change in definition, but how it's used changes) ---
class CNNModule(nn.Module):
    def __init__(self, input_channels=1, num_filters=[64, 128, 256], kernel_sizes=[7, 5, 3], pool_sizes=[2, 2, 2]): # Default input_channels=1
        super(CNNModule, self).__init__()
        self.layers = nn.ModuleList()
        
        current_channels = input_channels # This will be 12 for 12-lead ECG
        for i in range(len(num_filters)):
            self.layers.append(
                nn.Conv1d(in_channels=current_channels, 
                          out_channels=num_filters[i], 
                          kernel_size=kernel_sizes[i], 
                          padding='same') 
            )
            self.layers.append(nn.BatchNorm1d(num_filters[i]))
            self.layers.append(nn.ReLU())
            if pool_sizes[i] > 1:
                self.layers.append(nn.MaxPool1d(kernel_size=pool_sizes[i], stride=pool_sizes[i]))
            current_channels = num_filters[i]
            
    def forward(self, x):
        # x shape: (batch_size, input_channels, sequence_length)
        # For 12-lead ECG, input_channels will be 12
        for layer in self.layers:
            x = layer(x)
        return x # shape: (batch_size, last_num_filters, reduced_sequence_length)

# --- Positional Encoding (Often needed for Transformers) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0: # Handle odd d_model if necessary, though typically even
             pe[:, 1::2] = torch.cos(position * div_term)[:,:d_model // 2]
        else:
             pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- RNN Module (Bi-LSTM example - no change in definition) ---
class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(RNNModule, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        
    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)
        return outputs

# --- Transformer Encoder Module (no change in definition) ---
class TransformerEncoderModule(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoderModule, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=num_encoder_layers)
        
    def forward(self, src):
        output = self.transformer_encoder(src)
        return output

# --- Classification Head (no change in definition) ---
class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=None):
        super(ClassificationHead, self).__init__()
        layers = []
        if hidden_size:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            layers.append(nn.Linear(hidden_size, num_classes))
        else:
            layers.append(nn.Linear(input_size, num_classes))
        
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.fc(x)

# --- CRT-Net Main Model (Updated for clarity with 12 leads) ---
class CRTNet(nn.Module):
    def __init__(self, 
                 # CNN params
                 cnn_input_channels=12, # MODIFIED: Default to 12 for 12-lead
                 cnn_num_filters=[64, 128], 
                 cnn_kernel_sizes=[7, 5], 
                 cnn_pool_sizes=[2, 2],
                 # RNN params
                 rnn_hidden_size=128, 
                 rnn_num_layers=2,
                 # Transformer params
                 transformer_d_model=256, 
                 transformer_nhead=8, 
                 transformer_num_encoder_layers=3, 
                 transformer_dim_feedforward=512,
                 # Classifier params
                 num_classes=5, 
                 dropout_rate=0.1,
                 use_positional_encoding=True): # Added flag for PE
        super(CRTNet, self).__init__()
        
        self.cnn_module = CNNModule(input_channels=cnn_input_channels, # This will be 12
                                    num_filters=cnn_num_filters,
                                    kernel_sizes=cnn_kernel_sizes,
                                    pool_sizes=cnn_pool_sizes)
        
        cnn_output_channels = cnn_num_filters[-1] 
        
        self.rnn_module = RNNModule(input_size=cnn_output_channels, 
                                    hidden_size=rnn_hidden_size, 
                                    num_layers=rnn_num_layers,
                                    dropout=dropout_rate)
        
        current_transformer_input_dim = rnn_hidden_size * 2
        
        self.rnn_to_transformer_projection = nn.Identity()
        if current_transformer_input_dim != transformer_d_model:
            self.rnn_to_transformer_projection = nn.Linear(current_transformer_input_dim, transformer_d_model)
            current_transformer_input_dim = transformer_d_model


        self.use_positional_encoding = use_positional_encoding
        if self.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(transformer_d_model, dropout_rate)

        self.transformer_module = TransformerEncoderModule(d_model=transformer_d_model, 
                                                           nhead=transformer_nhead, 
                                                           num_encoder_layers=transformer_num_encoder_layers, 
                                                           dim_feedforward=transformer_dim_feedforward,
                                                           dropout=dropout_rate)
        
        self.classification_head = ClassificationHead(input_size=transformer_d_model, 
                                                      num_classes=num_classes)
        
    def forward(self, x):
        # x: (batch_size, num_leads (e.g., 12), sequence_length)
        
        # Ensure input has channel dimension if it's (batch, seq_len) for single lead
        if x.ndim == 2 and self.cnn_module.layers[0].in_channels == 1: # (batch_size, seq_len)
             x = x.unsqueeze(1) # (batch_size, 1, seq_len)
        # If x is (batch_size, num_leads, seq_len) it's already correct for cnn_input_channels > 1

        cnn_out = self.cnn_module(x) # (batch_size, cnn_output_channels, reduced_seq_len)
        
        rnn_input = cnn_out.permute(0, 2, 1) # (batch_size, reduced_seq_len, cnn_output_channels)
        rnn_out = self.rnn_module(rnn_input) # (batch_size, reduced_seq_len, rnn_hidden_size * 2)
        
        transformer_input_projected = self.rnn_to_transformer_projection(rnn_out) # (batch_size, reduced_seq_len, transformer_d_model)
        
        if self.use_positional_encoding:
            transformer_input_pe = self.pos_encoder(transformer_input_projected)
        else:
            transformer_input_pe = transformer_input_projected

        transformer_out = self.transformer_module(transformer_input_pe) # (batch_size, reduced_seq_len, transformer_d_model)
        
        pooled_output = transformer_out.mean(dim=1) # Global Average Pooling
        
        logits = self.classification_head(pooled_output) # (batch_size, num_classes)
        
        return logits