import torch
import torch.nn as nn
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1536):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

# Transformer Encoder Model
class TransformerECGClassifier(nn.Module):
    def __init__(self, input_dim=12, seq_len=1536, d_model=128, n_heads=8, num_layers=4, num_classes=9):
        super(TransformerECGClassifier, self).__init__()
        # self.embedding = nn.Linear(input_dim, d_model)  # Embedding for input leads
        self.embedding = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        )
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)
        x = self.embedding(x)  # Shape: (batch_size, seq_len, d_model)
        x = self.positional_encoding(x)  # Add positional encoding
        x = x.permute(1, 0, 2)  # Shape: (seq_len, batch_size, d_model) for Transformer
        x = self.transformer_encoder(x)  # Shape: (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2).contiguous().view(x.size(1), -1)  # Flatten
        return self.classifier(x)

# Example Usage
# model = TransformerECGClassifier().to("cuda")
# input_data = torch.rand(32, 1536, 12).to("cuda")  # Batch of size 32
# output = model(input_data)
# print(output.shape)  # Output shape: (32, 9)