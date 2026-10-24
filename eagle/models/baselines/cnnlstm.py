from torch import nn

class CNNLSTMModel(nn.Module):
    def __init__(self, input_channels=12, cnn_channels=[32, 64, 128], kernel_size=5, lstm_hidden_size=64, num_classes=9):
        super(CNNLSTMModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_channels[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
            # nn.MaxPool1d(2),

            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
            # nn.MaxPool1d(2),

            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.ReLU(),
            # nn.MaxPool1d(2),
        )

        self.lstm = nn.GRU(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
            # nn.Sigmoid()  # For multi-label classification
        )

    def forward(self, x):
        # x: (batch_size, 12, N)
        x = self.cnn(x)  # -> (batch, C, T)
        x = x.permute(0, 2, 1)  # -> (batch, T, C)
        x, _ = self.lstm(x)     # -> (batch, T, 2*hidden)
        x = x[:, -1, :]         # Take last time step
        out = self.fc(x)        # -> (batch, num_classes)
        return out