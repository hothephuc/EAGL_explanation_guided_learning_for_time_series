import torch
from torch import nn

class PCCRNN(nn.Module):
    def __init__(self, num_classes=4):
        super(PCCRNN, self).__init__()
        
        # Top Branch: LSTM for temporal features
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        
        # Bottom Branch: CNN for spatial features
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.1)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.1)
        
        # Adaptive pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fusion and Dense Layers
        self.fc1 = nn.Linear(64 + 64, 128)  # 64 from LSTM, 64 from CNN
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout_fc = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Input x is expected to be a 2D tensor (batch_size, height, width)
        # Add a channel dimension
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, height, width)
        
        # Bottom Branch: CNN
        x_cnn = self.relu(self.bn1(self.conv1(x)))
        x_cnn = self.maxpool1(x_cnn)
        x_cnn = self.dropout1(x_cnn)
        
        x_cnn = self.relu(self.bn2(self.conv2(x_cnn)))
        x_cnn = self.maxpool2(x_cnn)
        x_cnn = self.dropout2(x_cnn)
        
        x_cnn = self.relu(self.bn3(self.conv3(x_cnn)))
        x_cnn = self.maxpool3(x_cnn)
        x_cnn = self.dropout3(x_cnn)
        
        # Adaptive pooling to get a fixed-size output
        x_cnn = self.avgpool(x_cnn)
        x_cnn = torch.flatten(x_cnn, 1)  # Flatten to (batch_size, 64)
        
        # Top Branch: LSTM
        # Reshape input for LSTM (batch_size, sequence_length, input_size)
        x_lstm = x.squeeze(1)  # Remove channel dimension
        x_lstm = x_lstm.permute(0, 2, 1)  # Shape: (batch_size, width, height)
        
        x_lstm, _ = self.lstm1(x_lstm)
        x_lstm, _ = self.lstm2(x_lstm)
        x_lstm, _ = self.lstm3(x_lstm)
        x_lstm = x_lstm[:, -1, :]  # Take the last time step output (batch_size, 64)
        
        # Feature Fusion
        x = torch.cat((x_lstm, x_cnn), dim=1)  # Concatenate LSTM and CNN outputs
        x = self.fc1(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
