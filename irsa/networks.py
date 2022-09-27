import torch
from torch import nn
import torch.nn.functional as F


class PairedNeuralNet(nn.Module):
    def __init__(self):
        super(PairedNeuralNet, self).__init__()
        
        # Conv1d(input_channels, output_channels, kernel_size)
        self.conv1 = nn.Conv1d(1, 64, 10) 
        self.conv2 = nn.Conv1d(64, 128, 7)  
        self.conv3 = nn.Conv1d(128, 128, 4)
        self.conv4 = nn.Conv1d(128, 256, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc_out = nn.Linear(4096, 1)

        self.sigmoid = nn.Sigmoid()
    
    def convs(self, x):
        # out_dim = in_dim - kernel_size + 1  
        # 1, 105, 105
        x = F.relu(self.bn1(self.conv1(x)))
        # 64, 96, 96
        x = F.max_pool1d(x, (2, 2))
        # 64, 48, 48
        x = F.relu(self.bn2(self.conv2(x)))
        # 128, 42, 42
        x = F.max_pool1d(x, (2, 2))
        # 128, 21, 21
        x = F.relu(self.bn3(self.conv3(x)))
        # 128, 18, 18
        x = F.max_pool1d(x, (2, 2))
        # 128, 9, 9
        x = F.relu(self.bn4(self.conv4(x)))
        # 256, 6, 6
        return x

    def forward(self, x1, x2):
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256 * 6 * 6)
        x1 = self.sigmoid(self.fc1(x1))

        x2 = self.convs(x2)
        x2 = x2.view(-1, 256 * 6 * 6)
        x2 = self.sigmoid(self.fc1(x2))

        x = torch.abs(x1 - x2)
        x = self.fc_out(x)

        return x


class VGGPairedNeuralNet(nn.Module):
    def __init__(self):
        super(VGGPairedNeuralNet, self).__init__()

        self.conv11 = nn.Conv1d(1, 64, 3) 
        self.conv12 = nn.Conv1d(64, 64, 3)  
        self.conv21 = nn.Conv1d(64, 128, 3)
        self.conv22 = nn.Conv1d(128, 128, 3)
        self.conv31 = nn.Conv1d(128, 256, 3) 
        self.conv32 = nn.Conv1d(256, 256, 3)  
        self.conv33 = nn.Conv1d(256, 256, 3)

        self.pool = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(256 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc_out = nn.Linear(4096, 1)

        self.sigmoid = nn.Sigmoid()

    def convs(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.max_pool1d(x, (2, 2))
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.max_pool1d(x, (2, 2))
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv33(x))
        x = F.max_pool1d(x, (2, 2))
        return x

    def forward(self, x1, x2):
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256 * 8 * 8)
        x1 = self.fc1(x1)
        x1 = self.sigmoid(self.fc2(x1))

        x2 = self.convs(x2)
        x2 = x2.view(-1, 256 * 8 * 8)
        x2 = self.fc1(x2)
        x2 = self.sigmoid(self.fc2(x2))

        x = torch.abs(x1 - x2)
        x = self.fc_out(x)

        return x
