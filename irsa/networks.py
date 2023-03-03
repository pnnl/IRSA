import torch
import torch.nn.functional as F
from torch import nn


class DomainEncoder(nn.Module):
    def __init__(self, embedding_dim=2048):
        super(DomainEncoder, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, 21, padding='same')
        torch.nn.init.xavier_normal_(self.conv1.weight)

        self.conv2 = nn.Conv1d(32, 32, 21, padding='same')
        torch.nn.init.xavier_normal_(self.conv2.weight)

        self.conv3 = nn.Conv1d(32, 64, 21, padding='same')
        torch.nn.init.xavier_normal_(self.conv3.weight)

        self.conv4 = nn.Conv1d(64, 64, 21, padding='same')
        torch.nn.init.xavier_normal_(self.conv4.weight)

        # Corresponding batch normalization layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)

        # Fully conected output layer
        self.fc_out = nn.Linear(64 * 51, embedding_dim)

    def convs(self, x):
        # Conv - batchnorm - activate - pool - dropout block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = F.dropout1d(x, 0.5)

        # Conv - batchnorm - activate - pool - dropout block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = F.dropout1d(x, 0.5)

        # Conv - batchnorm - activate - pool - dropout block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = F.dropout1d(x, 0.5)

        # Conv - batchnorm - activate - pool - dropout block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = F.dropout1d(x, 0.5)

        return x

    def forward(self, x):
        # Convolutional layers
        x = self.convs(x)

        # Flatten
        x = x.view(-1, 64 * 51)

        # Fully connected output
        x = F.relu(self.fc_out(x))

        return x


class PairedNeuralNet(nn.Module):
    def __init__(self, embedding_dim=2048):
        super(PairedNeuralNet, self).__init__()

        # Domain encoders
        self.domain_embed1 = DomainEncoder(embedding_dim=embedding_dim)
        self.domain_embed2 = DomainEncoder(embedding_dim=embedding_dim)

        # Batch norm
        self.bn1 = nn.BatchNorm1d(embedding_dim)

        # Fully connected layers
        self.fc_out = nn.Linear(embedding_dim, 1)

    def forward(self, x1, x2):
        # Embed first input
        x1 = self.domain_embed1(x1)

        # Embed second input
        x2 = self.domain_embed2(x2)

        # "Learned distance function" starts here
        # Difference vector
        x = torch.abs(x1 - x2)

        # Batch norm
        x = self.bn1(x)

        # Fully connected output
        x = self.fc_out(x)

        # Activation will be handled by loss function
        return x
