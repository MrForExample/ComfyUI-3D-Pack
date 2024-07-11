import torch.nn as nn
import torch.nn.functional as F


class SdfMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.fc3 = nn.Linear(hidden_dim, 4, bias=bias)


    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class RgbMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.fc3 = nn.Linear(hidden_dim, 3, bias=bias)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return out

    