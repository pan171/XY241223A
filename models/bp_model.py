import torch
import torch.nn as nn

class BPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device='cpu'):
        super(BPModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size).to(device)
        self.layer2 = nn.Linear(hidden_size, output_size).to(device)
        self.device = device
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        self.layer1 = self.layer1.to(self.device)
        self.layer2 = self.layer2.to(self.device)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x
