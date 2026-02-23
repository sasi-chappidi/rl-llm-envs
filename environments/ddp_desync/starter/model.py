import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in: int = 32, d_hidden: int = 128, d_out: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)