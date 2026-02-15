import torch.nn as nn

class TriageNet(nn.Module):
    def __init__(self):
        super(TriageNet, self).__init__()
        # Input: 9 features (Age, HR, O2, etc.) -> Output: 4 classes
        self.fc = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4) # Classes: 0, 1, 2, 3
        )

    def forward(self, x):
        return self.fc(x)