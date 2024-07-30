
import torch.nn as nn

class LinearProb(nn.Module):
    def __init__(self, n_classes = 6):
        super().__init__()
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        logits = self.linear(x)
        return {'logits': logits}