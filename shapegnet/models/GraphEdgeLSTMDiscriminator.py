from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn


# Graph Edge LSTM Discriminator
class GraphEdgeLSTMDiscriminator(nn.Module):
    def __init__(self, h_size, y_size):
        super(GraphEdgeLSTMDiscriminator, self).__init__()
        # one layer MLP
        self.discriminator_output = nn.Sequential(
            nn.Linear(h_size + y_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, h, y):
        return self.discriminator_output(torch.cat((h, y), dim=2))
