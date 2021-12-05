from __future__ import unicode_literals, print_function, division

import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
    """
    """
    def __init__(self, h_size, embedding_size, y_size, device='cpu'):
        super(MLP, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, y_size)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data,
                                                     gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        return self.deterministic_output(h)
