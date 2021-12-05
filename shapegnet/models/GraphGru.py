from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.init as init
from rich import pretty
from rich.console import Console
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

pretty.install()


# plain GRU model
class GraphGRU(nn.Module):
    """

    """
    def __init__(self, input_size, batch_size, embedding_size, hidden_size, num_layers,
                 has_input=True, has_output=False,
                 output_size=None, bidirectional=False, device='cpu', verbose=False):
        """

        """

        if verbose:
            self.console = Console(width=128)
            style = "bold white on blue"
            self.console.print(locals())

        super(GraphGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output
        self.batch_size = batch_size
        self.device = device

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True).to(device)
        else:
            self.rnn = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first=True).to(device)

        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            ).to(device)

        self.relu = nn.ReLU().to(device)
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                gain = nn.init.calculate_gain('sigmoid')
                nn.init.xavier_uniform_(param, gain=gain).to(device)

        # init gains for xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                relu_gain = nn.init.calculate_gain('relu')
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=relu_gain).to(device)

        self.zero_tensor = Variable(torch.zeros(self.num_layers, batch_size,
                                                self.hidden_size)).to(self.device)

    def init_hidden(self, batch_size):
        # return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda(),
        #         Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda())

        # print("hidden size", batch_size, self.hidden_size)
        if batch_size != self.batch_size:
            print("batch mismatch")

        return self.zero_tensor

    # def _weights_init(m):
    #     if isinstance(m, nn.Conv2d or nn.Linear or nn.GRU or nn.LSTM):
    #         init.xavier_normal_(m.weight)
    #         m.bias.data.zero_()
    #     elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
    #         m.weight.data.fill_(1)
    #         m.bias.data.zero_()

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            cell_input = self.input(input_raw)
            cell_input = self.relu(cell_input)
        else:
            cell_input = input_raw
        # packed
        if pack:
            cell_input = pack_padded_sequence(cell_input, input_len, batch_first=True)

        output_raw, self.hidden = self.rnn(cell_input, self.hidden)

        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)

        # return hidden state at each time step
        return output_raw
