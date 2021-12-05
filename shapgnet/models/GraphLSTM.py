# Model Configurator
# Mustafa B

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class GraphLSTM(nn.Module):
    """

    """
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 has_input=True, has_output=False, output_size=None, device='cpu'):

        super(GraphLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output
        self.device = device

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.LSTM(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param, gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda(),
                Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda())

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            cell_input = self.input(input_raw)
            cell_input = self.relu(cell_input)
        else:
            cell_input = input_raw
        if pack:
            cell_input = pack_padded_sequence(cell_input, input_len, batch_first=True)

        output_raw, self.hidden = self.rnn(cell_input, self.hidden)

        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw


# current baseline model, generating a graph by lstm
class GraphGeneratorLSTM(nn.Module):
    """

    """
    def __init__(self, feature_size, input_size, hidden_size, output_size, batch_size, num_layers):
        """

        """
        super(GraphGeneratorLSTM, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.linear_input = nn.Linear(feature_size, input_size)
        self.linear_output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # initialize
        # self.hidden,self.cell = self.init_hidden()
        self.hidden = self.init_hidden()

        self.lstm.weight_ih_l0.data = init.xavier_uniform(self.lstm.weight_ih_l0.data,
                                                          gain=nn.init.calculate_gain('sigmoid'))
        self.lstm.weight_hh_l0.data = init.xavier_uniform(self.lstm.weight_hh_l0.data,
                                                          gain=nn.init.calculate_gain('sigmoid'))
        self.lstm.bias_ih_l0.data = torch.ones(self.lstm.bias_ih_l0.data.size(0)) * 0.25
        self.lstm.bias_hh_l0.data = torch.ones(self.lstm.bias_hh_l0.data.size(0)) * 0.25
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda(),
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda())

    def forward(self, input_raw, pack=False, len=None):
        input = self.linear_input(input_raw)
        input = self.relu(input)
        if pack:
            input = pack_padded_sequence(input, len, batch_first=True)
        output_raw, self.hidden = self.lstm(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        output = self.linear_output(output_raw)
        return output
