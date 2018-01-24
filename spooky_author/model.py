import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict


class Model(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 hidden_size,
                 linear_size,
                 num_class,
                 nlayers=1,
                 dropout=0.3):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=nlayers, batch_first=True, dropout=dropout)
        for name, param in self.lstm.named_parameters():
            if len(param.size()) == 2:
                nn.init.orthogonal(param)
            else:
                nn.init.constant(param, 1)
        self.sequential = nn.Sequential(
            OrderedDict([
                ('lin1', nn.Linear(hidden_size, linear_size)),
                ('tanh', nn.ReLU()),
                ('dropout', nn.Dropout(dropout)),
                ('lin2', nn.Linear(linear_size, num_class)),
            ]))
        self.count = None

    def forward(self, inputs):

        embeds = self.embedding(inputs)
        lstm, (h, _) = self.lstm(embeds)

        last = lstm[:, -1, :]

        # if self.count is None:
        #     print(inputs.shape, embeds.shape, lstm.shape)
        #     self.count = 1

        output = self.sequential(last)
        return output