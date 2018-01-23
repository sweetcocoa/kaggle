import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict


class Model(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 hidden_size,
                 linear_size,
                 nlayers=1):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=nlayers, batch_first=True)
        self.sequential = nn.Sequential(
            OrderedDict([
                ('lin1', nn.Linear(hidden_size, linear_size)),
                ('relu', nn.ReLU()),
                ('lin2', nn.Linear(linear_size, 3)),
            ]))

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        lstm, (h, _) = self.lstm(embeds)
        last = lstm[:, -1, :]
        output = self.sequential(last)
        return output