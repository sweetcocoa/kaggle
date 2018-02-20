import torch
import torch.nn as nn
from collections import OrderedDict


class LSTMNet(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 len_sentence,
                 padding_idx=1,
                 lstm_hidden=128,
                 lstm_dropout=0.1,
                 fcn_hidden=256
                 ):
        super(LSTMNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.len_sentence = len_sentence
        self.padding_idx = padding_idx
        self.lstm_hidden = lstm_hidden
        self.lstm_dropout = lstm_dropout
        self.fcn_hidden = fcn_hidden

        self.embedding = nn.Embedding(self.vocab_size + 2, embedding_dim=self.embedding_dim,
                                      padding_idx=self.padding_idx)
        self.lstm = nn.LSTM(input_size=self.embedding_dim + 1, hidden_size=self.lstm_hidden, num_layers=1,
                            batch_first=True, dropout=self.lstm_dropout)
        self.global_pool = nn.MaxPool2d((self.len_sentence, 1))
        self.fcn = nn.Sequential(
            OrderedDict([
                ('linear1', nn.Linear(self.lstm_hidden + 1, self.fcn_hidden)),
                ('dropout1', nn.Dropout(self.lstm_dropout, inplace=True)),
                ('relu1', nn.ReLU(inplace=True)),
                ('linear2', nn.Linear(self.fcn_hidden, 6)),
                ('sigmoid', nn.Sigmoid())
            ]))

    def forward(self, *input):
        var_x = input[0]
        var_cr = input[1]
        var_x2 = input[2]
        embed = self.embedding(var_x)
        cr = var_cr.unsqueeze(2)
        embed_cr = torch.cat([embed, cr], dim=2)
        out, (h, _) = self.lstm(embed_cr)
        pool_out = self.global_pool(out)
        pool_out.squeeze_(1)
        linear_input = torch.cat([pool_out, var_x2], dim=1)
        output = self.fcn(linear_input)
        return output



class Net(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 len_sentence,
                 channel_size=4,
                 x2_size=1, # additional data - cap ratio
                 fc_dim=128,
                 padding_idx=1,
                 dropout=0.3,
                 num_labels=7,
                 batch_size=32,
                 is_cuda=False,
                 n_gram=5,
                 additional_kernel_size=1,
                 ):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size +2, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.channel_size = channel_size
        self.len_sentence = len_sentence
        self.batch_size = batch_size
        self.x2_size = x2_size
        self.kernel_size = (n_gram, embedding_dim + additional_kernel_size)
        self.n_gram = n_gram
        self.additional_kernel_size = additional_kernel_size
        self.conv2d = nn.Conv2d(1, out_channels=channel_size, kernel_size=self.kernel_size, stride=1)
        # output : batch x channel x (len_sentence - 2) x 1

        # -> squeeze : batch x channel x (len_sentence - 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1d = nn.Dropout(p=dropout)
        self.pool1d = nn.AvgPool1d(kernel_size=2)
        # output : batch x channel x (len_sentence - 2) / 2

        self.bottleneck_size = channel_size * (len_sentence - (self. n_gram -1)) / 2
        assert self.bottleneck_size.is_integer()
        self.bottleneck_size = int(self.bottleneck_size) + self.x2_size

        self.fcn1 = nn.Linear(self.bottleneck_size, fc_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fcn2 = nn.Linear(fc_dim, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.fc_dim = fc_dim
        self.num_labels = num_labels

    def forward(self, sentence, cap_ratio, other_features):
        image = self.embedding(sentence)

        # batch x channel x sentence_length x embedding
        cap_ratio.unsqueeze_(2)
        new_image = torch.cat([image, cap_ratio], dim=2)
        new_image.unsqueeze_(1)
        bottleneck = self.conv2d(new_image)
        bottleneck.squeeze_(3)
        bottleneck = self.relu(bottleneck) # batch x channel x features
        bottleneck = self.dropout1d(bottleneck)
        bottleneck = self.pool1d(bottleneck)

        bottleneck = bottleneck.view(-1, self.bottleneck_size - self.x2_size)
        if self.x2_size > 0:
            bottleneck = torch.cat([bottleneck, other_features], dim=1)

        fcn = self.relu1(self.fcn1(bottleneck))
        fcn = self.fcn2(fcn)
        logit = self.sigmoid(fcn)

        return logit