import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchtext
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import random
import numpy as np
import argparse

from toxic_comment.model import Net
from toxic_comment.load_data import *


parser = argparse.ArgumentParser(description='Toxic comment Classification')
parser.add_argument("--vocab-size", type=int, default=20000, metavar='N',
                    help="vocab size")
parser.add_argument("--embedding-dim", type=int, default=100, metavar='N',
                    help="embedding dim")
parser.add_argument("--len-sentence", type=int, default=100, metavar='N',
                    help="sequence length's limit")
parser.add_argument("--num-labels", type=int, default=6, metavar='N',
                    help="label num")
parser.add_argument("--min-freq", type=int, default=1, metavar='N',
                    help="min_freq")
parser.add_argument("--channel-size", type=int, default=128, metavar='N',
                    help="channel size of cnn")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--valid_num', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--n-gram', type=int, default=5, metavar='N',
                    help='n_gram')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--dropout', type=float, default=0.5, metavar='LR',
                    help='dropout rate(default: 0.5)')
parser.add_argument('--x2-size', type=int, default=1, metavar='S',
                    help='x2-size (default: 1)')
parser.add_argument('--valid-num', type=int, default=10000, metavar='S',
                    help='valid_num (default: 10000)')


parser.add_argument("--valid", type=float, default=0.02, metavar='RATIO',
                    help="valid set's ratio")
config = parser.parse_args()


torch.cuda.manual_seed_all(config.seed)
torch.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)
torch.backends.cudnn.deterministic=True

saved = torch.load('./saved_state.tch')
tk_train = saved['tk_train']
tk_test = saved['tk_test']
tk_cap_ratio_train = saved['tk_cap_ratio_train']
tk_cap_ratio_test = saved['tk_cap_ratio_test']
y_labels = saved['y_labels']
x2 = saved['x2']

tk_train, x2, tk_cap_ratio_train, y_labels = shuffle_lists(tk_train, x2, tk_cap_ratio_train, y_labels)

tk_cap_ratio_train = np.array([padding_cap_ratio(row, config.len_sentence) for row in tk_cap_ratio_train])

tk_valid = tk_train[-config.valid_num:]
tk_cap_ratio_valid = tk_cap_ratio_train[-config.valid_num:]
x2_valid = x2[-config.valid_num:]
y_valid = y_labels[-config.valid_num:]

print ("valid : ", len(tk_valid), len(tk_cap_ratio_valid), len(x2_valid), len(y_valid))

tk_train = tk_train[:-config.valid_num]
y_train = y_labels[:-config.valid_num]
x2_train = x2[:-config.valid_num]
tk_cap_ratio_train = tk_cap_ratio_train[:-config.valid_num]

print("train : ", len(tk_train), len(tk_cap_ratio_train), len(x2_train), len(y_train))

from torchtext import data, datasets

TEXT = data.Field(sequential=True,
                  # 들어갈 데이터가 sequential 인가요? 우리는 tokenize한 word의 sequence를 다룰거니까 True입니다. Defualt로도 True임.
                  tokenize=tokenizer,
                  # 그 데이터를 tokenize할 함수를 지정할 수 있습니다. 우리는 gensim library의 tokenize 함수를 쓸건데요
                  # 뭐 굳이 그거 말고도 직접 정의해도 되고 str.split 같은걸 써넣어도 됩니다.
                  # :: 그런 줄 알았는데 아무 tokenize 함수나 쓰면 안되고, generator가 아닌 tokenized list 를 반환하는 함수여야합니다..
                  # :::: 이게 아닐거같기도 함.
                  fix_length=config.len_sentence,
                 # 아마 tokenize된 길이 제한 같은데 한번 확인해볼게요. 특이사항으로는 length 넘으면 자르고, 안넘으면 padding을 채웁니다
                  # :: 그게 아니고 vector화 했을 때의 길이 제한일 것 같아요. 확인해보겠습니다.
                  pad_first=True,
                  # padding이 앞에서부터 붙냐, 뒤에서부터 붙냐는 겁니다.
                  tensor_type=torch.cuda.LongTensor
                  # cuda를 써도 됩니다
                 )

TEXT.build_vocab(tk_train, tk_valid, max_size=config.vocab_size, min_freq=config.min_freq)

net = Net(vocab_size=config.vocab_size, embedding_dim=config.embedding_dim, len_sentence=config.len_sentence,
         x2_size=config.x2_size, channel_size=config.channel_size, dropout=config.dropout, num_labels=config.num_labels, batch_size=config.batch_size).cuda()

optimizer = optim.Adam(net.parameters())
criterion = nn.BCELoss()

from tqdm import tqdm


def validation(net, tk_valid, tk_cap_ratio_valid, x2_valid, y_valid, TEXT, batch_size, criterion):
    net.train(False)
    val_score = None
    valid_loss = 0
    for val_step, (batch_val, cr_val, x2_val, y_val) in enumerate(
            batchify(tk_valid, tk_cap_ratio_valid, x2_valid.values, y_valid.values, batch_size=batch_size)):
        var_x = TEXT.process(batch_val, device=0, train=False).transpose(dim0=0, dim1=1)
        var_cr = Variable(torch.cuda.FloatTensor(cr_val))
        var_y = Variable(torch.cuda.FloatTensor(y_val))
        var_x2 = Variable(torch.cuda.FloatTensor(x2_val))
        pred_score = net(var_x, var_cr, var_x2)
        if val_score is None:
            val_score = pred_score.data
        else:
            val_score = torch.cat([val_score, pred_score.data])
        y_loss = criterion(pred_score, var_y)
        valid_loss += y_loss.data[0]
    net.train(True)
    valid_loss /= len(tk_valid) / batch_size

    return valid_loss, val_score


def prediction(net, tk_test, tk_cap_ratio_test, x2_test, TEXT, batch_size):
    net.train(False)
    val_score = None
    for val_step, (batch_test, cr_test, x2_test) in enumerate(
            batchify(tk_test, tk_cap_ratio_test, x2_test.values, batch_size=batch_size)):
        var_x = TEXT.process(batch_test, device=0, train=False).transpose(dim0=0, dim1=1)
        var_cr = Variable(torch.cuda.FloatTensor(cr_test))
        var_x2 = Variable(torch.cuda.FloatTensor(x2_test))
        pred_score = net(var_x, var_cr, var_x2)
        if val_score is None:
            val_score = pred_score.data
        else:
            val_score = torch.cat([val_score, pred_score.data])
    net.train(True)

    return val_score


def train(net, tk_train, tk_cap_ratio_train, x2_train, y_train, TEXT , batch_size, criterion):
    net.train(True)
    for step, (x_train_, cr_train_, x2_train_, y_train_) in tqdm(enumerate(
            batchify(tk_train, tk_cap_ratio_train, x2_train.values, y_train.values, batch_size=batch_size))):
        var_x = TEXT.process(x_train_, device=0, train=True).transpose(dim0=0, dim1=1)
        var_cr = Variable(torch.cuda.FloatTensor(cr_train_))
        var_y = Variable(torch.cuda.FloatTensor(y_train_))
        var_x2 = Variable(torch.cuda.FloatTensor(x2_train_))
        pred_score = net(var_x, var_cr, var_x2)

        net.zero_grad()

        y_loss = criterion(pred_score, var_y)
        y_loss.backward()
        optimizer.step()


from sklearn.metrics import roc_auc_score

def save_config(checkpoint, is_max):
    cf = checkpoint['config']
    score = checkpoint['score']
    epoch = checkpoint['epoch']
    if is_max:
        # torch.save(checkpoint, './max_cf' + str(score))
        with open("./history", "a") as f:
            f.write(str(config) + " epoch %s: "%(epoch) + str(score) + '\n')
    else:
        # torch.save(checkpoint, './cf' + str(score))
        pass


maxscore = 0.5

for epoch in range(config.epochs):
    train(net, tk_train, tk_cap_ratio_train, x2_train, y_train, TEXT, config.batch_size, criterion)
    valid_loss, val_score = validation(net, tk_valid, tk_cap_ratio_valid, x2_valid, y_valid, TEXT, config.batch_size, criterion)
    rocauc = roc_auc_score(y_valid.values, val_score.cpu().numpy())
    checkpoint = dict({
        'config' : config,
        'score' : rocauc,
        'epoch' : epoch,
    })
    save_config(checkpoint, rocauc > maxscore)
    maxscore = max(maxscore, rocauc)
    print("valid loss", valid_loss, "score :", rocauc)