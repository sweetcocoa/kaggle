from spooky_author.model import *
from spooky_author.load_data import *
from spooky_author.config import config
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import argparse
import os
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random
from tqdm import tqdm


def validate(
        net,
        valid_dataloader,
        criterion,
        cuda,
):

    valid_correct = 0
    valid_loss = 0

    net.eval()

    for i, (sentense, author) in enumerate(valid_dataloader):
        if cuda:
            var_x, var_y = Variable(sentense).cuda(), Variable(author).cuda()
        else:
            var_x, var_y = Variable(sentense), Variable(author)

        pred_score = net(var_x)
        y_loss = criterion(pred_score, var_y)
        valid_loss += y_loss.data[0]
        _, pred = pred_score.max(dim=1)
        valid_correct += (pred == var_y).float().sum()

    acc = valid_correct.data[0] / len(valid_dataloader.dataset)
    valid_loss /= len(valid_dataloader.dataset) / valid_dataloader.batch_size
    return valid_loss, acc


def train(
        net,
        train_dataloader,
        criterion,
        optimizer,
        cuda,
):
    net.train(True)
    train_correct = 0
    train_loss = 0

    for i, (sentence, author) in enumerate(train_dataloader):
        if cuda:
            var_x, var_y = Variable(sentence).cuda(), Variable(author).cuda()
        else:
            var_x, var_y = Variable(sentence), Variable(author)

        pred_score = net(var_x)

        _, pred = pred_score.max(dim=1)
        train_correct += (pred == var_y).float().sum()
        y_loss = criterion(pred_score, var_y)
        train_loss += y_loss.data[0]

        optimizer.zero_grad()
        y_loss.backward()
        optimizer.step()


    acc = train_correct.data[0] / len(train_dataloader.dataset)
    train_loss /= len(train_dataloader.dataset) / train_dataloader.batch_size
    return train_loss, acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Spooky Author Classification')
    parser.add_argument('--batch-size', type=int, default=48, metavar='N',
                        help='input batch size for training (default: 48)')
    parser.add_argument('--test-batch-size', type=int, default=48, metavar='N',
                        help='input batch size for testing (default: 48)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument("--truncate", type=int, default=64, metavar='N',
                        help="sequence length's limit")
    parser.add_argument("--valid", type=float, default=0.2, metavar='RATIO',
                        help="valid set's ratio")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        dtype = torch.cuda.LongTensor
        torch.cuda.manual_seed(args.seed)
    else:
        dtype = torch.LongTensor

    train_data = load_raw(os.path.join("data", "train.csv"))
    encoded_authors = encode_authors(train_data)

    tokenized_texts = train_data['text'].str.lower().apply(tokenize)

    vocab = get_vocab_set(tokenized_texts)
    word_to_ix = {word : i + 1 for i, word in enumerate(vocab)}

    tokenized_ix = []
    for tokenized_text in tokenized_texts:
        tokenized_ix.append(tokens_to_ix(word_to_ix, tokenized_text, fixed_length=args.truncate))
    tokenized_ix = np.array(tokenized_ix)

    tokenized_ix, encoded_authors = shuffle_x_y(tokenized_ix, encoded_authors)

    train_x, train_y, valid_x, valid_y = divide_validation_set(tokenized_ix, encoded_authors, args.valid)
    train_dataset = TensorDataset(dtype(train_x), dtype(train_y))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = TensorDataset(dtype(valid_x), dtype(valid_y))
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)


    if args.cuda:
        net = Model(vocab_size= len(vocab) + 2,  # why add 2 :: one for zero-padding, one for unseen words
                    embedding_dim=config.EMBEDDING_DIM,
                    hidden_size=config.HIDDEN_SIZE,
                    linear_size=config.LINEAR_SIZE,
                    nlayers=config.NLAYERS).cuda()
    else:
        net = Model(vocab_size=len(vocab) + 2,
                    embedding_dim=config.EMBEDDING_DIM,
                    hidden_size=config.HIDDEN_SIZE,
                    linear_size=config.LINEAR_SIZE,
                    nlayers=config.NLAYERS)

    criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)


    best_acc = 0

    for epoch in tqdm(range(args.epochs)):
        train_loss, train_acc = train(net,
              train_dataloader,
              criterion,
              optimizer,
              args.cuda,)

        if epoch % 20 == 19:
            valid_loss, valid_acc = validate(net,
                     valid_dataloader,
                     criterion,
                     args.cuda)

            print("train loss :", train_loss,
                  "train acc :", train_acc,
                  "valid loss :", valid_loss,
                  "valid acc :", valid_acc,
                  )

            is_best = valid_acc > best_acc
            best_acc = max(valid_acc, best_acc)
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'best_acc' : best_acc,
                    'optimizer': optimizer.state_dict(),
                    'word_to_ix': word_to_ix,
                }, is_best
            )

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        torch.save(state, "best_" + filename)

if __name__ == "__main__":
    main()