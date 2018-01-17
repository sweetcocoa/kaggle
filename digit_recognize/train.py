import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils import data
import argparse

# Training settings
parser = argparse.ArgumentParser(description='Digit Recognizer')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)





class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.convnet = nn.Sequential(OrderedDict(
            [
                ('layer 1', nn.Conv2d(1, 10, 5)),
                # ('layer 1', nn.Conv2d(1, 3, 5)),
                ('relu 1', nn.ReLU()), # 10 * 24 * 24
                ('maxpool 1', nn.MaxPool2d(2, stride=2)), # 3 * 12 * 12
                ('dropout 1', nn.Dropout2d(p=0.2)),
                ('batchnorm 1', nn.BatchNorm2d(10)),
                ('layer 2', nn.Conv2d(10, 20, kernel_size=5)), # 5 * 8 * 8
                # ('layer 2', nn.Conv2d(3, 5, kernel_size=5)),  # 5 * 8 * 8
                ('relu 2', nn.ReLU()),
                ('maxpool 2', nn.MaxPool2d(2, stride=2)), # 5 * 4 * 4
                ('dropout 2', nn.Dropout2d(p=0.2)),
            ]
        ))
        self.fcn = nn.Sequential(OrderedDict(
        [
            ('linear 1', nn.Linear(320, 150)),
            # ('linear 1', nn.Linear(80, 160)),
            ('relu 3', nn.ReLU()),
            ('linear 2', nn.Linear(150, 10)),
        ]))

    def forward(self, x):
        conv = self.convnet(x)
        lin = conv.view(-1, 320)
        return self.fcn(lin)


def load_data():
    original_data = np.genfromtxt('train.csv', delimiter=',', dtype=np.float32)
    images = original_data[1:, 1:]
    labels = original_data[1:, 0]
    labels_onehot = np.eye(10)[labels.astype(np.int)]
    images_normalized = (images - images.mean()) / images.std()
    return images, labels_onehot


images, labels_onehot = load_data()

if args.cuda:
    net = DigitCNN().cuda()
    ft = torch.cuda.FloatTensor
else:
    net = DigitCNN()
    ft = torch.FloatTensor

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)


def divide_validset(cross_validation_index, images, labels_onehot):
    """
    :param cross_validation_index:
    :return:
    train_images, train_labels, valid_images, valid_labels
    """
    total_num = len(images)
    valid_ratio = 0.2
    valid_num = int(total_num * valid_ratio)
    for i, idx in enumerate(range(0, total_num, valid_num)):
        if i == cross_validation_index:
            to_idx = min(idx+valid_num, total_num)
            valid_images = images[idx:to_idx]
            valid_labels = labels_onehot[idx:to_idx]
            train_images = np.concatenate((images[:idx], images[to_idx:]), axis=0)
            train_labels = np.concatenate((labels_onehot[:idx], labels_onehot[to_idx:]), axis=0)
            break

    return train_images, train_labels, valid_images, valid_labels



cross_validation_index = 4
train_images, train_labels, valid_images, valid_labels = divide_validset(cross_validation_index, images, labels_onehot)


train_dataset = data.TensorDataset(ft(train_images), ft(train_labels))
train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


mnist_dataloader = data.DataLoader(
    datasets.MNIST('./', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
                   ),
    batch_size=args.batch_size, shuffle=True
)

def train(net):
    net.train(True)

    for i, (image, label) in enumerate(train_dataloader):
        image = image.view(-1, 1, 28, 28)
        var_image = Variable(image)
        var_label = Variable(label)
        y_pred = net(var_image)
        y_loss = criterion(y_pred, torch.max(var_label, 1)[1])
        optimizer.zero_grad()
        y_loss.backward()
        optimizer.step()

    for i, (image, label) in enumerate(mnist_dataloader):
        image = image.view(-1, 1, 28, 28)
        if args.cuda:
            var_image = Variable(image).cuda()
            var_label = Variable(label).cuda()
        else:
            var_image = Variable(image)
            var_label = Variable(label)
        y_pred = net(var_image)

        y_loss = criterion(y_pred, var_label)
        optimizer.zero_grad()
        y_loss.backward()
        optimizer.step()


valid_dataset = data.TensorDataset(ft(valid_images), ft(valid_labels))
valid_dataloader = data.DataLoader(valid_dataset, batch_size=args.batch_size)


def test(net, dataloader):
    cnt_correct = 0
    global_loss = 0.
    net.eval()

    for i, (image, label) in enumerate(dataloader):
        image = image.view(-1, 1, 28, 28)
        var_image = Variable(image)
        var_label = Variable(label)
        y_pred = net(var_image)
        y_loss = criterion(y_pred, torch.max(var_label, 1)[1])
        global_loss += y_loss.data[0]
        cnt_correct += torch.eq(torch.max(y_pred, 1)[1], torch.max(var_label,1)[1]).data.sum()


    # print("Test :: global_loss", global_loss / len(dataloader.dataset), "correct : {}/{}, {}".format(cnt_correct, len(dataloader.dataset), 100 * (cnt_correct/ len(dataloader.dataset))))
    return global_loss / len(dataloader.dataset), cnt_correct


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        torch.save(state, "best_"+str(cross_validation_index)+ filename)


best_precision = 0
for epoch in tqdm(range(args.epochs + 70)):

    train(net)
    global_loss, cnt_correct = test(net, valid_dataloader)
    if (epoch + 1) % 10 == 0:
        precision = 100 * (cnt_correct / len(valid_dataloader.dataset))
        is_best = precision > best_precision
        best_precision = max(precision, best_precision)
        print("loss : ", global_loss / len(valid_dataloader.dataset), "correct : {}/{}, {}".format(cnt_correct, len(valid_dataloader.dataset), precision), "best :", best_precision)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_prec' : best_precision,
                'optimizer': optimizer.state_dict(),
            }, is_best
        )


