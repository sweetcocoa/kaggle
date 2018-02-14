import random
import os

# parser = argparse.ArgumentParser(description='Toxic comment Classification')
# parser.add_argument("--vocab-size", type=int, default=20000, metavar='N',
#                     help="vocab size")
# parser.add_argument("--embedding-dim", type=int, default=100, metavar='N',
#                     help="embedding dim")
# parser.add_argument("--len-sentence", type=int, default=100, metavar='N',
#                     help="sequence length's limit")
# parser.add_argument("--num-labels", type=int, default=6, metavar='N',
#                     help="label num")
# parser.add_argument("--min-freq", type=int, default=1, metavar='N',
#                     help="min_freq")
# parser.add_argument("--channel-size", type=int, default=128, metavar='N',
#                     help="channel size of cnn")
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                     help='input batch size for training (default: 64)')
# parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
#                     help='input batch size for testing (default: 64)')
# parser.add_argument('--valid_num', type=int, default=10000, metavar='N',
#                     help='input batch size for testing (default: 64)')
# parser.add_argument('--n-gram', type=int, default=5, metavar='N',
#                     help='n_gram')
# parser.add_argument('--epochs', type=int, default=5, metavar='N',
#                     help='number of epochs to train (default: 5)')
# parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
#                     help='learning rate (default: 0.001)')
# parser.add_argument('--seed', type=int, default=0, metavar='S',
#                     help='random seed (default: 0)')
# parser.add_argument('--dropout', type=float, default=0.5, metavar='LR',
#                     help='dropout rate(default: 0.5)')
# parser.add_argument('--x2-size', type=int, default=1, metavar='S',
#                     help='x2-size (default: 1)')
# parser.add_argument('--valid-num', type=int, default=10000, metavar='S',
#                     help='valid_num (default: 10000)')
#
#
# parser.add_argument("--valid", type=float, default=0.02, metavar='RATIO',
#                     help="valid set's ratio")
# config = parser.parse_args()

for i in range(10000):
    vocab_size = random.randint(50,300) * 100
    embedding_dim = random.randint(50,300)
    len_sentence = random.randint(15,100) * 2
    min_freq = random.randint(1,5)
    channel_size = 2 ** random.randint(3,9)
    n_gram = 1 + random.randint(1,3) * 2
    dropout = random.random()
    os.system("python main.py --vocab-size %s --embedding-dim %s --len-sentence %s --min-freq %s --channel-size %s --n-gram %s --dropout %s"%(vocab_size, embedding_dim, len_sentence, min_freq, channel_size, n_gram, dropout))
