from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn import preprocessing
import random
from collections import Counter

def load_raw(path):
    train_data = pd.read_csv(path)
    return train_data


def encode_authors(train_data):
    """
    :param train_data: pd.Dataframe
    :return: encoded(0, 1, 2... ) authors
    """
    le = preprocessing.LabelEncoder()
    le.fit(train_data['author'])
    encoded_authors = le.transform(train_data['author'])
    return encoded_authors


def tokenize(text):
    ret = word_tokenize(text)
    return ret


def get_vocab_set(tokenized_texts, max_words=10000):

    if max_words == 0:
        vocab = set()
        for tokenized_text in tokenized_texts:
            vocab = vocab.union(set(tokenized_text))
        print("vocab size", len(vocab))
        return vocab
    else:
        counter = Counter()
        for tokenized_text in tokenized_texts:
            for word in tokenized_text:
                counter[word] += 1
        print("vocab size ", len(counter))
        return dict(counter.most_common(max_words)).keys()


def zero_padding(tokens, fixed_length=64):
    if len(tokens) > fixed_length:
        tokens = tokens[:fixed_length]
    pad = [0] * (fixed_length - len(tokens))
    return tokens + pad

def tokens_to_ix(word_to_ix, tokens, fixed_length=64):
    """
    :param word_to_ix:
    :param tokens:

    :param fixed_length:
    :return:

    list(text_tokens) -> list(index_tokens, zero padding)
    """
    pad = []
    ret = []
    for w in tokens:
        if w in word_to_ix.keys():
            ret.append(word_to_ix[w])
        else:
            ret.append(1)  # words that haven't seen before

    ret = zero_padding(ret, fixed_length)

    return np.array(ret + pad)

def divide_validation_set(x, y, validation_ratio):
    train_num = int(len(x) * (1 - validation_ratio))

    train_x, train_y = x[:train_num], y[:train_num]
    valid_x, valid_y = x[train_num:], y[train_num:]
    print('train ', train_num, 'valid ', len(valid_x))
    return train_x, train_y, valid_x, valid_y


def shuffle_x_y(x,y):
    shuffler = np.arange(len(x))
    random.shuffle(shuffler)
    x = x[shuffler]
    y = y[shuffler]
    return x, y