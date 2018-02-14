from gensim.utils import simple_tokenize
import pandas as pd
import numpy as np


def tokenizer(string : str):
    return [s for s in simple_tokenize(string)]


def get_pd_data(path : str):
    df = pd.read_csv(path)
    return df


def set_capital_ratio(df : pd.DataFrame):
    df['alphas'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isalpha()))
    df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['cap_ratio'] = df.apply(lambda row: float(row['capitals']) / (float(row['alphas']) + 1), axis=1)


def set_capital_ratio_of_words(tokens : list):
    return [(sum(1 for c in token if c.isupper()))/(sum(1 for c in token if c.isalpha()) + 1) for token in tokens]


def shuffle_lists(*pargs):
    shuffler = np.random.permutation(len(pargs[0]))
    retargs = []
    for x in pargs:
        x = x.iloc[shuffler]
        retargs.append(x)
    return retargs


def padding_cap_ratio(tokens : list, len_sentence : int):
    L = len(tokens)
    ret = tokens[:min(L, len_sentence)]
    ret = [1] * (len_sentence - L) + ret
    return np.array(ret)


def batchify(*pargs, batch_size=32):
    for i in range(0, len(pargs[0]), batch_size):
        end = min(i+batch_size, len(pargs[0]))
        yield [batch[i:end] for batch in pargs]