from gensim.utils import simple_tokenize
import pandas as pd

def tokenizer(string : str):
    return [s for s in simple_tokenize(string)]

def get_pd_data(path : str):
    df = pd.read_csv(path)
    return df

def set_capital_ratio(df : pd.DataFrame):
    df['alphas'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isalpha()))
    df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['cap_ratio'] = df.apply(lambda row: float(row['capitals']) / (float(row['alphas']) + 1), axis=1)

