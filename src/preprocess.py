import pandas as pd
import re
import pickle
import os
from collections import Counter

CSV_PATH = 'data/augmented/output_augmented.csv'
DATA_DIR = 'data/processed'
VOCAB_PATH = os.path.join(DATA_DIR, 'vocab.pkl')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'data.pkl')

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

def tokenize(text):
    text = str(text).lower()
    text = re.sub(r'([.,!?()])', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip().split()

def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sentence in sentences: counter.update(sentence)
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>', '<sep>']
    words = [word for word, freq in counter.items() if freq >= min_freq and word not in special_tokens]
    vocab = special_tokens + words
    return {word: i for i, word in enumerate(vocab)}, {i: word for i, word in enumerate(vocab)}

def main():
    if not os.path.exists(CSV_PATH): return
    df = pd.read_csv(CSV_PATH, header=0, names=['topic', 'question', 'answer']).dropna(subset=['question', 'answer'])
    
    tokenized_q = [tokenize(q) for q in df['question']]
    tokenized_a = [tokenize(a) for a in df['answer']]
    
    word2idx, idx2word = build_vocab(tokenized_q + tokenized_a)
    with open(VOCAB_PATH, 'wb') as f: pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)

    indexed_data = []
    for q, a in zip(tokenized_q, tokenized_a):
        q_idx = [word2idx.get(w, word2idx['<unk>']) for w in q]
        a_idx = [word2idx.get(w, word2idx['<unk>']) for w in a]
        indexed_data.append([word2idx['<sos>']] + q_idx + [word2idx['<sep>']] + a_idx + [word2idx['<eos>']])

    with open(PROCESSED_DATA_PATH, 'wb') as f: pickle.dump({'sequences': indexed_data, 'topics': df['topic'].tolist()}, f)

if __name__ == "__main__":
    main()