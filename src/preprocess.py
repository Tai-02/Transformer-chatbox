import pandas as pd
import re
import pickle
import os
from collections import Counter

TRAIN_CSV = 'data/augmented/train.csv'
VAL_CSV = 'data/augmented/val.csv'
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
    df = pd.read_csv(CSV_PATH, sep=';', header=None, names=['topic', 'question', 'answer'], encoding='utf-8-sig').dropna(subset=['question', 'answer'])
    
    tokenized_q = [tokenize(q) for q in df['question']]
    tokenized_a = [tokenize(a) for a in df['answer']]
    
    indexed_data = []
    for q, a in zip(tokenized_q, tokenized_a):
        q_idx = [word2idx.get(w, word2idx['<unk>']) for w in q]
        a_idx = [word2idx.get(w, word2idx['<unk>']) for w in a]
        indexed_data.append([word2idx['<sos>']] + q_idx + [word2idx['<sep>']] + a_idx + [word2idx['<eos>']])
    return indexed_data

def main():
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        print("CSV files not found. Run augment_data.py first.")
        return
        
    train_df = pd.read_csv(TRAIN_CSV, header=0, names=['topic', 'question', 'answer']).dropna(subset=['question', 'answer'])
    val_df = pd.read_csv(VAL_CSV, header=0, names=['topic', 'question', 'answer']).dropna(subset=['question', 'answer'])
    
    # Build vocab using ALL data (train + val)
    all_text = []
    for q, a in zip(train_df['question'], train_df['answer']):
        all_text.append(tokenize(q))
        all_text.append(tokenize(a))
    for q, a in zip(val_df['question'], val_df['answer']):
        all_text.append(tokenize(q))
        all_text.append(tokenize(a))
        
    word2idx, idx2word = build_vocab(all_text)
    with open(VOCAB_PATH, 'wb') as f: pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)

    train_indexed = process_df(train_df, word2idx)
    val_indexed = process_df(val_df, word2idx)

    with open(PROCESSED_DATA_PATH, 'wb') as f:
        pickle.dump({
            'train': {'sequences': train_indexed, 'topics': train_df['topic'].tolist()},
            'val': {'sequences': val_indexed, 'topics': val_df['topic'].tolist()}
        }, f)
    print(f"Processed {len(train_indexed)} train and {len(val_indexed)} val samples.")

if __name__ == "__main__":
    main()