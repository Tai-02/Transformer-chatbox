import pickle
import os

DATA_PATH = 'data/processed/data.pkl'
VOCAB_PATH = 'data/processed/vocab.pkl'

def debug():
    if not os.path.exists(VOCAB_PATH):
        print("Run preprocess.py first.")
        return

    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    word2idx = vocab['word2idx']
    vocab_size = len(word2idx)

    max_q = max([max(s) if s else 0 for s in data['questions']])
    max_a = max([max(s) if s else 0 for s in data['answers']])

    print(f"Number of samples: {len(data['questions'])}")
    print(f"Vocab size: {vocab_size}")
    print(f"Max index in questions: {max_q}")
    print(f"Max index in answers: {max_a}")

    if max_q >= vocab_size or max_a >= vocab_size:
        print("ERROR: Index out of range!")
    else:
        print("Indices are within range.")

if __name__ == "__main__":
    debug()