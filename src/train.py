import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import sys
import pickle
import os
from sklearn.model_selection import train_test_split
from model import make_model

sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = 'data/processed/data.pkl'
VOCAB_PATH = 'data/processed/vocab.pkl'
SAVE_PATH = 'models/model.pt'

BATCH_SIZE = 64
D_MODEL = 192
N_LAYER = 3
N_HEAD = 8
DROPOUT = 0.35
EPOCHS = 200
LR = 8e-4

class EarlyStopping:
    def __init__(self, patience=30, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class SeqDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx])

def collate_fn(batch, pad_idx):
    seqs = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_idx)
    x = seqs[:, :-1]
    y = seqs[:, 1:].clone()
    y[y == pad_idx] = -1
    return x, y

def train():
    device = torch.device("cpu")
    if not os.path.exists(DATA_PATH): return

    with open(VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    with open(DATA_PATH, 'rb') as f: data = pickle.load(f)

    word2idx, pad_idx, vocab_size = vocab['word2idx'], vocab['word2idx']['<pad>'], len(vocab['word2idx'])

    train_seq, val_seq, _, _ = train_test_split(data['sequences'], data['topics'], test_size=0.2, stratify=data['topics'], random_state=42)

    train_dl = DataLoader(SeqDataset(train_seq), batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx))
    val_dl = DataLoader(SeqDataset(val_seq), batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_idx))

    model = make_model(vocab_size, n_layer=N_LAYER, d_model=D_MODEL, n_head=N_HEAD, dropout=DROPOUT)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    early_stopping = EarlyStopping(patience=10, delta=0.01)
    best_val_loss = float('inf')
    best_val_acc = 0
    best_epoch = 0

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = total_train_acc = 0
        start_time = time.time()

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            mask = y != -1
            total_train_acc += (logits.argmax(dim=-1) == y)[mask].float().mean().item()

        scheduler.step()
        avg_train_loss, avg_train_acc = total_train_loss / len(train_dl), total_train_acc / len(train_dl) * 100

        model.eval()
        total_val_loss = total_val_acc = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                total_val_loss += loss.item()
                mask = y != -1
                total_val_acc += (logits.argmax(dim=-1) == y)[mask].float().mean().item()

        avg_val_loss, avg_val_acc = total_val_loss / len(val_dl), total_val_acc / len(val_dl) * 100
        
        is_overfitting = avg_val_loss > avg_train_loss + 0.5
        warn = " ⚠️ OVERFIT" if is_overfitting else ""
        msg = f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} ({avg_train_acc:.1f}%) | Val: {avg_val_loss:.4f} ({avg_val_acc:.1f}%) | {time.time()-start_time:.1f}s{warn}"
        print(msg)
        with open('logs/progress.txt', 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            best_epoch = epoch + 1
            torch.save({'model_state_dict': model.state_dict(), 'v_size': vocab_size, 'd_m': D_MODEL, 'n_l': N_LAYER, 'n_h': N_HEAD, 'dr': DROPOUT}, SAVE_PATH)

        if is_overfitting:
            stop_msg = f"Stopping training due to Overfitting (Gap > 0.5) at epoch {epoch+1}."
            print(stop_msg)
            with open('logs/progress.txt', 'a', encoding='utf-8') as f:
                f.write(stop_msg + "\n")
            break

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            stop_msg = f"Early stopping at epoch {epoch+1}."
            print(stop_msg)
            with open('logs/progress.txt', 'a', encoding='utf-8') as f:
                f.write(stop_msg + "\n")
            break

    final_msg = f"\n--- Training Complete ---\nBest Epoch: {best_epoch}\nBest Val Loss: {best_val_loss:.4f}\nBest Val Acc: {best_val_acc:.1f}%\nModel saved to: {SAVE_PATH}"
    print(final_msg)
    with open('logs/progress.txt', 'a', encoding='utf-8') as f:
        f.write(final_msg + "\n")

if __name__ == "__main__":
    train()