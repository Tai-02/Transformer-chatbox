import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import sys
import pickle
import os
import math
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
DROPOUT = 0.30
EPOCHS = 400
LR = 5e-4
WARMUP_EPOCHS = 10

class EarlyStopping:
    def __init__(self, patience=50, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_acc = 0.0  # track accuracy (higher = better)
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_acc):
        if val_acc > self.best_acc + self.delta:
            self.best_acc = val_acc
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

def train_gpu():
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(DATA_PATH):
        print("Data not found.")
        return

    print("Loading data...")
    with open(VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    with open(DATA_PATH, 'rb') as f: data = pickle.load(f)

    word2idx, pad_idx, vocab_size = vocab['word2idx'], vocab['word2idx']['<pad>'], len(vocab['word2idx'])

    train_seq = data['train']['sequences']
    val_seq = data['val']['sequences']
    
    print(f"Train: {len(train_seq)} | Val: {len(val_seq)} | Vocab: {vocab_size}")

    train_dl = DataLoader(SeqDataset(train_seq), batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, pad_idx), pin_memory=True)
    val_dl = DataLoader(SeqDataset(val_seq), batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, pad_idx), pin_memory=True)

    print("Initializing model...")
    model = make_model(vocab_size, n_layer=N_LAYER, d_model=D_MODEL, n_head=N_HEAD, dropout=DROPOUT)
    model.to(device)

    print("Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    # Warmup then ReduceLROnPlateau: LR tự giảm khi val_acc không tăng
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-6, verbose=True)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    early_stopping = EarlyStopping(patience=60, delta=0.1)  # acc-based, stop if no +0.1% in 60 epochs
    best_val_loss = float('inf')
    best_val_acc = 0
    best_epoch = 0

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    with open('logs/progress.txt', 'w', encoding='utf-8') as f:
        f.write("=== Training Log ===\n")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = total_train_acc = 0
        start_time = time.time()

        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_train_loss += loss.item()
            mask = y != -1
            total_train_acc += (logits.argmax(dim=-1) == y)[mask].float().mean().item()

        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        avg_train_loss = total_train_loss / len(train_dl)
        avg_train_acc = total_train_acc / len(train_dl) * 100

        model.eval()
        total_val_loss = total_val_acc = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits, loss = model(x, y) if scaler is None else (torch.amp.autocast('cuda')(lambda: model(x, y))())
                # Simplification:
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)

                total_val_loss += loss.item()
                mask = y != -1
                total_val_acc += (logits.argmax(dim=-1) == y)[mask].float().mean().item()

        avg_val_loss, avg_val_acc = total_val_loss / len(val_dl), total_val_acc / len(val_dl) * 100
        
        is_overfitting = avg_val_loss > avg_train_loss + 2.5
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
            stop_msg = f"Stopping training due to Overfitting (Gap > 1.5) at epoch {epoch+1}."
            print(stop_msg)
            with open('logs/progress.txt', 'a', encoding='utf-8') as f:
                f.write(stop_msg + "\n")
            break

        plateau_scheduler.step(avg_val_acc)  # ReduceLROnPlateau theo val_acc
        early_stopping(avg_val_acc)  # track accuracy
        if early_stopping.early_stop:
            stop_msg = f"Early stopping at epoch {epoch+1}. (no val acc improvement > 0.1% in {early_stopping.patience} epochs)"
            print(stop_msg)
            with open('logs/progress.txt', 'a', encoding='utf-8') as f:
                f.write(stop_msg + "\n")
            break

    final_msg = f"\n--- Training Complete ---\nBest Epoch: {best_epoch}\nBest Val Loss: {best_val_loss:.4f}\nBest Val Acc: {best_val_acc:.1f}%\nModel saved to: {SAVE_PATH}"
    print(final_msg)
    with open('logs/progress.txt', 'a', encoding='utf-8') as f:
        f.write(final_msg + "\n")

if __name__ == "__main__":
    train_gpu()