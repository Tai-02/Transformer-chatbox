import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import sys
import pickle
import os
import json
from sklearn.model_selection import train_test_split
from model import make_model

sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = 'data/processed/data.pkl'
VOCAB_PATH = 'data/processed/vocab.pkl'

# ============================================================
# CẤU HÌNH NHIỀU MÔ HÌNH ĐỂ SO SÁNH ĐÁNH GIÁ
# Mục đích: Train nhiều kích thước não khác nhau,
#           rồi so sánh xem cấu hình nào "thông minh" nhất.
# ============================================================

MODEL_CONFIGS = {
    "small": {
        "D_MODEL": 256,
        "N_LAYER": 4,
        "N_HEAD": 8,
        "DROPOUT": 0.3,       # Tăng từ 0.2 -> 0.3 để giảm học vẹt
        "BATCH_SIZE": 64,
        "EPOCHS": 200,
        "LR": 5e-4,
        "DESCRIPTION": "Mô hình GỐC (nhỏ) - Cấu hình của Đại"
    },
    "medium": {
        "D_MODEL": 512,       # Tăng gấp đôi số chiều vector
        "N_LAYER": 6,         # Tăng số lớp suy luận
        "N_HEAD": 8,
        "DROPOUT": 0.35,      # Tăng Dropout chống học vẹt
        "BATCH_SIZE": 32,     # Giảm batch size do mô hình to hơn
        "EPOCHS": 300,
        "LR": 3e-4,
        "DESCRIPTION": "Mô hình VỪA - Cân bằng giữa tốc độ và chất lượng"
    },
    "large": {
        "D_MODEL": 512,       # Giữ chiều vector 512
        "N_LAYER": 8,         # Tăng gấp đôi số lớp suy luận so với gốc
        "N_HEAD": 16,         # Tăng số đầu Attention
        "DROPOUT": 0.4,       # Dropout cao nhất để triệt để chống học vẹt
        "BATCH_SIZE": 16,     # Giảm batch size tránh tràn bộ nhớ
        "EPOCHS": 300,
        "LR": 2e-4,
        "DESCRIPTION": "Mô hình LỚN - Não to nhất, tiềm năng thông minh nhất"
    },
}

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

def count_parameters(model):
    """Đếm tổng số tham số của mô hình (kích thước não)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_model(config_name, config, device, train_seq, val_seq, pad_idx, vocab_size):
    """Train một cấu hình mô hình và trả về kết quả đánh giá."""
    
    D_MODEL = config["D_MODEL"]
    N_LAYER = config["N_LAYER"]
    N_HEAD = config["N_HEAD"]
    DROPOUT = config["DROPOUT"]
    BATCH_SIZE = config["BATCH_SIZE"]
    EPOCHS = config["EPOCHS"]
    LR = config["LR"]
    
    SAVE_PATH = f'models/model_{config_name}.pt'

    print(f"\n{'🧠'*25}")
    print(f"  BẮT ĐẦU TRAIN MÔ HÌNH: {config_name.upper()}")
    print(f"  {config['DESCRIPTION']}")
    print(f"{'🧠'*25}")
    print(f"  D_MODEL={D_MODEL} | N_LAYER={N_LAYER} | N_HEAD={N_HEAD}")
    print(f"  DROPOUT={DROPOUT} | BATCH={BATCH_SIZE} | LR={LR}")
    print(f"  EPOCHS={EPOCHS}")

    # --- Curriculum Learning ---
    # Giai đoạn đầu: Sắp xếp dữ liệu từ câu NGẮN → DÀI
    # Giúp bot học nền tảng vững từ câu đơn giản trước
    # Sau 20% epochs, chuyển sang shuffle bình thường
    curriculum_epochs = max(1, EPOCHS // 5)  # 20% epochs đầu dùng curriculum
    
    # Sắp xếp train_seq theo độ dài (ngắn → dài)
    sorted_train_seq = sorted(train_seq, key=lambda s: len(s))
    
    print(f"  📚 Curriculum Learning: {curriculum_epochs} epochs đầu học câu ngắn → dài")
    
    # DataLoader cho giai đoạn curriculum (không shuffle, giữ thứ tự ngắn→dài)
    curriculum_dl = DataLoader(SeqDataset(sorted_train_seq), batch_size=BATCH_SIZE, shuffle=False,
                               collate_fn=lambda b: collate_fn(b, pad_idx), pin_memory=True)
    # DataLoader cho giai đoạn bình thường (có shuffle)
    normal_dl = DataLoader(SeqDataset(train_seq), batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=lambda b: collate_fn(b, pad_idx), pin_memory=True)
    val_dl = DataLoader(SeqDataset(val_seq), batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, pad_idx), pin_memory=True)

    model = make_model(vocab_size, n_layer=N_LAYER, d_model=D_MODEL, n_head=N_HEAD, dropout=DROPOUT)
    model.to(device)
    
    num_params = count_parameters(model)
    print(f"  Tổng tham số  : {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"{'='*60}")

    # --- Optimizer với Weight Decay (L2 Regularization) ---
    # Weight Decay giúp giảm overfit bằng cách phạt các trọng số quá lớn
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    
    # --- Learning Rate Warmup + Cosine Annealing ---
    # Warmup: Tăng LR từ từ ở đầu để tránh "sốc" mô hình
    # Cosine: Giảm LR mượt mà giống hình sin để hội tụ tốt hơn
    warmup_epochs = min(10, EPOCHS // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Warmup tuyến tính
        else:
            import math
            progress = (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    early_stopping = EarlyStopping(patience=30, delta=0.005)
    best_val_loss = float('inf')
    best_val_acc = 0
    best_train_loss = float('inf')
    best_epoch = 0
    
    # Lưu lịch sử training để vẽ biểu đồ sau
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    log_file = f'logs/progress_{config_name}.txt'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Training Log: {config_name.upper()} ===\n")
        f.write(f"Config: D_MODEL={D_MODEL}, N_LAYER={N_LAYER}, N_HEAD={N_HEAD}\n")
        f.write(f"DROPOUT={DROPOUT}, BATCH={BATCH_SIZE}, LR={LR}\n")
        f.write(f"Params: {num_params:,}\n\n")

    total_start = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = total_train_acc = 0
        start_time = time.time()

        # Curriculum Learning: chọn DataLoader phù hợp theo giai đoạn
        if epoch < curriculum_epochs:
            train_dl = curriculum_dl  # Câu ngắn → dài
        else:
            train_dl = normal_dl     # Shuffle bình thường

        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, loss = model(x, y)
                scaler.scale(loss).backward()
                # --- Gradient Clipping ---
                # Giới hạn độ lớn gradient để tránh "nổ gradient" (Exploding gradient)
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

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        avg_train_loss = total_train_loss / len(train_dl)
        avg_train_acc = total_train_acc / len(train_dl) * 100

        model.eval()
        total_val_loss = total_val_acc = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)

                total_val_loss += loss.item()
                mask = y != -1
                total_val_acc += (logits.argmax(dim=-1) == y)[mask].float().mean().item()

        avg_val_loss, avg_val_acc = total_val_loss / len(val_dl), total_val_acc / len(val_dl) * 100

        # Lưu lịch sử
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)
        history["lr"].append(current_lr)

        # Cảnh báo Overfit
        gap = avg_val_loss - avg_train_loss
        if gap > 1.0:
            warn = " 🔴 OVERFIT NẶNG"
        elif gap > 0.5:
            warn = " ⚠️ OVERFIT NHẸ"
        else:
            warn = ""
            
        msg = f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {avg_train_loss:.4f} ({avg_train_acc:.1f}%) | Val: {avg_val_loss:.4f} ({avg_val_acc:.1f}%) | LR: {current_lr:.6f} | {time.time()-start_time:.1f}s{warn}"
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            torch.save({
                'model_state_dict': model.state_dict(),
                'v_size': vocab_size,
                'd_m': D_MODEL,
                'n_l': N_LAYER,
                'n_h': N_HEAD,
                'dr': DROPOUT
            }, SAVE_PATH)

        # Chỉ kích hoạt Early Stopping sau khi đã qua ít nhất 1/3 số Epochs
        # Để tránh mô hình bị dừng sớm quá nhạy cảm ở giai đoạn Warmup & Curriculum lúc đầu
        if epoch >= (EPOCHS // 3):
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print(f"⏹️  Early stopping tại epoch {epoch+1}.")
                break

    total_time = time.time() - total_start
    
    # Lưu lịch sử training
    history_file = f'logs/history_{config_name}.json'
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f)

    result = {
        "config_name": config_name,
        "description": config["DESCRIPTION"],
        "num_params": num_params,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "total_time": total_time,
        "total_epochs_ran": len(history["train_loss"]),
        "overfit_gap": best_val_loss - best_train_loss if best_train_loss != float('inf') else 0,
    }

    summary = f"""
{'='*60}
  ✅ HOÀN TẤT TRAIN: {config_name.upper()}
{'='*60}
  📝 {config['DESCRIPTION']}
  🧠 Tham số      : {num_params:,} ({num_params/1e6:.2f}M)
  🏆 Best Epoch   : {best_epoch}
  📉 Best Val Loss: {best_val_loss:.4f}
  📈 Best Val Acc : {best_val_acc:.1f}%
  ⏱️  Tổng thời gian: {total_time:.1f}s ({total_time/60:.1f} phút)
  💾 Model saved  : {SAVE_PATH}
  📊 History saved: {history_file}
{'='*60}
"""
    print(summary)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(summary)

    return result


def compare_models(results):
    """So sánh kết quả của tất cả các mô hình đã train."""
    
    print(f"\n{'🏆'*25}")
    print(f"  BẢNG SO SÁNH TẤT CẢ CÁC MÔ HÌNH")
    print(f"{'🏆'*25}\n")
    
    header = f"{'Mô hình':<12} | {'Tham số':>12} | {'Val Loss':>10} | {'Val Acc':>10} | {'Overfit Gap':>12} | {'Thời gian':>10}"
    separator = "-" * len(header)
    print(header)
    print(separator)
    
    for r in results:
        row = f"{r['config_name']:<12} | {r['num_params']:>10,}  | {r['best_val_loss']:>10.4f} | {r['best_val_acc']:>9.1f}% | {r['overfit_gap']:>12.4f} | {r['total_time']:>8.1f}s"
        print(row)
    
    print(separator)
    
    # Tìm mô hình tốt nhất (Val Loss thấp nhất)
    best = min(results, key=lambda x: x['best_val_loss'])
    print(f"\n🥇 MÔ HÌNH TỐT NHẤT: {best['config_name'].upper()}")
    print(f"   Val Loss = {best['best_val_loss']:.4f} | Val Acc = {best['best_val_acc']:.1f}%")
    print(f"   → File model: models/model_{best['config_name']}.pt")
    print(f"   → Copy file này thành 'models/model.pt' để sử dụng trên Web!\n")
    
    # Lưu kết quả so sánh ra file
    comparison_file = 'logs/model_comparison.json'
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"📊 Đã lưu bảng so sánh vào: {comparison_file}")

    return best


def train_gpu():
    """Hàm chính: Train tất cả các cấu hình mô hình."""
    
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("⚠️  Không tìm thấy GPU. Đang dùng CPU (sẽ chậm hơn).")
    else:
        device = torch.device("cuda")
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(DATA_PATH):
        print(f"❌ Không tìm thấy dữ liệu: {DATA_PATH}")
        print("   Hãy chạy augment_data.py và preprocess.py trước!")
        return

    with open(VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    with open(DATA_PATH, 'rb') as f: data = pickle.load(f)

    word2idx = vocab['word2idx']
    pad_idx = word2idx['<pad>']
    vocab_size = len(word2idx)

    train_seq, val_seq, train_topics, _ = train_test_split(
        data['sequences'], data['topics'], test_size=0.2, stratify=data['topics'], random_state=42
    )

    print(f"\n📊 THỐNG KÊ DỮ LIỆU:")
    print(f"   Train samples : {len(train_seq)}")
    print(f"   Val samples   : {len(val_seq)}")
    print(f"   Vocab size    : {vocab_size}")

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # ============================================================
    # CHỌN CHẾ ĐỘ TRAIN
    # Nếu muốn train chỉ 1 mô hình, đổi train_configs thành
    # danh sách chỉ chứa tên mô hình đó, ví dụ: ["medium"]
    # ============================================================
    
    train_configs = ["small", "medium", "large"]
    
    results = []
    for config_name in train_configs:
        if config_name not in MODEL_CONFIGS:
            print(f"⚠️  Cấu hình '{config_name}' không tồn tại, bỏ qua.")
            continue
        try:
            result = train_one_model(
                config_name, MODEL_CONFIGS[config_name],
                device, train_seq, val_seq, pad_idx, vocab_size
            )
            results.append(result)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ MÔ HÌNH {config_name.upper()} BỊ TRÀN BỘ NHỚ!")
                print(f"   Hãy giảm BATCH_SIZE hoặc D_MODEL rồi thử lại.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e

    # So sánh tất cả mô hình
    if len(results) > 1:
        best = compare_models(results)
    elif len(results) == 1:
        print(f"\n✅ Chỉ train 1 mô hình: {results[0]['config_name']}")
        print(f"   Val Loss: {results[0]['best_val_loss']:.4f}")
        print(f"   Val Acc: {results[0]['best_val_acc']:.1f}%")

if __name__ == "__main__":
    train_gpu()