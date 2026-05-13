import json
import matplotlib.pyplot as plt
import os

def plot_training_history():
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        print("❌ Không tìm thấy thư mục logs. Hãy train model trước!")
        return

    models = ['small', 'medium', 'large']
    histories = {}

    for m in models:
        path = os.path.join(logs_dir, f'history_{m}.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                histories[m] = json.load(f)

    if not histories:
        print("❌ Không tìm thấy file lịch sử nào (history_*.json).")
        return

    # Khởi tạo khung hình lớn (15x10 inches) chứa 4 biểu đồ phụ
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Biểu Đồ Đánh Giá Khả Năng Học Của Các Mô Hình Transformer', fontsize=18, fontweight='bold', y=0.98)

    # Màu sắc cho từng mô hình
    colors = {'small': '#1f77b4', 'medium': '#ff7f0e', 'large': '#2ca02c'}

    # 1. Biểu đồ Validation Loss (So sánh khả năng suy luận)
    ax = axs[0, 0]
    for m, h in histories.items():
        ax.plot(h['val_loss'], label=m.upper(), color=colors[m], linewidth=2)
    ax.set_title('Validation Loss (Càng thấp càng tốt)', fontsize=12)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # 2. Biểu đồ Validation Accuracy (So sánh độ chính xác)
    ax = axs[0, 1]
    for m, h in histories.items():
        ax.plot(h['val_acc'], label=m.upper(), color=colors[m], linewidth=2)
    ax.set_title('Validation Accuracy (%) (Càng cao càng tốt)', fontsize=12)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # 3. Biểu đồ Train Loss vs Val Loss (Kiểm tra Học Vẹt - Overfitting)
    ax = axs[1, 0]
    for m, h in histories.items():
        # Chỉ vẽ nét đứt cho Train Loss, nét liền cho Val Loss để dễ phân biệt
        ax.plot(h['train_loss'], linestyle=':', color=colors[m], alpha=0.7)
        ax.plot(h['val_loss'], label=f'{m.upper()} (Val)', color=colors[m], linewidth=2)
    ax.set_title('Train Loss (nét đứt) vs Val Loss (nét liền)\nKhoảng cách quá xa = Học vẹt', fontsize=12)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # 4. Biểu đồ Learning Rate
    ax = axs[1, 1]
    for m, h in histories.items():
        if 'lr' in h:
            ax.plot(h['lr'], label=m.upper(), color=colors[m], linewidth=2)
    ax.set_title('Learning Rate (Tốc độ học theo chu kỳ)', fontsize=12)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Lưu file ảnh độ phân giải cao
    save_path = os.path.join(logs_dir, 'training_comparison_plot.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Đã tạo thành công biểu đồ so sánh!")
    print(f"👉 File ảnh được lưu tại: {save_path}")
    
    # plt.show() # Tắt show() để không lỗi khi chạy trên Colab

if __name__ == "__main__":
    plot_training_history()
