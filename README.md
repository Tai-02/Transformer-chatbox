# 🤖 Transformer Chatbot - Toán Rời Rạc

Dự án xây dựng một chatbot dựa trên kiến trúc **Transformer (Decoder-only)** chuyên biệt cho việc giải đáp các câu hỏi trong lĩnh vực **Toán rời rạc**. Mô hình được thiết kế tinh gọn để đạt hiệu năng tối ưu trên cả CPU và GPU cá nhân.

---

## ✨ Tính năng nổi bật
- **Kiến trúc Modern GPT-style**: Sử dụng cơ chế Decoder-only với Self-Attention đa đầu.
- **Dual-Training Support**: Hỗ trợ huấn luyện tăng tốc bằng GPU (CUDA) hoặc chạy ổn định trên CPU.
- **Data Augmentation**: Tích hợp công cụ tăng cường dữ liệu tự động để cải thiện độ chính xác.
- **Early Stopping & Anti-Overfit**: Cơ chế tự động dừng và điều chỉnh giúp mô hình không bị "học vẹt".

---

## 📂 Cấu trúc thư mục
```text
Transformer/
├── data/
│   ├── raw/           <-- Chứa 'output.csv' (Dữ liệu gốc)
│   ├── augmented/     <-- Dữ liệu sau khi tăng cường (Tự động)
│   └── processed/     <-- Dữ liệu đã Tokenize (Tự động)
├── models/            <-- Lưu trữ model tốt nhất ('model.pt')
├── logs/              <-- Nhật ký quá trình huấn luyện
├── src/               <-- Mã nguồn cốt lõi
│   ├── model.py       │   - Định nghĩa kiến trúc Transformer
│   ├── preprocess.py  │   - Tiền xử lý văn bản
│   ├── train_gpu.py   │   - Huấn luyện trên GPU (Khuyên dùng)
│   └── train.py       │   - Huấn luyện trên CPU
└── chat.py            <-- Giao diện trò chuyện với Bot
```

---

## 🛠 Hướng dẫn sử dụng

### 1. Cài đặt môi trường
Đảm bảo bạn đã có Python 3.8+. Cài đặt các thư viện phụ thuộc:
```powershell
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu
1. **Tăng cường dữ liệu**:
   ```powershell
   python src/augment_data.py
   ```
2. **Tiền xử lý và Tokenize**:
   ```powershell
   python src/preprocess.py
   ```

### 3. Huấn luyện mô hình (Training)
Tùy chọn phương thức phù hợp với phần cứng của bạn:

*   **Nếu có GPU NVIDIA (Nên dùng):**
    ```powershell
    python src/train_gpu.py
    ```
*   **Nếu chỉ có CPU:**
    ```powershell
    python src/train.py
    ```

### 4. Chat với Bot
Sau khi huấn luyện kết thúc, chạy script sau để bắt đầu trò chuyện:
```powershell
python chat.py
```

---

## ⚙️ Thông số kỹ thuật (Hyperparameters)
Mô hình hiện tại đang sử dụng cấu hình **"Efficient-Base"**:

| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| **Layers** | 3 | Số lớp Transformer Block |
| **Heads** | 8 | Số đầu Attention |
| **D_Model** | 192 | Kích thước vector nhúng |
| **Dropout** | 0.35 | Tỉ lệ loại bỏ (Chống Overfit) |
| **Optimizer** | AdamW | Tốc độ học: 8e-4, Weight Decay: 0.1 |
| **Scheduler** | Cosine | Giảm dần LR theo chu kỳ |
| **Patience** | 10 Epochs | Tự động dừng nếu Loss không giảm |

---

## ⚠️ Lưu ý
- File `output.csv` trong `data/raw/` cần định dạng: `Chủ đề;Câu hỏi;Câu trả lời`.
- Nếu gặp lỗi bộ nhớ trên GPU (OOM), hãy giảm `BATCH_SIZE` trong file `train_gpu.py`.
