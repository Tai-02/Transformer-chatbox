# Transformer Chatbot - Toán Rời Rạc

Dự án xây dựng một chatbot dựa trên kiến trúc Transformer (Decoder-only) chuyên về giải đáp các câu hỏi Toán rời rạc. Mô hình được tối ưu hóa để chạy tốt trên cả CPU và GPU cá nhân.

## 📌 Quy trình thực hiện (Bắt buộc)

Dù bạn chọn chạy bằng CPU hay GPU, bạn đều phải thực hiện 2 bước chuẩn bị dữ liệu sau:

1. **Tăng cường dữ liệu (Data Augmentation):**
   ```powershell
   python src/augment_data.py
   ```
2. **Tiền xử lý và Tokenize:**
   ```powershell
   python src/preprocess.py
   ```

---

## 🚀 Cách 1: Huấn luyện bằng GPU (Khuyên dùng)
Dành cho máy tính có card đồ họa NVIDIA. Tốc độ xử lý nhanh gấp nhiều lần CPU.

*   **Lệnh chạy:**
    ```powershell
    python src/train_gpu.py
    ```
*   **Đặc điểm:** Sử dụng thư viện `CUDA`, hỗ trợ tự động nhận diện phần cứng và tối ưu hóa bộ nhớ.

## 💻 Cách 2: Huấn luyện bằng CPU
Dành cho máy tính không có card đồ họa rời hoặc muốn chạy tiết kiệm năng lượng.

*   **Lệnh chạy:**
    ```powershell
    python src/train.py
    ```
*   **Đặc điểm:** Chạy ổn định trên mọi hệ máy, các thông số đã được đồng bộ hóa hoàn toàn với bản GPU.

---

## 💬 Cách Chat với Bot
Sau khi huấn luyện xong, model tốt nhất sẽ được lưu tại `models/model.pt`. Để trò chuyện với Bot, hãy chạy:

```powershell
python chat.py
```

## 🛠 Thông số kỹ thuật (Cấu hình "Tốc độ")
- **Kiến trúc:** Transformer Decoder (4 Layers, 8 Heads)
- **D_Model:** 256
- **Dropout:** 0.2 (Chống học vẹt)
- **Optimizer:** AdamW (Weight Decay: 0.05)
- **Loss Function:** CrossEntropy (Label Smoothing: 0.1)
- **Early Stopping:** Tự động dừng nếu sai số tập Val không cải thiện quá 0.01 trong 10 Epoch liên tiếp.
