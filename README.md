# Transformer Chatbot - Toán Rời Rạc

Dự án xây dựng một chatbot dựa trên kiến trúc Transformer (Decoder-only) chuyên về giải đáp các câu hỏi Toán rời rạc. Mô hình được tối ưu hóa để chạy tốt trên cả CPU và GPU cá nhân.

---

## 📂 Cấu trúc thư mục (Directory Structure)
Do các file dữ liệu lớn và model checkpoints được liệt kê trong `.gitignore`, bạn cần chuẩn bị cấu trúc thư mục như sau trước khi chạy:

```text
Transformer/
├── data/
│   ├── raw/           <-- Đặt file 'output.csv' vào đây
│   ├── augmented/     <-- Tự động tạo khi chạy augment_data.py
│   └── processed/     <-- Tự động tạo khi chạy preprocess.py
├── models/            <-- Nơi lưu trữ 'model.pt' sau khi train
├── logs/              <-- Nơi lưu trữ 'progress.txt'
├── src/               <-- Mã nguồn (Preprocess, Train, Model)
├── chat.py            <-- Script để chat với bot
└── README.md
```

## 🛠 Cài đặt & Chuẩn bị

### 1. Tạo các thư mục cần thiết
Nếu bạn vừa clone project này, hãy chạy lệnh sau để tạo các thư mục bị thiếu:
```powershell
mkdir data/raw, data/augmented, data/processed, models, logs
```

### 2. Chuẩn bị dữ liệu thô
Tải file dữ liệu mẫu tại đây: [Google Drive - Dữ liệu mẫu](https://drive.google.com/drive/folders/17U5mYwNusa2OKSnUFA4fyNiI6aPK1_vd?usp=sharing)

Sau đó copy file vào đường dẫn: `data/raw/output.csv`.
*Định dạng file CSV: `Chủ đề;Câu hỏi;Câu trả lời` (Phân tách bằng dấu chấm phẩy `;`)*

---

## 📌 Quy trình thực hiện (Bắt buộc)

Dù bạn chọn chạy bằng CPU hay GPU, bạn đều phải thực hiện 2 bước chuẩn bị dữ liệu sau:

1. **Tăng cường dữ liệu (Data Augmentation):**
   ```powershell
   python src/augment_data.py
   ```
   *Lệnh này sẽ đọc từ `data/raw/output.csv` và tạo ra `data/augmented/output_augmented.csv`.*

2. **Tiền xử lý và Tokenize:**
   ```powershell
   python src/preprocess.py
   ```
   *Lệnh này sẽ tạo ra `vocab.pkl` và `data.pkl` trong thư mục `data/processed/`.*

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

---

## 🌐 Giao diện Web
Hệ thống đã được nâng cấp lên ứng dụng Web phong cách "Cyber-Academic" hiện đại, thay thế cho giao diện Terminal truyền thống.

**Tính năng nổi bật:**
- **Backend FastAPI:** Tải model vào RAM/VRAM một lần duy nhất (`lifespan`), cung cấp REST API tốc độ cao.
- **Frontend Hiện đại:** Vanilla JS kết hợp giao diện Glassmorphism, Dark Mode, Typing Effect.
- **Hiển thị Toán học:** Tích hợp `marked.js` và `KaTeX` để tự động render Markdown và các công thức Toán rời rạc (LaTeX).
- **Báo cáo hiệu năng:** Hiển thị thời gian Inference (ms) và số lượng tokens sinh ra ở mỗi câu trả lời.

**Cách khởi động Web UI:**
1. Đảm bảo bạn đã huấn luyện model thành công (đã có file `models/model.pt` và `data/processed/vocab.pkl`).
2. Chạy lệnh sau để bật máy chủ:
   ```powershell
   python api.py
   ```
3. Mở trình duyệt (Chrome/Edge) và truy cập vào: **[http://localhost:8000](http://localhost:8000)**
