# 🤖 Transformer Chatbot - Toán Rời Rạc
*(Mô hình Generative AI dựa trên Transformer)*

Dự án xây dựng một AI Chatbot giải đáp câu hỏi môn Toán rời rạc. Bot sử dụng kiến trúc **Transformer (Decoder-only)** tự huấn luyện từ con số 0 để tiếp nhận câu hỏi và tự động sinh ra (generate) câu trả lời dựa trên lượng tri thức đã học được.

---

## 🧠 Kiến Trúc Hệ Thống (Pure Transformer)
Trưởng nhóm quyết định loại bỏ hoàn toàn hệ thống truy xuất (RAG - Lớp 1 & 2) để sử dụng **100% sức mạnh của mạng Nơ-ron Transformer**.

- 🧬 **Transformer AI (Suy luận sáng tạo):** Mạng Nơ-ron Transformer sẽ tự động phân tích ngữ cảnh câu hỏi và tiến hành quá trình Decoding để tự múa bút sáng tác câu trả lời cho bất kỳ vấn đề nào về Toán rời rạc.
- 🛡️ **Lớp 0 (Chit-Chat Filter):** Kỹ năng giao tiếp cơ bản (chào hỏi, cảm ơn).

---

## 📂 Cấu trúc thư mục

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
1. **Tải dữ liệu thô**: 
   Tải file dữ liệu mẫu tại đây: [Google Drive - Dữ liệu mẫu](https://drive.google.com/drive/folders/17U5mYwNusa2OKSnUFA4fyNiI6aPK1_vd?usp=sharing)
   Sau đó, copy file vào đường dẫn: `data/raw/output.csv`.
   *(Lưu ý: Định dạng CSV sử dụng dấu chấm phẩy `;` làm dấu phân cách)*

2. **Tăng cường dữ liệu**:
   ```powershell
   python src/augment_data.py
   ```
3. **Tiền xử lý và Tokenize**:
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
