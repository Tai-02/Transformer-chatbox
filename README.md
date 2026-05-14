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

```text
Transformer/
├── data/
│   ├── raw/           <-- Dữ liệu gốc (output.csv)
│   ├── augmented/     <-- Dữ liệu đã nhân bản (output_augmented.csv)
│   └── processed/     <-- Dữ liệu đã mã hóa (vocab.pkl, data.pkl)
├── models/            <-- Lưu trữ 'model.pt' (Pre-trained Transformer)
├── src/               
│   ├── brain_rag.py   <-- 🧠 Bộ não lai RAG (Core)
│   ├── model.py       <-- Kiến trúc mạng Transformer
│   ├── augment_data.py<-- Script nhân bản dữ liệu
│   └── preprocess.py  <-- Tokenization
├── web/               <-- Giao diện Frontend (HTML, CSS, JS)
├── api.py             <-- Máy chủ FastAPI điều phối hệ thống
└── README.md
```

---

## 🛠 Hướng dẫn chạy Hệ thống

### 1. Chuẩn bị Dữ liệu & Huấn luyện (Nếu chưa có model)
Dù bạn chọn chạy bằng CPU hay GPU, quy trình chuẩn bị dữ liệu là bắt buộc:
```powershell
python src/augment_data.py  # Nhân bản dữ liệu
python src/preprocess.py    # Xây dựng từ điển (Vocab)
python src/train.py         # Huấn luyện mô hình
```

### 2. Khởi động Máy chủ API & Giao diện Web (Khuyên dùng)
Dự án được trang bị một hệ thống Web cực kỳ hiện đại mang phong cách Cyber-Academic. Để bật Web:

```powershell
python api.py
```
👉 Sau đó, mở trình duyệt và truy cập: **`http://localhost:8000`**

Dưới mỗi câu trả lời của Bot, bạn sẽ thấy các **Huy hiệu (Badge)** nhiều màu sắc (Xanh/Tím/Cam) minh bạch hóa quá trình suy luận và độ tin cậy (Confidence Score) của Bot!

---

## 🛠 Thông số kỹ thuật Model (Cấu hình Large)
- **Kiến trúc:** Transformer Decoder (8 Layers, 16 Heads)
- **D_Model:** 512 | **Tham số:** ~26 Triệu
- **Dropout:** 0.4 (Chống học vẹt tối đa)
- **Cơ chế Attention:** Masked Multi-Head Self-Attention
- **API Server:** FastAPI + Uvicorn

