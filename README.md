# 🤖 Transformer Chatbot - Toán Rời Rạc
*(Tích hợp Kiến trúc Não lai Hybrid RAG 3 Lớp)*

Dự án xây dựng một AI Chatbot giải đáp câu hỏi môn Toán rời rạc. Bot sử dụng kiến trúc **Hybrid RAG (Retrieval-Augmented Generation)** kết hợp cùng mô hình **Transformer (Decoder-only)** tự huấn luyện từ con số 0, nhằm đảm bảo độ chính xác 100% về mặt học thuật đồng thời giữ được sự linh hoạt trong giao tiếp.

---

## 🧠 Kiến Trúc Não Lai (Hybrid 3-Layer Architecture)
Thay vì phó mặc hoàn toàn cho Transformer (dễ sinh ra ảo giác - Hallucination), dự án áp dụng hệ thống điều phối 3 lớp thông minh:

- 🛡️ **Lớp 1 (Exact Match - Khớp chính xác):** Sử dụng thuật toán TF-IDF. Chặn đứng các câu hỏi trùng khớp 100% với sách giáo khoa và trả về đáp án siêu tốc trong `0.02s`.
- 🔮 **Lớp 2 (Semantic Search - Tìm kiếm ngữ nghĩa):** Tích hợp Vector Embedding để tính toán Jaccard + TF-IDF. Dù người dùng gõ tiếng lóng, sai chính tả hay đảo từ, hệ thống vẫn "hiểu ý" và trích xuất đúng định nghĩa toán học. (Ngưỡng an toàn tối ưu: `0.45`).
- 🧬 **Lớp 3 (Transformer AI - Suy luận sáng tạo):** Nếu câu hỏi hoàn toàn mới (Vượt ngoài sách), mạng Nơ-ron Transformer 26 triệu tham số sẽ tự động tiếp quản và tự múa bút sáng tác câu trả lời.

👉 **Tính năng bổ sung:** Tích hợp **Lớp 0 (Chit-Chat Filter)** giúp Bot có "Kỹ năng mềm" - Biết chào hỏi và cảm ơn cực kỳ dẻo miệng!

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

