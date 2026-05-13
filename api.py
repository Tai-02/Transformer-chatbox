"""
FastAPI Backend cho Transformer Chatbot - Toán Rời Rạc
======================================================
Kiến trúc Hybrid 3 Lớp:
  Lớp 1: TF-IDF Exact Match (Khớp chính xác)
  Lớp 2: Semantic Keyword Search (Tìm kiếm ngữ nghĩa)
  Lớp 3: Transformer Generate (AI sáng tạo câu trả lời)
"""

import torch
import pickle
import os
import re
import sys
import time

# Fix encoding trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ─── Paths ───
VOCAB_PATH = 'data/processed/vocab.pkl'
SAVE_PATH = 'models/model.pt'

# ─── Global state ───
model = None
word2idx = None
idx2word = None
device = None
model_info = {}
hybrid_brain = None  # 🧠 Bộ não lai Hybrid RAG


def tokenize(text):
    """Tokenize text - giống hệt logic trong chat.py và preprocess.py"""
    text = str(text).lower()
    text = re.sub(r'([.,!?()])', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip().split()


def load_model():
    """Load model và vocab vào bộ nhớ 1 lần duy nhất"""
    global model, word2idx, idx2word, device, model_info, hybrid_brain

    # Thêm src vào path để import model
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from model import make_model
    from brain_rag import HybridBrain

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(VOCAB_PATH):
        print(f"[WARN] Khong tim thay vocab: {VOCAB_PATH}. Hay chay preprocess.py truoc.")
        return False
    if not os.path.exists(SAVE_PATH):
        print(f"[WARN] Khong tim thay model: {SAVE_PATH}. Hay chay train.py truoc.")
        return False

    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    word2idx = vocab['word2idx']
    idx2word = vocab['idx2word']

    checkpoint = torch.load(SAVE_PATH, map_location=device)
    model = make_model(
        len(word2idx),
        n_layer=checkpoint['n_l'],
        d_model=checkpoint['d_m'],
        n_head=checkpoint['n_h'],
        dropout=checkpoint['dr']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    model_info = {
        "vocab_size": len(word2idx),
        "d_model": checkpoint['d_m'],
        "n_layers": checkpoint['n_l'],
        "n_heads": checkpoint['n_h'],
        "dropout": checkpoint['dr'],
        "device": str(device),
    }

    print(f"[OK] Model loaded on {device} | Vocab: {len(word2idx)} | d_model: {checkpoint['d_m']}")

    # ─── 🧠 Khởi động Hybrid RAG Brain ───
    hybrid_brain = HybridBrain()
    rag_success = hybrid_brain.load_knowledge_base()
    if rag_success:
        model_info["rag_status"] = "active"
        model_info["rag_qa_pairs"] = len(hybrid_brain.qa_pairs)
        print(f"[OK] Hybrid Brain ACTIVATED! 🧠 3-Layer Architecture Ready!")
    else:
        print("[WARN] RAG Brain could not load. Falling back to Transformer-only mode.")
        model_info["rag_status"] = "inactive"

    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model khi server khởi động (không crash nếu thiếu file)"""
    success = load_model()
    if not success:
        print("[WARN] Server started WITHOUT model. Chat API will return 503.")
    yield


# ─── FastAPI App ───
app = FastAPI(
    title="Transformer Chatbot - Toán Rời Rạc",
    description="API cho chatbot Toán Rời Rạc sử dụng kiến trúc Hybrid 3 Lớp (TF-IDF + Semantic + Transformer)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ───
class HistoryItem(BaseModel):
    role: str       # 'user' or 'bot'
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[HistoryItem]] = None
    temperature: float = 0.3    # Giảm từ 0.5 → 0.3 để bot chọn từ chắc chắn hơn, bớt nói sảng
    top_k: int = 3
    max_tokens: int = 100


class ChatResponse(BaseModel):
    response: str
    tokens_generated: int
    inference_time_ms: float
    source_layer: Optional[int] = None        # Lớp nào đã trả lời (1, 2, hoặc 3)
    source_method: Optional[str] = None       # Phương pháp đã dùng
    confidence: Optional[float] = None        # Độ tin cậy (0.0 - 1.0)
    matched_question: Optional[str] = None    # Câu hỏi gốc đã khớp (nếu có)


# ─── Endpoints ───
@app.get("/api/health")
async def health_check():
    """Kiểm tra trạng thái server và model"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_info": model_info,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Gửi câu hỏi và nhận câu trả lời từ chatbot.
    Hệ thống sẽ tự động chọn lớp phù hợp nhất để trả lời:
      Lớp 1: Khớp chính xác (nhanh nhất, chính xác nhất)
      Lớp 2: Tìm kiếm ngữ nghĩa (thông minh, linh hoạt)
      Lớp 3: Transformer AI (sáng tạo, cho câu hỏi hoàn toàn mới)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa được load.")

    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Tin nhắn không được để trống.")

    start_time = time.perf_counter()

    # ═══════════════════════════════════════════════════════════
    #  LỚP 0: BỘ LỌC GIAO TIẾP XÃ GIAO (CHIT-CHAT)
    # ═══════════════════════════════════════════════════════════
    msg_lower = message.lower()
    if len(msg_lower) < 25:
        if any(w in msg_lower for w in ["chào", "hello", "hi ", "helo", "xin chào", "alo", "ê"]):
            return ChatResponse(
                response="Chào bạn nha! 👋 Mình là Trợ lý AI môn Toán Rời Rạc siêu cấp đáng yêu đây. Hôm nay mình giúp gì được cho bạn nè? 🥰",
                tokens_generated=25,
                inference_time_ms=1.2,
                source_layer=1,
                source_method="Giao tiếp cơ bản",
                confidence=1.0,
                matched_question="[Xã giao]"
            )
        elif any(w in msg_lower for w in ["cảm ơn", "thank", "cám ơn", "tks"]):
            return ChatResponse(
                response="Dạ không có chi! 💖 Chúc bạn làm môn Toán rời rạc đạt điểm 10 tuyệt đối nha! Cần hỏi bài gì cứ gõ vào đây nhé! 🚀",
                tokens_generated=25,
                inference_time_ms=1.2,
                source_layer=1,
                source_method="Giao tiếp cơ bản",
                confidence=1.0,
                matched_question="[Xã giao]"
            )

    # ═══════════════════════════════════════════════════════════
    #  THỬ LỚP 1 & LỚP 2: Hybrid RAG Brain
    # ═══════════════════════════════════════════════════════════
    if hybrid_brain and hybrid_brain.is_loaded:
        rag_result = hybrid_brain.search(message)
        if rag_result:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            import random
            prefixes = [
                "Chào bạn! Theo mình được học thì:\n",
                "Câu này có trong sách giáo khoa nè! Đáp án là:\n",
                "Để mình giải thích phần này cho bạn nhé:\n",
                "Dạ, kiến thức chuẩn về phần này là:\n"
            ]
            suffixes = [
                "\n\nHy vọng câu trả lời này giúp ích cho bạn nha! 🥰",
                "\n\nBạn còn thắc mắc chỗ nào nữa không ạ? 😊",
                "\n\nChúc bạn học tốt môn Toán Rời Rạc nha! 🚀"
            ]
            # Chỉ bọc văn phong thân thiện nếu đáp án là chữ bình thường
            friendly_answer = f"{random.choice(prefixes)}{rag_result['answer']}{random.choice(suffixes)}"
            
            return ChatResponse(
                response=friendly_answer,
                tokens_generated=len(rag_result['answer'].split()),
                inference_time_ms=round(elapsed_ms, 2),
                source_layer=rag_result['layer'],
                source_method=rag_result['method'],
                confidence=rag_result['score'],
                matched_question=rag_result['matched_question'],
            )

    # ═══════════════════════════════════════════════════════════
    #  LỚP 3: Transformer Generate (Fallback)
    # ═══════════════════════════════════════════════════════════
    # Build prompt from current question ONLY
    # Vì mô hình train trên từng câu đơn, việc nối history sẽ làm hỏng Positional Encoding
    current_tokens = tokenize(message)
    if not current_tokens:
        raise HTTPException(status_code=400, detail="Không thể tokenize tin nhắn.")

    # Encode input: <sos> + question_tokens + <sep>
    prompt_indices = (
        [word2idx['<sos>']]
        + [word2idx.get(w, word2idx['<unk>']) for w in current_tokens]
        + [word2idx['<sep>']]
    )

    x = torch.tensor([prompt_indices]).to(device)

    # ─── Generate ───
    with torch.no_grad():
        out_ids = model.generate(
            x,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # ─── Decode output ───
    response_ids = out_ids[0][len(prompt_indices):]
    response_tokens = []
    for i in response_ids:
        sym = idx2word[i.item()]
        if sym == '<eos>':
            break
        if sym not in ['<pad>', '<sos>', '<unk>', '<sep>']:
            response_tokens.append(sym)

    response = " ".join(response_tokens)
    response = re.sub(r'\s+([.,!?()])', r'\1', response)
    response = response.strip().rstrip('?')

    if response:
        import random
        prefixes_ai = [
            "Hmm, phần này trong sách không có ghi rõ, nhưng theo mình suy luận thì:\n",
            "Câu hỏi này lạ quá! Nhưng mình thử ráp kiến thức lại xem sao nhé:\n",
            "Để mình thử tư duy một chút xem... Có vẻ như là:\n"
        ]
        friendly_ai_response = f"{random.choice(prefixes_ai)}{response}\n\n(Lưu ý: Đây là câu trả lời do AI tự sáng tạo, bạn nhớ kiểm tra lại nhé! 🤖)"
    else:
        friendly_ai_response = "Xin lỗi, mình chưa học tới phần kiến thức này rồi. Bạn hỏi câu khác nhé! 😅"

    return ChatResponse(
        response=friendly_ai_response,
        tokens_generated=len(response_tokens),
        inference_time_ms=round(elapsed_ms, 2),
        source_layer=3,
        source_method='Transformer Generate',
        confidence=None,
        matched_question=None,
    )


# ─── Serve Frontend ───
web_dir = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(web_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(web_dir, "assets")), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(web_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
