"""
FastAPI Backend cho Transformer Chatbot - Toán Rời Rạc
Bọc model PyTorch thành REST API endpoint.
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


def tokenize(text):
    """Tokenize text - giống hệt logic trong chat.py và preprocess.py"""
    text = str(text).lower()
    text = re.sub(r'([.,!?()])', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip().split()


def load_model():
    """Load model và vocab vào bộ nhớ 1 lần duy nhất"""
    global model, word2idx, idx2word, device, model_info

    # Thêm src vào path để import model
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from model import make_model

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
    description="API cho chatbot Toán Rời Rạc sử dụng kiến trúc Transformer Decoder-only",
    version="1.0.0",
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
    temperature: float = 0.1
    top_k: int = 1
    max_tokens: int = 100


class ChatResponse(BaseModel):
    response: str
    tokens_generated: int
    inference_time_ms: float


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
    """Gửi câu hỏi và nhận câu trả lời từ chatbot (hỗ trợ multi-turn history)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa được load.")

    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Tin nhắn không được để trống.")

    # ─── Build prompt from current question ONLY ───
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
    start_time = time.perf_counter()
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

    return ChatResponse(
        response=response if response else "Xin lỗi, mình chưa hiểu câu hỏi này.",
        tokens_generated=len(response_tokens),
        inference_time_ms=round(elapsed_ms, 2),
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
