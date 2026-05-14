import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.d_model = d_model

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Model(nn.Module):
    def __init__(self, vocab_size, n_layer=4, d_model=256, n_head=8, dropout=0.1, max_len=512):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, d_model),
            wpe = nn.Embedding(max_len, d_model),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(d_model),
        ))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, label_smoothing=0.1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, repetition_penalty=1.2):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            for idx_row in range(idx.size(0)):
                for token_id in set(idx[idx_row].tolist()):
                    if logits[idx_row, token_id] > 0:
                        logits[idx_row, token_id] /= repetition_penalty
                    else:
                        logits[idx_row, token_id] *= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == 2: # <eos> is usually 2, let's be careful
                break
        return idx

def make_model(vocab_size, n_layer=4, d_model=256, n_head=8, dropout=0.1):
    return Model(vocab_size, n_layer, d_model, n_head, dropout)