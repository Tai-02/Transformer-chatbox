import torch
import pickle
import os
import re
import sys
from src.model import make_model

VOCAB_PATH = 'data/processed/vocab.pkl'
SAVE_PATH = 'models/model.pt'

def tokenize(text):
    text = str(text).lower()
    text = re.sub(r'([.,!?()])', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip().split()

def chat():
    sys.stdout.reconfigure(encoding='utf-8')
    if not os.path.exists(SAVE_PATH): return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    word2idx = vocab['word2idx']
    idx2word = vocab['idx2word']

    checkpoint = torch.load(SAVE_PATH, map_location=device)
    model = make_model(len(word2idx), n_layer=checkpoint['n_l'], d_model=checkpoint['d_m'], n_head=checkpoint['n_h'], dropout=checkpoint['dr'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("\n--- CHATBOT TOÁN RỜI RẠC ---")
    while True:
        try:
            user_input = input("Bạn: ").strip()
            if user_input.lower() in ['quit', 'exit', 'thoát']: break
            
            tokens = tokenize(user_input)
            if not tokens: continue
            
            prompt_indices = [word2idx['<sos>']] + [word2idx.get(w, word2idx['<unk>']) for w in tokens] + [word2idx['<sep>']]
            x = torch.tensor([prompt_indices]).to(device)
            
            with torch.no_grad():                
                out_ids = model.generate(x, max_new_tokens=100, temperature=0.5, top_k=3)
            
            response_ids = out_ids[0][len(prompt_indices):]
            response_tokens = []
            for i in response_ids:
                sym = idx2word[i.item()]
                if sym == '<eos>': break
                if sym not in ['<pad>', '<sos>', '<unk>', '<sep>']:
                    response_tokens.append(sym)
            
            response = " ".join(response_tokens)
            response = re.sub(r'\s+([.,!?()])', r'\1', response)
            response = response.strip().rstrip('?')
            print(f"Bot: {response}\n")
            
        except KeyboardInterrupt: break
        except Exception as e: print(f"Error: {e}")

if __name__ == "__main__":
    chat()