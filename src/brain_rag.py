"""
🧠 HYBRID RAG BRAIN - Hệ Thống Não Lai 3 Lớp
==============================================
Kiến trúc xử lý câu hỏi thông minh cho Chatbot Toán Rời Rạc.

Lớp 1: Exact Match (TF-IDF) - Khớp chính xác bằng từ khóa
Lớp 2: Semantic Search (Embedding) - Tìm kiếm ngữ nghĩa bằng Vector
Lớp 3: Transformer Generate - Sáng tạo câu trả lời bằng AI

Tác giả: Nhóm đồ án Toán Rời Rạc - HCMUE
"""

import os
import re
import math
import pickle
import pandas as pd
from collections import Counter

# ─── Đường dẫn dữ liệu ───
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'output.csv')
AUGMENTED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'augmented', 'output_augmented.csv')


def tokenize_vn(text):
    """Tách từ tiếng Việt đơn giản (chuẩn hóa + split)"""
    text = str(text).lower().strip()
    # Tách dấu câu ra khỏi từ
    text = re.sub(r'([.,!?();:\"\'\\-])', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip().split()


# ═══════════════════════════════════════════════════════════════
#  STOPWORDS TIẾNG VIỆT (Các từ vô nghĩa cần loại bỏ khi so sánh)
# ═══════════════════════════════════════════════════════════════
STOPWORDS = {
    'là', 'gì', 'của', 'và', 'trong', 'có', 'cho', 'một', 'các', 'được',
    'để', 'với', 'này', 'đó', 'từ', 'theo', 'về', 'hay', 'hoặc', 'không',
    'thì', 'mà', 'khi', 'nếu', 'như', 'vì', 'tại', 'bởi', 'đã', 'đang',
    'sẽ', 'cũng', 'vẫn', 'rất', 'nào', 'ai', 'đâu', 'sao', 'thế',
    'bạn', 'tôi', 'mình', 'em', 'anh', 'chị', 'ơi', 'nhé', 'nha',
    'ạ', 'dạ', 'vậy', 'đi', 'rồi', 'à', 'ừ', 'hả', 'hã', 'hen',
    'giúp', 'giùm', 'hộ', 'tớ', 'cậu', 'bot', 'chatbot',
    'hãy', 'xin', 'cho', 'biết', 'nêu', 'trình', 'bày', 'giải', 'thích',
    'định', 'nghĩa', 'thế', 'nào', 'ra', 'sao', 'cái', 'con', 'người',
}


class HybridBrain:
    """
    Bộ não lai 3 lớp cho Chatbot Toán Rời Rạc.
    
    Lớp 1 (TF-IDF): So khớp từ khóa nhanh chóng
    Lớp 2 (Semantic Embedding): Tìm kiếm ngữ nghĩa bằng vector cosine
    Lớp 3 (Transformer): Để lại cho api.py gọi model.generate()
    """

    def __init__(self):
        self.qa_pairs = []          # Danh sách (câu hỏi, câu trả lời, chủ đề)
        self.question_tokens = []   # Danh sách tokens đã tách từ của mỗi câu hỏi
        self.idf = {}               # Inverse Document Frequency cho TF-IDF
        self.doc_vectors = []       # TF-IDF vectors cho mỗi câu hỏi
        self.is_loaded = False

    def load_knowledge_base(self):
        """
        Nạp toàn bộ kiến thức từ file CSV vào bộ nhớ.
        Ưu tiên file augmented (đã nhân bản), fallback về file gốc.
        """
        # Chọn file dữ liệu tốt nhất
        if os.path.exists(AUGMENTED_PATH):
            data_path = AUGMENTED_PATH
            print(f"[RAG] 📚 Đang nạp tri thức từ: output_augmented.csv")
        elif os.path.exists(DATA_PATH):
            data_path = DATA_PATH
            print(f"[RAG] 📚 Đang nạp tri thức từ: output.csv")
        else:
            print("[RAG] ❌ Không tìm thấy file dữ liệu CSV!")
            return False

        try:
            df = pd.read_csv(data_path, sep=',', encoding='utf-8-sig')
            df.columns = ['topic', 'question', 'answer']
            # Loại bỏ các dòng trùng lặp (chỉ giữ câu hỏi gốc duy nhất)
            df = df.drop_duplicates(subset=['question'], keep='first')
            df = df.dropna(subset=['question', 'answer'])
        except Exception as e:
            print(f"[RAG] ❌ Lỗi đọc CSV: {e}")
            return False

        # Nạp từng cặp Q&A vào bộ nhớ
        for _, row in df.iterrows():
            q = str(row['question']).strip()
            a = str(row['answer']).strip()
            topic = str(row['topic']).strip()
            if q and a:
                self.qa_pairs.append((q, a, topic))
                self.question_tokens.append(tokenize_vn(q))

        if not self.qa_pairs:
            print("[RAG] ❌ Không có cặp Q&A hợp lệ!")
            return False

        # Xây dựng chỉ mục TF-IDF
        self._build_tfidf_index()

        self.is_loaded = True
        print(f"[RAG] ✅ Nạp thành công {len(self.qa_pairs)} cặp Q&A vào bộ nhớ.")
        return True

    def _build_tfidf_index(self):
        """
        Xây dựng chỉ mục TF-IDF cho toàn bộ câu hỏi trong cơ sở tri thức.
        TF-IDF giúp đánh giá mức độ quan trọng của từng từ trong mỗi câu hỏi.
        """
        N = len(self.question_tokens)
        # Đếm số tài liệu chứa mỗi từ (Document Frequency)
        df_count = Counter()
        for tokens in self.question_tokens:
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df_count[t] += 1

        # Tính IDF: log(N / DF) - Từ càng hiếm thì IDF càng cao
        self.idf = {}
        for word, count in df_count.items():
            self.idf[word] = math.log((N + 1) / (count + 1)) + 1  # Smoothed IDF

        # Tính TF-IDF vector cho mỗi câu hỏi
        self.doc_vectors = []
        for tokens in self.question_tokens:
            vec = self._compute_tfidf_vector(tokens)
            self.doc_vectors.append(vec)

    def _compute_tfidf_vector(self, tokens):
        """Tính vector TF-IDF cho một danh sách tokens"""
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = {}
        for word, count in tf.items():
            tf_val = count / total
            idf_val = self.idf.get(word, 1.0)
            vec[word] = tf_val * idf_val
        return vec

    def _cosine_similarity(self, vec_a, vec_b):
        """Tính độ tương đồng Cosine giữa 2 vector TF-IDF (sparse dict)"""
        # Tìm các từ chung
        common_words = set(vec_a.keys()) & set(vec_b.keys())
        if not common_words:
            return 0.0

        # Tích vô hướng (Dot product)
        dot = sum(vec_a[w] * vec_b[w] for w in common_words)

        # Độ dài (Magnitude)
        mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)

    def _extract_keywords(self, text):
        """Trích xuất từ khóa quan trọng (loại bỏ stopwords)"""
        tokens = tokenize_vn(text)
        keywords = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        return keywords

    # ═══════════════════════════════════════════════════════════
    #  LỚP 1: EXACT MATCH (Khớp chính xác bằng từ khóa)
    # ═══════════════════════════════════════════════════════════
    def layer1_exact_match(self, user_question, threshold=0.85):
        """
        Lớp 1: Tìm kiếm chính xác bằng TF-IDF.
        So sánh câu hỏi của người dùng với toàn bộ câu hỏi trong database.
        
        Args:
            user_question: Câu hỏi của người dùng
            threshold: Ngưỡng tương đồng tối thiểu (0.85 = gần như khớp hoàn toàn)
            
        Returns:
            (answer, score, topic, matched_question) hoặc None nếu không tìm thấy
        """
        if not self.is_loaded:
            return None

        user_tokens = tokenize_vn(user_question)
        user_vec = self._compute_tfidf_vector(user_tokens)

        best_score = 0.0
        best_idx = -1

        for i, doc_vec in enumerate(self.doc_vectors):
            score = self._cosine_similarity(user_vec, doc_vec)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= threshold and best_idx >= 0:
            q, a, topic = self.qa_pairs[best_idx]
            return {
                'answer': a,
                'score': round(best_score, 4),
                'topic': topic,
                'matched_question': q,
                'layer': 1,
                'method': 'TF-IDF Exact Match'
            }

        return None

    # ═══════════════════════════════════════════════════════════
    #  LỚP 2: SEMANTIC SEARCH (Tìm kiếm ngữ nghĩa bằng Keyword Overlap)
    # ═══════════════════════════════════════════════════════════
    def layer2_semantic_search(self, user_question, threshold=0.45):
        """
        Lớp 2: Tìm kiếm ngữ nghĩa dựa trên Keyword Overlap + TF-IDF.
        Loại bỏ stopwords để tập trung vào từ khóa chuyên môn,
        sau đó tính tỷ lệ trùng khớp từ khóa quan trọng.
        
        Args:
            user_question: Câu hỏi của người dùng
            threshold: Ngưỡng tương đồng ngữ nghĩa tối thiểu
            
        Returns:
            (answer, score, topic, matched_question) hoặc None nếu không tìm thấy
        """
        if not self.is_loaded:
            return None

        user_keywords = set(self._extract_keywords(user_question))
        if not user_keywords:
            return None

        best_score = 0.0
        best_idx = -1

        for i, (q, a, topic) in enumerate(self.qa_pairs):
            doc_keywords = set(self._extract_keywords(q))
            if not doc_keywords:
                continue

            # Tính Jaccard Similarity (Tỷ lệ giao / hợp của 2 tập từ khóa)
            intersection = user_keywords & doc_keywords
            union = user_keywords | doc_keywords
            jaccard = len(intersection) / len(union) if union else 0

            # Kết hợp với TF-IDF cosine để có kết quả chuẩn hơn
            tfidf_score = self._cosine_similarity(
                self._compute_tfidf_vector(tokenize_vn(user_question)),
                self.doc_vectors[i]
            )

            # Trọng số: 40% Jaccard + 60% TF-IDF
            combined_score = 0.4 * jaccard + 0.6 * tfidf_score

            if combined_score > best_score:
                best_score = combined_score
                best_idx = i

        if best_score >= threshold and best_idx >= 0:
            q, a, topic = self.qa_pairs[best_idx]
            return {
                'answer': a,
                'score': round(best_score, 4),
                'topic': topic,
                'matched_question': q,
                'layer': 2,
                'method': 'Semantic Keyword Search'
            }

        return None

    # ═══════════════════════════════════════════════════════════
    #  TRUNG TÂM ĐIỀU PHỐI: Gọi lần lượt từng Lớp
    # ═══════════════════════════════════════════════════════════
    def search(self, user_question):
        """
        Hàm tìm kiếm chính - Điều phối 3 lớp.
        
        Quy trình:
        1. Thử Lớp 1 (Exact Match) → Nếu khớp cao ≥ 85% → Trả về ngay
        2. Thử Lớp 2 (Semantic Search) → Nếu khớp ≥ 45% → Trả về
        3. Nếu cả 2 lớp đều bó tay → Trả về None → api.py sẽ gọi Lớp 3 (Transformer)
        
        Returns:
            dict chứa answer, score, layer, method... hoặc None
        """
        if not self.is_loaded:
            return None

        # ── Lớp 1: Khớp chính xác ──
        result = self.layer1_exact_match(user_question, threshold=1)
        if result:
            print(f"[RAG] 🛡️  Lớp 1 (Exact Match) | Score: {result['score']} | Q: {result['matched_question'][:50]}...")
            return result

        # ── Lớp 2: Tìm kiếm ngữ nghĩa ──
        result = self.layer2_semantic_search(user_question, threshold=1)
        if result:
            print(f"[RAG] 🔮 Lớp 2 (Semantic)    | Score: {result['score']} | Q: {result['matched_question'][:50]}...")
            return result

        # ── Lớp 3: Chuyển giao cho Transformer ──
        print(f"[RAG] 🧬 Lớp 3 (Transformer) | Không tìm thấy câu tương tự → Chuyển cho AI sáng tạo...")
        return None
