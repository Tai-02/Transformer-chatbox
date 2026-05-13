import pandas as pd
import random
import os
import re

INPUT_CSV = 'data/raw/output.csv'
OUTPUT_CSV = 'data/augmented/output_augmented.csv'

# ============================================================
# CÁC MẪU ĐA DẠNG HÓA CÂU HỎI (QUESTION AUGMENTATION)
# Mục đích: Ép mô hình hiểu nhiều cách diễn đạt khác nhau
# cho cùng một ý nghĩa, giảm hiện tượng học vẹt (overfitting)
# ============================================================

def augment_question(question):
    """Tạo nhiều biến thể câu hỏi từ 1 câu gốc."""
    variations = [question]
    q_lower = question.lower().strip()

    # --- Mẫu 1: "... là gì?" ---
    if q_lower.endswith("là gì?"):
        base = question[:-6].strip()
        variations.extend([
            f"Thế nào là {base}?",
            f"Định nghĩa {base} là gì?",
            f"Khái niệm {base} được hiểu như thế nào?",
            f"Cho biết định nghĩa của {base}.",
            f"Hãy giải thích {base}.",
            f"Bạn hiểu {base} như thế nào?",
            f"{base} nghĩa là gì?",
            f"Giải thích khái niệm {base}?",
            f"Em muốn hỏi {base} là gì?",
        ])

    # --- Mẫu 2: "Thế nào là ...?" ---
    if q_lower.startswith("thế nào là"):
        base = question[10:].rstrip("?").strip()
        variations.extend([
            f"{base} là gì?",
            f"Định nghĩa {base}?",
            f"Giải thích {base} cho em hiểu?",
        ])

    # --- Mẫu 3: "Định nghĩa ...?" ---
    if q_lower.startswith("định nghĩa"):
        base = question[10:].rstrip("?").strip()
        if "là gì" not in base.lower():
            variations.extend([
                f"{base} là gì?",
                f"Thế nào là {base}?",
                f"Hãy cho biết {base}?",
            ])

    # --- Mẫu 4: "... như thế nào?" ---
    if "như thế nào?" in q_lower:
        variations.extend([
            question.replace("như thế nào?", "ra sao?"),
            question.replace("như thế nào?", "theo cách nào?"),
            question.replace("như thế nào?", "thế nào?"),
        ])

    # --- Mẫu 5: "Tại sao ...?" ---
    if q_lower.startswith("tại sao"):
        base = question[7:].strip()
        variations.extend([
            f"Vì sao {base}",
            f"Lý do tại sao {base}",
            f"Nguyên nhân nào khiến {base}",
        ])

    # --- Mẫu 6: "Nêu ..." ---
    if q_lower.startswith("nêu"):
        base = question[3:].strip()
        variations.extend([
            f"Hãy nêu {base}",
            f"Cho biết {base}",
            f"Liệt kê {base}",
            f"Trình bày {base}",
        ])

    # --- Mẫu 7: "Có bao nhiêu ..." ---
    if "có bao nhiêu" in q_lower:
        variations.extend([
            question.replace("có bao nhiêu", "bao nhiêu"),
            question.replace("có bao nhiêu", "số lượng"),
            question.replace("Có bao nhiêu", "Bao nhiêu"),
        ])

    # --- Mẫu 8: "Công thức ..." ---
    if "công thức" in q_lower:
        variations.extend([
            question.replace("Công thức", "Biểu thức").replace("công thức", "biểu thức"),
            question.replace("Công thức", "Cách tính").replace("công thức", "cách tính"),
        ])

    # --- Mẫu 9: "Điều kiện ..." ---
    if "điều kiện" in q_lower:
        variations.extend([
            question.replace("Điều kiện", "Yêu cầu").replace("điều kiện", "yêu cầu"),
        ])

    # --- Mẫu 10: Thêm prefix lịch sự ---
    if len(variations) < 4 and not q_lower.startswith(("hãy", "cho biết", "bạn")):
        variations.extend([
            f"Cho em hỏi, {question[0].lower() + question[1:]}",
            f"Bạn ơi, {question[0].lower() + question[1:]}",
        ])

    # --- Mẫu 11: "Phát biểu ..." ---
    if q_lower.startswith("phát biểu"):
        base = question[9:].strip()
        variations.extend([
            f"Nội dung của {base}",
            f"Trình bày {base}",
        ])

    # --- Mẫu 12: "Mối liên hệ / Mối quan hệ ..." ---
    if "mối liên hệ" in q_lower:
        variations.append(question.replace("mối liên hệ", "mối quan hệ").replace("Mối liên hệ", "Mối quan hệ"))
    if "mối quan hệ" in q_lower:
        variations.append(question.replace("mối quan hệ", "mối liên hệ").replace("Mối quan hệ", "Mối liên hệ"))

    # Loại bỏ trùng lặp và giữ nguyên thứ tự
    seen = set()
    unique = []
    for v in variations:
        v_clean = v.strip()
        if v_clean and v_clean.lower() not in seen:
            seen.add(v_clean.lower())
            unique.append(v_clean)

    return unique


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Không tìm thấy file: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, sep=';', encoding='utf-8-sig')
    # Xử lý header linh hoạt
    if df.columns[0].strip().lower() in ['chương', 'topic', 'chủ đề']:
        df.columns = ['topic', 'question', 'answer']
    else:
        df.columns = ['topic', 'question', 'answer']
    
    df = df.dropna(subset=['question', 'answer'])

    augmented_data = []
    for _, row in df.iterrows():
        for v in augment_question(str(row['question'])):
            augmented_data.append({
                'topic': row['topic'],
                'question': v,
                'answer': row['answer']
            })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    result_df = pd.DataFrame(augmented_data)
    result_df.to_csv(OUTPUT_CSV, sep=';', index=False, header=False, encoding='utf-8-sig')

    original_count = len(df)
    augmented_count = len(result_df)
    print(f"\n{'='*50}")
    print(f"  AUGMENTATION HOÀN TẤT!")
    print(f"{'='*50}")
    print(f"  Câu hỏi gốc   : {original_count}")
    print(f"  Sau đa dạng hóa: {augmented_count}")
    print(f"  Tỷ lệ tăng     : x{augmented_count/original_count:.1f}")
    print(f"  File output     : {OUTPUT_CSV}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()