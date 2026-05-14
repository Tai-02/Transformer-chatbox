import pandas as pd
import random
import os

INPUT_CSV = 'data/raw/output.csv'
OUTPUT_CSV = 'data/augmented/output_augmented.csv'

def augment_question(question):
    variations = [question]
    q_lower = question.lower()
    
    # Core definition logic
    if question.endswith("là gì?"):
        base = question[:-6].strip()
        variations.extend([
            f"Thế nào là {base}?",
            f"Giải thích {base} là gì?",
            f"Cho biết {base} là gì?",
            f"Định nghĩa của {base}."
        ])
    
    # Handle "Công thức" specifically
    if "công thức" in q_lower:
        if "là gì?" in q_lower:
            base = question.replace("là gì?", "").strip()
            variations.append(f"Nêu {base}")
            variations.append(f"Cho biết {base}")

    if "như thế nào?" in q_lower:
        variations.append(question.replace("như thế nào?", "ra sao?"))
        variations.append(question.replace("như thế nào?", "theo cách nào?"))

    if q_lower.startswith("tại sao"):
        variations.append("Vì sao" + question[7:])
        variations.append("Lý do tại sao" + question[7:])

    if q_lower.startswith("nêu"):
        variations.append("Hãy nêu" + question[3:])
        variations.append("Cho biết" + question[3:])

    if "có bao nhiêu" in q_lower:
        variations.append(question.replace("có bao nhiêu", "bao nhiêu"))
            
    return list(set(variations))

def main():
    if not os.path.exists(INPUT_CSV): return
    df = pd.read_csv(INPUT_CSV)
    df.columns = ['topic', 'question', 'answer']
    df = df.dropna(subset=['question', 'answer'])
    
    augmented_data = []
    for _, row in df.iterrows():
        for v in augment_question(row['question']):
            augmented_data.append({'topic': row['topic'], 'question': v, 'answer': row['answer']})

    pd.DataFrame(augmented_data).to_csv(OUTPUT_CSV, index=False, header=True, encoding='utf-8-sig')

if __name__ == "__main__":
    main()