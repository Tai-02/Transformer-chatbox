import pandas as pd
import random
import os

INPUT_CSV = 'data/raw/output.csv'
OUTPUT_CSV = 'data/augmented/output_augmented.csv'

def augment_question(question):
    variations = [question]
    if question.endswith("là gì?"):
        base = question[:-6].strip()
        variations.extend([f"Thế nào là {base}?", f"Định nghĩa {base} là gì?", f"Khái niệm {base} được hiểu như thế nào?", f"Cho biết định nghĩa của {base}."])
    if "như thế nào?" in question:
        variations.extend([question.replace("như thế nào?", "ra sao?"), question.replace("như thế nào?", "theo cách nào?")])
    if question.lower().startswith("tại sao"):
        variations.extend(["Vì sao" + question[7:], "Lý do tại sao" + question[7:]])
    if question.lower().startswith("nêu"):
        variations.extend(["Hãy nêu" + question[3:], "Cho biết" + question[3:]])
    if "có bao nhiêu" in question:
        variations.extend([question.replace("có bao nhiêu", "bao nhiêu"), question.replace("có bao nhiêu", "số lượng")])
    return list(set(variations))

def main():
    if not os.path.exists(INPUT_CSV): return
    df = pd.read_csv(INPUT_CSV, sep=';')
    df.columns = ['topic', 'question', 'answer']
    df = df.dropna(subset=['question', 'answer'])
    
    augmented_data = []
    for _, row in df.iterrows():
        for v in augment_question(row['question']):
            augmented_data.append({'topic': row['topic'], 'question': v, 'answer': row['answer']})

    pd.DataFrame(augmented_data).to_csv(OUTPUT_CSV, sep=';', index=False, header=False)

if __name__ == "__main__":
    main()