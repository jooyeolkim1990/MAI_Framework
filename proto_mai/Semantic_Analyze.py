# =======================
# 🔹 MAI Cross-Round Semantic Convergence Analyzer
# =======================
# Requirements: ChatGPT_to_Round3.txt, Gemini_to_Round3.txt 업로드 필수
# 목적: 라운드 간 논리적 수렴 정도 (semantic similarity) 계산
# =======================

import re
import math
import pandas as pd
from collections import Counter

# --- Tokenization & Vectorization ---
def tokenize(text):
    return re.findall(r'\b[\w가-힣]+\b', text.lower())

def weight_vector(tokens):
    return Counter(tokens)

def cosine_py(v1, v2):
    intersection = set(v1.keys()) & set(v2.keys())
    numerator = sum([v1[x] * v2[x] for x in intersection])
    sum1 = sum([v1[x] ** 2 for x in v1.keys()])
    sum2 = sum([v2[x] ** 2 for x in v2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return float(numerator) / denominator if denominator else 0.0

# --- 파일 불러오기 ---
chatgpt_text = open("ChatGPT_to_Round3.txt", encoding="utf-8").read()
gemini_text = open("Gemini_to_Round3.txt", encoding="utf-8").read()

# --- [RoundX_PhaseY] 태그별 내용 분리 ---
def read_phases(text):
    phases = {}
    current_tag = None
    for line in text.splitlines():
        tag = re.findall(r'\[Round(\d+)_Phase([\d\-]+)\]', line)
        if tag:
            current_tag = f"R{tag[0][0]}_P{tag[0][1]}"
            phases[current_tag] = ""
        elif current_tag:
            phases[current_tag] += line + "\n"
    return phases

gpt_phases = read_phases(chatgpt_text)
gem_phases = read_phases(gemini_text)

# --- 수동 정의된 cross-round mapping ---
mappings = [
    ("R1_P1", "R1_P2-1"),
    ("R1_P2-2", "R2_P1"),
    ("R2_P1", "R2_P2-1"),
    ("R2_P2-2", "R3_P1"),
    ("R3_P1", "R3_P2")
]

# --- 의미 유사도 계산 ---
results = []
for left, right in mappings:
    left_text = gem_phases.get(left, gpt_phases.get(left, ""))
    right_text = gpt_phases.get(right, gem_phases.get(right, ""))
    
    if left_text and right_text:
        vec1 = weight_vector(tokenize(left_text))
        vec2 = weight_vector(tokenize(right_text))
        sim = cosine_py(vec1, vec2)
        results.append((left, right, round(sim, 3), round(1 - sim, 3)))
    else:
        results.append((left, right, None, None))

# --- 결과 정리 ---
df = pd.DataFrame(results, columns=["From", "To", "Semantic_Similarity", "Semantic_Drift"])
print(df)

# --- 라운드별 평균 수렴도 ---
avg_sim = df["Semantic_Similarity"].dropna().mean()
print(f"\n📈 Average Semantic Convergence (MAI Score): {round(avg_sim, 3)}")

# --- 시각화 (옵션) ---
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))          # 그래프 크기 지정
    plt.plot(range(1, len(df)+1),
             df["Semantic_Similarity"],
             marker="o",
             linewidth=2)
    plt.title("MAI Cross-Round Semantic Convergence", fontsize=14)
    plt.xlabel("Mapping Step", fontsize=12)
    plt.ylabel("Semantic Similarity", fontsize=12)
    plt.grid(True)
    plt.show()
except Exception as e:
    print("Visualization skipped:", e)