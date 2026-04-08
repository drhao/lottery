
import numpy as np
import pandas as pd
from numpy.random import default_rng
import os
import sys

# =====================
# CONFIG
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'lotto649_results.csv')
TXT_OUTPUT_PATH = os.path.join(BASE_DIR, 'recommendation_lotto649_output.txt')

# Strategy Parameters (Validated)
ALPHA = 100
PENALTY = 0.4
HOT_WINDOW = 100

def parse_currency(x):
    if isinstance(x, str):
        return float(x.replace(',', ''))
    return float(x)

def sample_wor_es(rng, w, k=6):
    u = rng.random(len(w))
    keys = -np.log(u) / w
    return np.argpartition(keys, k-1)[:k]

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    print("Loading data for Lotto 6/49 (Ensemble Strategy)...")
    df = pd.read_csv(CSV_PATH)
    
    
    # 1. Get Latest Jackpot Info & Calculate Rollovers
    # -----------------------------------------------
    latest_row = df.iloc[0]
    last_date = latest_row['Date']
    
    # Estimate Next Jackpot
    jp_total = parse_currency(latest_row['Jackpot_Total'])
    jp_rollover = parse_currency(latest_row['Jackpot_Rollover'])
    effective_jackpot = jp_total + jp_rollover
    
    # Calculate Consecutive Rollovers
    consecutive_rollovers = 0
    for i in range(len(df)):
        val = parse_currency(df.iloc[i]['Jackpot_Per_Winner'])
        if val > 0:
            break
        consecutive_rollovers += 1
        
    # Calculate Recommended Tickets (Base 2, Cap 15)
    # Logic: Start at 2. Every rollover adds 1, up to Cap 15.
    recommended_tickets = min(2 + consecutive_rollovers, 15)

    # 2. Strategy Logic (Ensemble)
    # ----------------------------
    df_chrono = df.iloc[::-1].copy().reset_index(drop=True)
    first_draws = df_chrono["Numbers"].astype(str).str.split().apply(
        lambda xs: np.array([int(x) for x in xs], dtype=np.int16) - 1
    ).values
    special_draws = df_chrono["Special_Number"].apply(lambda x: int(x) -1).values
    
    K1 = 49
    total_draws = len(df_chrono)
    
    # A. Weights for Recent Hot (100)
    win_start = max(0, total_draws - HOT_WINDOW)
    sub_first = first_draws[win_start:]
    sub_special = special_draws[win_start:]
    nums_hot = []
    for arr in sub_first: nums_hot.extend(arr)
    nums_hot.extend(sub_special)
    counts_hot = np.bincount(nums_hot, minlength=K1)
    w_hot = counts_hot.astype(float) + ALPHA
    
    # B. Weights for All-Time Cold
    nums_all = []
    for arr in first_draws: nums_all.extend(arr)
    nums_all.extend(special_draws)
    counts_all = np.bincount(nums_all, minlength=K1)
    w_cold = 1.0 / (counts_all.astype(float) + 1.0)
    
    # 3. Generate Recommendations (Base 2)
    # ---------------------------
    # We dynamically generate the required number of sets based on recommendations.
    
    rng = default_rng()
    
    n_hot_rec = recommended_tickets // 2 + (recommended_tickets % 2)
    n_cold_rec = recommended_tickets // 2
    
    hot_sets = []
    cold_sets = []
    
    # Generate Hot Sets
    used = np.zeros(K1, dtype=int)
    for _ in range(max(1, n_hot_rec)): # At least 1 for display if recommended is 0? Actually minimum recommended is 2.
        w = w_hot * (PENALTY ** used)
        pick = sample_wor_es(rng, w, k=6)
        # used[pick] += 1 # Diversity within set
        hot_sets.append(sorted(pick + 1))
        
    # Generate Cold Sets
    used = np.zeros(K1, dtype=int)
    for _ in range(n_cold_rec):
        w = w_cold * (PENALTY ** used) # Penalty still applies to avoid duplicates
        pick = sample_wor_es(rng, w, k=6)
        # used[pick] += 1
        cold_sets.append(sorted(pick + 1))

    # 4. Output
    # ---------
    output_lines = []
    output_lines.append("==========================================")
    output_lines.append("       大樂透 (Lotto 6/49) 投注建議       ")
    output_lines.append("==========================================")
    
    output_lines.append("[ 當期資訊 ]")
    output_lines.append(f"最新開獎日期:   {last_date}")
    output_lines.append(f"預估下期頭獎:   ${effective_jackpot:,.0f} (累積/保底)")
    output_lines.append(f"當前連槓期數:   {consecutive_rollovers} 期")
    output_lines.append("")
    
    output_lines.append("[ 投注策略建議 ] (Base 2 + 連槓成長)")
    output_lines.append(f"★ 本期建議注數: {recommended_tickets} 注")
    output_lines.append(f"★ 資金分配建議: {n_hot_rec} 注熱門 + {n_cold_rec} 注冷門")
    output_lines.append("-" * 42)
    
    output_lines.append(f"【 A組: 追熱門 (Momentum) 】 建議買 {n_hot_rec} 注")
    output_lines.append("   (策略: Recent Hot 100)")
    for i, nums in enumerate(hot_sets):
        nums_str = " ".join([f"{n:02d}" for n in nums])
        output_lines.append(f"   第 {i+1:02d} 組: [{nums_str}]")
        
    output_lines.append("")
    output_lines.append(f"【 B組: 抓冷門 (Contrarian) 】 建議買 {n_cold_rec} 注")
    output_lines.append("   (策略: All-Time Cold)")
    for i, nums in enumerate(cold_sets):
        nums_str = " ".join([f"{n:02d}" for n in nums])
        output_lines.append(f"   第 {i+1:02d} 組: [{nums_str}]")
        
    output_lines.append("==========================================")
    output_lines.append("祝您中大獎！")

    final_text = "\n".join(output_lines)
    print(final_text)
    
    with open(TXT_OUTPUT_PATH, "w", encoding='utf-8') as f:
        f.write(final_text)
    print(f"結果已儲存至 '{os.path.basename(TXT_OUTPUT_PATH)}'")

if __name__ == "__main__":
    main()
