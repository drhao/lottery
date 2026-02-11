import numpy as np
import pandas as pd
from numpy.random import default_rng

import os

# =====================
# CONFIG
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'super_lotto638_results.csv')
INITIAL_TICKETS = 2
MAX_TICKETS = 10
THRESHOLD = 800000000 # 8-亿門檻
ALPHA = 100
PENALTY_BASE = 0.4
PENALTY_LARGE_BATCH = 0.2
K1, K2 = 38, 8

# =====================
# UTIL
# =====================
def sample_wor_es(rng, w, k=6):
    """Efraimidis–Spirakis weighted sampling without replacement."""
    u = rng.random(len(w))
    keys = -np.log(u) / w
    return np.argpartition(keys, k-1)[:k]

def pick_soft_diverse(rng, base_w, n_tickets, penalty, k=6, total_nums=K1):
    """N tickets, soft diversity within the same draw batch."""
    used = np.zeros(total_nums, dtype=int)
    tickets = []
    
    for _ in range(n_tickets):
        penalty_factors = penalty ** used
        w = base_w * penalty_factors
        w = np.clip(w, 1e-12, None)
        
        pick = sample_wor_es(rng, w, k=k)
        tickets.append(pick)
        used[pick] += 1
    return tickets

# =====================
# MAIN
# =====================
print("Loading data for recommendation (v3: Dynamic Penalty + 2nd Zone Model)...")
df = pd.read_csv(CSV_PATH)

# 1. Parse History for Weights
first_draws = df["Numbers"].astype(str).str.split().apply(
    lambda xs: np.array([int(x) for x in xs], dtype=np.int16) - 1
)
second_draws = df["Special_Number"].astype(np.int16).to_numpy() - 1

# Latest Draw Info
last_date = df["Date"].iloc[0]
try:
    last_jackpot = df["First_Prize_Total"].astype(str).str.replace(',', '').astype(float).iloc[0]
except:
    last_jackpot = df["First_Prize_Total"].iloc[0]

base_counts = np.zeros(K1, dtype=int)
for arr in first_draws:
    base_counts[arr] += 1

base_counts_s = np.zeros(K2, dtype=int)
for s in second_draws:
    base_counts_s[s] += 1

# 2. Determine Ticket Count (Dynamic Strategy)
try:
    first_prize_won = (df["First_Prize_Per_Winner"].astype(str).str.replace(',', '').astype(float) > 0).to_numpy()
except:
    first_prize_won = (df["First_Prize_Per_Winner"] > 0).to_numpy()

# BUG FIX: CSV is New -> Old, so Index 0 is the Latest Draw
# To calculate current ticket count, we need to know how many consecutive losses since last win
# working backwards from latest (Index 0)

current_n_tickets = INITIAL_TICKETS
# Iterate through recent history to find last win
consecutive_losses = 0
for won in first_prize_won: # starts from latest
    if won:
        break # Found the last jackpot reset
    consecutive_losses += 1

# Cap consecutive losses to determine ticket level
# 0 losses = Initial (2)
# 1 loss = 3
# ...
current_n_tickets = min(INITIAL_TICKETS + consecutive_losses, MAX_TICKETS)

# 3. Apply Threshold Strategy
# If jackpot is below threshold, we should theoretically skip or play minimal (1).
# Based on simulation, skipping is best EV, but for user experience we recommend 0 or 1.
is_below_threshold = last_jackpot < THRESHOLD
if is_below_threshold:
    # Force ticket count to 0 (Recommend Skip) or 1 (Optional)
    # Let's recommend 0 as primary advice to strictly follow strategy
    recommended_tickets = 0
else:
    recommended_tickets = current_n_tickets

# Generate play items anyway in case user wants to override
play_n_tickets = max(1, recommended_tickets)

# 3. Generate Recommendations
rng = default_rng() 
current_penalty = PENALTY_BASE if current_n_tickets < 5 else PENALTY_LARGE_BATCH

# First Zone
base_w = base_counts.astype(float) + ALPHA
tickets = pick_soft_diverse(rng, base_w, play_n_tickets, current_penalty, k=6, total_nums=K1)

# Second Zone
base_w_s = base_counts_s.astype(float) + ALPHA
specials_picked = pick_soft_diverse(rng, base_w_s, play_n_tickets, current_penalty, k=1, total_nums=K2)
specials = [p[0] for p in specials_picked]

# 4. Construct Output
output_lines = []
output_lines.append("==========================================")
output_lines.append("       下期投注建議 (v3)")
output_lines.append("==========================================")
output_lines.append(f"上次開獎日期:   {last_date}")
output_lines.append(f"目前頭獎累積:   ${last_jackpot:,.0f}")
output_lines.append(f"------------------------------------------")
output_lines.append(f"使用策略: 動態 Cap-10 + 第二區熱度模型")
output_lines.append(f"多樣性懲罰: {'增加強烈度 (0.2)' if current_n_tickets >= 5 else '基礎 (0.4)'}")
output_lines.append(f"上次開獎結果: {'頭獎已開出！' if first_prize_won[0] else '連摃 (頭獎未開出)'}")
output_lines.append(f"目前連摃期數: {consecutive_losses} 期")
output_lines.append(f"策略門檻 (8億): {'⛔ 未達標' if is_below_threshold else '✅ 已達標 (火力全開)'}")
if is_below_threshold:
    output_lines.append(f"建議行動:     暫停投注 (目前獎金低於獲利門檻，建議省錢等待)")
    output_lines.append(f"備註:         若手癢想買，下方提供 {play_n_tickets} 組參考號碼")
else:
    output_lines.append(f"下期建議注數: {recommended_tickets} 注 (總金額 ${recommended_tickets * 100})")
output_lines.append(f"------------------------------------------")

for i, pick in enumerate(tickets):
    nums = sorted(pick + 1)
    nums_str = " ".join(f"{x:02d}" for x in nums)
    spec_str = f"{specials[i]+1:02d}"
    output_lines.append(f"第 {i+1:02d} 組: [{nums_str}]  特別號: {spec_str}")

output_lines.append("==========================================")
output_lines.append("祝您中大獎！")

final_output = "\n".join(output_lines)

# Output to Console
print(final_output)

# Output to Text File
with open("recommendation_output.txt", "w", encoding='utf-8') as f:
    f.write(final_output)

# 5. Save to History Log (Indexed by last_date)
import json
import os

HISTORY_PATH = "recommendation_history.json"
history = {}

if os.path.exists(HISTORY_PATH):
    try:
        with open(HISTORY_PATH, "r", encoding='utf-8') as f:
            history = json.load(f)
    except:
        pass

# Structure the data for this date
current_record = {
    "jackpot": last_jackpot,
    "strategy": "Dynamic Cap-10 (800m Threshold)",
    "threshold_met": not is_below_threshold,
    "recommended_tickets": recommended_tickets,
    "recommendations": []
}

for i, pick in enumerate(tickets):
    nums = sorted(pick + 1)
    nums_str = " ".join(f"{x:02d}" for x in nums)
    spec_str = f"{specials[i]+1:02d}"
    current_record["recommendations"].append({
        "group": i + 1,
        "numbers": nums_str,
        "special": spec_str
    })

# Update history (using last_date as unique key)
history[last_date] = current_record

with open(HISTORY_PATH, "w", encoding='utf-8') as f:
    json.dump(history, f, indent=4, ensure_ascii=False)

print(f"結果已儲存至 'recommendation_output.txt' 與 '{HISTORY_PATH}'")

