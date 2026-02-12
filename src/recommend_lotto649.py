
import numpy as np
import pandas as pd
from numpy.random import default_rng
import os
import json

# =====================
# CONFIG
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'lotto649_results.csv')
INITIAL_TICKETS = 2
MAX_TICKETS = 10
THRESHOLD = 150000000 # 1.5-亿門檻
ALPHA = 100
PENALTY_BASE = 0.4
PENALTY_LARGE_BATCH = 0.2
K1 = 49

# =====================
# UTIL
# =====================
def sample_wor_es(rng, w, k=6):
    """Efraimidis–Spirakis weighted sampling without replacement."""
    u = rng.random(len(w))
    keys = -np.log(u) / w
    return np.argpartition(keys, k-1)[:k]

def parse_currency(x):
    if isinstance(x, str):
        return float(x.replace(',', ''))
    return float(x)

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
        # Note: In Lotto 6/49, we pick 6 numbers. 
        # The special number is mostly for checking winnings, not for picking (except for system bets, but here we generate standard tickets).
        # Wait, usually for Lotto 6/49 you just pick 6 numbers. 
        # The special number is just the 7th number drawn from the remaining 43.
        # So we don't predict a "Special Number" separately for a ticket.
        # A ticket just has 6 numbers.
        
        used[pick] += 1
    return tickets

# =====================
# MAIN
# =====================
def main():
    print("Loading data for Lotto 6/49 (Dynamic Penalty + Threshold)...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # 1. Parse History for Weights
    # Load all history
    first_draws = df["Numbers"].astype(str).str.split().apply(
        lambda xs: np.array([int(x) for x in xs], dtype=np.int16) - 1
    )
    # Special number is also 1-49
    special_draws = df["Special_Number"].apply(lambda x: int(x) - 1)

    # Latest Draw Info
    last_date = df["Date"].iloc[0]
    
    # Calculate Effective Jackpot for determining strategy state
    last_rollover = parse_currency(df["Jackpot_Rollover"].iloc[0])
    last_current_total = parse_currency(df["Jackpot_Total"].iloc[0])
    last_jackpot = last_rollover + last_current_total 
    # Note: If someone won, the next jackpot starts at 100M. 
    # Use the 'Jackpot_Per_Winner' of the latest draw to see if it was won.
    last_winner_prize = parse_currency(df["Jackpot_Per_Winner"].iloc[0])
    
    was_won = last_winner_prize > 0
    
    # If won, the next jackpot is reset to 100M.
    # If not won, the next jackpot is at least last_jackpot (it accumulates).
    # Actually, for prediction of NEXT draw, we should estimate the *advertised* jackpot.
    # If it was won, advertised is 100M.
    # If not won, advertised is approx last_jackpot + new sales allocation.
    
    estimated_next_jackpot = 100000000.0
    if not was_won:
        estimated_next_jackpot = last_jackpot # + estimated sales? 
        # Just use current accumulated as lower bound.
    
    # Counts
    base_counts = np.zeros(K1, dtype=int)
    for arr in first_draws:
        base_counts[arr] += 1
    for s in special_draws:
        base_counts[s] += 1 # Special number counts towards heat too

    # 2. Determine Ticket Count (Dynamic Strategy)
    # Consecutive losses since last win
    # Iterate backwards
    jp_per_winners = df["Jackpot_Per_Winner"].apply(parse_currency).values
    consecutive_losses = 0
    for prize in jp_per_winners:
        if prize > 0:
            break
        consecutive_losses += 1
    
    # If latest draw WAS won, consecutive losses is 0. 
    # If latest draw NOT won, consecutive losses is 1+ (current streak)
    
    # Wait, 'consecutive_losses' usually means 'how long has it been since a jackpot was hit'.
    # If index 0 (latest) has prize > 0, then 0 losses.
    # If index 0 has prize 0, index 1 has prize 0... then > 0.
    
    current_n_tickets = min(INITIAL_TICKETS + consecutive_losses, MAX_TICKETS)
    
    # 3. Apply Threshold Strategy
    is_below_threshold = estimated_next_jackpot < THRESHOLD
    
    recommended_tickets = current_n_tickets
    if is_below_threshold:
        recommended_tickets = 0 # Suggest skip
        
    play_n_tickets = max(1, current_n_tickets) # Generate some anyway

    # 4. Generate Recommendations
    rng = default_rng()
    current_penalty = PENALTY_BASE if current_n_tickets < 5 else PENALTY_LARGE_BATCH
    
    base_w = base_counts.astype(float) + ALPHA
    tickets = pick_soft_diverse(rng, base_w, play_n_tickets, current_penalty, k=6, total_nums=K1)
    
    # 5. Output
    output_lines = []
    output_lines.append("==========================================")
    output_lines.append("       大樂透 (Lotto 6/49) 投注建議")
    output_lines.append("==========================================")
    output_lines.append(f"最新開獎日期:   {last_date}")
    output_lines.append(f"預估下期頭獎:   ${estimated_next_jackpot:,.0f} (累積/保底)")
    output_lines.append(f"------------------------------------------")
    output_lines.append(f"使用策略: 動態 Cap-10 + 1.5億門檻")
    output_lines.append(f"連摃期數: {consecutive_losses} 期")
    output_lines.append(f"策略門檻 (1.5億): {'⛔ 未達標' if is_below_threshold else '✅ 已達標 (進場投資)'}")
    
    if is_below_threshold:
        output_lines.append(f"建議行動:     暫停投注 (目前獎金低於獲利期望值，建議省錢)")
        output_lines.append(f"備註:         若手癢想買，下方提供 {play_n_tickets} 組參考號碼")
    else:
        output_lines.append(f"下期建議注數: {recommended_tickets} 注 (總金額 ${recommended_tickets * 50})")
    
    output_lines.append(f"------------------------------------------")
    
    for i, pick in enumerate(tickets):
        nums = sorted(pick + 1)
        nums_str = " ".join(f"{x:02d}" for x in nums)
        output_lines.append(f"第 {i+1:02d} 組: [{nums_str}]")
        
    output_lines.append("==========================================")
    output_lines.append("祝您中大獎！")
    
    final_output = "\n".join(output_lines)
    print(final_output)
    
    # Save output
    with open("recommendation_lotto649_output.txt", "w", encoding='utf-8') as f:
        f.write(final_output)
        
    # Save History
    history_path = "recommendation_lotto649_history.json"
    history = {}
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding='utf-8') as f:
                history = json.load(f)
        except: pass
        
    current_record = {
        "jackpot_est": estimated_next_jackpot,
        "strategy": "Dynamic Cap-10 (150m Threshold)",
        "threshold_met": not is_below_threshold,
        "recommended_tickets": recommended_tickets,
        "recommendations": []
    }
    
    for i, pick in enumerate(tickets):
        nums = sorted(pick + 1)
        nums_str = " ".join(f"{x:02d}" for x in nums)
        current_record["recommendations"].append({
            "group": i + 1,
            "numbers": nums_str
        })
        
    history[last_date] = current_record
    with open(history_path, "w", encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
        
    print(f"結果已儲存至 'recommendation_lotto649_output.txt'")

if __name__ == "__main__":
    main()
