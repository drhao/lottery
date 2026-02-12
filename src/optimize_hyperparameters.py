
import pandas as pd
import numpy as np
import os
import time

# =====================
# CONFIG
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'lotto649_results.csv')
TICKET_PRICE = 50
MAX_TICKETS = 15
N_SIMS = 3000         # 3000 sims for sweep to be faster
K1 = 49 
BACKTEST_DRAWS = 1000 

def sample_wor_es(rng, w, k=6):
    u = rng.random(len(w))
    keys = -np.log(u) / w
    return np.argpartition(keys, k-1)[:k]

def parse_currency(x):
    if isinstance(x, str):
        return float(x.replace(',', ''))
    return float(x)

def calc_weights(strategy_type, window, inverse, i, start_index, counts, first_draws, special_draws, alpha):
    w = np.zeros(K1)
    if strategy_type == "all_time":
        base = counts.astype(float)
        if inverse: w = 1.0 / (base + 1.0)
        else: w = base + alpha
    elif strategy_type == "window":
        win_start = max(0, i - window)
        sub_first = first_draws[win_start:i]
        sub_special = special_draws[win_start:i]
        nums = []
        for arr in sub_first: nums.extend(arr)
        nums.extend(sub_special)
        local_counts = np.bincount(nums, minlength=K1)
        base = local_counts.astype(float)
        if inverse: w = 1.0 / (base + 1.0) # Inverse usually doesn't need alpha scaling as much? Or does it? 
        # Actually inverse 1/(x+1) is small range. Alpha was added to BASE weights.
        # For inverse, we usually don't add alpha to denominator? Or we add to result?
        # Original logic: w = base + ALPHA.
        # Inverse logic: w = 1.0 / (base + 1.0).
        # Wait, the user asked about adjusting alpha. Alpha is the "smoothing" for direct weights.
        # For inverse weights, 'Alpha' concept is different (maybe exponent?).
        # Let's stick to modifying Alpha for the Direct Strategy (Recent Hot).
        # For Cold strategy, maybe we just keep it as is or add a small epsilon?
        # Let's parameterize Alpha only for the "Hot" component to see how aggressive we should be.
        else: w = base + alpha
    return w

def run_sweep():
    print(f"\nStarting Hyperparameter Sweep (N={N_SIMS})...")
    
    if not os.path.exists(CSV_PATH):
        print("Data file not found.")
        return

    df = pd.read_csv(CSV_PATH)
    df_chrono = df.iloc[::-1].copy().reset_index(drop=True)
    jp_totals = (df_chrono['Jackpot_Rollover'].apply(parse_currency) + df_chrono['Jackpot_Total'].apply(parse_currency)).values
    jp_per_winner = df_chrono['Jackpot_Per_Winner'].apply(parse_currency).values
    
    # Prize Pool
    prize_pool = []
    for idx, row in df_chrono.iterrows():
        def gp(k_per, k_tot):
            v = parse_currency(row[k_per])
            return v if v > 0 else parse_currency(row[k_tot])
        prize_pool.append({
            '2nd': gp('Second_Prize_Per_Winner', 'Second_Prize_Total'),
            '3rd': gp('Third_Prize_Per_Winner', 'Third_Prize_Total'),
            '4th': gp('Fourth_Prize_Per_Winner', 'Fourth_Prize_Total')
        })
    PRIZE_FIXED = {'5th': 2000, '6th': 1000, '7th': 400, 'Normal': 400}
    
    first_draws = df_chrono["Numbers"].astype(str).str.split().apply(lambda xs: np.array([int(x) for x in xs], dtype=np.int16) - 1).values
    special_draws = df_chrono["Special_Number"].apply(lambda x: int(x) -1).values
    total_draws = len(df_chrono)
    start_index = max(0, total_draws - BACKTEST_DRAWS)

    # Param Grid
    alphas = [10, 50, 100, 200]
    penalties = [0.2, 0.4, 0.6] 
    
    print(f"{'Alpha':<6} | {'Penalty':<7} | {'ROI':<8} | {'Hits':<5}")
    print("-" * 45)
    
    results = []

    # Pre-calc Base Counts for Cold Strategy (Invariant of Alpha we assume? Or should we sweep Cold too?)
    # Cold strategy used 1/(x+1). Alpha doesn't affect it in current implementation.
    # So Alpha only affects Recent Hot.
    
    # Actually, let's pre-calc COLD weights once since they are static for this sweep.
    W_COLD = []
    counts = np.zeros(K1, dtype=int)
    for k in range(start_index):
        for num in first_draws[k]: counts[num] += 1
        counts[special_draws[k]] += 1
    
    for i in range(start_index, total_draws):
        w_cold = calc_weights("all_time", 0, True, i, start_index, counts, first_draws, special_draws, 0)
        W_COLD.append(w_cold)
        for num in first_draws[i]: counts[num] += 1
        counts[special_draws[i]] += 1
        
    
    for alpha in alphas:
        # Pre-calc HOT weights for this Alpha
        W_HOT = []
        for i in range(start_index, total_draws):
            w_hot = calc_weights("window", 100, False, i, start_index, None, first_draws, special_draws, alpha)
            W_HOT.append(w_hot)
            
        for penalty in penalties:
            sim_profits = []
            jackpot_hits = 0
            
            # Deterministic Cost (Same for all params)
            cost_sweep_draws = 0
            curr_tickets = 2
            total_cost = 0
            for i in range(start_index, total_draws):
                total_cost += curr_tickets * TICKET_PRICE
                if jp_per_winner[i] > 0: curr_tickets = 2
                else: curr_tickets = min(curr_tickets + 1, MAX_TICKETS)
            
            for sim in range(N_SIMS):
                rng = np.random.default_rng(sim + int(alpha*100) + int(penalty*10))
                curr_tickets = 2
                sim_prize = 0
                
                for i in range(start_index, total_draws):
                    n_hot = curr_tickets // 2
                    n_cold = curr_tickets - n_hot
                    
                    actual_m = set(first_draws[i])
                    actual_s = special_draws[i]
                    
                    # HOT
                    w = W_HOT[i-start_index]
                    used = np.zeros(K1, dtype=int)
                    for _ in range(n_hot):
                        w_pen = w * (penalty ** used)
                        pick = sample_wor_es(rng, w_pen, k=6)
                        used[pick] += 1
                        
                        pick_set = set(pick)
                        m = len(pick_set.intersection(actual_m))
                        s = 1 if actual_s in pick_set else 0
                        if m == 6: 
                            sim_prize += max(jp_totals[i], 100000000)
                            jackpot_hits += 1
                        elif m==5 and s: sim_prize += prize_pool[i-start_index]['2nd']
                        elif m==5: sim_prize += prize_pool[i-start_index]['3rd']
                        elif m==4 and s: sim_prize += prize_pool[i-start_index]['4th']
                        elif m==4: sim_prize += PRIZE_FIXED['5th']
                        elif m==3 and s: sim_prize += PRIZE_FIXED['6th']
                        elif m==2 and s: sim_prize += PRIZE_FIXED['7th']
                        elif m==3: sim_prize += PRIZE_FIXED['Normal']

                    # COLD
                    w = W_COLD[i-start_index]
                    used = np.zeros(K1, dtype=int)
                    for _ in range(n_cold):
                        w_pen = w * (penalty ** used)
                        pick = sample_wor_es(rng, w_pen, k=6)
                        used[pick] += 1 # Penalty logic for cold too
                        
                        pick_set = set(pick)
                        m = len(pick_set.intersection(actual_m))
                        s = 1 if actual_s in pick_set else 0
                        if m == 6: 
                            sim_prize += max(jp_totals[i], 100000000)
                            jackpot_hits += 1
                        elif m==5 and s: sim_prize += prize_pool[i-start_index]['2nd']
                        elif m==5: sim_prize += prize_pool[i-start_index]['3rd']
                        elif m==4 and s: sim_prize += prize_pool[i-start_index]['4th']
                        elif m==4: sim_prize += PRIZE_FIXED['5th']
                        elif m==3 and s: sim_prize += PRIZE_FIXED['6th']
                        elif m==2 and s: sim_prize += PRIZE_FIXED['7th']
                        elif m==3: sim_prize += PRIZE_FIXED['Normal']
                    
                    if jp_per_winner[i] > 0: curr_tickets = 2
                    else: curr_tickets = min(curr_tickets + 1, MAX_TICKETS)
                    
                sim_profits.append(sim_prize - total_cost)
            
            avg_profit = np.mean(sim_profits)
            roi = (avg_profit / total_cost) * 100
            print(f"{alpha:<6} | {penalty:<7} | {roi:>7.2f}% | {jackpot_hits:<5}")
            results.append({"Alpha": alpha, "Penalty": penalty, "ROI": roi, "Hits": jackpot_hits})

    print("-" * 45)

if __name__ == "__main__":
    run_sweep()
