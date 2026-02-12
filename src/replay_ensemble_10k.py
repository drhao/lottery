
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
MAX_TICKETS = 15      # Cap 15
N_SIMS = 10000        # 10,000 sims
ALPHA = 100
PENALTY_BASE = 0.4
PENALTY_LARGE = 0.2
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

def calc_weights(strategy_type, window, inverse, i, start_index, counts, first_draws, special_draws):
    w = np.zeros(K1)
    if strategy_type == "all_time":
        base = counts.astype(float)
        if inverse: w = 1.0 / (base + 1.0)
        else: w = base + ALPHA
    elif strategy_type == "window":
        win_start = max(0, i - window)
        sub_first = first_draws[win_start:i]
        sub_special = special_draws[win_start:i]
        nums = []
        for arr in sub_first: nums.extend(arr)
        nums.extend(sub_special)
        local_counts = np.bincount(nums, minlength=K1)
        base = local_counts.astype(float)
        if inverse: w = 1.0 / (base + 1.0)
        else: w = base + ALPHA
    return w

def run_ensemble_simulation():
    print(f"\nRunning Ensemble Simulation (Mixed Portfolio 50/50) (N={N_SIMS})...")
    
    if not os.path.exists(CSV_PATH):
        print("Data file not found.")
        return None

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
    
    # Pre-calc Weights for both strategies for all draws
    print("Pre-calculating weights...")
    W_HOT = []
    W_COLD = []
    
    counts = np.zeros(K1, dtype=int)
    # Init counts
    for k in range(start_index):
        for num in first_draws[k]: counts[num] += 1
        counts[special_draws[k]] += 1
        
    for i in range(start_index, total_draws):
        # 1. Hot (Window 100)
        w_hot = calc_weights("window", 100, False, i, start_index, None, first_draws, special_draws)
        W_HOT.append(w_hot)
        
        # 2. Cold (All-Time)
        w_cold = calc_weights("all_time", 0, True, i, start_index, counts, first_draws, special_draws)
        W_COLD.append(w_cold)
        
        # Update counts for All-Time
        for num in first_draws[i]: counts[num] += 1
        counts[special_draws[i]] += 1
        
    print("Weights ready.")

    # Calculate Total Cost (Independent of strategy, depends on JP history)
    curr_tickets = 2
    total_cost = 0
    for i in range(start_index, total_draws):
        total_cost += curr_tickets * TICKET_PRICE
        if jp_per_winner[i] > 0: curr_tickets = 2
        else: curr_tickets = min(curr_tickets + 1, MAX_TICKETS)

    sim_profits = []
    jackpot_hits = 0
    
    start_time = time.time()

    for sim in range(N_SIMS):
        if (sim+1) % 1000 == 0:
            print(f"  Progress: {sim+1}/{N_SIMS}...")
            
        rng = np.random.default_rng(sim + 777777)
        curr_tickets = 2
        sim_prize = 0
        
        for i in range(start_index, total_draws):
            
            # Strategy: Split tickets
            # If curr_tickets = 2: 1 Hot, 1 Cold
            # If 3: 2 Hot, 1 Cold (alternating? or random split?)
            # Let's do integer split.
            n_hot = curr_tickets // 2
            n_cold = curr_tickets - n_hot
            # If odd, give extra to COLD (since it performed slightly better in 10k sim)
            # Actually, let's alternate based on i to be fair? 
            # Or just n_cold = ceil, n_hot = floor.
            
            actual_m = set(first_draws[i])
            actual_s = special_draws[i]
            
            # --- HOT BATCH ---
            w = W_HOT[i-start_index]
            used = np.zeros(K1, dtype=int) 
            # Note: used counts should strictly be shared if we want valid penalty?
            # Yes, if I buy 2 tickets, the second should know about the first's numbers to avoid duplicates?
            # Actually, penalty is per-ticket generation to avoid picking same number in ONE ticket.
            # But across tickets? We want diversity.
            # Independent generations are fine.
            
            for _ in range(n_hot):
                w_pen = w * (PENALTY_BASE ** used) # Simple penalty against self
                pick = sample_wor_es(rng, w_pen, k=6)
                # used[pick] += 1 # Update strictly? 
                # Let's keep it simple: independent tickets.
                
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

            # --- COLD BATCH ---
            w = W_COLD[i-start_index]
            # used reset? Or shared? Independent tickets approach is standard.
            used = np.zeros(K1, dtype=int)
            
            for _ in range(n_cold):
                w_pen = w * (PENALTY_BASE ** used)
                pick = sample_wor_es(rng, w_pen, k=6)
                
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
        
    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s")
    
    print("\n========================================")
    print("ENSEMBLE RESULTS (10,000 Sims)")
    print("Strategy: 50% Recent Hot (100) / 50% All-Time Cold")
    print("========================================")
    roi = np.sum(sim_profits) / (total_cost * N_SIMS) * 100
    print(f"{'ROI':<20} | {roi:>10.2f}%")
    print(f"{'Avg Profit':<20} | ${np.mean(sim_profits):>10,.0f}")
    print(f"{'Jackpot Hits':<20} | {jackpot_hits:>10}")
    print(f"{'Cost':<20} | ${total_cost:>10,.0f}")

if __name__ == "__main__":
    run_ensemble_simulation()
