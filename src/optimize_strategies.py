
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
N_SIMS = 3000
MAX_TICKETS = 15      # Fixed at the "good" cap we found
THRESHOLD = 0         # NO THRESHOLD
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

def run_strategy_sweep():
    if not os.path.exists(CSV_PATH):
        print("Data file not found.")
        return

    df = pd.read_csv(CSV_PATH)
    df_chrono = df.iloc[::-1].copy().reset_index(drop=True)
    jp_totals = (df_chrono['Jackpot_Rollover'].apply(parse_currency) + df_chrono['Jackpot_Total'].apply(parse_currency)).values
    jp_per_winner = df_chrono['Jackpot_Per_Winner'].apply(parse_currency).values
    
    # Pre-calc prize pool
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
    
    # Define Strategies
    strategies = [
        {"name": "All-Time Hot", "type": "all_time", "window": 0, "inverse": False},
        {"name": "Recent Hot (10)", "type": "window", "window": 10, "inverse": False},
        {"name": "Recent Hot (50)", "type": "window", "window": 50, "inverse": False},
        {"name": "Recent Hot (100)", "type": "window", "window": 100, "inverse": False},
        {"name": "Cold Numbers (All-Time)", "type": "all_time", "window": 0, "inverse": True},
        {"name": "Cold Numbers (Recent 50)", "type": "window", "window": 50, "inverse": True},
    ]
    
    print(f"Starting Strategy Sweep (Cap 15, No Threshold, N={N_SIMS})...")
    print(f"{'Strategy':<25} | {'ROI':<8} | {'Avg Profit':<11} | {'Hits':<5}")
    print("-" * 60)
    
    # Pre-calc All-Time Initial Counts
    init_counts_all = np.zeros(K1, dtype=int)
    for i in range(start_index):
        for num in first_draws[i]: init_counts_all[num] += 1
        init_counts_all[special_draws[i]] += 1 

    for strat in strategies:
        # 1. Deterministic Cost is SAME for all (since strategy only changes numbers picked, not tickets bought)
        # Wait, tickets bought depends on 'jp_per_winner' history which is constant.
        # So Cost is effectively constant?
        # YES.
        
        # Calculate Cost Once
        if strat == strategies[0]:
            curr_tickets = 2
            shared_total_cost = 0
            for i in range(start_index, total_draws):
                shared_total_cost += curr_tickets * TICKET_PRICE
                if jp_per_winner[i] > 0: curr_tickets = 2
                else: curr_tickets = min(curr_tickets + 1, MAX_TICKETS)
        
        # 2. Run Sims
        sim_profits = []
        jackpot_hits = 0 
        
        for sim in range(N_SIMS):
            rng = np.random.default_rng(sim + 30000)
            curr_tickets = 2
            sim_prize = 0
            
            # State Management
            if strat["type"] == "all_time":
                counts = init_counts_all.copy()
            
            # For windowed, we don't maintain 'counts', we calc on fly or maintain specific window buffer
            # Optimization: Maintain full history in memory and slice?
            # first_draws is already full history array.
            
            for i in range(start_index, total_draws):
                # Determine Weights
                w = None
                
                if strat["type"] == "all_time":
                    base = counts.astype(float)
                    if strat["inverse"]:
                        # Inverse: 1 / (count + 1)
                        w = 1.0 / (base + 1.0)
                    else:
                        w = base + ALPHA
                elif strat["type"] == "window":
                    win_start = max(0, i - strat["window"])
                    # Slice history
                    # This is slower inside loop but correctness first.
                    window_nums = []
                    for k in range(win_start, i):
                        window_nums.extend(first_draws[k])
                        window_nums.append(special_draws[k]) # special counts? Yes.
                    
                    local_counts = np.bincount(window_nums, minlength=K1)
                    base = local_counts.astype(float)
                    
                    if strat["inverse"]:
                        w = 1.0 / (base + 1.0)
                    else:
                        w = base + ALPHA

                # Select
                penalty = PENALTY_BASE if curr_tickets < 5 else PENALTY_LARGE
                actual_m = set(first_draws[i])
                actual_s = special_draws[i]
                
                used = np.zeros(K1, dtype=int)
                for _ in range(curr_tickets):
                    w_pen = w * (penalty ** used) 
                    w_pen = np.clip(w_pen, 1e-12, None)
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

                # Update State (for next draw)
                if strat["type"] == "all_time":
                    for num in first_draws[i]: counts[num] += 1
                    counts[special_draws[i]] += 1
                
                # Ticket Step
                if jp_per_winner[i] > 0: curr_tickets = 2
                else: curr_tickets = min(curr_tickets + 1, MAX_TICKETS)
            
            sim_profits.append(sim_prize - shared_total_cost)
            
        avg_profit = np.mean(sim_profits)
        roi = (avg_profit / shared_total_cost) * 100
        
        print(f"{strat['name']:<25} | {roi:>7.2f}% | ${avg_profit:>10,.0f} | {jackpot_hits:<5}")

    print("-" * 60)

if __name__ == "__main__":
    run_strategy_sweep()
