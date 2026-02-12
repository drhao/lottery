
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

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print("Data file not found.")
        exit(1)

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
    dates = df_chrono["Date"].values
    
    total_draws = len(df_chrono)
    start_index = max(0, total_draws - BACKTEST_DRAWS)
    
    init_counts = np.zeros(K1, dtype=int)
    for i in range(start_index):
        for num in first_draws[i]: init_counts[num] += 1
        init_counts[special_draws[i]] += 1 

    caps = [10, 12, 14, 16, 18, 20]
    out_data = []
    
    print(f"Starting No-Threshold Optimization (N={N_SIMS})...")
    print(f"{'Cap':<5} | {'ROI':<8} | {'Avg Profit':<11} | {'Hits':<5} | {'Hit Locations (Draw Index)'}")
    print("-" * 100)
    
    for cap_max in caps:
        # 1. Deterministic Cost (No threshold, always play 2..cap)
        curr_tickets = 2
        total_cost = 0
        
        for i in range(start_index, total_draws):
            total_cost += curr_tickets * TICKET_PRICE
            if jp_per_winner[i] > 0:
                curr_tickets = 2
            else:
                curr_tickets = min(curr_tickets + 1, cap_max)
                
        # 2. Run Sims
        sim_profits = []
        jackpot_hits = [] 
        
        for sim in range(N_SIMS):
            rng = np.random.default_rng(sim + 10000) # Offset seed to verify it's not just the same 8 hits from before
            curr_tickets = 2
            sim_prize = 0
            
            counts = init_counts.copy()
            
            for i in range(start_index, total_draws):
                w = counts.astype(float) + ALPHA
                penalty = PENALTY_BASE if curr_tickets < 5 else PENALTY_LARGE
                actual_m = set(first_draws[i])
                actual_s = special_draws[i]
                
                used = np.zeros(K1, dtype=int)
                for _ in range(curr_tickets):
                    w_pen = w * (penalty ** used) 
                    pick = sample_wor_es(rng, w_pen, k=6)
                    used[pick] += 1
                    
                    pick_set = set(pick)
                    m = len(pick_set.intersection(actual_m))
                    s = 1 if actual_s in pick_set else 0
                    
                    if m == 6: 
                        win_amt = max(jp_totals[i], 100000000)
                        sim_prize += win_amt
                        jackpot_hits.append(f"{i}(${win_amt/100000000:.1f}E)")
                    elif m==5 and s: sim_prize += prize_pool[i-start_index]['2nd']
                    elif m==5: sim_prize += prize_pool[i-start_index]['3rd']
                    elif m==4 and s: sim_prize += prize_pool[i-start_index]['4th']
                    elif m==4: sim_prize += PRIZE_FIXED['5th']
                    elif m==3 and s: sim_prize += PRIZE_FIXED['6th']
                    elif m==2 and s: sim_prize += PRIZE_FIXED['7th']
                    elif m==3: sim_prize += PRIZE_FIXED['Normal']

                # Update counts
                for num in first_draws[i]: counts[num] += 1
                counts[special_draws[i]] += 1
                
                if jp_per_winner[i] > 0: curr_tickets = 2
                else: curr_tickets = min(curr_tickets + 1, cap_max)
            
            sim_profits.append(sim_prize - total_cost)
            
        avg_profit = np.mean(sim_profits)
        roi = (avg_profit / total_cost) * 100 if total_cost > 0 else 0
        hits_str = ", ".join(jackpot_hits[:5]) + ("..." if len(jackpot_hits)>5 else "")
        
        print(f"{cap_max:<5} | {roi:>7.2f}% | ${avg_profit:>10,.0f} | {len(jackpot_hits):<5} | {hits_str}")
        out_data.append({"Cap": cap_max, "ROI": roi, "Hits": len(jackpot_hits)})

    print("-" * 100)
