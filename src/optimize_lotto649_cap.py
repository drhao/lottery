
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
N_SIMS = 3000         # High enough for stable comparative results
THRESHOLD = 150000000 
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

def run_simulation(max_tickets):
    # Re-using the logic from replay_lotto649_150m.py but stripped down for speed/stats
    if not os.path.exists(CSV_PATH): return None
        
    df = pd.read_csv(CSV_PATH)
    df_chrono = df.iloc[::-1].copy().reset_index(drop=True)
    
    jp_rollover = df_chrono['Jackpot_Rollover'].apply(parse_currency).values
    jp_current = df_chrono['Jackpot_Total'].apply(parse_currency).values
    jp_totals = jp_rollover + jp_current
    jp_per_winner = df_chrono['Jackpot_Per_Winner'].apply(parse_currency).values
    
    # Simple prize approximations for speed in optimization sweep
    # For robust comparison, we assume:
    # Jackpot: Share of pot (if hit). To simplify, use CURRENT POT.
    # 2nd-4th: Average values from history
    avg_2nd = df_chrono['Second_Prize_Per_Winner'].apply(lambda x: parse_currency(x) if parse_currency(x)>0 else 0).mean()
    if avg_2nd == 0: avg_2nd = 2000000 # Fallback
    
    # Just use fixed map for non-jackpot to be faster? 
    # Or stick to detailed logic? Detailed is safer for accuracy.
    # Let's use the detailed prize lookup.
    
    def get_prize_value(row, col_per, col_tot):
        per = parse_currency(row[col_per])
        if per > 0: return per
        return parse_currency(row[col_tot])

    prize_pool = []
    for idx, row in df_chrono.iterrows():
        p = {}
        p['2nd'] = get_prize_value(row, 'Second_Prize_Per_Winner', 'Second_Prize_Total')
        p['3rd'] = get_prize_value(row, 'Third_Prize_Per_Winner', 'Third_Prize_Total')
        p['4th'] = get_prize_value(row, 'Fourth_Prize_Per_Winner', 'Fourth_Prize_Total')
        prize_pool.append(p)
        
    PRIZE_FIXED = {'5th': 2000, '6th': 1000, '7th': 400, 'Normal': 400}

    first_draws = df_chrono["Numbers"].astype(str).str.split().apply(
        lambda xs: np.array([int(x) for x in xs], dtype=np.int16) - 1
    ).values
    special_draws = df_chrono["Special_Number"].apply(lambda x: int(x) -1).values

    total_draws = len(df_chrono)
    start_index = max(0, total_draws - BACKTEST_DRAWS)
    
    # Pre-calc counts
    init_counts = np.zeros(K1, dtype=int)
    for i in range(start_index):
        for num in first_draws[i]: init_counts[num] += 1
        init_counts[special_draws[i]] += 1 

    total_cost = 0
    total_profit = 0
    jackpot_hits = 0
    
    # To save time, we might run sims in parallel or just use N_SIMS sequentially
    # Python is slow for loops.
    
    sim_profits = []
    
    for sim in range(N_SIMS):
        rng = np.random.default_rng(sim) # Consistent seed per sim ID across caps
        current_n_tickets = 2
        counts = init_counts.copy()
        
        sim_cost = 0
        sim_prize = 0
        
        for i in range(start_index, total_draws):
            current_jp = jp_totals[i]
            
            if current_jp < THRESHOLD:
                # Update but don't play
                for num in first_draws[i]: counts[num] += 1
                counts[special_draws[i]] += 1
                current_n_tickets = 2
                continue
            
            w = counts.astype(float) + ALPHA
            penalty = PENALTY_BASE if current_n_tickets < 5 else PENALTY_LARGE
            
            # Cost
            sim_cost += current_n_tickets * TICKET_PRICE
            
            # Play
            actual_m = set(first_draws[i])
            actual_s = special_draws[i]
            used = np.zeros(K1, dtype=int)
            
            draw_prize = 0
            
            for _ in range(current_n_tickets):
                cur_w = w * (penalty ** used)
                # Optimize: clip only if needed (usually just check for 0)
                # cur_w = np.maximum(cur_w, 1e-12) # Faster than clip?
                
                pick = sample_wor_es(rng, cur_w, k=6)
                used[pick] += 1
                
                pick_set = set(pick)
                match_main = len(pick_set.intersection(actual_m))
                match_special = 1 if actual_s in pick_set else 0
                
                if match_main == 6:
                    draw_prize += max(current_jp, 100000000)
                    if sim < 1: print(f"  [Cap-{max_tickets}] Jackpot Hit!")
                elif match_main == 5 and match_special: draw_prize += prize_pool[i]['2nd']
                elif match_main == 5: draw_prize += prize_pool[i]['3rd']
                elif match_main == 4 and match_special: draw_prize += prize_pool[i]['4th']
                elif match_main == 4: draw_prize += PRIZE_FIXED['5th']
                elif match_main == 3 and match_special: draw_prize += PRIZE_FIXED['6th']
                elif match_main == 2 and match_special: draw_prize += PRIZE_FIXED['7th']
                elif match_main == 3: draw_prize += PRIZE_FIXED['Normal']
            
            sim_prize += draw_prize
            
            # Update
            for num in first_draws[i]: counts[num] += 1
            counts[special_draws[i]] += 1
            
            if jp_per_winner[i] > 0:
                current_n_tickets = 2
            else:
                current_n_tickets = min(current_n_tickets + 1, max_tickets)
        
        sim_profits.append(sim_prize - sim_cost)
        if sim_prize > 100000000: jackpot_hits += 1
        
    return {
        "ROI": np.sum(sim_profits) / (np.sum([res['cost'] for res in [{'cost': sim_cost} for _ in range(N_SIMS)]] ) if np.sum(sim_profits)!=0 else 1) * 100, 
        # Wait, the cost calculation above is wrong if I don't track cost per sim properly in list.
        # Fixed below.
        "Mean_Profit": np.mean(sim_profits),
        "Median_Profit": np.median(sim_profits),
        "Jackpot_Hits": jackpot_hits
    }

def run_optimization():
    caps_to_test = [2, 5, 8, 10, 12, 15, 20]
    results = []
    
    print(f"Starting Optimization Sweep (N={N_SIMS} per cap)...")
    
    # First, run a baseline measure of TOTAL COST for ROI calc since it varies slightly by random path?
    # No, cost is deterministic if participation is deterministic? 
    # Ticket count depends on 'current_n_tickets' which depends on whether WE won?
    # No, it depends on whether the GLOBAL jackpot was won (jp_per_winner > 0).
    # Since we are backtesting on historical data, the 'reset' signal is external and fixed.
    # Therefore, the sequence of ticket counts is DETERMINISTIC for a given Cap strategy!
    # So Cost is constant for all N_SIMS for a specific Cap.
    
    # Let's calculate cost once per cap.
    
    for cap in caps_to_test:
        start_t = time.time()
        
        # We need to run the full simulation loop because we need Profit which IS random.
        # But we can simplify cost calc.
        
        # Let's just run the full sim function I wrote above, but correct the return.
        # I need to track cost inside `run_simulation`.
        
        # Refactored run_simulation logic for speed:
        # Actually simplest is just to run it. 3000 sims of 1000 draws is 3M loops. 
        # Python might take ~30-60s per cap. 7 caps = ~5-7 mins. 
        # Acceptable for "Agentic" work.
        
        # Wait, I need to fix the ROI calculation in `run_simulation` first.
        # I'll rewrite `run_simulation` inside the loop to be cleaner or fix it now.
        pass

    # ... (Implementation in the file below)

if __name__ == "__main__":
    # Redefine function cleanly for execution
    
    df = pd.read_csv(CSV_PATH)
    df_chrono = df.iloc[::-1].copy().reset_index(drop=True)
    jp_totals = (df_chrono['Jackpot_Rollover'].apply(parse_currency) + df_chrono['Jackpot_Total'].apply(parse_currency)).values
    jp_per_winner = df_chrono['Jackpot_Per_Winner'].apply(parse_currency).values
    
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
    
    init_counts = np.zeros(K1, dtype=int)
    for i in range(start_index):
        for num in first_draws[i]: init_counts[num] += 1
        init_counts[special_draws[i]] += 1 

    caps = [2, 4, 6, 8, 10, 12, 15, 20]
    out_data = []
    
    print(f"{'Cap':<5} | {'ROI':<8} | {'Avg Profit':<12} | {'Jackpots':<8} | {'Cost':<10}")
    print("-" * 60)
    
    for cap_max in caps:
        # 1. Calculate Deterministic Cost
        # Simulation of "Strategy State" only
        curr_tickets = 2
        total_cost = 0
        draws_played = 0
        
        # We need to simulate the 'counts' evolution? 
        # Actually random picks depend on counts.
        # But 'curr_tickets' evolution depends ONLY on history (jp_per_winner), not on our picks.
        # So Cost is deterministic.
        
        for i in range(start_index, total_draws):
            if jp_totals[i] < THRESHOLD:
                curr_tickets = 2
                continue
            
            total_cost += curr_tickets * TICKET_PRICE
            draws_played += 1
            
            if jp_per_winner[i] > 0:
                curr_tickets = 2
            else:
                curr_tickets = min(curr_tickets + 1, cap_max)
                
        # 2. Run Sims for Profit (Variance)
        sim_profits = []
        jackpot_hits = 0
        
        for sim in range(N_SIMS):
            rng = np.random.default_rng(sim)
            curr_tickets = 2
            sim_prize = 0
            
            # Reset counts for each sim? No, counts evolution is deterministic too because we update with ACTUAL draw results.
            # So we can just maintain one running count if we assume we just observe?
            # Yes, our recommendations don't change the history.
            # So we can pre-calculate the probability weights `w` for every draw!
            # Optimization: Pre-calculate `w` matrices?
            # `w` depends on counts. Counts depend on history. History is fixed.
            # So `w` at each step i is fixed!
            
            # Let's do a loop, but perform vectorized sampling if possible?
            # Hard to vector variable number of tickets.
            # Just optimize the inner loop.
            
            counts = init_counts.copy()
            
            for i in range(start_index, total_draws):
                # Update counts at start of step (from previous)
                # (Actually counts should be updated AFTER the draw... so for step i, we use counts from 0..i-1)
                
                if jp_totals[i] < THRESHOLD:
                    # Just update counts
                    for num in first_draws[i]: counts[num] += 1
                    counts[special_draws[i]] += 1
                    curr_tickets = 2
                    continue
                
                w = counts.astype(float) + ALPHA
                # Penalty depends on tickets used IN THIS BATCH.
                # So we can't pre-calc the penalized weights easily.
                
                penalty = PENALTY_BASE if curr_tickets < 5 else PENALTY_LARGE
                actual_m = set(first_draws[i])
                actual_s = special_draws[i]
                
                # Pick tickets
                used = np.zeros(K1, dtype=int)
                for _ in range(curr_tickets):
                    w_pen = w * (penalty ** used)
                    pick = sample_wor_es(rng, w_pen, k=6)
                    used[pick] += 1
                    
                    # Score
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

                # Update counts
                for num in first_draws[i]: counts[num] += 1
                counts[special_draws[i]] += 1
                
                if jp_per_winner[i] > 0: curr_tickets = 2
                else: curr_tickets = min(curr_tickets + 1, cap_max)
            
            sim_profits.append(sim_prize - total_cost)
            
        avg_profit = np.mean(sim_profits)
        roi = (avg_profit / total_cost) * 100 if total_cost > 0 else 0
        
        print(f"{cap_max:<5} | {roi:>7.2f}% | ${avg_profit:>11,.0f} | {jackpot_hits:>8} | ${total_cost:,.0f}")
        out_data.append({"Cap": cap_max, "ROI": roi, "EV": avg_profit, "Jackpots": jackpot_hits, "Cost": total_cost})

    # Find best
    best_roi = max(out_data, key=lambda x: x['ROI'])
    print("-" * 60)
    print(f"Best Cap by ROI: {best_roi['Cap']} ({best_roi['ROI']:.2f}%)")
