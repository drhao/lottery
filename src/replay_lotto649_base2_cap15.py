
import pandas as pd
import numpy as np
from numpy.random import default_rng
import os

# =====================
# CONFIG
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'lotto649_results.csv')
TICKET_PRICE = 50
MAX_TICKETS = 15      # Cap 15
N_SIMS = 10000        # 10,000 sims for robust analysis
THRESHOLD = 0         # NO THRESHOLD (Always Play)
ALPHA = 100
PENALTY_BASE = 0.4
PENALTY_LARGE = 0.2
K1 = 49 
BACKTEST_DRAWS = 1000 

def sample_wor_es(rng, w, k=6):
    """Efraimidis–Spirakis weighted sampling without replacement."""
    u = rng.random(len(w))
    keys = -np.log(u) / w
    return np.argpartition(keys, k-1)[:k]

def parse_currency(x):
    if isinstance(x, str):
        return float(x.replace(',', ''))
    return float(x)

def run_simulation():
    print(f"Initializing Lotto 6/49 Strategy (Base-2, Cap-15, NO THRESHOLD, N_SIMS={N_SIMS})...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: Data file not found at {CSV_PATH}")
        return pd.DataFrame(), {}
        
    df = pd.read_csv(CSV_PATH)
    df_chrono = df.iloc[::-1].copy().reset_index(drop=True)
    
    jp_rollover = df_chrono['Jackpot_Rollover'].apply(parse_currency).values
    jp_current = df_chrono['Jackpot_Total'].apply(parse_currency).values
    jp_totals = jp_rollover + jp_current
    jp_per_winner = df_chrono['Jackpot_Per_Winner'].apply(parse_currency).values
    
    # Prize Pool Logic
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
    
    init_counts = np.zeros(K1, dtype=int)
    for i in range(start_index):
        for num in first_draws[i]: init_counts[num] += 1
        init_counts[special_draws[i]] += 1 
        
    results = []
    hit_keys = ['Jackpot', '2nd', '3rd', '4th', '5th', '6th', '7th', 'Normal']
    total_hits = {k: 0 for k in hit_keys}
    
    jackpot_draw_indices = []

    for sim in range(N_SIMS):
        if (sim + 1) % 1000 == 0:
            print(f"  Progress: {sim+1}/{N_SIMS}...")
            
        rng = np.random.default_rng(sim + 50000) # Unique seed offset
        current_n_tickets = 2
        
        sim_cost = 0
        sim_prize = 0
        sim_draws_played = 0
        sim_max_drawdown = 0.0
        running_net_profit = 0
        
        counts = init_counts.copy()
        
        for i in range(start_index, total_draws):
            
            sim_draws_played += 1
            w = counts.astype(float) + ALPHA
            penalty = PENALTY_BASE if current_n_tickets < 5 else PENALTY_LARGE
            
            cost_increase = current_n_tickets * TICKET_PRICE
            sim_cost += cost_increase
            
            actual_m = set(first_draws[i])
            actual_s = special_draws[i]
            
            sim_draw_prize_total = 0
            used = np.zeros(K1, dtype=int)
            
            for _ in range(current_n_tickets):
                cur_w = w * (penalty ** used)
                cur_w = np.clip(cur_w, 1e-12, None)
                
                pick = sample_wor_es(rng, cur_w, k=6)
                used[pick] += 1
                
                pick_set = set(pick)
                match_main_count = len(pick_set.intersection(actual_m))
                match_special = 1 if actual_s in pick_set else 0
                
                prize = 0
                tier = None
                
                if match_main_count == 6:
                    tier = 'Jackpot'
                    prize = max(jp_totals[i], 100000000)
                    jackpot_draw_indices.append(i)
                elif match_main_count == 5 and match_special == 1:
                    tier = '2nd'
                    prize = prize_pool[i-start_index]['2nd']
                elif match_main_count == 5:
                    tier = '3rd'
                    prize = prize_pool[i-start_index]['3rd']
                elif match_main_count == 4 and match_special == 1:
                    tier = '4th'
                    prize = prize_pool[i-start_index]['4th']
                elif match_main_count == 4:
                    tier = '5th'
                    prize = PRIZE_FIXED['5th']
                elif match_main_count == 3 and match_special == 1:
                    tier = '6th'
                    prize = PRIZE_FIXED['6th']
                elif match_main_count == 2 and match_special == 1:
                    tier = '7th'
                    prize = PRIZE_FIXED['7th']
                elif match_main_count == 3:
                    tier = 'Normal'
                    prize = PRIZE_FIXED['Normal']
                
                if tier:
                    total_hits[tier] += 1
                    sim_draw_prize_total += prize
                    if tier == 'Jackpot':
                         print(f"!!! JACKPOT HIT !!! Sim {sim}, Draw {i} (Prize: {prize:,.0f})")

            sim_prize += sim_draw_prize_total
            running_net_profit += sim_draw_prize_total - cost_increase
            if running_net_profit < sim_max_drawdown:
                sim_max_drawdown = running_net_profit
            
            for num in first_draws[i]: counts[num] += 1
            counts[special_draws[i]] += 1
            
            if jp_per_winner[i] > 0: 
                current_n_tickets = 2
            else:
                current_n_tickets = min(current_n_tickets + 1, MAX_TICKETS)

        results.append({
            "cost": sim_cost,
            "prize": sim_prize,
            "profit": sim_prize - sim_cost,
            "max_drawdown": sim_max_drawdown
        })
        
    return pd.DataFrame(results), total_hits, jackpot_draw_indices

if __name__ == "__main__":
    df_res, hits, jp_indices = run_simulation()
    
    if not df_res.empty:
        total_profit = df_res['profit'].sum()
        total_cost = df_res['cost'].sum()
        roi = (total_profit / total_cost) * 100 if total_cost > 0 else 0
        
        # Calculate winning statistics
        profitable_runs = df_res[df_res['profit'] > 0]
        profit_prob = len(profitable_runs) / len(df_res) * 100
        
        print("\n" + "="*50)
        print("--- Detailed Analysis: Lotto 6/49 (Base 2, Cap 15) ---")
        print(f"Comparison: No Threshold, 10,000 Sims")
        print("="*50)
        print(f"總獲利次數 (Prob. of Profit): {len(profitable_runs)} ({profit_prob:.2f}%)")
        print(f"平均投入成本 (Avg Cost): ${df_res['cost'].mean():,.0f}")
        print(f"平均淨利 (EV): ${df_res['profit'].mean():,.0f}")
        print(f"淨利中位數 (Median P/L): ${df_res['profit'].median():,.0f}")
        print(f"投資報酬率 (ROI): {roi:.2f}%")
        print(f"平均最大回撤 (Max Drawdown): ${df_res['max_drawdown'].mean():,.0f}")
        
        print("\n--- Prize Tier Hits (Total across 10k sims) ---")
        for k, v in hits.items():
            print(f"{k:<10}: {v:>6}")
            
        print("\n--- Jackpot Analysis ---")
        print(f"Total Jackpot Hits: {hits['Jackpot']}")
        unique_jp_draws = sorted(list(set(jp_indices)))
        print(f"Unique Draws where Jackpot was hit ({len(unique_jp_draws)}): {unique_jp_draws[:10]}{'...' if len(unique_jp_draws)>10 else ''}")

        print("\n--- Risk Metrics ---")
        print(f"Worst Case Loss: ${df_res['profit'].min():,.0f}")
        print(f"Best Case Profit: ${df_res['profit'].max():,.0f}")
        print("="*50)
