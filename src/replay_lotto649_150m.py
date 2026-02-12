
import pandas as pd
import numpy as np
import os

# =====================
# CONFIG
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'lotto649_results.csv')
TICKET_PRICE = 50
MAX_TICKETS = 20      # Cap 20 strategies
N_SIMS = 2000         # 2,000 sims 
THRESHOLD = 150000000 # 1.5-亿門檻
ALPHA = 100
PENALTY_BASE = 0.4
PENALTY_LARGE = 0.2
K1 = 49 # Pool 1-49
BACKTEST_DRAWS = 1000 # Last 1000 draws

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
    print(f"Initializing Lotto 6/49 Strategy (Cap-10, Threshold {THRESHOLD/100000000}E, N_SIMS={N_SIMS})...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: Data file not found at {CSV_PATH}")
        return pd.DataFrame(), {}
        
    df = pd.read_csv(CSV_PATH)
    
    # Prepare data (Old -> New)
    df_chrono = df.iloc[::-1].copy().reset_index(drop=True)
    
    # Parse prize columns
    # New scraper uses Jackpot_Totals etc.
    # CSV Headers: Period, Date, Numbers, Special_Number, Jackpot_Rollover, Jackpot_Total, Jackpot_Per_Winner, ...
    
    jp_rollover = df_chrono['Jackpot_Rollover'].apply(parse_currency).values
    jp_current = df_chrono['Jackpot_Total'].apply(parse_currency).values
    # Effective Jackpot is Rollover + Current Allocation
    # However, if there is a winner (jp_per > 0), the pool might be topped up to 100M. 
    # But for THRESHOLD check (decision to buy), we look at the accumulated amount.
    jp_totals = jp_rollover + jp_current
    
    jp_per_winner = df_chrono['Jackpot_Per_Winner'].apply(parse_currency).values
    
    # Prepare prize dictionary for fixed prizes
    # Lotto 6/49 Prize Structure:
    # Jackpot (6): Float
    # 2nd (5+S): Float
    # 3rd (5): Float
    # 4th (4+S): Float
    # 5th (4): Fixed 2000
    # 6th (3+S): Fixed 1000
    # 7th (2+S): Fixed 400
    # Normal (3): Fixed 400
    
    # We need floating prizes for 2nd, 3rd, 4th
    sec_prizes = df_chrono['Second_Prize_Total'].apply(parse_currency).values / df_chrono['Second_Prize_Per_Winner'].apply(lambda x: float(str(x).replace(',','')) if float(str(x).replace(',','')) > 0 else 1).values * df_chrono['Second_Prize_Per_Winner'].apply(parse_currency).values
    # Wait, Total is total payout? Or per winner?
    # Scraper saves: sec_total, sec_per.
    # If strictly per winner is needed for user win, we should use 'per_winner' column if avaiable.
    # But usually for backtest we assume we share the prize if we win?
    # Or just take the per_winner value from history? 
    # If we win, we are likely one of few winners. 
    # Let's use the actual per_winner value from history as the prize amount (assuming consistent winners).
    # If nobody won (per_winner=0), we can estimate based on Total? Or just 0?
    # If nobody won 2nd prize, and we win in sim, we take the pot? 
    # For simplicity, if per_winner > 0, use it. If 0, use Total.
    
    def get_prize_value(row, col_per, col_tot):
        per = parse_currency(row[col_per])
        if per > 0: return per
        return parse_currency(row[col_tot])

    # Pre-calculate prize array for each draw
    # We only need this for floating prizes
    prize_pool = []
    for idx, row in df_chrono.iterrows():
        p = {}
        p['2nd'] = get_prize_value(row, 'Second_Prize_Per_Winner', 'Second_Prize_Total')
        p['3rd'] = get_prize_value(row, 'Third_Prize_Per_Winner', 'Third_Prize_Total')
        p['4th'] = get_prize_value(row, 'Fourth_Prize_Per_Winner', 'Fourth_Prize_Total')
        prize_pool.append(p)
        
    PRIZE_FIXED = {
        '5th': 2000,
        '6th': 1000,
        '7th': 400,
        'Normal': 400
    }

    # Numbers
    first_draws = df_chrono["Numbers"].astype(str).str.split().apply(
        lambda xs: np.array([int(x) for x in xs], dtype=np.int16) - 1
    ).values
    special_draws = df_chrono["Special_Number"].apply(lambda x: int(x) -1).values

    # Define Simulation Range
    total_draws = len(df_chrono)
    start_index = max(0, total_draws - BACKTEST_DRAWS)
    
    print(f"Simulation Range: Index {start_index} to {total_draws-1} ({total_draws - start_index} draws)")
    print(f"Date Range: {df_chrono.iloc[start_index]['Date']} to {df_chrono.iloc[-1]['Date']}")

    # Pre-calculate initial counts up to start_index
    init_counts = np.zeros(K1, dtype=int)
    for i in range(start_index):
        for num in first_draws[i]:
            init_counts[num] += 1
        init_counts[special_draws[i]] += 1 
        # Note: Special number is drawn from same 1-49 pool in Lotto 6/49
        # So we add it to the same frequency count? 
        # Strategy: "Hot" numbers. Frequently drawn numbers (main or special) are hot.
        # Yes, add to same count.

    results = []
    
    # Metrics
    # 6: Jackpot
    # 5+S: 2nd
    # 5: 3rd
    # 4+S: 4th
    # 4: 5th
    # 3+S: 6th
    # 2+S: 7th
    # 3: Normal
    
    hit_keys = ['Jackpot', '2nd', '3rd', '4th', '5th', '6th', '7th', 'Normal']
    total_hits = {k: 0 for k in hit_keys}

    for sim in range(N_SIMS):
        if (sim + 1) % 1000 == 0:
            print(f"  Progress: {sim+1}/{N_SIMS}...")
            
        rng = np.random.default_rng(sim)
        current_n_tickets = 2
        
        sim_cost = 0
        sim_prize = 0
        sim_draws_played = 0
        sim_hits = {k: 0 for k in hit_keys}
        sim_max_drawdown = 0.0
        running_net_profit = 0
        
        counts = init_counts.copy()
        
        for i in range(start_index, total_draws):
            current_jp = jp_totals[i]
            
            # --- THRESHOLD CHECK ---
            if current_jp < THRESHOLD:
                # Update counts but don't play
                for num in first_draws[i]:
                    counts[num] += 1
                counts[special_draws[i]] += 1
                current_n_tickets = 2 # Reset
                continue
            
            # Play
            sim_draws_played += 1
            w = counts.astype(float) + ALPHA
            penalty = PENALTY_BASE if current_n_tickets < 5 else PENALTY_LARGE
            
            cost_increase = current_n_tickets * TICKET_PRICE
            sim_cost += cost_increase
            
            actual_m = set(first_draws[i])
            actual_s = special_draws[i]
            
            # For checking win
            actual_all_7 = actual_m.union({actual_s})
            
            sim_draw_prize_total = 0
            
            used = np.zeros(K1, dtype=int)
            
            for _ in range(current_n_tickets):
                # Penalty logic
                cur_w = w * (penalty ** used)
                cur_w = np.clip(cur_w, 1e-12, None)
                
                # Pick 6 Main Numbers
                # In Lotto 6/49, we pick 6 numbers. 
                # There is NO separate special number to pick. 
                # The special number is drawn from the remaining 43 numbers in the machine.
                # So our ticket is just 6 numbers.
                
                pick = sample_wor_es(rng, cur_w, k=6)
                used[pick] += 1
                
                pick_set = set(pick)
                
                # Check match
                # Match Main
                match_main_count = len(pick_set.intersection(actual_m))
                # Match Special? 
                # Does our ticket contain the special number?
                match_special = 1 if actual_s in pick_set else 0
                
                # Check Prize Tier
                prize = 0
                tier = None
                
                if match_main_count == 6:
                    tier = 'Jackpot'
                    prize = max(current_jp, 100000000) # Guarantee header?
                elif match_main_count == 5 and match_special == 1:
                    tier = '2nd'
                    prize = prize_pool[i]['2nd']
                elif match_main_count == 5:
                    tier = '3rd'
                    prize = prize_pool[i]['3rd']
                elif match_main_count == 4 and match_special == 1:
                    tier = '4th'
                    prize = prize_pool[i]['4th']
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
                    sim_hits[tier] += 1
                    total_hits[tier] += 1
                    sim_draw_prize_total += prize
                    
                    if tier == 'Jackpot':
                         print(f"!!! JACKPOT HIT !!! Sim {sim}, Draw {i} (Prize: {prize:,.0f})")

            sim_prize += sim_draw_prize_total
            running_net_profit += sim_draw_prize_total - cost_increase
            if running_net_profit < sim_max_drawdown:
                sim_max_drawdown = running_net_profit
            
            # Update history
            for num in first_draws[i]:
                counts[num] += 1
            counts[special_draws[i]] += 1
            
            # Dynamic Next Tickets logic
            if jp_per_winner[i] > 0: # Jackpot won
                current_n_tickets = 2
            else:
                current_n_tickets = min(current_n_tickets + 1, MAX_TICKETS)

        results.append({
            "cost": sim_cost,
            "prize": sim_prize,
            "profit": sim_prize - sim_cost,
            "draws": sim_draws_played,
            "max_drawdown": sim_max_drawdown,
            "h_jp": sim_hits['Jackpot'],
            "h_2nd": sim_hits['2nd'],
            "h_3rd": sim_hits['3rd']
        })
        
    return pd.DataFrame(results), total_hits

if __name__ == "__main__":
    res_df, total_hits = run_simulation()
    
    print("\n" + "="*50)
    print(f"--- Lotto 6/49 (1.5億 Threshold) Backtest Report ---")
    print(f"Range: Last {BACKTEST_DRAWS} Draws")
    print("="*50)
    
    participating_sims = res_df[res_df['draws'] > 0]
    n_participating = len(participating_sims)
    
    if n_participating == 0:
        print("No simulations participated.")
    else:
        print(f"總獲利次數: {len(res_df[res_df['profit'] > 0])} ({len(res_df[res_df['profit'] > 0])/N_SIMS*100:.2f}%)")
        print(f"平均參與期數: {res_df['draws'].mean():.1f}")
        print(f"平均投入成本: ${res_df['cost'].mean():,.0f}")
        print(f"平均淨利 (EV): ${res_df['profit'].mean():,.0f}")
        print(f"淨利中位數: ${res_df['profit'].median():,.0f}")
        roi = (res_df['profit'].sum() / res_df['cost'].sum() * 100) if res_df['cost'].sum() > 0 else 0
        print(f"投資報酬率 (ROI): {roi:.2f}%")
        print(f"平均最大回撤 (Drawdown): ${res_df['max_drawdown'].mean():,.0f}")
        
        print("\n--- Hit Stats (Total) ---")
        print(f"Jackpot (6):   {total_hits['Jackpot']}")
        print(f"2nd (5+S):     {total_hits['2nd']}")
        print(f"3rd (5):       {total_hits['3rd']}")
        print(f"4th (4+S):     {total_hits['4th']}")
        print(f"5th (4):       {total_hits['5th']}")
        print(f"6th (3+S):     {total_hits['6th']}")
        print(f"7th (2+S):     {total_hits['7th']}")
        print(f"Normal (3):    {total_hits['Normal']}")

        print("\n--- Extremes ---")
        print(f"Max Profit: ${res_df['profit'].max():,.0f}")
        print(f"Max Loss: ${res_df['profit'].min():,.0f}")
        print("="*50)
    
    out_path = os.path.join(BASE_DIR, 'output', "replay_lotto649_150m_10k.csv")
    res_df.to_csv(out_path, index=False)
