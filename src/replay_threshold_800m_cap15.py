import pandas as pd
import numpy as np

import os

# =====================
# CONFIG
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'super_lotto638_results.csv')
TICKET_PRICE = 100
MAX_TICKETS = 15      # Cap 15 strategies (High Roller)
N_SIMS = 10000        # 10,000 sims 
THRESHOLD = 800000000 # 8-亿門檻
ALPHA = 100
PENALTY_BASE = 0.4
PENALTY_LARGE = 0.2
K1, K2 = 38, 8

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
    print(f"Initializing 800M Threshold Strategy (High Roller Cap-15, N_SIMS={N_SIMS})...")
    df = pd.read_csv(CSV_PATH)
    
    # Prepare data (Old -> New)
    df_chrono = df.iloc[::-1].copy().reset_index(drop=True)
    jp_totals = df_chrono['First_Prize_Total'].apply(parse_currency).values
    jp_per_winner = df_chrono['First_Prize_Per_Winner'].apply(parse_currency).values
    sect2_prizes = df_chrono['Second_Prize_Total'].apply(parse_currency).values
    
    first_draws = df_chrono["Numbers"].astype(str).str.split().apply(
        lambda xs: np.array([int(x) for x in xs], dtype=np.int16) - 1
    ).values
    second_draws = df_chrono["Special_Number"].astype(np.int16).to_numpy() - 1

    # Pre-calculate counts up to index 500
    init_counts1 = np.zeros(K1, dtype=int)
    for draw in first_draws[:500]:
        init_counts1[draw] += 1
    init_counts2 = np.bincount(second_draws[:500], minlength=K2)

    PRIZE_FIXED = {
        (5, 1): 150000,
        (5, 0): 20000,
        (4, 1): 4000,
        (4, 0): 800,
        (3, 1): 400,
        (2, 1): 200,
        (3, 0): 100,
        (1, 1): 100,
    }

    results = []
    
    # Track overall hits
    total_hits = {key: 0 for key in [(6,1), (6,0), (5,1)]}

    for sim in range(N_SIMS):
        if (sim + 1) % 1000 == 0:
            print(f"  Progress: {sim+1}/{N_SIMS}...")
            
        rng = np.random.default_rng(sim)
        current_n_tickets = 2
        
        sim_cost = 0
        sim_prize = 0
        sim_draws_played = 0
        sim_hits = {key: 0 for key in [(6,1), (6,0), (5,1)]}
        sim_max_drawdown = 0.0
        
        counts1 = init_counts1.copy()
        counts2 = init_counts2.copy()
        
        running_net_profit = 0
        
        for i in range(500, len(df_chrono)):
            current_jp = jp_totals[i]
            
            # --- THRESHOLD CHECK ---
            if current_jp < THRESHOLD:
                # Update counts but don't play
                counts1[first_draws[i]] += 1
                counts2[second_draws[i]] += 1
                current_n_tickets = 2 # Reset
                continue
            
            # Use Cap-15 Dynamic Strategy when >= 800M
            sim_draws_played += 1
            w1 = counts1.astype(float) + ALPHA
            w2 = counts2.astype(float) + ALPHA
            penalty = PENALTY_BASE if current_n_tickets < 5 else PENALTY_LARGE
            
            cost_increase = current_n_tickets * TICKET_PRICE
            sim_cost += cost_increase
            
            actual_m = set(first_draws[i])
            actual_s = second_draws[i]
            
            used1 = np.zeros(K1, dtype=int)
            used2 = np.zeros(K2, dtype=int)
            
            sim_draw_prize_total = 0
            
            for _ in range(current_n_tickets):
                cur_w1 = w1 * (penalty ** used1)
                cur_w1 = np.clip(cur_w1, 1e-12, None)
                pick1 = sample_wor_es(rng, cur_w1, k=6)
                used1[pick1] += 1
                
                cur_w2 = w2 * (penalty ** used2)
                cur_w2 = np.clip(cur_w2, 1e-12, None)
                pick_s = sample_wor_es(rng, cur_w2, k=1)[0]
                used2[pick_s] += 1
                
                m1_hit = len(actual_m.intersection(set(pick1)))
                m2_hit = 1 if pick_s == actual_s else 0
                
                prize = 0
                if (m1_hit, m2_hit) == (6, 1):
                    win_amount = max(jp_totals[i], 200000000)
                    prize = win_amount
                    sim_hits[(6,1)] += 1
                    total_hits[(6,1)] += 1
                    print(f"!!! JACKPOT HIT !!! Sim {sim}, Draw {i} (Prize: {win_amount:,.0f})")
                elif (m1_hit, m2_hit) == (6, 0):
                    prize = sect2_prizes[i]
                    sim_hits[(6,0)] += 1
                    total_hits[(6,0)] += 1
                elif (m1_hit, m2_hit) == (5, 1):
                    prize = PRIZE_FIXED[(5,1)]
                    sim_hits[(5,1)] += 1
                    total_hits[(5,1)] += 1
                else:
                    prize = PRIZE_FIXED.get((m1_hit, m2_hit), 0)
                
                sim_draw_prize_total += prize
            
            sim_prize += sim_draw_prize_total
            running_net_profit += sim_draw_prize_total - cost_increase
            if running_net_profit < sim_max_drawdown:
                sim_max_drawdown = running_net_profit
            
            # Update history
            counts1[first_draws[i]] += 1
            counts2[second_draws[i]] += 1
            
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
            "h61": sim_hits[(6,1)],
            "h60": sim_hits[(6,0)],
            "h51": sim_hits[(5,1)]
        })
        
    return pd.DataFrame(results), total_hits

if __name__ == "__main__":
    res_df, total_hits = run_simulation()
    
    print("\n" + "="*50)
    print(f"--- 800M Threshold Strategy (Cap-15) Analysis Report (10,000 Sims) ---")
    print("="*50)
    participating_sims = res_df[res_df['draws'] > 0]
    n_participating = len(participating_sims)
    
    if n_participating == 0:
        print("No simulations participated (Jackpot never reached 800M or threshold too high).")
    else:
        print(f"總獲利次數: {len(res_df[res_df['profit'] > 0])} ({len(res_df[res_df['profit'] > 0])/N_SIMS*100:.2f}%)")
        print(f"平均參與期數: {res_df['draws'].mean():.1f} (Draws Played)")
        print(f"平均投入成本: ${res_df['cost'].mean():,.0f}")
        print(f"平均淨利 (EV): ${res_df['profit'].mean():,.0f}")
        print(f"淨利中位數: ${res_df['profit'].median():,.0f}")
        roi = (res_df['profit'].sum() / res_df['cost'].sum() * 100) if res_df['cost'].sum() > 0 else 0
        print(f"投資報酬率 (ROI): {roi:.2f}%")
        print(f"平均最大回撤 (Drawdown): ${res_df['max_drawdown'].mean():,.0f}")
        
        print("\n--- Hit Stats (Total) ---")
        print(f"Jackpot (6+1): {total_hits[(6,1)]}")
        print(f"2nd Prize (6+0): {total_hits[(6,0)]}")
        print(f"3rd Prize (5+1): {total_hits[(5,1)]}")
        
        print("\n--- Extremes ---")
        print(f"Max Profit: ${res_df['profit'].max():,.0f}")
        print(f"Max Loss: ${res_df['profit'].min():,.0f}")
        print("="*50)
    
    out_path = os.path.join(BASE_DIR, 'output', "replay_result_threshold_800m_cap15_10k.csv")
    res_df.to_csv(out_path, index=False)
