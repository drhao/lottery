import pandas as pd
import numpy as np

# =====================
# CONFIG
# =====================
CSV_PATH = "super_lotto638_results.csv"
TICKET_PRICE = 100
MAX_TICKETS = 15      # 上限 15 注
N_SIMS = 10000        # 10,000 sims 
THRESHOLD = 800000000 # 8-亿門檻
ALPHA = 100
PENALTY_BASE = 0.4
PENALTY_LARGE = 0.2
K1, K2 = 38, 8

def parse_currency(x):
    if isinstance(x, str):
        return float(x.replace(',', ''))
    return float(x)

def run_simulation():
    print(f"Initializing 8-亿 Threshold Optimization Strategy (Cap-15, N_SIMS={N_SIMS})...")
    df = pd.read_csv(CSV_PATH)
    
    # Prepare data (Old -> New)
    df_chrono = df.iloc[::-1].copy()
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

    for sim in range(N_SIMS):
        if (sim + 1) % 1000 == 0:
            print(f"  Progress: {sim+1}/{N_SIMS}...")
            
        rng = np.random.default_rng(sim)
        current_n_tickets = 2
        
        sim_cost = 0
        sim_prize = 0
        sim_draws_played = 0
        sim_hits = {key: 0 for key in [(6,1), (6,0), (5,1)]}
        
        counts1 = init_counts1.copy()
        counts2 = init_counts2.copy()
        
        for i in range(500, len(df_chrono)):
            current_jp = jp_totals[i]
            
            # --- THRESHOLD CHECK ---
            if current_jp < THRESHOLD:
                counts1[first_draws[i]] += 1
                counts2[second_draws[i]] += 1
                current_n_tickets = 2
                continue
            
            sim_draws_played += 1
            w1 = counts1.astype(float) + ALPHA
            w2 = counts2.astype(float) + ALPHA
            penalty = PENALTY_BASE if current_n_tickets < 5 else PENALTY_LARGE
            
            sim_cost += current_n_tickets * TICKET_PRICE
            
            actual_m = set(first_draws[i])
            actual_s = second_draws[i]
            
            used1 = np.zeros(K1, dtype=int)
            used2 = np.zeros(K2, dtype=int)
            
            for _ in range(current_n_tickets):
                cur_w1 = w1 * (penalty ** used1)
                cur_w1 = np.clip(cur_w1, 1e-12, None)
                pick1 = rng.choice(K1, size=6, replace=False, p=cur_w1/cur_w1.sum())
                used1[pick1] += 1
                
                cur_w2 = w2 * (penalty ** used2)
                cur_w2 = np.clip(cur_w2, 1e-12, None)
                pick_s = rng.choice(K2, size=1, p=cur_w2/cur_w2.sum())[0]
                used2[pick_s] += 1
                
                m1_hit = len(actual_m.intersection(set(pick1)))
                m2_hit = 1 if pick_s == actual_s else 0
                
                if (m1_hit, m2_hit) == (6, 1):
                    sim_prize += max(jp_totals[i], 200000000)
                    sim_hits[(6,1)] += 1
                elif (m1_hit, m2_hit) == (6, 0):
                    sim_prize += sect2_prizes[i]
                    sim_hits[(6,0)] += 1
                elif (m1_hit, m2_hit) == (5, 1):
                    sim_prize += PRIZE_FIXED[(5,1)]
                    sim_hits[(5,1)] += 1
                else:
                    sim_prize += PRIZE_FIXED.get((m1_hit, m2_hit), 0)
            
            counts1[first_draws[i]] += 1
            counts2[second_draws[i]] += 1
            
            if jp_per_winner[i] > 0:
                current_n_tickets = 2
            else:
                current_n_tickets = min(current_n_tickets + 1, MAX_TICKETS)

    results.append({
            "cost": sim_cost,
            "prize": sim_prize,
            "profit": sim_prize - sim_cost,
            "draws": sim_draws_played,
            "h61": sim_hits[(6,1)],
            "h60": sim_hits[(6,0)],
            "h51": sim_hits[(5,1)]
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    res_df = run_simulation()
    
    print("\n" + "="*50)
    print(f"--- 8-亿門檻策略 (Cap-15) 分析報告 (10,000次模擬) ---")
    print("="*50)
    print(f"總獲利次數: {len(res_df[res_df['profit'] > 0])} ({len(res_df[res_df['profit'] > 0])/N_SIMS*100:.2f}%)")
    print(f"平均參與期數: {res_df['draws'].mean():.1f}")
    print(f"平均投入成本: ${res_df['cost'].mean():,.0f}")
    print(f"平均淨利 (EV): ${res_df['profit'].mean():,.0f}")
    print(f"淨利中位數: ${res_df['profit'].median():,.0f}")
    print(f"投資報酬率 (ROI): {(res_df['profit'].mean()/res_df['cost'].mean()*100):.2f}%")
    
    print("\n--- 大獎命中統計 ---")
    print(f"頭獎 (6+1) 總次數: {res_df['h61'].sum()}")
    print(f"貳獎 (6+0) 總次數: {res_df['h60'].sum()}")
    print(f"參獎 (5+1) 總次數: {res_df['h51'].sum()}")
    
    print("\n--- 極端值分析 ---")
    print(f"單輪最高獲利: ${res_df['profit'].max():,.0f}")
    print(f"單輪最高虧損: ${res_df['profit'].min():,.0f}")
    print("="*50)
    
    res_df.to_csv("replay_result_threshold_800m_cap15_10k.csv", index=False)
