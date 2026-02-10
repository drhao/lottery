import pandas as pd
import numpy as np

# =====================
# CONFIG
# =====================
CSV_PATH = "super_lotto638_results.csv"
TICKET_PRICE = 100
MAX_TICKETS = 10
N_SIMS = 1000  # Fewer sims for quick analysis of timing
K1, K2 = 38, 8

def parse_currency(x):
    if isinstance(x, str):
        return float(x.replace(',', ''))
    return float(x)

def simulate_with_threshold(df, threshold, n_sims=1000):
    # Prepare data (Old -> New)
    df_chrono = df.iloc[::-1].copy()
    jp_totals = df_chrono['First_Prize_Total'].apply(parse_currency).values
    jp_per_winner = df_chrono['First_Prize_Per_Winner'].apply(parse_currency).values
    sect2_prizes = df_chrono['Second_Prize_Total'].apply(parse_currency).values
    
    first_draws = df_chrono["Numbers"].astype(str).str.split().apply(
        lambda xs: np.array([int(x) for x in xs], dtype=np.int16) - 1
    ).values
    second_draws = df_chrono["Special_Number"].astype(np.int16).to_numpy() - 1

    total_cost_all_sims = 0
    total_prize_all_sims = 0
    draws_played_all_sims = 0

    # Prize tiers (simplified as in the replay scripts)
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

    # Running history for weights (using a simple rolling frequency)
    # To speed up, we pre-calculate frequencies up to each point
    # but for simplicity and correctness in simulation, we'll do it inside
    
    # We'll use the same model as v3 (Alpha=100)
    ALPHA = 100
    PENALTY_BASE = 0.4
    PENALTY_LARGE = 0.2

    for sim in range(n_sims):
        rng = np.random.default_rng(sim)
        current_n_tickets = 2
        
        # Start tracking from a reasonable point (e.g. after index 500 to have history)
        for i in range(500, len(df_chrono)):
            current_jp = jp_totals[i]
            
            # --- THRESHOLD CHECK ---
            if current_jp < threshold:
                # Skip this draw
                # Reset ticket count for next time we play? 
                # Let's say we follow the same logic: if it was won, reset. 
                # If we didn't play, we don't know "last result" in our session context.
                # Simplified: Reset to 2 if we skip.
                current_n_tickets = 2
                continue
            
            # --- We PLAY this draw ---
            draws_played_all_sims += 1
            
            # Calculate weights based on history [0:i]
            hist_main = [num for sublist in first_draws[:i] for num in sublist]
            hist_spec = second_draws[:i]
            
            counts1 = np.bincount(hist_main, minlength=K1)
            counts2 = np.bincount(hist_spec, minlength=K2)
            
            w1 = counts1.astype(float) + ALPHA
            w2 = counts2.astype(float) + ALPHA
            
            penalty = PENALTY_BASE if current_n_tickets < 5 else PENALTY_LARGE
            
            # Buy tickets
            cost = current_n_tickets * TICKET_PRICE
            total_cost_all_sims += cost
            
            # Simulate tickets
            actual_m = set(first_draws[i])
            actual_s = second_draws[i]
            
            # For each ticket...
            draw_prize = 0
            # Note: We simulate the tickets in a way that respects the penalty
            used1 = np.zeros(K1, dtype=int)
            used2 = np.zeros(K2, dtype=int)
            
            for _ in range(current_n_tickets):
                # Ticket Main
                cur_w1 = w1 * (penalty ** used1)
                cur_w1 = np.clip(cur_w1, 1e-12, None)
                pick1 = rng.choice(K1, size=6, replace=False, p=cur_w1/cur_w1.sum())
                used1[pick1] += 1
                
                # Ticket Spec
                cur_w2 = w2 * (penalty ** used2)
                cur_w2 = np.clip(cur_w2, 1e-12, None)
                pick_s = rng.choice(K2, size=1, p=cur_w2/cur_w2.sum())[0]
                used2[pick_s] += 1
                
                # Check Prize
                m1_hit = len(actual_m.intersection(set(pick1)))
                m2_hit = 1 if pick_s == actual_s else 0
                
                if (m1_hit, m2_hit) == (6, 1):
                    draw_prize += max(jp_totals[i], 200000000)
                elif (m1_hit, m2_hit) == (6, 0):
                    draw_prize += sect2_prizes[i]
                else:
                    draw_prize += PRIZE_FIXED.get((m1_hit, m2_hit), 0)
            
            total_prize_all_sims += draw_prize
            
            # Update current_n_tickets for NEXT draw based on REAL result
            if jp_per_winner[i] > 0:
                current_n_tickets = 2
            else:
                current_n_tickets = min(current_n_tickets + 1, MAX_TICKETS)

    roi = ((total_prize_all_sims - total_cost_all_sims) / total_cost_all_sims * 100) if total_cost_all_sims > 0 else 0
    return {
        "threshold": threshold,
        "draws_played": draws_played_all_sims / n_sims,
        "total_cost": total_cost_all_sims / n_sims,
        "total_prize": total_prize_all_sims / n_sims,
        "net_profit": (total_prize_all_sims - total_cost_all_sims) / n_sims,
        "roi": roi
    }

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    thresholds = [0, 300000000, 500000000, 800000000, 1000000000, 1200000000]
    
    results = []
    print(f"Analyzing Threshold Strategy (N_SIMS=1000)...")
    for t in thresholds:
        print(f"  Testing Threshold: {t/1e8:.1f} 億...")
        res = simulate_with_threshold(df, t, n_sims=1000)
        results.append(res)
    
    print("\n--- 門檻策略分析報告 ---")
    print(f"{'門檻 (億)':>10} | {'參與期數':>8} | {'平均成本':>10} | {'平均淨利':>12} | {'ROI (%)':>8}")
    print("-" * 65)
    for r in results:
        print(f"{r['threshold']/1e8:10.1f} | {r['draws_played']:8.1f} | {r['total_cost']:10,.0f} | {r['net_profit']:12,.0f} | {r['roi']:8.2f}%")
