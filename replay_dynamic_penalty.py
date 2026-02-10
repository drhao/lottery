# replay_dynamic_penalty.py
import numpy as np
import pandas as pd
from numpy.random import default_rng

# =====================
# CONFIG
# =====================
CSV_PATH = "super_lotto638_results.csv"
N_SIMS = 1000  # Reduced for faster experimentation
TEST_LEN = 1000
TICKET_PRICE = 100
INITIAL_TICKETS = 2
MAX_TICKETS = 10

# Dynamic Penalty Config
PENALTY_BASE = 0.4
# Experiment: If tickets > 5, use a different penalty.
# LOWER value = STRONGER penalty (more diverse)
# HIGHER value = WEAKER penalty (more concentrated)
PENALTY_LARGE_BATCH = 0.2  # Let's try "more diverse" for >= 5 tickets

K1, K2 = 38, 8
ALPHA = 100

PRIZE_FIXED = {
    (5,1):150000, (5,0):20000,
    (4,1):4000,   (4,0):800,
    (3,1):400,    (3,0):100,
    (2,1):200,
    (1,1):100
}

HIT_EVENTS = [
    (6,1), (6,0),
    (5,1), (5,0),
    (4,1), (4,0),
    (3,1), (3,0),
    (2,1),
    (1,1),
]

# =====================
# UTIL
# =====================
def sample_wor_es(rng, w, k=6):
    """Efraimidis–Spirakis weighted sampling without replacement."""
    u = rng.random(len(w))
    keys = -np.log(u) / w
    return np.argpartition(keys, k-1)[:k]

def pick_soft_diverse(rng, base_w, n_tickets, penalty, k=6, total_nums=K1):
    """N tickets, soft diversity within the same draw batch."""
    used = np.zeros(total_nums, dtype=int)
    tickets = []
    
    for _ in range(n_tickets):
        penalty_factors = penalty ** used
        w = base_w * penalty_factors
        w = np.clip(w, 1e-12, None)
        
        pick = sample_wor_es(rng, w, k=k)
        tickets.append(pick)
        used[pick] += 1
    return tickets

# =====================
# LOAD DATA
# =====================
df = pd.read_csv(CSV_PATH)
# The CSV has newest on top. Flip it to be chronological (oldest on top) for correct replay.
df = df.iloc[::-1].reset_index(drop=True)

first_draws = df["Numbers"].astype(str).str.split().apply(
    lambda xs: np.array([int(x) for x in xs], dtype=np.int16) - 1
)
second_draws = df["Special_Number"].astype(np.int16).to_numpy() - 1

try:
    first_prize_won = (df["First_Prize_Per_Bet"].astype(str).str.replace(',', '').astype(float) > 0).to_numpy()
except:
    first_prize_won = (df["First_Prize_Per_Bet"] > 0).to_numpy()

T = len(df)
start = T - TEST_LEN
test_first = [first_draws.iloc[t] for t in range(start, T)]
test_second = second_draws[start:T]

def parse_currency(series):
    try:
        return series.astype(str).str.replace(',', '').astype(float).fillna(0).to_numpy()
    except:
        return series.fillna(0).to_numpy()

first_prize_total = parse_currency(df["First_Prize_Total"])[start:T]
second_prize_total = parse_currency(df["Second_Prize_Total"])[start:T]

masks = np.zeros((TEST_LEN, K1), dtype=bool)
for i, arr in enumerate(test_first):
    masks[i, arr] = True

# warm-up counts
base_counts = np.zeros(K1, dtype=int)
base_counts_s = np.zeros(K2, dtype=int)
for t in range(start):
    base_counts[first_draws.iloc[t]] += 1
    base_counts_s[second_draws[t]] += 1

# =====================
# MAIN REPLAY
# =====================
rng = default_rng(999999)
records = []
jackpot_hits_info = []

for s in range(N_SIMS):
    counts = base_counts.copy()
    counts_s = base_counts_s.copy()
    total_prize_real = 0
    hit_counter = {evt: 0 for evt in HIT_EVENTS}
    current_n_tickets = INITIAL_TICKETS
    total_cost = 0
    max_drawdown = 0

    for i in range(TEST_LEN):
        # Choose Penalty based on batch size
        current_penalty = PENALTY_BASE if current_n_tickets < 5 else PENALTY_LARGE_BATCH

        # first-zone
        base_w = counts.astype(float) + ALPHA
        tickets = pick_soft_diverse(rng, base_w, current_n_tickets, current_penalty, k=6, total_nums=K1)
        
        # second-zone (Diversity approach)
        base_w_s = counts_s.astype(float) + ALPHA
        specials_picked = pick_soft_diverse(rng, base_w_s, current_n_tickets, current_penalty, k=1, total_nums=K2)
        specials = [p[0] for p in specials_picked]

        total_cost += current_n_tickets * TICKET_PRICE
        mask = masks[i]
        draw_s = int(test_second[i])

        for t_idx, pick in enumerate(tickets):
            m1 = int(mask[pick].sum())
            ms = int(specials[t_idx] == draw_s)

            if (m1, ms) in hit_counter:
                hit_counter[(m1, ms)] += 1

            if (m1, ms) in PRIZE_FIXED:
                total_prize_real += PRIZE_FIXED[(m1, ms)]
            elif (m1, ms) == (6, 1): # Jackpot
                jackpot_amount = max(first_prize_total[i], 200000000)
                total_prize_real += jackpot_amount 
                # Record details
                jackpot_hits_info.append({
                    "sim_idx": s,
                    "draw_idx_in_test": i,
                    "global_idx": start + i,
                    "period": df.iloc[start + i]["Period"],
                    "date": df.iloc[start + i]["Date"],
                    "amount": jackpot_amount
                })
            elif (m1, ms) == (6, 0): # Second Prize
                total_prize_real += second_prize_total[i]

        # Determine tickets for NEXT draw
        global_idx = start + i
        if first_prize_won[global_idx]:
           current_n_tickets = INITIAL_TICKETS
        else:
           current_n_tickets = min(current_n_tickets + 1, MAX_TICKETS)

        # update history
        counts[test_first[i]] += 1
        counts_s[test_second[i]] += 1
        
        current_net_profit = total_prize_real - total_cost
        if current_net_profit < max_drawdown:
            max_drawdown = current_net_profit

    records.append({
        "total_prize_real": total_prize_real,
        "net_profit_real": total_prize_real - total_cost,
        "max_drawdown": max_drawdown,
        **{f"hit_{m1}_{ms}": hit_counter[(m1, ms)] for (m1, ms) in HIT_EVENTS}
    })

# =====================
# OUTPUT
# =====================
out = pd.DataFrame(records)
out.to_csv("replay_dynamic_penalty_result.csv", index=False)

print(f"=== SUMMARY (Dynamic Penalty: <5:{PENALTY_BASE}, >=5:{PENALTY_LARGE_BATCH}) ===")
print(out["net_profit_real"].describe())

print("\nMax Drawdown (Worst cumulative loss):")
print(out["max_drawdown"].describe())

print("\nHit rates (mean per 1000 draws):")
for evt in [(6,1),(6,0),(5,1),(5,0),(4,1),(4,0)]:
    col = f"hit_{evt[0]}_{evt[1]}"
    print(f"{col}: {out[col].mean():.3f}")

if jackpot_hits_info:
    print("\n=== JACKPOT HIT DETAILS ===")
    for info in jackpot_hits_info:
        print(f"Sim {info['sim_idx']}: Hit on Draw {info['draw_idx_in_test']} (Period {info['period']}, Date {info['date']}) - Amount: {info['amount']:,.0f}")
else:
    print("\nNo Jackpot hits recorded.")
