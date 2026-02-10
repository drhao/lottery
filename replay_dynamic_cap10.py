import numpy as np
import pandas as pd
from numpy.random import default_rng

print("Initializing Dynamic Lottery Replay (v3: Cap=10, 3k Sims, Dynamic Penalty, 2nd Zone Model)...")

# =====================
# CONFIG
# =====================
CSV_PATH = "super_lotto638_results.csv"
N_SIMS = 10000        # Changed to 10k sims as requested
TEST_LEN = 1000       # Simulate 1000 draws per run
TICKET_PRICE = 100    # Cost per ticket

# Dynamic Strategy Settings
INITIAL_TICKETS = 2
MAX_TICKETS = 10      # Cap at 10 tickets ($1000) per draw
ALPHA = 100           # Hot number bias
WINDOW_LEN = 0        # 0 = All history, >0 = Moving window

# Dynamic Penalty Config
PENALTY_BASE = 0.4
PENALTY_LARGE_BATCH = 0.2  # Stronger diversity for larger batches (>= 5)

K1, K2 = 38, 8

# Fixed Prize Table (Small prizes)
PRIZE_FIXED = {
    (5,1):150000, (5,0):20000,
    (4,1):4000,   (4,0):800,
    (3,1):400,    (3,0):100,
    (2,1):200,
    (1,1):100
}

# Events to track
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

def parse_currency(series):
    try:
        return series.astype(str).str.replace(',', '').astype(float).fillna(0).to_numpy()
    except:
        return series.fillna(0).to_numpy()

# =====================
# LOAD & PREP DATA
# =====================
print("Loading data...")
df = pd.read_csv(CSV_PATH)
# The CSV is New -> Old. We want Old -> New for proper sequential simulation.
df = df.iloc[::-1].reset_index(drop=True)

first_draws = df["Numbers"].astype(str).str.split().apply(
    lambda xs: np.array([int(x) for x in xs], dtype=np.int16) - 1
)
second_draws = df["Special_Number"].astype(np.int16).to_numpy() - 1

# Check jackpot winners for dynamic strategy
try:
    first_prize_won = (df["First_Prize_Per_Winner"].astype(str).str.replace(',', '').astype(float) > 0).to_numpy()
except:
    first_prize_won = (df["First_Prize_Per_Winner"] > 0).to_numpy()

T = len(df)
if TEST_LEN >= T:
    raise ValueError("TEST_LEN must be < number of draws in CSV")

start = T - TEST_LEN
test_first = [first_draws.iloc[t] for t in range(start, T)]
test_second = second_draws[start:T]
full_first_prize_won = first_prize_won

# Load prize pools
first_prize_total = parse_currency(df["First_Prize_Total"])[start:T]
second_prize_total = parse_currency(df["Second_Prize_Total"])[start:T]

masks = np.zeros((TEST_LEN, K1), dtype=bool)
for i, arr in enumerate(test_first):
    masks[i, arr] = True

# Warm-up counts (Base)
base_counts = np.zeros(K1, dtype=int)
base_counts_s = np.zeros(K2, dtype=int)
for t in range(start):
    base_counts[first_draws.iloc[t]] += 1
    base_counts_s[second_draws[t]] += 1

# =====================
# MAIN LOOP
# =====================
print(f"Starting simulation ({N_SIMS} runs)...")
rng = default_rng(20260210)
records = []

for s in range(N_SIMS):
    if (s+1) % 500 == 0:
        print(f"  Run {s+1}/{N_SIMS}...")
        
    total_prize_real = 0
    hit_counter = {evt: 0 for evt in HIT_EVENTS}
    jackpot_hits_in_run = []
    current_n_tickets = INITIAL_TICKETS
    total_cost = 0
    max_drawdown = 0.0

    # Initialize rolling counts
    current_counts = base_counts.copy()
    current_counts_s = base_counts_s.copy()
    
    # If windowed, handle appropriately
    if WINDOW_LEN > 0:
        current_counts = np.zeros(K1, dtype=int)
        current_counts_s = np.zeros(K2, dtype=int)
        history_start_idx = max(0, start - WINDOW_LEN)
        for t in range(history_start_idx, start):
            current_counts[first_draws.iloc[t]] += 1
            current_counts_s[second_draws[t]] += 1

    for i in range(TEST_LEN):
        # 0. Choose Penalty based on batch size
        current_penalty = PENALTY_BASE if current_n_tickets < 5 else PENALTY_LARGE_BATCH

        # 1. Select Tickets (First Zone)
        base_w = current_counts.astype(float) + ALPHA
        tickets = pick_soft_diverse(rng, base_w, current_n_tickets, current_penalty, k=6, total_nums=K1)
        
        # 2. Select Specials (Second Zone Model)
        base_w_s = current_counts_s.astype(float) + ALPHA
        specials_picked = pick_soft_diverse(rng, base_w_s, current_n_tickets, current_penalty, k=1, total_nums=K2)
        specials = [p[0] for p in specials_picked]
        
        # 3. Track Cost
        total_cost += current_n_tickets * TICKET_PRICE

        mask = masks[i]
        draw_s = int(test_second[i])

        # 4. Check Hits
        for t_idx, pick in enumerate(tickets):
            m1 = int(mask[pick].sum())
            ms = int(specials[t_idx] == draw_s)

            if (m1, ms) in hit_counter:
                hit_counter[(m1, ms)] += 1

            # Prize Logic
            prize = 0
            if (m1, ms) in PRIZE_FIXED:
                prize = PRIZE_FIXED[(m1, ms)]
            elif (m1, ms) == (6, 1): # Jackpot
                prize = max(first_prize_total[i], 200000000)
                jackpot_hits_in_run.append({
                    "draw_idx": i,
                    "period": df.iloc[start + i]["Period"]
                })
            elif (m1, ms) == (6, 0): # Second Prize
                prize = second_prize_total[i]
            
            total_prize_real += prize

        # 5. Drawdown Tracking
        current_net_profit = total_prize_real - total_cost
        if current_net_profit < max_drawdown:
            max_drawdown = current_net_profit

        # 6. Dynamic Strategy Update for NEXT draw
        global_idx = start + i
        if full_first_prize_won[global_idx]:
           current_n_tickets = INITIAL_TICKETS
        else:
           current_n_tickets = min(current_n_tickets + 1, MAX_TICKETS)

        # 7. Update History (Rolling Window or Cumulative)
        current_counts[test_first[i]] += 1
        current_counts_s[test_second[i]] += 1
        
        if WINDOW_LEN > 0:
            remove_idx = (start + i) - WINDOW_LEN
            if remove_idx >= 0:
                current_counts[first_draws.iloc[remove_idx]] -= 1
                current_counts_s[second_draws[remove_idx]] -= 1

    records.append({
        "total_prize_real": total_prize_real,
        "net_profit_real": total_prize_real - total_cost,
        "max_drawdown": max_drawdown,
        "jackpot_hits": jackpot_hits_in_run,
        **{f"hit_{m1}_{ms}": hit_counter[(m1, ms)] for (m1, ms) in HIT_EVENTS}
    })

# =====================
# ANALYSIS
# =====================
out = pd.DataFrame(records)
print("\n=== FINAL REPORT (v3: Dynamic Penalty + 2nd Zone Model) ===")
print(f"Hit 6+1 (Jackpot) Count: {(out['hit_6_1'] > 0).sum()}")
print(f"Hit 6+0 (2nd Prize) Count: {(out['hit_6_0'] > 0).sum()}")

print("\n--- Net Profit Statistics ---")
print(out["net_profit_real"].describe().apply(lambda x: format(x, 'f')))

print("\n--- Max Drawdown (Risk) ---")
print(out["max_drawdown"].describe().apply(lambda x: format(x, 'f')))

# Analysis of Jackpot Timing
all_jackpot_hits = []
for r in records:
    for hit in r["jackpot_hits"]:
        all_jackpot_hits.append(hit["draw_idx"])

if all_jackpot_hits:
    print("\n--- Jackpot Timing Analysis ---")
    print(f"Total Jackpots: {len(all_jackpot_hits)}")
    print(f"Average hit at draw index: {np.mean(all_jackpot_hits):.1f}")
    print(f"Hit indices: {sorted(all_jackpot_hits)}")
    print("(Note: Index 0 is the start of the 1000-draw test period, 999 is the end)")
else:
    print("\nNo Jackpots hit in this simulation.")

# Save
out.to_csv("replay_result_v3_cap10_10k.csv", index=False)
print("\nResults saved to replay_result_v3_cap10_10k.csv")
