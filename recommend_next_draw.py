import numpy as np
import pandas as pd
from numpy.random import default_rng

# =====================
# CONFIG
# =====================
CSV_PATH = "super_lotto638_results.csv"
INITIAL_TICKETS = 2
MAX_TICKETS = 10
ALPHA = 100
PENALTY_BASE = 0.4
PENALTY_LARGE_BATCH = 0.2
K1, K2 = 38, 8

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
# MAIN
# =====================
print("Loading data for recommendation (v3: Dynamic Penalty + 2nd Zone Model)...")
df = pd.read_csv(CSV_PATH)

# 1. Parse History for Weights
first_draws = df["Numbers"].astype(str).str.split().apply(
    lambda xs: np.array([int(x) for x in xs], dtype=np.int16) - 1
)
second_draws = df["Special_Number"].astype(np.int16).to_numpy() - 1

base_counts = np.zeros(K1, dtype=int)
for arr in first_draws:
    base_counts[arr] += 1

base_counts_s = np.zeros(K2, dtype=int)
for s in second_draws:
    base_counts_s[s] += 1

# 2. Determine Ticket Count (Dynamic Strategy)
try:
    first_prize_won = (df["First_Prize_Per_Bet"].astype(str).str.replace(',', '').astype(float) > 0).to_numpy()
except:
    first_prize_won = (df["First_Prize_Per_Bet"] > 0).to_numpy()

current_n_tickets = INITIAL_TICKETS
for won in first_prize_won:
    if won:
        current_n_tickets = INITIAL_TICKETS
    else:
        current_n_tickets = min(current_n_tickets + 1, MAX_TICKETS)

print(f"\n==========================================")
print(f"       NEXT DRAW RECOMMENDATION (v3)")
print(f"==========================================")
print(f"Strategy: Dynamic Cap-10 + 2nd Zone Model")
print(f"Dynamic Penalty: {'Active (0.2)' if current_n_tickets >= 5 else 'Base (0.4)'}")
print(f"Last Draw Result: {'JACKPOT WON!' if first_prize_won[-1] else 'Rollover (No Jackpot)'}")
print(f"Adjusted Ticket Count for Next Draw: {current_n_tickets} Tickets (${current_n_tickets * 100})")
print(f"------------------------------------------")

# 3. Generate Recommendations
rng = default_rng() 
current_penalty = PENALTY_BASE if current_n_tickets < 5 else PENALTY_LARGE_BATCH

# First Zone
base_w = base_counts.astype(float) + ALPHA
tickets = pick_soft_diverse(rng, base_w, current_n_tickets, current_penalty, k=6, total_nums=K1)

# Second Zone
base_w_s = base_counts_s.astype(float) + ALPHA
specials_picked = pick_soft_diverse(rng, base_w_s, current_n_tickets, current_penalty, k=1, total_nums=K2)
specials = [p[0] for p in specials_picked]

# Output
for i, pick in enumerate(tickets):
    nums = sorted(pick + 1)
    nums_str = " ".join(f"{x:02d}" for x in nums)
    spec_str = f"{specials[i]+1:02d}"
    print(f"Ticket #{i+1:02d}: [{nums_str}]  Special: {spec_str}")

print(f"==========================================")
print(f"Good Luck!")
