# replay_2tickets_soft_p04_n500.py
import numpy as np
import pandas as pd
from numpy.random import default_rng

# =====================
# CONFIG
# =====================
CSV_PATH = "super_lotto638_results.csv"   # 改成你的檔案路徑
N_SIMS = 2000
TEST_LEN = 1000
PENALTY = 0.4
TICKET_PRICE = 100
# Dynamic ticket strategy: Starts at 2, increases by 1 on rollover
INITIAL_TICKETS = 2
MAX_TICKETS = 10







K1, K2 = 38, 8
ALPHA = 100  # Keep current ALPHA

# 固定獎金（不含 6+0, 6+1：浮動派彩）
PRIZE_FIXED = {
    (5,1):150000, (5,0):20000,
    (4,1):4000,   (4,0):800,
    (3,1):400,    (3,0):100,
    (2,1):200,
    (1,1):100
}

# 你想追蹤的命中事件（含 6+0, 6+1）
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

def pick_soft_diverse(rng, base_w, n_tickets):
    """N tickets, soft diversity within the same draw."""
    used = np.zeros(K1, dtype=int)
    # Penalize based on how many times a number has been used in this batch
    # power table: [1.0, PENALTY, PENALTY^2, PENALTY^3, ...]
    # We can compute it on the fly or pre-compute enough
    tickets = []
    
    for _ in range(n_tickets):
        # penalize weight if number is already used 'used' times
        penalty_factors = PENALTY ** used
        w = base_w * penalty_factors
        w = np.clip(w, 1e-12, None)
        
        pick = sample_wor_es(rng, w, k=6)
        tickets.append(pick)
        used[pick] += 1
    return tickets

# =====================
# LOAD DATA
# =====================
df = pd.read_csv(CSV_PATH)
first_draws = df["Numbers"].astype(str).str.split().apply(
    lambda xs: np.array([int(x) for x in xs], dtype=np.int16) - 1
)
second_draws = df["Special_Number"].astype(np.int16).to_numpy() - 1
# Check for jackpot winners (First_Prize_Per_Winner > 0 means won)
# If column is string with commas, remove them first
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

# Retrieve history of jackpot wins for the test period
# Note: ticket count depends on PREVIOUS draw's result.
# So for test step i (index 'start+i'), we look at result of 'start+i-1'.
# We need history leading up to start.
full_first_prize_won = first_prize_won

# Load prize pool data for test period
# If column contains commas, remove them
def parse_currency(series):
    try:
        return series.astype(str).str.replace(',', '').astype(float).fillna(0).to_numpy()
    except:
        return series.fillna(0).to_numpy()

first_prize_total = parse_currency(df["First_Prize_Total"])[start:T]
second_prize_total = parse_currency(df["Second_Prize_Total"])[start:T]
first_prize_count = parse_currency(df["First_Prize_Per_Winner"])[start:T] > 0 # rough check if anyone won
second_prize_count = parse_currency(df["Second_Prize_Per_Winner"])[start:T] > 0

# Original winners count is not directly available as integer, but we can estimate share.
# For simplicity:
# If I hit jackpot and nobody else did: I win Total.
# If I hit jackpot and someone else did: I win Total / 2 (assuming 1 other winner for simplicity, or just take reported Per_Bet)
# Let's use a logic:
# My Prize = First_Prize_Total / (Original_Winners + 1)
# Original_Winners = First_Prize_Total / First_Prize_Per_Winner (if Per_Winner > 0) else 0



masks = np.zeros((TEST_LEN, K1), dtype=bool)
for i, arr in enumerate(test_first):
    masks[i, arr] = True

# warm-up counts (history before test window)
base_counts = np.zeros(K1, dtype=int)
for t in range(start):
    base_counts[first_draws.iloc[t]] += 1

# =====================
# MAIN REPLAY
# =====================
rng = default_rng(20260210)
records = []

for s in range(N_SIMS):
    counts = base_counts.copy()
    total_prize_fixed = 0
    total_prize_real = 0  # Includes 6+1 and 6+0


    # 每個事件都先填 0，確保輸出欄位一定存在
    hit_counter = {evt: 0 for evt in HIT_EVENTS}

    # Initialize dynamic tickets
    # Check the draw immediately BEFORE the test start to set initial state?
    # Or just assume start fresh at 2?
    # Let's assume start fresh at INITIAL_TICKETS for simplicity of simulation state,
    # OR we can trace back actual history.
    # User's request: "從一期投注兩張開始" -> implied fresh start.
    current_n_tickets = INITIAL_TICKETS
    
    total_cost = 0
    max_drawdown = 0 # Worst profit seen so far (should be <= 0)

    for i in range(TEST_LEN):


        # first-zone base weights (Dirichlet-smoothed)
        base_w = counts.astype(float) + ALPHA

        # Dynamic Tickets Strategy
        # Use current_n_tickets
        tickets = pick_soft_diverse(rng, base_w, current_n_tickets)
        
        # Track cost
        total_cost += current_n_tickets * TICKET_PRICE

        # second-zone: M different specials each draw (if M <= 8)
        if current_n_tickets <= K2:
            specials = rng.choice(K2, size=current_n_tickets, replace=False)
        else:
            specials = rng.choice(K2, size=current_n_tickets, replace=True)

        mask = masks[i]
        draw_s = int(test_second[i])

        for t_idx, pick in enumerate(tickets):

            m1 = int(mask[pick].sum())                 # first-zone matches 0..6
            ms = int(specials[t_idx] == draw_s)        # special hit 0/1

            # 1) 記錄命中事件（含 6+0, 6+1）
            if (m1, ms) in hit_counter:
                hit_counter[(m1, ms)] += 1

            # 2) 固定獎金
            if (m1, ms) in PRIZE_FIXED:
                prize = PRIZE_FIXED[(m1, ms)]
                total_prize_fixed += prize
                total_prize_real += prize
            
            # 3) 浮動獎金 (6+1 頭獎, 6+0 貳獎)
            elif (m1, ms) == (6, 1): # Jackpot
                # Calculate share
                # If original per bet > 0, calculate original winners
                # We need data for this specific draw 'i'
                total_pool = first_prize_total[i]
                # Try to get per_bet for original winners count
                # We don't have per_bet array sliced yet, let's just use raw logic
                # Simplified: If nobody won (most cases I target), I win Total.
                # If somebody won, I share.
                # Let's just take the Total first. If multiple tickets of MINE hit, they share it too?
                # Actually, if I buy 2 winning tickets, I get 2 shares.
                # But let's assume I am the only additional winner.
                
                # Check original winners count
                # We need to read per_bet from DF again or pre-calc
                # Let's use a simpler heuristic for now: I win the FULL JACKPOT if previously 0 winners.
                # If there were winners, I get (Total / 2) roughly.
                # Actually let's just add Total. It's an optimistic simulation.
                
                # Apply Minimum Guaranteed Jackpot (200 Million)
                jackpot_amount = max(total_pool, 200000000)
                total_prize_real += jackpot_amount 
                
            elif (m1, ms) == (6, 0): # Second Prize
                total_prize_real += second_prize_total[i]

        # Determine tickets for NEXT draw based on THIS draw's global result
        # We need to know if the global jackpot was won in this draw (i)
        # The index in full arrays is 'start + i'
        global_idx = start + i
        if full_first_prize_won[global_idx]:
           # Jackpot won! Reset.
           current_n_tickets = INITIAL_TICKETS
        else:
           # Rollover! Add 1 ticket.
           current_n_tickets += 1
        
        # Apply Cap
        if current_n_tickets > MAX_TICKETS:
            current_n_tickets = MAX_TICKETS

        # update history after seeing the real outcome of this draw
        counts[test_first[i]] += 1
        
        # Track running net profit for drawdown calculation
        current_net_profit = total_prize_real - total_cost
        if current_net_profit < max_drawdown:
            max_drawdown = current_net_profit

    records.append({
        "total_prize_fixed": total_prize_fixed,
        "total_prize_real": total_prize_real,
        "net_profit_fixed": total_prize_fixed - total_cost,
        "net_profit_real": total_prize_real - total_cost,
        "max_drawdown": max_drawdown,
        # 展開命中次數欄位
        **{f"hit_{m1}_{ms}": hit_counter[(m1, ms)] for (m1, ms) in HIT_EVENTS}
    })

# =====================
# OUTPUT
# =====================
out = pd.DataFrame(records)
out.to_csv("replay_result_n500.csv", index=False)

print("=== SUMMARY (Real Prizes with Jackpot) ===")
print(out["total_prize_real"].describe())

print("\nNet profit (Real Prizes):")
print(out["net_profit_real"].describe())

print("\nMax Drawdown (Worst cumulative loss during run):")
print(out["max_drawdown"].describe())

print("\nHit rate check (mean counts per 1000 draws):")
for evt in [(6,1),(6,0),(5,1),(5,0),(4,1),(4,0)]:
    col = f"hit_{evt[0]}_{evt[1]}"
    print(f"{col}: {out[col].mean():.3f}")
    col = f"hit_{evt[0]}_{evt[1]}"
    print(f"{col}: {out[col].mean():.3f}")