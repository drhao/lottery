import numpy as np
from numpy.random import default_rng

K1 = 38
PENALTIES = [0.1, 0.4, 0.6, 0.8, 1.0]
N_TICKETS = 10
N_TRIALS = 1000

def sample_wor_es(rng, w, k=6):
    u = rng.random(len(w))
    keys = -np.log(u) / w
    return np.argpartition(keys, k-1)[:k]

def pick_soft_diverse(rng, base_w, n_tickets, penalty):
    used = np.zeros(K1, dtype=int)
    tickets = []
    for _ in range(n_tickets):
        penalty_factors = penalty ** used
        w = base_w * penalty_factors
        w = np.clip(w, 1e-12, None)
        pick = sample_wor_es(rng, w, k=6)
        tickets.append(pick)
        used[pick] += 1
    return tickets

rng = default_rng(42)
base_w = np.ones(K1) + 100 # Smooth weights

print(f"Testing diversity for {N_TICKETS} tickets ({N_TRIALS} trials):")
print(f"{'Penalty':<10} | {'Avg Unique Numbers':<20} | {'Avg Overlap (pairs)':<20}")
print("-" * 60)

for p in PENALTIES:
    unique_counts = []
    overlaps = []
    for _ in range(N_TRIALS):
        tickets = pick_soft_diverse(rng, base_w, N_TICKETS, p)
        all_nums = np.concatenate(tickets)
        unique_nums = len(np.unique(all_nums))
        unique_counts.append(unique_nums)
        
    print(f"{p:<10} | {np.mean(unique_counts):<20.2f} | {60 - np.mean(unique_counts):<20.2f}")
