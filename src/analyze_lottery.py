import csv
import random
from collections import Counter, defaultdict

def load_data(filename):
    data = []
    try:
        with open(filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    return data

def analyze_and_predict(data):
    # Section 1: Numbers 1-38
    # Section 2: Numbers 1-8
    
    sec1_history = []
    sec2_history = []
    
    # Track when a number was last seen (index of the draw, 0 being the most recent execution of the loop, but actually we iterate from oldest to newest usually? 
    # Let's verify data order. The content view showed Descending order (newest first).
    # "115000011,2026-02-05..." (Line 2)
    # "115000010,2026-02-02..." (Line 3)
    # So data[0] is the NEWEST.
    
    # For overdue calculation, it's easier to iterate from newest to oldest.
    # index 0 is "0 draws ago" (most recent).
    
    sec1_last_seen = {}
    sec2_last_seen = {}
    
    total_draws = len(data)
    
    for idx, row in enumerate(data):
        # Parse numbers
        try:
            nums = list(map(int, row["Numbers"].split()))
            special = int(row["Special_Number"])
        except ValueError:
            continue
            
        sec1_history.extend(nums)
        sec2_history.append(special)
        
        # Update last seen if not already recorded (finding the *first* occurrence from the start of the list = most recent)
        for n in nums:
            if n not in sec1_last_seen:
                sec1_last_seen[n] = idx
        
        if special not in sec2_last_seen:
            sec2_last_seen[special] = idx

    # Calculate Frequency
    sec1_freq = Counter(sec1_history)
    sec2_freq = Counter(sec2_history)
    
    # Calculate Overdue (Current draws since last appearance)
    # If a number hasn't appeared in the dataset (unlikely for long history), set it to total_draws
    sec1_overdue = {n: sec1_last_seen.get(n, total_draws) for n in range(1, 39)}
    sec2_overdue = {n: sec2_last_seen.get(n, total_draws) for n in range(1, 9)}

    print("="*40)
    print(f"Analysis of {total_draws} Draws")
    print("="*40)
    
    # Top 5 Hot Numbers
    print("\n[Section 1] Top 5 Hot Numbers (most frequent):")
    for n, count in sec1_freq.most_common(5):
        print(f"  No. {n:02d} : {count} times")
        
    print("\n[Section 2] Top 3 Hot Numbers:")
    for n, count in sec2_freq.most_common(3):
        print(f"  No. {n:02d} : {count} times")

    # Top 5 Overdue Numbers
    print("\n[Section 1] Top 5 Overdue Numbers (longest missing):")
    sorted_sec1_overdue = sorted(sec1_overdue.items(), key=lambda x: x[1], reverse=True)
    for n, draws in sorted_sec1_overdue[:5]:
        print(f"  No. {n:02d} : missing for {draws} draws")

    print("\n[Section 2] Top 2 Overdue Numbers:")
    sorted_sec2_overdue = sorted(sec2_overdue.items(), key=lambda x: x[1], reverse=True)
    for n, draws in sorted_sec2_overdue[:2]:
        print(f"  No. {n:02d} : missing for {draws} draws")

    # --- Prediction Models ---
    print("\n" + "="*40)
    print("Predictions for Next Draw")
    print("="*40)

    def generate_numbers(weights_sec1, weights_sec2):
        # Generate 6 unique numbers for Section 1
        # random.choices could pick duplicates, so we pick iteratively or use sample with weights if possible, 
        # but sample doesn't support weights directly in older python, choices does (with replacement).
        # We'll use a weighted shuffle approach or repeated choices until distinct.
        
        candidates_1 = list(range(1, 39))
        w1 = [weights_sec1.get(n, 0) for n in candidates_1]
        
        # Pick 6 distinct
        picks_1 = set()
        while len(picks_1) < 6:
            # Normalize w1 to avoid errors if sum is 0 (unlikely)
            p = random.choices(candidates_1, weights=w1, k=1)[0]
            picks_1.add(p)
            
        final_1 = sorted(list(picks_1))
        
        # Pick 1 for Section 2
        candidates_2 = list(range(1, 9))
        w2 = [weights_sec2.get(n, 0) for n in candidates_2]
        final_2 = random.choices(candidates_2, weights=w2, k=1)[0]
        
        return final_1, final_2

    # Model 1: Trend Follower (Weight = Frequency)
    w_trend_1 = {n: sec1_freq[n] for n in range(1, 39)}
    w_trend_2 = {n: sec2_freq[n] for n in range(1, 9)}
    
    pred_trend = generate_numbers(w_trend_1, w_trend_2)
    print(f"\n1. [The Trend Follower] (Favoring Hot Numbers)")
    print(f"   Section 1: {pred_trend[0]}")
    print(f"   Section 2: {pred_trend[1]}")

    # Model 2: The Contrarian (Weight = Overdue ^ 2 to emphasize outliers)
    w_cold_1 = {n: sec1_overdue[n]**2 for n in range(1, 39)}
    w_cold_2 = {n: sec2_overdue[n]**2 for n in range(1, 9)}
    
    pred_cold = generate_numbers(w_cold_1, w_cold_2)
    print(f"\n2. [The Contrarian] (Favoring Cold/Overdue Numbers)")
    print(f"   Section 1: {pred_cold[0]}")
    print(f"   Section 2: {pred_cold[1]}")

    # Model 3: Balanced Mix (Freq * Overdue + small random noise)
    # This tries to find numbers that are generally frequent but currently missing.
    # Add a base weight to ensure everything has a chance.
    w_bal_1 = {}
    for n in range(1, 39):
        score = (sec1_freq[n] * 0.5) + (sec1_overdue[n] * 2) 
        w_bal_1[n] = score

    w_bal_2 = {}
    for n in range(1, 9):
        score = (sec2_freq[n] * 0.5) + (sec2_overdue[n] * 2)
        w_bal_2[n] = score
        
    pred_bal = generate_numbers(w_bal_1, w_bal_2)
    print(f"\n3. [The Balanced Mix] (Weighted Hybrid)")
    print(f"   Section 1: {pred_bal[0]}")
    print(f"   Section 2: {pred_bal[1]}")

if __name__ == "__main__":
    csv_file = "super_lotto638_results.csv"
    data = load_data(csv_file)
    if data:
        analyze_and_predict(data)
