import pandas as pd
import numpy as np
import re
import sys

# Original script path
script_path = "replay_2tickets_soft_p04_n500.py"

# Function to run the actual simulation logic with a given cap
# We'll just define the core logic here to avoid subprocess issues
# BUT re-implementing 250 lines is risky.
# Let's use subprocess but properly.
import subprocess

caps_to_test = [15, 12, 10, 8, 6, 4, 3]
results = []

def update_cap(cap):
    with open(script_path, 'r') as f:
        content = f.read()
    # Replace MAX_TICKETS = ... with new value
    new_content = re.sub(r"MAX_TICKETS = \d+", f"MAX_TICKETS = {cap}", content)
    with open(script_path, 'w') as f:
        f.write(new_content)

print("Starting batch test...")

for cap in caps_to_test:
    print(f"Testing MAX_TICKETS = {cap}...", flush=True)
    update_cap(cap)
    
    # Run the script and capture output
    try:
        res = subprocess.run(["python3", script_path], capture_output=True, text=True, timeout=60) # 60s timeout
        output = res.stdout
        
        # Parse output manually
        # Look for Hit Rate 6_1
        hit_6_1_match = re.search(r"hit_6_1:\s+([\d\.]+)", output)
        hit_6_1 = float(hit_6_1_match.group(1)) if hit_6_1_match else 0.0

        # Look for Net Profit Mean
        profit_mean_match = re.search(r"Net profit \(Real Prizes\):.*?mean\s+([-\d\.]+e?\+?\d*)", output, re.DOTALL)
        profit_mean = float(profit_mean_match.group(1)) if profit_mean_match else 0.0

        # Look for Max Profit
        profit_max_match = re.search(r"Net profit \(Real Prizes\):.*?max\s+([-\d\.]+e?\+?\d*)", output, re.DOTALL)
        profit_max = float(profit_max_match.group(1)) if profit_max_match else 0.0

        # Look for Total Prize Mean (to calc cost)
        total_prize_match = re.search(r"SUMMARY \(Real Prizes.*?mean\s+([-\d\.]+e?\+?\d*)", output, re.DOTALL)
        total_prize = float(total_prize_match.group(1)) if total_prize_match else 0.0
        
        cost_mean = total_prize - profit_mean

        results.append({
            "Cap": cap,
            "Hit_6_1": hit_6_1,
            "Profit_Mean": profit_mean,
            "Profit_Max": profit_max,
            "Cost_Mean": cost_mean
        })
        print(f"  -> Done. Cost: {int(cost_mean)}, Max Profit: {int(profit_max)}")

    except Exception as e:
        print(f"  -> Error: {e}")

print("\n=== BATCH TEST RESULTS ===")
print(f"{'Cap':<5} | {'Cost(Avg)':<12} | {'Hit 6+1':<8} | {'Profit(Avg)':<15} | {'Profit(Max)':<15}")
print("-" * 65)
for r in results:
    print(f"{r['Cap']:<5} | {int(r['Cost_Mean']):<12} | {r['Hit_6_1']:<8.4f} | {int(r['Profit_Mean']):<15} | {int(r['Profit_Max']):<15}")
