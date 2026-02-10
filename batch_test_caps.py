import pandas as pd
import numpy as np
import subprocess
import re

caps_to_test = [15, 12, 10, 8, 6, 4, 3]
results = []

file_path = "replay_2tickets_soft_p04_n500.py"

print("Starting batch test...")

for cap in caps_to_test:
    print(f"Testing MAX_TICKETS = {cap}...")
    
    # Modify the script using sed (Mac specific sed requires empty extension for -i)
    subprocess.run(["sed", "-i", "", f"s/MAX_TICKETS = .*/MAX_TICKETS = {cap}/", file_path])
    
    # Run the script
    # We need to capture the output to parse hit rates and profits
    result = subprocess.run(["python3", file_path], capture_output=True, text=True)
    output = result.stdout
    
    # Parse output manually because we overwrite the csv each time
    # We want: 6_1 hit rate, Mean Net Profit, Max Profit
    
    # Simple regex parsing
    try:
        # P.S. The output format in python script:
        # hit_6_1: 0.000
        hit_6_1 = float(re.search(r"hit_6_1:\s+([\d\.]+)", output).group(1))
        
        # Net profit mean
        # Name: net_profit_real, dtype: float64
        # mean    -831926.500000
        net_profit_mean = float(re.search(r"Net profit \(Real Prizes\):.*?mean\s+([-\d\.]+)", output, re.DOTALL).group(1))
        
        # Max profit
        # max     -534000.000000
        net_profit_max = float(re.search(r"Net profit \(Real Prizes\):.*?max\s+([-\d\.]+)", output, re.DOTALL).group(1))

        # Mean Cost estimate (approx)
        total_prize_mean = float(re.search(r"SUMMARY \(Real Prizes.*?mean\s+([-\d\.]+)", output, re.DOTALL).group(1))
        cost_mean = total_prize_mean - net_profit_mean

        results.append({
            "Cap": cap,
            "Hit_6_1": hit_6_1,
            "Profit_Mean": net_profit_mean,
            "Profit_Max": net_profit_max,
            "Cost_Mean": cost_mean
        })
    except Exception as e:
        print(f"Error parsing result for cap {cap}: {e}")

print("\n=== BATCH TEST RESULTS ===")
print(f"{'Cap':<5} | {'Cost(Avg)':<10} | {'Hit 6+1':<8} | {'Profit(Avg)':<12} | {'Profit(Max)':<12}")
for r in results:
    print(f"{r['Cap']:<5} | {int(r['Cost_Mean']):<10} | {r['Hit_6_1']:<8.4f} | {int(r['Profit_Mean']):<12} | {int(r['Profit_Max']):<12}")
