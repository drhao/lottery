import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('replay_result_n500.csv')

# Calculate basic statistics
total_prizes = df['total_prize_fixed']
net_profits = df['net_profit_fixed']
# Assuming fixed cost or just derived from net_profit_fixed
# net_profit = total_prize - cost
# cost = total_prize - net_profit
costs = total_prizes - net_profits

print("=== Analysis of replay_result_n500.csv (Updated ALPHA=300) ===")
print(f"Number of simulations: {len(df)}")
print(f"Average Total Prize: {total_prizes.mean():.2f}")
print(f"Average Net Profit: {net_profits.mean():.2f}")
print(f"Average Cost per run: {costs.mean():.2f}")

# Calculate ROI
roi = (net_profits.sum() / costs.sum()) * 100
print(f"Overall ROI: {roi:.2f}%")

# Profit/Loss distribution
profitable_runs = (net_profits > 0).sum()
print(f"Profitable Runs: {profitable_runs} ({profitable_runs/len(df)*100:.2f}%)")
print(f"Max Profit: {net_profits.max()}")
print(f"Min Profit (Max Loss): {net_profits.min()}")

# Prize tier hits analysis
prize_tiers = [col for col in df.columns if col.startswith('hit_')]
# Order if possible: 6_1, 6_0, 5_1, 5_0, 4_1, 4_0, 3_1, 3_0...
ordered_tiers = ['hit_6_1', 'hit_6_0', 'hit_5_1', 'hit_5_0', 
                 'hit_4_1', 'hit_4_0', 'hit_3_1', 'hit_3_0', 
                 'hit_2_1', 'hit_1_1']
ordered_tiers = [t for t in ordered_tiers if t in prize_tiers]

print("\n=== Prize Tier Hits (Average per run) ===")
for tier in ordered_tiers:
    hits = df[tier].mean()
    prob = (df[tier] > 0).mean() * 100
    print(f"{tier}: Avg Hits = {hits:.4f}, Hit Probability = {prob:.2f}%")
