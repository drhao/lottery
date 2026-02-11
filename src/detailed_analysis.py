import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("replay_result_v3_cap10_10k.csv")

# 1. General Profit/Loss Analysis
total_sims = len(df)
profitable_sims = df[df['net_profit_real'] > 0]
num_profitable = len(profitable_sims)
profit_pct = (num_profitable / total_sims) * 100

print(f"=== 威力彩動態策略 (Cap-10) 一萬次模擬分析報告 ===")
print(f"總模擬次數: {total_sims}")
print(f"獲利次數: {num_profitable} ({profit_pct:.2f}%)")
print(f"虧損次數: {total_sims - num_profitable} ({100 - profit_pct:.2f}%)")

print("\n--- 損益統計 (Net Profit) ---")
print(f"平均損益: {df['net_profit_real'].mean():,.0f}")
print(f"損益中位數: {df['net_profit_real'].median():,.0f}")
print(f"最慘虧損: {df['net_profit_real'].min():,.0f}")
print(f"最高獲利: {df['net_profit_real'].max():,.0f}")

# 2. Hit Rates (Total across all simulations)
hit_cols = [col for col in df.columns if col.startswith('hit_')]
hit_sums = df[hit_cols].sum()

print("\n--- 總命中次數分布 (10,000次平均每輪結果) ---")
for col in hit_cols:
    total_hits = hit_sums[col]
    avg_hits = total_hits / total_sims
    print(f"{col:10}: 總計 {int(total_hits):5} 次 | 平均每千期出現 {avg_hits:.3f} 次")

# 3. Probability of hitting specific major prizes in a 1,000-draw period
print("\n--- 擊中大獎機率 (在一千期投注中至少出現一次的模擬比例) ---")
p_6_1 = (df['hit_6_1'] > 0).mean() * 100
p_6_0 = (df['hit_6_0'] > 0).mean() * 100
p_5_1 = (df['hit_5_1'] > 0).mean() * 100
print(f"頭獎 (6+1): {p_6_1:5.2f}%")
print(f"貳獎 (6+0): {p_6_0:5.2f}%")
print(f"參獎 (5+1): {p_5_1:5.2f}%")

# 4. Risk Analysis
print("\n--- 風險分析 (Max Drawdown) ---")
print(f"平均最大回撤: {df['max_drawdown'].mean():,.0f}")
print(f"極端最大回撤: {df['max_drawdown'].min():,.0f}")

# 5. ROI Analysis (Approximate)
# Each run simulates 1000 draws. Let's estimate total cost per run.
# Since current_n_tickets is dynamic, we need to infer cost.
# net_profit = total_prize - total_cost => total_cost = total_prize - net_profit
df['total_cost'] = df['total_prize_real'] - df['net_profit_real']
avg_cost = df['total_cost'].mean()
avg_prize = df['total_prize_real'].mean()
roi = (df['net_profit_real'].mean() / avg_cost) * 100

print("\n--- 投資報酬率 (ROI) ---")
print(f"平均每輪成本: {avg_cost:,.0f} (約為 {avg_cost/100:.0f} 注)")
print(f"平均投資報酬率: {roi:.2f}%")
