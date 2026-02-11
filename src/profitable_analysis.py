import pandas as pd
import ast

# Load the data
df = pd.read_csv("replay_result_v3_cap10_10k.csv")

# Filter for profitable simulations
profitable = df[df['net_profit_real'] > 0].copy()

print(f"=== 正回報模擬分析報告 (共 {len(profitable)} 筆) ===")

if len(profitable) == 0:
    print("沒有任何模擬達到正回報。")
else:
    # 1. Source of Profit
    hit_jackpot = profitable[profitable['hit_6_1'] > 0]
    hit_second = profitable[(profitable['hit_6_1'] == 0) & (profitable['hit_6_0'] > 0)]
    hit_other = profitable[(profitable['hit_6_1'] == 0) & (profitable['hit_6_0'] == 0)]

    print(f"\n--- 獲利來源分析 ---")
    print(f"核心動力: 擊中頭獎 (6+1): {len(hit_jackpot)} 筆")
    print(f"核心動力: 擊中貳獎 (6+0): {len(hit_second)} 筆")
    print(f"其他(依靠大量小獎/參獎): {len(hit_other)} 筆")

    # 2. Statistics of Profitable Runs
    print(f"\n--- 獲利模擬的損益統計 ---")
    print(f"平均獲利: {profitable['net_profit_real'].mean():,.0f}")
    print(f"獲利中位數: {profitable['net_profit_real'].median():,.0f}")
    print(f"最低獲利門檻: {profitable['net_profit_real'].min():,.0f}")
    
    # 3. Hit Distribution for Profitable Runs
    hit_cols = ['hit_6_1', 'hit_6_0', 'hit_5_1', 'hit_5_0', 'hit_4_1', 'hit_4_0']
    print(f"\n--- 獲利組的 key 命中次數 (平均值) ---")
    avg_hits_prof = profitable[hit_cols].mean()
    avg_hits_all = df[hit_cols].mean()
    
    for col in hit_cols:
        print(f"{col:10}: 獲利組平均 {avg_hits_prof[col]:.3f} 次 | 全體平均 {avg_hits_all[col]:.3f} 次")

    # 4. Detailed look at the "Other" category if it exists
    if len(hit_other) > 0:
        print(f"\n--- 靠小獎致富的奇蹟案例 (無頭二獎但獲利) ---")
        for idx, row in hit_other.iterrows():
            print(f"ID {idx}: 淨利 {row['net_profit_real']:,.0f} | 命中: 5+1:{row['hit_5_1']}, 5+0:{row['hit_5_0']}, 4+1:{row['hit_4_1']}")

    # 5. Timing Analysis (When did they win?)
    print(f"\n--- 中獎時機點分析 (僅限擊中頭獎者) ---")
    all_timings = []
    for _, row in hit_jackpot.iterrows():
        # jackpot_hits is stored as a string representation of a list of dicts
        try:
            hits = ast.literal_eval(row['jackpot_hits'])
            for h in hits:
                all_timings.append(h['draw_idx'])
        except:
            continue
            
    if all_timings:
        print(f"頭獎出現在期數: {sorted(all_timings)}")
        print(f"這代表獲利往往來自於這特定的幾次機運。")

