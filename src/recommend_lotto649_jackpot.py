"""
大樂透「頭獎門檻 + 避開熱門號碼」策略

依 strategy_lab_lotto649.py 的回測結論：
- 選號無法提高中獎率，中頭獎機率只取決於注數
- 固定年預算時，集中在頭獎池大的期數買，期望虧損最少、中獎時金額最大
- 選冷門組合可降低與他人平分頭獎的機會

做法：預估下期頭獎池 ≥ THRESHOLD 才建議投注，以「包牌」方式一次買一組號碼的全部 6 碼組合，
包牌大小依年預算決定（預設 8 碼 = 28 注），8 個號碼用熱門度模型挑選最冷門的組合。
"""
import json
import os
from itertools import combinations
from math import comb

import numpy as np

from strategy_lab_lotto649 import fit_popularity, load

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT_OUTPUT_PATH = os.path.join(BASE_DIR, 'recommendation_lotto649_jackpot_output.txt')
HISTORY_PATH = os.path.join(BASE_DIR, 'recommendation_lotto649_jackpot_history.json')

# Strategy Parameters
ANNUAL_BUDGET = 20000
THRESHOLD = 300_000_000
DRAWS_PER_YEAR = 14          # 2010 起頭獎池 ≥3 億平均每年期數
TICKET_COST = 50
N_CAND = 3000


def pack_size(budget=ANNUAL_BUDGET):
    """預算可負擔的最大包牌碼數（C(k,6) 注 ≤ 每期預算）。"""
    per_draw = budget / DRAWS_PER_YEAR / TICKET_COST
    k = 6
    while comb(k + 1, 6) <= per_draw:
        k += 1
    return k


def pick_pack(pop, prev, k, rng, n_cand=N_CAND):
    """挑 k 個號碼，使其全部 6 碼組合的平均熱門度最低。"""
    combos = np.array(list(combinations(range(k), 6)))
    # 偏向冷門號碼抽候選：用單一號碼的熱門度當權重
    single = pop(np.array([[n] * 6 for n in range(1, 50)]), prev)[6] ** (1 / 6)
    w = 1 / single
    keys = -np.log(rng.random((n_cand, 49))) / w
    cands = np.sort(np.argpartition(keys, k - 1, axis=1)[:, :k] + 1, axis=1)
    tickets = cands[:, combos].reshape(-1, 6)
    score = pop(tickets, prev)[6].reshape(n_cand, -1).mean(1)
    best = int(np.argmin(score))
    return cands[best], float(score[best])


def next_jackpot_estimate(df):
    """下期頭獎池預估：本期未開出則滾入，加上本期分配到頭獎的金額。"""
    last = df.iloc[-1]
    rollover = 0 if last.jp_won else last.J
    return rollover + last.Jackpot_Total


def main():
    df, first, sp = load()                       # Old -> New
    pop, _ = fit_popularity(df, first, train_end='9999-12-31')

    last_date = df.Date.iloc[-1]
    period = int(df.Period.iloc[-1])
    jackpot_est = next_jackpot_estimate(df)
    threshold_met = jackpot_est >= THRESHOLD

    k = pack_size()
    rng = np.random.default_rng(period)          # 以期別當 seed，結果可重現
    numbers, pi6 = pick_pack(pop, first[-1], k, rng)
    n_tickets = comb(k, 6)
    cost = n_tickets * TICKET_COST

    nums_str = " ".join(f"{n:02d}" for n in numbers)
    lines = [
        "==========================================",
        "   大樂透 頭獎門檻策略 (≥3億 + 避開熱門)   ",
        "==========================================",
        f"最新開獎日期:   {last_date}",
        f"預估下期頭獎:   ${jackpot_est:,.0f}",
        f"投注門檻:       ${THRESHOLD:,.0f}",
        f"年度預算:       ${ANNUAL_BUDGET:,}（預期每年約 {DRAWS_PER_YEAR} 期達標）",
        "------------------------------------------",
    ]
    if threshold_met:
        lines.append(f"建議行動:       ✅ 已達門檻，包牌 {k} 碼 = {n_tickets} 注 (${cost:,})")
    else:
        lines.append("建議行動:       ⛔ 未達門檻，本期不買（預算留給頭獎更大的期數）")
        lines.append(f"備註:           下方為參考號碼，達標時以包牌 {k} 碼投注")
    lines += [
        f"包牌號碼: [{nums_str}]",
        f"平均熱門度:     {pi6:.2f}（隨機選號 = 1.00，越低越不容易與人平分）",
        "==========================================",
        "祝您中大獎！",
    ]
    text = "\n".join(lines)
    print(text)
    with open(TXT_OUTPUT_PATH, "w", encoding='utf-8') as f:
        f.write(text)

    history = {}
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            pass
    history[last_date] = {
        "jackpot_est": float(jackpot_est),
        "threshold": THRESHOLD,
        "threshold_met": bool(threshold_met),
        "annual_budget": ANNUAL_BUDGET,
        "pack_size": int(k),
        "recommended_tickets": int(n_tickets) if threshold_met else 0,
        "cost": int(cost) if threshold_met else 0,
        "numbers": [int(n) for n in numbers],
        "popularity": round(pi6, 3),
    }
    with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    print(f"結果已儲存至 '{os.path.basename(TXT_OUTPUT_PATH)}' 與 '{os.path.basename(HISTORY_PATH)}'")


if __name__ == "__main__":
    main()
