"""
大樂透 50 策略實驗室

衡量兩件事：
1. 選號命中：2010 起每期每策略 20 注，實際對獎（固定獎項回收、平均中號數），
   與理論值比較，看選號方法能不能提高命中。
2. 期望值：2019 起（熱門度模型只用 2019 前資料訓練），每注依該期實際頭獎池、
   銷售量與「這組號碼被多少人買」估算稅後期望值（含分獎），除以成本 $50。

熱門度模型：用各期 普獎/伍獎/參獎 實際中獎人數 ÷ 理論人數，回歸開出號碼的特徵，
推估任一組號碼的相對熱門度 π（隨機號碼平均 = 1），再外推到 6 碼（頭獎）。
"""
import os
from math import comb

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'lotto649_results.csv')
OUT_PATH = os.path.join(BASE_DIR, 'output', 'lotto649_strategy_lab.csv')

TICKET_COST = 50
TAX = 0.8                 # 單注獎金超過 5,000 課 20%
K = 20                    # 每期每策略注數（選號類）
SIM_START = '2010-01-01'
TEST_START = '2019-01-01' # 熱門度模型訓練/測試切點
N_CAND = 3000             # 熱門度篩選候選數

T49 = comb(49, 6)


def p_tier(m, b):
    base = comb(6, m) * comb(43, 6 - m) / T49
    q = (6 - m) / 43
    return base * (q if b else 1 - q)


P = {k: p_tier(*k) for k in [(5, 1), (5, 0), (4, 1), (4, 0), (3, 1), (2, 1), (3, 0)]}
P6 = 1 / T49


# =====================
# 資料
# =====================
def load():
    df = pd.read_csv(CSV_PATH).iloc[::-1].reset_index(drop=True)
    first = np.array([list(map(int, s.split())) for s in df.Numbers])
    sp = df.Special_Number.to_numpy()
    fixed = df.Fifth_Prize_Total + df.Sixth_Prize_Total + df.Seventh_Prize_Total + df.Normal_Prize_Total
    flo = df.Jackpot_Total + df.Second_Prize_Total + df.Third_Prize_Total + df.Fourth_Prize_Total
    df['S'] = (flo + fixed) / 0.55 / TICKET_COST          # 估計售出注數
    df['J'] = df.Jackpot_Total + df.Jackpot_Rollover       # 本期頭獎池
    df['J_est'] = df.Jackpot_Rollover + df.Jackpot_Total.shift(1).fillna(df.Jackpot_Total)  # 開獎前可得的估計
    df['S_prev'] = df.S.shift(1).fillna(df.S)
    df['jp_won'] = df.Jackpot_Per_Winner > 0
    return df, first, sp


# =====================
# 熱門度模型
# =====================
def features(T, prev):
    """T: (n,6) 1-based；prev: 上一期 6 個號碼。"""
    n = len(T)
    X = np.zeros((n, 49 + 5))
    X[np.repeat(np.arange(n), 6), T.ravel() - 1] = 1
    s = np.sort(T, axis=1)
    d = np.diff(s, axis=1)
    X[:, 49] = (d == 1).sum(1)                                  # 連號對數
    X[:, 50] = (d == d[:, :1]).all(1)                           # 等差
    last = s % 10
    X[:, 51] = np.array([np.bincount(r, minlength=10).max() for r in last])  # 同尾數最多幾個
    X[:, 52] = np.isin(T, prev).sum(1)                          # 與上期重複
    X[:, 53] = np.array([np.bincount(r, minlength=5).max() for r in (s // 10)])  # 同十位段最多幾個
    return X


def fit_popularity(df, first, train_end=TEST_START):
    train = [t for t in range(1, len(df)) if df.Date[t] < train_end]
    X = features(first[train], first[0])  # 「與上期重複」欄位下面逐列修正
    for i, t in enumerate(train):
        X[i, 52] = np.isin(first[t], first[t - 1]).sum()
    Xc = np.hstack([np.ones((len(train), 1)), X])

    tiers = {
        3: ('Normal_Prize_Total', 'Normal_Prize_Per_Winner', P[(3, 0)]),
        4: ('Fifth_Prize_Total', 'Fifth_Prize_Per_Winner', P[(4, 0)]),
        5: ('Third_Prize_Total', 'Third_Prize_Per_Winner', P[(5, 0)]),
    }
    betas, ys = {}, {}
    for k, (tot, per, p) in tiers.items():
        w = (df[tot] / df[per].replace(0, np.nan)).fillna(0).to_numpy()
        E = (df.S * p).to_numpy()
        y = np.log((w + 0.5) / (E + 0.5))
        wt = np.minimum(E, 500)[train]
        A = Xc * np.sqrt(wt)[:, None]
        ridge = np.eye(Xc.shape[1]) * 2.0
        ridge[0, 0] = 0
        betas[k] = np.linalg.solve(A.T @ A + ridge, A.T @ (y[train] * np.sqrt(wt)))
        ys[k] = y
    # 外推到 6 碼：各層係數的放大倍率
    g = np.polyfit(betas[4][1:], betas[5][1:], 1)[0]
    betas[6] = np.r_[0, betas[5][1:] * g]
    for k in (4, 5, 6):
        betas[k][0] = 0

    # 正規化：均勻隨機號碼的平均 π = 1
    rng = np.random.default_rng(1)
    U = np.array([rng.choice(49, 6, replace=False) + 1 for _ in range(20000)])
    XU = features(U, first[-1])
    norm = {k: np.log(np.mean(np.exp(XU @ betas[k][1:]))) for k in (4, 5, 6)}

    # 測試期驗證
    test = [t for t in range(1, len(df)) if df.Date[t] >= TEST_START]
    Xt = features(first[test], first[0])
    for i, t in enumerate(test):
        Xt[i, 52] = np.isin(first[t], first[t - 1]).sum()
    r = {k: np.corrcoef(Xt @ betas[k][1:] if k > 3 else Xt @ betas[3][1:], ys[k][test])[0, 1] for k in (3, 4, 5)}
    info = {'growth_g': g, 'test_corr': r}

    def pop(T, prev):
        X = features(T, prev)
        return {k: np.exp(X @ betas[k][1:] - norm[k]) for k in (4, 5, 6)}
    return pop, info


# =====================
# 期望值
# =====================
def share(pool, lam):
    lam = np.maximum(lam, 1e-12)
    return pool * (1 - np.exp(-lam)) / lam


def ticket_ev(row, pi):
    S = row.S
    ev = (P[(4, 0)] * row.Fifth_Prize_Per_Winner + P[(3, 1)] * row.Sixth_Prize_Per_Winner
          + P[(2, 1)] * row.Seventh_Prize_Per_Winner + P[(3, 0)] * row.Normal_Prize_Per_Winner)
    ev = ev + TAX * (
        P6 * share(row.J, S * P6 * pi[6])
        + P[(5, 1)] * share(row.Second_Prize_Total, S * P[(5, 1)] * pi[5])
        + P[(5, 0)] * share(row.Third_Prize_Total, S * P[(5, 0)] * pi[5])
        + P[(4, 1)] * share(row.Fourth_Prize_Total, S * P[(4, 1)] * pi[4])
    )
    return ev


def ticket_ev_est(row, pi6=1.0):
    """開獎前可得資訊估計的期望值（給時機策略做決策用）。"""
    fixed = 14.58
    sec = 0.65 + 0.79 + 0.51
    return fixed + sec + TAX * P6 * share(row.J_est, row.S_prev * P6 * pi6)


def realized(T, draw, sp, row):
    m = np.isin(T, draw).sum(1)
    b = (T == sp).any(1)
    prize = np.zeros(len(T))
    prize[(m == 4) & ~b] = row.Fifth_Prize_Per_Winner
    prize[(m == 3) & b] = row.Sixth_Prize_Per_Winner
    prize[(m == 2) & b] = row.Seventh_Prize_Per_Winner
    prize[(m == 3) & ~b] = row.Normal_Prize_Per_Winner
    return m, prize


# =====================
# 抽樣工具
# =====================
def es(rng, w, n, k=6):
    w = np.clip(np.asarray(w, float), 1e-12, None)
    keys = -np.log(rng.random((n, 49))) / w
    return np.argpartition(keys, k - 1, axis=1)[:, :k] + 1


def uniform(rng, n):
    return es(rng, np.ones(49), n)


def filtered(rng, n, cond):
    out = []
    while sum(len(o) for o in out) < n:
        C = uniform(rng, 4000)
        out.append(C[cond(C)])
    return np.vstack(out)[:n]


def by_pop(rng, n, ctx, lowest=True, pool_frac=None):
    C = uniform(rng, N_CAND)
    pi6 = ctx['pop'](C, ctx['prev'])[6]
    order = np.argsort(pi6 if lowest else -pi6)
    if pool_frac:
        idx = order[:int(N_CAND * pool_frac)]
        return C[rng.choice(idx, n, replace=False)]
    return C[order[:n]]


def cover(rng, n, base_fn):
    """盡量不重複號碼：從候選中貪婪挑選與已選號碼重疊最少者。"""
    C = base_fn(n * 10)
    chosen, used = [], np.zeros(50)
    for _ in range(n):
        score = used[C].sum(1)
        i = int(np.argmin(score))
        chosen.append(C[i]); used[C[i]] += 1
        C = np.delete(C, i, axis=0)
    return np.array(chosen)


# =====================
# 50 個策略
# =====================
def hot(ctx, win, alpha=0.0):
    t = ctx['t']
    c = ctx['cum'][t] - ctx['cum'][max(0, t - win)]
    return c + alpha


def strategies():
    S = []
    add = lambda sid, cat, name, fn, **kw: S.append(dict(id=sid, cat=cat, name=name, fn=fn, **kw))

    # A. 選號技巧
    add('A01', '選號', '純隨機（基準）', lambda r, c: uniform(r, K))
    add('A02', '選號', '現行程式（熱100+α / 全期冷門）',
        lambda r, c: np.vstack([es(r, hot(c, 100, 100), K // 2), es(r, 1 / (c['cum'][c['t']] + 1), K // 2)]))
    for sid, w in [('A03', 10), ('A04', 30), ('A05', 50), ('A06', 100), ('A07', 300)]:
        add(sid, '選號', f'熱門 近{w}期（純次數權重）', lambda r, c, w=w: es(r, hot(c, w) + 0.5, K))
    for sid, w in [('A08', 30), ('A09', 100)]:
        add(sid, '選號', f'冷門 近{w}期', lambda r, c, w=w: es(r, 1 / (hot(c, w) + 0.5), K))
    add('A10', '選號', '冷門 全期（平方強化）', lambda r, c: es(r, 1 / (c['cum'][c['t']] + 1) ** 2 * 1e6, K))
    add('A11', '選號', '遺漏最久（追 overdue）', lambda r, c: es(r, c['gap'] + 1, K))
    add('A12', '選號', '剛開出（遺漏最短）', lambda r, c: es(r, 1 / (c['gap'] + 1), K))
    add('A13', '選號', '固定保留上期 2 號',
        lambda r, c: np.array([np.r_[r.choice(c['prev'], 2, replace=False),
                                     r.choice(np.setdiff1d(np.arange(1, 50), c['prev']), 4, replace=False)] for _ in range(K)]))
    add('A14', '選號', '排除上期號碼', lambda r, c: filtered(r, K, lambda C: ~np.isin(C, c['prev']).any(1)))
    add('A15', '選號', '奇偶 3:3', lambda r, c: filtered(r, K, lambda C: (C % 2).sum(1) == 3))
    add('A16', '選號', '大小 3:3（1-24 / 25-49）', lambda r, c: filtered(r, K, lambda C: (C <= 24).sum(1) == 3))
    add('A17', '選號', '和值 125–175', lambda r, c: filtered(r, K, lambda C: (C.sum(1) >= 125) & (C.sum(1) <= 175)))
    add('A18', '選號', '五個十位段都有', lambda r, c: filtered(r, K, lambda C: np.array([len(set(x // 10)) == 5 for x in C])))
    add('A19', '選號', '至少一組連號',
        lambda r, c: filtered(r, K, lambda C: (np.diff(np.sort(C, 1), axis=1) == 1).any(1)))
    add('A20', '選號', '不含連號',
        lambda r, c: filtered(r, K, lambda C: ~(np.diff(np.sort(C, 1), axis=1) == 1).any(1)))
    add('A21', '選號', '尾數全不同', lambda r, c: filtered(r, K, lambda C: np.array([len(set(x % 10)) == 6 for x in C])))
    add('A22', '選號', '共現配對最強（號碼對歷史同開次數）', lambda r, c: pair_pick(r, c))
    add('A23', '選號', '馬可夫轉移（上期號碼→下期）', lambda r, c: es(r, c['markov'][c['prev'] - 1].sum(0) + 1, K))
    add('A24', '選號', '特別號熱門加權', lambda r, c: es(r, c['cum_sp'][c['t']] + 1, K))
    add('A25', '選號', '固定一組號碼每期重買', lambda r, c: np.tile(c['fixed_set'], (K, 1)))
    add('A26', '選號', '均值回歸（近200期低於期望者）',
        lambda r, c: es(r, np.clip(200 * 7 / 49 - hot(c, 200), 0.1, None), K))

    # B. 反熱門（降低分獎）
    add('B27', '反熱門', '全部 >31（避開生日號）', lambda r, c: filtered(r, K, lambda C: (C > 31).all(1)))
    add('B28', '反熱門', '至少 4 個 >31', lambda r, c: filtered(r, K, lambda C: (C > 31).sum(1) >= 4))
    add('B29', '反熱門', '熱門度模型最低', lambda r, c: by_pop(r, K, c, lowest=True))
    add('B30', '反熱門', '熱門度最低 + 覆蓋不重複', lambda r, c: cover(r, K, lambda n: by_pop(r, n, c, lowest=True)))
    add('B31', '反熱門', '避開生日/連號/等差/上期',
        lambda r, c: filtered(r, K, lambda C: ((C > 31).sum(1) >= 3) & ~(np.diff(np.sort(C, 1), axis=1) == 1).any(1)
                              & ~np.isin(C, c['prev']).any(1)))
    add('B32', '反熱門', '熱門度模型最高（對照）', lambda r, c: by_pop(r, K, c, lowest=False))
    add('B33', '反熱門', '全部 ≤31（對照：生日號）', lambda r, c: filtered(r, K, lambda C: (C <= 31).all(1)))
    add('B34', '反熱門', '含 3 連號以上（對照：圖形號）',
        lambda r, c: filtered(r, K, lambda C: np.array([np.max(np.convolve(np.diff(np.sort(x)) == 1, [1, 1], 'valid')) == 2 for x in C])))
    add('B35', '反熱門', '熱門度後 20% 中隨機', lambda r, c: by_pop(r, K, c, lowest=True, pool_frac=0.2))
    add('B36', '反熱門', '現行程式 + 反熱門過濾',
        lambda r, c: anti_filter(r, c, lambda n: np.vstack([es(r, hot(c, 100, 100), n // 2),
                                                            es(r, 1 / (c['cum'][c['t']] + 1), n - n // 2)])))

    # C. 時機（隨機選號，只改何時買）
    add('C37', '時機', '每期都買（基準）', lambda r, c: uniform(r, K))
    for sid, th in [('C38', 1), ('C39', 2), ('C40', 3), ('C41', 4), ('C42', 5), ('C43', 6), ('C44', 8)]:
        add(sid, '時機', f'頭獎池 ≥ {th} 億才買',
            lambda r, c, th=th: uniform(r, K) if c['row'].J_est >= th * 1e8 else None)
    add('C45', '時機', '連槓 ≥ 10 期才買', lambda r, c: uniform(r, K) if c['roll'] >= 10 else None)
    add('C46', '時機', '頭獎池達近一年前 10% 才買',
        lambda r, c: uniform(r, K) if c['row'].J_est >= c['j_q90'] else None)

    # D. 注數
    add('D47', '注數', '現行注數 2+連槓（上限 15）', lambda r, c: uniform(r, min(2 + c['roll'], 15)), var_k=True)
    add('D48', '注數', '固定每期 2 注', lambda r, c: uniform(r, 2), var_k=True)
    add('D49', '注數', '依期望值加碼（EV 比 ×10 注）',
        lambda r, c: uniform(r, max(0, int(ticket_ev_est(c['row']) / TICKET_COST * 10) - 3)), var_k=True)
    add('D50', '綜合', '頭獎池 ≥ 5 億 + 熱門度最低選號',
        lambda r, c: by_pop(r, K, c, lowest=True) if c['row'].J_est >= 5e8 else None)
    return S


def pair_pick(rng, ctx):
    M = ctx['pair']
    out = []
    for _ in range(K):
        first_n = rng.integers(1, 50)
        pick = [first_n]
        while len(pick) < 6:
            w = M[np.array(pick) - 1].sum(0) + 1
            w[np.array(pick) - 1] = 0
            pick.append(int(es(rng, w, 1, k=1)[0, 0]))
        out.append(pick)
    return np.array(out)


def anti_filter(rng, ctx, base_fn):
    C = base_fn(N_CAND)
    pi6 = ctx['pop'](C, ctx['prev'])[6]
    return C[np.argsort(pi6)[:K]]


# =====================
# 主程式
# =====================
def main():
    df, first, sp = load()
    pop, info = fit_popularity(df, first)
    print(f"熱門度模型：6 碼外推倍率 g={info['growth_g']:.2f}；測試期相關係數 " +
          ", ".join(f"{k}碼 {v:.2f}" for k, v in info['test_corr'].items()))

    N = len(df)
    cum = np.zeros((N + 1, 50))
    cum_sp = np.zeros((N + 1, 50))
    for t in range(N):
        cum[t + 1] = cum[t]; cum[t + 1, first[t]] += 1
        cum_sp[t + 1] = cum_sp[t]; cum_sp[t + 1, sp[t]] += 1
    cum, cum_sp = cum[:, 1:], cum_sp[:, 1:]

    strats = strategies()
    rng = np.random.default_rng(2026)
    fixed_set = np.sort(rng.choice(49, 6, replace=False) + 1)
    stats = {s['id']: dict(tickets=0, draws=0, cost=0.0, m=[], fixed=[], ev=0.0, ev_cost=0.0,
                           ev_rand=0.0, real_prize=0.0, pi6=[]) for s in strats}

    gap = np.zeros(49)
    pair = np.zeros((49, 49))
    markov = np.zeros((49, 49))
    roll = 0
    start = df.index[df.Date >= SIM_START][0]

    for t in range(N):
        row = df.iloc[t]
        if t >= start:
            ctx = dict(t=t, cum=cum, cum_sp=cum_sp, gap=gap, pair=pair, markov=markov, prev=first[t - 1],
                       pop=pop, row=row, roll=roll, fixed_set=fixed_set,
                       j_q90=df.J_est.iloc[max(0, t - 104):t].quantile(0.9))
            in_test = row.Date >= TEST_START
            for s in strats:
                T = s['fn'](rng, ctx)
                if T is None or len(T) == 0:
                    continue
                T = np.asarray(T, int)
                st = stats[s['id']]
                m, prize = realized(T, first[t], sp[t], row)
                st['tickets'] += len(T); st['draws'] += 1; st['cost'] += len(T) * TICKET_COST
                st['m'].extend(m); st['fixed'].extend(prize)
                if in_test:
                    pi = pop(T, first[t - 1])
                    ev = ticket_ev(row, pi)
                    st['ev'] += ev.sum(); st['ev_cost'] += len(T) * TICKET_COST
                    st['ev_rand'] += len(T) * ticket_ev(row, {4: 1.0, 5: 1.0, 6: 1.0})
                    st['pi6'].extend(pi[6])
        # 更新狀態（只用到本期為止的資訊）
        gap += 1; gap[first[t] - 1] = 0
        idx = first[t] - 1
        pair[np.ix_(idx, idx)] += 1
        if t > 0:
            markov[np.ix_(first[t - 1] - 1, idx)] += 1
        roll = 0 if row.jp_won else roll + 1

    rows = []
    for s in strats:
        st = stats[s['id']]
        m = np.array(st['m']); f = np.array(st['fixed'])
        rows.append(dict(
            id=s['id'], 類別=s['cat'], 策略=s['name'], 期數=st['draws'], 注數=st['tickets'],
            平均中號=m.mean() if len(m) else np.nan,
            固定獎回收每注=f.mean() if len(f) else np.nan,
            固定獎SE=f.std() / np.sqrt(len(f)) if len(f) else np.nan,
            測試期注數=int(st['ev_cost'] / TICKET_COST),
            期望值比=st['ev'] / st['ev_cost'] if st['ev_cost'] else np.nan,
            同期隨機期望值比=st['ev_rand'] / st['ev_cost'] if st['ev_cost'] else np.nan,
            平均熱門度π6=np.mean(st['pi6']) if st['pi6'] else np.nan,
        ))
    out = pd.DataFrame(rows)
    theo = (P[(4, 0)] * 2000 + P[(3, 1)] * 1000 + P[(2, 1)] * 400 + P[(3, 0)] * 400)
    out['固定獎z'] = (out.固定獎回收每注 - out.loc[out.id == 'A01', '固定獎回收每注'].iloc[0]) / np.sqrt(
        out.固定獎SE ** 2 + out.loc[out.id == 'A01', '固定獎SE'].iloc[0] ** 2)
    out['分獎效益'] = out.期望值比 - out.同期隨機期望值比
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 60)
    print(f"理論平均中號 {36/49:.4f}；固定獎理論回收（2014 後含柒獎）約 {theo:.2f}")
    print(out.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
