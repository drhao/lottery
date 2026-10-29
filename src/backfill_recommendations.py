"""
回補缺漏期別的推薦號碼（供儀表板回測使用）

對每一期開獎（自 START_DATE 起），若上一期的日期不在推薦歷史中，
就把 CSV 截斷到上一期為止，在暫存資料夾中執行原本的推薦腳本，
重現「當時會給出的推薦」，並以 backfilled: true 標記寫回歷史 JSON。
既有的真實推薦紀錄不會被覆蓋。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START_DATE = '2026-02-01'

GAMES = [
    {
        'name': 'Lotto 6/49',
        'csv': 'lotto649_results.csv',
        'script': 'recommend_lotto649.py',
        'history': 'recommendation_lotto649_history.json',
    },
    {
        'name': 'Super Lotto (威力彩)',
        'csv': 'super_lotto638_results.csv',
        'script': 'recommend_next_draw.py',
        'history': 'recommendation_history.json',
    },
]


def run_recommender(game, df_until):
    """在暫存資料夾執行推薦腳本，回傳它寫出的那一筆歷史紀錄。"""
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'src'))
        os.makedirs(os.path.join(tmp, 'data'))
        shutil.copy(os.path.join(BASE_DIR, 'src', game['script']), os.path.join(tmp, 'src'))
        df_until.to_csv(os.path.join(tmp, 'data', game['csv']), index=False)

        subprocess.run([sys.executable, os.path.join('src', game['script'])],
                       cwd=tmp, check=True, capture_output=True)

        with open(os.path.join(tmp, game['history']), encoding='utf-8') as f:
            return json.load(f)[df_until['Date'].iloc[0]]


def backfill(game):
    print(f"\n--- {game['name']} ---")
    history_path = os.path.join(BASE_DIR, game['history'])
    with open(history_path, encoding='utf-8') as f:
        history = json.load(f)

    df = pd.read_csv(os.path.join(BASE_DIR, 'data', game['csv']))  # New -> Old
    dates = df['Date'].tolist()

    added = 0
    for i, target_date in enumerate(dates[:-1]):
        if target_date < START_DATE:
            break
        base_date = dates[i + 1]  # 推薦以上一期為最新資料產生
        if base_date in history:
            continue
        record = run_recommender(game, df.iloc[i + 1:].reset_index(drop=True))
        record['backfilled'] = True
        history[base_date] = record
        added += 1
        print(f"回補 {base_date} -> {target_date}")

    history = dict(sorted(history.items()))
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    print(f"{game['name']}: 新增 {added} 筆回補推薦")


if __name__ == '__main__':
    for game in GAMES:
        backfill(game)
