import csv
import json
import random
from collections import Counter
from datetime import datetime

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

def analyze_data(data):
    # data is Newest -> Oldest in CSV usually.
    # Let's verify by sorting or trusting the scraper. 
    # Based on previous output: Line 2 is 2026-02-05, Line 1885 is 2008.
    # So data[0] is Newest.
    
    sec1_history = []
    sec2_history = []
    sec1_last_seen = {}
    sec2_last_seen = {}
    total_draws = len(data)
    
    # Process from newest to oldest for basic stats
    for idx, row in enumerate(data):
        try:
            nums = list(map(int, row["Numbers"].split()))
            special = int(row["Special_Number"])
        except ValueError:
            continue
            
        sec1_history.extend(nums)
        sec2_history.append(special)
        
        for n in nums:
            if n not in sec1_last_seen:
                sec1_last_seen[n] = idx
        if special not in sec2_last_seen:
            sec2_last_seen[special] = idx
            
    # Frequency
    sec1_freq = Counter(sec1_history)
    sec2_freq = Counter(sec2_history)
    
    # Overdue
    sec1_overdue = {n: sec1_last_seen.get(n, total_draws) for n in range(1, 39)}
    sec2_overdue = {n: sec2_last_seen.get(n, total_draws) for n in range(1, 9)}
    
    return {
        "total_draws": total_draws,
        "sec1_freq": dict(sec1_freq),
        "sec2_freq": dict(sec2_freq),
        "sec1_overdue": sec1_overdue,
        "sec2_overdue": sec2_overdue,
        "last_draw_date": data[0]["Date"] if data else "N/A"
    }

def perform_backtest(data, start_index=500):
    # Data is Newest first data[0]. Oldest is data[-1].
    # We want to iterate chronologically.
    # chron_data[0] = Oldest (2008)
    chron_data = list(reversed(data))
    
    results = []
    
    # We need at least start_index history to begin
    if len(chron_data) <= start_index:
        return []

    # Iterate from draw 500 to the end
    for i in range(start_index, len(chron_data)):
        current_draw = chron_data[i]
        
        # Training data: 0 to i-1
        history = chron_data[:i]
        
        # Calculate Frequency Ranks based on history
        hist_nums = []
        for row in history:
            try:
                hist_nums.extend(list(map(int, row["Numbers"].split())))
            except ValueError:
                continue
                
        freq = Counter(hist_nums)
        
        # Rank: 1 = Most Common. 
        # Sort numbers 1-38 by frequency descending
        ranked_nums = sorted(range(1, 39), key=lambda x: freq[x], reverse=True)
        
        # Create a rank map: number -> rank (1-38)
        rank_map = {num: rank+1 for rank, num in enumerate(ranked_nums)}
        
        # Analyze outcome
        try:
            winning_nums = list(map(int, current_draw["Numbers"].split()))
        except ValueError:
            continue
            
        ranks = [rank_map.get(n, 38) for n in winning_nums]
        avg_rank = sum(ranks) / len(ranks)
        
        results.append({
            "period": current_draw["Period"],
            "date": current_draw["Date"],
            "avg_rank": round(avg_rank, 2),
            "ranks": ranks,
            "winning_nums": winning_nums
        })
        
    return results

def generate_predictions(analysis_data):
    sec1_freq = analysis_data["sec1_freq"]
    sec2_freq = analysis_data["sec2_freq"]
    sec1_overdue = analysis_data["sec1_overdue"]
    sec2_overdue = analysis_data["sec2_overdue"]
    
    def get_prediction(w1, w2):
        candidates_1 = list(range(1, 39))
        weight1 = [w1.get(n, 0) for n in candidates_1]
        picks_1 = set()
        while len(picks_1) < 6:
            p = random.choices(candidates_1, weights=weight1, k=1)[0]
            picks_1.add(p)
        
        candidates_2 = list(range(1, 9))
        weight2 = [w2.get(n, 0) for n in candidates_2]
        pick_2 = random.choices(candidates_2, weights=weight2, k=1)[0]
        
        return sorted(list(picks_1)), pick_2

    # 1. Trend (Freq)
    p1_nums, p1_sec2 = get_prediction(sec1_freq, sec2_freq)
    
    # 2. Contrarian (Overdue^2)
    w_cold_1 = {n: (sec1_overdue.get(n, 0)**2) for n in range(1, 39)}
    w_cold_2 = {n: (sec2_overdue.get(n, 0)**2) for n in range(1, 9)}
    p2_nums, p2_sec2 = get_prediction(w_cold_1, w_cold_2)
    
    # 3. Balanced
    w_bal_1 = {n: (sec1_freq.get(n,0)*0.5 + sec1_overdue.get(n,0)*2) for n in range(1, 39)}
    w_bal_2 = {n: (sec2_freq.get(n,0)*0.5 + sec2_overdue.get(n,0)*2) for n in range(1, 9)}
    p3_nums, p3_sec2 = get_prediction(w_bal_1, w_bal_2)
    
    return {
        "trend": {"nums": p1_nums, "sec2": p1_sec2},
        "contrarian": {"nums": p2_nums, "sec2": p2_sec2},
        "balanced": {"nums": p3_nums, "sec2": p3_sec2}
    }

def generate_html(analysis, predictions, backtest_results):
    js_data = json.dumps({
        "analysis": analysis,
        "predictions": predictions,
        "backtest": backtest_results
    })
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>威力彩數據分析 Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-hot: #f43f5e;
            --accent-cold: #3b82f6;
            --accent-mix: #10b981;
            --grid-color: #334155;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        
        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(to right, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .nav-btn {{
            display: inline-block;
            margin-top: 10px;
            padding: 8px 16px;
            background: rgba(255,255,255,0.1);
            color: var(--text-primary);
            text-decoration: none;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.2s;
        }}
        .nav-btn:hover {{
            background: rgba(255,255,255,0.2);
            transform: translateY(-2px);
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .card {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        
        .card h2 {{
            font-size: 1.25rem;
            margin-top: 0;
            margin-bottom: 20px;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--grid-color);
            padding-bottom: 10px;
        }}
        
        .prediction-box {{
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid transparent;
        }}
        
        .pred-trend {{ border-left-color: var(--accent-hot); }}
        .pred-cold {{ border-left-color: var(--accent-cold); }}
        .pred-mix {{ border-left-color: var(--accent-mix); }}
        
        .balls {{
            display: flex;
            gap: 8px;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        
        .ball {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9rem;
            color: white;
        }}
        
        .ball.sec1 {{ background: linear-gradient(135deg, #6366f1, #8b5cf6); }}
        .ball.sec2 {{ background: linear-gradient(135deg, #ef4444, #f97316); }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid var(--grid-color); }}
        th {{ color: var(--text-secondary); font-weight: 600; }}
        .rank-1 {{ color: #fbbf24; font-weight: bold; }}
        
        canvas {{ width: 100% !important; height: 300px !important; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔮 威力彩趨勢分析 (主儀表板)</h1>
            <p style="color: var(--text-secondary)">
                統計區間: 2008-01 至 <span id="last-date"></span> | 總期數: <span id="total-draws"></span>
            </p>
            <a href="backtest_dashboard.html" class="nav-btn">➡️ 前往詳細歷史回測報告</a>
        </header>

        <script>
            const DATA = {js_data};
            
            document.addEventListener("DOMContentLoaded", () => {{
                document.getElementById('last-date').textContent = DATA.analysis.last_draw_date;
                document.getElementById('total-draws').textContent = DATA.analysis.total_draws;
                
                renderPredictions(DATA.predictions);
                renderCharts(DATA.analysis);
                renderTables(DATA.analysis);
                renderBacktest(DATA.backtest);
            }});
        </script>

        <!-- Predictions -->
        <div class="grid">
            <div class="card" style="grid-column: 1 / -1;">
                <h2>🎯 下期預測 (基於統計模型)</h2>
                <div class="grid" style="margin-bottom: 0; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));">
                    <div class="prediction-box pred-trend">
                        <div style="font-weight: bold; color: var(--accent-hot);">🔥 熱門追蹤</div>
                        <div class="balls" id="pred-trend-balls"></div>
                    </div>
                    <div class="prediction-box pred-cold">
                        <div style="font-weight: bold; color: var(--accent-cold);">❄️ 冷門逆勢</div>
                        <div class="balls" id="pred-cold-balls"></div>
                    </div>
                    <div class="prediction-box pred-mix">
                        <div style="font-weight: bold; color: var(--accent-mix);">⚖️ 平衡精選</div>
                        <div class="balls" id="pred-mix-balls"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Backtest Analysis Summary -->
        <div class="grid">
            <div class="card" style="grid-column: 1 / -1;">
                <h2>📉 歷史趨勢摘要</h2>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 20px;">
                    冷熱值 (Rank) 越低代表越「熱門」。
                </p>
                <canvas id="chart-backtest"></canvas>
            </div>
        </div>

        <!-- Charts -->
        <div class="grid">
            <div class="card">
                <h2>📊 第一區頻率</h2>
                <canvas id="chart-freq-1"></canvas>
            </div>
            <div class="card">
                <h2>⏳ 第一區遺漏值</h2>
                <canvas id="chart-overdue-1"></canvas>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>📋 重點數據榜</h2>
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <div style="flex: 1;">
                        <h3 style="font-size: 0.9rem; color: var(--accent-hot);">🔥 最熱 (Top 5)</h3>
                        <table id="table-hot"></table>
                    </div>
                    <div style="flex: 1;">
                        <h3 style="font-size: 0.9rem; color: var(--accent-cold);">❄️ 最冷 (Top 5)</h3>
                        <table id="table-cold"></table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function createBall(num, type) {{
            const el = document.createElement('div');
            el.className = `ball ${{type}}`;
            el.textContent = num < 10 ? '0'+num : num;
            return el;
        }}

        function renderPredictions(preds) {{
            const types = [
                {{key: 'trend', id: 'pred-trend-balls'}},
                {{key: 'contrarian', id: 'pred-cold-balls'}},
                {{key: 'balanced', id: 'pred-mix-balls'}}
            ];
            types.forEach(t => {{
                const container = document.getElementById(t.id);
                const p = preds[t.key];
                p.nums.forEach(n => container.appendChild(createBall(n, 'sec1')));
                container.appendChild(createBall(p.sec2, 'sec2'));
            }});
        }}
        
        function renderBacktest(results) {{
            const ctx = document.getElementById('chart-backtest').getContext('2d');
            const dates = results.map(r => r.date);
            const ranks = results.map(r => r.avg_rank);
            
            // Moving Average (SMA 10)
            const sma = [];
            const windowSize = 10;
            for(let i=0; i<ranks.length; i++) {{
                if(i < windowSize) {{ sma.push(null); continue; }}
                const slice = ranks.slice(i-windowSize, i);
                const avg = slice.reduce((a,b)=>a+b,0) / windowSize;
                sma.push(avg);
            }}

            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: dates,
                    datasets: [
                        {{
                            label: '當期冷熱值',
                            data: ranks,
                            borderColor: 'rgba(96, 165, 250, 0.3)',
                            pointRadius: 0,
                            borderWidth: 1,
                            tension: 0.4
                        }},
                        {{
                            label: '10期趨勢線',
                            data: sma,
                            borderColor: '#f43f5e',
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.4
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            reverse: true,
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }},
                        x: {{ display: false }}
                    }},
                    indexAxis: 'x',
                    interaction: {{ mode: 'index', intersect: false }}
                }}
            }});
        }}

        function renderCharts(analysis) {{
            renderBarChart('chart-freq-1', analysis.sec1_freq, 38, 'rgba(99, 102, 241, 0.6)');
            renderBarChart('chart-overdue-1', analysis.sec1_overdue, 38, 'rgba(59, 130, 246, 0.6)');
        }}
        
        function renderBarChart(canvasId, dataMap, rangeMax, color) {{
            const ctx = document.getElementById(canvasId).getContext('2d');
            const labels = Array.from({{length: rangeMax}}, (_, i) => i + 1);
            const data = labels.map(l => dataMap[l] || 0);
            
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: '數值',
                        data: data,
                        backgroundColor: color,
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ grid: {{ color: 'rgba(255,255,255,0.1)' }} }},
                        x: {{ grid: {{ display: false }} }}
                    }}
                }}
            }});
        }}
        
        function renderTables(analysis) {{
            const makeTable = (data, elemId) => {{
                let html = '<thead><tr><th>號碼</th><th>數值</th></tr></thead><tbody>';
                data.forEach((item, idx) => {{
                    html += `<tr><td class="rank-${{idx+1}}">${{item[0]}}</td><td>${{item[1]}}</td></tr>`;
                }});
                document.getElementById(elemId).innerHTML = html + '</tbody>';
            }};
            
            makeTable(Object.entries(analysis.sec1_freq).sort((a,b)=>b[1]-a[1]).slice(0,5), 'table-hot');
            makeTable(Object.entries(analysis.sec1_overdue).sort((a,b)=>b[1]-a[1]).slice(0,5), 'table-cold');
        }}
    </script>
</body>
</html>
    """
    
    with open("lottery_dashboard.html", "w", encoding='utf-8') as f:
        f.write(html_content)
    print("Main Dashboard generated: 'lottery_dashboard.html'")

def generate_backtest_html(backtest_results):
    js_data = json.dumps(backtest_results)
    
    # Calculate distribution
    ranks = [r['avg_rank'] for r in backtest_results]
    bins = [0]*40 # bins for ranks 0-38
    for r in ranks:
        bins[int(r)] += 1
    
    dist_data = json.dumps(bins)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>威力彩歷史回測報告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --grid-color: #334155;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            border-bottom: 1px solid var(--grid-color);
            padding-bottom: 20px;
        }}
        
        h1 {{ margin: 0; background: linear-gradient(to right, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        
        .btn {{
            padding: 8px 16px;
            background: rgba(255,255,255,0.1);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        
        .card {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .grid-2 {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid var(--grid-color); }}
        th {{ position: sticky; top: 0; background: var(--bg-color); }}
        
        .rank-hot {{ color: #fbbf24; }}
        .rank-cold {{ color: #3b82f6; }}
        
        .scroll-box {{ height: 500px; overflow-y: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>📉 歷史回測詳細報告</h1>
                <p style="color: var(--text-secondary)">分析範圍: 第500期至今</p>
            </div>
            <a href="lottery_dashboard.html" class="btn">⬅️ 返回主儀表板</a>
        </header>

        <div class="card">
            <h2>🔎 趨勢總覽</h2>
            <div style="height: 400px;">
                <canvas id="chart-trend"></canvas>
            </div>
        </div>

        <div class="grid-2">
            <div class="card">
                <h2>📊 平均排名分佈 (Distribution)</h2>
                <div style="height: 300px;">
                    <canvas id="chart-dist"></canvas>
                </div>
                <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top:10px;">
                    此圖表顯示中獎號碼之「平均排名」的出現次數分佈。<br>
                    若分佈偏左 (低排名)，代表大樂透整體傾向開出熱門號碼。<br>
                    若分佈呈現常態分佈 (鐘形曲線)，代表開獎符合隨機機率。
                </p>
            </div>
            
            <div class="card">
                <h2>📋 詳細歷史紀錄</h2>
                <div class="scroll-box">
                    <table>
                        <thead>
                            <tr>
                                <th>日期</th>
                                <th>期數</th>
                                <th>平均排名</th>
                                <th>號碼(排名)</th>
                            </tr>
                        </thead>
                        <tbody id="table-body"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        const DATA = {js_data};
        const DIST = {dist_data};

        // Render Trend Chart
        const ctxTrend = document.getElementById('chart-trend').getContext('2d');
        const dates = DATA.map(r => r.date);
        const ranks = DATA.map(r => r.avg_rank);
        
        // MA30
        const ma30 = [];
        const w = 30;
        for(let i=0; i<ranks.length; i++) {{
            if(i < w) {{ ma30.push(null); continue; }}
            const s = ranks.slice(i-w, i);
            ma30.push(s.reduce((a,b)=>a+b,0)/w);
        }}

        new Chart(ctxTrend, {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: [
                    {{
                        label: '當期平均排名 (Raw)',
                        data: ranks,
                        borderColor: 'rgba(96, 165, 250, 0.4)',
                        pointRadius: 1,
                        borderWidth: 1,
                        tension: 0.4
                    }},
                    {{
                        label: '30期移動平均 (Trend)',
                        data: ma30,
                        borderColor: '#f43f5e',
                        backgroundColor: 'rgba(244, 63, 94, 0.1)',
                        borderWidth: 3,
                        pointRadius: 0,
                        fill: false
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ reverse: true, grid: {{ color: 'rgba(255,255,255,0.1)' }} }}
                }},
                interaction: {{ intersect: false, mode: 'index' }}
            }}
        }});

        // Render Distribution
        new Chart(document.getElementById('chart-dist'), {{
            type: 'bar',
            data: {{
                labels: Array.from({{length:40}}, (_,i)=>i),
                datasets: [{{
                    label: '出現次數',
                    data: DIST,
                    backgroundColor: '#10b981',
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255,255,255,0.1)' }} }},
                    x: {{ grid: {{ display: false }} }}
                }}
            }}
        }});

        // Render Table
        const tbody = document.getElementById('table-body');
        // Show reverse chronological
        [...DATA].reverse().forEach(row => {{
            const tr = document.createElement('tr');
            
            // Format numbers with ranks: e.g. "05(1)"
            const numDetails = row.winning_nums.map((n, i) => {{
                let r = row.ranks[i];
                return `<span title="Rank ${{r}}">${{n}}</span>`;
            }}).join(', ');

            tr.innerHTML = `
                <td>${{row.date}}</td>
                <td>${{row.period}}</td>
                <td class="${{row.avg_rank < 15 ? 'rank-hot' : (row.avg_rank > 25 ? 'rank-cold' : '')}}">
                    ${{row.avg_rank}}
                </td>
                <td>${{numDetails}}</td>
            `;
            tbody.appendChild(tr);
        }});
    </script>
</body>
</html>
    """
    
    with open("backtest_dashboard.html", "w", encoding='utf-8') as f:
        f.write(html_content)
    print("Backtest Dashboard generated: 'backtest_dashboard.html'")

def generate_ml_dashboard(ml_results):
    js_data = json.dumps(ml_results)
    
    # Calculate Hit Distribution
    rf_counts = [0]*7
    dl_counts = [0]*7
    xgb_counts = [0]*7
    ens_counts = [0]*7
    
    rf_high_hits = 0
    dl_high_hits = 0
    xgb_high_hits = 0
    ens_high_hits = 0
    overlap_high_hits = 0
    
    # Conditional Stats (When Overlap >= 1)
    cnt_overlap_any = 0
    rf_high_when_overlap = 0
    dl_high_when_overlap = 0
    xgb_high_when_overlap = 0
    ens_high_when_overlap = 0
    
    # Prize Totals
    rf_prize_total = 0
    dl_prize_total = 0
    xgb_prize_total = 0
    ens_prize_total = 0
    
    # Prize Calculation Logic
    def calculate_prize(h_main, h_spec):
        # 3rd: 5 + 1 ($150,000)
        if h_main == 5 and h_spec: return 150000
        # 4th: 5 + 0 ($20,000)
        if h_main == 5: return 20000
        # 5th: 4 + 1 ($4,000)
        if h_main == 4 and h_spec: return 4000
        # 6th: 4 + 0 ($800)
        if h_main == 4: return 800
        # 7th: 3 + 1 ($400)
        if h_main == 3 and h_spec: return 400
        # 8th: 2 + 1 ($200)
        if h_main == 2 and h_spec: return 200
        # 9th: 3 + 0 ($100)
        if h_main == 3: return 100
        # Ordinary: 1 + 1 ($100)
        if h_main == 1 and h_spec: return 100
        return 0
    
    # Filter high hits for separate table
    high_hit_records = []
    
    total = len(ml_results)
    
    for r in ml_results:
        h_rf = r.get('rf_hits_m', 0)
        h_dl = r.get('dl_hits_m', 0)
        h_xgb = r.get('xgb_hits_m', 0)
        h_ens = r.get('ens_hits_m', 0)
        
        # Check Special Numbers matches
        s_rf = r.get('rf_hits_s', 0) == 1
        s_dl = r.get('dl_hits_s', 0) == 1
        s_xgb = r.get('xgb_hits_s', 0) == 1
        s_ens = r.get('ens_hits_s', 0) == 1
        
        # Calculate Prizes
        p_rf = calculate_prize(h_rf, s_rf)
        p_dl = calculate_prize(h_dl, s_dl)
        p_xgb = calculate_prize(h_xgb, s_xgb)
        p_ens = calculate_prize(h_ens, s_ens)
        
        rf_prize_total += p_rf
        dl_prize_total += p_dl
        xgb_prize_total += p_xgb
        ens_prize_total += p_ens
        
        # Add prize to record for JS
        r['rf_prize'] = p_rf
        r['dl_prize'] = p_dl
        r['xgb_prize'] = p_xgb
        r['ens_prize'] = p_ens
        
        # Calculate Overlap Hits
        set_rf = set(r.get('rf_nums', []))
        set_dl = set(r.get('dl_nums', []))
        set_actual = set(r.get('actual_nums', []))
        
        overlap_nums = set_rf.intersection(set_dl)
        ov_hits = len(overlap_nums.intersection(set_actual))
        
        # Conditional Logic
        if len(overlap_nums) >= 1:
            cnt_overlap_any += 1
            if h_rf >= 4: rf_high_when_overlap += 1
            if h_dl >= 4: dl_high_when_overlap += 1
            if h_xgb >= 4: xgb_high_when_overlap += 1
            if h_ens >= 4: ens_high_when_overlap += 1
        
        if 0 <= h_rf <= 6: rf_counts[h_rf] += 1
        if 0 <= h_dl <= 6: dl_counts[h_dl] += 1
        if 0 <= h_xgb <= 6: xgb_counts[h_xgb] += 1
        if 0 <= h_ens <= 6: ens_counts[h_ens] += 1
        
        if h_rf >= 4: rf_high_hits += 1
        if h_dl >= 4: dl_high_hits += 1
        if h_xgb >= 4: xgb_high_hits += 1
        if h_ens >= 4: ens_high_hits += 1
        if ov_hits >= 4: overlap_high_hits += 1
        
        if h_rf >= 4 or h_dl >= 4 or h_xgb >= 4 or h_ens >= 4:
            high_hit_records.append(r)
            
    hit_dist = {
        "rf": rf_counts,
        "dl": dl_counts,
        "xgb": xgb_counts,
        "ens": ens_counts
    }
    js_dist = json.dumps(hit_dist)
    js_high_hits = json.dumps(high_hit_records)
    # Re-dump detailed results with prize info
    js_data = json.dumps(ml_results)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 預測模型分析</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --grid-color: #334155;
            --c-rf: #f59e0b;
            --c-dl: #8b5cf6;
            --c-xgb: #3b82f6;
            --c-ens: #06b6d4;
            --c-ov: #ec4899;
            --c-actual: #10b981;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        
        .container {{ max-width: 1800px; margin: 0 auto; }}
        
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            border-bottom: 1px solid var(--grid-color);
            padding-bottom: 20px;
        }}
        
        h1 {{ margin: 0; background: linear-gradient(to right, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        
        .btn {{
            padding: 8px 16px;
            background: rgba(255,255,255,0.1);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        
        .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
        
        .card {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid var(--grid-color); white-space: nowrap; }}
        th {{ position: sticky; top: 0; background: var(--bg-color); z-index: 10; }}
        
        .val-rf {{ color: var(--c-rf); font-weight: bold; }}
        .val-dl {{ color: var(--c-dl); font-weight: bold; }}
        .val-xgb {{ color: var(--c-xgb); font-weight: bold; }}
        .val-ens {{ color: var(--c-ens); font-weight: bold; }}
        .val-actual {{ color: var(--c-actual); font-weight: bold; }}
        
        .gap-good {{ color: #10b981; }}
        .gap-bad {{ color: #ef4444; }}
        
        .hit-high {{ color: #f59e0b; font-weight: bold; background: rgba(245, 158, 11, 0.1); padding: 2px 6px; border-radius: 4px; }}
        .hit-med {{ color: #fbbf24; }}
        .hit-zero {{ color: #64748b; }}
        
        .prize-tag {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.75rem; margin-left: 5px; }}
        .prize-money {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }}

        .scroll-box {{ height: 500px; overflow-y: auto; overflow-x: auto; }}
        
        .past-info {{ font-size: 0.75rem; color: #64748b; font-family: monospace; display: block; max-width: 200px; overflow: hidden; text-overflow: ellipsis; }}
        
        .big-stat {{ font-size: 2rem; font-weight: bold; }}
        .stat-label {{ color: var(--text-secondary); font-size: 0.9rem; }}
        .highlight-rf {{ color: var(--c-rf); }}
        .highlight-dl {{ color: var(--c-dl); }}
        .highlight-xgb {{ color: var(--c-xgb); }}
        .highlight-ens {{ color: var(--c-ens); }}
        .highlight-ov {{ color: var(--c-ov); }}

        /* High Hit Table Styles */
        .high-hit-row {{ background: rgba(245, 158, 11, 0.05); }}
        .hit-badge {{ 
            display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;
        }}
        .badge-rf {{ background: rgba(245, 158, 11, 0.2); color: var(--c-rf); border: 1px solid var(--c-rf); }}
        .badge-dl {{ background: rgba(139, 92, 246, 0.2); color: var(--c-dl); border: 1px solid var(--c-dl); }}
        .badge-xgb {{ background: rgba(59, 130, 246, 0.2); color: var(--c-xgb); border: 1px solid var(--c-xgb); }}
        .badge-ens {{ background: rgba(6, 182, 212, 0.2); color: var(--c-ens); border: 1px solid var(--c-ens); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🤖 AI 預測模型分析 (ML/DL/XGB/Ensemble)</h1>
                <p style="color: var(--text-secondary)">評估指標: Rank Gap (冷熱值誤差) & Hit Count (命中數)</p>
            </div>
            <div>
                <a href="lottery_dashboard.html" class="btn">⬅️ 主儀表板</a>
                <a href="backtest_dashboard.html" class="btn">📉 歷史回測</a>
            </div>
        </header>
        
        <!-- Summary Cards -->
        <div class="grid-3">
            <div class="card">
                <h2>🏆 高命中次數統計 (Hits >= 4)</h2>
                <div style="display: flex; justify-content: space-around; text-align: center; margin-top: 20px;">
                    <div>
                        <div class="big-stat highlight-rf">{rf_high_hits}</div>
                        <div class="stat-label">Random Forest</div>
                    </div>
                    <div>
                        <div class="big-stat highlight-dl">{dl_high_hits}</div>
                        <div class="stat-label">Deep Learning</div>
                    </div>
                    <div>
                        <div class="big-stat highlight-xgb">{xgb_high_hits}</div>
                        <div class="stat-label">XGB Base</div>
                    </div>
                    <div>
                        <div class="big-stat highlight-ens">{ens_high_hits}</div>
                        <div class="stat-label">Stacking Ens</div>
                    </div>
                    <div>
                        <div class="big-stat">{total}</div>
                        <div class="stat-label">總回測期數</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>💰 模擬累計獎金 (Total Prize)</h2>
                <div style="display: flex; justify-content: space-around; text-align: center; margin-top: 20px;">
                    <div>
                        <div class="big-stat highlight-rf">${rf_prize_total:,}</div>
                        <div class="stat-label">RF Total</div>
                    </div>
                    <div>
                        <div class="big-stat highlight-dl">${dl_prize_total:,}</div>
                        <div class="stat-label">DL Total</div>
                    </div>
                    <div>
                        <div class="big-stat highlight-xgb">${xgb_prize_total:,}</div>
                        <div class="stat-label">XGB Total</div>
                    </div>
                    <div>
                        <div class="big-stat highlight-ens">${ens_prize_total:,}</div>
                        <div class="stat-label">Ens Total</div>
                    </div>
                </div>
                <p style="text-align: center; margin-top: 10px; font-size: 0.8rem; color: #64748b;">
                    規則: 參/肆/伍/陸/柒/捌/玖/普獎 (不含頭二獎)
                </p>
            </div>
            
            <div class="card">
                <h2>🤝 共識效應分析 (當 RF & DL 至少有1號碼重疊時)</h2>
                <div style="display: flex; justify-content: space-around; text-align: center; margin-top: 20px;">
                    <div>
                        <div class="big-stat">{cnt_overlap_any}</div>
                        <div class="stat-label">發生次數 (Overlap >= 1)</div>
                    </div>
                    <div>
                        <div class="big-stat highlight-rf">{rf_high_when_overlap}</div>
                        <div class="stat-label">RF High Hits</div>
                    </div>
                    <div>
                        <div class="big-stat highlight-dl">{dl_high_when_overlap}</div>
                        <div class="stat-label">DL High Hits</div>
                    </div>
                    <div>
                        <div class="big-stat highlight-xgb">{xgb_high_when_overlap}</div>
                        <div class="stat-label">XGB High Hits</div>
                    </div>
                </div>
                <p style="text-align: center; margin-top: 15px; font-size: 0.85rem; color: var(--text-secondary);">
                    此區塊顯示當 RF 和 DL 達成某種程度的「共識」(至少1碼相同) 時，各模型的高命中率表現是否提升。
                </p>
            </div>
        </div>
        
        <div class="grid">
             <!-- High Hits Detail Table -->
            <div class="card">
                <h2>🌟 高命中詳細紀錄 (Hits >= 4)</h2>
                <div class="scroll-box" style="height: 300px;">
                    <table>
                        <thead>
                            <tr>
                                <th>期數</th>
                                <th>實際開獎 (SP)</th>
                                <th>RF 命中</th>
                                <th>RF 預測 (SP)</th>
                                <th>DL 命中</th>
                                <th>DL 預測 (SP)</th>
                                <th>XGB 命中</th>
                                <th>XGB 預測 (SP)</th>
                                <th>Ens 命中</th>
                                <th>Ens 預測 (SP)</th>
                            </tr>
                        </thead>
                        <tbody id="high-hits-body"></tbody>
                    </table>
                </div>
            </div>

            <div class="card">
                <h2>📊 命中數分佈 (Hit Distribution)</h2>
                <div style="height: 200px;">
                    <canvas id="chart-hit-dist"></canvas>
                </div>
            </div>

            <div class="card">
                <h2>📈 模型預測誤差趨勢 (Rank Gap)</h2>
                <div style="height: 300px;">
                    <canvas id="chart-gap"></canvas>
                </div>
                <p style="text-align: center; margin-top: 10px; font-size: 0.9rem;">
                    Gap 越小代表模型越準確地預測了當期的「冷熱傾向」。<br>
                    <span style="color:var(--c-rf)">RF</span> vs 
                    <span style="color:var(--c-dl)">DL</span> vs 
                    <span style="color:var(--c-xgb)">XGB</span> vs 
                    <span style="color:var(--c-ens)">Stacking</span>
                </p>
            </div>
            
            <div class="card">
                <h2>📋 每期詳細預測紀錄</h2>
                <div class="scroll-box">
                    <table>
                        <thead>
                            <tr>
                                <th rowspan="2">期數</th>
                                <th rowspan="2" style="min-width: 150px;">前10期冷熱值 (Avg Ranks)</th>
                                <th colspan="2">實際開獎</th>
                                <th rowspan="2" style="min-width: 100px; background: rgba(255,255,255,0.05);">Overlap (RF∩DL)</th>
                                <th colspan="4">RF (Random Forest)</th>
                                <th colspan="4">DL (Deep Learning)</th>
                                <th colspan="4">XGB (Base Model)</th>
                                <th colspan="4">Stacking Ensemble</th>
                            </tr>
                            <tr>
                                <th>號碼 (Rank)</th>
                                <th>SP (R)</th>
                                <th>Hit</th>
                                <th>Gap</th>
                                <th>號碼 (Rank)</th>
                                <th>SP (R)</th>
                                <th>Hit</th>
                                <th>Gap</th>
                                <th>號碼 (Rank)</th>
                                <th>SP (R)</th>
                                <th>Hit</th>
                                <th>Gap</th>
                                <th>號碼 (Rank)</th>
                                <th>SP (R)</th>
                                <th>Hit</th>
                                <th>Gap</th>
                            </tr>
                        </thead>
                        <tbody id="table-body"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        const DATA = {js_data};
        const HITS = {js_dist};
        const HIGH_HITS = {js_high_hits};

        // Render Hit Distribution Chart
        const ctxHist = document.getElementById('chart-hit-dist').getContext('2d');
        new Chart(ctxHist, {{
            type: 'bar',
            data: {{
                labels: ['0 Hit', '1 Hit', '2 Hits', '3 Hits', '4 Hits', '5 Hits', '6 Hits'],
                datasets: [
                    {{ label: 'RF', data: HITS.rf, backgroundColor: '#f59e0b', borderRadius: 4 }},
                    {{ label: 'DL', data: HITS.dl, backgroundColor: '#8b5cf6', borderRadius: 4 }},
                    {{ label: 'XGB', data: HITS.xgb, backgroundColor: '#3b82f6', borderRadius: 4 }},
                    {{ label: 'Ens', data: HITS.ens, backgroundColor: '#06b6d4', borderRadius: 4 }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255,255,255,0.1)' }} }},
                    x: {{ grid: {{ display: false }} }}
                }}
            }}
        }});

        // Render High Hits Table
        const hhBody = document.getElementById('high-hits-body');
        
        function highlightMatch(nums, winning) {{
            return nums.map(n => winning.includes(n) ? `<span style="color:#10b981;font-weight:bold">${{n}}</span>` : `<span style="color:#64748b">${{n}}</span>`).join(', ');
        }}
        
        function formatSP(actual, pred) {{
            return actual === pred ? `<strong style="color:#f43f5e">(${{pred}}) ✓</strong>` : `<span style="color:#64748b">(${{pred}})</span>`;
        }}

        [...HIGH_HITS].reverse().forEach(row => {{
            const tr = document.createElement('tr');
            tr.className = 'high-hit-row';
            
            let rfBadge = row.rf_hits_m >= 4 ? `<span class="hit-badge badge-rf">${{row.rf_hits_m}} Hits</span>` : `<span class="hit-zero">${{row.rf_hits_m}}</span>`;
            let dlBadge = row.dl_hits_m >= 4 ? `<span class="hit-badge badge-dl">${{row.dl_hits_m}} Hits</span>` : `<span class="hit-zero">${{row.dl_hits_m}}</span>`;
            let xgbBadge = row.xgb_hits_m >= 4 ? `<span class="hit-badge badge-xgb">${{row.xgb_hits_m}} Hits</span>` : `<span class="hit-zero">${{row.xgb_hits_m}}</span>`;
            let ensBadge = row.ens_hits_m >= 4 ? `<span class="hit-badge badge-ens">${{row.ens_hits_m}} Hits</span>` : `<span class="hit-zero">${{row.ens_hits_m}}</span>`;
            
            tr.innerHTML = `
                <td><strong>${{row.period}}</strong></td>
                <td>
                    <span class="val-actual">${{row.actual_nums.join(', ')}}</span>
                    <strong style="color:#f43f5e; margin-left:5px">(${{row.actual_spec}})</strong>
                </td>
                
                <td>${{rfBadge}}</td>
                <td>
                    ${{highlightMatch(row.rf_nums, row.actual_nums)}}
                    ${{formatSP(row.actual_spec, row.rf_spec)}}
                </td>
                
                <td>${{dlBadge}}</td>
                <td>
                    ${{highlightMatch(row.dl_nums, row.actual_nums)}}
                    ${{formatSP(row.actual_spec, row.dl_spec)}}
                </td>
                
                <td>${{xgbBadge}}</td>
                <td>
                    ${{highlightMatch(row.xgb_nums, row.actual_nums)}}
                    ${{formatSP(row.actual_spec, row.xgb_spec)}}
                </td>
                
                <td>${{ensBadge}}</td>
                <td>
                    ${{highlightMatch(row.ens_nums, row.actual_nums)}}
                    ${{formatSP(row.actual_spec, row.ens_spec)}}
                </td>
            `;
            hhBody.appendChild(tr);
        }});
        
        // Render Gap Chart
        const ctxGap = document.getElementById('chart-gap').getContext('2d');
        const periods = DATA.map(r => r.period);
        const gapsRF = DATA.map(r => r.rf_gap_m);
        const gapsDL = DATA.map(r => r.dl_gap_m);
        const gapsXGB = DATA.map(r => r.xgb_gap_m || 0);
        const gapsEns = DATA.map(r => r.ens_gap_m || 0); // Handle missing if any
        
        // Rolling Avg for smooth line
        function sma(arr, w) {{
            const content = [];
            for(let i=0; i<arr.length; i++) {{
                if(i<w) {{ content.push(null); continue; }}
                content.push( arr.slice(i-w,i).reduce((a,b)=>a+b,0)/w );
            }}
            return content;
        }}
        
        new Chart(ctxGap, {{
            type: 'line',
            data: {{
                labels: periods,
                datasets: [
                    {{ label: 'RF (SMA 20)', data: sma(gapsRF, 20), borderColor: '#f59e0b', borderWidth: 2, pointRadius: 0 }},
                    {{ label: 'DL (SMA 20)', data: sma(gapsDL, 20), borderColor: '#8b5cf6', borderWidth: 2, pointRadius: 0 }},
                    {{ label: 'XGB (SMA 20)', data: sma(gapsXGB, 20), borderColor: '#3b82f6', borderWidth: 2, pointRadius: 0 }},
                    {{ label: 'Ens (SMA 20)', data: sma(gapsEns, 20), borderColor: '#06b6d4', borderWidth: 2, pointRadius: 0 }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ grid: {{ color: 'rgba(255,255,255,0.1)' }}, title: {{ display: true, text: 'Gap (Lower is Better)' }} }},
                    x: {{ display: false }}
                }},
                interaction: {{ intersect: false, mode: 'index' }}
            }}
        }});

        // Render Table
        const tbody = document.getElementById('table-body');
        
        function hitClass(h) {{
            if(h >= 2) return 'hit-high';
            if(h >= 1) return 'hit-med';
            return 'hit-zero';
        }}
        
        function spHit(actual, pred) {{
            return actual === pred ? '<span class="hit-high" style="display:inline-block;width:20px;text-align:center">✓</span>' : '<span style="color:#64748b">✗</span>';
        }}
        
        function formatPrize(p) {{
            if (!p || p === 0) return '';
            return `<span class="prize-tag prize-money">$${{p}}</span>`;
        }}

        [...DATA].reverse().forEach(row => {{
            const tr = document.createElement('tr');
            
            // Format arrays
            // Past 10 avg ranks
            const past10 = row.past_10_avgs ? row.past_10_avgs.join(', ') : 'N/A';
            
            // Overlap Calculation
            const rfSet = new Set(row.rf_nums);
            const dlSet = new Set(row.dl_nums);
            const actualSet = new Set(row.actual_nums);
            const overlap = row.rf_nums.filter(x => dlSet.has(x));
            const overlapStr = overlap.length > 0 ? overlap.join(', ') : '-';
            const ovHits = overlap.filter(x => actualSet.has(x)).length;
            
            const hlOverlap = ovHits >= 4 ? 'hit-high' : (ovHits > 0 ? 'hit-med' : '');
            
            tr.innerHTML = `
                <td>${{row.period}}</td>
                <td><span class="past-info" title="${{past10}}">${{past10}}</span></td>
                
                <!-- Actual -->
                <td>
                    <div>${{row.actual_nums.join(', ')}}</div>
                    <div class="val-actual">R: ${{row.actual_val_m}}</div>
                </td>
                <td style="border-right: 2px solid var(--grid-color)">
                    <div style="font-weight:bold;color:#f43f5e">${{row.actual_spec}}</div>
                    <div class="val-actual">R:${{row.actual_val_s}}</div>
                </td>
                
                <!-- Overlap (RF ∩ DL) -->
                <td style="border-right: 2px solid var(--grid-color); background: rgba(255, 255, 255, 0.02);">
                    <div style="color: #e2e8f0; font-weight: bold;">${{overlapStr}}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">Count: ${{overlap.length}} <span class="${{hlOverlap}}">(Hits: ${{ovHits}})</span></div>
                </td>
                
                <!-- RF -->
                <td>
                    <div>${{row.rf_nums.join(', ')}}</div>
                    <div class="val-rf">R: ${{row.rf_val_m}}</div>
                </td>
                <td>
                    <div style="font-weight:bold;color:#f43f5e">${{row.rf_spec}}</div>
                    ${{spHit(row.actual_spec, row.rf_spec)}}
                </td>
                <td><span class="${{hitClass(row.rf_hits_m)}}">${{row.rf_hits_m}}</span>${{formatPrize(row.rf_prize)}}</td>
                <td class="${{row.rf_gap_m <= 5 ? 'gap-good' : 'gap-bad'}}" style="border-right: 2px solid var(--grid-color)">${{row.rf_gap_m}}</td>
                
                <!-- DL -->
                <td>
                    <div>${{row.dl_nums.join(', ')}}</div>
                    <div class="val-dl">R: ${{row.dl_val_m}}</div>
                </td>
                <td>
                    <div style="font-weight:bold;color:#f43f5e">${{row.dl_spec}}</div>
                    ${{spHit(row.actual_spec, row.dl_spec)}}
                </td>
                <td><span class="${{hitClass(row.dl_hits_m)}}">${{row.dl_hits_m}}</span>${{formatPrize(row.dl_prize)}}</td>
                <td class="${{row.dl_gap_m <= 5 ? 'gap-good' : 'gap-bad'}}" style="border-right: 2px solid var(--grid-color)">${{row.dl_gap_m}}</td>
                
                <!-- XGB -->
                <td>
                    <div>${{row.xgb_nums ? row.xgb_nums.join(', ') : '-'}}</div>
                    <div class="val-xgb">R: ${{row.xgb_val_m || '-'}}</div>
                </td>
                <td>
                    <div style="font-weight:bold;color:#f43f5e">${{row.xgb_spec || '-'}}</div>
                    ${{spHit(row.actual_spec, row.xgb_spec)}}
                </td>
                <td><span class="${{hitClass(row.xgb_hits_m || 0)}}">${{row.xgb_hits_m || 0}}</span>${{formatPrize(row.xgb_prize)}}</td>
                <td class="${{(row.xgb_gap_m || 99) <= 5 ? 'gap-good' : 'gap-bad'}}" style="border-right: 2px solid var(--grid-color)">${{row.xgb_gap_m || '-'}}</td>
                
                <!-- Ensemble -->
                <td>
                    <div>${{row.ens_nums ? row.ens_nums.join(', ') : '-'}}</div>
                    <div class="val-ens">R: ${{row.ens_val_m || '-'}}</div>
                </td>
                <td>
                    <div style="font-weight:bold;color:#f43f5e">${{row.ens_spec || '-'}}</div>
                    ${{spHit(row.actual_spec, row.ens_spec)}}
                </td>
                <td><span class="${{hitClass(row.ens_hits_m || 0)}}">${{row.ens_hits_m || 0}}</span>${{formatPrize(row.ens_prize)}}</td>
                <td class="${{(row.ens_gap_m || 99) <= 5 ? 'gap-good' : 'gap-bad'}}">${{row.ens_gap_m || '-'}}</td>
            `;
            tbody.appendChild(tr);
        }});
    </script>
</body>
</html>
    """
    with open("ml_dashboard.html", "w", encoding='utf-8') as f:
        f.write(html_content)
    print("ML Dashboard generated: 'ml_dashboard.html'")

if __name__ == "__main__":
    data = load_data("super_lotto638_results.csv")
    ml_results = []
    
    # Load ML Results if exist
    try:
        with open("ml_backtest_results.json", "r", encoding='utf-8') as f:
            ml_results = json.load(f)
    except:
        print("No ML results found. Run ml_lottery.py first.")

    if data:
        analysis = analyze_data(data)
        predictions = generate_predictions(analysis)
        backtest = perform_backtest(data)
        
        generate_html(analysis, predictions, backtest)
        generate_backtest_html(backtest)
        
        if ml_results:
            generate_ml_dashboard(ml_results)
