document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching Logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-target');
            
            // Remove active classes
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active classes
            btn.classList.add('active');
            document.getElementById(target).classList.add('active');
        });
    });

    // Data Fetching
    const RAW_GITHUB_URL_LOTTO = 'recommendation_lotto649_output.txt'; // Relative path for GitHub Pages
    const RAW_GITHUB_URL_SUPER = 'recommendation_output.txt';
    
    // We will attempt relative fetch first, if it fails for testing locally, we can fallback to raw githubusercontent if we knew the repo structure.

    fetchAndParseText(RAW_GITHUB_URL_LOTTO, 'lotto649-dashboard', '大樂透');
    fetchAndParseText(RAW_GITHUB_URL_SUPER, 'superlotto-dashboard', '威力彩');

    renderJackpotStrategy('recommendation_lotto649_jackpot_history.json', 'lotto649-jackpot');
    renderBacktest(BACKTEST_CONFIG.lotto649, 'lotto649-backtest');
    renderBacktest(BACKTEST_CONFIG.superlotto, 'superlotto-backtest');
});

async function fetchAndParseText(url, targetId, gameType) {
    try {
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const text = await response.text();
        renderTextAsDashboard(text, targetId, gameType);
    } catch (e) {
        console.error("Fetch failed, trying timestamp bust:", e);
        try {
            const bustResponse = await fetch(`${url}?t=${new Date().getTime()}`);
            const text = await bustResponse.text();
            renderTextAsDashboard(text, targetId, gameType);
        } catch (err) {
            document.getElementById(targetId).innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📂</div>
                    <h3>無法載入推薦數據</h3>
                    <p>請確認 ${url} 檔案是否存在於伺服器上。</p>
                    <p style="font-size: 0.8rem; opacity: 0.5;">${err.message}</p>
                </div>
            `;
        }
    }
}

function renderTextAsDashboard(text, targetId, gameType) {
    const lines = text.split('\n');
    let date = '未知';
    let jackpot = '未知';
    let ticketsInfo = '';
    
    let currentCategory = '投注推薦';
    const numberGroups = []; // Array of { category: "...", sets: [ {nums: [], special: ''} ] }
    let currentGroupSets = [];
    
    // Pattern matchers
    const dateRegex = /(?:日期|開獎日期):\s*([0-9-]+)/;
    const jackpotRegex = /(?:頭獎|累積)[:-]?\s*\$?([\d,]+)/;
    const ticketRegex = /建議注數?:\s*(\d+)/;
    const blockRegex = /【\s*(.+?)\s*】/;
    const setRegex = /第\s*\d+\s*組:\s*\[(.*?)\](?:.*?特別號:\s*(\d+))?/;

    for (let line of lines) {
        line = line.trim();
        if (!line || line.includes('=====') || line.includes('-----')) continue;

        // Info Extractor
        if (dateRegex.test(line)) date = line.match(dateRegex)[1];
        if (jackpotRegex.test(line)) jackpot = line.match(jackpotRegex)[1];
        if (ticketRegex.test(line) && !ticketsInfo) ticketsInfo = line.match(ticketRegex)[1];

        // Categories (A組 / B組)
        if (blockRegex.test(line)) {
            if (currentGroupSets.length > 0) {
                numberGroups.push({ category: currentCategory, sets: currentGroupSets });
                currentGroupSets = [];
            }
            currentCategory = line.match(blockRegex)[1];
            // If it has A組 or B組, let's style it later using hot/cold based on keyword
        }

        // Sets
        if (setRegex.test(line)) {
            const match = line.match(setRegex);
            const nums = match[1].trim().split(/\s+/);
            const special = match[2] ? match[2].trim() : null;
            currentGroupSets.push({ nums, special });
        }
    }
    
    if (currentGroupSets.length > 0) {
        numberGroups.push({ category: currentCategory, sets: currentGroupSets });
    }

    // Build the UI
    const container = document.getElementById(targetId);
    
    let html = `
        <div class="card">
            <div class="card-header">
                <div class="card-title">📊 ${gameType} 開獎資訊</div>
                <div class="live-indicator"><span class="pulsing-dot"></span> 最新</div>
            </div>
            <div class="info-grid">
                <div class="stat-box">
                    <div class="stat-label">最新開獎日期</div>
                    <div class="stat-value">${date}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">預估頭獎</div>
                    <div class="stat-value accent">$${jackpot}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">總建議注數</div>
                    <div class="stat-value">${ticketsInfo ? ticketsInfo + ' 注' : '動態判斷'}</div>
                </div>
            </div>
        </div>
    `;

    numberGroups.forEach(group => {
        const isHot = group.category.includes('熱') || group.category.includes('Momentum');
        const isCold = group.category.includes('冷') || group.category.includes('Contrarian');
        const ballClass = isHot ? 'hot' : (isCold ? 'cold' : '');
        
        let setsHtml = '';
        group.sets.forEach((set, idx) => {
            let ballsHtml = set.nums.map(n => `<div class="ball ${ballClass}">${n}</div>`).join('');
            if (set.special) {
                ballsHtml += `<div class="ball special" title="特別號">${set.special}</div>`;
            }
            
            setsHtml += `
                <div class="number-row">
                    <div class="row-label">第 ${String(idx + 1).padStart(2, '0')} 組</div>
                    <div class="balls-container">
                        ${ballsHtml}
                    </div>
                </div>
            `;
        });

        html += `
            <div class="card">
                <div class="card-header">
                    <div class="card-title">🎯 ${group.category}</div>
                </div>
                <div class="number-groups">
                    ${setsHtml || '<div class="empty-state">無推薦號碼</div>'}
                </div>
            </div>
        `;
    });

    if (numberGroups.length === 0) {
        html += `
            <div class="card">
                <div class="empty-state">
                    <div class="empty-icon">⏸️</div>
                    <h3>暫無號碼推薦</h3>
                    <p>目前系統設定可能未達建議投注門檻，請參考上方的文字提示。</p>
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

// ===== 回測：假設每期都照推薦號碼投注 =====
// 浮動獎項：該期有人中則用實際每注獎金；無人中則假設獨得整個獎池。
const floatPrize = (draw, perCol, ...poolCols) =>
    Number(draw[perCol]) || poolCols.reduce((sum, c) => sum + Number(draw[c] || 0), 0);

const BACKTEST_START = '2026-02-01';

const BACKTEST_CONFIG = {
    lotto649: {
        name: '大樂透',
        historyUrl: 'recommendation_lotto649_history.json',
        csvUrl: 'data/lotto649_results.csv',
        ticketCost: 50,
        tiers: [
            ['頭獎', '6 個號碼'], ['貳獎', '5 個 + 特別號'], ['參獎', '5 個號碼'], ['肆獎', '4 個 + 特別號'],
            ['伍獎', '4 個號碼'], ['陸獎', '3 個 + 特別號'], ['柒獎', '2 個 + 特別號'], ['普獎', '3 個號碼'],
        ],
        getTickets: rec => rec.recommendations
            ? rec.recommendations.map(r => ({ nums: r.numbers.split(/\s+/).map(Number) }))
            : [...(rec.hot_sets || []), ...(rec.cold_sets || [])].map(nums => ({ nums })),
        // 大樂透特別號與一般號同池：選號中含特別號即算中
        judge: (ticket, draw) => {
            const m = ticket.nums.filter(n => draw.nums.includes(n)).length;
            const sp = ticket.nums.includes(draw.special);
            if (m === 6) return ['頭獎', floatPrize(draw, 'Jackpot_Per_Winner', 'Jackpot_Total', 'Jackpot_Rollover')];
            if (m === 5 && sp) return ['貳獎', floatPrize(draw, 'Second_Prize_Per_Winner', 'Second_Prize_Total')];
            if (m === 5) return ['參獎', floatPrize(draw, 'Third_Prize_Per_Winner', 'Third_Prize_Total')];
            if (m === 4 && sp) return ['肆獎', floatPrize(draw, 'Fourth_Prize_Per_Winner', 'Fourth_Prize_Total')];
            if (m === 4) return ['伍獎', Number(draw.Fifth_Prize_Per_Winner) || 2000];
            if (m === 3 && sp) return ['陸獎', Number(draw.Sixth_Prize_Per_Winner) || 1000];
            if (m === 2 && sp) return ['柒獎', Number(draw.Seventh_Prize_Per_Winner) || 400];
            if (m === 3) return ['普獎', Number(draw.Normal_Prize_Per_Winner) || 400];
            return null;
        },
    },
    superlotto: {
        name: '威力彩',
        historyUrl: 'recommendation_history.json',
        csvUrl: 'data/super_lotto638_results.csv',
        ticketCost: 100,
        tiers: [
            ['頭獎', '6 個 + 第二區'], ['貳獎', '6 個號碼'], ['參獎', '5 個 + 第二區'], ['肆獎', '5 個號碼'],
            ['伍獎', '4 個 + 第二區'], ['陸獎', '4 個號碼'], ['柒獎', '3 個 + 第二區'], ['捌獎', '2 個 + 第二區'],
            ['玖獎', '3 個號碼'], ['普獎', '1 個 + 第二區'],
        ],
        getTickets: rec => (rec.recommendations || []).map(r => ({
            nums: r.numbers.split(/\s+/).map(Number),
            special: Number(r.special),
        })),
        // 威力彩第二區獨立開出：第二區號碼相同才算中
        judge: (ticket, draw) => {
            const m = ticket.nums.filter(n => draw.nums.includes(n)).length;
            const sp = ticket.special === draw.special;
            if (m === 6 && sp) return ['頭獎', floatPrize(draw, 'First_Prize_Per_Winner', 'First_Prize_Total')];
            if (m === 6) return ['貳獎', floatPrize(draw, 'Second_Prize_Per_Winner', 'Second_Prize_Total')];
            if (m === 5 && sp) return ['參獎', 150000];
            if (m === 5) return ['肆獎', 20000];
            if (m === 4 && sp) return ['伍獎', 4000];
            if (m === 4) return ['陸獎', 800];
            if (m === 3 && sp) return ['柒獎', 400];
            if (m === 2 && sp) return ['捌獎', 200];
            if (m === 3) return ['玖獎', 100];
            if (m === 1 && sp) return ['普獎', 100];
            return null;
        },
    },
};

function parseDrawsCsv(text) {
    const [header, ...rows] = text.trim().split('\n');
    const cols = header.split(',');
    return rows.map(line => {
        const vals = line.split(',');
        const row = Object.fromEntries(cols.map((c, i) => [c, vals[i]]));
        row.nums = row.Numbers.split(/\s+/).map(Number);
        row.special = Number(row.Special_Number);
        return row;
    }).sort((a, b) => a.Date.localeCompare(b.Date));
}

function computeBacktest(cfg, history, draws) {
    return Object.keys(history).sort().map(baseDate => {
        const tickets = cfg.getTickets(history[baseDate]);
        // 推薦以 baseDate 當期為最新資料產生，對應的是下一期開獎
        const draw = draws.find(d => d.Date > baseDate);
        const cost = tickets.length * cfg.ticketCost;
        if (!draw) return { baseDate, tickets, cost, draw: null };

        const results = tickets.map(t => {
            const hit = cfg.judge(t, draw);
            return { ...t, prizeName: hit ? hit[0] : null, prize: hit ? hit[1] : 0 };
        });
        const prize = results.reduce((s, r) => s + r.prize, 0);
        return { baseDate, tickets: results, cost, draw, prize, backfilled: !!history[baseDate].backfilled };
    }).filter(r => r.draw && r.draw.Date >= BACKTEST_START);
}

const fmtMoney = n => `${n < 0 ? '-' : ''}$${Math.abs(Math.round(n)).toLocaleString('en-US')}`;

async function renderBacktest(cfg, targetId) {
    const container = document.getElementById(targetId);
    try {
        const [history, csvText] = await Promise.all([
            fetch(cfg.historyUrl, { cache: 'no-store' }).then(r => r.json()),
            fetch(cfg.csvUrl, { cache: 'no-store' }).then(r => r.text()),
        ]);
        const rows = computeBacktest(cfg, history, parseDrawsCsv(csvText));
        container.innerHTML = buildBacktestHtml(cfg, rows);
    } catch (e) {
        console.error('Backtest failed:', e);
        container.innerHTML = `
            <div class="card">
                <div class="empty-state">
                    <div class="empty-icon">📉</div>
                    <h3>無法載入回測資料</h3>
                    <p style="font-size: 0.8rem; opacity: 0.5;">${e.message}</p>
                </div>
            </div>`;
    }
}

function buildBacktestHtml(cfg, rows) {
    const tickets = rows.flatMap(r => r.tickets);
    const totalCost = rows.reduce((s, r) => s + r.cost, 0);
    const totalPrize = rows.reduce((s, r) => s + r.prize, 0);
    const net = totalPrize - totalCost;
    const roi = totalCost ? (net / totalCost) * 100 : 0;
    const winningTickets = tickets.filter(t => t.prize > 0).length;
    const backfilledCount = rows.filter(r => r.backfilled).length;
    const lastDate = rows.length ? rows[rows.length - 1].draw.Date : '—';

    const tierRows = cfg.tiers.map(([name, rule]) => {
        const hits = tickets.filter(t => t.prizeName === name);
        const amount = hits.reduce((s, t) => s + t.prize, 0);
        return `
            <tr class="${hits.length ? 'hit' : ''}">
                <td>${name}</td>
                <td class="bt-rule">${rule}</td>
                <td class="num">${hits.length}</td>
                <td class="num">${fmtMoney(amount)}</td>
            </tr>`;
    }).join('');

    return `
        <div class="card">
            <div class="card-header">
                <div class="card-title">💰 ${cfg.name} 推薦回測</div>
                <div class="bt-note">${BACKTEST_START.slice(0, 7)} ~ ${lastDate}</div>
            </div>
            <div class="info-grid">
                <div class="stat-box">
                    <div class="stat-label">已開獎期數</div>
                    <div class="stat-value">${rows.length} 期</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">累積投注 (${tickets.length} 注 × $${cfg.ticketCost})</div>
                    <div class="stat-value">${fmtMoney(totalCost)}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">累積獎金 (中 ${winningTickets} 注)</div>
                    <div class="stat-value accent">${fmtMoney(totalPrize)}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">淨損益 (報酬率)</div>
                    <div class="stat-value ${net >= 0 ? 'accent' : 'negative'}">${fmtMoney(net)}
                        <span class="bt-roi">${roi.toFixed(1)}%</span></div>
                </div>
            </div>
            <table class="bt-table">
                <thead>
                    <tr><th>獎項</th><th>中獎條件</th><th class="num">次數</th><th class="num">獎金</th></tr>
                </thead>
                <tbody>${tierRows}</tbody>
            </table>
            <p class="bt-footnote">假設自 ${BACKTEST_START.slice(0, 7)} 起每期照推薦號碼全數投注。其中 ${backfilledCount} 期當時未留下推薦紀錄，以相同推薦邏輯、僅用當期之前的資料回補。浮動獎項若該期無人中獎，以獨得整個獎池計算；固定獎項依官方獎金。</p>
        </div>`;
}

// ===== 大樂透 頭獎門檻策略（≥3 億 + 避開熱門號碼）=====
async function renderJackpotStrategy(url, targetId) {
    const container = document.getElementById(targetId);
    try {
        const history = await fetch(url, { cache: 'no-store' }).then(r => r.json());
        const date = Object.keys(history).sort().pop();
        const rec = history[date];
        const tickets = combCount(rec.pack_size, 6);
        const action = rec.threshold_met
            ? `包牌 ${rec.pack_size} 碼 = ${tickets} 注 (${fmtMoney(tickets * 50)})`
            : '本期不買';
        const balls = rec.numbers.map(n => `<div class="ball cold">${String(n).padStart(2, '0')}</div>`).join('');

        container.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <div class="card-title">🎯 頭獎門檻策略（≥${rec.threshold / 1e8} 億＋避開熱門）</div>
                    <div class="strategy-badge ${rec.threshold_met ? 'go' : 'wait'}">${rec.threshold_met ? '✅ 已達門檻' : '⛔ 未達門檻'}</div>
                </div>
                <div class="info-grid">
                    <div class="stat-box">
                        <div class="stat-label">預估下期頭獎</div>
                        <div class="stat-value ${rec.threshold_met ? 'accent' : ''}">${fmtMoney(rec.jackpot_est)}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">投注門檻</div>
                        <div class="stat-value">${fmtMoney(rec.threshold)}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">本期建議</div>
                        <div class="stat-value">${action}</div>
                    </div>
                </div>
                <div class="number-groups strategy-numbers">
                    <div class="number-row">
                        <div class="row-label">包牌號碼</div>
                        <div class="balls-container">${balls}</div>
                    </div>
                </div>
                <p class="bt-footnote">
                    ${rec.threshold_met
                        ? `在選號單勾選包牌，填入上面 ${rec.pack_size} 個號碼，即為全部 ${tickets} 注組合。`
                        : `未達門檻時預算保留給頭獎更大的期數；以上為參考號碼，達標時以包牌 ${rec.pack_size} 碼投注（${tickets} 注）。`}
                    年預算 ${fmtMoney(rec.annual_budget)}，平均熱門度 ${rec.popularity.toFixed(2)}（隨機選號 = 1.00，越低越不容易與人平分頭獎）。
                    2019 起回測每 $1 期望拿回約 0.80（電腦選號 0.77、每期都買 0.49）；中頭獎機率只取決於注數，選號不會提高中獎率。
                </p>
            </div>`;
    } catch (e) {
        console.error('Jackpot strategy failed:', e);
        container.innerHTML = '';
    }
}

function combCount(n, k) {
    let r = 1;
    for (let i = 0; i < k; i++) r = r * (n - i) / (i + 1);
    return Math.round(r);
}
