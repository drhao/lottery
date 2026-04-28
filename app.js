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
