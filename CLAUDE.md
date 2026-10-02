# CLAUDE.md

## Project Overview

Taiwan Lottery analysis and recommendation system. Scrapes historical lottery data, performs statistical/ML analysis, and generates number-picking recommendations for three lottery types:

- **Super Lotto 638 (威力彩)** — 6 numbers from 1-38 + 1 special from 1-8
- **Lotto 6/49 (大樂透)** — 6 numbers from 1-49 + 1 special
- **Daily Cash 539 (樂透539)** — 5 numbers from 1-39

## Repository Structure

```
src/                          # All Python source code (~30 modules)
  auto_update_recommendation.py  # Main orchestrator (CI/CD entry point)
  scrape_lottery.py              # Super Lotto 638 scraper
  scrape_lotto649.py             # Lotto 6/49 scraper
  scrape_daily_cash.py           # Daily Cash 539 scraper
  analyze_lottery.py             # Frequency/overdue/pattern analysis
  analyze_prizes.py              # Prize structure analysis
  recommend_next_draw.py         # Super Lotto 638 recommendations
  recommend_lotto649.py          # Lotto 6/49 recommendations
  ml_lottery.py                  # ML predictions (XGBoost, RandomForest, MLP)
  replay_*.py                    # Strategy backtesting (10+ variants)
  optimize_*.py                  # Hyperparameter tuning
  generate_dashboard.py          # HTML dashboard generation
  test_diversity.py              # Weighted sampling tests
data/                         # CSV lottery results from API
  super_lotto638_results.csv
  lotto649_results.csv
  daily_cash_results.csv
output/                       # Generated dashboards, backtest CSVs, metrics
recommendation_history.json   # Historical Super Lotto picks
recommendation_output.txt     # Latest Super Lotto recommendation
recommendation_lotto649_output.txt  # Latest Lotto 6/49 recommendation
```

## Tech Stack

- **Language:** Python 3.10
- **Dependencies:** pandas, numpy, requests, xgboost, scikit-learn
- **CI/CD:** GitHub Actions (`.github/workflows/update_lottery.yml`)

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full daily update (scrape + analyze + recommend)
python src/auto_update_recommendation.py

# Run individual scrapers
python src/scrape_lottery.py
python src/scrape_lotto649.py
python src/scrape_daily_cash.py

# Generate recommendations
python src/recommend_next_draw.py
python src/recommend_lotto649.py

# Run ML predictions
python src/ml_lottery.py

# Run backtest/replay
python src/replay_threshold_strategy.py

# Run sampling diversity tests
python src/test_diversity.py

# Generate HTML dashboards
python src/generate_dashboard.py
```

## CI/CD

GitHub Actions runs daily at UTC 15:00 (Taiwan 23:00):
1. Scrapes latest results from `api.taiwanlottery.com`
2. Runs recommendation engine
3. Auto-commits updated `data/*.csv`, `recommendation_*.json`, and `recommendation_*.txt`

Manual trigger available via `workflow_dispatch`.

## Key Concepts

- **Jackpot threshold:** Recommendations trigger when jackpot exceeds 800M TWD (configurable)
- **Weighted sampling:** Efraimidis-Spirakis algorithm for sampling without replacement
- **Diversity penalty:** Soft penalty to reduce number overlap across multiple tickets
- **Hot/cold analysis:** Recent window vs. all-time frequency tracking
- **Rollover tracking:** Consecutive draws without winners drives ticket count scaling
- **Ticket scaling:** Dynamic ticket count based on jackpot size and rollover streak

## Code Conventions

- **Language:** Python with snake_case naming
- **Comments/output:** Mix of Chinese (Traditional) and English
- **Script naming:** Prefixed by function: `scrape_`, `analyze_`, `recommend_`, `replay_`, `optimize_`
- **Version suffixes:** `_v2`, `_v3`, `_10k` (sim count), `_cap15` (cap limit)
- **No linter/formatter configured** — no `.flake8`, `pyproject.toml`, or pre-commit hooks
- **No formal test framework** — `test_diversity.py` uses print-based verification

## Git Conventions

- **Commit messages:** Use conventional prefixes (`feat:`, `fix:`) or emoji (🤖 for automated)
- **Automated commits:** `🤖 Auto-update: 最新開獎結果與推薦號碼`
- **Branch naming:** `feature/`, `claude/` prefixes
- **Primary branch:** `main`

## Data Sources

All data scraped from Taiwan Lottery official API (`api.taiwanlottery.com`). CSV files in `data/` contain full historical draw results including period numbers, dates, winning numbers, special numbers, and prize breakdowns.
