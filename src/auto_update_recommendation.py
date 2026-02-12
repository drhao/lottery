
import subprocess
import os
import pandas as pd
import sys

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Paths - Lotto 6/49
CSV_649 = os.path.join(BASE_DIR, 'data', 'lotto649_results.csv')
SCRAPER_649 = os.path.join(BASE_DIR, 'src', 'scrape_lotto649.py')
REC_649 = os.path.join(BASE_DIR, 'src', 'recommend_lotto649.py')

# Paths - Super Lotto (威力彩)
CSV_SUPER = os.path.join(BASE_DIR, 'data', 'super_lotto638_results.csv')
SCRAPER_SUPER = os.path.join(BASE_DIR, 'src', 'scrape_lottery.py')
REC_SUPER = os.path.join(BASE_DIR, 'src', 'recommend_next_draw.py')

def get_latest_info(csv_path, date_col='Date', period_col='Period'):
    """Returns the latest Date and Period from the CSV file."""
    if not os.path.exists(csv_path):
        return None, None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None, None
            
        latest_date = df[date_col].max()
        # Get period corresponding to latest date
        latest_period = df.loc[df[date_col] == latest_date, period_col].iloc[0]
        
        return latest_date, str(latest_period)
    except Exception as e:
        print(f"Error reading CSV {os.path.basename(csv_path)}: {e}")
        return None, None

def check_and_update(name, csv_path, scraper_path, rec_path):
    print(f"\n--- Checking {name} ---")
    
    # 1. Get current latest
    curr_date, curr_period = get_latest_info(csv_path)
    print(f"Current Latest: {curr_date} (Period: {curr_period})")
    
    # 2. Run Scraper
    print(f"Running Scraper...")
    try:
        subprocess.run(["python3", scraper_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running scraper for {name}: {e}")
        return

    # 3. Get new latest
    new_date, new_period = get_latest_info(csv_path)
    print(f"Refreshed Latest: {new_date} (Period: {new_period})")
    
    # 4. Compare
    needs_update = False
    if curr_date is None:
        print("Database was empty. Initializing...")
        needs_update = True
    elif new_date != curr_date or new_period != curr_period:
        print(f">>> NEW DRAW DETECTED! ({curr_date} -> {new_date}) <<<")
        needs_update = True
    else:
        print(">>> No new draw detected. <<<")
        
    if needs_update:
        print(f"Updating Recommendations for {name}...")
        try:
            subprocess.run(["python3", rec_path], check=True)
            print(f"SUCCESS: {name} recommendations updated.")
        except subprocess.CalledProcessError as e:
            print(f"Error running recommendation script: {e}")
    else:
        print(f"{name} recommendations are up to date.")

def main():
    print("========================================")
    print("   Lottery Auto-Update System")
    print("========================================")
    
    # Check Lotto 6/49
    check_and_update("Lotto 6/49", CSV_649, SCRAPER_649, REC_649)
    
    # Check Super Lotto
    check_and_update("Super Lotto (威力彩)", CSV_SUPER, SCRAPER_SUPER, REC_SUPER)
    
    print("\n========================================")
    print("   All Checks Completed")
    print("========================================")

if __name__ == "__main__":
    main()
