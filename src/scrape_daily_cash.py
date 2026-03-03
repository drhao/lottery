import requests
import json
import csv
import os
from datetime import datetime

def scrape_daily_cash():
    url = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/Daily539Result"
    
    # Get current month dynamically to ensure we get up-to-date data
    current_date = datetime.now()
    end_month = current_date.strftime("%Y-%m")
    
    params = {
        "period": "",
        "month": "2007-01", 
        "endMonth": end_month,
        "pageNum": "1",
        "pageSize": "10000" # Request a large page size to attempt to get all data
    }

    try:
        print(f"Fetching data from {url}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "content" not in data or "daily539Res" not in data["content"]:
            print("Error: Unexpected API response format.")
            # Print keys to help debugging if format changes
            if "content" in data:
                print(f"Available keys in content: {data['content'].keys()}")
            else:
                print(f"Available keys in root: {data.keys()}")
            return
            
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_file = os.path.join(BASE_DIR, 'data', 'daily_cash_results.csv')

        results = data["content"]["daily539Res"]
        total_size = data["content"]["totalSize"]
        print(f"Total records found on server: {total_size}")
        print(f"Records fetched in this request: {len(results)}")
        
        if len(results) < total_size:
            print("Warning: Fetching fewer records than total available. Pagination might be needed if 10000 limit is hit.")

        # Ensure data directory exists
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Period", "Date", "Numbers", 
                "Jackpot_WinnerCount", "Jackpot_Prize",
                "Second_WinnerCount", "Second_Prize",
                "Third_WinnerCount", "Third_Prize",
                "Fourth_WinnerCount", "Fourth_Prize"
            ])

            for item in results:
                period = item.get("period")
                
                # Extract prize info
                def get_prize_info(key):
                    p_data = item.get(key, {})
                    return p_data.get("winnerCount", 0), p_data.get("perPrize", 0)

                jp_count, jp_prize = get_prize_info("d539JackpotAssign")
                sec_count, sec_prize = get_prize_info("d539SecondAssign")
                thi_count, thi_prize = get_prize_info("d539ThirdAssign")
                fou_count, fou_prize = get_prize_info("d539FourthAssign")

                # Format date to YYYY-MM-DD
                raw_date = item.get("lotteryDate")
                formatted_date = ""
                if raw_date:
                     try:
                        date_obj = datetime.fromisoformat(raw_date)
                        formatted_date = date_obj.strftime("%Y-%m-%d")
                     except ValueError:
                        formatted_date = raw_date

                # Daily Cash 539 has 5 numbers. 
                # 'drawNumberSize' gives them sorted.
                draw_numbers = item.get("drawNumberSize", [])
                
                if len(draw_numbers) == 5:
                    numbers_str = " ".join(map(str, draw_numbers))
                    
                    writer.writerow([
                        period, formatted_date, numbers_str,
                        jp_count, jp_prize,
                        sec_count, sec_prize,
                        thi_count, thi_prize,
                        fou_count, fou_prize
                    ])
                else:
                    print(f"Warning: Unexpected number count for period {period}: {draw_numbers}")

        print(f"Successfully saved {len(results)} records to {csv_file}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError:
        print("Failed to decode JSON response.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    scrape_daily_cash()
