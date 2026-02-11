import requests
import json
import csv
from datetime import datetime

def scrape_super_lotto_638():
    url = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/SuperLotto638Result"
    params = {
        "period": "",
        "month": "2008-01",
        "endMonth": "2026-02",
        "pageNum": "1",
        "pageSize": "5000"  # Request a large page size to get all data at once
    }

    try:
        print(f"Fetching data from {url}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "content" not in data or "superLotto638Res" not in data["content"]:
            print("Error: Unexpected API response format.")
            return
            
        import os
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_file = os.path.join(BASE_DIR, 'data', 'super_lotto638_results.csv')

        results = data["content"]["superLotto638Res"]
        total_size = data["content"]["totalSize"]
        print(f"Total records found: {total_size}")
        print(f"Records fetched: {len(results)}")

        # csv_file is defined above
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Period", "Date", "Numbers", "Special_Number", 
                "First_Prize_Rollover", "First_Prize_Total", "First_Prize_Per_Winner", 
                "Second_Prize_Rollover", "Second_Prize_Total", "Second_Prize_Per_Winner"
            ])

            for item in results:
                period = item.get("period")
                # Extract prize info
                first_prize_data = item.get("super638JackpotAssign", {})
                second_prize_data = item.get("super638SecondAssign", {})
                
                # First Prize
                first_prize_last = first_prize_data.get("lastPrize", 0)
                first_prize_current = first_prize_data.get("prize", 0)
                first_prize_total = first_prize_last + first_prize_current
                first_prize_per_winner = first_prize_data.get("perPrize", 0)
                
                # Second Prize
                second_prize_last = second_prize_data.get("lastPrize", 0)
                second_prize_current = second_prize_data.get("prize", 0)
                second_prize_total = second_prize_last + second_prize_current
                second_prize_per_winner = second_prize_data.get("perPrize", 0)

                # Format date to YYYY-MM-DD
                raw_date = item.get("lotteryDate")
                if raw_date:
                     try:
                        date_obj = datetime.fromisoformat(raw_date)
                        formatted_date = date_obj.strftime("%Y-%m-%d")
                     except ValueError:
                        formatted_date = raw_date
                else:
                    formatted_date = ""

                # The API returns 'drawNumberAppear' (draw order) and 'drawNumberSize' (sorted).
                # Super Lotto 638 has 6 numbers in the first section + 1 number in the second section.
                draw_numbers = item.get("drawNumberAppear", [])
                
                if len(draw_numbers) >= 7:
                    special_number = draw_numbers[-1]
                    primary_numbers = sorted(draw_numbers[:-1])
                    
                    # Convert to string for CSV
                    numbers_str = " ".join(map(str, primary_numbers))
                    
                    writer.writerow([
                        period, formatted_date, numbers_str, special_number, 
                        first_prize_last, first_prize_total, first_prize_per_winner,
                        second_prize_last, second_prize_total, second_prize_per_winner
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
    scrape_super_lotto_638()
