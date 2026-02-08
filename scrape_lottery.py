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

        results = data["content"]["superLotto638Res"]
        total_size = data["content"]["totalSize"]
        print(f"Total records found: {total_size}")
        print(f"Records fetched: {len(results)}")

        csv_file = "super_lotto638_results.csv"
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Period", "Date", "Numbers", "Special_Number"])

            for item in results:
                period = item.get("period")
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
                # In 'drawNumberAppear', it seems the last number is likely the second section number, 
                # but 'drawNumberSize' might sort them all mixed together if it's just a raw sort.
                # However, usually for these APIs, they separate them or we have to imply it.
                # Let's inspect the 'drawNumberAppear' list. It usually contains 7 numbers.
                # 6 from first section, 1 from second section.
                
                # Careful: The browser inspection said "second section number is the last element".
                # Let's trust 'drawNumberAppear' for the second section number being the last one.
                # And for the first section numbers, we can take the first 6 elements and sort them ourselves 
                # to be consistent with how lotteries usually display results (sorted).
                
                draw_numbers = item.get("drawNumberAppear", [])
                
                if len(draw_numbers) >= 7:
                    special_number = draw_numbers[-1]
                    primary_numbers = sorted(draw_numbers[:-1])
                    
                    # Convert to string for CSV
                    numbers_str = " ".join(map(str, primary_numbers))
                    
                    writer.writerow([period, formatted_date, numbers_str, special_number])
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
