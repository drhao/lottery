
import requests
import json
import csv
import os
from datetime import datetime

def scrape_lotto649():
    url = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/Lotto649Result"
    params = {
        "period": "",
        "month": "2004-01", 
        "endMonth": datetime.now().strftime("%Y-%m"),
        "pageNum": "1",
        "pageSize": "5000"
    }

    try:
        print(f"Fetching data from {url}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "content" not in data or "lotto649Res" not in data["content"]:
            print("Error: Unexpected API response format.")
            return
            
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_file = os.path.join(BASE_DIR, 'data', 'lotto649_results.csv')

        results = data["content"]["lotto649Res"]
        total_size = data["content"]["totalSize"]
        print(f"Total records found: {total_size}")
        print(f"Records fetched: {len(results)}")

        # Ensure data directory exists
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Period", "Date", "Numbers", "Special_Number", 
                "Jackpot_Rollover", "Jackpot_Total", "Jackpot_Per_Winner", 
                "Second_Prize_Total", "Second_Prize_Per_Winner",
                "Third_Prize_Total", "Third_Prize_Per_Winner",
                "Fourth_Prize_Total", "Fourth_Prize_Per_Winner",
                "Fifth_Prize_Total", "Fifth_Prize_Per_Winner",
                "Sixth_Prize_Total", "Sixth_Prize_Per_Winner",
                "Seventh_Prize_Total", "Seventh_Prize_Per_Winner",
                "Normal_Prize_Total", "Normal_Prize_Per_Winner"
            ])

            for item in results:
                period = item.get("period")
                
                # Extract prize info
                # Note: The API naming convention might differ slightly from super lotto.
                # Inspecting the curl output from earlier would help, but standardizing naming based on common patterns.
                # Based on previous curl output:
                # jackpotAssign, secondAssign, thirdAssign, fourthAssign, fifthAssign, sixthAssign, seventhAssign, normalAssign
                
                def get_prize_info(key):
                    p_data = item.get(key, {})
                    return p_data.get("prize", 0), p_data.get("perPrize", 0), p_data.get("lastPrize", 0)

                jp_total, jp_per, jp_last = get_prize_info("jackpotAssign")
                sec_total, sec_per, _ = get_prize_info("secondAssign")
                thi_total, thi_per, _ = get_prize_info("thirdAssign")
                fou_total, fou_per, _ = get_prize_info("fourthAssign")
                fif_total, fif_per, _ = get_prize_info("fifthAssign")
                six_total, six_per, _ = get_prize_info("sixthAssign")
                sev_total, sev_per, _ = get_prize_info("seventhAssign")
                nor_total, nor_per, _ = get_prize_info("normalAssign")

                # Format date to YYYY-MM-DD
                raw_date = item.get("lotteryDate")
                formatted_date = ""
                if raw_date:
                     try:
                        date_obj = datetime.fromisoformat(raw_date)
                        formatted_date = date_obj.strftime("%Y-%m-%d")
                     except ValueError:
                        formatted_date = raw_date

                # Lotto 6/49 has 6 numbers + 1 special number
                draw_numbers = item.get("drawNumberAppear", [])
                
                if len(draw_numbers) >= 7:
                    # In drawNumberAppear, the specific order isn't guaranteed to be [Main... Special] or [Special ... Main]?
                    # Actually, usually 'drawNumberAppear' is the order drawn.
                    # 'drawNumberSize' is sorted order.
                    # For Lotto 6/49, usually 6 main + 1 special.
                    # Let's assume the last one is special if it matches the count.
                    # BUT, usually the API returns 6 main numbers in 'drawNumberSize' and Special is separate?
                    # Let's check the curl output from Step 15 carefully.
                    # "drawNumberSize":[8,15,21,27,46,48,31],"drawNumberAppear":[21,8,48,46,27,15,31]
                    # There are 7 numbers.
                    # Standard Lotto 6/49: 6 numbers + 1 special.
                    # It seems the last number in the list is the special number.
                    # Let's verify with sorted list: [8,15,21,27,46,48] and [31]??
                    # Wait, 31 is not in the sorted part if we take first 6?
                    # Sorted: 8, 15, 21, 27, 46, 48, 31 (31 is in middle).
                    # 'drawNumberSize' is just the drawNumberAppear sorted.
                    # The special number is usually the LAST one drawn (in drawNumberAppear).
                    # Let's assume drawNumberAppear[-1] is special number.
                    
                    special_number = draw_numbers[-1]
                    primary_numbers = sorted(draw_numbers[:-1])
                    
                    numbers_str = " ".join(map(str, primary_numbers))
                    
                    writer.writerow([
                        period, formatted_date, numbers_str, special_number, 
                        jp_last, jp_total, jp_per,
                        sec_total, sec_per,
                        thi_total, thi_per,
                        fou_total, fou_per,
                        fif_total, fif_per,
                        six_total, six_per,
                        sev_total, sev_per,
                        nor_total, nor_per
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
    scrape_lotto649()
