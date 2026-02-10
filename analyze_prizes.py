import pandas as pd

def calculate_average_prizes(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # Calculate start and end dates from the Date column
        start_date = df['Date'].min()
        end_date = df['Date'].max()
        
        # Filter for First Prize winners (Per Winner > 0)
        first_prize_winners = df[df['First_Prize_Per_Winner'] > 0]['First_Prize_Per_Winner']
        avg_first_prize = first_prize_winners.mean()
        count_first_prize = first_prize_winners.count()

        # Filter for Second Prize winners (Per Winner > 0)
        second_prize_winners = df[df['Second_Prize_Per_Winner'] > 0]['Second_Prize_Per_Winner']
        avg_second_prize = second_prize_winners.mean()
        count_second_prize = second_prize_winners.count()

        print(f"Data Date Range: {start_date} to {end_date}")
        print(f"Total Draws: {len(df)}")
        print("-" * 30)
        
        if count_first_prize > 0:
            print(f"First Prize (Jackpot):")
            print(f"  Number of Winning Draws: {count_first_prize}")
            print(f"  Average Prize Per Winner: {avg_first_prize:,.2f}")
        else:
            print("First Prize (Jackpot): No winners found in the dataset.")
            
        print("-" * 30)

        if count_second_prize > 0:
            print(f"Second Prize:")
            print(f"  Number of Winning Draws: {count_second_prize}")
            print(f"  Average Prize Per Winner: {avg_second_prize:,.2f}")
        else:
            print("Second Prize: No winners found in the dataset.")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    calculate_average_prizes("super_lotto638_results.csv")
