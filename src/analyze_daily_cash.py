import pandas as pd
import numpy as np
import random
from collections import Counter
import os

class DailyCashAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = self.load_data()
        self.prizes = {
            5: 8000000,
            4: 20000,
            3: 300,
            2: 50,
            1: 0,
            0: 0
        }
        self.cost_per_bet = 50

    def load_data(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(BASE_DIR, self.data_path)
        df = pd.read_csv(full_path)
        df['Date'] = pd.to_datetime(df['Date'])
        # Sort by date ascending for simulation
        df = df.sort_values('Date').reset_index(drop=True)
        # Parse numbers
        df['Numbers_List'] = df['Numbers'].apply(lambda x: [int(n) for n in str(x).split()])
        return df

    def calculate_winnings(self, bet_numbers, draw_numbers):
        matches = len(set(bet_numbers) & set(draw_numbers))
        return self.prizes.get(matches, 0), matches

    def run_simulation(self, strategy_func, initial_window=50, test_draws=None):
        """
        Runs a simulation of a betting strategy.
        strategy_func: Function that takes (history_df, current_date) and returns list of 5 numbers.
        initial_window: Number of past draws required for strategy to start.
        test_draws: Optional limit on number of draws to test.
        """
        results = []
        total_cost = 0
        total_winnings = 0
        
        # Start simulation after initial window
        start_idx = initial_window
        if start_idx >= len(self.df):
            print("Error: Not enough data for initial window.")
            return

        test_data = self.df.iloc[start_idx:]
        if test_draws:
            test_data = test_data.head(test_draws)

        print(f"Running simulation for {len(test_data)} draws using {strategy_func.__name__}...")

        for idx, row in test_data.iterrows():
            # Get history up to this draw (exclusive)
            history = self.df.iloc[:idx]
            
            # Predict
            bet = strategy_func(history)
            if not bet or len(bet) != 5:
                continue # Skip if strategy can't make a valid bet
                
            # Check result
            actual_numbers = row['Numbers_List']
            prize, matches = self.calculate_winnings(bet, actual_numbers)
            
            total_cost += self.cost_per_bet
            total_winnings += prize
            
            results.append({
                'Date': row['Date'],
                'Bet': bet,
                'Actual': actual_numbers,
                'Matches': matches,
                'Prize': prize
            })

        # Calculate Stats
        total_bets = len(results)
        wins = sum(1 for r in results if r['Prize'] > 0)
        win_rate = (wins / total_bets) * 100 if total_bets > 0 else 0
        roi = ((total_winnings - total_cost) / total_cost) * 100 if total_cost > 0 else 0

        print(f"Results for {strategy_func.__name__}:")
        print(f"Total Bets: {total_bets}")
        print(f"Total Cost: {total_cost}")
        print(f"Total Winnings: {total_winnings}")
        print(f"Net Profit: {total_winnings - total_cost}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"ROI: {roi:.2f}%")
        print("-" * 30)
        
        return results

    # --- Strategies ---

    @staticmethod
    def strategy_random(history):
        return random.sample(range(1, 40), 5)

    @staticmethod
    def strategy_hot_numbers(history, n_draws=50):
        # Pick 5 numbers that appeared most frequently in the last n_draws
        recent_history = history.tail(n_draws)
        all_numbers = [num for sublist in recent_history['Numbers_List'] for num in sublist]
        counts = Counter(all_numbers)
        # Most common returns tuples (num, count), we want just num.
        # If there are ties, it picks arbitrarily among ties.
        most_common = counts.most_common(5)
        # If not enough numbers (unlikely), fill with random? Or return fewer?
        # Assuming we can always find 5.
        
        picks = [num for num, count in most_common]
        while len(picks) < 5:
            # Fill remaining with random unique
            new_pick = random.randint(1, 39)
            if new_pick not in picks:
                picks.append(new_pick)
        return sorted(picks)

    @staticmethod
    def strategy_cold_numbers(history, n_draws=100):
        # Pick numbers that appeared LEAST frequently in last n_draws
        recent_history = history.tail(n_draws)
        all_numbers = [num for sublist in recent_history['Numbers_List'] for num in sublist]
        counts = Counter(all_numbers)
        
        # All possible numbers
        all_possible = set(range(1, 40))
        
        # Numbers not in recent history have count 0
        present_numbers = set(counts.keys())
        missing_numbers = list(all_possible - present_numbers)
        
        if len(missing_numbers) >= 5:
            return sorted(random.sample(missing_numbers, 5))
        
        # If we need more, take the ones with lowest counts
        picks = missing_numbers
        least_common = counts.most_common()[:-6:-1] # Get last 5, reversed
        
        for num, count in least_common:
            if len(picks) >= 5:
                break
            if num not in picks:
                picks.append(num)
                
        return sorted(picks)

    @staticmethod
    def strategy_repeater(history):
        # Pick 5 numbers from the previous draw
        # If previous draw had < 5 numbers (error), return random
        last_draw = history.iloc[-1]['Numbers_List']
        if len(last_draw) == 5:
             # Daily Cash 539 numbers are unique in a draw
             return sorted(last_draw)
        return random.sample(range(1, 40), 5)

    @staticmethod
    def strategy_balanced_odd_even(history):
        # Pick 3 odd and 2 even, or 2 odd and 3 even (randomly decide)
        odds = [n for n in range(1, 40) if n % 2 != 0]
        evens = [n for n in range(1, 40) if n % 2 == 0]
        
        if random.random() < 0.5:
            # 3 odd, 2 even
            picks = random.sample(odds, 3) + random.sample(evens, 2)
        else:
            # 2 odd, 3 even
            picks = random.sample(odds, 2) + random.sample(evens, 3)
            
        return sorted(picks)

if __name__ == "__main__":
    analyzer = DailyCashAnalyzer('data/daily_cash_results.csv')
    
    print("Strategy: Random")
    analyzer.run_simulation(analyzer.strategy_random)
    print("Strategy: Hot Numbers (Last 50)")
    analyzer.run_simulation(analyzer.strategy_hot_numbers)
    print("Strategy: Cold Numbers (Last 100)")
    analyzer.run_simulation(analyzer.strategy_cold_numbers)
    print("Strategy: Repeater (Last Draw)")
    analyzer.run_simulation(analyzer.strategy_repeater)
    print("Strategy: Balanced Odd/Even")
    analyzer.run_simulation(analyzer.strategy_balanced_odd_even)
