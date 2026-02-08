import csv
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
from xgboost import XGBRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(filename):
    data = []
    try:
        with open(filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    # Reverse to Chronological: Index 0 = Oldest
    return list(reversed(data))

def calculate_ranks(history_numbers, check_numbers=None):
    """
    Calculates ranks for numbers 1-38 based on history_numbers.
    Returns:
        rank_map: dict {num: rank}
        avg_rank: float (if check_numbers provided)
    """
    freq = Counter(history_numbers)
    # Sort 1-38 by frequency desc
    ranked_nums = sorted(range(1, 39), key=lambda x: freq[x], reverse=True)
    rank_map = {num: r+1 for r, num in enumerate(ranked_nums)}
    
    avg_rank = 0
    if check_numbers:
        ranks = [rank_map.get(n, 38) for n in check_numbers]
        avg_rank = sum(ranks) / len(ranks) if ranks else 38
        
    return rank_map, avg_rank

def prepare_features(data):
    """
    Generates dataset for training.
    X: [
        [Past 1_Rank1..7, Past 2_Rank1..7 ... Past 10_Rank1..7, Current_Rank1..38]
    ]
    Y: [Target_Avg_Rank] (Use simplified target for regression: Predict the 'Temperature' of the next draw)
    
    Wait, user wants to predict the *Winning Numbers*, then eval the gap.
    Predicting 6 exact numbers is extremely hard. 
    Predicting the *Rank Distribution* or *Average Rank* is easier.
    
    User request: "Forecast this period's winning numbers... eval metric is Rank Gap".
    We will train a Multi-Output Regressor to predict the "Likelihood Score" for each of the 38 numbers.
    Top 6 scores = Predicted Numbers.
    """
    print("Preparing features...")
    
    # Pre-calculate history buffer
    X_samples = []
    Y_samples = []
    
    # We need to maintain a running history of ALL numbers to calculate ranks efficiently
    # But for correctness, we'll just slice the list. It's fast enough for 1800 items.
    
    # Convert all numbers to list of lists for easier access
    all_draws_nums = []
    all_draws_special = []
    
    for row in data:
        try:
            n = list(map(int, row["Numbers"].split()))
            s = int(row["Special_Number"])
            all_draws_nums.append(n)
            all_draws_special.append(s)
        except:
            all_draws_nums.append([])
            all_draws_special.append(0)

    # Start from 500
    # We need 10 past draws for features.
    
    records = []
    
    for i in range(500, len(data)):
        # 1. State at time i (Before draw i happened)
        # History: 0 to i-1
        history_pool_main = [n for sublist in all_draws_nums[:i] for n in sublist]
        rank_map_main, _ = calculate_ranks(history_pool_main)
        
        history_pool_spec = all_draws_special[:i]
        # Calculate rank for special (1-8)
        freq_s = Counter(history_pool_spec)
        ranked_s = sorted(range(1, 9), key=lambda x: freq_s[x], reverse=True)
        rank_map_spec = {num: r+1 for r, num in enumerate(ranked_s)}
        
    # Feature 1: Past 10 draws (i-10 to i-1)
        feat_seq_main = []
        feat_seq_spec = []
        
        for offset in range(1, 11):
            past_idx = i - offset
            if past_idx < 0: break
            
            # Main Numbers Feature
            p_nums = all_draws_nums[past_idx] 
            p_ranks = [rank_map_main.get(n, 38) for n in p_nums]
            feat_seq_main.extend(p_ranks)
            
            # Special Number Feature
            s_num = all_draws_special[past_idx]
            s_rank = rank_map_spec.get(s_num, 8)
            feat_seq_spec.append(s_rank)
            
        # Feature 2: Current State (Frequencies)
        # Main (1-38)
        counts_main = Counter(history_pool_main)
        max_c_main = max(counts_main.values()) if counts_main else 1
        feat_state_main = [counts_main[n]/max_c_main for n in range(1, 39)]
        
        # Special (1-8)
        counts_spec = Counter(history_pool_spec)
        max_c_spec = max(counts_spec.values()) if counts_spec else 1
        feat_state_spec = [counts_spec[n]/max_c_spec for n in range(1, 9)]
        
        # Combine Features
        # Main: 10*6 + 38 = 98 features
        X_main = feat_seq_main + feat_state_main
        
        # Special: 10*1 + 8 = 18 features
        X_spec = feat_seq_spec + feat_state_spec
        
        # Targets
        # Main (Multi-hot 38)
        Y_main = [0] * 38
        for n in all_draws_nums[i]:
            if 1 <= n <= 38:
                Y_main[n-1] = 1
                
        # Special (One-hot 8)
        Y_spec = [0] * 8
        s_target = all_draws_special[i]
        if 1 <= s_target <= 8:
            Y_spec[s_target-1] = 1
                
        records.append({
            "idx": i,
            "X_main": X_main,
            "Y_main": Y_main,
            "X_spec": X_spec,
            "Y_spec": Y_spec,
            "winning_nums": all_draws_nums[i],
            "winning_spec": all_draws_special[i],
            "rank_map_main": rank_map_main,
            "rank_map_spec": rank_map_spec
        })
        
    return records

def train_and_predict(records):
    results = []
    
    # Initialize Main Models
    rf_main = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    dl_main = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    xgb_base_main = XGBRegressor(n_estimators=50, learning_rate=0.1, n_jobs=-1, random_state=42)
    
    # Initialize Special Models
    rf_spec = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    dl_spec = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
    xgb_base_spec = XGBRegressor(n_estimators=50, learning_rate=0.1, n_jobs=-1, random_state=42)
    
    # Initialize Ensemble Models (Stacking XGB)
    xgb_main = XGBRegressor(n_estimators=30, learning_rate=0.1, n_jobs=-1, random_state=42)
    xgb_spec = XGBRegressor(n_estimators=30, learning_rate=0.1, n_jobs=-1, random_state=42)
    
    # Meta History buffers
    X_meta_main_hist = []
    Y_meta_main_hist = []
    X_meta_spec_hist = []
    Y_meta_spec_hist = []
    
    xgb_main_fitted = False
    xgb_spec_fitted = False
    
    # Gap History (for use as features)
    gap_hist_rf_m = []
    gap_hist_dl_m = []
    gap_hist_xgb_m = []
    
    gap_hist_rf_s = []
    gap_hist_dl_s = []
    gap_hist_xgb_s = []
    
    # Sliding window settings
    train_window = 300 
    retrain_interval = 50 
    
    print(f"Starting backtest from index {records[0]['idx']} to {records[-1]['idx']}...")
    
    for k, rec in enumerate(records):
        curr_idx = rec['idx']
        
        # Retrain
        if k % retrain_interval == 0 and k > 0:
            train_start = max(0, k - train_window)
            train_slice = records[train_start : k]
            
            # Train Main
            X_train_m = [r['X_main'] for r in train_slice]
            Y_train_m = [r['Y_main'] for r in train_slice]
            if len(X_train_m) > 10:
                rf_main.fit(X_train_m, Y_train_m)
                dl_main.fit(X_train_m, Y_train_m)
                xgb_base_main.fit(X_train_m, Y_train_m)
            
            # Train Special
            X_train_s = [r['X_spec'] for r in train_slice]
            Y_train_s = [r['Y_spec'] for r in train_slice]
            if len(X_train_s) > 10:
                rf_spec.fit(X_train_s, Y_train_s)
                dl_spec.fit(X_train_s, Y_train_s)
                xgb_base_spec.fit(X_train_s, Y_train_s)
                
            # Train Ensemble (XGB Stacking)
            if len(X_meta_main_hist) >= 50:
                xgb_main.fit(X_meta_main_hist, Y_meta_main_hist)
                xgb_main_fitted = True
            if len(X_meta_spec_hist) >= 50:
                xgb_spec.fit(X_meta_spec_hist, Y_meta_spec_hist)
                xgb_spec_fitted = True
        
        # Predict
        if k < 50:
            continue
            
        
        
        # --- Main Prediction ---
        X_test_m = [rec['X_main']]
        
        # RF Main
        pred_rf_m = rf_main.predict(X_test_m)[0]
        top_6_rf_m = sorted(range(len(pred_rf_m)), key=lambda i: pred_rf_m[i], reverse=True)[:6]
        res_rf_m = [x+1 for x in top_6_rf_m]
        
        # DL Main
        pred_dl_m = dl_main.predict(X_test_m)[0]
        top_6_dl_m = sorted(range(len(pred_dl_m)), key=lambda i: pred_dl_m[i], reverse=True)[:6]
        res_dl_m = [x+1 for x in top_6_dl_m]
        
        # XGB Base Main
        pred_xgb_m = xgb_base_main.predict(X_test_m)[0]
        top_6_xgb_m = sorted(range(len(pred_xgb_m)), key=lambda i: pred_xgb_m[i], reverse=True)[:6]
        res_xgb_m = [x+1 for x in top_6_xgb_m]
        
        # --- Special Prediction ---
        X_test_s = [rec['X_spec']]
        
        # RF Special (Pick top 1)
        pred_rf_s = rf_spec.predict(X_test_s)[0]
        top_1_rf_s = sorted(range(len(pred_rf_s)), key=lambda i: pred_rf_s[i], reverse=True)[:1]
        res_rf_s = top_1_rf_s[0] + 1
        
        # DL Special (Pick top 1)
        pred_dl_s = dl_spec.predict(X_test_s)[0]
        top_1_dl_s = sorted(range(len(pred_dl_s)), key=lambda i: pred_dl_s[i], reverse=True)[:1]
        res_dl_s = top_1_dl_s[0] + 1
        
        # XGB Base Special
        pred_xgb_s = xgb_base_spec.predict(X_test_s)[0]
        top_1_xgb_s = sorted(range(len(pred_xgb_s)), key=lambda i: pred_xgb_s[i], reverse=True)[:1]
        res_xgb_s = top_1_xgb_s[0] + 1
        
        # --- Ensemble Prediction (Stacking) ---
        # 1. Prepare Meta-Features
        # Base Features: [RF(38) + DL(38) + XGB(38)] -> 114 features
        base_meta_m = list(pred_rf_m) + list(pred_dl_m) + list(pred_xgb_m)
        base_meta_s = list(pred_rf_s) + list(pred_dl_s) + list(pred_xgb_s)
        
        # Context Features: Past 10 period gaps (errors)
        # Pad with average or 0 if < 10
        def get_past_10_gaps(hist_list):
            if not hist_list: return [0.0]*10
            return (hist_list[-10:] + [0.0]*10)[:10]

        context_m = get_past_10_gaps(gap_hist_rf_m) + get_past_10_gaps(gap_hist_dl_m) + get_past_10_gaps(gap_hist_xgb_m)
        context_s = get_past_10_gaps(gap_hist_rf_s) + get_past_10_gaps(gap_hist_dl_s) + get_past_10_gaps(gap_hist_xgb_s)
        
        meta_feat_m = base_meta_m + context_m
        meta_feat_s = base_meta_s + context_s
        
        res_ens_m = []
        res_ens_s = 0
        
        # Predict using Ensemble (if trained)
        if xgb_main_fitted and xgb_spec_fitted: 
             # Main
             pred_ens_scores_m = xgb_main.predict([meta_feat_m])[0]
             
             # Filter: Only select from base model predictions
             # Candidate Set = Union of top 6 from RF, DL, XGB
             candidates_m = set(res_rf_m) | set(res_dl_m) | set(res_xgb_m)
             
             # Create filtered scores pairs (index, score)
             # Note: pred_ens_scores_m is for 1..38
             filtered_scores = []
             for i in range(len(pred_ens_scores_m)):
                 num = i + 1
                 if num in candidates_m:
                     filtered_scores.append((num, pred_ens_scores_m[i]))
                 else:
                     filtered_scores.append((num, -9999.0)) # Exclude
            
             # Sort by score desc
             filtered_scores.sort(key=lambda x: x[1], reverse=True)
             res_ens_m = [x[0] for x in filtered_scores[:6]]
             
             # Special
             pred_ens_scores_s = xgb_spec.predict([meta_feat_s])[0]
             
             # Candidate Set for Special
             candidates_s = {res_rf_s, res_dl_s, res_xgb_s}
             
             filtered_scores_s = []
             for i in range(len(pred_ens_scores_s)):
                 num = i + 1
                 if num in candidates_s:
                     filtered_scores_s.append((num, pred_ens_scores_s[i]))
                 else:
                     filtered_scores_s.append((num, -9999.0))
            
             filtered_scores_s.sort(key=lambda x: x[1], reverse=True)
             res_ens_s = filtered_scores_s[0][0]
        else:
             # Fallback to RF if not enough data
             res_ens_m = res_rf_m
             res_ens_s = res_rf_s

        # 2. Store Meta-Data for FUTURE Training
        X_meta_main_hist.append(meta_feat_m)
        Y_meta_main_hist.append(rec['Y_main'])
        
        X_meta_spec_hist.append(meta_feat_s)
        Y_meta_spec_hist.append(rec['Y_spec'])

        # Evaluation
        actual_main = rec['winning_nums']
        actual_spec = rec['winning_spec']
        
        rm_main = rec['rank_map_main']
        rm_spec = rec['rank_map_spec']
        
        def get_avg_rank_main(nums):
            rs = [rm_main.get(n, 38) for n in nums]
            return sum(rs)/len(rs) if rs else 0
            
        def get_rank_spec(n):
            return rm_spec.get(n, 8)
            
        # Main Scores
        val_act_m = get_avg_rank_main(actual_main)
        val_rf_m = get_avg_rank_main(res_rf_m)
        val_dl_m = get_avg_rank_main(res_dl_m)
        val_xgb_m = get_avg_rank_main(res_xgb_m)
        val_ens_m = get_avg_rank_main(res_ens_m)
        
        gap_rf_m = abs(val_rf_m - val_act_m)
        gap_dl_m = abs(val_dl_m - val_act_m)
        gap_xgb_m = abs(val_xgb_m - val_act_m)
        gap_ens_m = abs(val_ens_m - val_act_m)
        
        # Update Gap History
        gap_hist_rf_m.append(gap_rf_m)
        gap_hist_dl_m.append(gap_dl_m)
        gap_hist_xgb_m.append(gap_xgb_m)
        
        hit_rf_m = len(set(actual_main).intersection(set(res_rf_m)))
        hit_dl_m = len(set(actual_main).intersection(set(res_dl_m)))
        hit_xgb_m = len(set(actual_main).intersection(set(res_xgb_m)))
        hit_ens_m = len(set(actual_main).intersection(set(res_ens_m)))
        
        # Special Scores
        val_act_s = get_rank_spec(actual_spec)
        val_rf_s = get_rank_spec(res_rf_s)
        val_dl_s = get_rank_spec(res_dl_s)
        val_xgb_s = get_rank_spec(res_xgb_s)
        val_ens_s = get_rank_spec(res_ens_s)
        
        gap_rf_s = abs(val_rf_s - val_act_s)
        gap_dl_s = abs(val_dl_s - val_act_s)
        gap_xgb_s = abs(val_xgb_s - val_act_s)
        gap_ens_s = abs(val_ens_s - val_act_s)
        
        # Update Special Gap History
        gap_hist_rf_s.append(gap_rf_s)
        gap_hist_dl_s.append(gap_dl_s)
        gap_hist_xgb_s.append(gap_xgb_s)
        
        hit_rf_s = 1 if res_rf_s == actual_spec else 0
        hit_dl_s = 1 if res_dl_s == actual_spec else 0
        hit_xgb_s = 1 if res_xgb_s == actual_spec else 0
        hit_ens_s = 1 if res_ens_s == actual_spec else 0
        
        # Past 10 Stats (Main only for brevity in log, or combine?)
        # Let's keep Main Past 10 for table
        feat_seq = rec['X_main'][:60] 
        past_10_m = []
        for i in range(10):
            chunk = feat_seq[i*6 : (i+1)*6]
            if not chunk: break
            avg = sum(chunk)/len(chunk)
            past_10_m.append(round(avg, 1))

        results.append({
            "period": f"{curr_idx}", 
            "actual_nums": actual_main,
            "actual_spec": actual_spec,
            "actual_val_m": round(val_act_m, 2),
            "actual_val_s": val_act_s,
            
            # RF
            "rf_nums": res_rf_m,
            "rf_spec": res_rf_s,
            "rf_val_m": round(val_rf_m, 2),
            "rf_val_s": val_rf_s,
            "rf_gap_m": round(gap_rf_m, 2),
            "rf_gap_s": gap_rf_s,
            "rf_hits_m": hit_rf_m,
            "rf_hits_s": hit_rf_s,
            
            # DL
            "dl_nums": res_dl_m,
            "dl_spec": res_dl_s,
            "dl_val_m": round(val_dl_m, 2),
            "dl_val_s": val_dl_s,
            "dl_gap_m": round(gap_dl_m, 2),
            "dl_gap_s": gap_dl_s,
            "dl_hits_m": hit_dl_m,
            "dl_hits_s": hit_dl_s,
            
            # XGB Base
            "xgb_nums": res_xgb_m,
            "xgb_spec": res_xgb_s,
            "xgb_val_m": round(val_xgb_m, 2),
            "xgb_val_s": val_xgb_s,
            "xgb_gap_m": round(gap_xgb_m, 2),
            "xgb_gap_s": gap_xgb_s,
            "xgb_hits_m": hit_xgb_m,
            "xgb_hits_s": hit_xgb_s,
            
            # Ensemble (Stacking)
            "ens_nums": res_ens_m,
            "ens_spec": res_ens_s,
            "ens_val_m": round(val_ens_m, 2),
            "ens_val_s": val_ens_s,
            "ens_gap_m": round(gap_ens_m, 2),
            "ens_gap_s": gap_ens_s,
            "ens_hits_m": hit_ens_m,
            "ens_hits_s": hit_ens_s,
            
            "past_10_avgs": past_10_m
        })
        
    return results

if __name__ == "__main__":
    data = load_data("super_lotto638_results.csv")
    print(f"Total draws: {len(data)}")
    
    if len(data) > 500:
        records = prepare_features(data)
        results = train_and_predict(records)
        
        # Save results
        with open("ml_backtest_results.json", "w", encoding='utf-8') as f:
            json.dump(results, f)
            
        print(f"Saved {len(results)} predictions to ml_backtest_results.json")
    else:
        print("Not enough data.")
