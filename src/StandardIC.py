import pandas as pd
import numpy as np
import matplotlib
# Set non-interactive backend
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import pickle
import random
from collections import defaultdict
import elo_per_game
from ddm_elo import dd_elo # DD-Elo core algorithm
from itertools import groupby
import argparse 

# ==========================================
# 0. Parse command-line arguments
# ==========================================
parser = argparse.ArgumentParser(description="Chess Player Analysis Script")
parser.add_argument('--cache', action='store_true', help="使用缓存模式，直接利用现有的临时文件")
args = parser.parse_args()
# ==========================================
# 1. Configuration
# ==========================================
INPUT_FILE = './real_data/copy1.csv' 
OUTPUT_DIR = './data_analysis'
DATA_STRUCT_DIR = './data_structure' # 【新增】
CACHE_FILE = os.path.join(DATA_STRUCT_DIR, 'player_dict.pkl') # 【新增】缓存文件名

if not os.path.exists(DATA_STRUCT_DIR):
    os.makedirs(DATA_STRUCT_DIR)
OUTPUT_FILENAME = 'data_analysis_game_person3.png'
# Temporary data folder configuration
TEMP_DATA_DIR = './temp_data'
BIG_DATA_FILE = 'real_data/lichess_db_standard_rated_2019-01.csv'
if not os.path.exists(TEMP_DATA_DIR):
    os.makedirs(TEMP_DATA_DIR)

# Hyperparameter settings
FILTER_WINDOW = 5         # filter window size (for smoothing Elo)
CONV_KERNEL_LARGE = 11    # large convolution kernel size (for trend detection)
TREND_THRESHOLD = 50      # trend threshold (+80 rise, -80 drop)
SIGNAL_SPREAD = 5         # signal spread forward/backward (total 5+1+5=11 games)
MIN_GAMES_THRESHOLD = 50  # minimum games threshold for random selection
NUM_RANDOM_PLAYERS = 1000  # total number of players for analysis

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. Core algorithm functions
# ==========================================

def apply_moving_average(data, window_size):
    """Moving average filter"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def apply_trend_convolution(data, window_size):
    """
    Trend detection convolution kernel (linear trend)
    Returns: trend score series (zero-padded at edges, same length as input)
    """
    kernel = np.linspace(-1, 1, window_size)
    kernel = kernel / (np.sum(np.abs(kernel)) if np.sum(np.abs(kernel)) != 0 else 1) 
    conv_valid = np.convolve(data, kernel, mode='valid')
    pad_len = window_size // 2
    zeros = np.zeros(pad_len)
    result = np.concatenate([zeros, conv_valid, zeros])
    return result

def calculate_signals(trend_series, threshold, spread):
    """
    Calculate signals based on trend values (-1, 0, 1).
    Includes conflict detection and elimination logic.
    """
    N = len(trend_series)
    signals = np.zeros(N)

    high_indices = np.where(trend_series >= threshold)[0]
    low_indices = np.where(trend_series <= -threshold)[0]

    triggers = []
    for idx in high_indices: triggers.append((idx, 1))
    for idx in low_indices: triggers.append((idx, -1))
    triggers.sort(key=lambda x: x[0])

    for idx, val in triggers:
        start = max(0, idx - spread)
        end = min(N, idx + spread + 1)
        opposite = -val
        current_window = signals[start:end]

        if np.any(current_window == opposite):
            conflict_rel_idx = np.where(current_window == opposite)[0][0]
            conflict_abs_idx = start + conflict_rel_idx
            signals[conflict_abs_idx] = 0
            curr = conflict_abs_idx - 1
            while curr >= 0 and signals[curr] != 0:
                signals[curr] = 0
                curr -= 1
            curr = conflict_abs_idx + 1
            while curr < N and signals[curr] != 0:
                signals[curr] = 0
                curr += 1
            continue

        else:
            signals[start:end] = val

    return signals

# ==========================================
# 3. Data loading and construction
# ==========================================
print("Reading data...")
player_dict = defaultdict(list)
cache_loaded = False

# Attempt to load cache
try:
    with open(CACHE_FILE, 'rb') as f:
        player_dict = pickle.load(f)
    print(f"Cache loaded successfully, containing {len(player_dict)} players.")
    cache_loaded = True
except Exception as e:
    print(f"Failed to load cache: {e}，will rebuild from source files.")


# ==========================================
# 4.0 Random sampling of players (adapt to cache mode)
# ==========================================
print(f"[CACHE MODE] Extracting player list from {TEMP_DATA_DIR}...")
import re
cached_files = os.listdir(TEMP_DATA_DIR)
unique_names = set()
for f in cached_files:
    match = re.match(r"(.+)_section\d+\.csv", f)
    if match:
        unique_names.add(match.group(1))

valid_players = list(unique_names)
print(f"Found cached players: {len(valid_players)}")
if len(valid_players) < NUM_RANDOM_PLAYERS:
    print(f"Warning: Not enough players meeting game count requirement (>{MIN_GAMES_THRESHOLD}).")
    selected_players = valid_players
else:
    selected_players = random.sample(valid_players, NUM_RANDOM_PLAYERS)


# ==========================================
# 4.1 Elo calculation
# ==========================================
print("Calculating Elo trajectories...")

for p_name in selected_players:
    games = player_dict[p_name]
    games.sort(key=lambda x: x['game_id'])
    
    # 1. Initialize first game
    if len(games) > 0:
        games[0]['real_elo'] = games[0]['elo']
        current_real_elo = games[0]['real_elo']
        
        for i in range(1, len(games)):
            prev_game = games[i-1]
            new_val = elo_per_game.calculate_new_elo(
                my_elo=current_real_elo, 
                op_elo=prev_game['op_elo'], 
                win=prev_game['win']
            )
            
            current_real_elo = new_val
            games[i]['real_elo'] = current_real_elo

print("Real Elo calculation complete. Now calculating signals...")

# ==========================================
# 4.2 Random sampling and signal calculation
# ==========================================
for p_name in selected_players:
    games = player_dict[p_name]
    games.sort(key=lambda x: x['game_id'])
    raw_elos = np.array([g['real_elo'] for g in games])
    raw_evidence = np.array([g['evidence'] for g in games])
    trend_large = apply_trend_convolution(raw_elos, CONV_KERNEL_LARGE)
    signals = calculate_signals(trend_large, TREND_THRESHOLD, SIGNAL_SPREAD)
    for i, g in enumerate(games):
        g['signal'] = signals[i]


# ==========================================
# 4.3 DD-Elo 
# ==========================================
print("\nProcessing DDCPS computation requirements...")

for p_name in selected_players:
    games = player_dict[p_name]
    
    segment_idx = 0
    current_game_idx = 0 
    for signal_val, group in groupby(games, key=lambda x: x['signal']):
        print(f"group: {group}")
        # Step 1: convert iterator to list immediately and save
        segment_games = list(group) 
        current_ids = [g['data_id'] for g in segment_games]
            
        # Step 3: retrieve data from saved list
        elo0 = segment_games[0]['real_elo']
        segment_idx += 1
        segment_len = len(segment_games)
        
        seg_data_ids = [g['data_id'] for g in segment_games]
        temp_csv_path = os.path.join(TEMP_DATA_DIR, f"{p_name}_section{segment_idx}.csv")
        
        if not os.path.exists(temp_csv_path):
            print(f"[CACHE MODE ERROR] Expected cache file not found: {temp_csv_path}, skip segment.")
            current_game_idx += segment_len
            continue
        
        # 4. Call DD-elo
        try:
            print(f"This segment has {segment_len} games")
            
            ddcps_result_df = dd_elo(temp_csv_path, p_name, elo0)
            if 'delta_ddcps' in ddcps_result_df.columns:
                deltas = ddcps_result_df['delta_ddcps'].values
                safe_len = min(len(deltas), segment_len)
                for i in range(safe_len):
                    games[current_game_idx + i]['delta_ddcps'] = deltas[i]           
            else:
                print(f"Error: dd_elo result missing 'delta_ddcps' column ({temp_csv_path})")
                
        except Exception as e:
            print(f"Calling dd_elo failed: {e}")
            
        current_game_idx += segment_len
print("DD-Elo computation complete, data backfilled.")


# ==========================================
# 6. Visualization
# ==========================================
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import os

print("\n正在进行 IC 指标分布计算 (Grouped by 50 Players)...")


GROUP_SIZE = 50  
ic_k_values = [1, 2, 3, 4, 5]


dist_std_ic = {k: [] for k in ic_k_values} 

player_batches = [selected_players[i:i + GROUP_SIZE] for i in range(0, len(selected_players), GROUP_SIZE)]

print(f"selected_players: {len(selected_players)}, player_batches: {len(player_batches)}, group size: {GROUP_SIZE}")


for batch_idx, batch_players in enumerate(player_batches):

    batch_data = {k: {'f': [], 'o': []} for k in ic_k_values}
    

    for p_name in batch_players:
        games = player_dict[p_name]

        games.sort(key=lambda x: x['game_id'])
        

        real_elos = np.array([g['real_elo'] for g in games])
        ddcps_elos = np.array([g['delta_ddcps'] for g in games])
        factor_arr = ddcps_elos - real_elos
        
        N_games = len(real_elos)
        indices = np.arange(N_games)
        

        for k in ic_k_values:
            valid_indices = indices[indices < (N_games - k)]
            if len(valid_indices) > 0:
                # Factor
                f_vals = factor_arr[valid_indices]
                # Outcome
                o_vals = real_elos[valid_indices + k] - real_elos[valid_indices]
                

                batch_data[k]['f'].extend(f_vals)
                batch_data[k]['o'].extend(o_vals)
    

    for k in ic_k_values:
        f_total = batch_data[k]['f']
        o_total = batch_data[k]['o']
        
        if len(f_total) > 10 and np.std(f_total) > 1e-6 and np.std(o_total) > 1e-6:
            corr, _ = stats.spearmanr(f_total, o_total)
            if not np.isnan(corr):
                dist_std_ic[k].append(corr)

print("Standard IC Group Analysis Complete.")
for k in ic_k_values:
    print(f" -> K={k}: dist_std_ic[k] {len(dist_std_ic[k])} (Mean IC: {np.mean(dist_std_ic[k]):.4f})")

# ==========================================
# 9. Visualization
# ==========================================
print("Generating experimental result visualizations...")

data_std = [dist_std_ic[k] for k in ic_k_values]


boxprops = dict(linestyle='-', linewidth=1.5, color='black')
medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick') 
meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=6)

fig, ax = plt.subplots(figsize=(12, 8))


ax.boxplot(data_std, vert=True, patch_artist=True,
           showmeans=True, showfliers=False, 
           boxprops=dict(facecolor='cornflowerblue', **boxprops),
           medianprops=medianprops, meanprops=meanprops)


all_values = [val for sublist in data_std for val in sublist]
if all_values:
    max_val = max(np.max(np.abs(all_values)), 0.05) 
else:
    max_val = 0.1


y_limit = max_val * 1.3  
ax.set_ylim(-y_limit, y_limit)


ax.set_title(f'Standard IC Distribution by Prediction Horizon (K)\n(Metric: Spearman Correlation, Grouped by {GROUP_SIZE} Players)', 
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Horizon K (Games Ahead)', fontsize=14)
ax.set_ylabel('Spearman IC', fontsize=14)
ax.set_xticklabels([f'k={k}' for k in ic_k_values], fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.6)


ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax.axhline(y=0.02, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.02)')
ax.legend(loc='upper right')

text_y_pos = -y_limit * 0.85 

x_positions = range(1, len(ic_k_values) + 1)
for i, k in enumerate(ic_k_values):
    vals = dist_std_ic[k]
    if len(vals) > 0:
        val_mean = np.mean(vals)
        val_median = np.median(vals)
        
        stats_text = (f"Avg: {val_mean:.3f}\n"
                      f"Med: {val_median:.3f}")

        ax.text(x_positions[i], text_y_pos, stats_text, 
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

out_path_std = os.path.join(OUTPUT_DIR, 'ic_analysis_standard.png')
plt.savefig(out_path_std, dpi=300, bbox_inches='tight')
plt.close()
print(f"Result saved to: {out_path_std}")