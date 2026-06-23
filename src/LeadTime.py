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
from ddm_elo import dd_elo 
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
    print(f"Failed to load cache: {e}, will rebuild from source files.")


# ==========================================
# 4.0 Random sampling of players (adapt to cache mode)
# ==========================================
print(f"【Cache mode】Extracting player名单 from {TEMP_DATA_DIR}...")
import re
cached_files = os.listdir(TEMP_DATA_DIR)
unique_names = set()
for f in cached_files:
    match = re.match(r"(.+)_section\d+\.csv", f)
    if match:
        unique_names.add(match.group(1))

valid_players = list(unique_names)
print(f"Found {len(valid_players)} unique players in cache.")
if len(valid_players) < NUM_RANDOM_PLAYERS:
    print(f"Warning: Only {len(valid_players)} players meet the game count requirement (> {MIN_GAMES_THRESHOLD}).")
    selected_players = valid_players
else:
    selected_players = random.sample(valid_players, NUM_RANDOM_PLAYERS)

# ==========================================
# 4.1 Elo calculation
# ==========================================
print("Calculating Real Elo trajectories...")
for p_name in selected_players:
    games = player_dict[p_name]
    # Must ensure games are sorted by time
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
print("Real Elo calculation completed. Calculating signals...")

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

for p_name in selected_players:
    games = player_dict[p_name] 
    
    segment_idx = 0
    current_game_idx = 0 
    for signal_val, group in groupby(games, key=lambda x: x['signal']):
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
            print(f"DD-Elo returned {len(ddcps_result_df)} games")
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

print("DD-Elo calculation completed, data has been filled back.")


# ==========================================
# 6. Visualization
# ==========================================
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import groupby

print("\nLead Time Distribution")

lead_time_k_values = range(2, 11)
dist_lead_time = {k: [] for k in lead_time_k_values}

valid_segment_count = 0

i = 0
for p_name in selected_players:
    i += 1
    games = player_dict[p_name]
    games.sort(key=lambda x: x['game_id'])
    
    real_elos = np.array([g['real_elo'] for g in games])
    ddcps_elos = np.array([g['delta_ddcps'] for g in games]) 
    signals_arr = np.array([g['signal'] for g in games])
    current_idx = 0
    for signal_val, group in groupby(signals_arr):
        segment_games_iter = list(group)
        seg_len = len(segment_games_iter)
        start_idx = current_idx
        end_idx = current_idx + seg_len
        
        if signal_val == -1 or signal_val == 1:
            valid_segment_count += 1
            seg_real = real_elos[start_idx:end_idx]
            seg_ddcps = ddcps_elos[start_idx:end_idx]
            
            # ---------------------------------------------------------
            # Area Improvement
            # ---------------------------------------------------------
            E_min = np.min(seg_real)
            E_max = np.max(seg_real)
            
            if E_max > E_min: 
                for K in lead_time_k_values:
                    
                    k_lead_sum_segment = 0.0
                    
                    for k in range(1, K + 1):
                        theta = E_min + (k / K) * (E_max - E_min)

                        def get_first_arrival_idx(arr, threshold, sig):
                            if sig == -1:
                                indices = np.where(arr >= threshold)[0]
                            else:
                                indices = np.where(arr <= threshold)[0]
                            
                            if len(indices) > 0:
                                return indices[0]
                            else:
                                return len(arr) - 1 
                        
                        t_elo = get_first_arrival_idx(seg_real, theta, signal_val)
                        t_ddcps = get_first_arrival_idx(seg_ddcps, theta, signal_val)
                        
                        k_lead_sum_segment += (t_elo - t_ddcps)
                    
                    avg_lead_segment = k_lead_sum_segment / K
                    
                    dist_lead_time[K].append(avg_lead_segment)
        
        current_idx += seg_len

print(f"valid segments: {valid_segment_count}")


# ---------------------------------------------------------
# Pic 4: Lead Time Distribution vs K (segment_metric_lead_time.png)
# ---------------------------------------------------------

data_lead = [dist_lead_time[k] for k in lead_time_k_values]

boxprops = dict(linestyle='-', linewidth=1.5, color='black')
medianprops = dict(linestyle='-', linewidth=2.5, color='gold') # 金色中位数线
meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=6)

plt.figure(figsize=(10, 7))

plt.boxplot(data_lead, vert=True, patch_artist=True,
            showmeans=True, showfliers=False,  
            boxprops=dict(facecolor='mediumpurple', **boxprops), 
            medianprops=medianprops, meanprops=meanprops,
            labels=[str(k) for k in lead_time_k_values]) 

plt.title('Lead Time Distribution by Threshold Count (K)\n(Positive = DDCPS Leads)', fontsize=16, fontweight='bold')
plt.xlabel('Number of Thresholds (K)', fontsize=14)
plt.ylabel('Lead Time (Games)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.8, label='Sync (No Lead)')
plt.legend()

if len(dist_lead_time[5]) > 0:
    k5_mean = np.mean(dist_lead_time[5])
    k5_median = np.median(dist_lead_time[5])
    stats_text = (f"[K=5 Stats]\n"
                  f"Mean: {k5_mean:.2f}\n"
                  f"Median: {k5_median:.2f}")
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                   verticalalignment='top', horizontalalignment='left',
                   fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

out_path_lead = os.path.join(OUTPUT_DIR, 'segment_metric_lead_time.png')
plt.savefig(out_path_lead, dpi=300, bbox_inches='tight')
plt.close()
print(f"Results path: {out_path_lead}")