# must use cache mode
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
from src import calculate_new_elo # Elo calculate module
from src import dd_elo # DD-Elo core algorithm
from itertools import groupby
import argparse 

# ==========================================
# 0. Parse command-line arguments
# ==========================================
parser = argparse.ArgumentParser(description="Chess Player Analysis Script")
parser.add_argument('--cache', action='store_true', help="using cache mode")
args = parser.parse_args()
# ==========================================
# 1. Configuration
# ==========================================
INPUT_FILE = 'data/cache/game_data.csv' 
OUTPUT_DIR = 'data_analysis'
DATA_STRUCT_DIR = 'data/cache'
CACHE_FILE = os.path.join(DATA_STRUCT_DIR, 'player_dict.pkl') 

if not os.path.exists(DATA_STRUCT_DIR):
    os.makedirs(DATA_STRUCT_DIR)
OUTPUT_FILENAME = 'data_analysis_game_person3.png'
# Temporary data folder configuration
TEMP_DATA_DIR = 'data/processed'
BIG_DATA_FILE = 'data/raw/lichess_db_standard_rated_2019-01.csv'
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

# Load cache
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
print(f"Found {len(valid_players)} unique players.")
if len(valid_players) < NUM_RANDOM_PLAYERS:
    print(f"Warning: Only {len(valid_players)} players meet the game count requirement (> {MIN_GAMES_THRESHOLD}).")
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
            new_val = calculate_new_elo(
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
# 4.3 新增需求：DDCPS 计算 (分段与大文件处理)
# ==========================================
print("calculating DD-ELo")

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

print("DD-Elo calculation complete for all segments.")


# ==========================================
# 6. Visualization
# ==========================================
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import groupby

print("\nSegment Distribution Analysis")
dist_area_pct = []  
dist_da = []       
dist_variant_ic = []

valid_segment_count = 0

i = 0
for p_name in selected_players:
    i += 1
    
    games = player_dict[p_name]
    games.sort(key=lambda x: x['game_id'])
    
    real_elos = np.array([g['real_elo'] for g in games])
    ddcps_elos = np.array([g['delta_ddcps'] for g in games]) 
    signals_arr = np.array([g['signal'] for g in games])
    
    # Factor: F_t = ddcps_t - real_elo_t
    factor_arr = ddcps_elos - real_elos
    
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
            seg_factor = factor_arr[start_idx:end_idx]
            
            # ---------------------------------------------------------
            # Area Improvement
            # ---------------------------------------------------------
            diff_abs_sum = np.sum(np.abs(seg_ddcps - seg_real))
            
            seg_metric_area = 0.0
            if diff_abs_sum > 0:
                improvement_sum = 0.0
                if signal_val == -1: 
                    # DD-Elo > Real Elo
                    improvement_sum = np.sum(np.maximum(seg_ddcps - seg_real, 0))
                elif signal_val == 1: 
                    # DD-Elo < Real Elo
                    improvement_sum = np.sum(np.maximum(seg_real - seg_ddcps, 0))
                
                seg_metric_area = (improvement_sum / diff_abs_sum) * 100
            
            dist_area_pct.append(seg_metric_area)
            
            # =========================================================
            # Metric B: Directional Accuracy (Per Segment)
            # =========================================================
            # Target Direction = -signal_val
            target_dir = -signal_val 
            
            pred_dirs = np.sign(seg_factor)
            

            hits = np.sum(pred_dirs == target_dir)
            seg_metric_da = hits / seg_len
            dist_da.append(seg_metric_da)
        current_idx += seg_len

    # -------------------------------------------------------------
    # Track A: IC Variant
    # -------------------------------------------------------------
    valid_mask = (signals_arr != 0)
    if np.sum(valid_mask) > 5: 
        p_factors = factor_arr[valid_mask]
        p_targets = -signals_arr[valid_mask] 
        
        # Variant IC
        corr_v, _ = stats.spearmanr(p_factors, p_targets)
        if not np.isnan(corr_v):
            dist_variant_ic.append(corr_v)



print(f"valid segments: {valid_segment_count}")
print(f"Area: {len(dist_area_pct)}, DA: {len(dist_da)}")


# ==========================================
# 9. Combined Smoothed Line Plot with Normalized Peaks
# ==========================================
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.lines as mlines
from scipy.interpolate import PchipInterpolator


data_area = np.array([x for x in dist_area_pct if not np.isnan(x) and x >= 10])
data_da = np.array([x for x in dist_da if not np.isnan(x) and (x * 100) >= 10])
data_ic_var = np.array([x for x in dist_variant_ic if not np.isnan(x) and x >= -0.8])

data_da_scaled = data_da * 100  

fig, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twiny()

bins_ax1 = np.linspace(10, 100, 21)
bins_ax2 = np.linspace(-0.8, 1, 21)

bin_centers_ax1 = (bins_ax1[:-1] + bins_ax1[1:]) / 2
bin_centers_ax2 = (bins_ax2[:-1] + bins_ax2[1:]) / 2

counts_area, _ = np.histogram(data_area, bins=bins_ax1)
counts_da, _ = np.histogram(data_da_scaled, bins=bins_ax1)
counts_ic, _ = np.histogram(data_ic_var, bins=bins_ax2)

norm_area = counts_area / counts_area.max() if counts_area.max() > 0 else counts_area
norm_da = counts_da / counts_da.max() if counts_da.max() > 0 else counts_da
norm_ic = counts_ic / counts_ic.max() if counts_ic.max() > 0 else counts_ic

x_smooth_ax1 = np.linspace(10, 100, 300)
x_smooth_ax2 = np.linspace(-0.8, 1, 300)

interp_area = PchipInterpolator(bin_centers_ax1, norm_area)
interp_da = PchipInterpolator(bin_centers_ax1, norm_da)
interp_ic = PchipInterpolator(bin_centers_ax2, norm_ic)

y_smooth_area = interp_area(x_smooth_ax1)
y_smooth_da = interp_da(x_smooth_ax1)
y_smooth_ic = interp_ic(x_smooth_ax2)

line_area, = ax1.plot(x_smooth_ax1, y_smooth_area, color='dodgerblue', linewidth=2.5, label='Area Improvement %')
line_da, = ax1.plot(x_smooth_ax1, y_smooth_da, color='limegreen', linewidth=2.5, label='Directional Accuracy')
line_ic, = ax2.plot(x_smooth_ax2, y_smooth_ic, color='darkorange', linewidth=2.5, label='IC Variant')

median_area = np.median(data_area)
median_da = np.median(data_da_scaled)
median_ic = np.median(data_ic_var)

ax1.plot(median_area, interp_area(median_area), marker='o', markersize=9, color='dodgerblue', markeredgecolor='white', markeredgewidth=2, zorder=5)
ax1.plot(median_da, interp_da(median_da), marker='s', markersize=9, color='limegreen', markeredgecolor='white', markeredgewidth=2, zorder=5)
ax2.plot(median_ic, interp_ic(median_ic), marker='^', markersize=10, color='darkorange', markeredgecolor='white', markeredgewidth=2, zorder=5)

ax1.axvline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax2.axvline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.5)

ax1.set_xlim(10, 100)
ax2.set_xlim(-0.8, 1)
ax1.set_ylim(0, 1.05) 

ax1.set_xlabel('Percentage (%) [Area Improvement & Directional Accuracy]', fontsize=12, fontweight='bold')
ax2.set_xlabel('Spearman Correlation [-0.8 to 1] [IC Variant]', fontsize=12, fontweight='bold')
ax1.set_ylabel('Normalized Sample Count (Peak = 1.0)', fontsize=12, fontweight='bold')

plt.title('Distribution of Area Improvement, DA, and IC Variant (Smoothed)', fontsize=15, fontweight='bold', pad=25)

# 9. 自定义图例
median_marker_legend = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, label='Median Point')
baseline_legend = mlines.Line2D([], [], color='red', linestyle='--', alpha=0.5, label='Center Baseline (50% / 0)')

ax1.legend(handles=[line_area, line_da, line_ic, median_marker_legend, baseline_legend], 
           loc='upper left', framealpha=0.9, bbox_to_anchor=(1.02, 1))

ax1.grid(axis='y', linestyle='--', alpha=0.6)

# 10. 保存最终画布
plt.tight_layout()
out_path_line = os.path.join(OUTPUT_DIR, 'segment_metrics_variant_smoothed.png')
plt.savefig(out_path_line, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f" -> smoothed plot saved: {out_path_line}")