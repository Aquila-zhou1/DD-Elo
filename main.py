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
import src.elo_per_game as elo_per_game # Elo calculate module
from src.ddm_elo import dd_elo # DD-Elo core algorithm
from itertools import groupby
import argparse # 【新增】

# ==========================================
# 0. Parse command-line arguments
# ==========================================
parser = argparse.ArgumentParser(description="Chess Player Analysis Script")
parser.add_argument('--cache', action='store_true', help="使用缓存模式，直接利用现有的临时文件")
args = parser.parse_args()

CACHE_MODE = args.cache # define global cache mode flag
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
NUM_RANDOM_PLAYERS = 1000  # [MODIFIED] total number of players for analysis (expanded)
NUM_DRAW_PLAYERS = 10     # [ADDED] number of players to draw (5 rows x 2 cols)

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

    # 1. 获取所有触发点
    high_indices = np.where(trend_series >= threshold)[0]
    low_indices = np.where(trend_series <= -threshold)[0]

    # 2. 合并触发点并按索引排序 (模拟时间线推进)
    triggers = []
    for idx in high_indices: triggers.append((idx, 1))
    for idx in low_indices: triggers.append((idx, -1))
    triggers.sort(key=lambda x: x[0])

    # 3. 遍历触发点进行赋值或冲突消除
    for idx, val in triggers:
        start = max(0, idx - spread)
        end = min(N, idx + spread + 1)
        opposite = -val
        current_window = signals[start:end]

        if np.any(current_window == opposite):
            # === 发现冲突，执行消除逻辑 ===
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
if os.path.exists(CACHE_FILE):
    print(f"Found cache file: {CACHE_FILE}, loading...")
    try:
        with open(CACHE_FILE, 'rb') as f:
            player_dict = pickle.load(f)
        print(f"Cache loaded successfully, contains {len(player_dict)} players.")
        cache_loaded = True
    except Exception as e:
        print(f"Failed to load cache: {e}, will rebuild from source.")

if not cache_loaded:
    print(f"Reading source data: {INPUT_FILE} ...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: file not found {INPUT_FILE}")
        exit()

    print("Building expanded player dictionary structure...")
    white_data = df[['white_player', 'white_elo', 'mask_cpl_diff_mean', 'game_id', 'win', 'elo_diff']].copy()
    white_data['mask_cpl_diff_mean'] *= 10  
    white_data['op_elo'] = white_data['white_elo'] - white_data['elo_diff']
    white_data.rename(columns={'white_player': 'player', 'white_elo': 'elo', 'mask_cpl_diff_mean': 'evidence'}, inplace=True)
    white_data_inter = white_data.set_index(pd.Index(range(0,len(white_data)*2,2)))

    black_data = pd.DataFrame({
        'player': df['black_player'],
        'elo': df['white_elo'] - df['elo_diff'],
        'evidence': -df['mask_cpl_diff_mean']*10, # invert to black player's perspective
        'game_id': df['game_id'],
        'op_elo': df['white_elo'], 
        'win': 1-df['win']
    })
    black_data_inter = black_data.set_index(pd.Index(range(1,len(black_data)*2+1,2)))

    combined_data = pd.concat([
        white_data_inter[['player', 'elo', 'evidence', 'game_id', 'op_elo', 'win']], 
        black_data_inter[['player', 'elo', 'evidence', 'game_id', 'op_elo', 'win']]
    ]).sort_index()

    # Populate dictionary (added evidence and default signal)
    for idx, row in combined_data.iterrows():
        player_dict[row['player']].append({
            'game_id': idx, 
            'elo': row['elo'],
            'evidence': row['evidence'],
            'signal': 0,
            'data_id': row['game_id'],
            'delta_ddcps': 0.0,
            'real_elo': 0.0,
            'op_elo': row['op_elo'], 
            'win': row['win']   
        })

    # After building, save to cache
    print(f"Saving player_dict to cache: {CACHE_FILE} ...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(player_dict, f)


# ==========================================
# 4.0 Random sampling of players (adapt to cache mode)
# ==========================================
if CACHE_MODE:
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
else:
    valid_players = [p for p, games in player_dict.items() if len(games) >= MIN_GAMES_THRESHOLD]
if len(valid_players) < NUM_RANDOM_PLAYERS:
    print(f"Warning: Not enough players with >{MIN_GAMES_THRESHOLD} games to reach {NUM_RANDOM_PLAYERS}.")
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
all_signal_values = []
all_evidence_values = []

for p_name in selected_players:
    games = player_dict[p_name]
    games.sort(key=lambda x: x['game_id'])
    raw_elos = np.array([g['real_elo'] for g in games])
    raw_evidence = np.array([g['evidence'] for g in games])
    
    # Trend (Large Kernel)
    trend_large = apply_trend_convolution(raw_elos, CONV_KERNEL_LARGE)
    signals = calculate_signals(trend_large, TREND_THRESHOLD, SIGNAL_SPREAD)
    for i, g in enumerate(games):
        g['signal'] = signals[i]        
    all_signal_values.extend(signals)
    all_evidence_values.extend(raw_evidence)



# ==========================================
# 4.3 DD-Elo 
# ==========================================
print("\nProcessing DDCPS computation requirements...")

# --- Step A: Optimize large file reading ---
# 1. Collect target ID set
target_game_ids = set()
for p_name in selected_players:
    for g in player_dict[p_name]:
        target_game_ids.add(g['data_id'])

print(f"Total number of Game IDs to retrieve: {len(target_game_ids)}")

# 3. Scan big file (Chunk Reading)
if not CACHE_MODE:
    print(f"Scanning big file {BIG_DATA_FILE} (this may take a few minutes)...")
    matched_rows = []
    try:
        chunk_iter = pd.read_csv(BIG_DATA_FILE, chunksize=100000, on_bad_lines='skip', low_memory=False)
        
        for chunk in chunk_iter:
            if 'game_id' in chunk.columns:
                mask = chunk['game_id'].isin(target_game_ids)
                matched_rows.append(chunk[mask])
            else:
                print("Error: 'game_id' column not found in big file; cannot match.")
                break
                

        if matched_rows:
            big_data_subset = pd.concat(matched_rows)
            big_data_subset.set_index('game_id', inplace=True)
            print(f"Big file scan complete, extracted related games: {len(big_data_subset)}")
        else:
            big_data_subset = pd.DataFrame()
            print("Warning: No matching games found in big file.")

    except Exception as e:
        print(f"Error reading big file: {e}")
        big_data_subset = pd.DataFrame()

else:
    print("[CACHE MODE] Skipping 40GB big file scan.")
    big_data_subset = pd.DataFrame() 


# --- Step B: Segment processing and DDCPS computation ---
print("Processing segments and calling DDCPS...")

for p_name in selected_players:
    games = player_dict[p_name] 
    print(f"Player {p_name} total games: {len(games)}")
    
    segment_idx = 0
    current_game_idx = 0 
    for signal_val, group in groupby(games, key=lambda x: x['signal']):
        print(f"group: {group}")
        # Step 1: convert iterator to list immediately and save
        segment_games = list(group) 
        current_ids = [g['data_id'] for g in segment_games]
        if not segment_games:
            continue
            
        # Step 3: retrieve data from saved list
        elo0 = segment_games[0]['real_elo']
        segment_idx += 1
        segment_len = len(segment_games)
        
        seg_data_ids = [g['data_id'] for g in segment_games]
        temp_csv_path = os.path.join(TEMP_DATA_DIR, f"{p_name}_section{segment_idx}.csv")
    
        if not CACHE_MODE:
            temp_frames = []
            for gid in seg_data_ids:
                game_moves = big_data_subset.loc[[gid]]
                if 'move_ply' in game_moves.columns:
                    game_moves = game_moves.sort_values(by='move_ply')            
                temp_frames.append(game_moves)
            
            if temp_frames:
                seg_df = pd.concat(temp_frames)
            else:
                print("Empty segment encountered")
                seg_df = pd.DataFrame()

            # Normal mode: generate/overwrite file from seg_df
            if not seg_df.empty:
                seg_df.to_csv(temp_csv_path)
            else:
                print(f"Skipping generation of empty file: {temp_csv_path}")
                current_game_idx += segment_len
                continue
        else:
            # Cache mode: check file existence
            if not os.path.exists(temp_csv_path):
                print(f"[CACHE MODE ERROR] Expected cache file not found: {temp_csv_path}, skip segment.")
                current_game_idx += segment_len
                continue
        
        # 4. Call DD-elo
        try:
            print(f"This segment has {segment_len} games")
            
            ddcps_result_df = dd_elo(temp_csv_path, p_name, elo0)
            print(f"DD-Elo returned {len(ddcps_result_df)} games")
            # 5. 回填 delta_ddcps 到 player_dict
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
# 5. Correlation calculation (Requirements 5)
# ==========================================
print("Calculating correlation...")
np_signals = np.array(all_signal_values)
np_evidences = np.array(all_evidence_values)
mask = np_signals != 0

# 应用筛选
filtered_signals = np_signals[mask]
filtered_evidences = np_evidences[mask]

if len(filtered_signals) > 1: # 至少要有两个点才能算相关系数
    # 计算 Pearson 相关系数
    corr_matrix = np.corrcoef(filtered_signals, filtered_evidences)
    correlation = corr_matrix[0, 1]

    print(f"\n=============================================")
    print(f"Correlation between Signal and Evidence (excluding Signal=0): {correlation:.4f}")
    print(f"Effective samples (Signal!=0): {len(filtered_signals)} / Total samples: {len(all_signal_values)}")
    print(f"=============================================\n")
else:
    print("Not enough non-zero Signal data to compute correlation.")

# ==========================================
# 6. Visualization
# ==========================================
print("Plotting charts...")

fig, axes = plt.subplots(5, 2, figsize=(24, 20))
plt.subplots_adjust(hspace=0.4, wspace=0.3)
axes_flat = axes.flatten()

drawing_players = selected_players[:NUM_DRAW_PLAYERS] 

for i, p_name in enumerate(drawing_players):
    if i >= len(axes_flat): break
    
    ax1 = axes_flat[i]
    games = player_dict[p_name]
    
    raw_elos = np.array([g['real_elo'] for g in games])
    evidence = np.array([g['evidence'] for g in games])
    signals = np.array([g['signal'] for g in games])
    
    trend_large = apply_trend_convolution(raw_elos, CONV_KERNEL_LARGE)
    ma_elos = apply_moving_average(raw_elos, FILTER_WINDOW)
    pad_len = (len(raw_elos) - len(ma_elos)) // 2
    ma_elos_padded = np.pad(ma_elos, (pad_len, len(raw_elos)-len(ma_elos)-pad_len), 'constant', constant_values=np.nan)
        
    ddcps_curve = np.full(len(raw_elos), np.nan)
    delta_ddcps_arr = np.array([g['delta_ddcps'] for g in games])
    signals_arr = np.array([g['signal'] for g in games])
    
    current_idx = 0
    from itertools import groupby 
    
    for signal_val, group in groupby(signals_arr):

        seg_len = len(list(group))
        start_idx = current_idx
        end_idx = current_idx + seg_len
        
        base_val = ma_elos_padded[start_idx]
        if np.isnan(base_val):
            print("Warning: encountered NaN values")
            base_val = raw_elos[start_idx]
            

        segment_deltas = delta_ddcps_arr[start_idx:end_idx]
        segment_cumsum = segment_deltas
        ddcps_curve[start_idx:end_idx] = segment_cumsum
        current_idx += seg_len

    
    x_seq = np.arange(1, len(raw_elos) + 1)

    real_elo_arr = np.array([g['real_elo'] for g in games])
    ax1.plot(x_seq, raw_elos, color='gray', alpha=0.3, linewidth=1, label='Raw Elo')
    ax1.plot(x_seq, real_elo_arr, color='magenta', linewidth=1.5, linestyle='--', label='Real Elo (Recalc)')
    ax1.plot(x_seq, ddcps_curve, color='black', linewidth=1.5, linestyle='-', label='DD-Elo')


    # 2. MA Filter
    points = np.array([x_seq, ma_elos_padded]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_signals = signals[:-1] 
    

    cmap = ListedColormap(['green', 'blue', 'red'])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(seg_signals)
    lc.set_linewidth(2)
    ax1.add_collection(lc)
    
    ax1.plot([], [], color='blue', label='MA (Stable)')
    ax1.plot([], [], color='red', label='MA (Drop Trend)')
    ax1.plot([], [], color='green', label='MA (Rise Trend)')
    
    ax1.set_ylabel('Elo Rating', color='blue')
    ax1.set_xlabel('Game Sequence')
    ax1.set_title(f'{p_name} (Signal vs Evidence)', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax2 = ax1.twinx() 
    
    # 3. Evidence (mask_cpl_diff_mean)
    l_evi = ax2.plot(x_seq, evidence, color='purple', alpha=0.3, linewidth=0.5, label='Evidence (CPL Diff)')
    
    # 4. Trend Conv (Large)
    l_trend = ax2.plot(x_seq, trend_large, color='orange', linestyle='-.', linewidth=1.5, label=f'Trend (K={CONV_KERNEL_LARGE})')
    ax2.axhline(TREND_THRESHOLD, color='red', linestyle=':', linewidth=0.5)
    ax2.axhline(-TREND_THRESHOLD, color='green', linestyle=':', linewidth=0.5)
    
    ax2.set_ylabel('Trend / Evidence Value', color='purple')
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize='x-small', framealpha=0.9)

for j in range(len(selected_players), len(axes_flat)):
    axes_flat[j].axis('off')

# ==========================================
# 7. Save results
# ==========================================
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to: {output_path}")


# ==========================================
# 8. Experimental Validation
# ==========================================
import scipy.stats as stats 
# Running experimental metric validation (Lead Time & AreaPct)
print("\nPerforming experimental metric validation (Lead Time & AreaPct)...")

lead_time_sums = {k: 0.0 for k in range(2, 11)}
lead_time_counts = {k: 0 for k in range(2, 11)}

# Area Percentage
total_area_improvement = 0.0 # Σ a 
total_area_diff = 0.0        # Σ (a + b)

valid_segment_count = 0

# 2. Information Coefficient (IC)
ic_k_values = [1, 2, 3, 4, 5]
ic_data_storage = {k: {'f': [], 'o': []} for k in ic_k_values}
ic_variant_f = [] # 变体 Factor
ic_variant_s = [] # 变体 Outcome 

# 3. Directional Accuracy (DA)
da_total_hits = 0
da_total_n = 0
for p_name in selected_players:
    games = player_dict[p_name]
    games.sort(key=lambda x: x['game_id'])
    
    
    real_elos = np.array([g['real_elo'] for g in games])
    ddcps_elos = np.array([g['delta_ddcps'] for g in games]) 
    signals_arr = np.array([g['signal'] for g in games])
    
    
    # Factor: F_t = ddcps_t - real_elo_t
    factor_arr = ddcps_elos - real_elos
    valid_mask = (signals_arr != 0)
    
    if np.sum(valid_mask) > 0:
        valid_factors = factor_arr[valid_mask]
        valid_signals = signals_arr[valid_mask]
        
        # === Metric: Directional Accuracy (DA) ===
        pred_direction = np.sign(valid_factors)
        target_direction = -valid_signals
        
        hits = np.sum(pred_direction == target_direction)
        da_total_hits += hits
        da_total_n += len(valid_factors)
        
        # === Metric: IC Variant (Outcome = Signal) ===
        ic_variant_f.extend(valid_factors)
        ic_variant_s.extend(-valid_signals)
        
    # === Metric: Standard IC (Outcome = Future Return) ===
    N_games = len(real_elos)
    indices = np.arange(N_games) 
    
    for k in ic_k_values:
        valid_indices_k = indices[indices < (N_games - k)]
        
        if len(valid_indices_k) > 0:
            f_vals = factor_arr[valid_indices_k]
            # Outcome: real_elo_(t+k) - real_elo_t
            o_vals = real_elos[valid_indices_k + k] - real_elos[valid_indices_k]
            
            ic_data_storage[k]['f'].extend(f_vals)
            ic_data_storage[k]['o'].extend(o_vals)

    # -------------------------------------------------------------
    # Part 2: Lead Time & Area Pct
    # -------------------------------------------------------------
    from itertools import groupby
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
            # Area Improvemen
            # ---------------------------------------------------------
            diff_abs_sum = np.sum(np.abs(seg_ddcps - seg_real))
            
            if diff_abs_sum > 0:
                improvement_sum = 0.0
                if signal_val == -1: 
                    # DD-Elo > Real Elo
                    improvement_sum = np.sum(np.maximum(seg_ddcps - seg_real, 0))
                elif signal_val == 1: 
                    # DD-Elo < Real Elo
                    improvement_sum = np.sum(np.maximum(seg_real - seg_ddcps, 0))
                
                total_area_improvement += improvement_sum
                total_area_diff += diff_abs_sum

            # ---------------------------------------------------------
            # Lead Time 
            # ---------------------------------------------------------
            E_min = np.min(seg_real)
            E_max = np.max(seg_real)
            
            if E_max > E_min: 
                
                for K in range(2, 11):
                    k_lead_sum = 0.0
                    
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
                        
                        k_lead_sum += (t_elo - t_ddcps)
                    
                    lead_time_sums[K] += (k_lead_sum / K)
                    lead_time_counts[K] += 1
        
        current_idx += seg_len

# 1. Area Pct
final_area_pct = 0.0
if total_area_diff > 0:
    final_area_pct = (total_area_improvement / total_area_diff) * 100

# 2. Lead Time
lead_time_results = []
for K in range(2, 11):
    if lead_time_counts[K] > 0:
        avg_lead = lead_time_sums[K] / lead_time_counts[K]
    else:
        avg_lead = 0.0
    lead_time_results.append(avg_lead)

# 3. DA Value
final_da = 0.0
if da_total_n > 0:
    final_da = da_total_hits / da_total_n

# 4. IC Calculation (Spearman)
ic_results = []
for k in ic_k_values:
    f_data = ic_data_storage[k]['f']
    o_data = ic_data_storage[k]['o']
    if len(f_data) > 1:
        corr, _ = stats.spearmanr(f_data, o_data)
        ic_results.append(corr)
    else:
        ic_results.append(0.0)

# 5. IC Variant Calculation
final_ic_variant = 0.0
if len(ic_variant_f) > 1:
    corr_v, _ = stats.spearmanr(ic_variant_f, ic_variant_s)
    final_ic_variant = corr_v


# ==========================================
# 9. Visualization
# ==========================================
print("Generating experimental result visualizations...")

# ------------------------------------------
# Pic 1: real_exp_result.png (Lead Time & Area)
# ------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.axis('off')

# Text: Area Pct
text_str1 = (
    f"METRIC SET A: LAG & AREA\n"
    f"========================\n"
    f"Segments Analyzed: {valid_segment_count}\n\n"
    f"[Area Improvement Percentage]\n"
    f"  Result: {final_area_pct:.2f}%\n"
    f"  (DDCPS reduces lag area vs Baseline)\n"
)
plt.text(0.05, 0.60, text_str1, fontsize=12, family='monospace', transform=ax1.transAxes, va='top')

# Table: Lead Time
table_data_lt = []
table_data_lt.append([0.0] * 9) # Baseline
table_data_lt.append([round(x, 2) for x in lead_time_results])
col_labels_lt = [f"K={k}" for k in range(2, 11)]
row_labels_lt = ['Real Elo (0)', 'DDCPS Lead']

table_lt = plt.table(cellText=table_data_lt, rowLabels=row_labels_lt, colLabels=col_labels_lt,
                     loc='center', cellLoc='center', bbox=[0.05, 0.1, 0.9, 0.2])
table_lt.auto_set_font_size(False)
table_lt.set_fontsize(10)

plt.text(0.5, 0.32, "Table 1: Average Lead Time (Games) by Threshold Splits (K)", 
         ha='center', fontsize=11, fontweight='bold', transform=ax1.transAxes)

out_path1 = os.path.join(OUTPUT_DIR, 'real_exp_result.png')
plt.savefig(out_path1, dpi=300, bbox_inches='tight')
plt.close(fig1) # close figure to free memory
print(f"  -> {out_path1} generated")


# ------------------------------------------
# Pic 2: real_exp_result1.png (IC & DA)
# ------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.axis('off')

# Text: DA & IC Variant
text_str2 = (
    f"METRIC SET B: PREDICTION & ACCURACY\n"
    f"===================================\n"
    f"[Directional Accuracy (DA)]\n"
    f"  Score: {final_da:.4f} ({final_da*100:.2f}%)\n"
    f"  Formula: mean( sgn(ddcps-real) == -signal )\n\n"
    f"[IC Variant (Outcome=Signal)]\n"
    f"  Spearman IC: {final_ic_variant:.4f}\n"
)
plt.text(0.05, 0.60, text_str2, fontsize=12, family='monospace', transform=ax2.transAxes, va='top')

# Table: IC Standard
table_data_ic = []
table_data_ic.append([0.0] * 5) # Baseline
table_data_ic.append([round(x, 4) for x in ic_results])
col_labels_ic = [f"k={k}" for k in ic_k_values]
row_labels_ic = ['Real Elo (0)', 'DDCPS IC']

table_ic = plt.table(cellText=table_data_ic, rowLabels=row_labels_ic, colLabels=col_labels_ic,
                     loc='center', cellLoc='center', bbox=[0.05, 0.1, 0.9, 0.2])
table_ic.auto_set_font_size(False)
table_ic.set_fontsize(10)

plt.text(0.5, 0.32, "Table 2: Information Coefficient (IC) Prediction Horizon", 
         ha='center', fontsize=11, fontweight='bold', transform=ax2.transAxes)
plt.text(0.5, 0.05, "Factor: (ddcps-real), Outcome: (real[t+k] - real[t])", 
         ha='center', fontsize=9, style='italic', transform=ax2.transAxes)

out_path2 = os.path.join(OUTPUT_DIR, 'real_exp_result1.png')
plt.savefig(out_path2, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"  -> {out_path2} generated")

print(f"DA: {final_da:.4f}, IC(k=1): {ic_results[0]:.4f}")