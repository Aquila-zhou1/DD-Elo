import pandas as pd
import numpy as np
import math
import re
from .elo_per_game import calculate_new_elo

a = 60
z = 0
sigma = 0.0
CPL_LIMIT = 3000
DECA = 0.75
X_DECA = 0.75

def f(t):
    x = float(t)
    if x == 0:
        return 10
    elif 1 <= x <= 4:
        return 9
    elif 5 <= x <= 15:
        return 8
    elif 16 <= x <= 50:
        return 4
    elif 51 <= x <= 100:
        return 0
    elif 101 <= x <= 200:
        return -2
    elif 201 <= x <= 1000:
        return -4
    elif x > 1001:
        return -8
    return 10

def f_2(t):
    return 10-10*math.log(1+0.1*t)

def expect(elo) -> int:
    # return 0
    if elo < 1500:
        return 10
    elif elo == 1500:
        return 5
    else:
        return 0

def clean_to_int(x):
    s = re.sub(r'[^0-9]', '', str(x))
    return int(s)

def drift_diffusion_model(v, x, a=1200, z=1000, sigma=0.1, dv=12, alpha=10):
    decision = 0
    noise = np.random.normal(0, sigma)
    dr = v * dv + noise
    x += dr
    if x >= z + a:
        decision += alpha
        x = z
    elif x <= z - a:
        decision += -alpha
        x = z
    return decision, x

def ddm_model(df,my_elo,x,decision,name):
    new_decision = 0
    is_white = True if df.iloc[0]["white_player"]==name else False
    oppo_elo = clean_to_int(df.iloc[0]["black_elo"]) if is_white else clean_to_int(df.iloc[0]["white_elo"])
    if df.iloc[0]["no_winner"]:
        win = 0.5
    else:
        win = 1 if (is_white and df.iloc[0]["white_won"]) or (not is_white and df.iloc[0]["black_won"]) else 0
    for _, row in df.iterrows():
        object_value = 1 if row["white_active"]==is_white else -1
        curr_cpl = min(row["cp"], CPL_LIMIT)
        curr_elo = (my_elo+new_decision) if row["white_active"]==is_white else oppo_elo
        curr_oppo_elo = oppo_elo if row["white_active"]==is_white else (my_elo+new_decision)
        f_val = f(max(float(curr_cpl) - expect(curr_elo),0))
        rate =  1 / (1 + 10**((curr_oppo_elo - curr_elo) / 400))
        rate = rate if object_value==1 else (1-rate)
        val = f_val*object_value*rate if f_val is not math.isnan(f_val) else 0
        add_dec, x = drift_diffusion_model(val,x,a,z,sigma)
        new_decision+=add_dec
    my_elo_new = calculate_new_elo(my_elo, oppo_elo, win)
    return my_elo_new, x, decision+new_decision


def dd_elo(input_csv, name, my_elo, output_csv="_tmp.csv"):
    df = pd.read_csv(input_csv)

    new_game_flag = (
        (df["move_ply"] == 0) |
        (df["move_ply"] < df["move_ply"].shift())
    )
    df["game_segment"] = new_game_flag.cumsum()

    results = []
    x = 0
    decision = 0

    results.append({
        "cal_elo": my_elo,
        "delta_ddcps": my_elo,
    })

    for _, gdf in df.groupby("game_segment", sort=False):
        my_elo, x, decision = ddm_model(gdf.copy(), my_elo, x, decision, name)
        results.append({
            "cal_elo": my_elo,
            "delta_ddcps": my_elo + decision,
        })
        decision *= DECA
        x *= X_DECA

    out_df = pd.DataFrame(results)
    # out_df.to_csv(output_csv, index=False)
    return out_df


if __name__ == "__main__":
    dd_elo("./1500_1900.csv","alice", 1500)