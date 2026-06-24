import pandas as pd

CPL_LIMIT = 3000

def process_games(input_path, output_path, chunk_size=10000):
    columns = [
        "game_id", "white_cpl_mean", "black_cpl_mean", "white_elo", "elo_diff", 
        "cpl_diff_mean", "white_player", "black_player", "win", 
        "mask_white_cpl_mean", "mask_black_cpl_mean", "mask_cpl_diff_mean"
    ]

    with open(output_path, 'w') as f:
        f.write(','.join(columns) + '\n')

    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        chunk["cp"] = pd.to_numeric(chunk["cp"], errors="coerce")
        groups = chunk.groupby("game_id")

        results = []
        
        for game_id, g in groups:
            white_elo = g["white_elo"].iloc[0]
            black_elo = g["black_elo"].iloc[0]
            
            g["cp"] = pd.to_numeric(g["cp"], errors='coerce')

            white_cpl = g.loc[g["white_active"] == True, "cp"].abs()
            black_cpl = g.loc[g["white_active"] == False, "cp"].abs()

            mask = white_cpl > CPL_LIMIT
            mask_white_cpl = white_cpl[~mask]
            mask = black_cpl > CPL_LIMIT
            mask_black_cpl = black_cpl[~mask]

            white_cpl_mean = white_cpl.mean()
            black_cpl_mean = black_cpl.mean()

            mask_white_cpl_mean = mask_white_cpl.mean()
            mask_black_cpl_mean = mask_black_cpl.mean()

            white_won_flag = g["white_won"].iloc[0]
            black_won_flag = g["black_won"].iloc[0]
            draw_flag = g["no_winner"].iloc[0]

            if white_won_flag:
                white_result = 1
            elif draw_flag:
                white_result = 0.5
            else:
                white_result = 0

            elo_diff = white_elo - black_elo

            if len(white_cpl) > 0 and len(black_cpl) > 0:
                cpl_diff_mean = white_cpl.mean() - black_cpl.mean()
            else:
                cpl_diff_mean = float("nan")

            if len(mask_white_cpl) > 0 and len(mask_black_cpl) > 0:
                mask_cpl_diff_mean = mask_white_cpl.mean() - mask_black_cpl.mean()
            else:
                mask_cpl_diff_mean = float("nan")

            results.append({
                "game_id": game_id,
                "white_cpl_mean": white_cpl_mean,
                "black_cpl_mean": black_cpl_mean,
                "white_elo": white_elo,
                "elo_diff": elo_diff,
                "cpl_diff_mean": cpl_diff_mean,
                "white_player": g["white_player"].iloc[0],
                "black_player": g["black_player"].iloc[0],
                "win": white_result,
                "mask_white_cpl_mean": mask_white_cpl_mean,
                "mask_black_cpl_mean": mask_black_cpl_mean,
                "mask_cpl_diff_mean": mask_cpl_diff_mean
            })

        with open(output_path, 'a') as f:
            for result in results:
                f.write(','.join(map(str, result.values())) + '\n')

    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    process_games("./data/raw/lichess_db_standard_rated_2019-01.csv", "./data/cache/game_data.csv")
