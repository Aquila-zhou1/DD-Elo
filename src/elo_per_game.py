def calculate_new_elo(my_elo, op_elo, win, k=40):
    # Ea = 1 / (1 + 10 ^ ((Rb - Ra) / 400))
    expected_score = 1 / (1 + 10 ** ((op_elo - my_elo) / 400))
    # Ra_new = Ra + K * (Sa - Ea)
    new_elo = my_elo + k * (win - expected_score)
    
    return new_elo