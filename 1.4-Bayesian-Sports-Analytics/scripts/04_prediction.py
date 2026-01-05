import numpy as np
import pickle
import json

with open("data/fit.pkl", "rb") as f:
    fit = pickle.load(f)

with open("data/team_mapping.json") as f:
    team2id = json.load(f)

posterior = fit.draws_pd()

attack = posterior.filter(like="attack[").mean().values
defense = posterior.filter(like="defense[").mean().values
home_adv = posterior["home_adv"].mean()

def simulate_match(home, away, n=10000):
    h = team2id[home] - 1
    a = team2id[away] - 1

    lam_home = np.exp(home_adv + attack[h] - defense[a])
    lam_away = np.exp(attack[a] - defense[h])

    gh = np.random.poisson(lam_home, n)
    ga = np.random.poisson(lam_away, n)

    return {
        "home_win": (gh > ga).mean(),
        "Odds_home_win": 1/(gh > ga).mean(),
        "draw": (gh == ga).mean(),
        "Odds_draw": 1/(gh == ga).mean(),
        "away_win": (gh < ga).mean()
        ,"Odds_away_win": 1/(gh < ga).mean(),
    }

print(simulate_match("Man United", "Chelsea"))
