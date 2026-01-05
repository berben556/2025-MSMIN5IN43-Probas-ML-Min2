import pandas as pd
from cmdstanpy import CmdStanModel
import pickle

df = pd.read_csv("data/premier_league_ready.csv")

stan_data = {
    "N": len(df),
    "T": max(df["home_id"].max(), df["away_id"].max()),
    "home_team": df["home_id"].values,
    "away_team": df["away_id"].values,
    "home_goals": df["HomeGoals"].values.astype(int),
    "away_goals": df["AwayGoals"].values.astype(int),
}

model = CmdStanModel(stan_file="stan/football_model.stan")

fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    adapt_delta=0.95
)

with open("data/fit.pkl", "wb") as f:
    pickle.dump(fit, f)

summary = fit.summary()
cols = [c for c in summary.columns if "R_hat" in c or "Eff" in c]
print(summary[cols])