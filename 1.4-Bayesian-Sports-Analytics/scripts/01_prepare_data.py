import pandas as pd
import json

# Chargement
df = pd.read_csv("data/football_all_leagues.csv")

df = df[
    (df["League"] == "Premier League") &
    (df["Season"].isin(["2019-20", "2020-21", "2021-22"]))
].copy()

#mapping équipes → ID
teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
team2id = {team: i + 1 for i, team in enumerate(teams)}
id2team = {i + 1: team for team, i in team2id.items()}

df["home_id"] = df["HomeTeam"].map(team2id)
df["away_id"] = df["AwayTeam"].map(team2id)

# Sauvegarde
df.to_csv("data/premier_league_ready.csv", index=False)

with open("data/team_mapping.json", "w") as f:
    json.dump(team2id, f, indent=2)

print(f"{len(df)} matchs | {len(teams)} équipes")
