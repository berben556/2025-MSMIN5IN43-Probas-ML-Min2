import pandas as pd
import numpy as np
import pickle
import json

# Charger le modèle (comme dans script 04)
with open("data/fit.pkl", "rb") as f:
    fit = pickle.load(f)

with open("data/team_mapping.json") as f:
    team2id = json.load(f)

posterior = fit.draws_pd()
attack = posterior.filter(like="attack[").mean().values
defense = posterior.filter(like="defense[").mean().values
home_adv = posterior["home_adv"].mean()


def simulate_match(home, away, n=10000):
    """Fonction de prédiction (identique au script 04)"""
    h = team2id[home] - 1
    a = team2id[away] - 1

    lam_home = np.exp(home_adv + attack[h] - defense[a])
    lam_away = np.exp(attack[a] - defense[h])

    gh = np.random.poisson(lam_home, n)
    ga = np.random.poisson(lam_away, n)

    return {
        "home_win": (gh > ga).mean(),
        "draw": (gh == ga).mean(),
        "away_win": (gh < ga).mean(),
    }


def compare_match(home_team, away_team):
    """Compare les prédictions du modèle avec les cotes des bookmakers"""
    
    # Vérifier que les équipes existent
    if home_team not in team2id or away_team not in team2id:
        print(f"Erreur: Equipe inconnue. Equipes disponibles:")
        print(sorted(team2id.keys()))
        return
    
    # Prédiction du modèle
    pred = simulate_match(home_team, away_team)
    
    # Charger les données avec cotes bookmakers
    df = pd.read_csv("data/premier_league_ready.csv")
    
    # Chercher les matchs historiques entre ces équipes
    matches = df[(df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)].copy()
    
    print(f"\n{'='*60}")
    print(f"{home_team} vs {away_team}")
    print(f"{'='*60}")
    
    if len(matches) == 0:
        print(f"Aucun match historique trouvé")
        print(f"\nPredictions du modele:")
        print(f"  Victoire {home_team:20s}: {pred['home_win']:.1%}  (Cote: {1/pred['home_win']:.2f})")
        print(f"  Match nul{' '*20}: {pred['draw']:.1%}  (Cote: {1/pred['draw']:.2f})")
        print(f"  Victoire {away_team:20s}: {pred['away_win']:.1%}  (Cote: {1/pred['away_win']:.2f})")
        return
    
    # Calculer les probabilités implicites des bookmakers
    matches["p_home_book"] = 1 / matches["OddsHome"]
    matches["p_draw_book"] = 1 / matches["OddsDraw"]
    matches["p_away_book"] = 1 / matches["OddsAway"]
    
    # Normalisation (enlever la marge du bookmaker)
    total = matches["p_home_book"] + matches["p_draw_book"] + matches["p_away_book"]
    matches["p_home_book"] /= total
    matches["p_draw_book"] /= total
    matches["p_away_book"] /= total
    
    # Moyennes
    p_h_book = matches["p_home_book"].mean()
    p_d_book = matches["p_draw_book"].mean()
    p_a_book = matches["p_away_book"].mean()
    
    print(f"Matchs historiques analyses: {len(matches)}\n")
    
    print("MODELE:")
    print(f"  Victoire {home_team:20s}: {pred['home_win']:.1%}  (Cote: {1/pred['home_win']:.2f})")
    print(f"  Match nul{' '*20}: {pred['draw']:.1%}  (Cote: {1/pred['draw']:.2f})")
    print(f"  Victoire {away_team:20s}: {pred['away_win']:.1%}  (Cote: {1/pred['away_win']:.2f})")
    
    print(f"\nBOOKMAKERS (moyenne sur {len(matches)} matchs):")
    print(f"  Victoire {home_team:20s}: {p_h_book:.1%}  (Cote: {1/p_h_book:.2f})")
    print(f"  Match nul{' '*20}: {p_d_book:.1%}  (Cote: {1/p_d_book:.2f})")
    print(f"  Victoire {away_team:20s}: {p_a_book:.1%}  (Cote: {1/p_a_book:.2f})")
    
    print("\nDIFFERENCE (Modele - Bookmakers):")
    diff_h = pred['home_win'] - p_h_book
    diff_d = pred['draw'] - p_d_book
    diff_a = pred['away_win'] - p_a_book
    
    print(f"  Victoire {home_team:20s}: {diff_h:+.1%}")
    print(f"  Match nul{' '*20}: {diff_d:+.1%}")
    print(f"  Victoire {away_team:20s}: {diff_a:+.1%}")
    
    # Identifier les paris à valeur
    max_diff = max(abs(diff_h), abs(diff_d), abs(diff_a))
    if max_diff > 0.05:
        if abs(diff_h) == max_diff:
            result = f"Victoire {home_team}" if diff_h > 0 else f"Contre {home_team}"
        elif abs(diff_d) == max_diff:
            result = "Match nul" if diff_d > 0 else "Contre match nul"
        else:
            result = f"Victoire {away_team}" if diff_a > 0 else f"Contre {away_team}"
        print(f"\n=> VALEUR POTENTIELLE: {result}")
    else:
        print("\n=> Pas de difference significative")


if __name__ == "__main__":
    # Exemples
    compare_match("Man United", "Chelsea")

