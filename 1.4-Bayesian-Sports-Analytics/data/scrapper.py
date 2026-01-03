import pandas as pd
import os
from datetime import datetime

def telecharger_donnees_championnat(championnat_code, championnat_nom, saisons):
    """
    Télécharge les données d'un championnat sur plusieurs saisons
    Ne garde que les colonnes importantes pour l'analyse bayésienne
    """
    all_data = []
    
    print(f"\n{'='*60}")
    print(f"📥 Téléchargement: {championnat_nom}")
    print(f"{'='*60}")
    
    for saison_code, saison_nom in saisons.items():
        url = f"https://www.football-data.co.uk/mmz4281/{saison_code}/{championnat_code}.csv"
        
        try:
            print(f"\n  Saison {saison_nom}...", end=" ")
            df = pd.read_csv(url, encoding='latin1')
            
            # Colonnes essentielles pour notre analyse bayésienne
            colonnes_importantes = {
                'Date': 'Date',
                'HomeTeam': 'HomeTeam', 
                'AwayTeam': 'AwayTeam',
                'FTHG': 'HomeGoals',      # Full Time Home Goals
                'FTAG': 'AwayGoals',      # Full Time Away Goals
                'FTR': 'Result',          # Full Time Result (H/D/A)
                'HS': 'HomeShots',        # Home Shots
                'AS': 'AwayShots',        # Away Shots
                'HST': 'HomeShotsTarget', # Home Shots on Target
                'AST': 'AwayShotsTarget', # Away Shots on Target
            }
            
            # Optionnel: cotes des bookmakers si disponibles
            colonnes_cotes = {
                'B365H': 'OddsHome',      # Bet365 Home odds
                'B365D': 'OddsDraw',      # Bet365 Draw odds
                'B365A': 'OddsAway',      # Bet365 Away odds
            }
            
            # Sélectionner les colonnes disponibles
            colonnes_a_garder = {}
            for old_col, new_col in colonnes_importantes.items():
                if old_col in df.columns:
                    colonnes_a_garder[old_col] = new_col
            
            # Ajouter les cotes si disponibles
            for old_col, new_col in colonnes_cotes.items():
                if old_col in df.columns:
                    colonnes_a_garder[old_col] = new_col
            
            # Renommer les colonnes
            df_clean = df[list(colonnes_a_garder.keys())].copy()
            df_clean.rename(columns=colonnes_a_garder, inplace=True)
            
            # Ajouter des métadonnées
            df_clean['Season'] = saison_nom
            df_clean['League'] = championnat_nom
            df_clean['LeagueCode'] = championnat_code
            
            # Supprimer les lignes avec données manquantes essentielles
            df_clean = df_clean.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals'])
            
            all_data.append(df_clean)
            print(f"{len(df_clean)} matchs")
            
        except Exception as e:
            print(f"Erreur: {e}")
            continue
    
    if all_data:
        df_final = pd.concat(all_data, ignore_index=True)
        print(f"\n  Total: {len(df_final)} matchs sur {len(all_data)} saisons")
        return df_final
    else:
        print(f"\n  Aucune donnée récupérée")
        return None


def main():
    """Script principal de téléchargement"""
    
    # Définir les championnats à télécharger
    championnats = {
        'F1': 'Ligue 1',
        'E0': 'Premier League',
        'SP1': 'La Liga',
        'I1': 'Serie A',
        'D1': 'Bundesliga',
    }
    
    # Définir les saisons (format: code_url: nom_lisible)
    saisons = {
        '2324': '2023-24',
        '2223': '2022-23',
        '2122': '2021-22',
        '2021': '2020-21',
        '1920': '2019-20',
    }
    
    print("=" * 60)
    print("TÉLÉCHARGEMENT DES DONNÉES FOOTBALL")
    print("=" * 60)
    print(f"Championnats: {len(championnats)}")
    print(f"Saisons: {len(saisons)}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Télécharger chaque championnat
    datasets = {}
    for code, nom in championnats.items():
        df = telecharger_donnees_championnat(code, nom, saisons)
        if df is not None:
            datasets[code] = df
            # Sauvegarder individuellement
            filename = f'{code}_{nom.replace(" ", "_")}.csv'
            df.to_csv(filename, index=False)
            print(f"  Sauvegardé: {filename}")
    
    # Combiner tous les championnats
    if datasets:
        print(f"\n{'='*60}")
        print("Création du dataset combiné...")
        print(f"{'='*60}")
        
        df_all = pd.concat(datasets.values(), ignore_index=True)
        
        # Sauvegarder le dataset complet
        filename_all = 'football_all_leagues.csv'
        df_all.to_csv(filename_all, index=False)
        
        print(f"\nTERMINÉ!")
        print(f"  Total matchs: {len(df_all)}")
        print(f"  Championnats: {df_all['League'].nunique()}")
        print(f"  Équipes uniques: {pd.concat([df_all['HomeTeam'], df_all['AwayTeam']]).nunique()}")
        print(f"  Fichier principal: {filename_all}")
        
        # Statistiques rapides
        print(f"\nRépartition par championnat:")
        for league in df_all['League'].unique():
            count = len(df_all[df_all['League'] == league])
            print(f"  {league:20s}: {count:4d} matchs")
        
        return df_all
    else:
        print("\nAucune donnée téléchargée")
        return None


if __name__ == "__main__":
    df = main()
    
    if df is not None:
        print(f"\n{'='*60}")
        print("APERÇU DES DONNÉES")
        print(f"{'='*60}")
        print(df.head(10))
        print(f"\nColonnes: {list(df.columns)}")