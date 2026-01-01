import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Config
DATA_DIR = "3_4_captcha/data/archive"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
LABELS_FILE = os.path.join(DATA_DIR, "trainLabels.csv")

def prepare_data():
    print("Lecture du fichier CSV...")
    if not os.path.exists(LABELS_FILE):
        print(f"Erreur: Fichier {LABELS_FILE} introuvable.")
        return

    df = pd.read_csv(LABELS_FILE, names=['ID', 'Label'])
    
    # 1. Renommer les images avec extension .jpg si ce n'est pas déjà fait
    print("Renommage des images...")
    count = 0
    valid_indices = []
    
    # On parcourt le dataframe pour être sûr de traiter les fichiers existants
    for idx, row in df.iterrows():
        img_id = str(row['ID'])
        old_path = os.path.join(TRAIN_IMG_DIR, img_id)
        new_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.jpg")
        
        # Cas où le fichier existe sans extension
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            count += 1
            valid_indices.append(idx)
        # Cas où il a déjà été renommé (si on relance le script)
        elif os.path.exists(new_path):
            valid_indices.append(idx)
            pass
        else:
            # L'image référencée dans le CSV n'existe pas
            pass

    print(f"{count} images renommées en .jpg")
    
    # Filtrer le dataframe pour ne garder que les images trouvées
    df = df.loc[valid_indices].copy()
    
    # Ajouter l'extension au nom de fichier dans le dataframe pour faciliter la suite
    df['filename'] = df['ID'].astype(str) + ".jpg"
    
    # 2. Split Train / Validation
    print("Séparation Train / Validation (90/10)...")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Sauvegarde des nouveaux CSV
    train_df.to_csv(os.path.join(DATA_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val_split.csv"), index=False)
    
    print(f"Terminé. Train: {len(train_df)}, Val: {len(val_df)}")

if __name__ == "__main__":
    prepare_data()
