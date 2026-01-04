Vision par Ordinateur : Santé & Diagnostic
L'IA pour l'aide au diagnostic médical est un enjeu éthique et technique majeur.

Sujets :
Détection de tumeurs : Segmentation d'images IRM ou histopathologiques.
Classification de radiographies : Détecter pneumonie/COVID sur des radios thoraciques (Dataset CheXNet).
Défis : Travailler avec des données très déséquilibrées (peu de cas malades) et fournir des cartes de chaleur (Grad-CAM) pour expliquer la décision au médecin.

# Projet : Détection de tumeurs IRM avec Grad-CAM

## 1. Contexte

Ce projet a pour objectif d’explorer et d’analyser un dataset d’images IRM cérébrales, puis d’utiliser un modèle de deep learning pour détecter la présence de tumeurs et visualiser les zones d’attention à l’aide de Grad-CAM.  

Les étapes principales sont :

1. **Exploration du dataset** : visualisation d’images et inspection des labels.  
2. **Analyse des données** : distribution des classes, équilibrage, statistiques.  
3. **Détection avec Grad-CAM** : utilisation d’un modèle CNN (ResNet18) pour classifier les images et générer des cartes de chaleur interprétables.

L’objectif est de fournir une aide à l’interprétation médicale tout en conservant l’explicabilité des décisions du modèle.

---

## 2. Structure du projet

```bash
tumor-detection/
│
├─ data/
| └─ processed/
| | └─ test/
| | └─ train/
│ └─ raw/
|   └─Testing
│   └─Training/ # Images IRM organisées par classe
│
├─ notebooks
| └─01_exploration_dataset.ipynb
| └─02_data_analysis.ipynb
| └─03_gradcam_demo.ipynb
|
├─ src/
│ ├─ pycache/
│ ├─ datasets/
│ │ └─ mri_dataset.py
│ ├─ models/
│ ├─training/
│ ├─utils/
│ ├─ __init__.py
│ └─ preprocess.py
│
├─ README.md
└─ requirements.txt
└─ app.py
```
---

## 3. Installation

### 3.1 Prérequis

- Python 3.10 ou 3.11  
- Pip  
- GPU NVIDIA compatible CUDA (optionnel, le CPU fonctionne aussi)  

### 3.2 Installer les dépendances

```bash
# Créer un environnement virtuel recommandé
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 3.3 Contenu du requirements.txt

```bash
# Core
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
Pillow>=10.0.0
scikit-learn>=1.3.0

# PyTorch
torch>=2.2.0
torchvision>=0.17.0

# Notebook execution
nbclient>=0.7.4
nbformat>=5.9.0

# Front-end
streamlit>=1.22.0

# Image processing
opencv-python>=4.7.0

# Utilities
pathlib>=1.0.1
tqdm
```

---

## 4. Usage

### 4.1 Exploration du dataset

```bash
python notebooks/01_exploration_dataset.ipynb
```
Affiche plusieurs images IRM avec leur label.
Permet de vérifier l’intégrité et la structure du dataset.

### 4.2 Analyse des données

```bash
python notebooks/02_data_analysis.ipynb
```
Compte le nombre d’images par classe.
Affiche un graphique de distribution pour identifier les déséquilibres.

### 4. Grad-CAM

```bash
python notebooks/03_gradcam_demo.ipynb
```
Charge le modèle (ResNet18) et adapte la dernière couche à 2 classes.
Prépare l’image et génère le Grad-CAM.
Superpose la carte de chaleur sur l’image originale pour visualiser les zones importantes pour la décision du modèle.

---

## 5. Résulatats attendus

Exploration dataset : affichage de 8 images aléatoires avec leur label.
Analyse des données : graphique barre avec le nombre d’images par classe (tumor / no_tumor).
Grad-CAM : image IRM originale avec superposition d’une heatmap indiquant les zones influençant la décision du modèle.

---

## 6. Bonnes pratiques

Toujours exécuter le pipeline dans l’ordre : exploration → analyse → Grad-CAM.
Vérifier les dimensions des images avant le passage au modèle (512x512 recommandé).
Sauvegarder les cartes Grad-CAM pour interprétation médicale.

---

## 7. Excécution

```bash
python -m streamlit run app.py
```
Affichage du front avec streamlite, 3 boutons permettant chaqu'un d'afficher l'image associée à chaque notebook :
- Dataset Visualization
- Analysis Visualization
- Grad-CAM Result

## Auteurs
CARLINO Chiara, FINET Lucille, PAOLANTONI Jules