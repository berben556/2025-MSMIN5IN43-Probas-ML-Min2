# 2025 - MSMIN5IN43 - IA probabiliste, théorie de jeux et machine learning

Projet pédagogique d'exploration des approches d'intelligence artificielle probabilistes, de la théorie des jeux et du machine learning pour les étudiants de l'EPF.

---

## 📅 Modalités du projet

### Échéances importantes
- **15 décembre 2025** : Présentation des sujets proposés
- **5 janvier 2026** : Deadline de soumission des projets par Pull Request sur ce dépôt
- **6 janvier 2026** : Présentation finale et rendu

### Date de livraison
Le code avec le README devront être livrés dans un sous-dossier de ce dépôt pour chaque groupe 1 jour au plus tard avant la présentation.

### Taille des groupes
La taille standard d'un groupe est de **3 personnes**.
- Groupes de 2 : toléré (+1 point bonus potentiel pour la charge)
- Groupes de 4 : toléré (-1 point malus potentiel pour la dilution)
- Individuel : exceptionnel (+3 points bonus potentiel)

### Évaluation collégiale
L'évaluation portera sur :
1.  **Présentation/Communication** : Clarté, pédagogie, qualité des slides.
2.  **Contenu théorique** : Compréhension des enjeux, état de l'art, contexte.
3.  **Contenu technique** : Qualité du code, résultats obtenus, démos.
4.  **Organisation/Collaboration** : Activité Git, répartition du travail.

### Livrables attendus
- **Code source** propre et documenté.
- **README** complet (contexte, installation, usage, résultats).
- **Slides** de la présentation (PDF ou lien).

---

## 💡 Liste des sujets proposés

Vous êtes libres de choisir l'un des sujets ci-dessous ou de proposer un sujet personnel (à faire valider par les encadrants).
**Technologie libre** : Python (recommandé pour l'écosystème ML), C#/.NET (historique du cours), C++, Julia, etc.

### 🎲 Catégorie 1 : IA Probabiliste & Modèles Graphiques

Ces sujets explorent l'incertitude, l'inférence bayésienne et la modélisation statistique. Ils demandent une bonne compréhension des distributions de probabilités et des graphes de facteurs.

#### 1.1. TrueSkill & Matchmaking (Compétition)
Le classement de joueurs dans les jeux en ligne (Xbox Live, LoL, Chess) est un problème probabiliste complexe. Au-delà du simple système ELO, le système TrueSkill utilise des graphes de facteurs pour modéliser l'incertitude sur la compétence de chaque joueur (une gaussienne avec moyenne et variance).
- **Travail attendu** :
    - Implémenter un moteur d'inférence (via Expectation Propagation ou Variational Inference) pour mettre à jour les scores après chaque match.
    - Visualiser la convergence de l'incertitude (sigma) au fil des parties.
- **Extensions** : Gérer les équipes hétérogènes, le "draw margin" (probabilité de nul), ou la dynamique temporelle (un joueur progresse ou régresse).
- **Ressources** :
    - [Papier TrueSkill (Microsoft)](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/)
    - [TrueSkill 2 (Papier récent)](https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/)
    - [Chapitre du livre MBML](http://mbmlbook.com/TrueSkill.html)

#### 1.2. Inférence Causale (Causal Inference)
Corrélation n'est pas causalité. Comment savoir si une promo a *causé* une vente ou si c'est juste la saisonnalité ?
- **Objectif** : Estimer l'effet causal moyen (ATE) d'une intervention (traitement médical, politique publique) à partir de données observationnelles.
- **Outils** : Utiliser **Pyro** ou **DoWhy** pour modéliser les contrefactuels.
- **Ressources** :
    - [Tutoriel Causal Inference avec Pyro](https://pyro.ai/examples/intro_long.html)
    - [DoWhy Library](https://microsoft.github.io/dowhy/)

#### 1.3. Marketing Mix Modeling (MMM) Bayésien
Un sujet très demandé en entreprise : optimiser le budget pub.
- **Problème** : Attribuer les ventes aux différents canaux (TV, Facebook, Google) sachant qu'il y a des effets de saturation (rendements décroissants) et de délai (Adstock).
- **Approche** : Utiliser **PyMC** pour construire un modèle hiérarchique qui estime ces paramètres inconnus.
- **Ressources** :
    - [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing)
    - [Google LightweightMMM](https://github.com/google/lightweight_mmm)

#### 1.4. Bayesian Sports Analytics
Prédire les résultats sportifs mieux que les bookmakers.
- **Objectif** : Modéliser la force des équipes (attaque/défense) dans un championnat (Foot, NBA) en prenant en compte l'avantage du terrain.
- **Technique** : Modèles hiérarchiques sous **Stan** (via CmdStanPy ou RStan).
- **Ressources** :
    - [Stan Case Studies: Sports](https://mc-stan.org/users/documentation/case-studies.html)
    - [Baio & Blangiardo (2010) - Hierarchical model for Serie A](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf)

#### 1.5. Bayesian Neural Networks (BNNs)
Le pont entre le Deep Learning et les Probabilités.
- **Concept** : Au lieu d'avoir des poids fixes, chaque poids du réseau de neurones est une distribution de probabilité. Cela permet au réseau de dire "je ne sais pas" (incertitude épistémique).
- **Travail attendu** : Implémenter un BNN simple sous **Pyro** ou **TyXe** pour la classification d'images et visualiser l'incertitude sur des exemples hors distribution (OOD).
- **Ressources** : [TyXe (Pyro BNNs)](https://github.com/cifkao/tyxe), [Tutoriel Pyro BNN](https://pyro.ai/examples/bnn.html).

#### 1.6. Bio-informatique : Identification de motifs & Santé
La biologie regorge de données bruitées où les modèles probabilistes excellent.
- **Sujet A : Motif Finder (HMM)**.
    - Le problème : Retrouver des patterns cachés (ex: sites de liaison de protéines) dans des séquences d'ADN longues et bruitées.
    - L'approche : Utiliser un Modèle de Markov Caché (HMM) ou un modèle de mélange pour séparer le signal du bruit de fond.
    - [Tutoriel Motif Finder](https://dotnet.github.io/infer/userguide/Motif%20Finder.html)
- **Sujet B : Compréhension de l'asthme**.
    - Le problème : Modéliser les relations causales complexes entre génétique, environnement et symptômes.
    - L'approche : Construire un Réseau Bayésien pour effectuer des diagnostics probabilistes et de l'inférence causale.
    - [Chapitre Asthma (MBML)](http://mbmlbook.com/Asthma.html)

#### 1.7. Modèles Probabilistes Modernes (Pyro / Gaussian Processes)
Explorez les frameworks probabilistes modernes sous Python qui combinent Deep Learning et Probabilités.
- **Sujet A : Rational Speech Acts (RSA)**.
    - Modéliser la pragmatique du langage : comment un locuteur choisit ses mots pour être compris, et comment un auditeur interprète l'ambiguïté (ironie, hyperbole).
    - Utiliser le framework **Pyro** (basé sur PyTorch) pour simuler ces agents récursifs.
    - [Tutoriel Pyro RSA](https://pyro.ai/examples/RSA-implicature.html)
- **Sujet B : Processus Gaussiens (Gaussian Processes)**.
    - Une méthode puissante pour la régression non-paramétrique, offrant une estimation de l'incertitude "gratuite". Idéal pour les données spatiales (géologie) ou temporelles.
    - Utiliser **GPyTorch** pour passer à l'échelle sur GPU.
    - [Deep Kernel Learning](https://arxiv.org/abs/1511.02222) : Apprendre le noyau (kernel) du GP avec un réseau de neurones.

#### 1.8. Physics-Informed Neural Networks (PINNs)
Un domaine en pleine explosion : utiliser le Deep Learning pour résoudre des équations différentielles partielles (PDEs) en physique (mécanique des fluides, chaleur).
- **Concept** : Au lieu d'entraîner le réseau seulement sur des données, on ajoute un terme dans la fonction de perte qui pénalise le non-respect des équations physiques (ex: Navier-Stokes).
- **Travail attendu** : Résoudre une équation simple (ex: Burgers ou Heat Equation) avec un PINN et comparer avec une résolution numérique classique.
- **Ressources** :
    - [DeepXDE Library](https://deepxde.readthedocs.io/en/latest/)
    - [Papier fondateur PINNs](https://arxiv.org/abs/1711.10561)

---

### ♟️ Catégorie 2 : Théorie des Jeux & Systèmes Multi-Agents

Ces sujets traitent de la prise de décision stratégique, de la coopération et de la compétition entre agents autonomes.

#### 2.1. Poker AI & Information Imparfaite
Le Poker est le "drosophile" de l'IA en information imparfaite (on ne voit pas les cartes de l'adversaire). C'est un problème bien plus dur que les Échecs ou le Go.
- **Technique clé** : **Counterfactual Regret Minimization (CFR)**. L'agent apprend en minimisant son "regret" d'avoir joué une action plutôt qu'une autre a posteriori.
- **Travail attendu** :
    - Implémenter un algorithme CFR (ou MCCFR) sur une version simplifiée du Poker (Leduc Hold'em ou Kuhn Poker).
    - Analyser la stratégie obtenue (Nash Equilibrium).
- **Ressources** :
    - [OpenSpiel (DeepMind)](https://github.com/deepmind/open_spiel)
    - [Libratus](https://science.sciencemag.org/content/359/6374/418) et [Pluribus](https://science.sciencemag.org/content/365/6456/885).

#### 2.2. Hanabi AI : Coopération & Theory of Mind
Hanabi est un jeu de cartes coopératif unique où l'on voit les cartes des autres mais pas les siennes. Il faut communiquer des indices limités.
- **Défi** : L'agent doit modéliser ce que les autres savent ("Theory of Mind") et interpréter les indices comme des signaux implicites.
- **Travail attendu** : Entraîner un agent RL (ex: Rainbow DQN ou PPO) capable de jouer avec des humains ou d'autres bots.
- **Ressources** :
    - [Hanabi Learning Environment](https://github.com/deepmind/hanabi-learning-environment)
    - [The Hanabi Challenge (Papier)](https://arxiv.org/abs/1902.00506)

#### 2.3. Stratego AI : Bluff & Planification (DeepNash)
Stratego est un jeu de plateau à information imparfaite (pièces cachées) qui nécessite du bluff et une planification à long terme.
- **Technique** : **R-NaD (Regularized Nash Dynamics)**. Une approche sans recherche arborescente (MCTS) qui converge vers un équilibre de Nash.
- **Objectif** : Implémenter une version simplifiée de R-NaD sur un mini-Stratego.
- **Ressources** : [DeepNash (DeepMind)](https://www.deepmind.com/blog/mastering-stratego-the-classic-game-of-imperfect-information).

#### 2.4. Mean Field Games (Jeux à Champ Moyen)
Comment modéliser l'interaction stratégique d'une foule immense (ex: traders sur un marché, banc de poissons) ?
- **Concept** : Au lieu de modéliser N agents, on modélise un agent représentatif face à une "distribution moyenne" des autres.
- **Approche ML** : Utiliser des réseaux de neurones (Neural ODEs) pour résoudre les équations différentielles stochastiques couplées (Hamilton-Jacobi-Bellman + Fokker-Planck).
- **Ressources** : [Mean Field Games & ML (Papier)](https://arxiv.org/abs/2003.06069), [Tutoriel MFG](https://github.com/Nathan-Sanglier/M2MO-Mean-Field-Games).

#### 2.5. Deep Learning for Mechanism Design (Enchères)
Concevoir des règles économiques (enchères) optimales pour maximiser le revenu, via le Deep Learning ("Differentiable Economics").
- **Problème** : Concevoir une enchère multi-objets optimale est mathématiquement impossible analytiquement.
- **Solution** : Entraîner un réseau de neurones (RegretNet) qui prend en entrée les valorisations des acheteurs et sort les allocations et les prix, en maximisant le revenu sous contrainte d'incitation (IC).
- **Ressources** : [Optimal Auctions through Deep Learning](https://arxiv.org/abs/1905.05533), [GitHub RegretNet](https://github.com/srp3/regretnet).

#### 2.6. Théorie des Jeux appliquée à la Santé & Biologie
La théorie des jeux ne sert pas qu'à jouer, elle modélise le vivant et la société.
- **Sujet A : Échange de reins (Kidney Exchange)**.
    - Problème : Des patients ont des donneurs incompatibles. Comment organiser des chaînes d'échanges croisés pour sauver le maximum de vies ?
    - C'est un problème d'optimisation combinatoire et de théorie des jeux coopératifs.
    - [Travaux de Tuomas Sandholm](http://www.cs.cmu.edu/~sandholm/)
- **Sujet B : Théorie des jeux évolutionniste**.
    - Modéliser pourquoi certains comportements (altruisme, agressivité) survivent dans une population.
    - Simuler des dynamiques de type "Hawk-Dove" ou "Rock-Paper-Scissors" dans des populations biologiques.

---

### 🧠 Catégorie 3 : Machine Learning Avancé & Deep Learning

Sujets classiques mais exigeants, nécessitant une rigueur méthodologique (gestion des données, métriques, validation).

#### 3.1. Trading Algorithmique & Finance Quantitative
La finance quantitative est un terrain de jeu idéal pour les séries temporelles et le RL.
- **Plateforme** : Utiliser **[QuantConnect](https://www.quantconnect.com/)** (moteur LEAN). C'est une plateforme professionnelle qui permet de backtester des stratégies en Python/C# sur des données historiques de haute qualité.
- **Sujets** :
    - **Stratégie Alpha** : Créer un algo qui bat le marché (S&P500) sur 5 ans.
    - **GANs in Finance** : Utiliser des GANs (TimeGAN) pour générer des données synthétiques de marché et entraîner des modèles de manière plus robuste.
    - **Sentiment Analysis** : Trader en fonction des news financières (NLP sur titres de presse).

#### 3.2. Vision par Ordinateur : Santé & Diagnostic
L'IA pour l'aide au diagnostic médical est un enjeu éthique et technique majeur.
- **Sujets** :
    - **Détection de tumeurs** : Segmentation d'images IRM ou histopathologiques.
    - **Classification de radiographies** : Détecter pneumonie/COVID sur des radios thoraciques (Dataset CheXNet).
- **Défis** : Travailler avec des données très déséquilibrées (peu de cas malades) et fournir des cartes de chaleur (Grad-CAM) pour expliquer la décision au médecin.

#### 3.3. NLP Avancé : Analyse de Sentiment & Fake News
Le traitement du langage naturel (NLP) a été révolutionné par les Transformers.
- **Sujet A : Détection de Fake News**.
    - Entraîner un modèle (BERT/RoBERTa) pour classifier des articles comme fiables ou non, en se basant sur le style, le vocabulaire et la source.
- **Sujet B : Analyse de Sentiment Fine**.
    - Ne pas se limiter à Positif/Négatif. Détecter l'ironie, le sarcasme, ou des émotions spécifiques (colère, joie, peur) dans des tweets ou commentaires.
- **Outils** : [HuggingFace Transformers](https://huggingface.co/transformers/), [CamemBERT](https://camembert-model.fr/).

#### 3.4. Résolution de Captcha par Deep Learning
Un classique de la vision par ordinateur qui combine segmentation et reconnaissance de caractères (OCR).
- **Objectif** : Entraîner un modèle capable de lire des captchas alphanumériques bruités.
- **Méthode** :
    - Générer son propre dataset de captchas synthétiques.
    - Utiliser un CNN pour l'extraction de features et un RNN (LSTM/GRU) avec CTC loss pour la lecture de séquence, ou une approche purement attentionnelle (Vision Transformer).
- **Ressources** : [Kaggle Captcha Dataset](https://www.kaggle.com/codingnirvana/captcha-images).

#### 3.5. Reinforcement Learning (RL) : Contrôle & Jeux
Apprendre par essai-erreur dans un environnement dynamique.
- **Sujet** : Apprendre à un agent à jouer à un jeu vidéo (Snake, Mario, Doom) ou à contrôler un système physique (pendule inversé, atterrisseur lunaire).
- **Algos** : Comparer les performances de PPO (Proximal Policy Optimization), DQN (Deep Q-Network) et SAC (Soft Actor-Critic).
- **Lib** : [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [Gymnasium](https://gymnasium.farama.org/).

---

### 🚀 Catégorie 4 : Confidentialité & ML (Privacy Preserving ML)

Comment entraîner des modèles sans voir les données ? Sujet critique pour la santé et la banque (RGPD).

#### 4.1. Chiffrement Homomorphe
Le Saint Graal de la privacy : effectuer des calculs (inférence ML) directement sur des données chiffrées, sans jamais les déchiffrer.
- **Travail attendu** : Utiliser une librairie spécialisée pour entraîner un modèle simple (Régression, Arbre de décision) qui peut prédire sur des données chiffrées.
- **Ressources** :
    - [Microsoft SEAL](https://github.com/Microsoft/SEAL)
    - [Concrete ML (Zama)](https://github.com/zama-ai/concrete-ml) : Permet de convertir des modèles Scikit-learn en équivalents chiffrés.

#### 4.2. Federated Learning (Apprentissage Fédéré)
Entraîner un modèle global sur des données décentralisées (ex: téléphones utilisateurs, hôpitaux) sans jamais centraliser les données brutes.
- **Concept** : Le modèle voyage vers les données, apprend localement, et renvoie uniquement les mises à jour de poids (gradients) au serveur central.
- **Ressources** : [TensorFlow Federated](https://www.tensorflow.org/federated), [PySyft](https://github.com/OpenMined/PySyft).

---

### 🔬 Catégorie 5 : Recherche & Innovation (2024-2025)

Sujets exploratoires basés sur des publications récentes (NeurIPS, ICML). Pour les étudiants qui veulent toucher à la recherche.

#### 5.1. GFlowNets (Generative Flow Networks)
Une nouvelle famille de modèles génératifs probabilistes (introduite par Yoshua Bengio) conçue pour échantillonner des objets composites (molécules, graphes) proportionnellement à une récompense.
- **Application** : Découverte de médicaments (générer des molécules valides avec haute affinité) ou génération de plans.
- **Travail attendu** : Implémenter un GFlowNet simple sur un environnement de grille ou de génération de chaînes de caractères.
- **Ressources** : [Tutoriel GFlowNet (Mila)](https://mila.quebec/fr/article/gflownet-tutorial), [TorchGFN Library](https://github.com/GFNOrg/torchgfn).

#### 5.2. Diffusion for Combinatorial Optimization (DIFUSCO)
Utiliser les modèles de diffusion (ceux qui génèrent des images) pour résoudre des problèmes d'optimisation discrète difficiles (NP-hard).
- **Concept** : Transformer le problème du Voyageur de Commerce (TSP) ou du SAT en un problème de débruitage. Le modèle apprend à reconstruire la solution optimale à partir de bruit.
- **Ressources** : [Papier DIFUSCO](https://arxiv.org/abs/2302.08224), [Dépôt GitHub](https://github.com/zuwang12/DIFUSCO).

#### 5.3. Liquid Neural Networks (LNNs)
Une nouvelle architecture de réseaux de neurones inspirée du cerveau (C. elegans), capable d'adapter sa dynamique en temps continu.
- **Avantage** : Extrêmement robuste aux données bruitées et capable de généraliser hors distribution (OOD) mieux que les RNNs classiques.
- **Application** : Pilotage de drone, prédiction de séries temporelles financières ou médicales.
- **Ressources** : [Liquid Time-constant Networks (GitHub)](https://github.com/raminmh/liquid_time_constant_networks), [Papier Nature Machine Intelligence](https://www.nature.com/articles/s42256-020-00267-3).

#### 5.4. Conformal Prediction (Quantification d'Incertitude)
Comment garantir qu'une prédiction est "sûre" ? La prédiction conforme permet de générer des intervalles de confiance valides mathématiquement, quel que soit le modèle sous-jacent.
- **Travail attendu** : Prendre un modèle "boîte noire" (ex: Random Forest ou Réseau de Neurones) et utiliser la prédiction conforme pour transformer ses prédictions ponctuelles en intervalles (ex: "Le prix est entre 10€ et 12€ avec 95% de certitude").
- **Ressources** : [MAPIE (Library Python)](https://github.com/scikit-learn-contrib/MAPIE), [Awesome Conformal Prediction](https://github.com/valeman/awesome-conformal-prediction).

#### 5.5. World Models (DreamerV3)
En Reinforcement Learning, au lieu d'apprendre juste une politique, l'agent apprend un "modèle du monde" (comment l'environnement réagit) et rêve dans ce modèle pour s'entraîner.
- **Objectif** : Implémenter une version simplifiée d'un World Model sur un jeu simple (Minigrid ou Atari).
- **Ressources** : [DreamerV3 (Papier)](https://arxiv.org/abs/2301.04104), [Dépôt GitHub](https://github.com/danijar/dreamerv3).

---

## 📚 Ressources Générales

- **HuggingFace** : Pour les modèles et datasets (NLP, CV, Audio).
- **Kaggle** : Pour trouver des datasets propres et des notebooks d'exemple.
- **PapersWithCode** : Pour trouver l'état de l'art sur une tâche donnée.
- **ArXiv** : Pour les papiers de recherche originaux.
