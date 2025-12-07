# Learning Analytics Platform

Voici l'organisation du projet `learning-analytics-platform` :

```text
learning-analytics-platform/
│
├── data/
│   ├── raw/                # Données brutes exportées des LMS (CSV, JSON, logs…)
│   ├── processed/          # Données nettoyées + features
│   ├── models/             # Modèles ML sauvegardés (.pkl, .json)
│   ├── resources/          # PDF, vidéos, exercices, FAQ…
│   └── recommendations/    # Recommandations finales générées
│
├── notebooks/
│   ├── 01_LMSConnector.ipynb        # Import + normalisation des données
│   ├── 02_PrepaData.ipynb           # Nettoyage + feature engineering
│   ├── 03_StudentProfiler.ipynb     # Clustering + PCA
│   ├── 04_PathPredictor.ipynb       # Modèle de prédiction XGBoost
│   ├── 05_RecoBuilder.ipynb         # Recommandations BERT + FAISS
│   └── 06_Dashboard.ipynb           # Tableau de bord final / visualisation
│
├── libs/                    # Modules Python réutilisables
│   ├── lms_connector.py
│   ├── prepa_data.py
│   ├── profiler.py
│   ├── predictor.py
│   ├── recommender.py
│   └── utils.py             # fonctions communes (plots, logs…)
│
├── environment.yml          # Conda environment (BERT + XGBoost + Pandas)
├── README.md                # Documentation
└── requirements.txt         # Librairies (si pas conda)
