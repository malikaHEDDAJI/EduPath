# Instructions de Test - PrepaData Microservice

## Prérequis

Assurez-vous d'avoir Python 3.7+ installé.

## Étapes de test

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Vérifier que les données normalisées existent

Les fichiers suivants doivent être présents dans `data/processed/` :
- `student_info_normalized.csv`
- `courses_normalized.csv`
- `registrations_normalized.csv`
- `assessments_normalized.csv`
- `student_assessment_normalized.csv`
- `student_vle_normalized.csv`
- `vle_info_normalized.csv`

Si ces fichiers n'existent pas, exécutez d'abord le notebook `01_LMSConnector.ipynb`.

### 3. Tester le pipeline PrepaData

#### Option A : Test rapide en ligne de commande

```bash
python -c "from libs.prepa_data import run_prepa_data_pipeline; metrics = run_prepa_data_pipeline(); print(f'✓ Succès! {len(metrics)} lignes générées')"
```

#### Option B : Utiliser le notebook Jupyter

```bash
# Démarrer Jupyter
jupyter notebook

# Puis ouvrir et exécuter: notebooks/02_PrepaData.ipynb
```

#### Option C : Script Python interactif

```bash
python
```

Puis dans l'interpréteur Python :
```python
from libs.prepa_data import run_prepa_data_pipeline
import pandas as pd

# Exécuter le pipeline
metrics = run_prepa_data_pipeline()

# Vérifier les résultats
print(f"Shape: {metrics.shape}")
print(f"Colonnes: {list(metrics.columns)}")
print(metrics.head())
print(metrics.describe())
```

### 4. Vérifier le fichier généré

```bash
# Vérifier que le fichier existe
python -c "from pathlib import Path; f = Path('data/processed/student_module_metrics.csv'); print(f'✓ Fichier existe: {f.exists()}'); import pandas as pd; df = pd.read_csv(f); print(f'✓ Lignes: {len(df)}'); print(f'✓ Colonnes: {list(df.columns)}')"
```

## Résultat attendu

Le fichier `data/processed/student_module_metrics.csv` doit être créé avec :
- Environ 32 000+ lignes
- 8 colonnes : `student_id`, `code_module`, `code_presentation`, `avg_score`, `completion_rate`, `total_clicks`, `active_days`, `final_result`

## Dépannage

### Erreur : ModuleNotFoundError
```bash
# Réinstaller les dépendances
pip install -r requirements.txt
```

### Erreur : Fichiers CSV manquants
Exécutez d'abord le notebook `01_LMSConnector.ipynb` pour générer les données normalisées.

### Erreur : ImportError
Assurez-vous d'être dans le répertoire racine du projet (EduPath).

