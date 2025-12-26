import pandas as pd
import os


class ApiBridge:
    def __init__(self):
        # Chemins vers vos CSV normalisés
        self.data_path = "data/processed/"
        self.load_data()

    def load_data(self):
        try:
            self.students = pd.read_csv(os.path.join(self.data_path, "student_info_normalized.csv"))
            self.metrics = pd.read_csv(os.path.join(self.data_path, "student_module_metrics.csv"))
        except Exception as e:
            print(f"Erreur chargement CSV: {e}")

    def get_prediction(self, student_id, module_code):
        name = f"Étudiant {student_id}"
        avg_score = 75.0

        # Recherche sécurisée
        try:
            if hasattr(self, 'students') and not self.students.empty:
                # On teste les noms de colonnes courants (id_student ou student_id)
                col = 'id_student' if 'id_student' in self.students.columns else 'student_id'
                student_data = self.students[self.students[col] == student_id]

                if not student_data.empty:
                    name = f"Profil Réel (#{student_id})"
                    # Si vous avez une colonne score, utilisez-la, sinon 75.0
                    avg_score = student_data.iloc[0].get('avg_score', 75.0)
        except Exception as e:
            print(f"⚠️ Erreur lookup étudiant: {e}")
        # Logique de probabilité simple (sera remplacée par XGBoost)
        proba = min(float(avg_score) / 100.0, 0.95)
        risk = "Low" if proba > 0.8 else "Medium" if proba > 0.6 else "High"
        return {
            "student_id": student_id,
            "module_code": module_code,
            "success_proba": round(proba, 2),
            "risk_level": risk,
            "message": f"Analyse générée pour {name}"
        }

    def get_recommendations(self, student_id, module_code):
        return [
            {
                "resource_id": "r1",
                "title": f"Support de cours {module_code}",
                "url": "#",
                "type": "video",
                "reason": "Améliorer les bases du module"
            },
            {
                "resource_id": "r2",
                "title": "Quiz d'entraînement",
                "url": "#",
                "type": "quiz",
                "reason": "Valider les acquis récents"
            }
        ]