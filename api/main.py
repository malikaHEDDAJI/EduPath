from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
# --- MODELES (Copiez tout ça) ---
class PredictionRequest(BaseModel):
    student_id: int
    module_code: str
class PredictionResponse(BaseModel):
    student_id: int
    module_code: str
    success_proba: float
    risk_level: str
    message: str
class Recommendation(BaseModel):
    resource_id: str
    title: str
    url: str
    type: str
    reason: str
class RecommendationResponse(BaseModel):
    student_id: int
    module_code: str
    recommendations: List[Recommendation]
# --- APP SETUP ---
app = FastAPI(title="EduPath AI Service")
# ⚠️ CRUCIAL : CORS DOIT ETRE ICI ⚠️
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise TOUT (Frontend locale)
    allow_credentials=True,
    allow_methods=["*"],  # Autorise POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)
# --- ROUTES ---
@app.get("/health")
async def health():
    return {"status": "ok"}
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    print(f"🔮 PREDICT appelé pour: {request.student_id}")
    # Simule une réponse toujours valide (évite les 404/500)
    return {
        "student_id": request.student_id,
        "module_code": request.module_code,
        "success_proba": 0.88,
        "risk_level": "Low",
        "message": "Succès prédit (Mode Réparation)"
    }
@app.get("/reco/{student_id}/{module_code}", response_model=RecommendationResponse)
async def get_recommendations(student_id: int, module_code: str):
    print(f"📚 RECO appelé pour: {student_id}")
    return {
        "student_id": student_id,
        "module_code": module_code,
        "recommendations": [
            {
                "resource_id": "repair-1",
                "title": "Vidéo de Réparation",
                "url": "#",
                "type": "video",
                "reason": "Test de connexion réussi"
            }
        ]
    }
if __name__ == "__main__":
    import uvicorn
    # Lance sur le port 8001 comme configuré dans le frontend
    uvicorn.run(app, host="127.0.0.1", port=8001)