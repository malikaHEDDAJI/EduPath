from pydantic import BaseModel, Field
from typing import List, Optional
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