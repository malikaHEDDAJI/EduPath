"""
AIService: FastAPI microservice for AI functionalities.

This service exposes:
- Prediction endpoint: Predicts student success probability
- Recommendations endpoint: Generates personalized learning recommendations
- Health endpoint: Service health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import existing functions from libs
try:
    from libs.predictor import load_trained_model, predict_student
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not import from libs.predictor: {e}")
    print("Note: Please implement load_trained_model() and predict_student() in libs/predictor.py")
    # Define stub functions for development
    def load_trained_model():
        raise NotImplementedError("load_trained_model() must be implemented in libs/predictor.py")
    
    def predict_student(student_id, module_code, model, feature_columns):
        raise NotImplementedError("predict_student() must be implemented in libs/predictor.py")

try:
    from libs.recommender import generate_recommendations_for_student
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not import from libs.recommender: {e}")
    print("Note: Please implement generate_recommendations_for_student() in libs/recommender.py")
    # Define stub function for development
    def generate_recommendations_for_student(student_id, module_code):
        raise NotImplementedError("generate_recommendations_for_student() must be implemented in libs/recommender.py")

# Initialize FastAPI app
app = FastAPI(
    title="AIService",
    description="AI Service for Learning Analytics Platform - Prediction and Recommendations",
    version="1.0.0"
)

# Global variables to store loaded model and feature columns
model = None
feature_columns = None


@app.on_event("startup")
async def startup_event():
    """
    Load the trained model once at startup.
    This ensures the model is loaded only once, not on every request.
    """
    global model, feature_columns
    try:
        print("Loading trained model at startup...")
        model, feature_columns = load_trained_model()
        print(f"✓ Model loaded successfully. Feature columns: {len(feature_columns) if feature_columns else 0}")
    except Exception as e:
        print(f"⚠ Error loading model at startup: {e}")
        print("⚠ Service will start but prediction endpoints may not work until model is loaded.")
        model = None
        feature_columns = None


# ==================== Pydantic Models ====================

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    student_id: int
    module_code: str


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    student_id: int
    module_code: str
    success_proba: float
    risk_level: str
    message: str


class Recommendation(BaseModel):
    """Model for a single recommendation."""
    resource_id: str
    title: str
    url: str
    type: str
    reason: str


class RecommendationResponse(BaseModel):
    """Response model for recommendations endpoint."""
    student_id: int
    module_code: str
    recommendations: List[Recommendation]


# ==================== Endpoints ====================

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON response with status "ok"
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict student success probability for a given module.
    
    Args:
        request: PredictionRequest containing student_id and module_code
        
    Returns:
        PredictionResponse with success probability, risk level, and message
        
    Raises:
        HTTPException 404: If student or module is not found
        HTTPException 500: If model is not loaded or prediction fails
    """
    global model, feature_columns
    
    # Check if model is loaded
    if model is None or feature_columns is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please ensure the model is available and restart the service."
        )
    
    try:
        # Call the existing predict_student function
        result = predict_student(
            student_id=request.student_id,
            module_code=request.module_code,
            model=model,
            feature_columns=feature_columns
        )
        
        # Handle different return types from predict_student
        if isinstance(result, dict):
            success_proba = result.get("success_proba", result.get("probability", 0.0))
            risk_level = result.get("risk_level", _determine_risk_level(success_proba))
            message = result.get("message", _generate_message(success_proba, risk_level))
        elif isinstance(result, tuple):
            # If function returns (proba, risk_level, message)
            success_proba = result[0] if len(result) > 0 else 0.0
            risk_level = result[1] if len(result) > 1 else _determine_risk_level(success_proba)
            message = result[2] if len(result) > 2 else _generate_message(success_proba, risk_level)
        else:
            # If function returns just probability
            success_proba = float(result) if result is not None else 0.0
            risk_level = _determine_risk_level(success_proba)
            message = _generate_message(success_proba, risk_level)
        
        return PredictionResponse(
            student_id=request.student_id,
            module_code=request.module_code,
            success_proba=float(success_proba),
            risk_level=str(risk_level),
            message=str(message)
        )
        
    except ValueError as e:
        # Handle "not found" errors
        if "not found" in str(e).lower() or "introuvable" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/reco/{student_id}/{module_code}", response_model=RecommendationResponse)
async def get_recommendations(student_id: int, module_code: str):
    """
    Get personalized learning recommendations for a student and module.
    
    Args:
        student_id: ID of the student
        module_code: Code of the module
        
    Returns:
        RecommendationResponse with list of recommendations
        
    Raises:
        HTTPException 404: If student or module is not found
        HTTPException 500: If recommendation generation fails
    """
    try:
        # Call the existing generate_recommendations_for_student function
        recommendations = generate_recommendations_for_student(
            student_id=student_id,
            module_code=module_code
        )
        
        # Handle different return types
        if recommendations is None:
            recommendations = []
        elif not isinstance(recommendations, list):
            recommendations = [recommendations] if recommendations else []
        
        # Convert recommendations to the expected format
        recommendation_list = []
        for rec in recommendations:
            if isinstance(rec, dict):
                recommendation_list.append(Recommendation(
                    resource_id=str(rec.get("resource_id", rec.get("id", ""))),
                    title=str(rec.get("title", "")),
                    url=str(rec.get("url", rec.get("link", ""))),
                    type=str(rec.get("type", rec.get("resource_type", "unknown"))),
                    reason=str(rec.get("reason", rec.get("description", "")))
                ))
            else:
                # If it's a simple object, try to extract attributes
                recommendation_list.append(Recommendation(
                    resource_id=str(getattr(rec, "resource_id", getattr(rec, "id", ""))),
                    title=str(getattr(rec, "title", "")),
                    url=str(getattr(rec, "url", getattr(rec, "link", ""))),
                    type=str(getattr(rec, "type", getattr(rec, "resource_type", "unknown"))),
                    reason=str(getattr(rec, "reason", getattr(rec, "description", "")))
                ))
        
        return RecommendationResponse(
            student_id=student_id,
            module_code=module_code,
            recommendations=recommendation_list
        )
        
    except ValueError as e:
        # Handle "not found" errors
        if "not found" in str(e).lower() or "introuvable" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")


# ==================== Helper Functions ====================

def _determine_risk_level(success_proba: float) -> str:
    """
    Determine risk level based on success probability.
    
    Args:
        success_proba: Success probability (0.0 to 1.0)
        
    Returns:
        Risk level string: "low", "medium", or "high"
    """
    if success_proba >= 0.7:
        return "low"
    elif success_proba >= 0.5:
        return "medium"
    else:
        return "high"


def _generate_message(success_proba: float, risk_level: str) -> str:
    """
    Generate a human-readable message based on success probability and risk level.
    
    Args:
        success_proba: Success probability (0.0 to 1.0)
        risk_level: Risk level string
        
    Returns:
        Human-readable message
    """
    proba_percent = int(success_proba * 100)
    
    if risk_level == "low":
        return f"Student has a {proba_percent}% chance of success. Low risk - student is on track."
    elif risk_level == "medium":
        return f"Student has a {proba_percent}% chance of success. Medium risk - additional support may be beneficial."
    else:
        return f"Student has a {proba_percent}% chance of success. High risk - intervention recommended."


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

