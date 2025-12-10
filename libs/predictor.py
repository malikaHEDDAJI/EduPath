"""
PathPredictor microservice: Student success prediction using XGBoost.

This module provides functions to load a trained XGBoost model and make
predictions about student success probability for specific modules.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from libs.utils import get_data_paths
from libs.prepa_data import load_normalized_tables

# Global variables to cache loaded data
_cached_student_data = None
_cached_label_encoders = None


def _load_student_features() -> pd.DataFrame:
    """
    Load and prepare student features for prediction.
    This function caches the data to avoid reloading on every prediction.
    
    Returns:
        DataFrame with student features including student_id
    """
    global _cached_student_data
    
    if _cached_student_data is not None:
        return _cached_student_data
    
    # Load normalized tables
    _, processed_dir = get_data_paths()
    processed_dir = Path(processed_dir)
    
    student_info = pd.read_csv(processed_dir / "student_info_normalized.csv")
    student_vle = pd.read_csv(processed_dir / "student_vle_normalized.csv")
    student_assessment = pd.read_csv(processed_dir / "student_assessment_normalized.csv")
    
    # Rename id_student to student_id if needed
    if "id_student" in student_vle.columns:
        student_vle = student_vle.rename(columns={"id_student": "student_id"})
    
    # 1) VLE activity features
    vle_features = student_vle.groupby("student_id").agg({
        "sum_click": ["sum", "mean", "std"]
    })
    vle_features.columns = ["click_total", "click_mean", "click_std"]
    vle_features = vle_features.fillna(0).reset_index()
    
    # 2) Assessment score features
    score_features = student_assessment.groupby("student_id").agg({
        "score": ["mean", "std", "min", "max"]
    })
    score_features.columns = ["score_mean", "score_std", "score_min", "score_max"]
    score_features = score_features.fillna(0).reset_index()
    
    # 3) Merge with student_info
    df = student_info.merge(vle_features, on="student_id", how="left")
    df = df.merge(score_features, on="student_id", how="left")
    df = df.fillna(0)
    
    # Cache the result
    _cached_student_data = df
    
    return df


def load_trained_model(model_path: Optional[Path] = None) -> Tuple[xgb.XGBClassifier, list]:
    """
    Load the trained XGBoost model and return it with feature columns.
    
    Args:
        model_path: Path to the saved model JSON file. If None, uses default path.
        
    Returns:
        Tuple of (model, feature_columns) where:
        - model: Loaded XGBoost classifier
        - feature_columns: List of feature column names expected by the model
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        ValueError: If the model cannot be loaded
    """
    if model_path is None:
        model_path = Path("models/path_predictor.json")
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please train the model first using 04_PathPredictor.ipynb"
        )
    
    try:
        # Load XGBoost model
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        
        # Get feature columns from the model
        # XGBoost models store feature names if they were provided during training
        # If not, we need to infer them from the training data structure
        feature_columns = [
            "code_module", "code_presentation", "highest_education",
            "imd_band", "age_band", "num_of_prev_attempts", "studied_credits",
            "disability", "click_total", "click_mean", "click_std",
            "score_mean", "score_std", "score_min", "score_max"
        ]
        
        print(f"✓ Model loaded successfully from {model_path}")
        print(f"  Model type: XGBoost Classifier")
        print(f"  Expected features: {len(feature_columns)}")
        
        return model, feature_columns
        
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {str(e)}")


def predict_student(
    student_id: int,
    module_code: str,
    model: xgb.XGBClassifier,
    feature_columns: list
) -> Dict[str, Any]:
    """
    Predict success probability for a student in a specific module.
    
    Args:
        student_id: ID of the student
        module_code: Code of the module (e.g., "AAA", "BBB")
        model: Trained XGBoost classifier
        feature_columns: List of feature column names expected by the model
        
    Returns:
        Dictionary with keys:
        - success_proba: Success probability (0.0 to 1.0)
        - risk_level: "low", "medium", or "high"
        - message: Human-readable message
        
    Raises:
        ValueError: If student is not found or prediction fails
    """
    # Load student features
    df = _load_student_features()
    
    # Check if student exists
    if student_id not in df["student_id"].values:
        raise ValueError(f"Student {student_id} not found")
    
    # Filter student data
    student_row = df[df["student_id"] == student_id].iloc[0].copy()
    
    # Prepare features for prediction
    x = pd.DataFrame([student_row])
    
    # Set the module code
    x["code_module"] = module_code
    
    # Drop columns not used in prediction
    drop_cols = ["final_result", "gender", "region", "student_id", "target"]
    x = x.drop(columns=[c for c in drop_cols if c in x.columns])
    
    # Handle categorical columns with LabelEncoder
    categorical_cols = [
        "code_module", "code_presentation", "highest_education",
        "imd_band", "age_band", "disability"
    ]
    
    for col in categorical_cols:
        if col in x.columns:
            x[col] = x[col].fillna("Unknown")
            # Create a LabelEncoder and fit_transform
            # Note: In production, you'd want to save/load encoders from training
            le = LabelEncoder()
            x[col] = le.fit_transform(x[col].astype(str))
    
    # Handle numeric columns
    numeric_cols = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    x[numeric_cols] = x[numeric_cols].fillna(0)
    
    # Ensure columns match expected feature order
    # Reorder columns to match feature_columns
    available_cols = [col for col in feature_columns if col in x.columns]
    x = x[available_cols]
    
    # Make prediction
    try:
        proba = model.predict_proba(x)[0][1]  # Probability of success (class 1)
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")
    
    # Determine risk level
    if proba >= 0.7:
        risk_level = "low"
    elif proba >= 0.5:
        risk_level = "medium"
    else:
        risk_level = "high"
    
    # Generate message
    proba_percent = int(proba * 100)
    if risk_level == "low":
        message = f"Student has a {proba_percent}% chance of success. Low risk - student is on track."
    elif risk_level == "medium":
        message = f"Student has a {proba_percent}% chance of success. Medium risk - additional support may be beneficial."
    else:
        message = f"Student has a {proba_percent}% chance of success. High risk - intervention recommended."
    
    return {
        "success_proba": float(proba),
        "risk_level": risk_level,
        "message": message
    }

