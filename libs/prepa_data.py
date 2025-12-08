"""
PrepaData microservice: Feature engineering and metrics computation.

This module processes normalized data to compute student-module level metrics
for use in profiling, prediction, and recommendation systems.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from libs.utils import get_data_paths


def load_normalized_tables(processed_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all normalized CSV files from the processed data directory.
    
    Args:
        processed_dir: Path to processed data directory. If None, uses default from get_data_paths()
        
    Returns:
        Dictionary mapping table names to DataFrames:
        - student_info: Student demographic and enrollment info
        - courses: Course/module information
        - registrations: Student course registrations
        - assessments: Assessment definitions with weights
        - student_assessment: Student assessment scores
        - student_vle: Student VLE interaction clicks
        - vle_info: VLE resource information
    """
    if processed_dir is None:
        _, processed_dir = get_data_paths()
    
    processed_dir = Path(processed_dir)
    
    tables = {
        "student_info": "student_info_normalized.csv",
        "courses": "courses_normalized.csv",
        "registrations": "registrations_normalized.csv",
        "assessments": "assessments_normalized.csv",
        "student_assessment": "student_assessment_normalized.csv",
        "student_vle": "student_vle_normalized.csv",
        "vle_info": "vle_info_normalized.csv"
    }
    
    data = {}
    for name, filename in tables.items():
        filepath = processed_dir / filename
        if filepath.exists():
            data[name] = pd.read_csv(filepath)
            print(f"✓ Loaded {name}: {len(data[name])} rows")
        else:
            print(f"⚠ Warning: {filepath} not found")
            data[name] = pd.DataFrame()
    
    return data


def build_student_module_metrics(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute aggregated metrics per (student_id, code_module, code_presentation).
    
    Metrics computed:
    - avg_score: Weighted average of assessment scores
    - completion_rate: Ratio of completed assessments to total assessments
    - total_clicks: Total VLE clicks
    - active_days: Number of unique activity dates
    - final_result: Final result from student_info (Pass/Fail/Withdrawn/Distinction)
    
    Args:
        data: Dictionary of DataFrames from load_normalized_tables()
        
    Returns:
        DataFrame with columns:
        - student_id
        - code_module
        - code_presentation
        - avg_score
        - completion_rate
        - total_clicks
        - active_days
        - final_result
    """
    # Extract tables
    student_info = data.get("student_info", pd.DataFrame())
    registrations = data.get("registrations", pd.DataFrame())
    assessments = data.get("assessments", pd.DataFrame())
    student_assessment = data.get("student_assessment", pd.DataFrame())
    student_vle = data.get("student_vle", pd.DataFrame())
    
    # Start with registrations to get all (student, module, presentation) combinations
    if registrations.empty:
        print("⚠ Warning: No registrations data available")
        return pd.DataFrame()
    
    # Base: all student-module-presentation combinations from registrations
    metrics = registrations[["student_id", "code_module", "code_presentation"]].copy()
    metrics = metrics.drop_duplicates()
    
    # 1. Compute weighted average score from assessments
    if not student_assessment.empty and not assessments.empty:
        # Since student_assessment doesn't have module/presentation directly,
        # we'll compute average scores per student, then merge with module info from registrations
        student_scores = student_assessment.groupby("student_id").agg({
            "score": ["mean", "count"]
        }).reset_index()
        student_scores.columns = ["student_id", "avg_score_raw", "assessment_count"]
        
        # Merge with base metrics to get student-module-presentation combinations
        metrics = metrics.merge(student_scores, on="student_id", how="left")
        
        # Get total assessments per module-presentation from assessments table
        module_assessments = assessments.groupby(["code_module", "code_presentation"]).agg({
            "id_assessment": "count"
        }).reset_index()
        module_assessments.columns = ["code_module", "code_presentation", "total_assessments"]
        
        metrics = metrics.merge(module_assessments, on=["code_module", "code_presentation"], how="left")
        
        # Completion rate: assessments done / total assessments
        metrics["completion_rate"] = metrics["assessment_count"] / metrics["total_assessments"]
        metrics["completion_rate"] = metrics["completion_rate"].fillna(0.0)
        metrics["completion_rate"] = metrics["completion_rate"].clip(0.0, 1.0)
        
        # Use average score (weighted calculation would require id_assessment in student_assessment)
        metrics["avg_score"] = metrics["avg_score_raw"]
        
        # Drop intermediate columns
        metrics = metrics.drop(columns=["avg_score_raw", "assessment_count", "total_assessments"], errors="ignore")
    else:
        metrics["avg_score"] = np.nan
        metrics["completion_rate"] = 0.0
    
    # 2. Compute total clicks from VLE interactions
    if not student_vle.empty:
        # Check if column is id_student or student_id
        student_col = "id_student" if "id_student" in student_vle.columns else "student_id"
        vle_agg = student_vle.groupby([student_col, "code_module", "code_presentation"]).agg({
            "sum_click": "sum"
        }).reset_index()
        vle_agg.columns = ["student_id", "code_module", "code_presentation", "total_clicks"]
        
        metrics = metrics.merge(vle_agg, on=["student_id", "code_module", "code_presentation"], how="left")
        metrics["total_clicks"] = metrics["total_clicks"].fillna(0).astype(int)
    else:
        metrics["total_clicks"] = 0
    
    # 3. Compute active days (unique activity dates)
    if not student_vle.empty:
        # Check if column is id_student or student_id
        student_col = "id_student" if "id_student" in student_vle.columns else "student_id"
        active_days = student_vle.groupby([student_col, "code_module", "code_presentation"]).agg({
            "activity_date": "nunique"
        }).reset_index()
        active_days.columns = ["student_id", "code_module", "code_presentation", "active_days"]
        
        metrics = metrics.merge(active_days, on=["student_id", "code_module", "code_presentation"], how="left")
        metrics["active_days"] = metrics["active_days"].fillna(0).astype(int)
    else:
        metrics["active_days"] = 0
    
    # 4. Get final_result from student_info
    if not student_info.empty:
        student_results = student_info[["student_id", "code_module", "code_presentation", "final_result"]].copy()
        metrics = metrics.merge(student_results, on=["student_id", "code_module", "code_presentation"], how="left")
    else:
        metrics["final_result"] = None
    
    # Clean up intermediate columns
    cols_to_keep = [
        "student_id", "code_module", "code_presentation",
        "avg_score", "completion_rate", "total_clicks", "active_days", "final_result"
    ]
    metrics = metrics[cols_to_keep].copy()
    
    # Fill NaN values appropriately
    metrics["avg_score"] = metrics["avg_score"].fillna(0.0)
    
    return metrics


def save_student_module_metrics(metrics_df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """
    Save student-module metrics DataFrame to CSV.
    
    Args:
        metrics_df: DataFrame from build_student_module_metrics()
        output_path: Path to save the CSV. If None, uses default processed directory.
        
    Returns:
        Path to the saved file
    """
    if output_path is None:
        _, processed_dir = get_data_paths()
        output_path = processed_dir / "student_module_metrics.csv"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    metrics_df.to_csv(output_path, index=False)
    print(f"✓ Saved metrics to {output_path}")
    
    return output_path


def run_prepa_data_pipeline(processed_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Run the complete PrepaData pipeline:
    1. Load normalized tables
    2. Compute student-module metrics
    3. Save results to CSV
    
    Args:
        processed_dir: Path to processed data directory. If None, uses default.
        
    Returns:
        DataFrame with student-module metrics
    """
    print("=" * 60)
    print("PrepaData Pipeline: Starting...")
    print("=" * 60)
    
    # Step 1: Load normalized tables
    print("\n[Step 1] Loading normalized tables...")
    data = load_normalized_tables(processed_dir)
    
    # Step 2: Build metrics
    print("\n[Step 2] Computing student-module metrics...")
    metrics = build_student_module_metrics(data)
    
    if metrics.empty:
        print("⚠ Warning: No metrics computed. Returning empty DataFrame.")
        return metrics
    
    print(f"✓ Computed metrics for {len(metrics)} student-module combinations")
    
    # Step 3: Save results
    print("\n[Step 3] Saving metrics...")
    output_path = save_student_module_metrics(metrics, processed_dir)
    
    print("\n" + "=" * 60)
    print("PrepaData Pipeline: Completed successfully!")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"Shape: {metrics.shape}")
    print(f"\nFirst few rows:")
    print(metrics.head())
    
    return metrics

