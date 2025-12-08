"""
StudentProfiler microservice: Student profiling based on learning metrics.

This module creates student profiles using rule-based logic and optionally
clustering techniques to identify risk levels and engagement patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from libs.utils import get_data_paths

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Clustering will be disabled.")


def load_student_module_metrics(metrics_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load student-module metrics from CSV file.
    
    Args:
        metrics_path: Path to student_module_metrics.csv. If None, uses default.
        
    Returns:
        DataFrame with student-module metrics
        
    Raises:
        FileNotFoundError: If the metrics file doesn't exist
        ValueError: If required columns are missing
    """
    if metrics_path is None:
        _, processed_dir = get_data_paths()
        metrics_path = processed_dir / "student_module_metrics.csv"
    
    metrics_path = Path(metrics_path)
    
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}\n"
            "Please run the PrepaData pipeline first (02_PrepaData.ipynb)."
        )
    
    df = pd.read_csv(metrics_path)
    
    # Validate required columns
    required_cols = [
        "student_id", "code_module", "code_presentation",
        "avg_score", "completion_rate", "total_clicks", "active_days", "final_result"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns in metrics file: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    print(f"✓ Loaded {len(df)} student-module records from {metrics_path}")
    
    return df


def compute_rule_based_profiles(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rule-based student profiles based on metrics.
    
    Profiles computed:
    - risk_level: "HIGH", "MEDIUM", or "LOW" based on avg_score, completion_rate, and final_result
    - engagement_profile: "LOW_ENGAGEMENT", "REGULAR", or "HIGH_ENGAGEMENT" based on clicks and active_days
    - global_profile: Combination of risk_level and engagement_profile
    
    Args:
        metrics_df: DataFrame with student-module metrics
        
    Returns:
        DataFrame with original columns + profile columns
    """
    df = metrics_df.copy()
    
    # 1. Compute risk_level
    def assign_risk_level(row):
        """Assign risk level based on multiple factors."""
        # High risk: low score OR low completion OR withdrawn/fail
        if pd.isna(row['avg_score']) or row['avg_score'] < 50:
            return "HIGH"
        if row['completion_rate'] < 0.5:
            return "HIGH"
        if row['final_result'] in ['Withdrawn', 'Fail']:
            return "HIGH"
        
        # Medium risk: moderate score or completion
        if row['avg_score'] < 70 or row['completion_rate'] < 0.75:
            return "MEDIUM"
        
        # Low risk: good score and completion
        return "LOW"
    
    df['risk_level'] = df.apply(assign_risk_level, axis=1)
    
    # 2. Compute engagement_profile based on clicks and active_days
    # Use quantiles to define thresholds, but handle edge cases where quantiles are 0
    clicks_q33 = df['total_clicks'].quantile(0.33)
    clicks_q66 = df['total_clicks'].quantile(0.66)
    days_q33 = df['active_days'].quantile(0.33)
    days_q66 = df['active_days'].quantile(0.66)
    
    # If quantiles are all 0, use median and mean as fallback thresholds
    if clicks_q66 == 0:
        clicks_q66 = df['total_clicks'].median() if df['total_clicks'].median() > 0 else df['total_clicks'].mean()
    if days_q66 == 0:
        days_q66 = df['active_days'].median() if df['active_days'].median() > 0 else df['active_days'].mean()
    if clicks_q33 == 0:
        clicks_q33 = clicks_q66 * 0.33
    if days_q33 == 0:
        days_q33 = days_q66 * 0.33
    
    def assign_engagement(row):
        """Assign engagement profile based on activity levels."""
        clicks = row['total_clicks']
        days = row['active_days']
        
        # High engagement: high clicks AND high active days
        if clicks >= clicks_q66 and days >= days_q66:
            return "HIGH_ENGAGEMENT"
        
        # Low engagement: low clicks OR low active days
        if clicks < clicks_q33 or days < days_q33:
            return "LOW_ENGAGEMENT"
        
        # Regular engagement: everything else
        return "REGULAR"
    
    df['engagement_profile'] = df.apply(assign_engagement, axis=1)
    
    # 3. Compute global_profile combining risk and engagement
    def assign_global_profile(row):
        """Create a global profile label combining risk and engagement."""
        risk = row['risk_level']
        engagement = row['engagement_profile']
        
        # Priority: risk level first, then engagement
        if risk == "HIGH":
            if engagement == "HIGH_ENGAGEMENT":
                return "HIGH_RISK_HIGH_ENGAGEMENT"
            elif engagement == "LOW_ENGAGEMENT":
                return "HIGH_RISK_LOW_ENGAGEMENT"
            else:
                return "HIGH_RISK_REGULAR"
        elif risk == "MEDIUM":
            if engagement == "HIGH_ENGAGEMENT":
                return "MEDIUM_RISK_HIGH_ENGAGEMENT"
            elif engagement == "LOW_ENGAGEMENT":
                return "MEDIUM_RISK_LOW_ENGAGEMENT"
            else:
                return "MEDIUM_RISK_REGULAR"
        else:  # LOW risk
            if engagement == "HIGH_ENGAGEMENT":
                return "LOW_RISK_HIGH_ENGAGEMENT"
            elif engagement == "LOW_ENGAGEMENT":
                return "LOW_RISK_LOW_ENGAGEMENT"
            else:
                return "LOW_RISK_REGULAR"
    
    df['global_profile'] = df.apply(assign_global_profile, axis=1)
    
    print(f"✓ Computed rule-based profiles for {len(df)} records")
    print(f"  Risk levels: {df['risk_level'].value_counts().to_dict()}")
    print(f"  Engagement profiles: {df['engagement_profile'].value_counts().to_dict()}")
    
    return df


def compute_cluster_profiles(metrics_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Compute cluster-based profiles using KMeans clustering.
    
    This is an OPTIONAL function that uses unsupervised learning to discover
    latent student profiles based on numeric features.
    
    Args:
        metrics_df: DataFrame with student-module metrics (must include profile columns)
        n_clusters: Number of clusters to create (default: 3)
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with added cluster_label column
        
    Raises:
        ImportError: If scikit-learn is not available
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for clustering. "
            "Install it with: pip install scikit-learn"
        )
    
    df = metrics_df.copy()
    
    # Select features for clustering
    feature_cols = ['avg_score', 'completion_rate', 'total_clicks', 'active_days']
    
    # Prepare data: handle NaN values
    X = df[feature_cols].copy()
    X = X.fillna(X.median())  # Fill NaN with median
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['cluster_label'] = kmeans.fit_predict(X_scaled)
    
    # Map cluster IDs to human-readable labels
    # Analyze cluster centers to assign meaningful names
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Simple mapping based on average score in cluster
    cluster_avg_scores = {}
    for i in range(n_clusters):
        cluster_mask = df['cluster_label'] == i
        cluster_avg_scores[i] = df.loc[cluster_mask, 'avg_score'].mean()
    
    # Sort clusters by average score and assign labels
    sorted_clusters = sorted(cluster_avg_scores.items(), key=lambda x: x[1], reverse=True)
    
    cluster_label_map = {}
    if n_clusters == 3:
        cluster_label_map[sorted_clusters[0][0]] = "TOP_PERFORMERS"
        cluster_label_map[sorted_clusters[1][0]] = "AVERAGE_PERFORMERS"
        cluster_label_map[sorted_clusters[2][0]] = "STRUGGLING_STUDENTS"
    else:
        # Generic labels for other cluster counts
        for idx, (cluster_id, _) in enumerate(sorted_clusters):
            cluster_label_map[cluster_id] = f"CLUSTER_{idx+1}"
    
    df['cluster_profile'] = df['cluster_label'].map(cluster_label_map)
    
    print(f"✓ Computed {n_clusters} clusters using KMeans")
    print(f"  Cluster distribution: {df['cluster_label'].value_counts().sort_index().to_dict()}")
    print(f"  Cluster profiles: {df['cluster_profile'].value_counts().to_dict()}")
    
    return df


def save_student_profiles(profiles_df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """
    Save student profiles DataFrame to CSV.
    
    Args:
        profiles_df: DataFrame with student profiles
        output_path: Path to save the CSV. If None, uses default processed directory.
        
    Returns:
        Path to the saved file
    """
    if output_path is None:
        _, processed_dir = get_data_paths()
        output_path = processed_dir / "student_module_profiles.csv"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    profiles_df.to_csv(output_path, index=False)
    print(f"✓ Saved profiles to {output_path}")
    
    return output_path


def run_student_profiler_pipeline(
    metrics_path: Optional[Path] = None,
    use_clustering: bool = False,
    n_clusters: int = 3
) -> pd.DataFrame:
    """
    Run the complete StudentProfiler pipeline.
    
    Steps:
    1. Load student-module metrics
    2. Compute rule-based profiles (risk_level, engagement_profile, global_profile)
    3. Optionally compute cluster-based profiles
    4. Save results to CSV
    
    Args:
        metrics_path: Path to student_module_metrics.csv. If None, uses default.
        use_clustering: Whether to apply KMeans clustering (requires scikit-learn)
        n_clusters: Number of clusters if clustering is enabled
        
    Returns:
        DataFrame with student profiles
    """
    print("=" * 60)
    print("StudentProfiler Pipeline: Starting...")
    print("=" * 60)
    
    # Step 1: Load metrics
    print("\n[Step 1] Loading student-module metrics...")
    metrics_df = load_student_module_metrics(metrics_path)
    
    # Step 2: Compute rule-based profiles
    print("\n[Step 2] Computing rule-based profiles...")
    profiles_df = compute_rule_based_profiles(metrics_df)
    
    # Step 3: Optionally compute cluster profiles
    if use_clustering:
        if not SKLEARN_AVAILABLE:
            print("\n⚠ Warning: scikit-learn not available. Skipping clustering.")
        else:
            print("\n[Step 3] Computing cluster-based profiles...")
            profiles_df = compute_cluster_profiles(profiles_df, n_clusters=n_clusters)
    else:
        print("\n[Step 3] Skipping clustering (use_clustering=False)")
    
    # Step 4: Save results
    print("\n[Step 4] Saving profiles...")
    output_path = save_student_profiles(profiles_df)
    
    print("\n" + "=" * 60)
    print("StudentProfiler Pipeline: Completed successfully!")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"Shape: {profiles_df.shape}")
    print(f"\nProfile columns added:")
    profile_cols = [col for col in profiles_df.columns if col not in metrics_df.columns]
    print(f"  {profile_cols}")
    print(f"\nFirst few rows:")
    print(profiles_df.head())
    
    return profiles_df

