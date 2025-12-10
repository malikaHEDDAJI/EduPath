"""
RecoBuilder microservice: Personalized learning resource recommendations.

This module provides functions to generate personalized learning recommendations
for students based on their performance and engagement patterns using semantic
similarity (BERT embeddings) and FAISS for efficient similarity search.
"""

import os
# Disable TensorFlow to avoid DLL loading issues on Windows
# sentence-transformers will use PyTorch instead
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from libs.utils import get_data_paths
from libs.profiler import load_student_module_metrics

# Global variables to cache loaded models and data
_cached_embedding_model = None
_cached_resources_df = None
_cached_resource_embeddings = None
_cached_faiss_index = None
_cached_student_metrics = None


def _load_embedding_model():
    """
    Load the sentence transformer model for generating embeddings.
    Caches the model to avoid reloading.
    
    Returns:
        SentenceTransformer model
    """
    global _cached_embedding_model
    
    if _cached_embedding_model is not None:
        return _cached_embedding_model
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        _cached_embedding_model = model
        return model
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for recommendations. "
            "Install it with: pip install sentence-transformers"
        )


def _load_resources() -> pd.DataFrame:
    """
    Load learning resources. If resources file doesn't exist, creates default resources.
    
    Returns:
        DataFrame with columns: resource_id, title, url, type, topic, difficulty
    """
    global _cached_resources_df
    
    if _cached_resources_df is not None:
        return _cached_resources_df
    
    _, processed_dir = get_data_paths()
    processed_dir = Path(processed_dir)
    resources_file = processed_dir / "learning_resources.csv"
    
    if resources_file.exists():
        resources_df = pd.read_csv(resources_file)
    else:
        # Create default resources based on available modules
        courses_file = processed_dir / "courses_normalized.csv"
        if courses_file.exists():
            courses = pd.read_csv(courses_file)
            modules = courses["code_module"].unique() if "code_module" in courses.columns else ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
        else:
            modules = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
        
        # Generate default resources for each module
        resources = []
        resource_id = 1
        
        for module in modules:
            resources.extend([
                {
                    "resource_id": f"R{resource_id:04d}",
                    "title": f"Introduction to {module} Concepts",
                    "url": f"https://lms.example.com/{module}/intro",
                    "type": "video",
                    "topic": f"{module} Basics",
                    "difficulty": "medium"
                },
                {
                    "resource_id": f"R{resource_id+1:04d}",
                    "title": f"Quiz on {module} Key Concepts",
                    "url": f"https://lms.example.com/{module}/quiz",
                    "type": "exercise",
                    "topic": f"{module} Basics",
                    "difficulty": "high"
                },
                {
                    "resource_id": f"R{resource_id+2:04d}",
                    "title": f"{module} Study Notes",
                    "url": f"https://lms.example.com/{module}/notes",
                    "type": "pdf",
                    "topic": f"{module} Notes",
                    "difficulty": "medium"
                },
                {
                    "resource_id": f"R{resource_id+3:04d}",
                    "title": f"Advanced {module} Topic Deep Dive",
                    "url": f"https://lms.example.com/{module}/advanced",
                    "type": "tutoring",
                    "topic": f"{module} Advanced",
                    "difficulty": "high"
                },
                {
                    "resource_id": f"R{resource_id+4:04d}",
                    "title": f"Video: {module} Overview",
                    "url": f"https://lms.example.com/{module}/overview",
                    "type": "video",
                    "topic": f"{module} Overview",
                    "difficulty": "low"
                }
            ])
            resource_id += 5
        
        resources_df = pd.DataFrame(resources)
    
    _cached_resources_df = resources_df
    return resources_df


def _build_faiss_index() -> tuple:
    """
    Build FAISS index for resource embeddings.
    Caches the index and embeddings to avoid rebuilding.
    
    Returns:
        Tuple of (faiss_index, resource_embeddings)
    """
    global _cached_faiss_index, _cached_resource_embeddings
    
    if _cached_faiss_index is not None and _cached_resource_embeddings is not None:
        return _cached_faiss_index, _cached_resource_embeddings
    
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss-cpu is required for recommendations. "
            "Install it with: pip install faiss-cpu"
        )
    
    # Load resources and embedding model
    resources_df = _load_resources()
    model = _load_embedding_model()
    
    # Generate embeddings for resources
    resource_texts = (resources_df["title"] + " " + resources_df.get("topic", "")).tolist()
    resource_embeddings = model.encode(resource_texts, convert_to_numpy=True)
    
    # Build FAISS index
    embedding_dim = resource_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(resource_embeddings.astype('float32'))
    
    _cached_faiss_index = index
    _cached_resource_embeddings = resource_embeddings
    
    return index, resource_embeddings


def _get_student_embedding(student_id: int, module_code: str) -> Optional[np.ndarray]:
    """
    Get embedding vector for a student based on their weak topics.
    
    Args:
        student_id: ID of the student
        module_code: Code of the module
        
    Returns:
        Embedding vector or None if student not found
    """
    global _cached_student_metrics
    
    # Load student metrics
    if _cached_student_metrics is None:
        _, processed_dir = get_data_paths()
        metrics_file = processed_dir / "student_module_metrics.csv"
        
        if not metrics_file.exists():
            raise FileNotFoundError(
                f"Student metrics file not found: {metrics_file}\n"
                "Please run the PrepaData pipeline first."
            )
        
        _cached_student_metrics = pd.read_csv(metrics_file)
    
    # Find student's metrics for the module
    student_row = _cached_student_metrics[
        (_cached_student_metrics["student_id"] == student_id) &
        (_cached_student_metrics["code_module"] == module_code)
    ]
    
    if student_row.empty:
        # Try to find student in any module
        student_row = _cached_student_metrics[
            _cached_student_metrics["student_id"] == student_id
        ]
        
        if student_row.empty:
            return None
        
        # Use the first available module's data
        student_row = student_row.iloc[0]
    else:
        student_row = student_row.iloc[0]
    
    # Determine weak topics based on performance
    weak_topics = []
    if pd.notna(student_row.get("avg_score")) and student_row.get("avg_score", 100) < 50:
        weak_topics.append(module_code)
    if pd.notna(student_row.get("completion_rate")) and student_row.get("completion_rate", 1.0) < 0.5:
        weak_topics.append(module_code)
    
    weak_topics_str = " ".join(weak_topics) if weak_topics else module_code
    
    # Generate embedding
    model = _load_embedding_model()
    embedding = model.encode([weak_topics_str], convert_to_numpy=True)[0]
    
    return embedding


def generate_recommendations_for_student(
    student_id: int,
    module_code: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate personalized learning recommendations for a student and module.
    
    Args:
        student_id: ID of the student
        module_code: Code of the module (e.g., "AAA", "BBB")
        top_k: Number of recommendations to return (default: 5)
        
    Returns:
        List of recommendation dictionaries, each with:
        - resource_id: Unique identifier for the resource
        - title: Title of the resource
        - url: URL to access the resource
        - type: Type of resource (video, exercise, pdf, tutoring, etc.)
        - reason: Explanation of why this resource is recommended
        
    Raises:
        ValueError: If student is not found
        FileNotFoundError: If required data files are missing
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss-cpu is required for recommendations. "
            "Install it with: pip install faiss-cpu"
        )
    
    # Get student embedding
    student_embedding = _get_student_embedding(student_id, module_code)
    
    if student_embedding is None:
        raise ValueError(f"Student {student_id} not found")
    
    # Build or get FAISS index
    index, resource_embeddings = _build_faiss_index()
    resources_df = _load_resources()
    
    # Search for similar resources
    student_vector = np.array([student_embedding], dtype='float32')
    distances, indices = index.search(student_vector, min(top_k, len(resources_df)))
    
    # Build recommendations list
    recommendations = []
    for rank, idx in enumerate(indices[0]):
        if idx >= len(resources_df):
            continue
        
        res = resources_df.iloc[idx]
        
        # Determine reason based on distance and resource type
        distance = float(distances[0][rank])
        if distance < 0.5:
            reason = f"Highly relevant to your learning needs in {module_code}"
        elif distance < 1.0:
            reason = f"Relevant resource for {module_code} based on your performance"
        else:
            reason = f"Recommended resource for {module_code} to strengthen your understanding"
        
        recommendations.append({
            "resource_id": str(res.get("resource_id", f"R{idx:04d}")),
            "title": str(res.get("title", "Untitled Resource")),
            "url": str(res.get("url", f"https://lms.example.com/{module_code}/resource/{idx}")),
            "type": str(res.get("type", "unknown")),
            "reason": reason
        })
    
    return recommendations

