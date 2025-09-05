"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines import (
    create_data_processing_pipeline,
    create_feature_engineering_pipeline,
    create_model_training_pipeline,
    create_reporting_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Pipelines individuales
    data_processing_pipeline = create_data_processing_pipeline()
    feature_engineering_pipeline = create_feature_engineering_pipeline()
    model_training_pipeline = create_model_training_pipeline()
    reporting_pipeline = create_reporting_pipeline()
    
    # Pipeline completo (todos los pipelines en secuencia)
    full_pipeline = (
        data_processing_pipeline + 
        feature_engineering_pipeline + 
        model_training_pipeline + 
        reporting_pipeline
    )
    
    return {
        "data_processing": data_processing_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "model_training": model_training_pipeline,
        "reporting": reporting_pipeline,
        "full": full_pipeline,
        "__default__": full_pipeline,
    }
