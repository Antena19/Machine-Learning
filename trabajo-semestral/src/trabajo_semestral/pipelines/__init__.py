"""Pipelines del proyecto de recomendación musical."""

from .data_processing.pipeline import create_pipeline as create_data_processing_pipeline
from .feature_engineering.pipeline import create_pipeline as create_feature_engineering_pipeline
from .model_training.pipeline import create_pipeline as create_model_training_pipeline
from .reporting.pipeline import create_pipeline as create_reporting_pipeline

__all__ = [
    "create_data_processing_pipeline",
    "create_feature_engineering_pipeline", 
    "create_model_training_pipeline",
    "create_reporting_pipeline",
]
