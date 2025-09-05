"""Pipeline de entrenamiento de modelos de machine learning."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    split_data,
    train_recommendation_model,
    train_genre_classifier,
    train_popularity_predictor,
    evaluate_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crear el pipeline de entrenamiento de modelos.

    Returns:
        Pipeline: Pipeline de Kedro para entrenamiento de modelos.
    """
    return pipeline(
        [
            # División de datos
            node(
                func=split_data,
                inputs="audio_features_engineered",
                outputs=["train_data", "validation_data", "test_data"],
                name="split_data",
                tags=["data_splitting", "preprocessing"],
            ),
            # Entrenamiento de modelo de recomendación
            node(
                func=train_recommendation_model,
                inputs=["train_data", "validation_data"],
                outputs="recommendation_model",
                name="train_recommendation_model",
                tags=["model_training", "recommendation"],
            ),
            # Entrenamiento de clasificador de géneros
            node(
                func=train_genre_classifier,
                inputs=["train_data", "validation_data"],
                outputs="genre_classifier",
                name="train_genre_classifier",
                tags=["model_training", "classification"],
            ),
            # Entrenamiento de predictor de popularidad
            node(
                func=train_popularity_predictor,
                inputs=["train_data", "validation_data"],
                outputs="popularity_predictor",
                name="train_popularity_predictor",
                tags=["model_training", "regression"],
            ),
            # Evaluación de modelos
            node(
                func=evaluate_models,
                inputs=["test_data", "recommendation_model", "genre_classifier", "popularity_predictor"],
                outputs="model_metrics",
                name="evaluate_models",
                tags=["model_evaluation", "metrics"],
            ),
        ],
        tags="model_training",
    )
