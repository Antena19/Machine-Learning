"""Pipeline de reportes y visualizaciones."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_genre_distribution_plot,
    create_audio_features_plot,
    create_popularity_trends_plot,
    create_confusion_matrix_plot,
    generate_model_performance_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crear el pipeline de reportes y visualizaciones.

    Returns:
        Pipeline: Pipeline de Kedro para reportes.
    """
    return pipeline(
        [
            # Visualizaciones de datos
            node(
                func=create_genre_distribution_plot,
                inputs="preprocessed_main_dataset",
                outputs="genre_distribution_plot",
                name="create_genre_distribution_plot",
                tags=["visualization", "genres"],
            ),
            node(
                func=create_audio_features_plot,
                inputs="audio_features_engineered",
                outputs="audio_features_plot",
                name="create_audio_features_plot",
                tags=["visualization", "audio_features"],
            ),
            node(
                func=create_popularity_trends_plot,
                inputs="preprocessed_main_dataset",
                outputs="popularity_trends_plot",
                name="create_popularity_trends_plot",
                tags=["visualization", "trends"],
            ),
            # Visualizaciones de modelos
            node(
                func=create_confusion_matrix_plot,
                inputs=["test_data", "genre_classifier"],
                outputs="confusion_matrix_plot",
                name="create_confusion_matrix_plot",
                tags=["visualization", "model_evaluation"],
            ),
            # Reporte de rendimiento
            node(
                func=generate_model_performance_report,
                inputs="model_metrics",
                outputs="model_performance_report",
                name="generate_model_performance_report",
                tags=["reporting", "performance"],
            ),
        ],
        tags="reporting",
    )
