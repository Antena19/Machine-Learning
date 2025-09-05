"""Pipeline de ingeniería de características para extraer features relevantes."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    extract_audio_features,
    create_popularity_features,
    create_genre_features,
    create_temporal_features,
    combine_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crear el pipeline de ingeniería de características.

    Returns:
        Pipeline: Pipeline de Kedro para ingeniería de características.
    """
    return pipeline(
        [
            # Extracción de características de audio
            node(
                func=extract_audio_features,
                inputs="preprocessed_main_dataset",
                outputs="audio_features",
                name="extract_audio_features",
                tags=["feature_engineering", "audio"],
            ),
            # Creación de características de popularidad
            node(
                func=create_popularity_features,
                inputs="preprocessed_main_dataset",
                outputs="popularity_features",
                name="create_popularity_features",
                tags=["feature_engineering", "popularity"],
            ),
            # Creación de características de géneros
            node(
                func=create_genre_features,
                inputs="preprocessed_main_dataset",
                outputs="genre_features",
                name="create_genre_features",
                tags=["feature_engineering", "genres"],
            ),
            # Creación de características temporales
            node(
                func=create_temporal_features,
                inputs="preprocessed_main_dataset",
                outputs="temporal_features",
                name="create_temporal_features",
                tags=["feature_engineering", "temporal"],
            ),
            # Combinación de todas las características
            node(
                func=combine_features,
                inputs=["audio_features", "popularity_features", "genre_features", "temporal_features"],
                outputs="audio_features_engineered",
                name="combine_features",
                tags=["feature_engineering", "combination"],
            ),
        ],
        tags="feature_engineering",
    )
