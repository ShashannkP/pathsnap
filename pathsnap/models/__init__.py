"""
pathsnap.models
===============
Model architectures, HistGradientBoosting pipeline, and 5-model benchmarking.
"""

from .hist_gradient_boosting import HistGradientBoostingPipeline
from .benchmark import get_benchmark_models, evaluate_model_cv

__all__ = [
    "HistGradientBoostingPipeline",
    "get_benchmark_models",
    "evaluate_model_cv",
]
