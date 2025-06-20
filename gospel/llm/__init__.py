"""
LLM module for Gospel.
"""

from gospel.llm.model import GospelLLM, HuggingFaceGenomicModel, GENOMIC_MODELS
from gospel.llm.trainer import ModelTrainer
from gospel.llm.distiller import KnowledgeDistiller
from gospel.llm.genomic_models import GenomicModelManager, create_analysis_config
from gospel.llm.experiment_manager import ExperimentLLMManager, ExperimentContext, LoRAConfiguration, create_experiment_llm

__all__ = [
    "GospelLLM", 
    "ModelTrainer", 
    "KnowledgeDistiller",
    "HuggingFaceGenomicModel",
    "GenomicModelManager",
    "create_analysis_config",
    "GENOMIC_MODELS",
    "ExperimentLLMManager",
    "ExperimentContext",
    "LoRAConfiguration",
    "create_experiment_llm"
] 