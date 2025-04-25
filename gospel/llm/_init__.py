"""
LLM module for Gospel.
"""

from gospel.llm.model import GospelLLM
from gospel.llm.trainer import ModelTrainer
from gospel.llm.distiller import KnowledgeDistiller

__all__ = ["GospelLLM", "ModelTrainer", "KnowledgeDistiller"]
