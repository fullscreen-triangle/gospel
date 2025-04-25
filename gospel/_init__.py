"""
Gospel - Genomic analysis framework for sprint performance with LLM integration
"""

__version__ = "0.1.0"
__author__ = "Kundai Chasinda"

from gospel.core import GospelAnalyzer
from gospel.knowledge_base import KnowledgeBase
from gospel.llm import GospelLLM

# Make core classes available at the package level
__all__ = ["GospelAnalyzer", "KnowledgeBase", "GospelLLM"]
