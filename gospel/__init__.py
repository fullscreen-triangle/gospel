"""
Gospel - Genomic analysis framework with real data integration and HuggingFace models

A comprehensive bioinformatics framework for genomic analysis that:
- Uses real data from public genomic databases (Ensembl, ClinVar, GWAS Catalog)
- Integrates HuggingFace transformer models for sequence analysis
- Provides cross-domain analysis (genomics + pharmacogenomics + systems biology)
- Supports variant annotation, gene network analysis, and clinical interpretation
"""

__version__ = "0.1.0"
__author__ = "Kundai Sachikonye"

# Core genomic analysis modules
from gospel.core import (
    VariantProcessor, 
    VariantAnnotator, 
    GenomicScorer,
    ExpressionAnalyzer,
    NetworkAnalyzer
)

# LLM and AI integration
from gospel.llm import GospelLLM

# Export public API
__all__ = [
    "VariantProcessor",
    "VariantAnnotator", 
    "GenomicScorer",
    "ExpressionAnalyzer",
    "NetworkAnalyzer",
    "GospelLLM"
] 