"""
Gospel - Metacognitive Genomic Analysis Framework with Bayesian Optimization

A comprehensive bioinformatics framework for genomic analysis that:
- Uses metacognitive Bayesian networks for tool orchestration
- Implements fuzzy logic for uncertainty quantification  
- Provides visual understanding verification through circuit diagrams
- Integrates with external tools (Autobahn, Hegel, Borgia, etc.)
- Supports high-performance Rust acceleration for large datasets
- Enables per-experiment LLM specialization
"""

__version__ = "0.2.0"
__author__ = "Kundai Sachikonye"

# Main metacognitive analyzer
from gospel.core.gospel_analyzer import GospelAnalyzer

# Core genomic analysis modules (legacy compatibility)
from gospel.core import (
    VariantProcessor, 
    VariantAnnotator, 
    GenomicScorer,
    # New metacognitive components
    MetacognitiveBayesianNetwork,
    GenomicFuzzySystem,
    GenomicCircuitVisualizer,
    VisualUnderstandingVerifier,
    ToolOrchestrator
)

# LLM and AI integration
from gospel.llm import GospelLLM

# Turbulance DSL compiler
from gospel.turbulance import TurbulanceCompiler, compile_turbulance_script, validate_turbulance_script

# Export public API
__all__ = [
    # Main analyzer
    "GospelAnalyzer",
    
    # Legacy components (backward compatibility)
    "VariantProcessor",
    "VariantAnnotator", 
    "GenomicScorer",
    "GospelLLM",
    
    # New metacognitive framework
    "MetacognitiveBayesianNetwork",
    "GenomicFuzzySystem",
    "GenomicCircuitVisualizer", 
    "VisualUnderstandingVerifier",
    "ToolOrchestrator",
    
    # Turbulance DSL compiler
    "TurbulanceCompiler",
    "compile_turbulance_script",
    "validate_turbulance_script"
] 