"""
Domain-specific modules for Gospel.

This package contains domain-specific implementations for genomic analysis
across fitness, pharmacogenetics, and nutrition domains.
"""

from .fitness import FitnessDomain
from .pharmacogenetics import PharmacogeneticsDomain
from .nutrition import (
    NutritionAnalyzer, 
    NutritionReport,
    analyze_nutrition,
    save_nutrition_results,
    get_nutrition_summary
)

__all__ = [
    'FitnessDomain', 
    'PharmacogeneticsDomain', 
    'NutritionAnalyzer',
    'NutritionReport',
    'analyze_nutrition',
    'save_nutrition_results',
    'get_nutrition_summary'
] 