"""
Nutrition domain package for the Gospel genomic analysis framework.

This package provides models, analysis tools, and reference data for nutritional genomics,
enabling personalized nutrition insights based on genetic variation.
"""

from gospel.domains.nutrition.models import (
    DietaryPattern,
    DietaryResponse,
    EffectType,
    FoodSensitivity,
    NutrientEffect,
    NutrientNeed,
    NutritionPlan,
    NutritionReport,
    ResponseType,
    SensitivityLevel,
    VariantEffect,
)
from gospel.domains.nutrition.analyzer import NutritionAnalyzer
from gospel.domains.nutrition.constants import (
    NUTRIENTS,
    FOOD_SENSITIVITIES,
    GENE_NUTRIENT_RELATIONSHIPS,
    SNP_EFFECT_MAPPINGS,
)
from gospel.domains.nutrition.handler import (
    analyze_nutrition,
    save_nutrition_results,
    get_nutrition_summary,
    convert_variants_for_nutrition_analysis
)

__all__ = [
    # Models
    "DietaryPattern",
    "DietaryResponse",
    "EffectType",
    "FoodSensitivity",
    "NutrientEffect",
    "NutrientNeed",
    "NutritionPlan",
    "NutritionReport",
    "ResponseType",
    "SensitivityLevel",
    "VariantEffect",
    
    # Analysis
    "NutritionAnalyzer",
    
    # Constants
    "NUTRIENTS",
    "FOOD_SENSITIVITIES",
    "GENE_NUTRIENT_RELATIONSHIPS",
    "SNP_EFFECT_MAPPINGS",
    
    # Handler functions
    "analyze_nutrition",
    "save_nutrition_results",
    "get_nutrition_summary",
    "convert_variants_for_nutrition_analysis",
] 