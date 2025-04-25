"""
Nutrition domain handler for Gospel.

This module provides functions to integrate the nutrition analyzer with the 
core application flow, including analyzing variants and saving results.
"""

import json
import logging
import os
from typing import Dict, List, Any

from gospel.core.variant import Variant
from gospel.domains.nutrition import NutritionAnalyzer, NutritionReport

logger = logging.getLogger(__name__)


def convert_variants_for_nutrition_analysis(variants: List[Variant]) -> Dict[str, Dict[str, Any]]:
    """
    Convert Gospel Variant objects to the format expected by NutritionAnalyzer.
    
    Args:
        variants: List of Gospel Variant objects
        
    Returns:
        Dictionary of variant data formatted for NutritionAnalyzer
    """
    formatted_variants = {}
    
    for variant in variants:
        variant_id = variant.id
        gene = variant.functional_impact.get("gene_name", "")
        genotype = variant.genotype if hasattr(variant, "genotype") else ""
        
        # Skip variants without required information
        if not gene or not genotype:
            continue
        
        formatted_variants[variant_id] = {
            "gene": gene,
            "genotype": genotype,
            "chromosome": variant.chromosome,
            "position": variant.position,
            "reference": variant.reference,
            "alternate": variant.alternate,
            "type": variant.type,
        }
    
    return formatted_variants


def analyze_nutrition(variants: List[Variant], individual_id: str, 
                      confidence_threshold: float = 0.6) -> NutritionReport:
    """
    Analyze variants for nutritional implications.
    
    Args:
        variants: List of variants to analyze
        individual_id: ID of the individual being analyzed
        confidence_threshold: Confidence threshold for reporting findings
        
    Returns:
        Nutrition analysis report
    """
    logger.info(f"Analyzing {len(variants)} variants for nutrition domain")
    
    # Convert variants to the format expected by NutritionAnalyzer
    formatted_variants = convert_variants_for_nutrition_analysis(variants)
    logger.info(f"Converted {len(formatted_variants)} variants with sufficient information")
    
    # Initialize the analyzer
    analyzer = NutritionAnalyzer(formatted_variants, confidence_threshold=confidence_threshold)
    
    # Run the analysis
    logger.info("Running nutrition analysis")
    analyzer.analyze_variants()
    analyzer.aggregate_nutrient_needs()
    analyzer.aggregate_food_sensitivities()
    analyzer.aggregate_dietary_responses()
    
    # Generate the report
    logger.info("Generating nutrition report")
    report = analyzer.generate_report(individual_id=individual_id)
    
    return report


def save_nutrition_results(report: NutritionReport, output_dir: str) -> str:
    """
    Save nutrition analysis results to disk.
    
    Args:
        report: Nutrition analysis report
        output_dir: Directory to save results to
        
    Returns:
        Path to the saved results file
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert report to dictionary
    report_dict = report.to_dict()
    
    # Save to a nutrition-specific JSON file
    output_path = os.path.join(output_dir, "nutrition.json")
    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    
    logger.info(f"Saved nutrition results to {output_path}")
    
    return output_path


def get_nutrition_summary(report: NutritionReport) -> Dict[str, Any]:
    """
    Extract a summary of nutrition analysis results.
    
    Args:
        report: Nutrition analysis report
        
    Returns:
        Summary dictionary for inclusion in overall analysis summary
    """
    # Count of findings by category
    num_nutrient_needs = len(report.nutrient_needs)
    num_food_sensitivities = len(report.food_sensitivities)
    num_dietary_responses = len(report.dietary_responses)
    num_significant_variants = len(report.significant_variants)
    
    # Identify top nutrient needs
    top_nutrients = []
    if report.nutrition_plan and report.nutrition_plan.key_nutrient_needs:
        for need in report.nutrition_plan.key_nutrient_needs[:3]:  # Top 3
            direction = "increased" if need.adjustment_factor > 1.0 else "decreased" if need.adjustment_factor < 1.0 else "normal"
            top_nutrients.append({
                "name": need.name,
                "direction": direction,
                "adjustment_factor": need.adjustment_factor
            })
    
    # Identify key food sensitivities
    key_sensitivities = []
    for food_id, sensitivity in list(report.food_sensitivities.items())[:3]:  # Top 3
        key_sensitivities.append({
            "food": food_id,
            "severity": sensitivity.severity.value,
            "confidence": sensitivity.confidence
        })
    
    # Recommended diet
    primary_diet = "Balanced"
    diet_score = 0.0
    if report.nutrition_plan and report.nutrition_plan.primary_diet_pattern:
        primary_diet = report.nutrition_plan.primary_diet_pattern.name
        diet_score = report.nutrition_plan.primary_diet_pattern.score
    
    # Create summary
    return {
        "total_variants_analyzed": report.total_variants_analyzed,
        "significant_variants": num_significant_variants,
        "nutrient_needs": num_nutrient_needs,
        "food_sensitivities": num_food_sensitivities,
        "dietary_responses": num_dietary_responses,
        "top_nutrient_needs": top_nutrients,
        "key_sensitivities": key_sensitivities,
        "recommended_diet": primary_diet,
        "diet_score": diet_score
    } 