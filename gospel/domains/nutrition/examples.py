"""
Example usage of the nutrition analysis framework.

This module demonstrates how to use the nutrition analysis functionality
to analyze genetic variants and generate personalized nutrition reports.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from gospel.domains.nutrition import (
    NutritionAnalyzer,
    NutritionReport
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_example_variants() -> Dict[str, Dict[str, Any]]:
    """
    Load example variant data for demonstration purposes.
    
    Returns:
        Dictionary of variant data
    """
    # This is example data - in a real application, this would come from 
    # a variant calling pipeline or external genetic data source
    example_variants = {
        "rs1801133": {
            "gene": "MTHFR",
            "genotype": "CT",
            "chromosome": "1",
            "position": 11856378,
            "reference": "G",
            "alternate": "A"
        },
        "rs1801394": {
            "gene": "MTRR",
            "genotype": "AG",
            "chromosome": "5",
            "position": 7870973,
            "reference": "A",
            "alternate": "G"
        },
        "rs1800795": {
            "gene": "IL6",
            "genotype": "CG",
            "chromosome": "7",
            "position": 22766645,
            "reference": "G",
            "alternate": "C"
        },
        "rs4988235": {
            "gene": "MCM6",
            "genotype": "CT",
            "chromosome": "2",
            "position": 136608646,
            "reference": "G",
            "alternate": "A"
        },
        "rs762551": {
            "gene": "CYP1A2",
            "genotype": "AC",
            "chromosome": "15",
            "position": 75041917,
            "reference": "A",
            "alternate": "C"
        },
        "rs1799983": {
            "gene": "NOS3",
            "genotype": "GT",
            "chromosome": "7",
            "position": 150696111,
            "reference": "G",
            "alternate": "T"
        },
        "rs1801282": {
            "gene": "PPARG",
            "genotype": "CC",
            "chromosome": "3",
            "position": 12393125,
            "reference": "C",
            "alternate": "G"
        },
        "rs2228570": {
            "gene": "VDR",
            "genotype": "CT",
            "chromosome": "12",
            "position": 48272895,
            "reference": "A",
            "alternate": "G"
        },
        "rs1544410": {
            "gene": "VDR",
            "genotype": "CT",
            "chromosome": "12",
            "position": 48239835,
            "reference": "C",
            "alternate": "T"
        },
        "rs2282679": {
            "gene": "GC",
            "genotype": "AC",
            "chromosome": "4",
            "position": 72618334,
            "reference": "A",
            "alternate": "C"
        }
    }
    
    return example_variants


def run_example_analysis() -> NutritionReport:
    """
    Run an example nutrition analysis.
    
    Returns:
        Generated nutrition report
    """
    logger.info("Loading example variant data")
    variant_data = load_example_variants()
    
    logger.info(f"Loaded {len(variant_data)} variants")
    
    # Initialize the analyzer
    analyzer = NutritionAnalyzer(variant_data, confidence_threshold=0.6)
    
    # Run the analysis
    logger.info("Analyzing variants for nutritional effects")
    variant_effects = analyzer.analyze_variants()
    logger.info(f"Found nutritional effects for {len(variant_effects)} variants")
    
    # Aggregate nutrient needs
    logger.info("Aggregating nutrient needs")
    nutrient_needs = analyzer.aggregate_nutrient_needs()
    logger.info(f"Identified {len(nutrient_needs)} nutrient needs")
    
    # Aggregate food sensitivities
    logger.info("Aggregating food sensitivities")
    food_sensitivities = analyzer.aggregate_food_sensitivities()
    logger.info(f"Identified {len(food_sensitivities)} food sensitivities")
    
    # Aggregate dietary responses
    logger.info("Aggregating dietary responses")
    dietary_responses = analyzer.aggregate_dietary_responses()
    logger.info(f"Identified {len(dietary_responses)} dietary responses")
    
    # Generate nutrition plan
    logger.info("Generating personalized nutrition plan")
    nutrition_plan = analyzer.generate_nutrition_plan()
    
    # Generate final report
    logger.info("Generating final nutrition report")
    report = analyzer.generate_report(individual_id="example-user-123")
    
    return report


def save_example_report(report: NutritionReport, output_path: str = "nutrition_report.json") -> None:
    """
    Save the nutrition report to a JSON file.
    
    Args:
        report: The nutrition report to save
        output_path: Path to save the report to
    """
    # Convert the report to a dictionary
    report_dict = report.to_dict()
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    
    logger.info(f"Saved nutrition report to {output_path}")


def main() -> None:
    """Run the example nutrition analysis and save the report."""
    try:
        # Run the analysis
        report = run_example_analysis()
        
        # Create output directory if it doesn't exist
        Path("results").mkdir(exist_ok=True)
        
        # Save the report
        save_example_report(report, "results/nutrition_report.json")
        
        # Print a summary of the findings
        print("\n=== Nutrition Analysis Summary ===")
        print(f"Analyzed {report.total_variants_analyzed} variants")
        print(f"Found {len(report.significant_variants)} variants with significant effects")
        print(f"Identified {len(report.nutrient_needs)} nutrient needs")
        print(f"Identified {len(report.food_sensitivities)} food sensitivities")
        print(f"Identified {len(report.dietary_responses)} dietary responses")
        
        if report.nutrition_plan:
            print(f"\nRecommended Diet: {report.nutrition_plan.primary_diet_pattern.name}")
            print(f"Diet Score: {report.nutrition_plan.primary_diet_pattern.score:.2f}")
            print(f"Diet Confidence: {report.nutrition_plan.primary_diet_pattern.confidence:.2f}")
            
            if report.nutrition_plan.key_nutrient_needs:
                print("\nTop Nutrient Priorities:")
                for need in report.nutrition_plan.key_nutrient_needs:
                    direction = "Increased" if need.adjustment_factor > 1.0 else "Decreased" if need.adjustment_factor < 1.0 else "Normal"
                    print(f"  - {need.name}: {direction} need (adjustment: {need.adjustment_factor:.2f}x)")
            
            if report.nutrition_plan.food_sensitivities:
                print("\nFood Sensitivities:")
                for sensitivity in report.nutrition_plan.food_sensitivities:
                    print(f"  - {sensitivity.food_id}: {sensitivity.severity.value} sensitivity")
            
            if report.nutrition_plan.supplement_recommendations:
                print("\nSupplement Recommendations:")
                for supp in report.nutrition_plan.supplement_recommendations:
                    print(f"  - {supp['nutrient']}: {supp['dosage']} ({supp['priority']} priority)")
        
        print("\nFull report saved to results/nutrition_report.json")
        
    except Exception as e:
        logger.error(f"Error running nutrition analysis: {e}")
        raise


if __name__ == "__main__":
    main() 