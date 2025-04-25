"""
Fitness domain analysis for Gospel.

This module provides analysis functions for fitness-related genomic insights.
"""

import logging
import statistics
from typing import Dict, List, Optional, Set, Tuple, Union

from ...core.variant import Variant
from .models import FitnessDomain, TraitProfile, GeneticEffect
from .constants import FITNESS_TRAITS, FITNESS_SNPS

logger = logging.getLogger(__name__)


def analyze_variant_fitness_impact(variant: Variant, config: Dict = None) -> Dict:
    """Analyze the fitness impact of a single variant.
    
    Args:
        variant: The variant to analyze
        config: Optional configuration parameters
        
    Returns:
        Dictionary of fitness impact scores and traits
    """
    # Create a fitness domain instance and analyze the variant
    fitness_domain = FitnessDomain(config)
    return fitness_domain.analyze_variant(variant)


def analyze_genotype_fitness(variants: List[Variant], config: Dict = None) -> Dict:
    """Analyze the fitness profile of a complete genotype.
    
    Args:
        variants: List of all variants to analyze
        config: Optional configuration parameters
        
    Returns:
        Complete fitness profile with trait summaries and recommendations
    """
    # Create a fitness domain instance and analyze all variants
    fitness_domain = FitnessDomain(config)
    return fitness_domain.analyze_variants(variants)


def identify_key_fitness_variants(variants: List[Variant]) -> List[Variant]:
    """Identify variants with known fitness impacts.
    
    Args:
        variants: List of variants to screen
        
    Returns:
        List of variants with known fitness impacts
    """
    key_variants = []
    
    # Check for known fitness SNPs
    for variant in variants:
        variant_id = variant.id.lower()
        
        # Check if this is a known fitness-related SNP
        if variant_id in FITNESS_SNPS or variant_id in [snp.lower() for snp in FITNESS_SNPS]:
            key_variants.append(variant)
            continue
        
        # Check gene-based associations
        gene_name = variant.functional_impact.get("gene_name", "")
        if gene_name:
            from .constants import PERFORMANCE_GENES, ENDURANCE_GENES, POWER_GENES
            from .constants import INJURY_RISK_GENES, RECOVERY_GENES
            
            if (gene_name in PERFORMANCE_GENES or 
                gene_name in ENDURANCE_GENES or 
                gene_name in POWER_GENES or
                gene_name in INJURY_RISK_GENES or 
                gene_name in RECOVERY_GENES):
                key_variants.append(variant)
    
    return key_variants


def calculate_trait_distribution(variants: List[Variant]) -> Dict[str, float]:
    """Calculate the distribution of fitness traits in a variant set.
    
    Args:
        variants: List of variants to analyze
        
    Returns:
        Dictionary mapping trait names to prevalence scores
    """
    trait_counts = {trait: 0 for trait in FITNESS_TRAITS}
    
    # Count traits across all variants
    for variant in variants:
        if "fitness" in variant.domain_scores:
            fitness_score = variant.domain_scores["fitness"]
            for trait in fitness_score.get("relevant_traits", []):
                if trait in trait_counts:
                    trait_counts[trait] += 1
    
    # Calculate distribution percentages
    total_counts = sum(trait_counts.values())
    if total_counts > 0:
        trait_distribution = {trait: count/total_counts for trait, count in trait_counts.items()}
    else:
        trait_distribution = {trait: 0.0 for trait in FITNESS_TRAITS}
    
    return trait_distribution


def generate_fitness_report(variants: List[Variant], config: Dict = None) -> Dict:
    """Generate a comprehensive fitness report from genomic data.
    
    Args:
        variants: List of variants to analyze
        config: Optional configuration parameters
        
    Returns:
        Complete fitness report with summaries and recommendations
    """
    # Run complete fitness analysis
    fitness_domain = FitnessDomain(config)
    analysis_results = fitness_domain.analyze_variants(variants)
    
    # Identify key variants
    key_variants = identify_key_fitness_variants(variants)
    
    # Calculate trait distribution
    trait_distribution = calculate_trait_distribution(variants)
    
    # Determine primary fitness type
    primary_traits = sorted(
        trait_distribution.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    # Generate overall recommendation
    if primary_traits[0][0] in ["power", "sprint", "muscle_composition"]:
        primary_type = "Power/Strength"
        overall_recommendation = "Your genetic profile suggests a predisposition toward power and strength-based activities. Consider focusing on resistance training, sprinting, and explosive movements."
    elif primary_traits[0][0] in ["endurance", "vo2_max", "metabolism"]:
        primary_type = "Endurance"
        overall_recommendation = "Your genetic profile suggests a predisposition toward endurance activities. Consider focusing on aerobic training, distance running, cycling, or swimming."
    else:
        primary_type = "Balanced"
        overall_recommendation = "Your genetic profile shows a balanced distribution of fitness traits. Consider a mixed training approach that incorporates both strength and endurance elements."
    
    # Assemble the report
    report = {
        "summary": analysis_results["summary"],
        "primary_fitness_type": primary_type,
        "overall_recommendation": overall_recommendation,
        "trait_distribution": trait_distribution,
        "trait_profiles": analysis_results["traits"],
        "key_variants": [
            {
                "id": v.id,
                "gene": v.functional_impact.get("gene_name", "Unknown"),
                "score": v.domain_scores.get("fitness", {}).get("score", 0),
                "traits": v.domain_scores.get("fitness", {}).get("relevant_traits", [])
            }
            for v in key_variants[:10]  # Include top 10 key variants
        ]
    }
    
    return report


def compare_fitness_profiles(profile1: Dict, profile2: Dict) -> Dict:
    """Compare two fitness profiles to identify similarities and differences.
    
    Args:
        profile1: First fitness profile
        profile2: Second fitness profile
        
    Returns:
        Comparison results highlighting similarities and differences
    """
    # Compare trait distributions
    trait_similarities = {}
    trait_differences = {}
    
    for trait in FITNESS_TRAITS:
        score1 = profile1.get("trait_distribution", {}).get(trait, 0)
        score2 = profile2.get("trait_distribution", {}).get(trait, 0)
        
        difference = abs(score1 - score2)
        if difference < 0.1:
            trait_similarities[trait] = (score1, score2)
        else:
            trait_differences[trait] = (score1, score2)
    
    # Compare overall fitness types
    type1 = profile1.get("primary_fitness_type", "Unknown")
    type2 = profile2.get("primary_fitness_type", "Unknown")
    same_type = type1 == type2
    
    # Find shared key variants
    variants1 = {v["id"] for v in profile1.get("key_variants", [])}
    variants2 = {v["id"] for v in profile2.get("key_variants", [])}
    shared_variants = variants1.intersection(variants2)
    
    return {
        "same_fitness_type": same_type,
        "type1": type1,
        "type2": type2,
        "trait_similarities": trait_similarities,
        "trait_differences": trait_differences,
        "shared_key_variants": len(shared_variants),
        "similarity_score": len(trait_similarities) / len(FITNESS_TRAITS) if FITNESS_TRAITS else 0.0
    }


def identify_optimal_exercise_types(fitness_profile: Dict) -> List[Dict]:
    """Identify optimal exercise types based on a fitness profile.
    
    Args:
        fitness_profile: Fitness analysis results
        
    Returns:
        List of recommended exercise types with compatibility scores
    """
    # Exercise type definitions with trait compatibility
    exercise_types = [
        {
            "name": "Weight Lifting",
            "traits": {"power": 0.9, "muscle_composition": 0.8, "recovery": 0.6, "metabolism": 0.4}
        },
        {
            "name": "Sprinting",
            "traits": {"sprint": 0.95, "power": 0.8, "muscle_composition": 0.6, "metabolism": 0.5}
        },
        {
            "name": "Marathon Running",
            "traits": {"endurance": 0.95, "vo2_max": 0.9, "metabolism": 0.8, "recovery": 0.7}
        },
        {
            "name": "Swimming",
            "traits": {"endurance": 0.8, "power": 0.6, "recovery": 0.7, "coordination": 0.8}
        },
        {
            "name": "Cycling",
            "traits": {"endurance": 0.85, "power": 0.7, "metabolism": 0.7, "recovery": 0.6}
        },
        {
            "name": "HIIT Training",
            "traits": {"power": 0.8, "endurance": 0.7, "recovery": 0.8, "metabolism": 0.9}
        },
        {
            "name": "Yoga",
            "traits": {"joint_flexibility": 0.9, "balance": 0.8, "coordination": 0.7, "recovery": 0.7}
        },
        {
            "name": "Team Sports",
            "traits": {"coordination": 0.8, "power": 0.6, "endurance": 0.7, "balance": 0.7}
        }
    ]
    
    # Get trait distribution from profile
    trait_distribution = fitness_profile.get("trait_distribution", {})
    if not trait_distribution:
        return []
    
    # Calculate compatibility score for each exercise type
    exercise_compatibility = []
    for exercise in exercise_types:
        score = 0.0
        weight_sum = 0.0
        
        for trait, weight in exercise["traits"].items():
            trait_score = trait_distribution.get(trait, 0.0)
            score += trait_score * weight
            weight_sum += weight
        
        if weight_sum > 0:
            compatibility = score / weight_sum
        else:
            compatibility = 0.0
            
        exercise_compatibility.append({
            "name": exercise["name"],
            "compatibility_score": compatibility,
            "primary_traits": list(exercise["traits"].keys())[:2]
        })
    
    # Sort by compatibility score
    exercise_compatibility.sort(key=lambda x: x["compatibility_score"], reverse=True)
    
    return exercise_compatibility


def calculate_injury_risk_profile(variants: List[Variant]) -> Dict:
    """Calculate an injury risk profile from genetic variants.
    
    Args:
        variants: List of variants to analyze
        
    Returns:
        Injury risk profile with affected tissues and risk scores
    """
    # Create fitness domain and analyze variants
    fitness_domain = FitnessDomain()
    fitness_domain.analyze_variants(variants)
    
    # Extract injury risk profile
    if "injury_risk" in fitness_domain.trait_profiles:
        profile = fitness_domain.trait_profiles["injury_risk"]
    else:
        # No significant injury risk variants found
        return {
            "overall_risk": "low",
            "risk_score": 0.2,
            "affected_tissues": {},
            "risk_factors": [],
            "prevention_recommendations": [
                "Follow standard injury prevention protocols",
                "Ensure proper warm-up before exercise",
                "Maintain proper form during activities"
            ]
        }
    
    # Categorize by affected tissue
    tissues = {
        "tendon": 0.0,
        "ligament": 0.0,
        "muscle": 0.0,
        "bone": 0.0,
        "joint": 0.0
    }
    
    risk_factors = []
    
    # Analyze contributing variants
    for variant in profile.contributing_variants:
        gene = variant.functional_impact.get("gene_name", "")
        
        # Classify tissues based on gene function
        if gene in ["COL1A1", "COL5A1", "MMP3", "TIMP2"]:
            tissues["tendon"] += 0.2
            tissues["ligament"] += 0.15
            risk_factors.append(f"{gene} variant affecting connective tissue")
            
        elif gene in ["ACTN3", "MYLK", "IGF1"]:
            tissues["muscle"] += 0.2
            risk_factors.append(f"{gene} variant affecting muscle properties")
            
        elif gene in ["ACAN", "GDF5", "VDR"]:
            tissues["joint"] += 0.2
            tissues["bone"] += 0.15
            risk_factors.append(f"{gene} variant affecting joint/bone structure")
    
    # Cap tissue risks at 1.0
    tissues = {t: min(1.0, s) for t, s in tissues.items()}
    
    # Overall risk classification
    risk_score = profile.score
    if risk_score > 0.7:
        risk_level = "high"
    elif risk_score > 0.4:
        risk_level = "moderate"
    else:
        risk_level = "low"
    
    # Generate prevention recommendations
    recommendations = []
    
    if tissues["tendon"] > 0.5 or tissues["ligament"] > 0.5:
        recommendations.append("Focus on gradual progression in training intensity to allow connective tissue adaptation")
        recommendations.append("Consider supportive equipment for vulnerable areas during high-intensity activities")
        
    if tissues["joint"] > 0.5:
        recommendations.append("Incorporate joint-friendly exercise options like swimming or cycling")
        recommendations.append("Consider supplements supporting joint health (glucosamine, chondroitin)")
        
    if tissues["muscle"] > 0.5:
        recommendations.append("Ensure adequate warm-up focused on muscle activation")
        recommendations.append("Pay special attention to eccentric strength training for injury prevention")
    
    # Always include general recommendations
    recommendations.append("Maintain proper form during exercise")
    recommendations.append("Ensure adequate recovery between intense training sessions")
    
    return {
        "overall_risk": risk_level,
        "risk_score": risk_score,
        "affected_tissues": tissues,
        "risk_factors": risk_factors[:5],  # Top 5 risk factors
        "prevention_recommendations": recommendations
    } 