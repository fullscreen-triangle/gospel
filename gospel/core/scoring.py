"""
Variant scoring module for Gospel.

This module handles the scoring of variants based on their annotations,
calculating impact scores for different domains (fitness, pharmacogenetics, nutrition).
"""

import logging
import math
import json
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .variant import Variant, VariantType

logger = logging.getLogger(__name__)


class DomainScorer:
    """Base class for domain-specific variant scoring."""
    
    def __init__(self, config: Dict):
        """Initialize the domain scorer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.weights = config.get("weights", {})
        self.thresholds = config.get("thresholds", {})
    
    def score_variant(self, variant: Variant) -> Dict:
        """Score a variant for the specific domain.
        
        Args:
            variant: The variant to score
            
        Returns:
            Domain-specific scores and metadata
        """
        # This method should be implemented by subclasses
        raise NotImplementedError


class FitnessScorer(DomainScorer):
    """Scorer for fitness-related impacts of variants."""
    
    def score_variant(self, variant: Variant) -> Dict:
        """Score a variant for fitness-related impacts.
        
        Args:
            variant: The variant to score
            
        Returns:
            Fitness domain scores and related traits
        """
        # Extract relevant features from the variant's functional impact
        impact = variant.functional_impact
        gene_name = impact.get("gene_name", "")
        consequence = impact.get("consequence", "")
        
        # Default scores
        sprint_score = 0.0
        power_score = 0.0
        endurance_score = 0.0
        injury_risk_score = 0.0
        recovery_score = 0.0
        overall_score = 0.0
        
        # Check for fitness genes
        # In a real implementation, this would use actual gene-trait associations
        # For demonstration, use simulated associations based on gene name hash
        if gene_name:
            gene_hash = hash(gene_name) % 1000
            
            # ACTN3 - associated with sprint/power performance
            if gene_name == "ACTN3" or gene_hash % 100 < 10:
                if variant.reference == "C" and variant.alternate == "T":  # R577X
                    sprint_score = 0.92
                    power_score = 0.85
                    relevant_traits = ["sprint", "power", "muscle_composition"]
                    
            # PPARGC1A - associated with endurance
            elif gene_name == "PPARGC1A" or gene_hash % 100 < 20:
                endurance_score = 0.78
                recovery_score = 0.65
                relevant_traits = ["endurance", "mitochondrial_function"]
                
            # COL1A1 - associated with injury risk
            elif gene_name == "COL1A1" or gene_hash % 100 < 30:
                injury_risk_score = 0.70
                relevant_traits = ["connective_tissue", "injury_risk"]
                
            # IL6 - associated with recovery
            elif gene_name == "IL6" or gene_hash % 100 < 40:
                recovery_score = 0.82
                relevant_traits = ["inflammation", "recovery"]
                
            # Other fitness genes with moderate scores
            else:
                # Generate scores based on gene hash for demonstration
                sprint_score = (gene_hash % 10) / 10.0
                power_score = ((gene_hash + 1) % 10) / 10.0
                endurance_score = ((gene_hash + 2) % 10) / 10.0
                injury_risk_score = ((gene_hash + 3) % 10) / 10.0
                recovery_score = ((gene_hash + 4) % 10) / 10.0
                relevant_traits = []
        
        # Apply consequence modifiers
        consequence_factor = self._get_consequence_factor(consequence)
        sprint_score *= consequence_factor
        power_score *= consequence_factor
        endurance_score *= consequence_factor
        injury_risk_score *= consequence_factor
        recovery_score *= consequence_factor
        
        # Calculate overall fitness score - weighted average of component scores
        weights = {
            "sprint": self.weights.get("sprint", 0.2),
            "power": self.weights.get("power", 0.2),
            "endurance": self.weights.get("endurance", 0.2),
            "injury_risk": self.weights.get("injury_risk", 0.2),
            "recovery": self.weights.get("recovery", 0.2)
        }
        
        overall_score = (
            sprint_score * weights["sprint"] +
            power_score * weights["power"] +
            endurance_score * weights["endurance"] +
            injury_risk_score * weights["injury_risk"] +
            recovery_score * weights["recovery"]
        )
        
        # Filter out low scores for relevant traits
        score_threshold = self.thresholds.get("relevant_trait", 0.4)
        relevant_traits = []
        
        if sprint_score >= score_threshold:
            relevant_traits.append("sprint")
        if power_score >= score_threshold:
            relevant_traits.append("power")
        if endurance_score >= score_threshold:
            relevant_traits.append("endurance")
        if injury_risk_score >= score_threshold:
            relevant_traits.append("injury_risk")
        if recovery_score >= score_threshold:
            relevant_traits.append("recovery")
        
        return {
            "score": overall_score,
            "component_scores": {
                "sprint": sprint_score,
                "power": power_score,
                "endurance": endurance_score,
                "injury_risk": injury_risk_score,
                "recovery": recovery_score
            },
            "relevant_traits": relevant_traits
        }
    
    def _get_consequence_factor(self, consequence: str) -> float:
        """Get a scaling factor based on variant consequence.
        
        Args:
            consequence: The variant consequence
            
        Returns:
            Scaling factor for scores
        """
        # Higher impact consequences have higher factors
        if consequence in ["stop_gained", "frameshift"]:
            return 1.0
        elif consequence in ["missense", "splice_region", "splice_disruption"]:
            return 0.8
        elif consequence in ["inframe_insertion", "inframe_deletion"]:
            return 0.7
        elif consequence == "regulatory_region":
            return 0.5
        elif consequence == "synonymous":
            return 0.1
        else:
            return 0.3


class PharmacoScorer(DomainScorer):
    """Scorer for pharmacogenetic impacts of variants."""
    
    def score_variant(self, variant: Variant) -> Dict:
        """Score a variant for pharmacogenetic impacts.
        
        Args:
            variant: The variant to score
            
        Returns:
            Pharmacogenetic domain scores and related drugs
        """
        # Extract relevant features from the variant's functional impact
        impact = variant.functional_impact
        gene_name = impact.get("gene_name", "")
        consequence = impact.get("consequence", "")
        
        # Default scores and empty lists
        metabolism_score = 0.0
        efficacy_score = 0.0
        safety_score = 0.0
        overall_score = 0.0
        relevant_drugs = []
        
        # Check for pharmacogenetic genes
        # In a real implementation, this would use actual gene-drug associations
        # For demonstration, use simulated associations based on gene name hash
        if gene_name:
            gene_hash = hash(gene_name) % 1000
            
            # CYP2D6 - metabolizes many drugs
            if gene_name == "CYP2D6" or gene_hash % 100 < 10:
                metabolism_score = 0.95
                safety_score = 0.80
                relevant_drugs = ["codeine", "tamoxifen", "antidepressants"]
                
            # VKORC1 - affects warfarin response
            elif gene_name == "VKORC1" or gene_hash % 100 < 20:
                efficacy_score = 0.90
                safety_score = 0.85
                relevant_drugs = ["warfarin"]
                
            # SLCO1B1 - affects statin metabolism
            elif gene_name == "SLCO1B1" or gene_hash % 100 < 30:
                metabolism_score = 0.75
                safety_score = 0.70
                relevant_drugs = ["statins", "simvastatin"]
                
            # Other pharmacogenetic genes with moderate scores
            else:
                # Generate scores based on gene hash for demonstration
                metabolism_score = (gene_hash % 10) / 10.0
                efficacy_score = ((gene_hash + 1) % 10) / 10.0
                safety_score = ((gene_hash + 2) % 10) / 10.0
                relevant_drugs = []
        
        # Apply consequence modifiers
        consequence_factor = self._get_consequence_factor(consequence)
        metabolism_score *= consequence_factor
        efficacy_score *= consequence_factor
        safety_score *= consequence_factor
        
        # Calculate overall pharmacogenetic score - weighted average of component scores
        weights = {
            "metabolism": self.weights.get("metabolism", 0.4),
            "efficacy": self.weights.get("efficacy", 0.3),
            "safety": self.weights.get("safety", 0.3)
        }
        
        overall_score = (
            metabolism_score * weights["metabolism"] +
            efficacy_score * weights["efficacy"] +
            safety_score * weights["safety"]
        )
        
        return {
            "score": overall_score,
            "component_scores": {
                "metabolism": metabolism_score,
                "efficacy": efficacy_score,
                "safety": safety_score
            },
            "relevant_drugs": relevant_drugs
        }
    
    def _get_consequence_factor(self, consequence: str) -> float:
        """Get a scaling factor based on variant consequence.
        
        Args:
            consequence: The variant consequence
            
        Returns:
            Scaling factor for scores
        """
        # Higher impact consequences have higher factors
        if consequence in ["stop_gained", "frameshift"]:
            return 1.0
        elif consequence in ["missense", "splice_region", "splice_disruption"]:
            return 0.9
        elif consequence in ["inframe_insertion", "inframe_deletion"]:
            return 0.7
        elif consequence == "regulatory_region":
            return 0.5
        elif consequence == "synonymous":
            return 0.2
        else:
            return 0.3


class NutritionScorer(DomainScorer):
    """Scorer for nutritional impacts of variants."""
    
    def score_variant(self, variant: Variant) -> Dict:
        """Score a variant for nutritional impacts.
        
        Args:
            variant: The variant to score
            
        Returns:
            Nutrition domain scores and related nutrients
        """
        # Extract relevant features from the variant's functional impact
        impact = variant.functional_impact
        gene_name = impact.get("gene_name", "")
        consequence = impact.get("consequence", "")
        
        # Default scores and empty lists
        macronutrient_score = 0.0
        micronutrient_score = 0.0
        sensitivity_score = 0.0
        oxidative_stress_score = 0.0
        overall_score = 0.0
        relevant_nutrients = []
        
        # Check for nutrition-related genes
        # In a real implementation, this would use actual gene-nutrient associations
        # For demonstration, use simulated associations based on gene name hash
        if gene_name:
            gene_hash = hash(gene_name) % 1000
            
            # MTHFR - affects folate metabolism
            if gene_name == "MTHFR" or gene_hash % 100 < 10:
                micronutrient_score = 0.85
                relevant_nutrients = ["folate", "vitamin_b12", "vitamin_b6"]
                
            # FTO - affects fat metabolism
            elif gene_name == "FTO" or gene_hash % 100 < 20:
                macronutrient_score = 0.80
                relevant_nutrients = ["fat", "carbohydrates"]
                
            # MCM6/LCT - affects lactose metabolism
            elif gene_name in ["MCM6", "LCT"] or gene_hash % 100 < 30:
                sensitivity_score = 0.90
                relevant_nutrients = ["lactose", "dairy"]
                
            # SOD2 - affects oxidative stress
            elif gene_name == "SOD2" or gene_hash % 100 < 40:
                oxidative_stress_score = 0.75
                relevant_nutrients = ["antioxidants", "vitamin_e", "vitamin_c"]
                
            # Other nutrition genes with moderate scores
            else:
                # Generate scores based on gene hash for demonstration
                macronutrient_score = (gene_hash % 10) / 10.0
                micronutrient_score = ((gene_hash + 1) % 10) / 10.0
                sensitivity_score = ((gene_hash + 2) % 10) / 10.0
                oxidative_stress_score = ((gene_hash + 3) % 10) / 10.0
                relevant_nutrients = []
        
        # Apply consequence modifiers
        consequence_factor = self._get_consequence_factor(consequence)
        macronutrient_score *= consequence_factor
        micronutrient_score *= consequence_factor
        sensitivity_score *= consequence_factor
        oxidative_stress_score *= consequence_factor
        
        # Calculate overall nutrition score - weighted average of component scores
        weights = {
            "macronutrient": self.weights.get("macronutrient", 0.25),
            "micronutrient": self.weights.get("micronutrient", 0.25),
            "sensitivity": self.weights.get("sensitivity", 0.25),
            "oxidative_stress": self.weights.get("oxidative_stress", 0.25)
        }
        
        overall_score = (
            macronutrient_score * weights["macronutrient"] +
            micronutrient_score * weights["micronutrient"] +
            sensitivity_score * weights["sensitivity"] +
            oxidative_stress_score * weights["oxidative_stress"]
        )
        
        # Filter out low scores for relevant nutrients
        score_threshold = self.thresholds.get("relevant_nutrient", 0.4)
        filtered_nutrients = []
        
        if macronutrient_score >= score_threshold and "carbohydrates" in relevant_nutrients:
            filtered_nutrients.append("carbohydrates")
        if macronutrient_score >= score_threshold and "fat" in relevant_nutrients:
            filtered_nutrients.append("fat")
        if macronutrient_score >= score_threshold and "protein" in relevant_nutrients:
            filtered_nutrients.append("protein")
        
        if micronutrient_score >= score_threshold:
            for nutrient in relevant_nutrients:
                if nutrient in ["folate", "vitamin_b12", "vitamin_b6", "vitamin_d", "iron", "calcium"]:
                    filtered_nutrients.append(nutrient)
        
        if sensitivity_score >= score_threshold:
            for nutrient in relevant_nutrients:
                if nutrient in ["lactose", "gluten", "dairy", "caffeine", "alcohol"]:
                    filtered_nutrients.append(nutrient)
        
        if oxidative_stress_score >= score_threshold:
            for nutrient in relevant_nutrients:
                if nutrient in ["antioxidants", "vitamin_e", "vitamin_c", "selenium"]:
                    filtered_nutrients.append(nutrient)
        
        return {
            "score": overall_score,
            "component_scores": {
                "macronutrient": macronutrient_score,
                "micronutrient": micronutrient_score,
                "sensitivity": sensitivity_score,
                "oxidative_stress": oxidative_stress_score
            },
            "relevant_nutrients": filtered_nutrients
        }
    
    def _get_consequence_factor(self, consequence: str) -> float:
        """Get a scaling factor based on variant consequence.
        
        Args:
            consequence: The variant consequence
            
        Returns:
            Scaling factor for scores
        """
        # Higher impact consequences have higher factors
        if consequence in ["stop_gained", "frameshift"]:
            return 1.0
        elif consequence in ["missense", "splice_region", "splice_disruption"]:
            return 0.85
        elif consequence in ["inframe_insertion", "inframe_deletion"]:
            return 0.7
        elif consequence == "regulatory_region":
            return 0.6
        elif consequence == "synonymous":
            return 0.1
        else:
            return 0.4


class VariantScorer:
    """Master scorer for calculating variant scores across all domains."""
    
    def __init__(self, config: Dict):
        """Initialize the variant scorer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize domain-specific scorers
        self.scorers = {
            "fitness": FitnessScorer(config.get("fitness_params", {})),
            "pharmacogenetics": PharmacoScorer(config.get("pharma_params", {})),
            "nutrition": NutritionScorer(config.get("nutrition_params", {}))
        }
        
        # Domain weights for integrated score
        self.domain_weights = {
            "fitness": config.get("domain_weights", {}).get("fitness", 1.0),
            "pharmacogenetics": config.get("domain_weights", {}).get("pharmacogenetics", 1.0),
            "nutrition": config.get("domain_weights", {}).get("nutrition", 1.0)
        }
        
        # Conservation weight
        self.conservation_weight = config.get("conservation_weight", 0.5)
    
    def score_variants(self, variants: List[Variant]) -> List[Variant]:
        """Score a list of variants across all domains.
        
        Args:
            variants: List of variants to score
            
        Returns:
            List of scored variants
        """
        logger.info(f"Scoring {len(variants)} variants")
        
        scored_variants = []
        
        for variant in variants:
            scored_variant = self.score_variant(variant)
            scored_variants.append(scored_variant)
        
        logger.info(f"Scored {len(scored_variants)} variants")
        return scored_variants
    
    def score_variant(self, variant: Variant) -> Variant:
        """Score a single variant across all domains.
        
        Args:
            variant: The variant to score
            
        Returns:
            Scored variant
        """
        # Apply domain-specific scoring
        for domain, scorer in self.scorers.items():
            domain_scores = scorer.score_variant(variant)
            variant.domain_scores[domain] = domain_scores
        
        # Calculate integrated score
        integrated_score = self._calculate_integrated_score(variant)
        variant.domain_scores["integrated"] = {
            "score": integrated_score
        }
        
        return variant
    
    def _calculate_integrated_score(self, variant: Variant) -> float:
        """Calculate an integrated score across all domains.
        
        Args:
            variant: The scored variant
            
        Returns:
            Integrated score
        """
        # Extract domain scores
        fitness_score = variant.domain_scores.get("fitness", {}).get("score", 0.0)
        pharma_score = variant.domain_scores.get("pharmacogenetics", {}).get("score", 0.0)
        nutrition_score = variant.domain_scores.get("nutrition", {}).get("score", 0.0)
        
        # Extract conservation score if available
        conservation_score = variant.functional_impact.get("conservation_score", 0.5)
        
        # Apply domain weights
        weighted_fitness = fitness_score * self.domain_weights["fitness"]
        weighted_pharma = pharma_score * self.domain_weights["pharmacogenetics"]
        weighted_nutrition = nutrition_score * self.domain_weights["nutrition"]
        
        # Sum the weighted scores
        sum_weights = sum(self.domain_weights.values())
        if sum_weights == 0:
            sum_weights = 1.0  # Avoid division by zero
            
        weighted_sum = (weighted_fitness + weighted_pharma + weighted_nutrition) / sum_weights
        
        # Apply conservation adjustment
        conservation_factor = 1.0 + (conservation_score - 0.5) * self.conservation_weight
        integrated_score = weighted_sum * conservation_factor
        
        # Ensure score is between 0 and 1
        integrated_score = max(0.0, min(1.0, integrated_score))
        
        return integrated_score


def compute_domain_scores(variant: Variant, domain: str, config: Dict) -> Dict:
    """Compute domain-specific scores for a variant.
    
    Args:
        variant: The variant to score
        domain: The domain to score for
        config: Configuration dictionary
        
    Returns:
        Domain-specific scores
    """
    if domain == "fitness":
        scorer = FitnessScorer(config.get("fitness_params", {}))
    elif domain == "pharmacogenetics":
        scorer = PharmacoScorer(config.get("pharma_params", {}))
    elif domain == "nutrition":
        scorer = NutritionScorer(config.get("nutrition_params", {}))
    else:
        logger.warning(f"Unknown domain: {domain}")
        return {}
    
    return scorer.score_variant(variant)
