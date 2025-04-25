"""
Fitness domain models for Gospel.

This module defines data models and classes for fitness-related genomic analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from ...core.variant import Variant, VariantType

logger = logging.getLogger(__name__)


@dataclass
class GeneticEffect:
    """Representation of a genetic effect on a fitness trait."""
    trait: str
    score: float
    direction: str  # 'positive', 'negative', or 'neutral'
    confidence: float
    mechanism: str = ""
    source_studies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate the effect data."""
        if self.score < 0 or self.score > 1:
            logger.warning(f"Effect score {self.score} outside expected range [0,1]")
            self.score = max(0, min(1, self.score))
            
        if self.direction not in ['positive', 'negative', 'neutral']:
            logger.warning(f"Invalid effect direction: {self.direction}")
            self.direction = 'neutral'
            
        if self.confidence < 0 or self.confidence > 1:
            logger.warning(f"Confidence value {self.confidence} outside expected range [0,1]")
            self.confidence = max(0, min(1, self.confidence))


@dataclass
class TraitProfile:
    """Profile of genetic effects on a specific fitness trait."""
    trait: str
    score: float
    contributing_variants: List[Variant] = field(default_factory=list)
    gene_contributions: Dict[str, float] = field(default_factory=dict)
    interpretation: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    def add_variant(self, variant: Variant, contribution: float):
        """Add a contributing variant to the trait profile.
        
        Args:
            variant: The variant contributing to this trait
            contribution: The contribution score of this variant
        """
        self.contributing_variants.append(variant)
        
        # Update gene contribution
        gene = variant.functional_impact.get("gene_name")
        if gene:
            if gene in self.gene_contributions:
                self.gene_contributions[gene] += contribution
            else:
                self.gene_contributions[gene] = contribution
                
        # Recalculate overall score
        self._calculate_score()
    
    def _calculate_score(self):
        """Calculate the overall trait score based on contributing variants."""
        if not self.contributing_variants:
            self.score = 0
            return
            
        total_contribution = 0
        for variant in self.contributing_variants:
            if "fitness" in variant.domain_scores:
                fitness_scores = variant.domain_scores["fitness"]
                for component, value in fitness_scores.get("component_scores", {}).items():
                    if component.lower() == self.trait.lower():
                        total_contribution += value
                        break
        
        # Normalize to 0-1 scale with diminishing returns for many variants
        if total_contribution > 0:
            self.score = min(1.0, total_contribution / (1.0 + 0.1 * (len(self.contributing_variants) - 1)))
        else:
            self.score = 0
    
    def generate_interpretation(self):
        """Generate a human-readable interpretation of the trait profile."""
        if self.score < 0.3:
            impact = "minimal"
        elif self.score < 0.6:
            impact = "moderate"
        else:
            impact = "significant"
            
        num_variants = len(self.contributing_variants)
        top_genes = sorted(self.gene_contributions.items(), key=lambda x: x[1], reverse=True)[:3]
        gene_text = ", ".join([gene for gene, _ in top_genes]) if top_genes else "unknown genes"
        
        self.interpretation = (
            f"Your genetic profile shows a {impact} impact on {self.trait} performance, "
            f"based on {num_variants} genetic variants primarily in {gene_text}."
        )
    
    def generate_recommendations(self):
        """Generate trait-specific recommendations based on the genetic profile."""
        # These would be customized for each specific trait
        if self.trait == "endurance":
            if self.score > 0.7:
                self.recommendations = [
                    "Focus on endurance training like distance running and cycling",
                    "Consider longer rest intervals between high-intensity sets",
                    "Your genetics favor stamina-based activities"
                ]
            elif self.score > 0.4:
                self.recommendations = [
                    "Balanced approach to training with moderate endurance focus",
                    "Include both interval and steady-state cardio in your routine"
                ]
            else:
                self.recommendations = [
                    "Focus on shorter, higher intensity training",
                    "Your genetics may favor power/strength over endurance"
                ]
        
        elif self.trait == "power":
            if self.score > 0.7:
                self.recommendations = [
                    "Focus on power-based activities like sprinting or weight lifting",
                    "Consider higher weights with lower repetitions",
                    "Your genetics favor explosive movements and fast-twitch muscle fibers"
                ]
            elif self.score > 0.4:
                self.recommendations = [
                    "Balanced approach to training with moderate power focus",
                    "Include both strength and power exercises in your routine"
                ]
            else:
                self.recommendations = [
                    "Consider endurance-focused training methods",
                    "Your genetics may favor slow-twitch muscle fibers"
                ]
                
        elif self.trait == "injury_risk":
            if self.score > 0.7:
                self.recommendations = [
                    "Pay extra attention to proper warm-up routines",
                    "Consider longer recovery periods between intense training sessions",
                    "Focus on proper technique and form to prevent injuries"
                ]
            else:
                self.recommendations = [
                    "Standard injury prevention practices are sufficient",
                    "Maintain good training habits and adequate recovery"
                ]
                
        # Add other trait-specific recommendations here
        else:
            self.recommendations = [
                "Maintain a balanced approach to fitness training",
                "Monitor your response to different exercise types",
                "Consider consulting with a fitness professional for personalized advice"
            ]


class FitnessDomain:
    """Domain implementation for fitness-related genomic analysis."""
    
    def __init__(self, config: Dict = None):
        """Initialize the fitness domain with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.trait_profiles = {}
        self.analyzed_variants = []
        
        # Configure weights for different traits
        self.trait_weights = self.config.get("trait_weights", {
            "endurance": 1.0,
            "power": 1.0,
            "injury_risk": 1.0,
            "recovery": 1.0,
            "muscle_composition": 0.8,
            "vo2_max": 0.9,
            "metabolism": 0.7,
            "tendon_strength": 0.6
        })
        
        # Configure scoring thresholds
        self.thresholds = self.config.get("thresholds", {
            "high_impact": 0.7,
            "moderate_impact": 0.4,
            "relevant_contribution": 0.2
        })
    
    def analyze_variant(self, variant: Variant) -> Dict:
        """Analyze a variant for fitness-related effects.
        
        Args:
            variant: The variant to analyze
            
        Returns:
            Dictionary with fitness-related scores and traits
        """
        # Skip if already analyzed
        if variant in self.analyzed_variants:
            return variant.domain_scores.get("fitness", {})
            
        # Extract relevant features from variant
        impact = variant.functional_impact
        gene_name = impact.get("gene_name", "")
        consequence = impact.get("consequence", "")
        
        # Default scores
        sprint_score = 0.0
        power_score = 0.0
        endurance_score = 0.0
        injury_risk_score = 0.0
        recovery_score = 0.0
        muscle_comp_score = 0.0
        metabolism_score = 0.0
        
        # Apply gene-specific knowledge
        from .constants import PERFORMANCE_GENES, ENDURANCE_GENES, POWER_GENES
        from .constants import INJURY_RISK_GENES, RECOVERY_GENES
        
        # Calculate scores based on gene associations
        if gene_name in POWER_GENES:
            power_score = POWER_GENES[gene_name].get("score", 0.0)
            sprint_score = POWER_GENES[gene_name].get("sprint_factor", 0.8) * power_score
            muscle_comp_score = POWER_GENES[gene_name].get("muscle_comp_factor", 0.7) * power_score
            
        if gene_name in ENDURANCE_GENES:
            endurance_score = ENDURANCE_GENES[gene_name].get("score", 0.0)
            metabolism_score = ENDURANCE_GENES[gene_name].get("metabolism_factor", 0.7) * endurance_score
            
        if gene_name in INJURY_RISK_GENES:
            injury_risk_score = INJURY_RISK_GENES[gene_name].get("score", 0.0)
            
        if gene_name in RECOVERY_GENES:
            recovery_score = RECOVERY_GENES[gene_name].get("score", 0.0)
        
        # Apply consequence modifiers
        consequence_factor = self._get_consequence_factor(consequence)
        sprint_score *= consequence_factor
        power_score *= consequence_factor
        endurance_score *= consequence_factor
        injury_risk_score *= consequence_factor
        recovery_score *= consequence_factor
        muscle_comp_score *= consequence_factor
        metabolism_score *= consequence_factor
        
        # Calculate overall fitness score using weighted average
        component_scores = {
            "sprint": sprint_score,
            "power": power_score,
            "endurance": endurance_score,
            "injury_risk": injury_risk_score,
            "recovery": recovery_score,
            "muscle_composition": muscle_comp_score,
            "metabolism": metabolism_score
        }
        
        overall_score = self._calculate_overall_score(component_scores)
        
        # Determine relevant traits
        relevant_traits = []
        for trait, score in component_scores.items():
            if score >= self.thresholds["relevant_contribution"]:
                relevant_traits.append(trait)
        
        # Create fitness domain score
        fitness_score = {
            "score": overall_score,
            "component_scores": component_scores,
            "relevant_traits": relevant_traits
        }
        
        # Update variant with score
        if hasattr(variant, 'domain_scores'):
            variant.domain_scores["fitness"] = fitness_score
        
        # Add to list of analyzed variants
        self.analyzed_variants.append(variant)
        
        # Update trait profiles
        for trait, score in component_scores.items():
            if score >= self.thresholds["relevant_contribution"]:
                if trait not in self.trait_profiles:
                    self.trait_profiles[trait] = TraitProfile(trait=trait, score=0.0)
                
                self.trait_profiles[trait].add_variant(variant, score)
        
        return fitness_score
    
    def analyze_variants(self, variants: List[Variant]) -> Dict:
        """Analyze a list of variants for fitness-related effects.
        
        Args:
            variants: List of variants to analyze
            
        Returns:
            Dictionary with fitness domain analysis results
        """
        logger.info(f"Analyzing {len(variants)} variants for fitness domain")
        
        # Process each variant
        for variant in variants:
            self.analyze_variant(variant)
        
        # Generate interpretations and recommendations for each trait profile
        for profile in self.trait_profiles.values():
            profile.generate_interpretation()
            profile.generate_recommendations()
        
        # Prepare domain summary
        total_variants = len(self.analyzed_variants)
        relevant_variants = sum(1 for v in self.analyzed_variants 
                              if v.domain_scores.get("fitness", {}).get("score", 0) >= 
                              self.thresholds["relevant_contribution"])
        
        high_impact_variants = sum(1 for v in self.analyzed_variants 
                                 if v.domain_scores.get("fitness", {}).get("score", 0) >= 
                                 self.thresholds["high_impact"])
        
        # Prepare trait summaries
        trait_summaries = {}
        for trait, profile in self.trait_profiles.items():
            trait_summaries[trait] = {
                "score": profile.score,
                "relevant_variants": len(profile.contributing_variants),
                "top_genes": dict(sorted(profile.gene_contributions.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True)[:5]),
                "interpretation": profile.interpretation,
                "recommendations": profile.recommendations
            }
        
        # Overall domain impact assessment
        domain_scores = [v.domain_scores.get("fitness", {}).get("score", 0) 
                        for v in self.analyzed_variants]
        
        if domain_scores:
            impact_score = sum(domain_scores) / len(domain_scores)
        else:
            impact_score = 0.0
        
        return {
            "summary": {
                "total_variants": total_variants,
                "relevant_variants": relevant_variants,
                "high_impact_variants": high_impact_variants,
                "impact_score": impact_score
            },
            "traits": trait_summaries
        }
    
    def _calculate_overall_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate overall fitness score from component scores.
        
        Args:
            component_scores: Dictionary of component scores
            
        Returns:
            Overall fitness score
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for trait, score in component_scores.items():
            weight = self.trait_weights.get(trait, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
            
        return weighted_sum / total_weight
    
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