"""
Pharmacogenetics domain models for Gospel.

This module defines data models and classes for pharmacogenetic analysis.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

from ...core.variant import Variant, VariantType

logger = logging.getLogger(__name__)


class MetabolizerStatus(Enum):
    """Classification of drug metabolizer status."""
    POOR = "poor"
    INTERMEDIATE = "intermediate"
    NORMAL = "normal"
    RAPID = "rapid"
    ULTRARAPID = "ultrarapid"
    UNKNOWN = "unknown"


@dataclass
class DrugInteraction:
    """Representation of a drug-gene interaction."""
    drug_name: str
    gene_name: str
    impact_score: float
    interaction_type: str  # 'metabolism', 'target', 'transport', 'adverse'
    recommendation: str = ""
    evidence_level: str = ""
    source_studies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate the interaction data."""
        if self.impact_score < 0 or self.impact_score > 1:
            logger.warning(f"Impact score {self.impact_score} outside expected range [0,1]")
            self.impact_score = max(0, min(1, self.impact_score))


@dataclass
class DrugProfile:
    """Profile of genomic effects for a specific drug."""
    drug_name: str
    overall_score: float = 0.0
    metabolism_score: float = 0.0
    efficacy_score: float = 0.0
    safety_score: float = 0.0
    contributing_variants: List[Variant] = field(default_factory=list)
    gene_contributions: Dict[str, float] = field(default_factory=dict)
    metabolizer_status: MetabolizerStatus = MetabolizerStatus.UNKNOWN
    recommendations: List[str] = field(default_factory=list)
    
    def add_variant(self, variant: Variant, interaction: DrugInteraction):
        """Add a contributing variant to the drug profile.
        
        Args:
            variant: The variant contributing to this drug interaction
            interaction: The drug interaction details
        """
        self.contributing_variants.append(variant)
        
        # Update gene contribution
        gene = interaction.gene_name
        if gene:
            if gene in self.gene_contributions:
                self.gene_contributions[gene] += interaction.impact_score
            else:
                self.gene_contributions[gene] = interaction.impact_score
                
        # Update scores based on interaction type
        if interaction.interaction_type == "metabolism":
            self.metabolism_score = max(self.metabolism_score, interaction.impact_score)
        elif interaction.interaction_type == "target":
            self.efficacy_score = max(self.efficacy_score, interaction.impact_score)
        elif interaction.interaction_type == "adverse":
            self.safety_score = max(self.safety_score, interaction.impact_score)
            
        # Recalculate overall score
        self._calculate_overall_score()
        
        # Determine metabolizer status if relevant
        if interaction.interaction_type == "metabolism":
            self._determine_metabolizer_status()
    
    def _calculate_overall_score(self):
        """Calculate the overall drug interaction score."""
        # Weighted average of component scores
        weights = {
            "metabolism": 0.4,
            "efficacy": 0.3,
            "safety": 0.3
        }
        
        self.overall_score = (
            self.metabolism_score * weights["metabolism"] +
            self.efficacy_score * weights["efficacy"] +
            self.safety_score * weights["safety"]
        )
    
    def _determine_metabolizer_status(self):
        """Determine the metabolizer status based on variants."""
        # This would be more sophisticated in practice
        # Here we use a simplified approach based on metabolism score
        if self.metabolism_score > 0.9:
            self.metabolizer_status = MetabolizerStatus.ULTRARAPID
        elif self.metabolism_score > 0.7:
            self.metabolizer_status = MetabolizerStatus.RAPID
        elif self.metabolism_score > 0.3:
            self.metabolizer_status = MetabolizerStatus.NORMAL
        elif self.metabolism_score > 0.1:
            self.metabolizer_status = MetabolizerStatus.INTERMEDIATE
        else:
            self.metabolizer_status = MetabolizerStatus.POOR
    
    def generate_recommendations(self):
        """Generate drug-specific recommendations based on genetic profile."""
        from .constants import PGX_DRUGS
        
        # Clear existing recommendations
        self.recommendations = []
        
        # Get drug-specific recommendation logic if available
        if self.drug_name in PGX_DRUGS:
            drug_info = PGX_DRUGS[self.drug_name]
            recommendation_logic = drug_info.get("recommendation_logic", {})
            
            # Apply metabolizer status recommendations
            if str(self.metabolizer_status.value) in recommendation_logic:
                self.recommendations.append(
                    recommendation_logic[str(self.metabolizer_status.value)]
                )
                
            # Add general recommendations if available
            if "general" in recommendation_logic:
                self.recommendations.append(recommendation_logic["general"])
        
        # Generate generic recommendations based on scores
        if not self.recommendations:
            if self.metabolizer_status == MetabolizerStatus.POOR:
                self.recommendations.append(
                    f"Consider dose reduction for {self.drug_name} due to poor metabolism."
                )
            elif self.metabolizer_status == MetabolizerStatus.ULTRARAPID:
                self.recommendations.append(
                    f"Standard doses of {self.drug_name} may be insufficient due to ultrarapid metabolism."
                )
                
            if self.safety_score > 0.7:
                self.recommendations.append(
                    f"Monitor closely for adverse effects with {self.drug_name}."
                )
                
            if self.efficacy_score < 0.3:
                self.recommendations.append(
                    f"{self.drug_name} may have reduced efficacy based on your genetic profile."
                )
        
        # Always add a general caution
        if not any("consult" in r.lower() for r in self.recommendations):
            self.recommendations.append(
                "Always consult with a healthcare provider before making medication changes."
            )


class PharmacogeneticsDomain:
    """Domain implementation for pharmacogenetic genomic analysis."""
    
    def __init__(self, config: Dict = None):
        """Initialize the pharmacogenetics domain with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.drug_profiles = {}
        self.analyzed_variants = []
        
        # Configure scoring thresholds
        self.thresholds = self.config.get("thresholds", {
            "high_impact": 0.7,
            "moderate_impact": 0.4,
            "relevant_contribution": 0.2
        })
    
    def analyze_variant(self, variant: Variant) -> Dict:
        """Analyze a variant for pharmacogenetic effects.
        
        Args:
            variant: The variant to analyze
            
        Returns:
            Dictionary with pharmacogenetic scores and drug interactions
        """
        # Skip if already analyzed
        if variant in self.analyzed_variants:
            return variant.domain_scores.get("pharmacogenetics", {})
            
        # Extract relevant features from variant
        impact = variant.functional_impact
        gene_name = impact.get("gene_name", "")
        consequence = impact.get("consequence", "")
        
        # No analysis if no gene information
        if not gene_name:
            return {}
        
        # Default scores
        metabolism_score = 0.0
        efficacy_score = 0.0
        safety_score = 0.0
        
        # Analyze gene-drug interactions
        drug_interactions = []
        
        # Apply gene-specific knowledge
        from .constants import METABOLISM_GENES, DRUG_TARGET_GENES
        from .constants import DRUG_PATHWAY_GENES, ADVERSE_EFFECT_GENES
        
        # Check if variant affects drug metabolism
        if gene_name in METABOLISM_GENES:
            gene_info = METABOLISM_GENES[gene_name]
            metabolism_score = gene_info.get("impact_score", 0.0)
            
            # Create interactions for relevant drugs
            for drug_name in gene_info.get("affected_drugs", []):
                interaction = DrugInteraction(
                    drug_name=drug_name,
                    gene_name=gene_name,
                    impact_score=metabolism_score * self._get_consequence_factor(consequence),
                    interaction_type="metabolism",
                    evidence_level=gene_info.get("evidence_level", "moderate")
                )
                drug_interactions.append(interaction)
                
                # Update drug profile
                self._update_drug_profile(drug_name, variant, interaction)
        
        # Check if variant affects drug targets
        if gene_name in DRUG_TARGET_GENES:
            gene_info = DRUG_TARGET_GENES[gene_name]
            efficacy_score = gene_info.get("impact_score", 0.0)
            
            # Create interactions for relevant drugs
            for drug_name in gene_info.get("affected_drugs", []):
                interaction = DrugInteraction(
                    drug_name=drug_name,
                    gene_name=gene_name,
                    impact_score=efficacy_score * self._get_consequence_factor(consequence),
                    interaction_type="target",
                    evidence_level=gene_info.get("evidence_level", "moderate")
                )
                drug_interactions.append(interaction)
                
                # Update drug profile
                self._update_drug_profile(drug_name, variant, interaction)
                
        # Check if variant affects drug pathways
        if gene_name in DRUG_PATHWAY_GENES:
            gene_info = DRUG_PATHWAY_GENES[gene_name]
            pathway_score = gene_info.get("impact_score", 0.0)
            
            # Create interactions for relevant drugs
            for drug_name in gene_info.get("affected_drugs", []):
                interaction = DrugInteraction(
                    drug_name=drug_name,
                    gene_name=gene_name,
                    impact_score=pathway_score * self._get_consequence_factor(consequence),
                    interaction_type="pathway",
                    evidence_level=gene_info.get("evidence_level", "moderate")
                )
                drug_interactions.append(interaction)
                
                # Update drug profile
                self._update_drug_profile(drug_name, variant, interaction)
        
        # Check if variant affects adverse drug reactions
        if gene_name in ADVERSE_EFFECT_GENES:
            gene_info = ADVERSE_EFFECT_GENES[gene_name]
            safety_score = gene_info.get("impact_score", 0.0)
            
            # Create interactions for relevant drugs
            for drug_name in gene_info.get("affected_drugs", []):
                interaction = DrugInteraction(
                    drug_name=drug_name,
                    gene_name=gene_name,
                    impact_score=safety_score * self._get_consequence_factor(consequence),
                    interaction_type="adverse",
                    evidence_level=gene_info.get("evidence_level", "moderate")
                )
                drug_interactions.append(interaction)
                
                # Update drug profile
                self._update_drug_profile(drug_name, variant, interaction)
        
        # Calculate overall pharmacogenetics score
        overall_score = self._calculate_overall_score(
            metabolism_score, efficacy_score, safety_score
        )
        
        # Keep only significant drug interactions
        significant_interactions = []
        for interaction in drug_interactions:
            if interaction.impact_score >= self.thresholds["relevant_contribution"]:
                significant_interactions.append({
                    "drug_name": interaction.drug_name,
                    "gene_name": interaction.gene_name,
                    "score": interaction.impact_score,
                    "interaction_type": interaction.interaction_type,
                    "evidence_level": interaction.evidence_level
                })
        
        # Create pharmacogenetics domain score
        pgx_score = {
            "score": overall_score,
            "component_scores": {
                "metabolism": metabolism_score,
                "efficacy": efficacy_score,
                "safety": safety_score
            },
            "drug_interactions": significant_interactions
        }
        
        # Update variant with score
        if hasattr(variant, 'domain_scores'):
            variant.domain_scores["pharmacogenetics"] = pgx_score
        
        # Add to list of analyzed variants
        self.analyzed_variants.append(variant)
        
        return pgx_score
    
    def _update_drug_profile(self, drug_name: str, variant: Variant, 
                            interaction: DrugInteraction):
        """Update or create a drug profile with a new interaction.
        
        Args:
            drug_name: Name of the drug
            variant: The variant affecting the drug
            interaction: The interaction details
        """
        if drug_name not in self.drug_profiles:
            self.drug_profiles[drug_name] = DrugProfile(drug_name=drug_name)
            
        self.drug_profiles[drug_name].add_variant(variant, interaction)
    
    def analyze_variants(self, variants: List[Variant]) -> Dict:
        """Analyze a list of variants for pharmacogenetic effects.
        
        Args:
            variants: List of variants to analyze
            
        Returns:
            Dictionary with pharmacogenetics analysis results
        """
        logger.info(f"Analyzing {len(variants)} variants for pharmacogenetics domain")
        
        # Process each variant
        for variant in variants:
            self.analyze_variant(variant)
        
        # Generate recommendations for each drug profile
        for profile in self.drug_profiles.values():
            profile.generate_recommendations()
        
        # Prepare domain summary
        total_variants = len(self.analyzed_variants)
        relevant_variants = sum(1 for v in self.analyzed_variants 
                              if v.domain_scores.get("pharmacogenetics", {}).get("score", 0) >= 
                              self.thresholds["relevant_contribution"])
        
        high_impact_variants = sum(1 for v in self.analyzed_variants 
                                 if v.domain_scores.get("pharmacogenetics", {}).get("score", 0) >= 
                                 self.thresholds["high_impact"])
        
        # Prepare drug summaries
        drug_summaries = {}
        for drug_name, profile in self.drug_profiles.items():
            drug_summaries[drug_name] = {
                "score": profile.overall_score,
                "metabolism_score": profile.metabolism_score,
                "efficacy_score": profile.efficacy_score,
                "safety_score": profile.safety_score,
                "metabolizer_status": profile.metabolizer_status.value,
                "relevant_variants": len(profile.contributing_variants),
                "key_genes": dict(sorted(profile.gene_contributions.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True)[:5]),
                "recommendations": profile.recommendations
            }
        
        # Calculate overall impact score
        domain_scores = [v.domain_scores.get("pharmacogenetics", {}).get("score", 0) 
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
            "drugs": drug_summaries
        }
    
    def _calculate_overall_score(self, metabolism: float, efficacy: float, 
                               safety: float) -> float:
        """Calculate overall pharmacogenetics score from component scores.
        
        Args:
            metabolism: Metabolism score
            efficacy: Efficacy score
            safety: Safety score
            
        Returns:
            Overall pharmacogenetics score
        """
        # Weighted average of component scores
        weights = {
            "metabolism": 0.4,
            "efficacy": 0.3,
            "safety": 0.3
        }
        
        return (
            metabolism * weights["metabolism"] +
            efficacy * weights["efficacy"] +
            safety * weights["safety"]
        )
    
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