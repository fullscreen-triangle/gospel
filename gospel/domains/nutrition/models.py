"""
Data models for the nutrition domain.

This module defines the data structures used for nutrition analysis,
including nutrient effects, food sensitivities, dietary responses,
and the final nutrition report format.
"""

import logging
import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from datetime import datetime

from gospel.core.variant import Variant
from gospel.core.annotation import Annotation
from gospel.core.scoring import NutritionScorer
from gospel.domains.nutrition.constants import (
    NUTRIENTS, 
    NUTRIENT_METABOLISM_GENES,
    FOOD_SENSITIVITY_GENES, 
    DIETARY_RESPONSE_GENES,
    NUTRITION_SNPs
)

logger = logging.getLogger(__name__)


class EffectType(str, enum.Enum):
    """Types of nutrient effects determined by genetic variants."""
    NORMAL = "normal"
    INCREASED_NEED = "increased_need"
    DECREASED_NEED = "decreased_need"
    METABOLISM_ISSUE = "metabolism_issue"


class SensitivityLevel(str, enum.Enum):
    """Severity levels for food sensitivities."""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ResponseType(str, enum.Enum):
    """Types of responses to dietary factors."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ConfidenceLevel(float, enum.Enum):
    """Confidence levels for findings."""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class Nutrient:
    """Information about a specific nutrient."""
    id: str
    name: str
    category: str
    rda: float
    units: str
    description: str
    food_sources: List[str] = field(default_factory=list)


@dataclass
class NutrientEffect:
    """Effect of a genetic variant on a specific nutrient."""
    nutrient_id: str
    effect_type: EffectType
    magnitude: float  # 0.0 to 1.0, with 1.0 being the strongest effect
    confidence: float  # 0.0 to 1.0, with 1.0 being the highest confidence
    
    @property
    def is_significant(self) -> bool:
        """Return whether the effect is significant enough to report."""
        return self.magnitude >= 0.3 and self.confidence >= 0.5


@dataclass
class NutrientNeed:
    """Aggregated nutrient need after analyzing multiple variants."""
    nutrient_id: str
    name: str
    adjustment_factor: float  # Relative to RDA: 1.0 = no change, >1.0 = increased, <1.0 = decreased
    confidence: float
    rda: float
    units: str
    adjusted_intake: float
    food_sources: List[str] = field(default_factory=list)
    supplement_forms: List[str] = field(default_factory=list)
    description: str = ""
    priority: int = 0  # Higher number = higher priority


@dataclass
class FoodSensitivity:
    """Food sensitivity identified from genetic analysis."""
    food_id: str
    severity: SensitivityLevel
    confidence: float
    foods_to_avoid: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)


@dataclass
class FoodCategory:
    """A category of foods with associated information."""
    id: str
    name: str
    foods: List[str]
    foods_to_avoid: List[str]
    alternatives: List[str]


@dataclass
class DietaryResponse:
    """Response to a specific dietary pattern based on genetics."""
    diet_factor: str
    response_type: ResponseType
    magnitude: float
    confidence: float


@dataclass
class VariantEffect:
    """Aggregated effects of a specific genetic variant."""
    variant_id: str
    gene: str
    genotype: str
    nutrient_effects: List[NutrientEffect] = field(default_factory=list)
    food_sensitivities: List[FoodSensitivity] = field(default_factory=list)
    dietary_responses: List[DietaryResponse] = field(default_factory=list)


@dataclass
class DietaryPattern:
    """Recommended dietary pattern based on genetic analysis."""
    name: str
    score: float  # 0.0 to 1.0, with 1.0 being the strongest match
    confidence: float
    description: str
    key_components: List[str] = field(default_factory=list)
    meal_examples: List[str] = field(default_factory=list)


@dataclass
class NutritionPlan:
    """Personalized nutrition plan with recommendations."""
    primary_diet_pattern: DietaryPattern
    alternative_diet_patterns: List[DietaryPattern] = field(default_factory=list)
    key_nutrient_needs: List[NutrientNeed] = field(default_factory=list)
    food_sensitivities: List[FoodSensitivity] = field(default_factory=list)
    supplement_recommendations: List[Dict[str, str]] = field(default_factory=list)
    meal_plan_suggestions: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class NutritionReport:
    """Complete nutrition analysis report."""
    report_id: str
    date_generated: datetime
    individual_id: str
    total_variants_analyzed: int
    significant_variants: List[VariantEffect] = field(default_factory=list)
    nutrient_needs: Dict[str, NutrientNeed] = field(default_factory=dict)
    food_sensitivities: Dict[str, FoodSensitivity] = field(default_factory=dict)
    dietary_responses: Dict[str, DietaryResponse] = field(default_factory=dict)
    nutrition_plan: Optional[NutritionPlan] = None
    
    def to_dict(self) -> Dict:
        """Convert the report to a dictionary for serialization."""
        return {
            "report_id": self.report_id,
            "date_generated": self.date_generated.isoformat(),
            "individual_id": self.individual_id,
            "total_variants_analyzed": self.total_variants_analyzed,
            "significant_variants": [
                {
                    "variant_id": v.variant_id,
                    "gene": v.gene,
                    "genotype": v.genotype,
                    "nutrient_effects": [
                        {
                            "nutrient_id": e.nutrient_id,
                            "effect_type": e.effect_type.value,
                            "magnitude": e.magnitude,
                            "confidence": e.confidence
                        } for e in v.nutrient_effects
                    ],
                    "food_sensitivities": [
                        {
                            "food_id": s.food_id,
                            "severity": s.severity.value,
                            "confidence": s.confidence
                        } for s in v.food_sensitivities
                    ],
                    "dietary_responses": [
                        {
                            "diet_factor": r.diet_factor,
                            "response_type": r.response_type.value,
                            "magnitude": r.magnitude,
                            "confidence": r.confidence
                        } for r in v.dietary_responses
                    ]
                } for v in self.significant_variants
            ],
            "nutrient_needs": {
                nutrient_id: {
                    "name": need.name,
                    "adjustment_factor": need.adjustment_factor,
                    "confidence": need.confidence,
                    "rda": need.rda,
                    "units": need.units,
                    "adjusted_intake": need.adjusted_intake,
                    "food_sources": need.food_sources,
                    "supplement_forms": need.supplement_forms,
                    "description": need.description,
                    "priority": need.priority
                } for nutrient_id, need in self.nutrient_needs.items()
            },
            "food_sensitivities": {
                food_id: {
                    "severity": sensitivity.severity.value,
                    "confidence": sensitivity.confidence,
                    "foods_to_avoid": sensitivity.foods_to_avoid,
                    "alternatives": sensitivity.alternatives
                } for food_id, sensitivity in self.food_sensitivities.items()
            },
            "dietary_responses": {
                diet_factor: {
                    "response_type": response.response_type.value,
                    "magnitude": response.magnitude,
                    "confidence": response.confidence
                } for diet_factor, response in self.dietary_responses.items()
            },
            "nutrition_plan": None if self.nutrition_plan is None else {
                "primary_diet_pattern": {
                    "name": self.nutrition_plan.primary_diet_pattern.name,
                    "score": self.nutrition_plan.primary_diet_pattern.score,
                    "confidence": self.nutrition_plan.primary_diet_pattern.confidence,
                    "description": self.nutrition_plan.primary_diet_pattern.description,
                    "key_components": self.nutrition_plan.primary_diet_pattern.key_components,
                    "meal_examples": self.nutrition_plan.primary_diet_pattern.meal_examples
                },
                "alternative_diet_patterns": [
                    {
                        "name": pattern.name,
                        "score": pattern.score,
                        "confidence": pattern.confidence,
                        "description": pattern.description,
                        "key_components": pattern.key_components,
                        "meal_examples": pattern.meal_examples
                    } for pattern in self.nutrition_plan.alternative_diet_patterns
                ],
                "key_nutrient_needs": [
                    {
                        "nutrient_id": need.nutrient_id,
                        "name": need.name,
                        "adjustment_factor": need.adjustment_factor,
                        "confidence": need.confidence,
                        "rda": need.rda,
                        "units": need.units,
                        "adjusted_intake": need.adjusted_intake,
                        "food_sources": need.food_sources,
                        "description": need.description,
                        "priority": need.priority
                    } for need in self.nutrition_plan.key_nutrient_needs
                ],
                "food_sensitivities": [
                    {
                        "food_id": sensitivity.food_id,
                        "severity": sensitivity.severity.value,
                        "confidence": sensitivity.confidence,
                        "foods_to_avoid": sensitivity.foods_to_avoid,
                        "alternatives": sensitivity.alternatives
                    } for sensitivity in self.nutrition_plan.food_sensitivities
                ],
                "supplement_recommendations": self.nutrition_plan.supplement_recommendations,
                "meal_plan_suggestions": self.nutrition_plan.meal_plan_suggestions
            }
        }


class NutritionDomain:
    """Domain implementation for nutrition-related genomic analysis."""
    
    def __init__(self):
        """Initialize the nutrition domain."""
        self.nutrient_profiles: Dict[str, NutrientProfile] = {}
        self.analyzed_variants: Set[str] = set()
        self.scorer = NutritionScorer()
        
    def analyze_variant(self, variant: Variant, annotation: Annotation) -> List[GeneticEffect]:
        """Analyze a variant for nutrition-related effects."""
        effects = []
        variant_id = variant.variant_id
        
        if variant_id in self.analyzed_variants:
            return effects
            
        self.analyzed_variants.add(variant_id)
        
        # Check if variant is a known nutrition SNP
        if variant_id in NUTRITION_SNPs:
            snp_data = NUTRITION_SNPs[variant_id]
            genotype = variant.genotype
            
            if genotype in snp_data["significance"]:
                significance = snp_data["significance"][genotype]
                
                for nutrient in snp_data["associated_nutrients"]:
                    # Create effect based on significance
                    effect = GeneticEffect(
                        nutrient=nutrient,
                        score=significance["score"],
                        effect_type=NutrientEffect[significance["impact"].upper()] 
                            if significance["impact"].upper() in NutrientEffect.__members__ 
                            else NutrientEffect.UNKNOWN,
                        confidence=0.8,  # High confidence for known SNPs
                        source_variants=[variant_id],
                        source_genes=[snp_data["gene"]]
                    )
                    effects.append(effect)
                    self._update_nutrient_profile(nutrient, variant_id, effect)
        
        # Check if variant is in gene with known nutrient effects
        gene_symbol = annotation.gene_symbol
        if gene_symbol in NUTRIENT_METABOLISM_GENES:
            gene_data = NUTRIENT_METABOLISM_GENES[gene_symbol]
            
            # Calculate impact based on variant consequence
            impact_modifier = 1.0
            if annotation.impact == "HIGH":
                impact_modifier = 1.0
            elif annotation.impact == "MODERATE":
                impact_modifier = 0.7
            elif annotation.impact == "LOW":
                impact_modifier = 0.3
            else:
                impact_modifier = 0.1
                
            for effect_data in gene_data["nutrient_effects"]:
                effect = GeneticEffect(
                    nutrient=effect_data["nutrient"],
                    score=effect_data["score"] * impact_modifier,
                    effect_type=NutrientEffect[effect_data["effect_type"].upper()] 
                        if effect_data["effect_type"].upper() in NutrientEffect.__members__ 
                        else NutrientEffect.UNKNOWN,
                    confidence=0.7 if effect_data["evidence_level"] == "high" else 0.5,
                    source_variants=[variant_id],
                    source_genes=[gene_symbol]
                )
                effects.append(effect)
                self._update_nutrient_profile(effect_data["nutrient"], variant_id, effect)
        
        # Check for food sensitivity genes
        if gene_symbol in FOOD_SENSITIVITY_GENES:
            gene_data = FOOD_SENSITIVITY_GENES[gene_symbol]
            
            # Similar impact calculation as above
            impact_modifier = 1.0 if annotation.impact == "HIGH" else (
                0.7 if annotation.impact == "MODERATE" else (
                0.3 if annotation.impact == "LOW" else 0.1
            ))
            
            for sensitivity in gene_data["sensitivities"]:
                if "nutrient" in sensitivity:
                    effect = GeneticEffect(
                        nutrient=sensitivity["nutrient"],
                        score=sensitivity["score"] * impact_modifier,
                        effect_type=NutrientEffect.SENSITIVITY,
                        confidence=0.7 if sensitivity["evidence_level"] == "high" else 0.5,
                        source_variants=[variant_id],
                        source_genes=[gene_symbol]
                    )
                    effects.append(effect)
                    self._update_nutrient_profile(sensitivity["nutrient"], variant_id, effect)
        
        # Check for dietary response genes
        if gene_symbol in DIETARY_RESPONSE_GENES:
            gene_data = DIETARY_RESPONSE_GENES[gene_symbol]
            
            impact_modifier = 1.0 if annotation.impact == "HIGH" else (
                0.7 if annotation.impact == "MODERATE" else (
                0.3 if annotation.impact == "LOW" else 0.1
            ))
            
            for response in gene_data["dietary_responses"]:
                if "nutrient" in response and "effect_type" in response:
                    effect = GeneticEffect(
                        nutrient=response["nutrient"],
                        score=response["score"] * impact_modifier,
                        effect_type=NutrientEffect[response["effect_type"].upper()] 
                            if response["effect_type"].upper() in NutrientEffect.__members__ 
                            else NutrientEffect.UNKNOWN,
                        confidence=0.7 if response["evidence_level"] == "high" else 0.5,
                        source_variants=[variant_id],
                        source_genes=[gene_symbol]
                    )
                    effects.append(effect)
                    self._update_nutrient_profile(response["nutrient"], variant_id, effect)
        
        return effects
    
    def _update_nutrient_profile(self, nutrient: str, variant_id: str, effect: GeneticEffect) -> None:
        """Update the nutrient profile with a new effect."""
        if nutrient not in self.nutrient_profiles:
            self.nutrient_profiles[nutrient] = NutrientProfile()
        
        self.nutrient_profiles[nutrient].add_nutrient(NutrientRecommendation(
            nutrient_id=nutrient,
            name=NUTRIENTS.get(nutrient, {}).get("name", nutrient.title()),
            category="",
            status="",
            score=effect.score,
            rda=0.0,
            units="",
            recommended_intake=0.0,
            recommended_sources=[],
            relevant_variants=[],
            relevant_genes=[],
            explanation=""
        ))
    
    def get_nutrient_profile(self, nutrient: str) -> Optional[Dict]:
        """Get the profile for a specific nutrient."""
        if nutrient in self.nutrient_profiles:
            return self.nutrient_profiles[nutrient].to_dict()
        return None
    
    def get_all_nutrient_profiles(self) -> Dict[str, Dict]:
        """Get all nutrient profiles."""
        return {
            nutrient: profile.to_dict()
            for nutrient, profile in self.nutrient_profiles.items()
        }
    
    def get_nutrient_summary(self) -> Dict:
        """Get a summary of nutrition analysis across all nutrients."""
        if not self.nutrient_profiles:
            return {
                "overall_score": 0.0,
                "nutrients_analyzed": 0,
                "variants_analyzed": len(self.analyzed_variants),
                "top_nutrients": [],
                "summary": "No nutrition-related genetic variants found."
            }
        
        # Calculate overall nutrition score
        nutrient_scores = [
            profile.aggregate_score * profile.aggregate_score  # Square to emphasize strong effects
            for profile in self.nutrient_profiles.values()
            if profile.aggregate_score > 0.3  # Only consider moderate-to-strong effects
        ]
        
        overall_score = 0.0
        if nutrient_scores:
            # Root mean square to combine scores
            overall_score = (sum(nutrient_scores) / len(nutrient_scores)) ** 0.5
        
        # Get top nutrients by score
        top_nutrients = sorted(
            [
                {
                    "nutrient": profile.nutrient,
                    "name": profile.nutrient_name,
                    "score": profile.aggregate_score,
                    "effect_type": profile.dominant_effect_type.value
                }
                for profile in self.nutrient_profiles.values()
                if profile.aggregate_score > 0.3
            ],
            key=lambda x: x["score"],
            reverse=True
        )[:5]  # Top 5 nutrients
        
        # Generate summary text
        if not top_nutrients:
            summary = "No significant nutrition-related genetic effects found."
        else:
            top_needs = [n for n in top_nutrients if n["effect_type"] == NutrientEffect.INCREASED_NEED.value]
            top_sensitivities = [n for n in top_nutrients if n["effect_type"] == NutrientEffect.SENSITIVITY.value]
            
            summary_parts = []
            
            if top_needs:
                needs_text = ", ".join([n["name"] for n in top_needs[:3]])
                summary_parts.append(f"Increased need for {needs_text}")
                
            if top_sensitivities:
                sens_text = ", ".join([n["name"] for n in top_sensitivities[:3]])
                summary_parts.append(f"Sensitivity to {sens_text}")
                
            if summary_parts:
                summary = ". ".join(summary_parts) + "."
            else:
                other_effects = [n["name"] for n in top_nutrients[:3]]
                summary = f"Genetic effects on {', '.join(other_effects)}."
        
        return {
            "overall_score": overall_score,
            "nutrients_analyzed": len(self.nutrient_profiles),
            "variants_analyzed": len(self.analyzed_variants),
            "top_nutrients": top_nutrients,
            "summary": summary
        }
    
    def get_dietary_recommendations(self) -> Dict:
        """Generate dietary recommendations based on genetic analysis."""
        if not self.nutrient_profiles:
            return {
                "recommendations": [],
                "summary": "No specific dietary recommendations based on genetics."
            }
        
        recommendations = []
        
        # Process increased needs
        increased_needs = [
            profile for profile in self.nutrient_profiles.values()
            if profile.dominant_effect_type == NutrientEffect.INCREASED_NEED
            and profile.aggregate_score > 0.4
        ]
        
        for profile in increased_needs:
            nutrient_data = NUTRIENTS.get(profile.nutrient, {})
            food_sources = nutrient_data.get("food_sources", [])
            
            if food_sources:
                food_text = ", ".join(food_sources[:3])
                recommendation = {
                    "type": "increased_intake",
                    "nutrient": profile.nutrient,
                    "nutrient_name": profile.nutrient_name,
                    "score": profile.aggregate_score,
                    "recommendation": f"Increase intake of {profile.nutrient_name} through {food_text}.",
                    "foods": food_sources
                }
                recommendations.append(recommendation)
        
        # Process sensitivities
        sensitivities = [
            profile for profile in self.nutrient_profiles.values()
            if profile.dominant_effect_type == NutrientEffect.SENSITIVITY
            and profile.aggregate_score > 0.4
        ]
        
        for profile in sensitivities:
            recommendation = {
                "type": "sensitivity",
                "nutrient": profile.nutrient,
                "nutrient_name": profile.nutrient_name,
                "score": profile.aggregate_score,
                "recommendation": f"Monitor response to {profile.nutrient_name} and consider limiting if symptoms occur."
            }
            recommendations.append(recommendation)
        
        # Process decreased needs
        decreased_needs = [
            profile for profile in self.nutrient_profiles.values()
            if profile.dominant_effect_type == NutrientEffect.DECREASED_NEED
            and profile.aggregate_score > 0.4
        ]
        
        for profile in decreased_needs:
            recommendation = {
                "type": "decreased_intake",
                "nutrient": profile.nutrient,
                "nutrient_name": profile.nutrient_name,
                "score": profile.aggregate_score,
                "recommendation": f"Standard or slightly reduced {profile.nutrient_name} intake recommended."
            }
            recommendations.append(recommendation)
        
        # Process metabolism issues
        metabolism_issues = [
            profile for profile in self.nutrient_profiles.values()
            if profile.dominant_effect_type == NutrientEffect.METABOLISM_ISSUE
            and profile.aggregate_score > 0.4
        ]
        
        for profile in metabolism_issues:
            recommendation = {
                "type": "metabolism_issue",
                "nutrient": profile.nutrient,
                "nutrient_name": profile.nutrient_name,
                "score": profile.aggregate_score,
                "recommendation": f"Consider specialized testing for {profile.nutrient_name} metabolism."
            }
            recommendations.append(recommendation)
        
        # Sort recommendations by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        # Generate summary
        if not recommendations:
            summary = "No specific dietary recommendations based on genetics."
        else:
            types_count = {}
            for rec in recommendations:
                types_count[rec["type"]] = types_count.get(rec["type"], 0) + 1
                
            summary_parts = []
            if types_count.get("increased_intake", 0) > 0:
                summary_parts.append(f"{types_count['increased_intake']} nutrients with increased needs")
                
            if types_count.get("sensitivity", 0) > 0:
                summary_parts.append(f"{types_count['sensitivity']} potential sensitivities")
                
            if summary_parts:
                summary = "Genetic analysis indicates " + ", ".join(summary_parts) + "."
            else:
                summary = "Dietary recommendations based on genetic analysis."
        
        return {
            "recommendations": recommendations[:10],  # Limit to top 10
            "summary": summary
        } 