"""
Nutrition analyzer module for genomic data.

This module provides functionality to analyze genetic variants for nutritional implications,
including nutrient requirements, food sensitivities, and dietary recommendations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union
import uuid

from gospel.domains.nutrition.constants import (
    NUTRIENTS, 
    FOOD_SENSITIVITIES, 
    GENE_NUTRIENT_RELATIONSHIPS,
    SNP_EFFECT_MAPPINGS
)
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
    VariantEffect
)

logger = logging.getLogger(__name__)


class NutritionAnalyzer:
    """Analyzer for nutritional genomics data."""

    def __init__(self, variant_data: Dict[str, Dict], confidence_threshold: float = 0.6):
        """
        Initialize the nutrition analyzer.
        
        Args:
            variant_data: Dictionary of variant data with variant_id as key
            confidence_threshold: Minimum confidence threshold for reporting findings
        """
        self.variant_data = variant_data
        self.confidence_threshold = confidence_threshold
        self.variant_effects: Dict[str, VariantEffect] = {}
        self.nutrient_needs: Dict[str, NutrientNeed] = {}
        self.food_sensitivities: Dict[str, FoodSensitivity] = {}
        self.dietary_responses: Dict[str, DietaryResponse] = {}

    def analyze_variants(self) -> Dict[str, VariantEffect]:
        """
        Analyze variants for nutritional effects.
        
        Returns:
            Dictionary of variant effects with variant_id as key
        """
        for variant_id, variant_data in self.variant_data.items():
            # Skip variants without required data
            if not all(k in variant_data for k in ["gene", "genotype"]):
                logger.warning(f"Skipping variant {variant_id}: missing required data")
                continue
                
            gene = variant_data["gene"]
            genotype = variant_data["genotype"]
            
            # Check if this variant has known nutritional effects
            if variant_id in SNP_EFFECT_MAPPINGS:
                effect_map = SNP_EFFECT_MAPPINGS[variant_id]
                
                # Create the variant effect object
                variant_effect = VariantEffect(
                    variant_id=variant_id,
                    gene=gene,
                    genotype=genotype,
                    nutrient_effects=[],
                    food_sensitivities=[],
                    dietary_responses=[]
                )
                
                # Process nutrient effects
                if "nutrient_effects" in effect_map and genotype in effect_map["nutrient_effects"]:
                    for effect_data in effect_map["nutrient_effects"][genotype]:
                        nutrient_id = effect_data["nutrient_id"]
                        effect_type = EffectType(effect_data["effect_type"])
                        magnitude = effect_data["magnitude"]
                        confidence = effect_data["confidence"]
                        
                        nutrient_effect = NutrientEffect(
                            nutrient_id=nutrient_id,
                            effect_type=effect_type,
                            magnitude=magnitude,
                            confidence=confidence
                        )
                        variant_effect.nutrient_effects.append(nutrient_effect)
                
                # Process food sensitivities
                if "food_sensitivities" in effect_map and genotype in effect_map["food_sensitivities"]:
                    for sensitivity_data in effect_map["food_sensitivities"][genotype]:
                        food_id = sensitivity_data["food_id"]
                        severity = SensitivityLevel(sensitivity_data["severity"])
                        confidence = sensitivity_data["confidence"]
                        
                        foods_to_avoid = []
                        alternatives = []
                        if food_id in FOOD_SENSITIVITIES:
                            foods_to_avoid = FOOD_SENSITIVITIES[food_id].get("foods_to_avoid", [])
                            alternatives = FOOD_SENSITIVITIES[food_id].get("alternatives", [])
                        
                        food_sensitivity = FoodSensitivity(
                            food_id=food_id,
                            severity=severity,
                            confidence=confidence,
                            foods_to_avoid=foods_to_avoid,
                            alternatives=alternatives
                        )
                        variant_effect.food_sensitivities.append(food_sensitivity)
                
                # Process dietary responses
                if "dietary_responses" in effect_map and genotype in effect_map["dietary_responses"]:
                    for response_data in effect_map["dietary_responses"][genotype]:
                        diet_factor = response_data["diet_factor"]
                        response_type = ResponseType(response_data["response_type"])
                        magnitude = response_data["magnitude"]
                        confidence = response_data["confidence"]
                        
                        dietary_response = DietaryResponse(
                            diet_factor=diet_factor,
                            response_type=response_type,
                            magnitude=magnitude,
                            confidence=confidence
                        )
                        variant_effect.dietary_responses.append(dietary_response)
                
                # Store the variant effect if it has any nutritional implications
                if (variant_effect.nutrient_effects or 
                    variant_effect.food_sensitivities or 
                    variant_effect.dietary_responses):
                    self.variant_effects[variant_id] = variant_effect
            
            # Also check gene-nutrient relationships
            if gene in GENE_NUTRIENT_RELATIONSHIPS:
                # If the variant wasn't processed above, create a new effect object
                if variant_id not in self.variant_effects:
                    variant_effect = VariantEffect(
                        variant_id=variant_id,
                        gene=gene,
                        genotype=genotype,
                        nutrient_effects=[],
                        food_sensitivities=[],
                        dietary_responses=[]
                    )
                    self.variant_effects[variant_id] = variant_effect
                else:
                    variant_effect = self.variant_effects[variant_id]
                
                # Add general gene-nutrient relationship effects
                gene_nutrient_data = GENE_NUTRIENT_RELATIONSHIPS[gene]
                for nutrient_id in gene_nutrient_data.get("associated_nutrients", []):
                    # Skip if we already have a specific effect for this nutrient
                    if any(e.nutrient_id == nutrient_id for e in variant_effect.nutrient_effects):
                        continue
                        
                    # Add a general effect with lower confidence
                    nutrient_effect = NutrientEffect(
                        nutrient_id=nutrient_id,
                        effect_type=EffectType.NORMAL,  # Default to normal unless specified
                        magnitude=0.5,  # Medium magnitude
                        confidence=0.5  # Medium confidence
                    )
                    variant_effect.nutrient_effects.append(nutrient_effect)
        
        return self.variant_effects

    def aggregate_nutrient_needs(self) -> Dict[str, NutrientNeed]:
        """
        Aggregate nutrient effects across all variants.
        
        Returns:
            Dictionary of aggregated nutrient needs
        """
        # Ensure variants have been analyzed
        if not self.variant_effects:
            self.analyze_variants()
        
        # Collect all nutrient effects
        nutrient_effects_by_id: Dict[str, List[NutrientEffect]] = {}
        
        for variant_effect in self.variant_effects.values():
            for nutrient_effect in variant_effect.nutrient_effects:
                if nutrient_effect.nutrient_id not in nutrient_effects_by_id:
                    nutrient_effects_by_id[nutrient_effect.nutrient_id] = []
                nutrient_effects_by_id[nutrient_effect.nutrient_id].append(nutrient_effect)
        
        # Aggregate effects for each nutrient
        for nutrient_id, effects in nutrient_effects_by_id.items():
            # Skip if no effects meet confidence threshold
            confident_effects = [e for e in effects if e.confidence >= self.confidence_threshold]
            if not confident_effects:
                continue
            
            # Get nutrient info from constants
            if nutrient_id not in NUTRIENTS:
                logger.warning(f"Skipping unknown nutrient: {nutrient_id}")
                continue
                
            nutrient_info = NUTRIENTS[nutrient_id]
            
            # Calculate weighted effect
            increased_need_weight = sum(e.magnitude * e.confidence 
                                       for e in confident_effects 
                                       if e.effect_type == EffectType.INCREASED_NEED)
            decreased_need_weight = sum(e.magnitude * e.confidence 
                                       for e in confident_effects 
                                       if e.effect_type == EffectType.DECREASED_NEED)
            metabolism_issue_weight = sum(e.magnitude * e.confidence 
                                         for e in confident_effects 
                                         if e.effect_type == EffectType.METABOLISM_ISSUE)
            
            # Determine the overall adjustment factor
            net_effect = increased_need_weight - decreased_need_weight + (metabolism_issue_weight * 0.5)
            
            # Convert to adjustment factor: 1.0 = no change, >1.0 = increased need, <1.0 = decreased need
            # Cap adjustments to reasonable ranges
            if net_effect > 0:
                # Increase by up to 50%
                adjustment_factor = 1.0 + min(net_effect, 0.5)
            elif net_effect < 0:
                # Decrease by up to 25%
                adjustment_factor = max(1.0 + net_effect, 0.75)
            else:
                adjustment_factor = 1.0
            
            # Calculate average confidence
            average_confidence = sum(e.confidence for e in confident_effects) / len(confident_effects)
            
            # Get RDA from nutrient info
            rda = nutrient_info.get("rda", 0)
            units = nutrient_info.get("units", "")
            
            # Calculate adjusted intake
            adjusted_intake = rda * adjustment_factor
            
            # Create the aggregated nutrient need
            nutrient_need = NutrientNeed(
                nutrient_id=nutrient_id,
                name=nutrient_info.get("name", nutrient_id),
                adjustment_factor=adjustment_factor,
                confidence=average_confidence,
                rda=rda,
                units=units,
                adjusted_intake=adjusted_intake,
                food_sources=nutrient_info.get("food_sources", []),
                supplement_forms=nutrient_info.get("supplement_forms", []),
                description=nutrient_info.get("description", ""),
                # Prioritize nutrients with larger adjustments and higher confidence
                priority=int(10 * abs(adjustment_factor - 1.0) * average_confidence)
            )
            
            self.nutrient_needs[nutrient_id] = nutrient_need
        
        return self.nutrient_needs

    def aggregate_food_sensitivities(self) -> Dict[str, FoodSensitivity]:
        """
        Aggregate food sensitivity effects across all variants.
        
        Returns:
            Dictionary of aggregated food sensitivities
        """
        # Ensure variants have been analyzed
        if not self.variant_effects:
            self.analyze_variants()
        
        # Collect all food sensitivity effects
        sensitivities_by_food: Dict[str, List[FoodSensitivity]] = {}
        
        for variant_effect in self.variant_effects.values():
            for sensitivity in variant_effect.food_sensitivities:
                if sensitivity.food_id not in sensitivities_by_food:
                    sensitivities_by_food[sensitivity.food_id] = []
                sensitivities_by_food[sensitivity.food_id].append(sensitivity)
        
        # Aggregate effects for each food
        for food_id, sensitivities in sensitivities_by_food.items():
            # Skip if no sensitivities meet confidence threshold
            confident_sensitivities = [s for s in sensitivities if s.confidence >= self.confidence_threshold]
            if not confident_sensitivities:
                continue
            
            # Calculate the most common severity and average confidence
            severity_counts = {
                SensitivityLevel.NONE: 0,
                SensitivityLevel.MILD: 0,
                SensitivityLevel.MODERATE: 0,
                SensitivityLevel.SEVERE: 0
            }
            
            for s in confident_sensitivities:
                severity_counts[s.severity] += 1
            
            # Determine the most common severity (prioritize higher severities in case of ties)
            for severity in [SensitivityLevel.SEVERE, SensitivityLevel.MODERATE, 
                           SensitivityLevel.MILD, SensitivityLevel.NONE]:
                if severity_counts[severity] > 0:
                    most_common_severity = severity
                    break
            
            # Skip if the most common severity is NONE
            if most_common_severity == SensitivityLevel.NONE:
                continue
            
            # Calculate average confidence
            average_confidence = sum(s.confidence for s in confident_sensitivities) / len(confident_sensitivities)
            
            # Get foods to avoid and alternatives from the first sensitivity (they should be the same for all)
            foods_to_avoid = []
            alternatives = []
            if confident_sensitivities:
                foods_to_avoid = confident_sensitivities[0].foods_to_avoid
                alternatives = confident_sensitivities[0].alternatives
            
            # Create the aggregated food sensitivity
            food_sensitivity = FoodSensitivity(
                food_id=food_id,
                severity=most_common_severity,
                confidence=average_confidence,
                foods_to_avoid=foods_to_avoid,
                alternatives=alternatives
            )
            
            self.food_sensitivities[food_id] = food_sensitivity
        
        return self.food_sensitivities

    def aggregate_dietary_responses(self) -> Dict[str, DietaryResponse]:
        """
        Aggregate dietary response effects across all variants.
        
        Returns:
            Dictionary of aggregated dietary responses
        """
        # Ensure variants have been analyzed
        if not self.variant_effects:
            self.analyze_variants()
        
        # Collect all dietary response effects
        responses_by_factor: Dict[str, List[DietaryResponse]] = {}
        
        for variant_effect in self.variant_effects.values():
            for response in variant_effect.dietary_responses:
                if response.diet_factor not in responses_by_factor:
                    responses_by_factor[response.diet_factor] = []
                responses_by_factor[response.diet_factor].append(response)
        
        # Aggregate effects for each dietary factor
        for diet_factor, responses in responses_by_factor.items():
            # Skip if no responses meet confidence threshold
            confident_responses = [r for r in responses if r.confidence >= self.confidence_threshold]
            if not confident_responses:
                continue
            
            # Calculate the weighted response type
            positive_weight = sum(r.magnitude * r.confidence 
                                 for r in confident_responses 
                                 if r.response_type == ResponseType.POSITIVE)
            negative_weight = sum(r.magnitude * r.confidence 
                                 for r in confident_responses 
                                 if r.response_type == ResponseType.NEGATIVE)
            
            # Determine the overall response type
            if positive_weight > negative_weight:
                response_type = ResponseType.POSITIVE
                magnitude = positive_weight / sum(r.confidence for r in confident_responses 
                                                if r.response_type == ResponseType.POSITIVE)
            elif negative_weight > positive_weight:
                response_type = ResponseType.NEGATIVE
                magnitude = negative_weight / sum(r.confidence for r in confident_responses 
                                                if r.response_type == ResponseType.NEGATIVE)
            else:
                response_type = ResponseType.NEUTRAL
                magnitude = 0.5
            
            # Calculate average confidence
            average_confidence = sum(r.confidence for r in confident_responses) / len(confident_responses)
            
            # Create the aggregated dietary response
            dietary_response = DietaryResponse(
                diet_factor=diet_factor,
                response_type=response_type,
                magnitude=magnitude,
                confidence=average_confidence
            )
            
            self.dietary_responses[diet_factor] = dietary_response
        
        return self.dietary_responses

    def generate_nutrition_plan(self) -> NutritionPlan:
        """
        Generate a personalized nutrition plan based on analysis results.
        
        Returns:
            NutritionPlan object with dietary recommendations
        """
        # Ensure all aggregation has been done
        if not self.variant_effects:
            self.analyze_variants()
        if not self.nutrient_needs:
            self.aggregate_nutrient_needs()
        if not self.food_sensitivities:
            self.aggregate_food_sensitivities()
        if not self.dietary_responses:
            self.aggregate_dietary_responses()
        
        # Determine the best dietary patterns based on responses
        # Start with basic patterns
        patterns = [
            {"name": "Mediterranean", "score": 0.5, "confidence": 0.5},
            {"name": "Low-Carb", "score": 0.5, "confidence": 0.5},
            {"name": "Plant-Based", "score": 0.5, "confidence": 0.5},
            {"name": "Paleo", "score": 0.5, "confidence": 0.5}
        ]
        
        # Adjust pattern scores based on dietary responses
        for diet_factor, response in self.dietary_responses.items():
            if diet_factor == "carbohydrates" and response.response_type == ResponseType.NEGATIVE:
                patterns[1]["score"] += response.magnitude * response.confidence
                patterns[1]["confidence"] = max(patterns[1]["confidence"], response.confidence)
            
            if diet_factor == "plant_foods" and response.response_type == ResponseType.POSITIVE:
                patterns[2]["score"] += response.magnitude * response.confidence
                patterns[2]["confidence"] = max(patterns[2]["confidence"], response.confidence)
            
            if diet_factor == "processed_foods" and response.response_type == ResponseType.NEGATIVE:
                patterns[3]["score"] += response.magnitude * response.confidence
                patterns[3]["confidence"] = max(patterns[3]["confidence"], response.confidence)
            
            if diet_factor == "healthy_fats" and response.response_type == ResponseType.POSITIVE:
                patterns[0]["score"] += response.magnitude * response.confidence
                patterns[0]["confidence"] = max(patterns[0]["confidence"], response.confidence)
        
        # Sort patterns by score
        sorted_patterns = sorted(patterns, key=lambda p: p["score"], reverse=True)
        
        # Convert to DietaryPattern objects
        dietary_patterns = []
        for pattern in sorted_patterns:
            pattern_info = self._get_dietary_pattern_info(pattern["name"])
            dietary_pattern = DietaryPattern(
                name=pattern["name"],
                score=pattern["score"],
                confidence=pattern["confidence"],
                description=pattern_info["description"],
                key_components=pattern_info["key_components"],
                meal_examples=pattern_info["meal_examples"]
            )
            dietary_patterns.append(dietary_pattern)
        
        # Get the primary pattern (highest score) and alternatives
        primary_pattern = dietary_patterns[0]
        alternative_patterns = dietary_patterns[1:3]  # Top 2 alternatives
        
        # Get top nutrient needs (highest priority)
        sorted_nutrient_needs = sorted(
            self.nutrient_needs.values(), 
            key=lambda n: n.priority,
            reverse=True
        )
        top_nutrient_needs = sorted_nutrient_needs[:5]  # Top 5 nutrient needs
        
        # Get significant food sensitivities (moderate or severe)
        significant_sensitivities = [
            s for s in self.food_sensitivities.values()
            if s.severity in (SensitivityLevel.MODERATE, SensitivityLevel.SEVERE)
        ]
        
        # Generate supplement recommendations
        supplement_recommendations = []
        for need in top_nutrient_needs:
            if need.adjustment_factor > 1.2 and need.confidence > 0.7:
                forms = ", ".join(need.supplement_forms[:2]) if need.supplement_forms else "standard form"
                supplement_recommendations.append({
                    "nutrient": need.name,
                    "recommendation": f"Consider supplementing with {need.name} ({forms})",
                    "dosage": f"{int(need.adjusted_intake)} {need.units} daily",
                    "priority": "High" if need.priority > 7 else "Medium"
                })
        
        # Generate meal plan suggestions
        meal_plan_suggestions = []
        if primary_pattern.name == "Mediterranean":
            meal_plan_suggestions = [
                {"meal": "Breakfast", "suggestion": "Greek yogurt with honey, nuts, and fresh fruit"},
                {"meal": "Lunch", "suggestion": "Salad with olive oil, vegetables, and grilled fish"},
                {"meal": "Dinner", "suggestion": "Whole grain pasta with tomato sauce, vegetables, and a small portion of lean protein"},
                {"meal": "Snacks", "suggestion": "Nuts, olives, fresh fruit, or whole grain bread with hummus"}
            ]
        elif primary_pattern.name == "Low-Carb":
            meal_plan_suggestions = [
                {"meal": "Breakfast", "suggestion": "Eggs with avocado and vegetables"},
                {"meal": "Lunch", "suggestion": "Salad with protein (chicken, fish, or tofu) and olive oil dressing"},
                {"meal": "Dinner", "suggestion": "Grilled protein with non-starchy vegetables"},
                {"meal": "Snacks", "suggestion": "Nuts, cheese, or vegetable sticks with dip"}
            ]
        elif primary_pattern.name == "Plant-Based":
            meal_plan_suggestions = [
                {"meal": "Breakfast", "suggestion": "Oatmeal with fruits, nuts, and plant-based milk"},
                {"meal": "Lunch", "suggestion": "Grain bowl with legumes, vegetables, and tahini dressing"},
                {"meal": "Dinner", "suggestion": "Vegetable stir-fry with tofu and whole grains"},
                {"meal": "Snacks", "suggestion": "Fruit, nuts, or vegetable sticks with hummus"}
            ]
        elif primary_pattern.name == "Paleo":
            meal_plan_suggestions = [
                {"meal": "Breakfast", "suggestion": "Eggs with vegetables and avocado"},
                {"meal": "Lunch", "suggestion": "Salad with grilled meat and olive oil dressing"},
                {"meal": "Dinner", "suggestion": "Grilled protein with roasted vegetables"},
                {"meal": "Snacks", "suggestion": "Nuts, fruit, or homemade beef jerky"}
            ]
        
        # Adjust meal suggestions based on sensitivities
        for sensitivity in significant_sensitivities:
            if sensitivity.food_id in FOOD_SENSITIVITIES:
                sensitivity_info = FOOD_SENSITIVITIES[sensitivity.food_id]
                meal_plan_suggestions.append({
                    "meal": "Sensitivity Adjustment",
                    "suggestion": f"Avoid {sensitivity_info.get('name', sensitivity.food_id)} products. "
                                  f"Consider alternatives like {', '.join(sensitivity.alternatives[:3])}"
                })
        
        # Create and return the nutrition plan
        nutrition_plan = NutritionPlan(
            primary_diet_pattern=primary_pattern,
            alternative_diet_patterns=alternative_patterns,
            key_nutrient_needs=top_nutrient_needs,
            food_sensitivities=significant_sensitivities,
            supplement_recommendations=supplement_recommendations,
            meal_plan_suggestions=meal_plan_suggestions
        )
        
        return nutrition_plan

    def _get_dietary_pattern_info(self, pattern_name: str) -> Dict[str, List[str]]:
        """
        Get information about a dietary pattern.
        
        Args:
            pattern_name: Name of the dietary pattern
            
        Returns:
            Dictionary with pattern description, key components, and meal examples
        """
        patterns = {
            "Mediterranean": {
                "description": "Emphasizes plant foods, healthy fats (especially olive oil), and moderate consumption of fish, poultry, and dairy. Limited red meat and processed foods.",
                "key_components": [
                    "Abundant plant foods (fruits, vegetables, legumes, nuts, seeds, whole grains)",
                    "Olive oil as the primary fat source",
                    "Moderate consumption of fish and seafood",
                    "Limited intake of dairy, poultry, and eggs",
                    "Minimal red meat consumption",
                    "Moderate wine consumption (optional)"
                ],
                "meal_examples": [
                    "Greek salad with olive oil, feta, vegetables, and olives",
                    "Whole grain pasta with tomato sauce, vegetables, and a small portion of fish",
                    "Hummus with vegetable sticks and whole grain pita",
                    "Grilled fish with roasted vegetables and olive oil"
                ]
            },
            "Low-Carb": {
                "description": "Reduces carbohydrate intake while increasing protein and fat. May improve metabolic health and weight management for those with carbohydrate metabolism issues.",
                "key_components": [
                    "Limited intake of carbohydrates (typically 50-150g per day)",
                    "Increased consumption of protein",
                    "Higher intake of healthy fats",
                    "Focus on non-starchy vegetables",
                    "Avoidance of refined carbohydrates and sugars",
                    "Moderate fruit consumption (primarily low-glycemic fruits)"
                ],
                "meal_examples": [
                    "Eggs with avocado and spinach",
                    "Salad with grilled chicken, olive oil, and seeds",
                    "Grilled salmon with non-starchy vegetables",
                    "Greek yogurt with berries and nuts"
                ]
            },
            "Plant-Based": {
                "description": "Focuses on foods derived from plants with minimal or no animal products. Rich in phytonutrients, fiber, and antioxidants.",
                "key_components": [
                    "Abundant fruits and vegetables",
                    "Whole grains and legumes",
                    "Nuts and seeds",
                    "Plant oils (olive, flax, etc.)",
                    "Limited or no animal products",
                    "Emphasis on whole, unprocessed foods"
                ],
                "meal_examples": [
                    "Oatmeal with fruit, flaxseeds, and plant-based milk",
                    "Lentil soup with whole grain bread",
                    "Buddha bowl with quinoa, roasted vegetables, and tahini dressing",
                    "Smoothie with leafy greens, fruit, and plant protein"
                ]
            },
            "Paleo": {
                "description": "Based on foods presumed to have been available to paleolithic humans. Emphasizes whole foods and excludes processed foods, grains, legumes, and dairy.",
                "key_components": [
                    "Lean meats, fish, and seafood",
                    "Fruits and vegetables",
                    "Nuts and seeds",
                    "Healthy oils (olive, walnut, flaxseed, etc.)",
                    "Exclusion of grains, legumes, dairy, refined sugar, and processed foods"
                ],
                "meal_examples": [
                    "Scrambled eggs with vegetables and avocado",
                    "Grilled chicken salad with olive oil and vegetables",
                    "Baked salmon with roasted sweet potatoes and broccoli",
                    "Apple slices with almond butter"
                ]
            }
        }
        
        return patterns.get(pattern_name, {
            "description": f"Generic {pattern_name} diet pattern",
            "key_components": [f"{pattern_name} components"],
            "meal_examples": [f"{pattern_name} meal example"]
        })

    def generate_report(self, individual_id: str) -> NutritionReport:
        """
        Generate a comprehensive nutrition report.
        
        Args:
            individual_id: ID of the individual for whom to generate the report
            
        Returns:
            NutritionReport object with all analysis results
        """
        # Ensure all analysis has been completed
        if not self.variant_effects:
            self.analyze_variants()
        if not self.nutrient_needs:
            self.aggregate_nutrient_needs()
        if not self.food_sensitivities:
            self.aggregate_food_sensitivities()
        if not self.dietary_responses:
            self.aggregate_dietary_responses()
        
        # Generate nutrition plan if not already done
        nutrition_plan = self.generate_nutrition_plan()
        
        # Get significant variants (those with effects that meet the confidence threshold)
        significant_variants = []
        for variant in self.variant_effects.values():
            has_significant_effects = any(
                effect.confidence >= self.confidence_threshold
                for effect in (
                    variant.nutrient_effects +
                    variant.food_sensitivities +
                    variant.dietary_responses
                )
            )
            if has_significant_effects:
                significant_variants.append(variant)
        
        # Create and return the report
        report = NutritionReport(
            report_id=str(uuid.uuid4()),
            date_generated=datetime.now(),
            individual_id=individual_id,
            total_variants_analyzed=len(self.variant_data),
            significant_variants=significant_variants,
            nutrient_needs=self.nutrient_needs,
            food_sensitivities=self.food_sensitivities,
            dietary_responses=self.dietary_responses,
            nutrition_plan=nutrition_plan
        )
        
        return report 