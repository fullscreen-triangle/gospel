"""
Fuzzy Logic System for Genomic Uncertainty Quantification

This module implements fuzzy membership functions and inference systems
for handling the continuous uncertainty inherent in genomic analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from scipy import integrate
from scipy.stats import norm, beta
from sklearn.preprocessing import MinMaxScaler


@dataclass
class FuzzySet:
    """Represents a fuzzy set with membership function"""
    name: str
    membership_function: 'MembershipFunction'
    universe: np.ndarray
    
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute membership degree(s) for input value(s)"""
        return self.membership_function.membership(x)


class MembershipFunction(ABC):
    """Abstract base class for membership functions"""
    
    @abstractmethod
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute membership degree for input value(s)"""
        pass


class TrapezoidalMF(MembershipFunction):
    """Trapezoidal membership function"""
    
    def __init__(self, a: float, b: float, c: float, d: float):
        """
        Initialize trapezoidal membership function.
        
        Args:
            a, b, c, d: Trapezoidal parameters where a ≤ b ≤ c ≤ d
        """
        if not (a <= b <= c <= d):
            raise ValueError("Parameters must satisfy a ≤ b ≤ c ≤ d")
        self.a, self.b, self.c, self.d = a, b, c, d
    
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute trapezoidal membership"""
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        
        # Left slope
        mask1 = (x >= self.a) & (x < self.b)
        if self.b != self.a:
            result[mask1] = (x[mask1] - self.a) / (self.b - self.a)
        
        # Flat top
        mask2 = (x >= self.b) & (x <= self.c)
        result[mask2] = 1.0
        
        # Right slope
        mask3 = (x > self.c) & (x <= self.d)
        if self.d != self.c:
            result[mask3] = (self.d - x[mask3]) / (self.d - self.c)
        
        return result.item() if result.shape == () else result


class GaussianMF(MembershipFunction):
    """Gaussian membership function"""
    
    def __init__(self, center: float, sigma: float):
        """
        Initialize Gaussian membership function.
        
        Args:
            center: Center of the Gaussian
            sigma: Standard deviation
        """
        self.center = center
        self.sigma = sigma
    
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute Gaussian membership"""
        return np.exp(-0.5 * ((x - self.center) / self.sigma) ** 2)


class SigmoidMF(MembershipFunction):
    """Sigmoid membership function"""
    
    def __init__(self, center: float, slope: float):
        """
        Initialize sigmoid membership function.
        
        Args:
            center: Center point of sigmoid
            slope: Slope parameter (positive for increasing, negative for decreasing)
        """
        self.center = center
        self.slope = slope
    
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute sigmoid membership"""
        return 1.0 / (1.0 + np.exp(-self.slope * (x - self.center)))


class ExponentialMF(MembershipFunction):
    """Exponential membership function"""
    
    def __init__(self, center: float, rate: float):
        """
        Initialize exponential membership function.
        
        Args:
            center: Center point
            rate: Decay rate
        """
        self.center = center
        self.rate = rate
    
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute exponential membership"""
        return np.exp(-self.rate * np.abs(x - self.center))


@dataclass
class FuzzyRule:
    """Fuzzy inference rule"""
    antecedent: Dict[str, str]  # Variable -> fuzzy set name
    consequent: Dict[str, str]  # Variable -> fuzzy set name
    weight: float = 1.0


class GenomicFuzzySystem:
    """
    Fuzzy logic system for genomic uncertainty quantification.
    
    This system implements Mamdani-type fuzzy inference with genomic-specific
    membership functions for variant pathogenicity, expression significance,
    conservation scores, and other genomic features.
    """
    
    def __init__(self):
        """Initialize the genomic fuzzy system"""
        self.input_variables = {}
        self.output_variables = {}
        self.rules = []
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)
        
        # Initialize genomic-specific fuzzy sets
        self._initialize_genomic_fuzzy_sets()
    
    def _initialize_genomic_fuzzy_sets(self):
        """Initialize fuzzy sets for genomic features"""
        
        # Variant pathogenicity (CADD scores)
        self.input_variables['pathogenicity'] = {
            'very_low': FuzzySet('very_low', TrapezoidalMF(0, 0, 5, 10), np.linspace(0, 50, 1000)),
            'low': FuzzySet('low', TrapezoidalMF(5, 10, 15, 20), np.linspace(0, 50, 1000)),
            'moderate': FuzzySet('moderate', TrapezoidalMF(15, 20, 25, 30), np.linspace(0, 50, 1000)),
            'high': FuzzySet('high', TrapezoidalMF(25, 30, 40, 50), np.linspace(0, 50, 1000)),
            'very_high': FuzzySet('very_high', TrapezoidalMF(40, 50, 50, 50), np.linspace(0, 50, 1000))
        }
        
        # Conservation scores (0-1 scale)
        self.input_variables['conservation'] = {
            'low': FuzzySet('low', TrapezoidalMF(0, 0, 0.2, 0.4), np.linspace(0, 1, 1000)),
            'moderate': FuzzySet('moderate', GaussianMF(0.5, 0.15), np.linspace(0, 1, 1000)),
            'high': FuzzySet('high', TrapezoidalMF(0.6, 0.8, 1.0, 1.0), np.linspace(0, 1, 1000))
        }
        
        # Allele frequency (0-1 scale)
        self.input_variables['frequency'] = {
            'very_rare': FuzzySet('very_rare', SigmoidMF(0.001, -1000), np.linspace(0, 1, 1000)),
            'rare': FuzzySet('rare', TrapezoidalMF(0.001, 0.005, 0.01, 0.05), np.linspace(0, 1, 1000)),
            'common': FuzzySet('common', TrapezoidalMF(0.01, 0.05, 0.5, 1.0), np.linspace(0, 1, 1000))
        }
        
        # Expression fold change (log2 scale, -10 to 10)
        self.input_variables['expression'] = {
            'highly_downregulated': FuzzySet('highly_downregulated', 
                                           SigmoidMF(-3, -2), np.linspace(-10, 10, 1000)),
            'downregulated': FuzzySet('downregulated', 
                                    TrapezoidalMF(-5, -3, -1, 0), np.linspace(-10, 10, 1000)),
            'unchanged': FuzzySet('unchanged', GaussianMF(0, 0.5), np.linspace(-10, 10, 1000)),
            'upregulated': FuzzySet('upregulated', 
                                  TrapezoidalMF(0, 1, 3, 5), np.linspace(-10, 10, 1000)),
            'highly_upregulated': FuzzySet('highly_upregulated', 
                                         SigmoidMF(3, 2), np.linspace(-10, 10, 1000))
        }
        
        # Clinical significance confidence (0-1 scale)
        self.output_variables['confidence'] = {
            'very_low': FuzzySet('very_low', TrapezoidalMF(0, 0, 0.1, 0.3), np.linspace(0, 1, 1000)),
            'low': FuzzySet('low', TrapezoidalMF(0.1, 0.3, 0.4, 0.5), np.linspace(0, 1, 1000)),
            'moderate': FuzzySet('moderate', TrapezoidalMF(0.4, 0.5, 0.6, 0.7), np.linspace(0, 1, 1000)),
            'high': FuzzySet('high', TrapezoidalMF(0.6, 0.7, 0.8, 0.9), np.linspace(0, 1, 1000)),
            'very_high': FuzzySet('very_high', TrapezoidalMF(0.8, 0.9, 1.0, 1.0), np.linspace(0, 1, 1000))
        }
        
        # Initialize fuzzy rules
        self._initialize_fuzzy_rules()
    
    def _initialize_fuzzy_rules(self):
        """Initialize fuzzy inference rules for genomic analysis"""
        
        # High pathogenicity + high conservation + rare frequency = high confidence
        self.rules.append(FuzzyRule(
            antecedent={'pathogenicity': 'high', 'conservation': 'high', 'frequency': 'rare'},
            consequent={'confidence': 'very_high'},
            weight=1.0
        ))
        
        # Very high pathogenicity + any conservation + very rare = very high confidence
        self.rules.append(FuzzyRule(
            antecedent={'pathogenicity': 'very_high', 'frequency': 'very_rare'},
            consequent={'confidence': 'very_high'},
            weight=0.9
        ))
        
        # Low pathogenicity + low conservation = low confidence
        self.rules.append(FuzzyRule(
            antecedent={'pathogenicity': 'low', 'conservation': 'low'},
            consequent={'confidence': 'low'},
            weight=0.8
        ))
        
        # High frequency variants = lower confidence (unless very high pathogenicity)
        self.rules.append(FuzzyRule(
            antecedent={'frequency': 'common', 'pathogenicity': 'moderate'},
            consequent={'confidence': 'low'},
            weight=0.7
        ))
        
        # Moderate pathogenicity + moderate conservation = moderate confidence
        self.rules.append(FuzzyRule(
            antecedent={'pathogenicity': 'moderate', 'conservation': 'moderate'},
            consequent={'confidence': 'moderate'},
            weight=0.8
        ))
        
        # Expression-based rules
        self.rules.append(FuzzyRule(
            antecedent={'expression': 'highly_upregulated', 'pathogenicity': 'high'},
            consequent={'confidence': 'high'},
            weight=0.6
        ))
        
        self.rules.append(FuzzyRule(
            antecedent={'expression': 'highly_downregulated', 'pathogenicity': 'high'},
            consequent={'confidence': 'high'},
            weight=0.6
        ))
    
    def compute_fuzzy_confidence(self, variant_data: Dict[str, float]) -> Dict[str, float]:
        """
        Compute fuzzy confidence scores for variant pathogenicity.
        
        Args:
            variant_data: Dictionary with genomic features
            
        Returns:
            Dictionary with confidence scores and intermediate results
        """
        # Extract input values
        inputs = {
            'pathogenicity': variant_data.get('cadd_score', 0),
            'conservation': variant_data.get('conservation_score', 0),
            'frequency': variant_data.get('allele_frequency', 0.5),
            'expression': variant_data.get('log2_fold_change', 0)
        }
        
        # Compute membership degrees for all input variables
        memberships = {}
        for var_name, value in inputs.items():
            if var_name in self.input_variables:
                var_memberships = {}
                for set_name, fuzzy_set in self.input_variables[var_name].items():
                    var_memberships[set_name] = fuzzy_set.membership(value)
                memberships[var_name] = var_memberships
        
        # Apply fuzzy rules (Mamdani inference)
        rule_outputs = []
        for rule in self.rules:
            # Compute rule strength (minimum of antecedent memberships)
            rule_strength = 1.0
            for var_name, set_name in rule.antecedent.items():
                if var_name in memberships and set_name in memberships[var_name]:
                    rule_strength = min(rule_strength, memberships[var_name][set_name])
                else:
                    rule_strength = 0.0
                    break
            
            # Apply weight
            rule_strength *= rule.weight
            
            # Store rule output
            if rule_strength > 0:
                for var_name, set_name in rule.consequent.items():
                    rule_outputs.append({
                        'variable': var_name,
                        'set': set_name,
                        'strength': rule_strength
                    })
        
        # Aggregate rule outputs and defuzzify
        final_outputs = {}
        for var_name in self.output_variables:
            # Collect all rule outputs for this variable
            var_outputs = [r for r in rule_outputs if r['variable'] == var_name]
            
            if var_outputs:
                # Aggregate using maximum (OR operation)
                aggregated_sets = {}
                for output in var_outputs:
                    set_name = output['set']
                    strength = output['strength']
                    if set_name in aggregated_sets:
                        aggregated_sets[set_name] = max(aggregated_sets[set_name], strength)
                    else:
                        aggregated_sets[set_name] = strength
                
                # Defuzzify using centroid method
                final_value = self._defuzzify_centroid(var_name, aggregated_sets)
                final_outputs[var_name] = final_value
            else:
                final_outputs[var_name] = 0.5  # Default moderate value
        
        # Prepare detailed results
        results = {
            'confidence_score': final_outputs.get('confidence', 0.5),
            'input_memberships': memberships,
            'rule_activations': len([r for r in rule_outputs if r['strength'] > 0]),
            'inputs': inputs,
            'fuzzy_outputs': final_outputs
        }
        
        return results
    
    def _defuzzify_centroid(self, var_name: str, aggregated_sets: Dict[str, float]) -> float:
        """
        Defuzzify using centroid method.
        
        Args:
            var_name: Output variable name
            aggregated_sets: Dictionary of set names to activation strengths
            
        Returns:
            Defuzzified crisp value
        """
        if var_name not in self.output_variables:
            return 0.5
        
        universe = self.output_variables[var_name][list(aggregated_sets.keys())[0]].universe
        
        # Compute aggregated membership function
        aggregated_membership = np.zeros_like(universe)
        
        for set_name, strength in aggregated_sets.items():
            if set_name in self.output_variables[var_name]:
                fuzzy_set = self.output_variables[var_name][set_name]
                set_membership = fuzzy_set.membership(universe)
                # Clip membership at rule strength (Mamdani method)
                clipped_membership = np.minimum(set_membership, strength)
                # Combine using maximum
                aggregated_membership = np.maximum(aggregated_membership, clipped_membership)
        
        # Compute centroid
        if np.sum(aggregated_membership) > 0:
            centroid = np.sum(universe * aggregated_membership) / np.sum(aggregated_membership)
        else:
            centroid = np.mean(universe)  # Default to center if no activation
        
        return float(centroid)
    
    def compute_uncertainty_bounds(self, 
                                 variant_data: Dict[str, float],
                                 confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Compute uncertainty bounds for confidence estimates.
        
        Args:
            variant_data: Genomic feature data
            confidence_level: Confidence level for bounds (0-1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Get base confidence
        fuzzy_result = self.compute_fuzzy_confidence(variant_data)
        base_confidence = fuzzy_result['confidence_score']
        
        # Estimate uncertainty based on rule activation
        n_activated_rules = fuzzy_result['rule_activations']
        total_rules = len(self.rules)
        
        # More activated rules = lower uncertainty
        rule_coverage = n_activated_rules / total_rules
        uncertainty = (1 - rule_coverage) * 0.2  # Max 20% uncertainty
        
        # Compute bounds
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha/2)
        
        margin = z_score * uncertainty
        lower_bound = max(0, base_confidence - margin)
        upper_bound = min(1, base_confidence + margin)
        
        return lower_bound, upper_bound
    
    def add_custom_rule(self, rule: FuzzyRule):
        """Add a custom fuzzy rule to the system"""
        self.rules.append(rule)
        self.logger.info(f"Added custom rule: {rule}")
    
    def evaluate_rule_importance(self, test_data: List[Dict[str, float]]) -> Dict[int, float]:
        """
        Evaluate the importance of each rule using test data.
        
        Args:
            test_data: List of test cases with genomic features
            
        Returns:
            Dictionary mapping rule index to importance score
        """
        rule_importance = {}
        
        for rule_idx in range(len(self.rules)):
            # Temporarily remove rule
            removed_rule = self.rules.pop(rule_idx)
            
            # Compute difference in outputs
            total_difference = 0
            for test_case in test_data:
                # With rule removed
                result_without = self.compute_fuzzy_confidence(test_case)
                
                # Add rule back temporarily
                self.rules.insert(rule_idx, removed_rule)
                result_with = self.compute_fuzzy_confidence(test_case)
                self.rules.pop(rule_idx)
                
                # Compute absolute difference
                diff = abs(result_with['confidence_score'] - result_without['confidence_score'])
                total_difference += diff
            
            # Restore rule
            self.rules.insert(rule_idx, removed_rule)
            
            # Average importance
            rule_importance[rule_idx] = total_difference / len(test_data)
        
        return rule_importance
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state for debugging/monitoring"""
        return {
            'input_variables': list(self.input_variables.keys()),
            'output_variables': list(self.output_variables.keys()),
            'num_rules': len(self.rules),
            'rule_summary': [
                {
                    'antecedent': rule.antecedent,
                    'consequent': rule.consequent,
                    'weight': rule.weight
                }
                for rule in self.rules[:5]  # First 5 rules
            ]
        } 