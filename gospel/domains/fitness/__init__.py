"""
Fitness domain module for Gospel.

This module handles fitness-related genomic analysis, including athletic performance,
exercise response, injury risk, and recovery traits.
"""

from .models import FitnessDomain, TraitProfile, GeneticEffect
from .constants import (
    FITNESS_TRAITS, 
    PERFORMANCE_GENES, 
    ENDURANCE_GENES, 
    POWER_GENES,
    INJURY_RISK_GENES,
    RECOVERY_GENES
)

__all__ = [
    'FitnessDomain',
    'TraitProfile',
    'GeneticEffect',
    'FITNESS_TRAITS',
    'PERFORMANCE_GENES',
    'ENDURANCE_GENES',
    'POWER_GENES',
    'INJURY_RISK_GENES',
    'RECOVERY_GENES'
] 