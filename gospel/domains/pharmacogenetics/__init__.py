"""
Pharmacogenetics domain module for Gospel.

This module handles pharmacogenetic analysis, including drug metabolism,
efficacy, and safety profiles based on genomic variants.
"""

from .models import PharmacogeneticsDomain, DrugProfile, DrugInteraction
from .constants import (
    METABOLISM_GENES, 
    DRUG_TARGET_GENES, 
    PGX_DRUGS,
    DRUG_PATHWAY_GENES,
    ADVERSE_EFFECT_GENES
)

__all__ = [
    'PharmacogeneticsDomain',
    'DrugProfile',
    'DrugInteraction',
    'METABOLISM_GENES',
    'DRUG_TARGET_GENES',
    'PGX_DRUGS',
    'DRUG_PATHWAY_GENES',
    'ADVERSE_EFFECT_GENES'
] 