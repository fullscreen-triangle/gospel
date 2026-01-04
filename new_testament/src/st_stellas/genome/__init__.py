"""
St. Stella's Genome - Whole Genome Analysis Module
Population-scale genomic analysis using coordinate transformation and computational pharmacology.

This module extends the sequence analysis capabilities to whole genome processing,
enabling VCF analysis, pharmaceutical predictions, and multi-framework integration
using the oscillatory genomics principles.

Modules:
- VCF analysis and pharmacogenomic variant processing
- Oscillatory hole identification and therapeutic targeting
- Multi-framework integration (Nebuchadnezzar, Borgia, Bene Gesserit, Hegel)
- Pharmaceutical response prediction through oscillatory matching
- Gene-as-oscillator circuit modeling
- Intracellular Bayesian network analysis

Active implementations:
- Dante Labs VCF computational pharmacology analysis
- Multi-scale oscillatory signature extraction
- Cross-framework validation and evidence rectification
- Personalized pharmaceutical recommendation generation
"""

__version__ = "1.0.0"
__author__ = "Kundai Farai Sachikonye"

# Import VCF Analysis Pipeline
from .dante_labs_vcf_analyzer import (
    DanteLabsVCFAnalyzer,
    PharmacogenomicVariant
)

from .multi_framework_integrator import (
    MultiFrameworkIntegrator,
    FrameworkIntegrationResults
)

from .complete_vcf_pipeline import (
    run_complete_pipeline,
    create_vcf_analysis_pipeline
)

# Import Core Pharmaceutical Analysis
from .pharmaceutical_response import (
    PharmaceuticalOscillatoryMatcher,
    Drug
)

from .genomic_oscillators import (
    GeneAsOscillatorModel,
    GeneOscillator
)

from .intracellular_bayesian import (
    IntracellularBayesianNetwork,
    BayesianNetworkState
)

# Import other genome modules if they exist
try:
    from .membrane_quantum import *
except ImportError:
    pass

try:
    from .microbiome_network import *
except ImportError:
    pass

try:
    from .universal_oscillatory import *
except ImportError:
    pass

__all__ = [
    # VCF Analysis Pipeline
    'DanteLabsVCFAnalyzer',
    'PharmacogenomicVariant',
    'MultiFrameworkIntegrator',
    'FrameworkIntegrationResults',
    'run_complete_pipeline',
    'create_vcf_analysis_pipeline',

    # Core Pharmaceutical Analysis
    'PharmaceuticalOscillatoryMatcher',
    'Drug',

    # Genomic Oscillators
    'GeneAsOscillatorModel',
    'GeneOscillator',

    # Intracellular Dynamics
    'IntracellularBayesianNetwork',
    'BayesianNetworkState',
]
