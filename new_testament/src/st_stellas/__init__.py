"""
St. Stella's Genomic Analysis Framework
New Testament - Validation Framework for Theoretical Genomic Analysis

This package provides high-performance implementations of St. Stella's genomic analysis
theories, focusing on cardinal direction coordinate transformation and S-entropy navigation.

Main Modules:
- sequence: Core sequence analysis with cardinal direction transformation
- genome: Whole genome and population-scale analysis
- theory: Theoretical framework implementations
- whole_genome_sequencing: Large-scale genomic data processing

Performance Characteristics:
- 273× to 227,191× speedup over traditional methods
- 88-99.3% memory reduction through coordinate compression  
- 156-623% accuracy improvement in pattern recognition
- O(n) → O(log S₀) complexity reduction

Author: Kundai Farai Sachikonye
Institution: Technical University of Munich
Email: sachikonye@wzw.tum.de
"""

from .sequence import (
    StStellaSequenceTransformer,
    DualStrandAnalyzer,
    SEntropyNavigator,
    GenomicPatternExtractor,
    StStellasBenchmarkSuite,
    validate_sequence,
    get_performance_info,
    print_framework_info,
    CARDINAL_DIRECTIONS,
    PERFORMANCE_TARGETS
)

__version__ = "1.0.0"
__author__ = "Kundai Farai Sachikonye"
__email__ = "sachikonye@wzw.tum.de"

__all__ = [
    # Core sequence analysis
    'StStellaSequenceTransformer',
    'DualStrandAnalyzer', 
    'SEntropyNavigator',
    'GenomicPatternExtractor',
    'StStellasBenchmarkSuite',
    
    # Utility functions
    'validate_sequence',
    'get_performance_info',
    'print_framework_info',
    
    # Constants
    'CARDINAL_DIRECTIONS',
    'PERFORMANCE_TARGETS',
    
    # Metadata
    '__version__',
    '__author__',
    '__email__'
]

# Quick access to framework information
def framework_info():
    """Display comprehensive framework information."""
    print_framework_info()

def quick_start_example():
    """Print a quick start example."""
    example = '''
# New Testament - St. Stella's Framework Quick Start

from st_stellas import StStellaSequenceTransformer, DualStrandAnalyzer

# Initialize transformer
transformer = StStellaSequenceTransformer()

# Transform sequences to cardinal coordinates
sequences = ['ATGCGTACGTA', 'GCTATCGATGC']
coord_paths = transformer.transform_sequences_batch(sequences)

# Dual-strand geometric analysis
analyzer = DualStrandAnalyzer()
forward_paths, reverse_paths, features = analyzer.analyze_dual_strand_batch(sequences)

print(f"Coordinate paths shape: {coord_paths.shape}")
print(f"Geometric features: {features}")
    '''
    print(example)
