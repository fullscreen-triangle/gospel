#!/usr/bin/env python3
"""
Installation and Basic Functionality Tests
Verify that the New Testament framework is properly installed and functioning.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

def test_python_version():
    """Test that Python version meets requirements."""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"

def test_core_dependencies():
    """Test that core dependencies are available."""
    try:
        import numpy
        import numba
        import pandas
        import matplotlib
        import biopython
        assert True
    except ImportError as e:
        pytest.fail(f"Core dependency missing: {e}")

def test_package_import():
    """Test that the main package can be imported."""
    try:
        import st_stellas
        from st_stellas.sequence import StStellaSequenceTransformer
        assert True
    except ImportError as e:
        pytest.fail(f"Package import failed: {e}")

def test_cardinal_directions_constant():
    """Test that cardinal directions are properly defined."""
    from st_stellas.sequence import CARDINAL_DIRECTIONS
    
    expected = {
        'A': (0, 1),   # North
        'T': (0, -1),  # South  
        'G': (1, 0),   # East
        'C': (-1, 0)   # West
    }
    
    assert CARDINAL_DIRECTIONS == expected
    assert len(CARDINAL_DIRECTIONS) == 4

def test_sequence_validation():
    """Test basic sequence validation functionality."""
    from st_stellas.sequence import validate_sequence
    
    # Valid sequences
    assert validate_sequence("ATGC") == True
    assert validate_sequence("atgc") == True  # Case insensitive
    assert validate_sequence("ATGCGTACGTA") == True
    
    # Invalid sequences
    assert validate_sequence("ATGCN") == False  # Contains N
    assert validate_sequence("ATGC123") == False  # Contains numbers
    assert validate_sequence("") == True  # Empty sequence is valid

def test_basic_coordinate_transformation():
    """Test basic coordinate transformation functionality."""
    from st_stellas.sequence import StStellaSequenceTransformer
    
    transformer = StStellaSequenceTransformer()
    
    # Test single sequence
    sequence = "ATGC"
    coords = transformer.transform_sequence(sequence)
    
    # Expected path: A(0,1) -> T(0,0) -> G(1,0) -> C(0,0)
    expected = np.array([[0, 1], [0, 0], [1, 0], [0, 0]], dtype=np.float64)
    
    np.testing.assert_array_equal(coords, expected)

def test_batch_transformation():
    """Test batch sequence transformation."""
    from st_stellas.sequence import StStellaSequenceTransformer
    
    transformer = StStellaSequenceTransformer()
    
    sequences = ["AT", "GC"]
    coord_paths = transformer.transform_sequences_batch(sequences)
    
    assert coord_paths.shape == (2, 2, 2)  # 2 sequences, 2 positions, 2D coordinates
    assert coord_paths.dtype == np.float64

def test_performance_info():
    """Test that performance information is accessible."""
    from st_stellas.sequence import get_performance_info, PERFORMANCE_TARGETS
    
    info = get_performance_info()
    
    assert isinstance(info, dict)
    assert 'framework_name' in info
    assert 'version' in info
    assert 'performance_targets' in info
    assert info['framework_name'] == "St. Stella's Sequence"
    
    # Test performance targets
    assert 'min_speedup_factor' in PERFORMANCE_TARGETS
    assert 'max_speedup_factor' in PERFORMANCE_TARGETS
    assert PERFORMANCE_TARGETS['min_speedup_factor'] == 273
    assert PERFORMANCE_TARGETS['max_speedup_factor'] == 227191

def test_framework_info_function():
    """Test that framework info function works without errors."""
    from st_stellas.sequence import print_framework_info
    
    # This should not raise any exceptions
    try:
        print_framework_info()
        assert True
    except Exception as e:
        pytest.fail(f"Framework info function failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
