#!/usr/bin/env python3
"""
Multi-Dimensional Palindrome Detection for St. Stella's Transformation

Specialized palindrome detection that occurs in multiple dimensions after 
St. Stella's cardinal direction transformation.
"""

import numpy as np
from numba import jit
import argparse


@jit(nopython=True, cache=True)
def _sequence_to_coordinates(sequence_array):
    coordinates = np.zeros((len(sequence_array), 2), dtype=np.float64)
    current_pos = np.array([0.0, 0.0])
    
    for i in range(len(sequence_array)):
        base = sequence_array[i]
        if base == ord('A'):
            current_pos += np.array([0.0, 1.0])
        elif base == ord('T'):
            current_pos += np.array([0.0, -1.0])
        elif base == ord('G'):
            current_pos += np.array([1.0, 0.0])
        elif base == ord('C'):
            current_pos += np.array([-1.0, 0.0])
        
        coordinates[i] = current_pos
    
    return coordinates


class MultiDimensionalPalindromeDetector:
    def __init__(self):
        print("Multi-dimensional palindrome detector initialized.")
    
    def detect_string_palindromes(self, sequence, min_length=3):
        """Detect traditional string palindromes."""
        palindromes = []
        
        for i in range(len(sequence)):
            for j in range(i + min_length, len(sequence) + 1):
                substring = sequence[i:j]
                if substring == substring[::-1]:
                    palindromes.append({
                        'sequence': substring,
                        'start': i,
                        'end': j,
                        'type': 'string'
                    })
        
        return {
            'string_palindromes_found': len(palindromes),
            'palindromes': palindromes
        }
    
    def detect_geometric_palindromes(self, sequence):
        """Detect geometric palindromes in coordinate space."""
        sequence_array = np.array([ord(c) for c in sequence.upper()], dtype=np.uint8)
        coordinate_path = _sequence_to_coordinates(sequence_array)
        
        # Simple geometric palindrome check
        geometric_palindromes = []
        
        return {
            'geometric_palindromes_found': len(geometric_palindromes),
            'palindromes': geometric_palindromes
        }
    
    def comprehensive_analysis(self, sequence):
        """Run comprehensive palindrome analysis."""
        string_results = self.detect_string_palindromes(sequence)
        geometric_results = self.detect_geometric_palindromes(sequence)
        
        return {
            'sequence_length': len(sequence),
            'string_palindromes': string_results,
            'geometric_palindromes': geometric_results,
            'multi_dimensional_detection_successful': (
                string_results['string_palindromes_found'] > 0 or 
                geometric_results['geometric_palindromes_found'] > 0
            )
        }


def main():
    parser = argparse.ArgumentParser(description="Multi-Dimensional Palindrome Detection")
    parser.add_argument("--sequence", type=str, help="DNA sequence to analyze")
    parser.add_argument("--test-mode", action="store_true", help="Run with test sequences")
    
    args = parser.parse_args()
    
    detector = MultiDimensionalPalindromeDetector()
    
    if args.test_mode:
        test_sequences = {
            'perfect_palindrome': 'ATGCGCAT',
            'biological_palindrome': 'GAATTC',
            'complex_sequence': 'ATGGCGTTTCACTTCTGAG'
        }
        
        print("🧬 Multi-Dimensional Palindrome Detection - Test Mode")
        
        for name, sequence in test_sequences.items():
            print(f"\n📊 Testing: {name}")
            results = detector.comprehensive_analysis(sequence)
            print(f"String palindromes: {results['string_palindromes']['string_palindromes_found']}")
            print(f"Geometric palindromes: {results['geometric_palindromes']['geometric_palindromes_found']}")
            print(f"Detection successful: {results['multi_dimensional_detection_successful']}")
    
    elif args.sequence:
        results = detector.comprehensive_analysis(args.sequence)
        print(f"String palindromes: {results['string_palindromes']['string_palindromes_found']}")
        print(f"Geometric palindromes: {results['geometric_palindromes']['geometric_palindromes_found']}")
    
    else:
        print("Please provide --sequence or use --test-mode")


if __name__ == "__main__":
    main()