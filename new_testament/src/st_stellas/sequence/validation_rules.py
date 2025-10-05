#!/usr/bin/env python3
"""
Genomic Validation Rules for Oscillatory Sequence Analysis

Implements fundamental genomic test signals equivalent to electronic circuit test signals:
1. Chargaff's Rules (Fundamental DNA Test)
2. K-mer Distribution Tests  
3. GC Content Distribution
4. Complexity Measures
5. Oscillatory-Specific Validation Rules

Author: Based on mathematical-necessity.tex oscillatory theoretical framework
"""

import numpy as np
from collections import Counter
from scipy import stats
from typing import Dict
import argparse


class GenomicValidationRules:
    """Comprehensive genomic validation rules combining standard tests with oscillatory analysis."""
    
    def __init__(self):
        self.cardinal_map = {'A': (0, 1), 'T': (0, -1), 'G': (1, 0), 'C': (-1, 0)}
        self.thresholds = {
            'chargaff_tolerance': 0.05,
            'coordinate_balance_max': 0.1,
            'kmer_deviation_threshold': 2.0,
            'complexity_min': 1.0,
            'complexity_max': 1.9
        }
    
    def validate_chargaffs_rules(self, sequence: str) -> Dict:
        """Validate Chargaff's Rules: A ≈ T and G ≈ C with oscillatory coordinate validation."""
        sequence = sequence.upper()
        base_counts = Counter(sequence)
        total = len(sequence)
        
        freq_A = base_counts.get('A', 0) / total
        freq_T = base_counts.get('T', 0) / total  
        freq_G = base_counts.get('G', 0) / total
        freq_C = base_counts.get('C', 0) / total
        
        at_deviation = abs(freq_A - freq_T)
        gc_deviation = abs(freq_G - freq_C)
        
        # Oscillatory coordinate validation
        coordinate_path = self._sequence_to_coordinates(sequence)
        net_displacement = coordinate_path[-1] if len(coordinate_path) > 0 else np.array([0, 0])
        coordinate_balance = np.linalg.norm(net_displacement) / len(sequence) if len(sequence) > 0 else 0
        
        chargaff_valid = (at_deviation < self.thresholds['chargaff_tolerance'] and 
                         gc_deviation < self.thresholds['chargaff_tolerance'])
        oscillatory_valid = coordinate_balance < self.thresholds['coordinate_balance_max']
        
        return {
            'base_frequencies': {'A': freq_A, 'T': freq_T, 'G': freq_G, 'C': freq_C},
            'at_deviation': at_deviation,
            'gc_deviation': gc_deviation,
            'coordinate_balance': coordinate_balance,
            'chargaff_rules_satisfied': chargaff_valid,
            'oscillatory_balance_valid': oscillatory_valid,
            'overall_validation': chargaff_valid and oscillatory_valid
        }
    
    def validate_kmer_distribution(self, sequence: str, k: int = 3) -> Dict:
        """Validate k-mer distribution patterns for real genomic vs random sequences."""
        sequence = sequence.upper()
        
        if len(sequence) < k:
            return {'error': f'Sequence too short for k={k}'}
        
        kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
        kmer_counts = Counter(kmers)
        
        # Expected frequency for random sequence
        expected_freq = 1 / (4 ** k)
        total_kmers = len(kmers)
        
        # Chi-square test for randomness
        all_possible_kmers = self._generate_all_kmers(k)
        observed_counts = [kmer_counts.get(kmer, 0) for kmer in all_possible_kmers]
        expected_counts = [expected_freq * total_kmers] * len(all_possible_kmers)
        
        chi2_stat, chi2_pvalue = stats.chisquare(observed_counts, expected_counts)
        
        # Calculate diversity
        observed_freqs = {kmer: count/total_kmers for kmer, count in kmer_counts.items()}
        shannon_diversity = -sum(freq * np.log2(freq) for freq in observed_freqs.values() if freq > 0)
        
        # Deviation from randomness
        deviations = [(obs - exp)/np.sqrt(exp) for obs, exp in zip(observed_counts, expected_counts)]
        mean_deviation = np.mean(np.abs(deviations))
        
        is_real_genomic = mean_deviation > self.thresholds['kmer_deviation_threshold']
        
        return {
            'k': k,
            'total_kmers': total_kmers,
            'unique_kmers': len(kmer_counts),
            'shannon_diversity': shannon_diversity,
            'chi2_pvalue': chi2_pvalue,
            'mean_deviation_from_random': mean_deviation,
            'is_real_genomic_pattern': is_real_genomic
        }
    
    def validate_gc_content(self, sequence: str, expected_gc: float = 0.41) -> Dict:
        """Validate GC content distribution."""
        sequence = sequence.upper()
        gc_count = sequence.count('G') + sequence.count('C')
        gc_content = gc_count / len(sequence) if len(sequence) > 0 else 0
        gc_deviation = abs(gc_content - expected_gc)
        
        return {
            'gc_content': gc_content,
            'expected_gc_content': expected_gc,
            'gc_deviation': gc_deviation,
            'gc_content_valid': gc_deviation < 0.2
        }
    
    def validate_complexity_measures(self, sequence: str) -> Dict:
        """Validate sequence complexity - real DNA should show intermediate complexity."""
        sequence = sequence.upper()
        base_counts = Counter(sequence)
        total = len(sequence)
        
        # Shannon entropy
        shannon_entropy = -sum((count/total) * np.log2(count/total) 
                              for count in base_counts.values() if count > 0)
        
        # Real DNA complexity check
        is_real_dna_like = (self.thresholds['complexity_min'] <= shannon_entropy <= 
                           self.thresholds['complexity_max'])
        
        return {
            'shannon_entropy': shannon_entropy,
            'complexity_valid': is_real_dna_like
        }
    
    def validate_oscillatory_signatures(self, sequence: str) -> Dict:
        """Validate oscillatory-specific sequence signatures."""
        coordinate_path = self._sequence_to_coordinates(sequence)
        
        if len(coordinate_path) < 10:
            return {'error': 'Sequence too short for oscillatory analysis'}
        
        # Path coherence
        path_coherence = self._calculate_path_coherence(coordinate_path)
        
        # Compare with random sequence
        random_sequence = ''.join(np.random.choice(['A', 'T', 'G', 'C'], len(sequence)))
        random_coords = self._sequence_to_coordinates(random_sequence)
        random_coherence = self._calculate_path_coherence(random_coords)
        
        coherence_enhancement = path_coherence / random_coherence if random_coherence > 0 else float('inf')
        
        return {
            'path_coherence': path_coherence,
            'coherence_enhancement': coherence_enhancement,
            'oscillatory_signatures_valid': coherence_enhancement > 1.5
        }
    
    def run_comprehensive_validation(self, sequence: str, expected_organism: str = 'human') -> Dict:
        """Run all validation rules on a sequence."""
        expected_gc = {'human': 0.41, 'ecoli': 0.51, 'yeast': 0.38}.get(expected_organism, 0.41)
        
        results = {
            'chargaffs_rules': self.validate_chargaffs_rules(sequence),
            'kmer_distribution': self.validate_kmer_distribution(sequence),
            'gc_content': self.validate_gc_content(sequence, expected_gc),
            'complexity_measures': self.validate_complexity_measures(sequence),
            'oscillatory_signatures': self.validate_oscillatory_signatures(sequence)
        }
        
        # Overall validation
        validations = [
            results['chargaffs_rules']['overall_validation'],
            results['kmer_distribution']['is_real_genomic_pattern'],
            results['gc_content']['gc_content_valid'],
            results['complexity_measures']['complexity_valid'],
            results['oscillatory_signatures']['oscillatory_signatures_valid']
        ]
        
        results['overall_validation'] = {
            'total_passed': sum(validations),
            'total_tests': len(validations),
            'overall_valid': sum(validations) >= len(validations) * 0.6
        }
        
        return results
    
    # Helper methods
    def _sequence_to_coordinates(self, sequence: str) -> np.ndarray:
        """Convert sequence to cardinal coordinate path."""
        coordinates = []
        current_pos = np.array([0.0, 0.0])
        
        for base in sequence.upper():
            if base in self.cardinal_map:
                current_pos += np.array(self.cardinal_map[base])
                coordinates.append(current_pos.copy())
        
        return np.array(coordinates) if coordinates else np.array([[0, 0]])
    
    def _generate_all_kmers(self, k: int) -> list:
        """Generate all possible k-mers."""
        if k == 1:
            return ['A', 'T', 'G', 'C']
        
        kmers = []
        for base in ['A', 'T', 'G', 'C']:
            for kmer in self._generate_all_kmers(k-1):
                kmers.append(base + kmer)
        return kmers
    
    def _calculate_path_coherence(self, coordinate_path: np.ndarray) -> float:
        """Calculate oscillatory path coherence."""
        if len(coordinate_path) < 2:
            return 0
        
        steps = np.diff(coordinate_path, axis=0)
        if len(steps) == 0:
            return 0
        
        step_angles = np.arctan2(steps[:, 1], steps[:, 0])
        angle_consistency = 1 - np.var(step_angles) / (np.pi**2)
        
        return max(0, angle_consistency)


def main():
    """Main function for testing validation rules."""
    parser = argparse.ArgumentParser(description="Genomic Validation Rules Testing")
    parser.add_argument("--sequence", type=str, help="DNA sequence to validate")
    parser.add_argument("--test-mode", action="store_true", help="Run with test sequences")
    
    args = parser.parse_args()
    
    validator = GenomicValidationRules()
    
    if args.test_mode:
        test_sequences = {
            'perfect_palindrome': 'ATGCGCGCAT',
            'tandem_repeat': 'ATGC' * 25,
            'random_sequence': ''.join(np.random.choice(['A', 'T', 'G', 'C'], 1000)),
            'high_gc': 'GCGCGCGCGCGCGCGCGCGC'
        }
        
        print("🧬 Genomic Validation Rules - Test Mode")
        print("=" * 60)
        
        for name, sequence in test_sequences.items():
            print(f"\n📊 Testing: {name}")
            results = validator.run_comprehensive_validation(sequence)
            print(f"✅ Overall validation: {results['overall_validation']['overall_valid']}")
            print(f"📈 Tests passed: {results['overall_validation']['total_passed']}/{results['overall_validation']['total_tests']}")
    
    elif args.sequence:
        results = validator.run_comprehensive_validation(args.sequence)
        print(f"Overall validation: {results['overall_validation']['overall_valid']}")
        print(f"Tests passed: {results['overall_validation']['total_passed']}/{results['overall_validation']['total_tests']}")
    
    else:
        print("Please provide --sequence or use --test-mode")


if __name__ == "__main__":
    main()