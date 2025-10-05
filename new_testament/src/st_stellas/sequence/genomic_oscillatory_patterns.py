#!/usr/bin/env python3
"""
Genomic Oscillatory Pattern Recognition

Validate functional element detection through oscillatory signatures (promoters, coding regions, etc.)
Test cross-sequence pattern transfer and universal genomic motifs
Verify oscillatory hierarchy across different genomic scales

Based on mathematical-necessity.tex oscillatory theoretical framework
"""

import numpy as np
from numba import jit
import argparse
from typing import Dict, List
from collections import defaultdict


@jit(nopython=True, cache=True)
def _sequence_to_coordinates_numba(sequence_array):
    """Numba-optimized cardinal direction transformation."""
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


@jit(nopython=True, cache=True)
def _calculate_oscillatory_features(coordinate_path):
    """Calculate oscillatory features for functional element detection."""
    if len(coordinate_path) < 5:
        return np.zeros(4)
    
    # 1. Path curvature
    curvatures = []
    for i in range(1, len(coordinate_path) - 1):
        v1 = coordinate_path[i] - coordinate_path[i-1]
        v2 = coordinate_path[i+1] - coordinate_path[i]
        
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        norm1 = np.sqrt(v1[0]**2 + v1[1]**2)
        norm2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        if norm1 > 0 and norm2 > 0:
            curvatures.append(abs(cross) / (norm1 * norm2))
    
    avg_curvature = np.mean(np.array(curvatures)) if len(curvatures) > 0 else 0
    
    # 2. Oscillation frequency
    x_coords = coordinate_path[:, 0]
    y_coords = coordinate_path[:, 1]
    
    x_diffs = np.diff(x_coords)
    y_diffs = np.diff(y_coords)
    
    x_oscillations = np.sum((x_diffs[:-1] * x_diffs[1:]) < 0)
    y_oscillations = np.sum((y_diffs[:-1] * y_diffs[1:]) < 0)
    
    oscillation_frequency = (x_oscillations + y_oscillations) / len(coordinate_path)
    
    # 3. Path variance
    coord_variance = np.var(coordinate_path[:, 0]) + np.var(coordinate_path[:, 1])
    
    # 4. Path length vs displacement ratio
    path_length = 0
    for i in range(1, len(coordinate_path)):
        diff = coordinate_path[i] - coordinate_path[i-1]
        path_length += np.sqrt(diff[0]**2 + diff[1]**2)
    
    displacement = np.sqrt(coordinate_path[-1, 0]**2 + coordinate_path[-1, 1]**2)
    complexity_ratio = path_length / (displacement + 1e-6)
    
    return np.array([avg_curvature, oscillation_frequency, coord_variance, complexity_ratio])


class GenomicOscillatoryPatternRecognizer:
    """Recognizes functional genomic elements through oscillatory signatures."""
    
    def __init__(self):
        # Simple thresholds for functional element classification
        self.element_thresholds = {
            'promoter': {'curvature': (0.3, 0.8), 'oscillation': (0.1, 0.4), 'variance': (10, 100), 'complexity': (1.2, 2.0)},
            'coding': {'curvature': (0.1, 0.4), 'oscillation': (0.05, 0.2), 'variance': (50, 200), 'complexity': (1.5, 3.0)},
            'regulatory': {'curvature': (0.4, 1.0), 'oscillation': (0.2, 0.6), 'variance': (5, 80), 'complexity': (1.0, 1.8)}
        }
        print("GenomicOscillatoryPatternRecognizer initialized.")
    
    def detect_functional_elements(self, sequence: str, window_size: int = 50) -> Dict:
        """Detect functional elements through oscillatory signatures."""
        sequence = sequence.upper()
        detections = []
        element_counts = defaultdict(int)
        
        # Sliding window analysis
        for start in range(0, len(sequence) - window_size + 1, window_size // 2):
            end = start + window_size
            window_seq = sequence[start:end]
            
            # Convert to coordinates and extract features
            sequence_array = np.array([ord(c) for c in window_seq], dtype=np.uint8)
            coordinate_path = _sequence_to_coordinates_numba(sequence_array)
            features = _calculate_oscillatory_features(coordinate_path)
            
            # Classify based on oscillatory signatures
            element_type = self._classify_functional_element(features)
            
            if element_type != 'unknown':
                detections.append({
                    'start': start,
                    'end': end,
                    'element_type': element_type,
                    'features': features.tolist()
                })
                element_counts[element_type] += 1
        
        return {
            'sequence_length': len(sequence),
            'functional_elements_detected': len(detections),
            'element_counts': dict(element_counts),
            'detections': detections,
            'functional_detection_successful': len(detections) > 0
        }
    
    def analyze_cross_sequence_patterns(self, sequences: List[str]) -> Dict:
        """Test cross-sequence pattern transfer."""
        universal_patterns = defaultdict(list)
        
        for i, sequence in enumerate(sequences):
            analysis = self.detect_functional_elements(sequence)
            
            for detection in analysis.get('detections', []):
                pattern_key = detection['element_type']
                universal_patterns[pattern_key].append(i)
        
        # Find patterns appearing in multiple sequences
        universal_motifs = {pattern: seq_indices for pattern, seq_indices in universal_patterns.items() 
                           if len(seq_indices) > 1}
        
        total_patterns = sum(len(seq_indices) for seq_indices in universal_patterns.values())
        universal_pattern_count = sum(len(seq_indices) for seq_indices in universal_motifs.values())
        transfer_rate = universal_pattern_count / total_patterns if total_patterns > 0 else 0
        
        return {
            'sequences_analyzed': len(sequences),
            'universal_motifs_found': len(universal_motifs),
            'pattern_transfer_rate': transfer_rate,
            'cross_sequence_validation_successful': transfer_rate > 0.1
        }
    
    def validate_oscillatory_hierarchy(self, sequence: str, scales: List[int] = [25, 50, 100]) -> Dict:
        """Verify oscillatory hierarchy across different scales."""
        hierarchy_results = {}
        
        for scale in scales:
            if len(sequence) >= scale:
                analysis = self.detect_functional_elements(sequence, window_size=scale)
                hierarchy_results[scale] = {
                    'detection_count': analysis['functional_elements_detected'],
                    'element_types': len(analysis['element_counts'])
                }
        
        # Check for scale-dependent patterns
        scales_analyzed = list(hierarchy_results.keys())
        if len(scales_analyzed) > 1:
            detection_counts = [hierarchy_results[scale]['detection_count'] for scale in scales_analyzed]
            scale_correlation = np.corrcoef(scales_analyzed, detection_counts)[0, 1] if len(scales_analyzed) > 1 else 0
        else:
            scale_correlation = 0
        
        return {
            'scales_analyzed': scales_analyzed,
            'hierarchy_results': hierarchy_results,
            'scale_correlation': scale_correlation,
            'oscillatory_hierarchy_detected': abs(scale_correlation) > 0.3
        }
    
    def _classify_functional_element(self, features: np.ndarray) -> str:
        """Classify functional element based on oscillatory features."""
        if len(features) != 4:
            return 'unknown'
        
        curvature, oscillation, variance, complexity = features
        
        for element_type, thresholds in self.element_thresholds.items():
            score = 0
            total = 0
            
            for i, (feature_name, (min_val, max_val)) in enumerate(thresholds.items()):
                if min_val <= features[i] <= max_val:
                    score += 1
                total += 1
            
            if total > 0 and score / total > 0.5:  # At least 50% features match
                return element_type
        
        return 'unknown'


def main():
    """Main function for testing genomic oscillatory pattern recognition."""
    parser = argparse.ArgumentParser(description="Genomic Oscillatory Pattern Recognition")
    parser.add_argument("--sequence", type=str, help="DNA sequence to analyze")
    parser.add_argument("--test-mode", action="store_true", help="Run with test sequences")
    parser.add_argument("--window-size", type=int, default=50, help="Window size for analysis")
    
    args = parser.parse_args()
    
    recognizer = GenomicOscillatoryPatternRecognizer()
    
    if args.test_mode:
        test_sequences = [
            'TATAATGGCGCGTATACCGGGCCCAATTGGCCTTAAGGTCGACCTGCAG',  # Promoter-like
            'ATGGCGTTTCACTTCTGAGTTCGGCATGGCATCTCTTGCCGACAATCGC',  # Coding-like
            'CGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA',       # Regulatory-like
            ''.join(np.random.choice(['A', 'T', 'G', 'C'], 100))    # Random
        ]
        
        print("🧬 Genomic Oscillatory Pattern Recognition - Test Mode")
        print("=" * 70)
        
        for i, sequence in enumerate(test_sequences):
            print(f"\n📊 Testing sequence {i+1}:")
            detection_results = recognizer.detect_functional_elements(sequence, args.window_size)
            print(f"🔬 Functional elements detected: {detection_results['functional_elements_detected']}")
            print(f"🎯 Element types found: {list(detection_results['element_counts'].keys())}")
        
        # Cross-sequence analysis
        cross_analysis = recognizer.analyze_cross_sequence_patterns(test_sequences)
        print(f"\n🔄 Cross-Sequence Analysis:")
        print(f"Universal motifs found: {cross_analysis['universal_motifs_found']}")
        print(f"Pattern transfer rate: {cross_analysis['pattern_transfer_rate']:.2%}")
        
        # Hierarchy analysis
        hierarchy_results = recognizer.validate_oscillatory_hierarchy(test_sequences[0])
        print(f"\n🏗️ Hierarchical Analysis:")
        print(f"Scales analyzed: {hierarchy_results['scales_analyzed']}")
        print(f"Hierarchy detected: {hierarchy_results['oscillatory_hierarchy_detected']}")
    
    elif args.sequence:
        print("🧬 Genomic Oscillatory Pattern Recognition")
        print("=" * 50)
        
        detection_results = recognizer.detect_functional_elements(args.sequence, args.window_size)
        
        print(f"Sequence length: {len(args.sequence)}")
        print(f"Functional elements detected: {detection_results['functional_elements_detected']}")
        print(f"Element types: {detection_results['element_counts']}")
    
    else:
        print("Please provide --sequence or use --test-mode")


if __name__ == "__main__":
    main()