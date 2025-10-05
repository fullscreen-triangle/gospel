#!/usr/bin/env python3
"""
St. Stella's Sequence - Genomic Pattern Extractor
High Performance Computing Implementation for Multi-Dimensional Pattern Recognition

Extracts geometric patterns invisible to linear sequence analysis through
coordinate-based topological pattern recognition.
"""

import numpy as np
import numba
from numba import jit, prange
import time
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import psutil
import json
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


@jit(nopython=True, cache=True)
def compute_curvature_profile(coordinate_path: np.ndarray) -> np.ndarray:
    """Compute curvature along coordinate path for pattern detection."""
    path_length = coordinate_path.shape[0]
    if path_length < 3:
        return np.zeros(1)
    
    curvature = np.zeros(path_length - 2)
    
    for i in range(1, path_length - 1):
        # Three consecutive points
        p1 = coordinate_path[i - 1]
        p2 = coordinate_path[i]
        p3 = coordinate_path[i + 1]
        
        # Vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Cross product for 2D curvature
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        # Magnitudes
        mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 > 1e-10 and mag2 > 1e-10:
            curvature[i - 1] = abs(cross) / (mag1 * mag2)
        else:
            curvature[i - 1] = 0.0
    
    return curvature


@jit(nopython=True, cache=True)
def compute_velocity_profile(coordinate_path: np.ndarray) -> np.ndarray:
    """Compute velocity (step size) along coordinate path."""
    path_length = coordinate_path.shape[0]
    if path_length < 2:
        return np.zeros(1)
    
    velocity = np.zeros(path_length - 1)
    
    for i in range(path_length - 1):
        diff = coordinate_path[i + 1] - coordinate_path[i]
        velocity[i] = np.sqrt(diff[0]**2 + diff[1]**2)
    
    return velocity


@jit(nopython=True, cache=True)
def compute_turning_angles(coordinate_path: np.ndarray) -> np.ndarray:
    """Compute turning angles along coordinate path."""
    path_length = coordinate_path.shape[0]
    if path_length < 3:
        return np.zeros(1)
    
    angles = np.zeros(path_length - 2)
    
    for i in range(1, path_length - 1):
        v1 = coordinate_path[i] - coordinate_path[i - 1]
        v2 = coordinate_path[i + 1] - coordinate_path[i]
        
        # Calculate angle between vectors
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 > 1e-10 and mag2 > 1e-10:
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
            angles[i - 1] = np.arccos(cos_angle)
        else:
            angles[i - 1] = 0.0
    
    return angles


@jit(nopython=True, cache=True)
def detect_convergence_patterns(coordinate_path: np.ndarray, threshold: float = 0.1) -> Dict:
    """Detect coordinate path convergence patterns (returns to origin-like behavior)."""
    path_length = coordinate_path.shape[0]
    if path_length < 10:
        return {'has_convergence': False, 'convergence_points': 0}
    
    # Calculate distance from origin at each point
    distances = np.zeros(path_length)
    for i in range(path_length):
        distances[i] = np.sqrt(coordinate_path[i, 0]**2 + coordinate_path[i, 1]**2)
    
    # Find local minima (convergence points)
    convergence_count = 0
    min_distance = np.min(distances)
    
    for i in range(1, path_length - 1):
        if (distances[i] < distances[i - 1] and 
            distances[i] < distances[i + 1] and 
            distances[i] < min_distance + threshold):
            convergence_count += 1
    
    has_convergence = convergence_count > 0
    
    return {
        'has_convergence': has_convergence,
        'convergence_points': convergence_count,
        'min_distance': min_distance,
        'final_distance': distances[-1]
    }


@jit(nopython=True, cache=True)
def detect_loop_patterns(coordinate_path: np.ndarray, min_loop_size: int = 5) -> Dict:
    """Detect loop patterns in coordinate paths."""
    path_length = coordinate_path.shape[0]
    if path_length < min_loop_size * 2:
        return {'has_loops': False, 'loop_count': 0}
    
    loop_count = 0
    proximity_threshold = 0.5
    
    # Look for points that return close to previous positions
    for i in range(min_loop_size, path_length):
        current_pos = coordinate_path[i]
        
        # Check against previous positions
        for j in range(max(0, i - min_loop_size * 3), i - min_loop_size):
            prev_pos = coordinate_path[j]
            distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + 
                             (current_pos[1] - prev_pos[1])**2)
            
            if distance < proximity_threshold:
                loop_count += 1
                break  # Count only one loop per position
    
    return {
        'has_loops': loop_count > 0,
        'loop_count': loop_count,
        'loop_density': loop_count / path_length if path_length > 0 else 0.0
    }


class GenomicPatternExtractor:
    """High-performance genomic pattern extraction using coordinate geometry."""
    
    def __init__(self, min_pattern_length: int = 50):
        self.min_pattern_length = min_pattern_length
        self.performance_stats = {
            'sequences_processed': 0,
            'patterns_extracted': 0,
            'total_time': 0.0,
            'memory_peak': 0
        }
    
    def sequence_to_coordinates(self, sequence: str) -> np.ndarray:
        """Convert sequence to coordinate path."""
        cardinal_map = {'A': [0, 1], 'T': [0, -1], 'G': [1, 0], 'C': [-1, 0]}
        
        coordinates = []
        position = np.array([0.0, 0.0])
        
        for nucleotide in sequence.upper():
            if nucleotide in cardinal_map:
                position += np.array(cardinal_map[nucleotide])
                coordinates.append(position.copy())
        
        return np.array(coordinates)
    
    def extract_geometric_features(self, coordinate_path: np.ndarray) -> Dict:
        """Extract comprehensive geometric features from coordinate path."""
        if len(coordinate_path) < self.min_pattern_length:
            return {}
        
        # Basic geometric features
        curvature = compute_curvature_profile(coordinate_path)
        velocity = compute_velocity_profile(coordinate_path)
        turning_angles = compute_turning_angles(coordinate_path)
        
        # Pattern detection
        convergence_info = detect_convergence_patterns(coordinate_path)
        loop_info = detect_loop_patterns(coordinate_path)
        
        # Statistical features
        final_position = coordinate_path[-1]
        path_complexity = np.sum(velocity)
        coordinate_variance = np.var(coordinate_path, axis=0)
        
        # Advanced features
        mean_curvature = np.mean(curvature) if len(curvature) > 0 else 0.0
        max_curvature = np.max(curvature) if len(curvature) > 0 else 0.0
        curvature_std = np.std(curvature) if len(curvature) > 0 else 0.0
        
        mean_velocity = np.mean(velocity) if len(velocity) > 0 else 0.0
        velocity_variance = np.var(velocity) if len(velocity) > 0 else 0.0
        
        mean_turning_angle = np.mean(turning_angles) if len(turning_angles) > 0 else 0.0
        
        # Directionality analysis
        net_displacement = np.linalg.norm(final_position)
        displacement_efficiency = net_displacement / path_complexity if path_complexity > 0 else 0.0
        
        return {
            'mean_curvature': mean_curvature,
            'max_curvature': max_curvature,
            'curvature_std': curvature_std,
            'mean_velocity': mean_velocity,
            'velocity_variance': velocity_variance,
            'mean_turning_angle': mean_turning_angle,
            'path_complexity': path_complexity,
            'net_displacement': net_displacement,
            'displacement_efficiency': displacement_efficiency,
            'coordinate_variance_x': coordinate_variance[0],
            'coordinate_variance_y': coordinate_variance[1],
            'final_position_x': final_position[0],
            'final_position_y': final_position[1],
            **convergence_info,
            **loop_info
        }
    
    def classify_genomic_patterns(self, sequences: List[str], region_types: List[str]) -> Dict:
        """Classify genomic sequences based on their coordinate patterns."""
        results = {}
        
        for i, (sequence, region_type) in enumerate(zip(sequences, region_types)):
            if region_type not in results:
                results[region_type] = {
                    'sequences': [],
                    'features': [],
                    'coordinate_paths': []
                }
            
            # Convert to coordinates and extract features
            coord_path = self.sequence_to_coordinates(sequence)
            features = self.extract_geometric_features(coord_path)
            
            if features:  # Only store if features were extracted
                results[region_type]['sequences'].append(sequence)
                results[region_type]['features'].append(features)
                results[region_type]['coordinate_paths'].append(coord_path)
        
        return results
    
    def identify_functional_elements(self, sequences: List[str]) -> Dict:
        """
        Identify potential functional genomic elements based on geometric signatures.
        
        Based on theoretical framework:
        - Promoters: High curvature, convergence patterns
        - Coding sequences: Systematic displacement patterns
        - Regulatory elements: Loop structures, coordinate returns
        """
        results = {
            'promoter_candidates': [],
            'coding_candidates': [],
            'regulatory_candidates': [],
            'other_patterns': []
        }
        
        for sequence in sequences:
            coord_path = self.sequence_to_coordinates(sequence)
            features = self.extract_geometric_features(coord_path)
            
            if not features:
                continue
            
            # Classification based on geometric signatures
            if (features['mean_curvature'] > 2.3 and 
                features['has_convergence'] and 
                features['velocity_variance'] > 0.5):
                # High curvature + convergence = Promoter signature
                results['promoter_candidates'].append({
                    'sequence': sequence,
                    'features': features,
                    'confidence': min(1.0, features['mean_curvature'] / 5.0)
                })
            
            elif (features['displacement_efficiency'] > 0.3 and 
                  features['velocity_variance'] < 0.2 and 
                  features['mean_velocity'] > 0.8):
                # Systematic displacement = Coding signature
                results['coding_candidates'].append({
                    'sequence': sequence,
                    'features': features,
                    'confidence': features['displacement_efficiency']
                })
            
            elif (features['has_loops'] and 
                  features['convergence_points'] > 1 and 
                  features['loop_density'] > 0.05):
                # Loops + convergence = Regulatory signature
                results['regulatory_candidates'].append({
                    'sequence': sequence,
                    'features': features,
                    'confidence': min(1.0, features['loop_density'] * 10)
                })
            
            else:
                results['other_patterns'].append({
                    'sequence': sequence,
                    'features': features
                })
        
        return results
    
    def cross_domain_pattern_analysis(self, sequences: List[str]) -> Dict:
        """Extract patterns suitable for cross-domain transfer."""
        optimization_patterns = {
            'convergence_optimizers': [],
            'efficiency_patterns': [],
            'complexity_reducers': [],
            'stability_patterns': []
        }
        
        for sequence in sequences:
            coord_path = self.sequence_to_coordinates(sequence)
            features = self.extract_geometric_features(coord_path)
            
            if not features:
                continue
            
            # Patterns suitable for optimization transfer
            if features['has_convergence'] and features['displacement_efficiency'] > 0.8:
                optimization_patterns['convergence_optimizers'].append({
                    'pattern_type': 'convergence_optimizer',
                    'features': features,
                    'transferability_score': features['displacement_efficiency'] * (1 + features['convergence_points'])
                })
            
            if features['mean_velocity'] > 1.0 and features['velocity_variance'] < 0.1:
                optimization_patterns['efficiency_patterns'].append({
                    'pattern_type': 'efficiency_pattern',
                    'features': features,
                    'transferability_score': features['mean_velocity'] / (1 + features['velocity_variance'])
                })
            
            if features['path_complexity'] < 50 and features['net_displacement'] > 10:
                optimization_patterns['complexity_reducers'].append({
                    'pattern_type': 'complexity_reducer',
                    'features': features,
                    'transferability_score': features['net_displacement'] / features['path_complexity']
                })
            
            if features['curvature_std'] < 0.5 and features['mean_curvature'] > 0:
                optimization_patterns['stability_patterns'].append({
                    'pattern_type': 'stability_pattern',
                    'features': features,
                    'transferability_score': features['mean_curvature'] / (1 + features['curvature_std'])
                })
        
        return optimization_patterns
    
    def benchmark_pattern_extraction(self, genome_file: Path, test_sizes: List[int]) -> Dict:
        """Benchmark pattern extraction performance across different dataset sizes."""
        benchmark_results = {}
        
        for size in test_sizes:
            print(f"Benchmarking pattern extraction with {size} sequences...")
            
            # Load test sequences
            sequences = self.load_test_sequences(genome_file, size)
            
            start_time = time.time()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Extract patterns
            functional_elements = self.identify_functional_elements(sequences)
            cross_domain_patterns = self.cross_domain_pattern_analysis(sequences)
            
            end_time = time.time()
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Calculate metrics
            total_patterns = sum(len(patterns) for patterns in functional_elements.values())
            total_transfers = sum(len(patterns) for patterns in cross_domain_patterns.values())
            
            benchmark_results[size] = {
                'total_time': end_time - start_time,
                'sequences_per_second': size / (end_time - start_time) if (end_time - start_time) > 0 else 0,
                'patterns_extracted': total_patterns,
                'transfer_patterns': total_transfers,
                'memory_usage_mb': peak_memory - initial_memory,
                'extraction_efficiency': total_patterns / size if size > 0 else 0
            }
        
        return benchmark_results
    
    def load_test_sequences(self, genome_file: Path, count: int, sequence_length: int = 500) -> List[str]:
        """Load test sequences from genome file."""
        sequences = []
        
        with open(genome_file, 'r') as f:
            genome_data = ""
            for line in f:
                if not line.startswith('>'):
                    genome_data += line.strip().upper()
        
        # Extract sequences
        step_size = len(genome_data) // (count * 2)
        for i in range(0, len(genome_data) - sequence_length, step_size):
            if len(sequences) >= count:
                break
            
            sequence = genome_data[i:i + sequence_length]
            sequences.append(sequence)
        
        return sequences
    
    def save_results(self, results: Dict, output_file: Path):
        """Save pattern extraction results to JSON file."""
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {output_file}")


def main():
    """Main function for genomic pattern extraction."""
    parser = argparse.ArgumentParser(description="St. Stella's Genomic Pattern Extractor")
    parser.add_argument("--genome-file", type=Path, required=True, help="Path to genome FASTA file")
    parser.add_argument("--n-sequences", type=int, default=200, help="Number of sequences to analyze")
    parser.add_argument("--sequence-length", type=int, default=500, help="Length of sequences")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--extract-functional", action="store_true", 
                       help="Extract functional genomic elements")
    parser.add_argument("--cross-domain", action="store_true", 
                       help="Extract cross-domain transfer patterns")
    parser.add_argument("--output-dir", type=Path, default=Path("./results"),
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    extractor = GenomicPatternExtractor()
    
    print("St. Stella's Genomic Pattern Extractor")
    print("=" * 50)
    
    if args.benchmark:
        # Run performance benchmark
        test_sizes = [50, 100, 200, 500, 1000]
        benchmark_results = extractor.benchmark_pattern_extraction(args.genome_file, test_sizes)
        
        # Save benchmark results
        output_file = args.output_dir / "pattern_extraction_benchmark.json"
        extractor.save_results(benchmark_results, output_file)
        
        # Print benchmark summary
        print("\nBenchmark Results:")
        print("-" * 70)
        for size, metrics in benchmark_results.items():
            print(f"Size: {size:4d} | Time: {metrics['total_time']:.3f}s | "
                  f"Patterns: {metrics['patterns_extracted']:3d} | "
                  f"Efficiency: {metrics['extraction_efficiency']:.2f}")
    
    else:
        # Load sequences
        sequences = extractor.load_test_sequences(args.genome_file, args.n_sequences, args.sequence_length)
        
        results = {}
        
        if args.extract_functional:
            print("Extracting functional genomic elements...")
            functional_elements = extractor.identify_functional_elements(sequences)
            results['functional_elements'] = functional_elements
            
            # Print summary
            print(f"Functional Elements Found:")
            for element_type, candidates in functional_elements.items():
                print(f"  {element_type}: {len(candidates)} candidates")
        
        if args.cross_domain:
            print("Extracting cross-domain transfer patterns...")
            cross_domain_patterns = extractor.cross_domain_pattern_analysis(sequences)
            results['cross_domain_patterns'] = cross_domain_patterns
            
            # Print summary
            print(f"Cross-Domain Patterns Found:")
            for pattern_type, patterns in cross_domain_patterns.items():
                print(f"  {pattern_type}: {len(patterns)} patterns")
        
        if not args.extract_functional and not args.cross_domain:
            # Default: extract both
            functional_elements = extractor.identify_functional_elements(sequences)
            cross_domain_patterns = extractor.cross_domain_pattern_analysis(sequences)
            results = {
                'functional_elements': functional_elements,
                'cross_domain_patterns': cross_domain_patterns
            }
        
        # Save results
        output_file = args.output_dir / "genomic_patterns.json"
        extractor.save_results(results, output_file)


if __name__ == "__main__":
    main()
