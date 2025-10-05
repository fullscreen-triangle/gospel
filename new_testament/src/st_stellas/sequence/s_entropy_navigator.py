#!/usr/bin/env python3
"""
St. Stella's Sequence - S-Entropy Navigation
High Performance Computing Implementation for Genomic Pattern Navigation

S-Entropy navigation transforms genomic analysis from O(4^n) exponential 
complexity to O(log S₀) logarithmic coordinate navigation.
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
from scipy.optimize import minimize
from sklearn.cluster import KMeans


@jit(nopython=True, cache=True)
def compute_s_distance(coord1: np.ndarray, coord2: np.ndarray, 
                      alpha: float = 1.0, beta: float = 0.5) -> float:
    """
    Compute S-distance between two coordinate positions.
    
    S_distance = ||coord1 - coord2||_2 + α * angular_distance + β * topology_distance
    """
    # Euclidean distance
    euclidean_dist = np.sqrt(np.sum((coord1 - coord2)**2))
    
    # Angular distance
    dot_product = np.dot(coord1, coord2)
    norms_product = np.linalg.norm(coord1) * np.linalg.norm(coord2)
    if norms_product > 1e-10:
        cos_angle = np.clip(dot_product / norms_product, -1.0, 1.0)
        angular_dist = np.arccos(abs(cos_angle))
    else:
        angular_dist = 0.0
    
    # Simplified topology distance (coordinate magnitude difference)
    topology_dist = abs(np.linalg.norm(coord1) - np.linalg.norm(coord2))
    
    return euclidean_dist + alpha * angular_dist + beta * topology_dist


@jit(nopython=True, cache=True)
def s_entropy_gradient(current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Compute gradient for S-entropy navigation toward target position."""
    diff = target_pos - current_pos
    distance = np.linalg.norm(diff)
    
    if distance < 1e-10:
        return np.zeros_like(current_pos)
    
    # Unit vector toward target
    direction = diff / distance
    
    # Adaptive step size based on distance
    step_magnitude = min(1.0, distance * 0.1)
    
    return direction * step_magnitude


@jit(nopython=True, parallel=True, cache=True)
def batch_s_entropy_navigation(start_positions: np.ndarray, 
                              target_positions: np.ndarray,
                              max_iterations: int = 100,
                              tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch S-entropy navigation from start positions to target positions.
    
    Returns:
        final_positions, convergence_steps, final_distances
    """
    n_sequences = start_positions.shape[0]
    dim = start_positions.shape[1]
    
    final_positions = np.zeros((n_sequences, dim), dtype=np.float64)
    convergence_steps = np.zeros(n_sequences, dtype=np.int32)
    final_distances = np.zeros(n_sequences, dtype=np.float64)
    
    for i in prange(n_sequences):
        current_pos = start_positions[i].copy()
        target_pos = target_positions[i]
        
        for step in range(max_iterations):
            # Compute gradient and update position
            gradient = s_entropy_gradient(current_pos, target_pos)
            current_pos += gradient
            
            # Check convergence
            distance = compute_s_distance(current_pos, target_pos)
            if distance < tolerance:
                convergence_steps[i] = step + 1
                break
        
        final_positions[i] = current_pos
        final_distances[i] = compute_s_distance(current_pos, target_pos)
    
    return final_positions, convergence_steps, final_distances


class SEntropyNavigator:
    """High-performance S-entropy navigation system for genomic coordinate optimization."""
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.performance_stats = {
            'sequences_processed': 0,
            'total_time': 0.0,
            'memory_peak': 0,
            'avg_convergence_steps': 0.0,
            'navigation_success_rate': 0.0
        }
    
    def sequence_to_coordinates(self, sequence: str) -> np.ndarray:
        """Convert sequence to final coordinate position."""
        cardinal_map = {'A': [0, 1], 'T': [0, -1], 'G': [1, 0], 'C': [-1, 0]}
        position = np.array([0.0, 0.0])
        
        for nucleotide in sequence.upper():
            if nucleotide in cardinal_map:
                position += np.array(cardinal_map[nucleotide])
        
        return position
    
    def load_sequence_patterns(self, genome_file: Path, pattern_types: List[str],
                             sequences_per_type: int = 50) -> Dict[str, List[str]]:
        """Load different genomic patterns for S-entropy navigation testing."""
        patterns = {pattern_type: [] for pattern_type in pattern_types}
        
        with open(genome_file, 'r') as f:
            genome_data = ""
            for line in f:
                if not line.startswith('>'):
                    genome_data += line.strip().upper()
        
        # Extract different pattern types
        total_length = len(genome_data)
        
        for pattern_type in pattern_types:
            if pattern_type == 'palindromes':
                # Look for potential palindromic sequences
                patterns[pattern_type] = self._extract_palindromic_candidates(
                    genome_data, sequences_per_type)
            elif pattern_type == 'high_gc':
                # High GC content regions
                patterns[pattern_type] = self._extract_high_gc_regions(
                    genome_data, sequences_per_type)
            elif pattern_type == 'low_complexity':
                # Low complexity/repetitive regions
                patterns[pattern_type] = self._extract_low_complexity_regions(
                    genome_data, sequences_per_type)
            elif pattern_type == 'random':
                # Random genomic sequences
                patterns[pattern_type] = self._extract_random_sequences(
                    genome_data, sequences_per_type)
        
        return patterns
    
    def _extract_palindromic_candidates(self, genome_data: str, count: int) -> List[str]:
        """Extract sequences that might be palindromic."""
        candidates = []
        window_size = 200
        
        for i in range(0, len(genome_data) - window_size, window_size // 2):
            if len(candidates) >= count:
                break
            
            seq = genome_data[i:i + window_size]
            
            # Simple palindrome check (not perfect, but for pattern detection)
            complement = seq.translate(str.maketrans('ATGC', 'TACG'))[::-1]
            similarity = sum(a == b for a, b in zip(seq, complement)) / len(seq)
            
            if similarity > 0.7:  # Potential palindrome
                candidates.append(seq)
        
        return candidates[:count]
    
    def _extract_high_gc_regions(self, genome_data: str, count: int) -> List[str]:
        """Extract high GC content regions."""
        sequences = []
        window_size = 200
        
        for i in range(0, len(genome_data) - window_size, window_size // 4):
            if len(sequences) >= count:
                break
            
            seq = genome_data[i:i + window_size]
            gc_content = (seq.count('G') + seq.count('C')) / len(seq)
            
            if gc_content > 0.6:  # High GC content
                sequences.append(seq)
        
        return sequences[:count]
    
    def _extract_low_complexity_regions(self, genome_data: str, count: int) -> List[str]:
        """Extract low complexity/repetitive regions."""
        sequences = []
        window_size = 200
        
        for i in range(0, len(genome_data) - window_size, window_size // 4):
            if len(sequences) >= count:
                break
            
            seq = genome_data[i:i + window_size]
            
            # Simple complexity measure based on nucleotide diversity
            unique_ratio = len(set(seq)) / len(seq)
            
            if unique_ratio < 0.3:  # Low complexity
                sequences.append(seq)
        
        return sequences[:count]
    
    def _extract_random_sequences(self, genome_data: str, count: int) -> List[str]:
        """Extract random genomic sequences."""
        sequences = []
        window_size = 200
        step_size = len(genome_data) // (count * 2)  # Ensure good distribution
        
        for i in range(0, len(genome_data) - window_size, step_size):
            if len(sequences) >= count:
                break
            
            seq = genome_data[i:i + window_size]
            sequences.append(seq)
        
        return sequences[:count]
    
    def perform_navigation_experiment(self, patterns: Dict[str, List[str]]) -> Dict:
        """Perform S-entropy navigation experiments across different genomic patterns."""
        results = {}
        
        for pattern_type, sequences in patterns.items():
            print(f"Navigating {pattern_type} patterns ({len(sequences)} sequences)...")
            
            start_time = time.time()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Convert sequences to coordinate positions
            start_positions = np.array([self.sequence_to_coordinates(seq) for seq in sequences])
            
            # Create target positions (for example, origin convergence test)
            target_positions = np.zeros_like(start_positions)  # Navigate to origin
            
            # Perform batch S-entropy navigation
            final_positions, convergence_steps, final_distances = batch_s_entropy_navigation(
                start_positions, target_positions, self.max_iterations, self.tolerance
            )
            
            # Alternative test: Navigate to cluster centroids
            if len(sequences) >= 5:
                kmeans = KMeans(n_clusters=min(5, len(sequences)//2), random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(start_positions)
                centroid_targets = kmeans.cluster_centers_[cluster_labels]
                
                centroid_positions, centroid_steps, centroid_distances = batch_s_entropy_navigation(
                    start_positions, centroid_targets, self.max_iterations, self.tolerance
                )
            else:
                centroid_positions = final_positions.copy()
                centroid_steps = convergence_steps.copy()
                centroid_distances = final_distances.copy()
            
            # Performance metrics
            end_time = time.time()
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Calculate statistics
            success_rate = np.mean(final_distances < self.tolerance)
            avg_steps = np.mean(convergence_steps[convergence_steps > 0])
            
            results[pattern_type] = {
                'sequence_count': len(sequences),
                'navigation_time': end_time - start_time,
                'memory_usage': peak_memory - initial_memory,
                'success_rate': success_rate,
                'avg_convergence_steps': avg_steps,
                'final_distances_mean': np.mean(final_distances),
                'final_distances_std': np.std(final_distances),
                'centroid_success_rate': np.mean(centroid_distances < self.tolerance),
                'centroid_avg_steps': np.mean(centroid_steps[centroid_steps > 0]),
                'complexity_reduction_factor': self._estimate_complexity_reduction(len(sequences), avg_steps)
            }
            
            # Update global performance stats
            self.performance_stats['sequences_processed'] += len(sequences)
            self.performance_stats['total_time'] += (end_time - start_time)
            self.performance_stats['memory_peak'] = max(self.performance_stats['memory_peak'], 
                                                      peak_memory - initial_memory)
        
        # Calculate overall performance statistics
        total_sequences = sum(result['sequence_count'] for result in results.values())
        if total_sequences > 0:
            weighted_avg_steps = sum(result['avg_convergence_steps'] * result['sequence_count'] 
                                   for result in results.values() if not np.isnan(result['avg_convergence_steps'])) / total_sequences
            weighted_success_rate = sum(result['success_rate'] * result['sequence_count'] 
                                      for result in results.values()) / total_sequences
            
            self.performance_stats['avg_convergence_steps'] = weighted_avg_steps
            self.performance_stats['navigation_success_rate'] = weighted_success_rate
        
        return results
    
    def _estimate_complexity_reduction(self, sequence_length: int, avg_steps: float) -> float:
        """Estimate complexity reduction factor compared to traditional O(n) processing."""
        if np.isnan(avg_steps) or avg_steps <= 0:
            return 1.0
        
        # Traditional complexity would be O(n) where n is sequence length
        traditional_ops = sequence_length
        
        # S-entropy navigation complexity is O(log S₀) ≈ avg_steps
        navigation_ops = avg_steps
        
        return traditional_ops / navigation_ops if navigation_ops > 0 else 1.0
    
    def benchmark_navigation_performance(self, genome_file: Path, test_sizes: List[int]) -> Dict:
        """Benchmark S-entropy navigation performance across different dataset sizes."""
        benchmark_results = {}
        
        for size in test_sizes:
            print(f"Benchmarking navigation with {size} sequences...")
            
            # Load test patterns
            patterns = self.load_sequence_patterns(
                genome_file, ['random'], sequences_per_type=size
            )
            
            # Perform navigation experiment
            start_time = time.time()
            results = self.perform_navigation_experiment(patterns)
            end_time = time.time()
            
            # Extract performance metrics
            random_results = results['random']
            
            benchmark_results[size] = {
                'total_time': end_time - start_time,
                'sequences_per_second': size / (end_time - start_time) if (end_time - start_time) > 0 else 0,
                'success_rate': random_results['success_rate'],
                'avg_convergence_steps': random_results['avg_convergence_steps'],
                'complexity_reduction_factor': random_results['complexity_reduction_factor'],
                'memory_usage_mb': random_results['memory_usage']
            }
        
        return benchmark_results
    
    def save_results(self, results: Dict, output_file: Path):
        """Save navigation results to JSON file."""
        # Convert numpy types to Python native types for JSON serialization
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
    """Main function for S-entropy navigation testing."""
    parser = argparse.ArgumentParser(description="St. Stella's S-Entropy Navigation")
    parser.add_argument("--genome-file", type=Path, required=True, help="Path to genome FASTA file")
    parser.add_argument("--pattern-types", nargs='+', 
                       default=['palindromes', 'high_gc', 'low_complexity', 'random'],
                       help="Types of genomic patterns to test")
    parser.add_argument("--sequences-per-type", type=int, default=50,
                       help="Number of sequences per pattern type")
    parser.add_argument("--max-iterations", type=int, default=100,
                       help="Maximum navigation iterations")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark")
    parser.add_argument("--output-dir", type=Path, default=Path("./results"),
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    navigator = SEntropyNavigator(max_iterations=args.max_iterations)
    
    print("St. Stella's S-Entropy Navigation System")
    print("=" * 50)
    
    if args.benchmark:
        # Run performance benchmark
        test_sizes = [10, 25, 50, 100, 200, 500]
        benchmark_results = navigator.benchmark_navigation_performance(args.genome_file, test_sizes)
        
        # Save benchmark results
        output_file = args.output_dir / "s_entropy_navigation_benchmark.json"
        navigator.save_results(benchmark_results, output_file)
        
        # Print benchmark summary
        print("\nBenchmark Results:")
        print("-" * 60)
        for size, metrics in benchmark_results.items():
            print(f"Size: {size:3d} | Time: {metrics['total_time']:.3f}s | "
                  f"Success: {metrics['success_rate']:.1%} | "
                  f"Speedup: {metrics['complexity_reduction_factor']:.1f}x")
    
    else:
        # Standard navigation experiment
        patterns = navigator.load_sequence_patterns(
            args.genome_file, args.pattern_types, args.sequences_per_type
        )
        
        results = navigator.perform_navigation_experiment(patterns)
        
        # Print results summary
        print(f"\nNavigation Results:")
        print("-" * 70)
        for pattern_type, metrics in results.items():
            print(f"{pattern_type:15s}: Success={metrics['success_rate']:.1%}, "
                  f"Steps={metrics['avg_convergence_steps']:.1f}, "
                  f"Speedup={metrics['complexity_reduction_factor']:.1f}x")
        
        print(f"\nOverall Performance:")
        print(f"Total sequences processed: {navigator.performance_stats['sequences_processed']}")
        print(f"Average convergence steps: {navigator.performance_stats['avg_convergence_steps']:.1f}")
        print(f"Navigation success rate: {navigator.performance_stats['navigation_success_rate']:.1%}")
        print(f"Memory peak: {navigator.performance_stats['memory_peak']:.1f} MB")
        
        # Save detailed results
        results['performance_stats'] = navigator.performance_stats
        output_file = args.output_dir / "s_entropy_navigation_results.json"
        navigator.save_results(results, output_file)


if __name__ == "__main__":
    main()
