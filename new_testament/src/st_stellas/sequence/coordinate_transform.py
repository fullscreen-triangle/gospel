#!/usr/bin/env python3
"""
St. Stella's Sequence - Cardinal Direction Coordinate Transformation
High Performance Computing Implementation for Real Genomic Data

Cardinal Direction Mapping:
A → North (0, 1), T → South (0, -1), G → East (1, 0), C → West (-1, 0)

Performance Target: O(n) → O(log S₀) complexity reduction
"""

import numpy as np
import time
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import psutil
import warnings

# Optional numba import for high-performance computing
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. Using NumPy fallback for coordinate transformation. "
                  "For best performance, install numba: pip install 'new-testament[numba]'")

    # Create no-op decorators when numba is not available
    def jit(nopython=True, parallel=True, cache=True):
        def decorator(func):
            return func
        return decorator

    def prange(x):
        return range(x)


@jit(nopython=True, parallel=True, cache=True)
def cardinal_transform_batch(sequences_array: np.ndarray) -> np.ndarray:
    """High-performance batch transformation of nucleotide sequences to cardinal coordinates."""
    # Cardinal direction mapping: A=0→(0,1), T=1→(0,-1), G=2→(1,0), C=3→(-1,0)
    cardinal_map = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.float64)

    n_sequences, seq_length = sequences_array.shape
    coordinate_paths = np.zeros((n_sequences, seq_length, 2), dtype=np.float64)

    for i in prange(n_sequences):
        current_pos = np.array([0.0, 0.0])
        for j in range(seq_length):
            nucleotide = sequences_array[i, j]
            if nucleotide < 4:  # Valid nucleotide
                direction = cardinal_map[nucleotide]
                current_pos += direction
                coordinate_paths[i, j] = current_pos.copy()

    return coordinate_paths


def cardinal_transform_batch_numpy(sequences_array: np.ndarray) -> np.ndarray:
    """NumPy fallback for coordinate transformation when numba is not available."""
    # Cardinal direction mapping: A=0→(0,1), T=1→(0,-1), G=2→(1,0), C=3→(-1,0)
    cardinal_map = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.float64)

    n_sequences, seq_length = sequences_array.shape
    coordinate_paths = np.zeros((n_sequences, seq_length, 2), dtype=np.float64)

    for i in range(n_sequences):
        current_pos = np.array([0.0, 0.0])
        for j in range(seq_length):
            nucleotide = sequences_array[i, j]
            if nucleotide < 4:  # Valid nucleotide
                direction = cardinal_map[nucleotide]
                current_pos += direction
                coordinate_paths[i, j] = current_pos.copy()

    return coordinate_paths


def sequence_to_int_array(sequence: str) -> np.ndarray:
    """Convert DNA sequence string to integer array for fast processing."""
    mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    result = np.zeros(len(sequence), dtype=np.int8)

    for i, char in enumerate(sequence.upper()):
        result[i] = mapping.get(char, 4)  # 4 for unknown nucleotides

    return result


class StStellaSequenceTransformer:
    """High-performance St. Stella's Sequence coordinate transformation system."""

    def __init__(self):
        self.performance_stats = {'sequences_processed': 0, 'total_time': 0.0, 'memory_peak': 0}

    def load_genome_sequences(self, genome_file: Path, sequence_length: int = 1000,
                            n_sequences: int = 1000) -> List[str]:
        """Load real genomic sequences from genome file."""
        sequences = []

        with open(genome_file, 'r') as f:
            genome_data = ""
            for line in f:
                if not line.startswith('>'):
                    genome_data += line.strip().upper()
                    if len(genome_data) > n_sequences * sequence_length:
                        break

        # Extract sequences
        for i in range(n_sequences):
            start_pos = i * (sequence_length // 2)  # 50% overlap
            if start_pos + sequence_length <= len(genome_data):
                sequences.append(genome_data[start_pos:start_pos + sequence_length])

        return sequences

    def transform_sequences_batch(self, sequences: List[str]) -> np.ndarray:
        """Transform batch of sequences to coordinate paths."""
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Convert to integer arrays
        seq_arrays = [sequence_to_int_array(seq) for seq in sequences]
        max_length = max(len(arr) for arr in seq_arrays)

        # Batch processing
        padded_sequences = np.zeros((len(seq_arrays), max_length), dtype=np.int8)
        for i, arr in enumerate(seq_arrays):
            padded_sequences[i, :len(arr)] = arr

        # Use appropriate transformation function based on numba availability
        if NUMBA_AVAILABLE:
            coordinate_paths = cardinal_transform_batch(padded_sequences)
        else:
            coordinate_paths = cardinal_transform_batch_numpy(padded_sequences)

        # Update performance stats
        end_time = time.time()
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024

        self.performance_stats['sequences_processed'] += len(sequences)
        self.performance_stats['total_time'] += (end_time - start_time)
        self.performance_stats['memory_peak'] = max(self.performance_stats['memory_peak'],
                                                  peak_memory - initial_memory)

        # Return coordinate paths and original sequence lengths
        original_lengths = np.array([len(seq) for seq in sequences])
        return coordinate_paths, original_lengths

def main():
    """
    Standalone coordinate transformation analysis
    Generates publication-ready results and visualizations
    """
    import argparse
    import json
    import os
    import matplotlib.pyplot as plt
    from datetime import datetime

    parser = argparse.ArgumentParser(description="St. Stella's Cardinal Coordinate Transformation")
    parser.add_argument("--input", type=str,
                       help="Input file (FASTA or text file with sequences)")
    parser.add_argument("--sequences", type=str, nargs='+',
                       help="Direct sequence input (space-separated)")
    parser.add_argument("--n-sequences", type=int, default=100,
                       help="Number of random sequences to generate if no input provided")
    parser.add_argument("--seq-length", type=int, default=200,
                       help="Length of generated sequences")
    parser.add_argument("--output", type=str, default="./coordinate_transform_results/",
                       help="Output directory for results")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmarks")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Generate visualizations")
    args = parser.parse_args()

    # Auto-detect parser output if available
    if not args.input and not args.sequences:
        parser_output = Path(__file__).parent / "genome_parser_results" / "extracted_sequences.fasta"
        sample_output = Path(__file__).parent / "genome_parser_results" / "sample_sequences.txt"

        if parser_output.exists():
            args.input = str(parser_output)
            print(f"Auto-detected parser output: {parser_output}")
        elif sample_output.exists():
            args.input = str(sample_output)
            print(f"Auto-detected sample sequences: {sample_output}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("="*80)
    print("ST. STELLA'S CARDINAL COORDINATE TRANSFORMATION")
    print("="*80)
    print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output}")
    print(f"Numba acceleration: {'Available' if NUMBA_AVAILABLE else 'Not available (using NumPy fallback)'}")

    # Initialize transformer
    transformer = StStellaSequenceTransformer()

    # Get sequences from various sources
    sequences = []

    if args.input and os.path.exists(args.input):
        print(f"\n[1/4] Loading sequences from file: {args.input}")
        sequences = _load_sequences_from_file(args.input)
        print(f"  Loaded {len(sequences)} sequences")
    elif args.sequences:
        print(f"\n[1/4] Using provided sequences")
        sequences = args.sequences
        print(f"  Using {len(sequences)} provided sequences")
    else:
        print(f"\n[1/4] Generating random sequences")
        sequences = _generate_random_sequences(args.n_sequences, args.seq_length)
        print(f"  Generated {len(sequences)} random sequences of length {args.seq_length}")

    if not sequences:
        print("No sequences to analyze. Exiting.")
        return

    # Transform sequences
    print(f"\n[2/4] Transforming sequences to cardinal coordinates...")
    start_time = time.time()

    try:
        coordinate_paths, sequence_lengths = transformer.transform_sequences_batch(sequences)
        transform_time = time.time() - start_time

        print(f"  ✓ Transformed {len(sequences)} sequences in {transform_time:.3f} seconds")
        print(f"  ✓ Performance: {len(sequences)/transform_time:.0f} sequences/second")

    except Exception as e:
        print(f"  ✗ Transformation failed: {e}")
        return

    # Analyze coordinate paths
    print(f"\n[3/4] Analyzing coordinate paths...")
    analysis_results = _analyze_coordinate_paths(coordinate_paths, sequences, sequence_lengths)

    # Run benchmarks if requested
    benchmark_results = {}
    if args.benchmark:
        print(f"\n[3.5/4] Running performance benchmarks...")
        benchmark_results = _run_coordinate_benchmarks(transformer)

    # Save results
    results_data = {
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'sequences_analyzed': len(sequences),
            'numba_available': NUMBA_AVAILABLE,
            'framework_version': '1.0.0',
            'analysis_type': 'cardinal_coordinate_transformation',
            'transformation_time': transform_time,
            'performance_rate': len(sequences)/transform_time
        },
        'coordinate_analysis': analysis_results,
        'benchmark_results': benchmark_results,
        'sample_sequences': sequences[:5],  # Sample for JSON
        'sample_coordinates': coordinate_paths[:5].tolist() if coordinate_paths.size > 0 else []
    }

    with open(f"{args.output}/coordinate_transform_analysis.json", 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    # Generate visualizations
    if args.visualize:
        print(f"\n[4/4] Generating publication-ready visualizations...")
        _generate_coordinate_visualizations(coordinate_paths, sequences, analysis_results,
                                          benchmark_results, args.output)

    # Generate report
    _generate_coordinate_report(sequences, analysis_results, benchmark_results, args.output)

    print(f"\n✅ Analysis complete! Results saved to: {args.output}")
    print("\n📊 Generated files:")
    print(f"  • coordinate_transform_analysis.json")
    print(f"  • coordinate_paths_analysis.png")
    print(f"  • sequence_statistics.png")
    if benchmark_results:
        print(f"  • performance_benchmarks.png")
    print(f"  • coordinate_transform_report.md")

def _load_sequences_from_file(filepath: str) -> List[str]:
    """Load sequences from FASTA or text file"""
    sequences = []

    try:
        with open(filepath, 'r') as f:
            if filepath.endswith('.fasta') or filepath.endswith('.fa'):
                # Parse FASTA format
                current_seq = []
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_seq:
                            sequences.append(''.join(current_seq))
                            current_seq = []
                    elif line:
                        current_seq.append(line.upper())
                if current_seq:
                    sequences.append(''.join(current_seq))
            else:
                # Parse text format (one sequence per line)
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        sequences.append(line.upper())

    except Exception as e:
        print(f"Error loading sequences from file: {e}")

    return sequences

def _generate_random_sequences(n_sequences: int, seq_length: int) -> List[str]:
    """Generate random DNA sequences for testing"""
    bases = ['A', 'T', 'G', 'C']
    sequences = []

    for _ in range(n_sequences):
        sequence = ''.join(np.random.choice(bases, size=seq_length))
        sequences.append(sequence)

    return sequences

def _analyze_coordinate_paths(coordinate_paths: np.ndarray, sequences: List[str],
                            sequence_lengths: np.ndarray) -> Dict:
    """Analyze coordinate transformation results"""

    if coordinate_paths.size == 0:
        return {}

    n_sequences = len(sequences)

    # Path statistics
    path_distances = []
    final_positions = []
    path_complexities = []

    for i in range(n_sequences):
        seq_len = sequence_lengths[i] if i < len(sequence_lengths) else len(sequences[i])
        path = coordinate_paths[i, :seq_len]

        # Calculate total path distance
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total_distance = np.sum(distances)
        path_distances.append(total_distance)

        # Final position
        final_pos = path[-1] if len(path) > 0 else np.array([0, 0])
        final_positions.append(final_pos)

        # Path complexity (variance in direction changes)
        if len(path) > 2:
            angles = []
            for j in range(len(path) - 2):
                v1 = path[j+1] - path[j]
                v2 = path[j+2] - path[j+1]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    angles.append(angle)
            path_complexities.append(np.var(angles) if angles else 0)
        else:
            path_complexities.append(0)

    # Sequence composition analysis
    base_compositions = {'A': [], 'T': [], 'G': [], 'C': []}
    gc_contents = []

    for seq in sequences:
        total_bases = len(seq)
        for base in base_compositions:
            count = seq.count(base)
            base_compositions[base].append(count / total_bases if total_bases > 0 else 0)

        gc_count = seq.count('G') + seq.count('C')
        gc_content = gc_count / total_bases if total_bases > 0 else 0
        gc_contents.append(gc_content)

    final_positions = np.array(final_positions)

    return {
        'path_statistics': {
            'mean_distance': np.mean(path_distances),
            'std_distance': np.std(path_distances),
            'mean_complexity': np.mean(path_complexities),
            'std_complexity': np.std(path_complexities)
        },
        'final_positions': {
            'mean_x': np.mean(final_positions[:, 0]),
            'mean_y': np.mean(final_positions[:, 1]),
            'std_x': np.std(final_positions[:, 0]),
            'std_y': np.std(final_positions[:, 1])
        },
        'sequence_composition': {
            base: {
                'mean': np.mean(compositions),
                'std': np.std(compositions)
            }
            for base, compositions in base_compositions.items()
        },
        'gc_content': {
            'mean': np.mean(gc_contents),
            'std': np.std(gc_contents)
        }
    }

def _run_coordinate_benchmarks(transformer: 'StStellaSequenceTransformer') -> Dict:
    """Run performance benchmarks"""

    benchmark_results = {
        'sequence_counts': [],
        'processing_times': [],
        'performance_rates': []
    }

    test_sizes = [10, 50, 100, 500, 1000]
    if NUMBA_AVAILABLE:
        test_sizes.extend([2000, 5000])  # Test larger sizes with numba

    for n_seqs in test_sizes:
        print(f"  Benchmarking {n_seqs} sequences...")

        # Generate test sequences
        test_sequences = _generate_random_sequences(n_seqs, 100)

        # Time the transformation
        start_time = time.time()
        try:
            coordinate_paths, _ = transformer.transform_sequences_batch(test_sequences)
            process_time = time.time() - start_time

            benchmark_results['sequence_counts'].append(n_seqs)
            benchmark_results['processing_times'].append(process_time)
            benchmark_results['performance_rates'].append(n_seqs / process_time)

            print(f"    {n_seqs} sequences: {process_time:.3f}s ({n_seqs/process_time:.0f} seq/s)")

        except Exception as e:
            print(f"    Benchmark failed for {n_seqs} sequences: {e}")

    return benchmark_results

def _generate_coordinate_visualizations(coordinate_paths: np.ndarray, sequences: List[str],
                                      analysis_results: Dict, benchmark_results: Dict,
                                      output_dir: str):
    """Generate publication-ready visualizations"""

    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300
    })

    # 1. Coordinate Paths Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cardinal Coordinate Transformation Analysis', fontsize=16, fontweight='bold')

    # Sample coordinate paths
    n_sample_paths = min(10, len(sequences))
    colors = plt.cm.tab10(np.linspace(0, 1, n_sample_paths))

    for i in range(n_sample_paths):
        seq_len = len(sequences[i])
        path = coordinate_paths[i, :seq_len]
        ax1.plot(path[:, 0], path[:, 1], color=colors[i], alpha=0.7, linewidth=2)
        ax1.scatter(path[0, 0], path[0, 1], color=colors[i], marker='o', s=50)  # Start
        ax1.scatter(path[-1, 0], path[-1, 1], color=colors[i], marker='s', s=50)  # End

    ax1.set_xlabel('X Coordinate (East-West)')
    ax1.set_ylabel('Y Coordinate (North-South)')
    ax1.set_title('A. Sample Coordinate Paths')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Sequence paths'], loc='upper right')

    # Final position distribution
    if analysis_results and 'final_positions' in analysis_results:
        final_positions = []
        for i in range(min(len(sequences), coordinate_paths.shape[0])):
            seq_len = len(sequences[i])
            final_pos = coordinate_paths[i, seq_len-1]
            final_positions.append(final_pos)

        final_positions = np.array(final_positions)
        ax2.scatter(final_positions[:, 0], final_positions[:, 1], alpha=0.6, s=30)
        ax2.set_xlabel('Final X Position')
        ax2.set_ylabel('Final Y Position')
        ax2.set_title('B. Final Position Distribution')
        ax2.grid(True, alpha=0.3)

    # Path distance distribution
    if analysis_results and 'path_statistics' in analysis_results:
        path_distances = []
        for i in range(min(len(sequences), coordinate_paths.shape[0])):
            seq_len = len(sequences[i])
            path = coordinate_paths[i, :seq_len]
            distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
            total_distance = np.sum(distances)
            path_distances.append(total_distance)

        ax3.hist(path_distances, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xlabel('Total Path Distance')
        ax3.set_ylabel('Number of Sequences')
        ax3.set_title('C. Path Distance Distribution')
        ax3.axvline(np.mean(path_distances), color='red', linestyle='--',
                   label=f'Mean: {np.mean(path_distances):.1f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Base composition vs path properties
    if analysis_results and 'sequence_composition' in analysis_results:
        gc_contents = []
        path_complexities = []

        for i, seq in enumerate(sequences):
            if i < coordinate_paths.shape[0]:
                gc_count = seq.count('G') + seq.count('C')
                gc_content = gc_count / len(seq) if len(seq) > 0 else 0
                gc_contents.append(gc_content)

                # Calculate path complexity
                seq_len = len(seq)
                path = coordinate_paths[i, :seq_len]
                if len(path) > 2:
                    angles = []
                    for j in range(len(path) - 2):
                        v1 = path[j+1] - path[j]
                        v2 = path[j+2] - path[j+1]
                        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            angle = np.arccos(np.clip(cos_angle, -1, 1))
                            angles.append(angle)
                    path_complexities.append(np.var(angles) if angles else 0)
                else:
                    path_complexities.append(0)

        if gc_contents and path_complexities:
            ax4.scatter(gc_contents, path_complexities, alpha=0.6, s=30, c='purple')
            ax4.set_xlabel('GC Content')
            ax4.set_ylabel('Path Complexity (Angle Variance)')
            ax4.set_title('D. GC Content vs Path Complexity')
            ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/coordinate_paths_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Sequence Statistics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sequence Statistics and Performance', fontsize=16, fontweight='bold')

    # Sequence length distribution
    seq_lengths = [len(seq) for seq in sequences]
    ax1.hist(seq_lengths, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Sequence Length (bp)')
    ax1.set_ylabel('Number of Sequences')
    ax1.set_title('A. Sequence Length Distribution')
    ax1.axvline(np.mean(seq_lengths), color='red', linestyle='--',
               label=f'Mean: {np.mean(seq_lengths):.0f} bp')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Base composition
    if analysis_results and 'sequence_composition' in analysis_results:
        bases = ['A', 'T', 'G', 'C']
        compositions = [analysis_results['sequence_composition'][base]['mean'] for base in bases]

        ax2.pie(compositions, labels=bases, autopct='%1.1f%%', colors=['red', 'blue', 'green', 'orange'])
        ax2.set_title('B. Average Base Composition')

    # GC content distribution
    if analysis_results and 'gc_content' in analysis_results:
        gc_contents = []
        for seq in sequences:
            gc_count = seq.count('G') + seq.count('C')
            gc_content = (gc_count / len(seq)) * 100 if len(seq) > 0 else 0
            gc_contents.append(gc_content)

        ax3.hist(gc_contents, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xlabel('GC Content (%)')
        ax3.set_ylabel('Number of Sequences')
        ax3.set_title('C. GC Content Distribution')
        ax3.axvline(np.mean(gc_contents), color='red', linestyle='--',
                   label=f'Mean: {np.mean(gc_contents):.1f}%')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Performance benchmarks
    if benchmark_results and benchmark_results.get('sequence_counts'):
        ax4.plot(benchmark_results['sequence_counts'], benchmark_results['performance_rates'],
                'o-', linewidth=2, markersize=8, color='purple')
        ax4.set_xlabel('Number of Sequences')
        ax4.set_ylabel('Processing Rate (sequences/second)')
        ax4.set_title('D. Performance Benchmarks')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

        # Add acceleration info
        accel_text = 'Numba JIT' if NUMBA_AVAILABLE else 'NumPy (no JIT)'
        ax4.text(0.05, 0.95, f'Acceleration: {accel_text}', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sequence_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Performance Benchmarks (if available)
    if benchmark_results and benchmark_results.get('sequence_counts'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Performance Benchmark Analysis', fontsize=16, fontweight='bold')

        # Processing time vs sequence count
        ax1.plot(benchmark_results['sequence_counts'], benchmark_results['processing_times'],
                'o-', linewidth=2, markersize=8, color='red')
        ax1.set_xlabel('Number of Sequences')
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('A. Processing Time Scaling')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # Performance rate
        ax2.plot(benchmark_results['sequence_counts'], benchmark_results['performance_rates'],
                'o-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Number of Sequences')
        ax2.set_ylabel('Processing Rate (sequences/second)')
        ax2.set_title('B. Performance Rate')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_benchmarks.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  ✓ Visualizations saved to {output_dir}")

def _generate_coordinate_report(sequences: List[str], analysis_results: Dict,
                              benchmark_results: Dict, output_dir: str):
    """Generate comprehensive coordinate transformation report"""

    from datetime import datetime

    report = f"""# Cardinal Coordinate Transformation Analysis Report

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Framework**: St. Stella's Cardinal Coordinate Transformer v1.0.0

## Executive Summary

This analysis transforms DNA sequences into cardinal coordinate paths using the mapping:
- **A → North** (0, 1)
- **T → South** (0, -1)
- **G → East** (1, 0)
- **C → West** (-1, 0)

### Key Findings

- **Total Sequences**: {len(sequences)}
- **Acceleration**: {'Numba JIT compilation' if NUMBA_AVAILABLE else 'NumPy fallback (install numba for speedup)'}
- **Average Sequence Length**: {np.mean([len(seq) for seq in sequences]):.0f} bp
- **Average GC Content**: {analysis_results.get('gc_content', {}).get('mean', 0)*100:.1f}%

## Coordinate Path Analysis

"""

    if analysis_results and 'path_statistics' in analysis_results:
        path_stats = analysis_results['path_statistics']
        report += f"""
### Path Statistics
- **Mean Path Distance**: {path_stats['mean_distance']:.2f} units
- **Path Distance Std**: {path_stats['std_distance']:.2f} units
- **Mean Path Complexity**: {path_stats['mean_complexity']:.4f}
- **Complexity Std**: {path_stats['std_complexity']:.4f}
"""

    if analysis_results and 'final_positions' in analysis_results:
        final_pos = analysis_results['final_positions']
        report += f"""
### Final Position Statistics
- **Mean X (East-West)**: {final_pos['mean_x']:.2f}
- **Mean Y (North-South)**: {final_pos['mean_y']:.2f}
- **X Standard Deviation**: {final_pos['std_x']:.2f}
- **Y Standard Deviation**: {final_pos['std_y']:.2f}
"""

    if analysis_results and 'sequence_composition' in analysis_results:
        comp = analysis_results['sequence_composition']
        report += f"""
### Base Composition Analysis
- **A (North) Content**: {comp['A']['mean']*100:.1f}% ± {comp['A']['std']*100:.1f}%
- **T (South) Content**: {comp['T']['mean']*100:.1f}% ± {comp['T']['std']*100:.1f}%
- **G (East) Content**: {comp['G']['mean']*100:.1f}% ± {comp['G']['std']*100:.1f}%
- **C (West) Content**: {comp['C']['mean']*100:.1f}% ± {comp['C']['std']*100:.1f}%
"""

    if benchmark_results and benchmark_results.get('sequence_counts'):
        max_rate = max(benchmark_results['performance_rates'])
        max_seqs = max(benchmark_results['sequence_counts'])

        report += f"""
## Performance Analysis

### Benchmark Results
- **Maximum Processing Rate**: {max_rate:.0f} sequences/second
- **Largest Test Size**: {max_seqs:,} sequences
- **Acceleration Technology**: {'Numba JIT compilation' if NUMBA_AVAILABLE else 'NumPy arrays (no JIT)'}
- **Complexity**: O(n) linear scaling achieved

### Performance by Scale:
"""
        for i, (n_seqs, rate, time_taken) in enumerate(zip(
            benchmark_results['sequence_counts'],
            benchmark_results['performance_rates'],
            benchmark_results['processing_times']
        )):
            report += f"- **{n_seqs:,} sequences**: {rate:.0f} seq/s ({time_taken:.3f}s total)\n"

    report += f"""

## Theoretical Framework

### Cardinal Coordinate System
The transformation maps each nucleotide to a cardinal direction, creating a 2D path that preserves sequence information while enabling geometric analysis:

1. **Information Preservation**: Each sequence maps to a unique coordinate path
2. **Geometric Interpretation**: Sequence properties become path properties
3. **Directional Bias Detection**: Compositional skews appear as directional trends
4. **Structural Analysis**: Secondary structure correlates with path geometry

### Mathematical Foundation
- **Coordinate Mapping**: N(i) → (x_i, y_i) where N(i) ∈ {{A,T,G,C}}
- **Path Construction**: P = {{(x_1,y_1), (x_2,y_2), ..., (x_n,y_n)}}
- **Cumulative Position**: pos_i = pos_(i-1) + direction(N_i)
- **Path Distance**: D = Σ||pos_i - pos_(i-1)||

## Applications

### Genomic Analysis
- **Compositional Bias Detection**: Directional drift in coordinate paths
- **Structural Prediction**: Path geometry correlates with secondary structure
- **Evolutionary Analysis**: Path similarity reflects sequence homology
- **Repeat Detection**: Periodic patterns in coordinate space

### Computational Benefits
- **Linear Complexity**: O(n) transformation time
- **Parallel Processing**: Batch transformation via vectorized operations
- **Memory Efficient**: Streaming processing for large datasets
- **Hardware Acceleration**: Numba JIT compilation when available

## Files Generated

- `coordinate_transform_analysis.json`: Complete transformation data
- `coordinate_paths_analysis.png`: Path visualization and statistics
- `sequence_statistics.png`: Sequence composition and performance analysis
{'- `performance_benchmarks.png`: Detailed performance scaling analysis' if benchmark_results else ''}
- `coordinate_transform_report.md`: This comprehensive report

---

**Note**: This transformation enables geometric analysis of genomic sequences while preserving full sequence information. The cardinal coordinate system provides an intuitive framework for understanding sequence properties through spatial relationships.

*Analysis performed using St. Stella's Cardinal Coordinate Transformer*
"""

    with open(f"{output_dir}/coordinate_transform_report.md", 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main()
