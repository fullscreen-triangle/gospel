#!/usr/bin/env python3
"""
St. Stella's Sequence - Cardinal Direction Coordinate Transformation
High Performance Computing Implementation for Real Genomic Data

Cardinal Direction Mapping:
A → North (0, 1), T → South (0, -1), G → East (1, 0), C → West (-1, 0)

Performance Target: O(n) → O(log S₀) complexity reduction
"""

import numpy as np
import numba
from numba import jit, prange
import time
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import psutil


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
        
        coordinate_paths = cardinal_transform_batch(padded_sequences)
        
        # Update performance stats
        end_time = time.time()
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        self.performance_stats['sequences_processed'] += len(sequences)
        self.performance_stats['total_time'] += (end_time - start_time)
        self.performance_stats['memory_peak'] = max(self.performance_stats['memory_peak'], 
                                                  peak_memory - initial_memory)
        
        return coordinate_paths


def main():
    """Main function for coordinate transformation testing."""
    parser = argparse.ArgumentParser(description="St. Stella's Sequence Coordinate Transformation")
    parser.add_argument("--genome-file", type=Path, required=True, help="Path to genome FASTA file")
    parser.add_argument("--sequence-length", type=int, default=1000, help="Length of sequences")
    parser.add_argument("--n-sequences", type=int, default=1000, help="Number of sequences")
    
    args = parser.parse_args()
    
    transformer = StStellaSequenceTransformer()
    
    print("St. Stella's Sequence Coordinate Transformation")
    print("=" * 50)
    
    # Load and process sequences
    sequences = transformer.load_genome_sequences(args.genome_file, args.sequence_length, args.n_sequences)
    
    start_time = time.time()
    coordinate_paths = transformer.transform_sequences_batch(sequences)
    end_time = time.time()
    
    # Results
    total_time = end_time - start_time
    throughput = len(sequences) / total_time
    
    print(f"Processed {len(sequences)} sequences in {total_time:.3f}s")
    print(f"Throughput: {throughput:.1f} sequences/second")
    print(f"Memory peak: {transformer.performance_stats['memory_peak']:.1f} MB")
    print(f"Coordinate paths shape: {coordinate_paths.shape}")


if __name__ == "__main__":
    main()