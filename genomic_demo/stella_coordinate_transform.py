"""
St. Stella's Genomic Coordinate Transformation
Layer 1: Transform DNA sequences to navigable S-entropy coordinates

Based on the revolutionary coordinate transformation:
A→North (0,1), T→South (0,-1), G→East (1,0), C→West (-1,0)
"""

import numpy as np
from typing import List, Tuple, Dict
import random

class StellaCoordinateTransformer:
    """Implements St. Stella's genomic coordinate transformation system"""
    
    def __init__(self):
        # Base coordinate mappings
        self.base_coordinates = {
            'A': np.array([0, 1]),   # North
            'T': np.array([0, -1]),  # South  
            'G': np.array([1, 0]),   # East
            'C': np.array([-1, 0])   # West
        }
        
    def transform_sequence_to_coordinates(self, sequence: str) -> np.ndarray:
        """Transform DNA sequence to spatial coordinates"""
        coordinates = np.array([0.0, 0.0])
        
        for base in sequence.upper():
            if base in self.base_coordinates:
                coordinates += self.base_coordinates[base]
                
        return coordinates
    
    def generate_s_entropy_coordinates(self, spatial_coords: np.ndarray, 
                                     sequence: str) -> Dict[str, float]:
        """Generate S-entropy coordinates from spatial coordinates"""
        
        # Extract S-entropy dimensions
        knowledge_content = self._calculate_information_content(spatial_coords, sequence)
        temporal_requirement = self._calculate_temporal_requirement(spatial_coords, sequence) 
        entropy_potential = self._calculate_optimization_potential(spatial_coords, sequence)
        
        return {
            'knowledge': knowledge_content,
            'time': temporal_requirement,
            'entropy': entropy_potential
        }
    
    def _calculate_information_content(self, coords: np.ndarray, sequence: str) -> float:
        """Calculate knowledge dimension from coordinate information"""
        # Information content based on coordinate magnitude and sequence complexity
        coord_magnitude = np.linalg.norm(coords)
        sequence_entropy = self._calculate_sequence_entropy(sequence)
        return coord_magnitude * sequence_entropy
    
    def _calculate_temporal_requirement(self, coords: np.ndarray, sequence: str) -> float:
        """Calculate time dimension from processing requirements"""
        # Temporal requirement based on coordinate path complexity
        coord_distance = np.linalg.norm(coords)
        sequence_length = len(sequence)
        return np.log(1 + coord_distance * sequence_length)
    
    def _calculate_optimization_potential(self, coords: np.ndarray, sequence: str) -> float:
        """Calculate entropy dimension from optimization potential"""
        # Optimization potential from coordinate structure
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        coord_complexity = abs(coords[0]) + abs(coords[1])
        return coord_complexity * (1 - gc_content)
    
    def _calculate_sequence_entropy(self, sequence: str) -> float:
        """Calculate Shannon entropy of sequence"""
        if not sequence:
            return 0.0
            
        counts = {}
        for base in sequence:
            counts[base] = counts.get(base, 0) + 1
            
        length = len(sequence)
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                prob = count / length
                entropy -= prob * np.log2(prob)
                
        return entropy

def generate_small_genome(length: int = 100) -> str:
    """Generate a small random genome sequence"""
    bases = ['A', 'T', 'G', 'C']
    return ''.join(random.choices(bases, k=length))

def demonstrate_coordinate_transformation():
    """Demonstrate the coordinate transformation process"""
    print("=== St. Stella's Coordinate Transformation Demo ===\n")
    
    transformer = StellaCoordinateTransformer()
    
    # Generate test sequences
    test_sequences = [
        "ATGCATGC",                    # Simple pattern
        "AAATTTGGGCCC",               # Repetitive  
        generate_small_genome(50),     # Random small genome
        generate_small_genome(100),    # Larger random genome
    ]
    
    print("Base Coordinate Mappings:")
    for base, coord in transformer.base_coordinates.items():
        print(f"  {base} → {coord} ({['', 'North', 'South', '', 'East', '', '', '', 'West'][coord[0]+1+3*(coord[1]+1)])})")
    print()
    
    for i, sequence in enumerate(test_sequences, 1):
        print(f"Sequence {i}: {sequence[:20]}{'...' if len(sequence) > 20 else ''}")
        print(f"  Length: {len(sequence)}")
        
        # Transform to spatial coordinates
        spatial_coords = transformer.transform_sequence_to_coordinates(sequence)
        print(f"  Spatial Coordinates: ({spatial_coords[0]:.2f}, {spatial_coords[1]:.2f})")
        
        # Generate S-entropy coordinates  
        s_coords = transformer.generate_s_entropy_coordinates(spatial_coords, sequence)
        print(f"  S-Entropy Coordinates:")
        print(f"    Knowledge: {s_coords['knowledge']:.4f}")
        print(f"    Time: {s_coords['time']:.4f}")
        print(f"    Entropy: {s_coords['entropy']:.4f}")
        print()

if __name__ == "__main__":
    demonstrate_coordinate_transformation()
