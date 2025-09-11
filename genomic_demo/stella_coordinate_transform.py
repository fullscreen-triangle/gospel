"""
St. Stella's Genomic Coordinate Transformation
Layer 1: Transform DNA sequences to navigable S-entropy coordinates

Based on the revolutionary coordinate transformation:
A→North (0,1), T→South (0,-1), G→East (1,0), C→West (-1,0)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random
import os

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
    
    def plot_coordinate_path(self, coordinates: np.ndarray, sequence: str, filename: str = "coordinate_path.png"):
        """Plot the coordinate path for visualization"""
        plt.figure(figsize=(12, 8))
        
        # Plot the path
        plt.plot(coordinates[:, 0], coordinates[:, 1], 'b-', linewidth=2, alpha=0.7, label='DNA Path')
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c=range(len(coordinates)), 
                   cmap='viridis', s=50, alpha=0.8, label='Base Positions')
        
        # Mark start and end
        plt.scatter(coordinates[0, 0], coordinates[0, 1], c='green', s=100, marker='o', label='Start')
        plt.scatter(coordinates[-1, 0], coordinates[-1, 1], c='red', s=100, marker='s', label='End')
        
        # Add some sequence annotations
        for i in range(0, len(coordinates), max(1, len(coordinates)//10)):
            if i < len(sequence):
                plt.annotate(sequence[i], (coordinates[i, 0], coordinates[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('X Coordinate (East-West)')
        plt.ylabel('Y Coordinate (North-South)')
        plt.title(f'St. Stella\'s Coordinate Transformation\nSequence: {sequence[:30]}{"..." if len(sequence) > 30 else ""}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Save the plot
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
        print(f"Coordinate path visualization saved to: outputs/{filename}")
        plt.close()

def generate_small_genome(length: int = 100) -> str:
    """Generate a small random genome sequence"""
    bases = ['A', 'T', 'G', 'C']
    return ''.join(random.choices(bases, k=length))

def demonstrate_coordinate_transformation():
    """Demonstrate the coordinate transformation process with visualizations"""
    print("=== St. Stella's Coordinate Transformation Demo ===\n")
    
    transformer = StellaCoordinateTransformer()
    
    # Generate test sequences
    test_sequences = [
        ("Simple Pattern", "ATGCATGC"),
        ("Repetitive", "AAATTTGGGCCC"),
        ("Random Small", generate_small_genome(50)),
        ("Random Large", generate_small_genome(100)),
    ]
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    print("Base Coordinate Mappings:")
    for base, coord in transformer.base_coordinates.items():
        direction_map = {(-1, 0): 'West', (1, 0): 'East', (0, 1): 'North', (0, -1): 'South'}
        direction = direction_map.get(tuple(coord), 'Unknown')
        print(f"  {base} -> {coord} ({direction})")
    print()
    
    # Store results for summary
    results = []
    
    for i, (name, sequence) in enumerate(test_sequences, 1):
        print(f"Processing Sequence {i} ({name}): {sequence[:20]}{'...' if len(sequence) > 20 else ''}")
        print(f"  Length: {len(sequence)}")
        
        # Transform to path coordinates for visualization
        coordinates_path = []
        current_pos = np.array([0.0, 0.0])
        coordinates_path.append(current_pos.copy())
        
        for base in sequence:
            if base in transformer.base_coordinates:
                current_pos += transformer.base_coordinates[base]
                coordinates_path.append(current_pos.copy())
        
        coordinates_path = np.array(coordinates_path)
        
        # Transform to final spatial coordinates
        spatial_coords = transformer.transform_sequence_to_coordinates(sequence)
        print(f"  Final Spatial Coordinates: ({spatial_coords[0]:.2f}, {spatial_coords[1]:.2f})")
        
        # Generate S-entropy coordinates  
        s_coords = transformer.generate_s_entropy_coordinates(spatial_coords, sequence)
        print(f"  S-Entropy Coordinates:")
        print(f"    Knowledge: {s_coords['knowledge']:.4f}")
        print(f"    Time: {s_coords['time']:.4f}")
        print(f"    Entropy: {s_coords['entropy']:.4f}")
        
        # Create visualization
        filename = f"sequence_{i}_{name.lower().replace(' ', '_')}_transformation.png"
        transformer.plot_coordinate_path(coordinates_path, sequence, filename)
        
        # Store results
        results.append({
            'name': name,
            'sequence': sequence,
            'length': len(sequence),
            'final_coords': spatial_coords.tolist(),
            's_entropy': s_coords,
            'plot_file': filename
        })
        print()
    
    # Save summary results
    import json
    with open('outputs/coordinate_transformation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison visualization
    create_comparison_plot(results)
    
    print(f"All results and visualizations saved to 'outputs/' directory:")
    print(f"  - Individual sequence plots: {len(results)} files") 
    print(f"  - Summary data: coordinate_transformation_results.json")
    print(f"  - Comparison plot: coordinate_comparison.png")

def create_comparison_plot(results):
    """Create a comparison plot of all sequences"""
    plt.figure(figsize=(15, 10))
    
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, result in enumerate(results):
        coords = np.array(result['final_coords'])
        plt.scatter(coords[0], coords[1], c=colors[i % len(colors)], 
                   s=result['length']*2, alpha=0.7, 
                   label=f"{result['name']} (L={result['length']})")
    
    plt.xlabel('X Coordinate (East-West)')
    plt.ylabel('Y Coordinate (North-South)')
    plt.title('St. Stella\'s Coordinate Transformation - Sequence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save comparison plot
    plt.savefig('outputs/coordinate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    demonstrate_coordinate_transformation()
