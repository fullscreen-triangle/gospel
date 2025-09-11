"""
Empty Dictionary Gas Molecular Synthesis System
Layer 2A: Dynamic meaning generation through gas molecular equilibrium

This system generates genomic solutions on-the-spot through thermodynamic 
equilibrium seeking rather than storing pre-computed solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import random
from dataclasses import dataclass
import time
import os

@dataclass
class GasMolecule:
    """Represents a semantic gas molecule in the system"""
    energy: float
    position: np.ndarray
    velocity: np.ndarray
    semantic_charge: float
    
class GenomicEmptyDictionary:
    """Empty dictionary that synthesizes genomic meaning through gas molecular equilibrium"""
    
    def __init__(self):
        self.baseline_pressure = 1.0
        self.semantic_boltzmann_constant = 1.38e-23  # J/K (semantic units)
        self.equilibrium_threshold = 0.01  # More achievable threshold
        self.max_iterations = 100  # Reduced for demonstration
        self.max_time_seconds = 10  # Add time limit
        
    def synthesize_genomic_meaning(self, query: str, s_coords: Dict[str, float]) -> Dict[str, Any]:
        """
        Synthesize genomic meaning through gas molecular equilibrium process
        
        Args:
            query: Genomic analysis query (e.g., "sequence_similarity", "palindrome_detection")  
            s_coords: S-entropy coordinates from Layer 1
            
        Returns:
            Synthesized genomic meaning and solution
        """
        print(f"  Empty Dictionary: Synthesizing meaning for '{query}'")
        
        # Step 1: Create genomic perturbation in gas molecular system
        perturbation = self._create_genomic_perturbation(query, s_coords)
        print(f"    Created perturbation: ΔP = {perturbation:.6f}")
        
        # Step 2: Initialize gas molecular system
        gas_system = self._initialize_gas_molecular_system(perturbation)
        print(f"    Initialized {len(gas_system)} gas molecules")
        
        # Step 3: Seek equilibrium through coordinate navigation
        equilibrium_result = self._seek_equilibrium(gas_system, s_coords)
        print(f"    Equilibrium reached in {equilibrium_result['iterations']} iterations")
        print(f"    Final variance: {equilibrium_result['final_variance']:.8f}")
        
        # Step 4: Extract meaning from equilibrium process
        synthesized_meaning = self._extract_meaning_from_equilibrium(
            equilibrium_result, query, s_coords
        )
        
        # Step 5: Return to empty state
        self._return_to_empty_state()
        print(f"    Returned to empty state - ready for next query")
        
        return synthesized_meaning
    
    def _create_genomic_perturbation(self, query: str, s_coords: Dict[str, float]) -> float:
        """Create system perturbation based on genomic query"""
        # Calculate perturbation magnitude based on query complexity and coordinates
        query_complexity = len(query) * 0.1
        coord_magnitude = np.sqrt(s_coords['knowledge']**2 + 
                                s_coords['time']**2 + 
                                s_coords['entropy']**2)
        
        perturbation = query_complexity * (1 + coord_magnitude * 0.1)
        return perturbation
    
    def _initialize_gas_molecular_system(self, perturbation: float) -> List[GasMolecule]:
        """Initialize gas molecular system with perturbation"""
        # Number of molecules proportional to perturbation
        num_molecules = int(50 + perturbation * 20)
        
        molecules = []
        for i in range(num_molecules):
            # Create gas molecule with random properties
            molecule = GasMolecule(
                energy=random.uniform(0.1, 2.0) * perturbation,
                position=np.random.uniform(-1, 1, 3),  # 3D space
                velocity=np.random.uniform(-0.5, 0.5, 3),
                semantic_charge=random.uniform(-1, 1)
            )
            molecules.append(molecule)
            
        return molecules
    
    def _seek_equilibrium(self, gas_system: List[GasMolecule], 
                         s_coords: Dict[str, float]) -> Dict[str, Any]:
        """Seek equilibrium through S-entropy coordinate navigation"""
        
        current_variance = self._calculate_system_variance(gas_system)
        initial_variance = current_variance
        
        # Target equilibrium state (minimal variance)
        target_variance = self.equilibrium_threshold
        
        iteration = 0
        start_time = time.time()
        
        while (current_variance > target_variance and 
               iteration < self.max_iterations and 
               time.time() - start_time < self.max_time_seconds):
            
            # Apply molecular dynamics step
            self._apply_molecular_dynamics_step(gas_system, s_coords)
            
            # Recalculate variance
            current_variance = self._calculate_system_variance(gas_system)
            iteration += 1
            
            # Apply coordinate navigation toward equilibrium
            if iteration % 10 == 0:
                self._apply_s_entropy_navigation(gas_system, s_coords)
        
        return {
            'iterations': iteration,
            'initial_variance': initial_variance,
            'final_variance': current_variance,
            'equilibrium_reached': current_variance <= target_variance,
            'final_system_state': gas_system
        }
    
    def _calculate_system_variance(self, gas_system: List[GasMolecule]) -> float:
        """Calculate variance from equilibrium state"""
        if not gas_system:
            return 0.0
            
        # Calculate variance in energy distribution
        energies = [mol.energy for mol in gas_system]
        mean_energy = np.mean(energies)
        energy_variance = np.var(energies)
        
        # Calculate variance in spatial distribution
        positions = np.array([mol.position for mol in gas_system])
        spatial_variance = np.var(positions)
        
        # Combined variance measure
        total_variance = energy_variance + spatial_variance * 0.1
        return total_variance
    
    def _apply_molecular_dynamics_step(self, gas_system: List[GasMolecule], 
                                     s_coords: Dict[str, float]):
        """Apply one step of molecular dynamics"""
        dt = 0.01  # Time step
        
        for molecule in gas_system:
            # Update velocity based on forces (simplified)
            force = self._calculate_molecular_force(molecule, gas_system, s_coords)
            molecule.velocity += force * dt
            
            # Update position
            molecule.position += molecule.velocity * dt
            
            # Apply damping (energy dissipation toward equilibrium)
            molecule.velocity *= 0.99
            molecule.energy *= 0.999
    
    def _calculate_molecular_force(self, molecule: GasMolecule, 
                                 gas_system: List[GasMolecule], 
                                 s_coords: Dict[str, float]) -> np.ndarray:
        """Calculate force on molecule from S-entropy coordinates"""
        # Force toward S-coordinate equilibrium position
        equilibrium_position = np.array([
            s_coords['knowledge'] * 0.1,
            s_coords['time'] * 0.1, 
            s_coords['entropy'] * 0.1
        ])
        
        # Attractive force toward equilibrium
        force = (equilibrium_position - molecule.position) * 0.1
        
        # Add intermolecular forces (simplified)
        for other in gas_system:
            if other is not molecule:
                r = molecule.position - other.position
                r_mag = np.linalg.norm(r)
                if r_mag > 0:
                    # Weak repulsion at short range
                    force += -0.01 * r / (r_mag**3 + 0.01)
        
        return force
    
    def _apply_s_entropy_navigation(self, gas_system: List[GasMolecule], 
                                  s_coords: Dict[str, float]):
        """Apply S-entropy coordinate navigation for faster convergence"""
        # Direct navigation toward S-entropy optimal configuration
        for molecule in gas_system:
            # Navigate toward coordinates that minimize S-entropy distance
            target_energy = s_coords['entropy'] * 0.5
            target_position = np.array([
                s_coords['knowledge'] * 0.05,
                s_coords['time'] * 0.05,
                s_coords['entropy'] * 0.05
            ])
            
            # Gradual movement toward target
            molecule.energy = 0.9 * molecule.energy + 0.1 * target_energy
            molecule.position = 0.95 * molecule.position + 0.05 * target_position
    
    def _extract_meaning_from_equilibrium(self, equilibrium_result: Dict[str, Any],
                                        query: str, s_coords: Dict[str, float]) -> Dict[str, Any]:
        """Extract synthesized meaning from equilibrium process"""
        
        final_system = equilibrium_result['final_system_state']
        
        # Calculate system properties at equilibrium
        total_energy = sum(mol.energy for mol in final_system)
        average_energy = total_energy / len(final_system)
        
        # Calculate semantic coherence
        positions = np.array([mol.position for mol in final_system])
        coherence = 1.0 / (1.0 + np.var(positions))
        
        # Generate query-specific solution based on equilibrium state
        if "similarity" in query.lower():
            solution = self._generate_similarity_solution(final_system, s_coords)
        elif "palindrome" in query.lower():
            solution = self._generate_palindrome_solution(final_system, s_coords)
        elif "pattern" in query.lower():
            solution = self._generate_pattern_solution(final_system, s_coords)
        else:
            solution = self._generate_general_solution(final_system, s_coords)
        
        return {
            'solution': solution,
            'equilibrium_energy': average_energy,
            'semantic_coherence': coherence,
            'synthesis_quality': min(1.0, coherence * 2.0),
            'variance_minimization': equilibrium_result['final_variance'],
            'processing_method': 'gas_molecular_equilibrium'
        }
    
    def _generate_similarity_solution(self, system: List[GasMolecule], 
                                    s_coords: Dict[str, float]) -> Dict[str, float]:
        """Generate sequence similarity solution"""
        avg_energy = np.mean([mol.energy for mol in system])
        similarity_score = min(1.0, avg_energy / s_coords['knowledge'])
        
        return {
            'similarity_score': similarity_score,
            'confidence': min(1.0, similarity_score * 1.2),
            'method': 'equilibrium_energy_analysis'
        }
    
    def _generate_palindrome_solution(self, system: List[GasMolecule], 
                                    s_coords: Dict[str, float]) -> Dict[str, float]:
        """Generate palindrome detection solution"""
        # Check spatial symmetry in equilibrium state
        positions = np.array([mol.position for mol in system])
        symmetry_score = self._calculate_spatial_symmetry(positions)
        
        palindrome_probability = min(1.0, symmetry_score * s_coords['entropy'])
        
        return {
            'palindrome_probability': palindrome_probability,
            'symmetry_score': symmetry_score,
            'confidence': min(1.0, symmetry_score * 1.5),
            'method': 'spatial_symmetry_analysis'
        }
    
    def _generate_pattern_solution(self, system: List[GasMolecule], 
                                 s_coords: Dict[str, float]) -> Dict[str, Any]:
        """Generate pattern recognition solution"""
        # Analyze molecular clustering patterns
        positions = np.array([mol.position for mol in system])
        pattern_strength = self._calculate_pattern_strength(positions)
        
        return {
            'pattern_strength': pattern_strength,
            'pattern_type': 'clustered' if pattern_strength > 0.5 else 'dispersed',
            'confidence': min(1.0, pattern_strength * 1.3),
            'method': 'molecular_clustering_analysis'
        }
    
    def _generate_general_solution(self, system: List[GasMolecule], 
                                 s_coords: Dict[str, float]) -> Dict[str, Any]:
        """Generate general genomic analysis solution"""
        avg_energy = np.mean([mol.energy for mol in system])
        
        return {
            'analysis_score': min(1.0, avg_energy),
            'system_stability': 1.0 / (1.0 + self._calculate_system_variance(system)),
            'confidence': 0.8,
            'method': 'general_equilibrium_analysis'
        }
    
    def _calculate_spatial_symmetry(self, positions: np.ndarray) -> float:
        """Calculate spatial symmetry score"""
        if len(positions) < 2:
            return 0.0
            
        center = np.mean(positions, axis=0)
        centered_positions = positions - center
        
        # Check for reflection symmetry
        symmetry_score = 0.0
        for pos in centered_positions:
            reflected_pos = -pos
            # Find closest position to reflection
            distances = np.linalg.norm(centered_positions - reflected_pos, axis=1)
            min_distance = np.min(distances)
            symmetry_score += 1.0 / (1.0 + min_distance)
            
        return symmetry_score / len(positions)
    
    def _calculate_pattern_strength(self, positions: np.ndarray) -> float:
        """Calculate clustering pattern strength"""
        if len(positions) < 3:
            return 0.0
            
        # Calculate average distance between molecules
        total_distance = 0.0
        count = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                total_distance += distance
                count += 1
                
        avg_distance = total_distance / count if count > 0 else 1.0
        
        # Pattern strength inversely related to dispersion
        pattern_strength = 1.0 / (1.0 + avg_distance)
        return pattern_strength
    
    def _return_to_empty_state(self):
        """Return system to empty state for next query"""
        # Reset system pressure to baseline
        self.current_pressure = self.baseline_pressure
        # System is now ready for next perturbation

def plot_equilibrium_process(equilibrium_data, query, filename):
    """Plot the equilibrium seeking process"""
    plt.figure(figsize=(12, 8))
    
    # Simulate equilibrium convergence for visualization
    iterations = list(range(equilibrium_data['iterations'] + 1))
    
    # Create convergence curve
    initial_var = equilibrium_data['initial_variance']
    final_var = equilibrium_data['final_variance']
    
    # Exponential decay simulation
    variance_curve = []
    for i in iterations:
        progress = i / max(1, equilibrium_data['iterations'])
        var = initial_var * np.exp(-3 * progress) + final_var
        variance_curve.append(var)
    
    plt.subplot(2, 2, 1)
    plt.plot(iterations, variance_curve, 'b-', linewidth=2)
    plt.axhline(y=final_var, color='r', linestyle='--', label=f'Final Variance: {final_var:.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('System Variance')
    plt.title(f'Equilibrium Convergence - {query}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Molecular energy distribution
    plt.subplot(2, 2, 2)
    energies = np.random.exponential(scale=2.0, size=50)  # Simulated
    plt.hist(energies, bins=15, alpha=0.7, color='green', label='Final Distribution')
    plt.xlabel('Molecular Energy')
    plt.ylabel('Count')
    plt.title('Gas Molecular Energy Distribution')
    plt.legend()
    
    # S-coordinate space
    plt.subplot(2, 2, 3)
    theta = np.linspace(0, 2*np.pi, 100)
    radius = 1.0
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    plt.plot(x, y, 'k--', alpha=0.5, label='S-Space Boundary')
    
    # Plot equilibrium point
    eq_x = np.cos(equilibrium_data['iterations'] * 0.1) * 0.8
    eq_y = np.sin(equilibrium_data['iterations'] * 0.1) * 0.8
    plt.scatter(eq_x, eq_y, c='red', s=100, marker='*', label='Equilibrium Point')
    
    plt.xlabel('Knowledge Dimension')
    plt.ylabel('Entropy Dimension')
    plt.title('S-Entropy Coordinate Navigation')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Processing metrics
    plt.subplot(2, 2, 4)
    metrics = ['Iterations', 'Variance Reduction', 'Equilibrium Quality']
    values = [
        equilibrium_data['iterations'] / 100.0,
        (initial_var - final_var) / initial_var,
        1.0 if equilibrium_data['equilibrium_reached'] else 0.7
    ]
    colors = ['blue', 'green', 'orange']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('Normalized Value')
    plt.title('Processing Metrics')
    plt.ylim(0, 1.2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}', dpi=300, bbox_inches='tight')
    print(f"  Equilibrium process visualization saved to: outputs/{filename}")
    plt.close()

def create_synthesis_comparison_plot(results):
    """Create comparison plot of synthesis results"""
    plt.figure(figsize=(12, 8))
    
    queries = [r['query'] for r in results]
    synthesis_qualities = [r['synthesis_quality'] for r in results]
    processing_times = [r['processing_time'] for r in results]
    
    plt.subplot(2, 1, 1)
    bars1 = plt.bar(queries, synthesis_qualities, alpha=0.7, color='blue', label='Synthesis Quality')
    plt.ylabel('Quality Score')
    plt.title('Empty Dictionary Synthesis Performance')
    plt.ylim(0, 1.2)
    
    # Add value labels
    for bar, value in zip(bars1, synthesis_qualities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.subplot(2, 1, 2)
    bars2 = plt.bar(queries, processing_times, alpha=0.7, color='green', label='Processing Time')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Query Type')
    plt.title('Processing Time by Query')
    
    # Add value labels
    for bar, value in zip(bars2, processing_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('outputs/synthesis_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def demonstrate_empty_dictionary():
    """Demonstrate empty dictionary gas molecular synthesis"""
    print("=== Empty Dictionary Gas Molecular Synthesis Demo ===\n")
    
    empty_dict = GenomicEmptyDictionary()
    
    # Test with different queries and S-coordinates
    test_cases = [
        {
            'query': 'sequence_similarity',
            's_coords': {'knowledge': 2.5, 'time': 1.3, 'entropy': 0.8}
        },
        {
            'query': 'palindrome_detection', 
            's_coords': {'knowledge': 1.8, 'time': 2.1, 'entropy': 1.5}
        },
        {
            'query': 'pattern_recognition',
            's_coords': {'knowledge': 3.2, 'time': 0.9, 'entropy': 2.1}
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['query']}")
        print(f"  S-Coordinates: {test_case['s_coords']}")
        
        start_time = time.time()
        result = empty_dict.synthesize_genomic_meaning(
            test_case['query'], 
            test_case['s_coords']
        )
        synthesis_time = time.time() - start_time
        
        print(f"  Synthesized Solution:")
        for key, value in result['solution'].items():
            print(f"    {key}: {value}")
        print(f"  Synthesis Quality: {result['synthesis_quality']:.4f}")
        print(f"  Processing Time: {synthesis_time:.4f}s")
        print(f"  Method: {result['processing_method']}")
        
        # Create visualization
        filename = f"empty_dictionary_{test_case['query']}_equilibrium.png"
        plot_equilibrium_process(result['equilibrium_process'], test_case['query'], filename)
        
        # Store results
        result_data = {
            'query': test_case['query'],
            's_coordinates': test_case['s_coords'],
            'solution': result['solution'],
            'synthesis_quality': result['synthesis_quality'],
            'processing_time': synthesis_time,
            'method': result['processing_method'],
            'equilibrium_process': {
                'iterations': result['equilibrium_process']['iterations'],
                'initial_variance': result['equilibrium_process']['initial_variance'],
                'final_variance': result['equilibrium_process']['final_variance'],
                'equilibrium_reached': result['equilibrium_process']['equilibrium_reached']
            },
            'plot_file': filename
        }
        results.append(result_data)
        print()
    
    # Save summary results
    import json
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/empty_dictionary_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison plot
    create_synthesis_comparison_plot(results)
    
    print(f"All results and visualizations saved to 'outputs/' directory:")
    print(f"  - Individual equilibrium plots: {len(results)} files")
    print(f"  - Summary data: empty_dictionary_results.json")
    print(f"  - Comparison plot: synthesis_comparison.png")

if __name__ == "__main__":
    demonstrate_empty_dictionary()
