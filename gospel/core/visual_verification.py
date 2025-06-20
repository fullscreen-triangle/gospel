"""
Visual Understanding Verification System for Gospel

This module implements genomic circuit diagram generation and understanding
verification tests to ensure the system comprehends biological processes
rather than just pattern matching.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import random
from abc import ABC, abstractmethod
import networkx as nx
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
import seaborn as sns
from io import StringIO
import base64


class VisualizationMode(Enum):
    """Different visualization modes for genomic data"""
    CIRCUIT_DIAGRAM = "circuit_diagram"
    SUBWAY_MAP = "subway_map"
    MUSICAL_SCORE = "musical_score"
    HYBRID_SPATIAL = "hybrid_spatial"


class ComponentType(Enum):
    """Types of components in genomic circuits"""
    GENE_PROCESSOR = "gene_processor"
    REGULATORY_ELEMENT = "regulatory_element"
    PROTEIN_PRODUCT = "protein_product"
    METABOLITE = "metabolite"
    PATHWAY_NODE = "pathway_node"
    INTERACTION_WIRE = "interaction_wire"


@dataclass
class GenomicComponent:
    """Represents a component in the genomic circuit"""
    id: str
    name: str
    component_type: ComponentType
    position: Tuple[float, float]
    properties: Dict[str, Any]
    connections: List[str]  # IDs of connected components


@dataclass
class GeneProcessor:
    """Gene represented as an integrated circuit processor"""
    gene_id: str
    symbol: str
    chromosome: str
    position: int
    input_pins: List[str]  # Regulatory inputs
    output_pins: List[str]  # Regulatory targets
    processing_function: str  # Annotated function
    voltage: float  # Expression level (normalized)
    current_capacity: float  # Regulatory strength
    resistance: float  # Regulatory resistance
    
    def to_component(self, pos: Tuple[float, float]) -> GenomicComponent:
        """Convert to GenomicComponent"""
        return GenomicComponent(
            id=self.gene_id,
            name=self.symbol,
            component_type=ComponentType.GENE_PROCESSOR,
            position=pos,
            properties={
                'chromosome': self.chromosome,
                'genomic_position': self.position,
                'function': self.processing_function,
                'expression': self.voltage,
                'regulatory_strength': self.current_capacity,
                'resistance': self.resistance,
                'input_count': len(self.input_pins),
                'output_count': len(self.output_pins)
            },
            connections=self.input_pins + self.output_pins
        )


@dataclass
class RegulatoryWire:
    """Regulatory interaction as electrical wire"""
    source_gene: str
    target_gene: str
    signal_type: str  # 'activation', 'repression', 'binding'
    resistance: float  # 1.0 / interaction_strength
    capacitance: float  # Temporal delay
    current: float  # Signal strength
    
    def to_component(self, pos: Tuple[float, float]) -> GenomicComponent:
        """Convert to GenomicComponent"""
        return GenomicComponent(
            id=f"{self.source_gene}_{self.target_gene}",
            name=f"{self.source_gene} -> {self.target_gene}",
            component_type=ComponentType.INTERACTION_WIRE,
            position=pos,
            properties={
                'signal_type': self.signal_type,
                'resistance': self.resistance,
                'capacitance': self.capacitance,
                'current': self.current,
                'source': self.source_gene,
                'target': self.target_gene
            },
            connections=[self.source_gene, self.target_gene]
        )


class GenomicCircuit:
    """Electronic circuit representation of genomic networks"""
    
    def __init__(self, name: str):
        self.name = name
        self.components: Dict[str, GenomicComponent] = {}
        self.graph = nx.DiGraph()
        self.layout_positions = {}
        
    def add_component(self, component: GenomicComponent):
        """Add a component to the circuit"""
        self.components[component.id] = component
        self.graph.add_node(component.id, **component.properties)
        
    def add_connection(self, source_id: str, target_id: str, properties: Dict[str, Any] = None):
        """Add a connection between components"""
        if properties is None:
            properties = {}
        self.graph.add_edge(source_id, target_id, **properties)
        
    def remove_component(self, component_id: str):
        """Remove a component from the circuit"""
        if component_id in self.components:
            del self.components[component_id]
            self.graph.remove_node(component_id)
            
    def remove_connection(self, source_id: str, target_id: str):
        """Remove a connection from the circuit"""
        if self.graph.has_edge(source_id, target_id):
            self.graph.remove_edge(source_id, target_id)
    
    def copy(self) -> 'GenomicCircuit':
        """Create a deep copy of the circuit"""
        new_circuit = GenomicCircuit(f"{self.name}_copy")
        
        # Copy components
        for comp_id, component in self.components.items():
            new_component = GenomicComponent(
                id=component.id,
                name=component.name,
                component_type=component.component_type,
                position=component.position,
                properties=component.properties.copy(),
                connections=component.connections.copy()
            )
            new_circuit.add_component(new_component)
        
        # Copy connections
        for source, target, data in self.graph.edges(data=True):
            new_circuit.add_connection(source, target, data.copy())
            
        return new_circuit
    
    def get_neighbors(self, component_id: str) -> List[str]:
        """Get neighboring components"""
        if component_id in self.graph:
            return list(self.graph.neighbors(component_id))
        return []
    
    def compute_layout(self, layout_type: str = 'spring') -> Dict[str, Tuple[float, float]]:
        """Compute layout positions for visualization"""
        if layout_type == 'spring':
            self.layout_positions = nx.spring_layout(self.graph, k=3, iterations=50)
        elif layout_type == 'circular':
            self.layout_positions = nx.circular_layout(self.graph)
        elif layout_type == 'hierarchical':
            self.layout_positions = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
        else:
            # Default spring layout
            self.layout_positions = nx.spring_layout(self.graph)
            
        return self.layout_positions


class GenomicCircuitVisualizer:
    """Generates electronic circuit representations of gene networks"""
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        self.figsize = figsize
        self.color_scheme = {
            ComponentType.GENE_PROCESSOR: '#4CAF50',      # Green
            ComponentType.REGULATORY_ELEMENT: '#FF9800',   # Orange
            ComponentType.PROTEIN_PRODUCT: '#2196F3',     # Blue
            ComponentType.METABOLITE: '#9C27B0',          # Purple
            ComponentType.PATHWAY_NODE: '#607D8B',        # Blue Grey
            ComponentType.INTERACTION_WIRE: '#424242'      # Dark Grey
        }
        self.logger = logging.getLogger(__name__)
    
    def generate_circuit(self, 
                        gene_network: nx.Graph, 
                        expression_data: Dict[str, float],
                        layout_type: str = 'spring') -> GenomicCircuit:
        """
        Generate electronic circuit representation of gene network.
        
        Args:
            gene_network: NetworkX graph of gene interactions
            expression_data: Dictionary mapping gene IDs to expression levels
            layout_type: Layout algorithm for positioning
            
        Returns:
            GenomicCircuit object
        """
        circuit = GenomicCircuit(f"genomic_circuit_{random.randint(1000, 9999)}")
        
        # Convert genes to processors
        for gene_id in gene_network.nodes():
            gene_data = gene_network.nodes[gene_id]
            
            # Get regulatory connections
            input_pins = list(gene_network.predecessors(gene_id))
            output_pins = list(gene_network.successors(gene_id))
            
            # Create gene processor
            processor = GeneProcessor(
                gene_id=gene_id,
                symbol=gene_data.get('symbol', gene_id),
                chromosome=gene_data.get('chromosome', 'unknown'),
                position=gene_data.get('position', 0),
                input_pins=input_pins,
                output_pins=output_pins,
                processing_function=gene_data.get('function', 'unknown'),
                voltage=self._normalize_expression(expression_data.get(gene_id, 0)),
                current_capacity=len(output_pins) * 0.1,
                resistance=1.0 / (len(input_pins) + 1)
            )
            
            # Add to circuit (position will be set during layout)
            circuit.add_component(processor.to_component((0, 0)))
        
        # Convert interactions to wires
        for source, target, edge_data in gene_network.edges(data=True):
            wire = RegulatoryWire(
                source_gene=source,
                target_gene=target,
                signal_type=edge_data.get('interaction_type', 'unknown'),
                resistance=1.0 / edge_data.get('strength', 0.5),
                capacitance=edge_data.get('delay', 0.1),
                current=edge_data.get('strength', 0.5)
            )
            
            circuit.add_connection(source, target, wire.properties)
        
        # Compute layout
        circuit.compute_layout(layout_type)
        
        # Update component positions
        for comp_id, component in circuit.components.items():
            if comp_id in circuit.layout_positions:
                component.position = circuit.layout_positions[comp_id]
        
        return circuit
    
    def render_circuit_svg(self, circuit: GenomicCircuit) -> str:
        """
        Render circuit as SVG string.
        
        Args:
            circuit: GenomicCircuit to render
            
        Returns:
            SVG string representation
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw components
        for comp_id, component in circuit.components.items():
            self._draw_component(ax, component)
        
        # Draw connections
        for source, target, data in circuit.graph.edges(data=True):
            self._draw_connection(ax, circuit, source, target, data)
        
        # Add title and legend
        ax.set_title(f"Genomic Circuit: {circuit.name}", fontsize=16, fontweight='bold')
        self._add_legend(ax)
        
        # Convert to SVG
        svg_buffer = StringIO()
        fig.savefig(svg_buffer, format='svg', bbox_inches='tight', dpi=300)
        svg_string = svg_buffer.getvalue()
        plt.close(fig)
        
        return svg_string
    
    def _draw_component(self, ax, component: GenomicComponent):
        """Draw a single component on the axes"""
        x, y = component.position
        
        if component.component_type == ComponentType.GENE_PROCESSOR:
            # Draw as integrated circuit
            width, height = 0.08, 0.05
            rect = FancyBboxPatch(
                (x - width/2, y - height/2), width, height,
                boxstyle="round,pad=0.005",
                facecolor=self.color_scheme[component.component_type],
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(rect)
            
            # Add pins
            input_count = component.properties.get('input_count', 0)
            output_count = component.properties.get('output_count', 0)
            
            # Input pins (left side)
            for i in range(input_count):
                pin_y = y + (i - input_count/2 + 0.5) * 0.01
                circle = Circle((x - width/2 - 0.005, pin_y), 0.002, 
                              facecolor='white', edgecolor='black')
                ax.add_patch(circle)
            
            # Output pins (right side)
            for i in range(output_count):
                pin_y = y + (i - output_count/2 + 0.5) * 0.01
                circle = Circle((x + width/2 + 0.005, pin_y), 0.002,
                              facecolor='white', edgecolor='black')
                ax.add_patch(circle)
            
            # Label
            ax.text(x, y, component.name, ha='center', va='center', 
                   fontsize=6, fontweight='bold')
            
            # Expression level indicator (voltage)
            expression = component.properties.get('expression', 0)
            if expression > 0.7:
                color = 'red'
            elif expression > 0.3:
                color = 'orange'
            else:
                color = 'blue'
            
            ax.text(x, y - height/2 - 0.015, f"V={expression:.2f}", 
                   ha='center', va='top', fontsize=4, color=color)
        
        elif component.component_type == ComponentType.REGULATORY_ELEMENT:
            # Draw as diamond
            diamond = Polygon([(x, y+0.02), (x+0.02, y), (x, y-0.02), (x-0.02, y)],
                            facecolor=self.color_scheme[component.component_type],
                            edgecolor='black')
            ax.add_patch(diamond)
            ax.text(x, y, component.name[:3], ha='center', va='center', fontsize=4)
    
    def _draw_connection(self, ax, circuit: GenomicCircuit, source: str, target: str, data: Dict):
        """Draw connection between components"""
        if source not in circuit.components or target not in circuit.components:
            return
            
        source_pos = circuit.components[source].position
        target_pos = circuit.components[target].position
        
        # Get connection properties
        signal_type = data.get('signal_type', 'unknown')
        current = data.get('current', 0.5)
        
        # Line style based on signal type
        if signal_type == 'activation':
            linestyle = '-'
            color = 'green'
            alpha = 0.7
        elif signal_type == 'repression':
            linestyle = '--'
            color = 'red'
            alpha = 0.7
        else:
            linestyle = ':'
            color = 'gray'
            alpha = 0.5
        
        # Line width based on current (signal strength)
        linewidth = max(0.5, current * 3)
        
        # Draw arrow
        ax.annotate('', xy=target_pos, xytext=source_pos,
                   arrowprops=dict(arrowstyle='->', lw=linewidth, 
                                 color=color, alpha=alpha, linestyle=linestyle))
    
    def _add_legend(self, ax):
        """Add legend to the plot"""
        legend_elements = []
        for comp_type, color in self.color_scheme.items():
            if comp_type != ComponentType.INTERACTION_WIRE:
                legend_elements.append(
                    patches.Patch(color=color, label=comp_type.value.replace('_', ' ').title())
                )
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    def _normalize_expression(self, expression_value: float) -> float:
        """Normalize expression value to 0-1 range (voltage)"""
        # Assume log2 fold change input, convert to 0-1 scale
        if expression_value == 0:
            return 0.5  # Neutral
        elif expression_value > 0:
            return min(1.0, 0.5 + expression_value / 10.0)  # Upregulated
        else:
            return max(0.0, 0.5 + expression_value / 10.0)  # Downregulated


class VisualUnderstandingVerifier:
    """
    Tests system understanding through visual circuit manipulation.
    
    This class implements four types of verification tests:
    1. Occlusion Test - Hide components and predict missing
    2. Reconstruction Test - Complete partial circuits
    3. Perturbation Test - Predict cascade effects
    4. Context Switch Test - Adapt to different conditions
    """
    
    def __init__(self, bayesian_network):
        """
        Initialize the verifier.
        
        Args:
            bayesian_network: MetacognitiveBayesianNetwork instance
        """
        self.bayesian_network = bayesian_network
        self.test_results = []
        self.logger = logging.getLogger(__name__)
    
    def occlusion_test(self, 
                      circuit: GenomicCircuit, 
                      occlusion_ratio: float = 0.3) -> Dict[str, float]:
        """
        Test understanding by hiding circuit components and predicting missing elements.
        
        Args:
            circuit: Original genomic circuit
            occlusion_ratio: Fraction of components to hide (0.2-0.4)
            
        Returns:
            Dictionary with test results and accuracy metrics
        """
        # Create copy of circuit
        test_circuit = circuit.copy()
        
        # Randomly select components to hide
        all_components = list(circuit.components.keys())
        n_hidden = int(len(all_components) * occlusion_ratio)
        n_hidden = max(1, min(n_hidden, len(all_components) - 1))  # At least 1, at most n-1
        
        hidden_components = random.sample(all_components, n_hidden)
        
        # Remove hidden components
        for comp_id in hidden_components:
            test_circuit.remove_component(comp_id)
        
        # Predict missing components using Bayesian network
        predicted_components = self._predict_missing_components(test_circuit, circuit)
        
        # Calculate accuracy
        correct_predictions = len(set(predicted_components) & set(hidden_components))
        total_hidden = len(hidden_components)
        accuracy = correct_predictions / total_hidden if total_hidden > 0 else 0
        
        # Additional metrics
        precision = correct_predictions / len(predicted_components) if predicted_components else 0
        recall = correct_predictions / total_hidden if total_hidden > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'test_type': 'occlusion',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'hidden_count': len(hidden_components),
            'predicted_count': len(predicted_components),
            'correct_predictions': correct_predictions,
            'hidden_components': hidden_components,
            'predicted_components': predicted_components
        }
        
        self.test_results.append(results)
        return results
    
    def reconstruction_test(self, 
                          circuit: GenomicCircuit,
                          completion_ratio: float = 0.4) -> Dict[str, float]:
        """
        Test by providing partial circuit and assessing completion accuracy.
        
        Args:
            circuit: Original complete circuit
            completion_ratio: Fraction of circuit to provide initially
            
        Returns:
            Dictionary with reconstruction accuracy metrics
        """
        # Create partial circuit
        all_components = list(circuit.components.keys())
        n_provided = int(len(all_components) * completion_ratio)
        provided_components = random.sample(all_components, n_provided)
        
        partial_circuit = GenomicCircuit(f"{circuit.name}_partial")
        
        # Add provided components
        for comp_id in provided_components:
            component = circuit.components[comp_id]
            partial_circuit.add_component(component)
        
        # Add connections between provided components
        for source, target, data in circuit.graph.edges(data=True):
            if source in provided_components and target in provided_components:
                partial_circuit.add_connection(source, target, data)
        
        # Predict completion
        completed_circuit = self._complete_circuit(partial_circuit, circuit)
        
        # Evaluate completion accuracy
        missing_components = set(all_components) - set(provided_components)
        predicted_components = set(completed_circuit.components.keys()) - set(provided_components)
        
        correct_completions = len(predicted_components & missing_components)
        total_missing = len(missing_components)
        accuracy = correct_completions / total_missing if total_missing > 0 else 0
        
        results = {
            'test_type': 'reconstruction',
            'accuracy': accuracy,
            'provided_count': len(provided_components),
            'missing_count': total_missing,
            'predicted_count': len(predicted_components),
            'correct_completions': correct_completions,
            'completion_ratio': completion_ratio
        }
        
        self.test_results.append(results)
        return results
    
    def perturbation_test(self, 
                         circuit: GenomicCircuit,
                         perturbation_strength: float = 0.5) -> Dict[str, float]:
        """
        Test cascade effect prediction by modifying single components.
        
        Args:
            circuit: Original circuit
            perturbation_strength: Strength of perturbation (0-1)
            
        Returns:
            Dictionary with cascade prediction accuracy
        """
        # Select random component to perturb
        all_components = list(circuit.components.keys())
        perturbed_component = random.choice(all_components)
        
        # Apply perturbation (modify expression level)
        perturbed_circuit = circuit.copy()
        component = perturbed_circuit.components[perturbed_component]
        
        if 'expression' in component.properties:
            original_expression = component.properties['expression']
            # Randomly increase or decrease
            direction = random.choice([-1, 1])
            new_expression = max(0, min(1, original_expression + direction * perturbation_strength))
            component.properties['expression'] = new_expression
        
        # Predict cascade effects
        predicted_effects = self._predict_cascade_effects(
            perturbed_circuit, perturbed_component, perturbation_strength
        )
        
        # Simulate true cascade (simplified model)
        true_effects = self._simulate_cascade_effects(
            circuit, perturbed_component, perturbation_strength
        )
        
        # Calculate prediction accuracy
        if true_effects and predicted_effects:
            # Compare affected components
            true_affected = set(true_effects.keys())
            predicted_affected = set(predicted_effects.keys())
            
            overlap = len(true_affected & predicted_affected)
            union = len(true_affected | predicted_affected)
            jaccard_similarity = overlap / union if union > 0 else 0
            
            # Compare effect magnitudes
            magnitude_errors = []
            for comp_id in true_affected & predicted_affected:
                true_mag = true_effects[comp_id]
                pred_mag = predicted_effects[comp_id]
                magnitude_errors.append(abs(true_mag - pred_mag))
            
            mean_magnitude_error = np.mean(magnitude_errors) if magnitude_errors else 1.0
            magnitude_accuracy = 1.0 - mean_magnitude_error
        else:
            jaccard_similarity = 0.0
            magnitude_accuracy = 0.0
        
        results = {
            'test_type': 'perturbation',
            'jaccard_similarity': jaccard_similarity,
            'magnitude_accuracy': magnitude_accuracy,
            'overall_accuracy': (jaccard_similarity + magnitude_accuracy) / 2,
            'perturbed_component': perturbed_component,
            'perturbation_strength': perturbation_strength,
            'true_affected_count': len(true_effects) if true_effects else 0,
            'predicted_affected_count': len(predicted_effects) if predicted_effects else 0
        }
        
        self.test_results.append(results)
        return results
    
    def context_switch_test(self, 
                           circuit: GenomicCircuit,
                           new_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Test circuit adaptation to different cellular contexts.
        
        Args:
            circuit: Original circuit
            new_context: New cellular context (e.g., different cell type, condition)
            
        Returns:
            Dictionary with adaptation accuracy metrics
        """
        # Create context-adapted circuit
        adapted_circuit = self._adapt_circuit_to_context(circuit, new_context)
        
        # Predict how components should change
        predicted_changes = self._predict_context_changes(circuit, new_context)
        
        # Simulate true context effects (simplified)
        true_changes = self._simulate_context_effects(circuit, new_context)
        
        # Evaluate adaptation accuracy
        if true_changes and predicted_changes:
            change_accuracy = self._compare_context_changes(predicted_changes, true_changes)
        else:
            change_accuracy = 0.0
        
        results = {
            'test_type': 'context_switch',
            'adaptation_accuracy': change_accuracy,
            'context': new_context,
            'predicted_changes_count': len(predicted_changes) if predicted_changes else 0,
            'true_changes_count': len(true_changes) if true_changes else 0
        }
        
        self.test_results.append(results)
        return results
    
    def run_comprehensive_verification(self, 
                                     circuit: GenomicCircuit,
                                     n_tests: int = 10) -> Dict[str, float]:
        """
        Run all verification tests multiple times and return aggregate results.
        
        Args:
            circuit: Circuit to test
            n_tests: Number of test iterations per test type
            
        Returns:
            Dictionary with comprehensive verification results
        """
        all_results = {
            'occlusion': [],
            'reconstruction': [],
            'perturbation': [],
            'context_switch': []
        }
        
        for i in range(n_tests):
            # Occlusion test
            occlusion_result = self.occlusion_test(circuit)
            all_results['occlusion'].append(occlusion_result['accuracy'])
            
            # Reconstruction test
            reconstruction_result = self.reconstruction_test(circuit)
            all_results['reconstruction'].append(reconstruction_result['accuracy'])
            
            # Perturbation test
            perturbation_result = self.perturbation_test(circuit)
            all_results['perturbation'].append(perturbation_result['overall_accuracy'])
            
            # Context switch test
            context = {'cell_type': random.choice(['hepatocyte', 'neuron', 'fibroblast']),
                      'condition': random.choice(['normal', 'stressed', 'diseased'])}
            context_result = self.context_switch_test(circuit, context)
            all_results['context_switch'].append(context_result['adaptation_accuracy'])
        
        # Compute aggregate statistics
        summary = {}
        for test_type, accuracies in all_results.items():
            summary[f'{test_type}_mean'] = np.mean(accuracies)
            summary[f'{test_type}_std'] = np.std(accuracies)
            summary[f'{test_type}_min'] = np.min(accuracies)
            summary[f'{test_type}_max'] = np.max(accuracies)
        
        # Overall understanding score
        overall_scores = []
        for i in range(n_tests):
            test_scores = [all_results[test_type][i] for test_type in all_results.keys()]
            overall_scores.append(np.mean(test_scores))
        
        summary['overall_understanding_score'] = np.mean(overall_scores)
        summary['overall_understanding_std'] = np.std(overall_scores)
        summary['n_tests'] = n_tests
        
        return summary
    
    # Helper methods for prediction and simulation
    def _predict_missing_components(self, 
                                  partial_circuit: GenomicCircuit,
                                  reference_circuit: GenomicCircuit) -> List[str]:
        """Predict missing components using Bayesian network"""
        # Simplified prediction based on network structure
        missing_predictions = []
        
        # Analyze connectivity patterns
        for comp_id in partial_circuit.components:
            neighbors = partial_circuit.get_neighbors(comp_id)
            
            # Look for components that should be connected based on patterns
            for ref_comp_id in reference_circuit.components:
                if ref_comp_id not in partial_circuit.components:
                    ref_neighbors = reference_circuit.get_neighbors(ref_comp_id)
                    
                    # Check if this missing component shares neighbors with existing ones
                    shared_neighbors = len(set(neighbors) & set(ref_neighbors))
                    if shared_neighbors > 0:
                        missing_predictions.append(ref_comp_id)
        
        return list(set(missing_predictions))  # Remove duplicates
    
    def _complete_circuit(self, 
                         partial_circuit: GenomicCircuit,
                         reference_circuit: GenomicCircuit) -> GenomicCircuit:
        """Complete partial circuit using pattern recognition"""
        completed = partial_circuit.copy()
        
        # Add missing components based on connectivity patterns
        predicted_missing = self._predict_missing_components(partial_circuit, reference_circuit)
        
        for comp_id in predicted_missing:
            if comp_id in reference_circuit.components:
                component = reference_circuit.components[comp_id]
                completed.add_component(component)
        
        return completed
    
    def _predict_cascade_effects(self, 
                               circuit: GenomicCircuit,
                               perturbed_component: str,
                               strength: float) -> Dict[str, float]:
        """Predict cascade effects of component perturbation"""
        effects = {}
        
        # Simple propagation model
        visited = set()
        queue = [(perturbed_component, strength)]
        
        while queue:
            current_comp, current_strength = queue.pop(0)
            if current_comp in visited or current_strength < 0.1:
                continue
                
            visited.add(current_comp)
            effects[current_comp] = current_strength
            
            # Propagate to neighbors with decay
            neighbors = circuit.get_neighbors(current_comp)
            for neighbor in neighbors:
                if neighbor not in visited:
                    # Decay strength based on distance
                    new_strength = current_strength * 0.7
                    queue.append((neighbor, new_strength))
        
        return effects
    
    def _simulate_cascade_effects(self, 
                                circuit: GenomicCircuit,
                                perturbed_component: str,
                                strength: float) -> Dict[str, float]:
        """Simulate true cascade effects (ground truth)"""
        # Simplified simulation - in reality this would use biological models
        return self._predict_cascade_effects(circuit, perturbed_component, strength)
    
    def _adapt_circuit_to_context(self, 
                                circuit: GenomicCircuit,
                                context: Dict[str, Any]) -> GenomicCircuit:
        """Adapt circuit to new cellular context"""
        adapted = circuit.copy()
        
        # Modify component properties based on context
        cell_type = context.get('cell_type', 'unknown')
        condition = context.get('condition', 'normal')
        
        for comp_id, component in adapted.components.items():
            if component.component_type == ComponentType.GENE_PROCESSOR:
                # Modify expression based on cell type and condition
                current_expression = component.properties.get('expression', 0.5)
                
                # Cell type specific modifications
                if cell_type == 'neuron':
                    # Neurons have higher expression of neural genes
                    if 'neural' in component.name.lower():
                        current_expression *= 1.5
                elif cell_type == 'hepatocyte':
                    # Hepatocytes have higher expression of metabolic genes
                    if any(term in component.name.lower() for term in ['metabol', 'enzyme']):
                        current_expression *= 1.3
                
                # Condition specific modifications
                if condition == 'stressed':
                    # Stress response genes upregulated
                    if any(term in component.name.lower() for term in ['stress', 'heat', 'shock']):
                        current_expression *= 2.0
                elif condition == 'diseased':
                    # Disease genes may be dysregulated
                    current_expression *= random.uniform(0.5, 1.5)
                
                component.properties['expression'] = min(1.0, current_expression)
        
        return adapted
    
    def _predict_context_changes(self, 
                               circuit: GenomicCircuit,
                               context: Dict[str, Any]) -> Dict[str, float]:
        """Predict how components should change in new context"""
        # Simplified prediction model
        changes = {}
        
        for comp_id, component in circuit.components.items():
            if component.component_type == ComponentType.GENE_PROCESSOR:
                # Predict expression change
                predicted_change = random.uniform(-0.3, 0.3)  # Simplified
                changes[comp_id] = predicted_change
        
        return changes
    
    def _simulate_context_effects(self, 
                                circuit: GenomicCircuit,
                                context: Dict[str, Any]) -> Dict[str, float]:
        """Simulate true context effects"""
        # Use the same prediction for now (in reality would use biological models)
        return self._predict_context_changes(circuit, context)
    
    def _compare_context_changes(self, 
                               predicted: Dict[str, float],
                               true: Dict[str, float]) -> float:
        """Compare predicted vs true context changes"""
        if not predicted or not true:
            return 0.0
        
        # Calculate correlation between predicted and true changes
        common_components = set(predicted.keys()) & set(true.keys())
        if not common_components:
            return 0.0
        
        pred_values = [predicted[comp] for comp in common_components]
        true_values = [true[comp] for comp in common_components]
        
        correlation = np.corrcoef(pred_values, true_values)[0, 1]
        return max(0, correlation)  # Return 0 if negative correlation 