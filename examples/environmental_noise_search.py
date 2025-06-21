#!/usr/bin/env python3
"""
Environmental Gradient Search for Genomic Analysis

Demonstrates the swamp metaphor: modulating environmental noise (water level)
to reveal relevant signals (trees) that emerge above the noise floor.

This exemplifies how nature uses environmental noise as a discovery mechanism,
rather than trying to artificially isolate variables like laboratory experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add gospel to path
sys.path.append(str(Path(__file__).parent.parent))

from gospel.core.metacognitive import (
    MetacognitiveOrchestrator, 
    EnvironmentalGradientSearch,
    NoiseProfile,
    SignalEmergence
)

def generate_synthetic_genomic_data(n_samples: int = 10000) -> dict:
    """
    Generate synthetic genomic data with embedded signals and environmental noise
    """
    np.random.seed(42)
    
    # Background genomic noise (like bushes and grass in swamp)
    background_noise = np.random.exponential(scale=0.3, size=n_samples)
    
    # Add some genomic structure (repetitive elements, GC content variation)
    gc_variation = 0.1 * np.sin(np.linspace(0, 20*np.pi, n_samples))
    repetitive_elements = 0.05 * np.random.poisson(lam=2, size=n_samples)
    
    # Environmental noise layer
    environmental_base = background_noise + gc_variation + repetitive_elements
    
    # Embed true signals (like trees in the swamp)
    true_signals = np.zeros(n_samples)
    
    # Important genes/regulatory regions (the "trees")
    signal_positions = [1000, 2500, 4200, 6800, 8500]  # Gene locations
    signal_strengths = [2.5, 3.2, 1.8, 4.1, 2.9]      # Expression levels
    
    for pos, strength in zip(signal_positions, signal_strengths):
        # Gene with some regulatory spread
        start, end = max(0, pos-50), min(n_samples, pos+150)
        signal_profile = strength * np.exp(-0.1 * np.arange(end-start))
        true_signals[start:end] += signal_profile
    
    # Add some false positives (tall grass that looks like trees)
    false_positive_positions = [1500, 3200, 5500, 7200]
    for pos in false_positive_positions:
        if pos < n_samples - 20:
            # Sharp spikes that don't have the characteristic gene profile
            true_signals[pos:pos+10] += np.random.exponential(scale=1.5, size=10)
    
    # Combine everything
    observed_data = environmental_base + true_signals
    
    # Add measurement noise (sequencing errors, technical variation)
    measurement_noise = 0.1 * np.random.normal(size=n_samples)
    observed_data += measurement_noise
    
    return {
        'observed': observed_data,
        'environmental_base': environmental_base,
        'true_signals': true_signals,
        'signal_positions': signal_positions,
        'false_positives': false_positive_positions,
        'metadata': {
            'total_samples': n_samples,
            'true_genes': len(signal_positions),
            'false_positives': len(false_positive_positions)
        }
    }

def demonstrate_swamp_metaphor(genomic_data: dict):
    """
    Demonstrate the swamp metaphor: modulate water level to find trees
    """
    print("🌊 Demonstrating Environmental Gradient Search (Swamp Metaphor)")
    print("=" * 70)
    
    observed_data = genomic_data['observed']
    environmental_search = EnvironmentalGradientSearch(
        noise_resolution=1000,
        gradient_steps=20,
        emergence_threshold=1.5
    )
    
    # Model the environmental noise (characterize the swamp)
    print("\n📊 Modeling Environmental Noise...")
    noise_profile = environmental_search.model_environmental_noise(
        observed_data, ['genomic_sequence', 'gc_content', 'repetitive_elements']
    )
    
    print(f"   Baseline Level: {noise_profile.baseline_level:.3f}")
    print(f"   Entropy Measure: {noise_profile.entropy_measure:.3f}")
    print(f"   Gradient Sensitivity: {noise_profile.gradient_sensitivity:.3f}")
    
    # Test different water levels (noise modulation factors)
    print("\n🌊 Testing Different Water Levels...")
    water_levels = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]  # Different noise amplitudes
    emergence_results = []
    
    for water_level in water_levels:
        print(f"   Water Level {water_level}x: ", end="")
        
        # Generate noise at this water level
        modulated_noise = environmental_search.modulate_noise_level(
            observed_data, noise_profile, water_level
        )
        
        # Detect what emerges above this water level
        signal_emergence = environmental_search.detect_signal_emergence(
            observed_data, modulated_noise
        )
        
        emergence_results.append({
            'water_level': water_level,
            'emergence': signal_emergence,
            'modulated_noise': modulated_noise
        })
        
        print(f"Found {np.sum(signal_emergence.emergence_trajectory > 1.5)} emergent features")
    
    # Find optimal water level
    optimal_idx = np.argmax([r['emergence'].signal_strength for r in emergence_results])
    optimal_result = emergence_results[optimal_idx]
    
    print(f"\n🎯 Optimal Water Level: {optimal_result['water_level']}x")
    print(f"   Signal Strength: {optimal_result['emergence'].signal_strength:.3f}")
    print(f"   Noise Contrast: {optimal_result['emergence'].noise_contrast_ratio:.3f}")
    print(f"   Stability: {optimal_result['emergence'].stability_measure:.3f}")
    
    return emergence_results, optimal_result

def demonstrate_metacognitive_orchestrator(genomic_data: dict):
    """
    Show how the metacognitive orchestrator uses noise modeling for decisions
    """
    print("\n🧠 Metacognitive Orchestrator Analysis")
    print("=" * 50)
    
    orchestrator = MetacognitiveOrchestrator()
    observed_data = genomic_data['observed']
    
    # Analyze genomic regions using noise-first approach
    regions = [
        {'start': 0, 'end': 2000, 'id': 'region_1'},
        {'start': 2000, 'end': 4000, 'id': 'region_2'},
        {'start': 4000, 'end': 6000, 'id': 'region_3'},
        {'start': 6000, 'end': 8000, 'id': 'region_4'},
        {'start': 8000, 'end': 10000, 'id': 'region_5'}
    ]
    
    analysis_results = []
    
    for region in regions:
        region_data = observed_data[region['start']:region['end']]
        
        result = orchestrator.analyze_genomic_region(
            region_data, 
            region['id'],
            ['variant_calling', 'expression_analysis', 'regulatory_prediction']
        )
        
        analysis_results.append(result)
        
        print(f"\n📍 {region['id']} (positions {region['start']}-{region['end']}):")
        print(f"   Posterior Belief: {result['posterior_belief']:.3f}")
        print(f"   Optimization Fitness: {result['optimization_fitness']:.3f}")
        print(f"   Noise Entropy: {result['noise_profile'].entropy_measure:.3f}")
        
        # Show decision for this region
        for node_id, decision in result['decisions'].items():
            if decision['decision']:
                print(f"   ✅ SIGNIFICANT: {decision['reasoning']}")
            else:
                print(f"   ❌ Not significant: Confidence {decision['confidence']:.3f}")
    
    return analysis_results

def compare_with_traditional_approach(genomic_data: dict):
    """
    Compare environmental gradient search with traditional signal detection
    """
    print("\n⚖️  Comparison: Environmental Gradient vs Traditional Signal Detection")
    print("=" * 70)
    
    observed_data = genomic_data['observed']
    true_positions = genomic_data['signal_positions']
    
    # Traditional approach: threshold-based signal detection
    print("\n📊 Traditional Approach (Threshold-based):")
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    for threshold in thresholds:
        detected_positions = np.where(observed_data > threshold)[0]
        
        # Calculate precision/recall
        true_detections = 0
        for true_pos in true_positions:
            if any(abs(detected_positions - true_pos) < 100):  # Within 100bp
                true_detections += 1
        
        precision = true_detections / len(detected_positions) if len(detected_positions) > 0 else 0
        recall = true_detections / len(true_positions)
        
        print(f"   Threshold {threshold}: {len(detected_positions)} detections, "
              f"Precision: {precision:.3f}, Recall: {recall:.3f}")
    
    # Environmental gradient approach
    print("\n🌊 Environmental Gradient Approach:")
    environmental_search = EnvironmentalGradientSearch()
    
    def gene_detection_objective(data):
        # Objective: find regions with sustained elevation (gene-like)
        if len(data) < 10:
            return 0
        
        # Look for sustained signal rather than just peaks
        smoothed = np.convolve(data, np.ones(10)/10, mode='valid')
        return np.mean(smoothed) * np.std(smoothed)  # High mean + variation
    
    search_result = environmental_search.environmental_gradient_search(
        observed_data, gene_detection_objective, ['genomic_sequence']
    )
    
    if search_result['best_solution'] is not None:
        # Convert solution back to positions
        emergence_trajectory = search_result['emergence_history'][-1]['signal_emergence'].emergence_trajectory
        detected_positions = np.where(emergence_trajectory > 1.5)[0]
        
        # Calculate precision/recall for environmental approach
        true_detections = 0
        for true_pos in true_positions:
            if any(abs(detected_positions - true_pos) < 100):
                true_detections += 1
        
        precision = true_detections / len(detected_positions) if len(detected_positions) > 0 else 0
        recall = true_detections / len(true_positions)
        
        print(f"   Environmental Gradient: {len(detected_positions)} detections, "
              f"Precision: {precision:.3f}, Recall: {recall:.3f}")
        print(f"   Optimization Fitness: {search_result['best_fitness']:.3f}")
    else:
        print("   No significant signals emerged from noise modulation")

def visualize_results(genomic_data: dict, emergence_results: list, optimal_result: dict):
    """
    Create visualizations showing the swamp metaphor in action
    """
    print("\n📈 Generating Visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Environmental Gradient Search: The Swamp Metaphor', fontsize=16)
    
    observed_data = genomic_data['observed']
    positions = np.arange(len(observed_data))
    
    # Plot 1: Original data with true signals highlighted
    ax1 = axes[0, 0]
    ax1.plot(positions, observed_data, alpha=0.7, color='darkblue', linewidth=0.5)
    ax1.plot(positions, genomic_data['environmental_base'], alpha=0.5, color='brown', 
             label='Environmental Noise')
    
    # Highlight true gene positions
    for pos in genomic_data['signal_positions']:
        ax1.axvline(x=pos, color='green', alpha=0.7, linestyle='--', label='True Genes' if pos == genomic_data['signal_positions'][0] else "")
    
    ax1.set_title('Original Genomic Data with Environmental Noise')
    ax1.set_xlabel('Genomic Position')
    ax1.set_ylabel('Signal Intensity')
    ax1.legend()
    
    # Plot 2: Different water levels
    ax2 = axes[0, 1]
    water_levels = [r['water_level'] for r in emergence_results]
    signal_strengths = [r['emergence'].signal_strength for r in emergence_results]
    
    ax2.plot(water_levels, signal_strengths, 'o-', color='blue', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_result['water_level'], color='red', linestyle='--', 
                label=f'Optimal Level: {optimal_result["water_level"]}x')
    ax2.set_title('Signal Strength vs Water Level')
    ax2.set_xlabel('Water Level (Noise Modulation Factor)')
    ax2.set_ylabel('Signal Strength')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Optimal emergence pattern
    ax3 = axes[1, 0]
    emergence_trajectory = optimal_result['emergence'].emergence_trajectory
    
    # Show what emerges above the optimal water level
    emergent_mask = emergence_trajectory > 1.5
    
    ax3.plot(positions, emergence_trajectory, alpha=0.7, color='darkblue', linewidth=0.8)
    ax3.fill_between(positions, 0, emergence_trajectory, where=emergent_mask, 
                     alpha=0.6, color='gold', label='Emergent Signals')
    ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Emergence Threshold')
    
    # Mark true gene positions
    for pos in genomic_data['signal_positions']:
        ax3.axvline(x=pos, color='green', alpha=0.7, linestyle=':', 
                   label='True Genes' if pos == genomic_data['signal_positions'][0] else "")
    
    ax3.set_title(f'Signal Emergence at Optimal Water Level ({optimal_result["water_level"]}x)')
    ax3.set_xlabel('Genomic Position')
    ax3.set_ylabel('Signal-to-Noise Ratio')
    ax3.legend()
    
    # Plot 4: Noise characteristics
    ax4 = axes[1, 1]
    noise_profile = optimal_result['emergence']
    
    # Show noise contrast and stability metrics
    metrics = ['Signal Strength', 'Noise Contrast', 'Stability', 'Confidence']
    values = [
        noise_profile.signal_strength,
        noise_profile.noise_contrast_ratio,
        noise_profile.stability_measure,
        (noise_profile.confidence_interval[1] - noise_profile.confidence_interval[0]) / 2
    ]
    
    bars = ax4.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
    ax4.set_title('Emergence Quality Metrics')
    ax4.set_ylabel('Metric Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('environmental_gradient_search.png', dpi=300, bbox_inches='tight')
    print("   Saved visualization as 'environmental_gradient_search.png'")
    
    return fig

def main():
    """
    Main demonstration of environmental gradient search
    """
    print("🧬 Environmental Gradient Search for Genomic Analysis")
    print("Using Nature's Approach: Noise Modulation for Signal Discovery")
    print("=" * 70)
    
    # Generate synthetic genomic data
    print("\n🔬 Generating Synthetic Genomic Data...")
    genomic_data = generate_synthetic_genomic_data(n_samples=10000)
    print(f"   Generated {genomic_data['metadata']['total_samples']} genomic positions")
    print(f"   Embedded {genomic_data['metadata']['true_genes']} true genes")
    print(f"   Added {genomic_data['metadata']['false_positives']} false positives")
    
    # Demonstrate swamp metaphor
    emergence_results, optimal_result = demonstrate_swamp_metaphor(genomic_data)
    
    # Show metacognitive orchestrator in action
    analysis_results = demonstrate_metacognitive_orchestrator(genomic_data)
    
    # Compare approaches
    compare_with_traditional_approach(genomic_data)
    
    # Create visualizations
    fig = visualize_results(genomic_data, emergence_results, optimal_result)
    
    print("\n🎯 Key Insights from Environmental Gradient Search:")
    print("   • Nature uses noise as a discovery mechanism, not an obstacle")
    print("   • Modulating environmental conditions reveals signal topology")
    print("   • Signals emerge naturally when noise is well-characterized")
    print("   • The 'swamp metaphor': adjust water level to see the trees")
    print("   • Better than traditional threshold-based approaches")
    
    print(f"\n📊 Results Summary:")
    print(f"   Optimal noise modulation: {optimal_result['water_level']}x")
    print(f"   Signal emergence strength: {optimal_result['emergence'].signal_strength:.3f}")
    print(f"   Noise modeling entropy: {optimal_result['emergence'].confidence_interval}")
    
    return genomic_data, emergence_results, analysis_results

if __name__ == "__main__":
    main() 