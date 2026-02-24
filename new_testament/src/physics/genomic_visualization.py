"""
Genomic Validation Visualization
=================================

Creates visualizations for paper validation results:
1. Charged fluid equation of state components
2. Transport coefficient scaling
3. Section size prediction vs features
4. Empty dictionary performance comparison
5. Feature detection accuracy improvements
6. Cross-organism validation

Generates publication-quality figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Any, Optional

try:
    from .genomic_charged_fluid import (
        GenomicChargedFluid,
        EmptyDictionaryValidator
    )
except ImportError:
    from genomic_charged_fluid import (
        GenomicChargedFluid,
        EmptyDictionaryValidator
    )


def plot_equation_of_state_components(results: Dict[str, Any],
                                     save_path: Optional[str] = None):
    """
    Plot contributions to charged fluid equation of state.

    Shows: PV = NkBT + U_cap + U_screen
    """
    eos = results['equation_of_state']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Absolute contributions
    components = ['Thermal\n(NkBT)', 'Capacitive\n(U_cap)', 'Screening\n(U_screen)']
    values = [eos['nkbt'], eos['u_capacitive'], eos['u_screening']]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax1.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=eos['pv_predicted'], color='black', linestyle='--',
                label='Total PV', linewidth=2)

    ax1.set_ylabel('Energy (J)', fontsize=12)
    ax1.set_title('Equation of State Components', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}',
                ha='center', va='bottom', fontsize=10)

    # Right: Fractional contributions
    fractions = [
        eos['thermal_fraction'],
        eos['capacitive_fraction'],
        abs(eos['screening_fraction'])  # Absolute value for pie chart
    ]

    ax2.pie(fractions, labels=components, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
    ax2.set_title('Relative Contributions', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_transport_coefficients(results: Dict[str, Any],
                                save_path: Optional[str] = None):
    """
    Plot transport coefficients and their scaling with partition parameters.
    """
    coeffs = results['transport_coefficients']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Coefficient names and values
    coeff_data = [
        ('Chromatin\nViscosity\nμ (Pa·s)', coeffs.chromatin_viscosity, '#3498db'),
        ('Charge\nResistivity\nρ (Ω·m)', coeffs.charge_resistivity, '#e74c3c'),
        ('Diffusivity\nD (m²/s)', coeffs.diffusivity, '#2ecc71')
    ]

    for ax, (name, value, color) in zip(axes, coeff_data):
        ax.bar([name], [value], color=color, alpha=0.7, edgecolor='black', width=0.6)
        ax.set_ylabel('Magnitude', fontsize=11)
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)

        # Add value label
        ax.text(0, value, f'{value:.2e}', ha='center', va='bottom', fontsize=10)

    fig.suptitle('Transport Coefficients from Partition Theory\nΞ = (1/N) Σ τ_p g',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_section_size_prediction(results: Dict[str, Any],
                                 save_path: Optional[str] = None):
    """
    Plot predicted section size vs typical genomic features.
    """
    section = results['section_prediction']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Feature sizes
    features = [
        ('Predicted\nSection', section.section_length_bp, '#9b59b6'),
        ('Regulatory\nElement', section.typical_regulatory_size, '#3498db'),
        ('Typical\nGene', section.typical_gene_size, '#e74c3c'),
    ]

    names = [f[0] for f in features]
    sizes = [f[1] for f in features]
    colors = [f[2] for f in features]

    bars = ax.barh(names, sizes, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xlabel('Size (base pairs)', fontsize=12)
    ax.set_title('Optimal Section Size: L = √(Dτ_p)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, size in zip(bars, sizes):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {size:,} bp',
                ha='left', va='center', fontsize=11, fontweight='bold')

    # Add formula annotation
    ax.text(0.98, 0.02,
            f'D = {section.diffusivity:.2e} m²/s\nτ_p = {section.partition_lag:.2e} s',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_complexity_comparison(results: Dict[str, Any],
                               save_path: Optional[str] = None):
    """
    Plot complexity comparison: Sequential vs Coordinate.
    """
    storage = results['storage_comparison']
    complexity = results['complexity_comparison']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Storage comparison
    storage_data = [
        ('Traditional\nO(n)', storage['traditional_storage_bits'], '#e74c3c'),
        ('Coordinate\nO(log n)', storage['coordinate_storage_bits'], '#2ecc71')
    ]

    names_s = [d[0] for d in storage_data]
    values_s = [d[1] for d in storage_data]
    colors_s = [d[2] for d in storage_data]

    bars1 = ax1.bar(names_s, values_s, color=colors_s, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Storage (bits)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('Storage Requirements', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add labels
    for bar, val in zip(bars1, values_s):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}',
                ha='center', va='bottom', fontsize=10)

    # Add reduction annotation
    ax1.text(0.5, 0.95,
            f'Reduction: {storage["reduction_orders_of_magnitude"]:.1f} orders',
            transform=ax1.transAxes, fontsize=11, fontweight='bold',
            ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Time complexity comparison
    complexity_data = [
        ('Traditional\nO(n²)', complexity['traditional_operations'], '#e74c3c'),
        ('Coordinate\nO(log S₀)', complexity['coordinate_operations'], '#2ecc71')
    ]

    names_c = [d[0] for d in complexity_data]
    values_c = [d[1] for d in complexity_data]
    colors_c = [d[2] for d in complexity_data]

    bars2 = ax2.bar(names_c, values_c, color=colors_c, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Operations', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title('Time Complexity', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add labels
    for bar, val in zip(bars2, values_c):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}',
                ha='center', va='bottom', fontsize=10)

    # Add speedup annotation
    ax2.text(0.5, 0.95,
            f'Speedup: {complexity["speedup_orders_of_magnitude"]:.1f} orders',
            transform=ax2.transAxes, fontsize=11, fontweight='bold',
            ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    fig.suptitle('Empty Dictionary Performance', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_feature_detection_accuracy(save_path: Optional[str] = None):
    """
    Plot feature detection accuracy improvements.
    """
    validator = EmptyDictionaryValidator()

    features = ['palindrome', 'regulatory', 'coding']
    feature_names = ['Palindromes', 'Regulatory\nElements', 'Coding\nSequences']

    sequential_acc = []
    coordinate_acc = []
    improvements = []

    for feature in features:
        result = validator.validate_feature_detection(feature, n_tests=500)
        sequential_acc.append(result.sequential_accuracy * 100)
        coordinate_acc.append(result.coordinate_accuracy * 100)
        improvements.append(result.improvement_percent)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy comparison
    x = np.arange(len(features))
    width = 0.35

    bars1 = ax1.bar(x - width/2, sequential_acc, width, label='Sequential',
                    color='#e74c3c', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, coordinate_acc, width, label='Coordinate',
                    color='#2ecc71', alpha=0.7, edgecolor='black')

    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Detection Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_names)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)

    # Improvement percentages
    bars3 = ax2.bar(feature_names, improvements, color='#9b59b6',
                    alpha=0.7, edgecolor='black')

    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('Accuracy Improvement', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Add value labels
    for bar, imp in zip(bars3, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'+{imp:.0f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.suptitle('Feature Detection Performance', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_all_validation_figures(output_dir: str = '.'):
    """
    Create all validation figures for the paper.

    Args:
        output_dir: Directory to save figures
    """
    from pathlib import Path
    import os

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("Generating validation figures...")
    print("-" * 60)

    # Run validation to get results
    from genomic_charged_fluid import validate_paper_claims
    results = validate_paper_claims()

    # Figure 1: Equation of state
    print("Creating Figure 1: Equation of State Components...")
    fig1 = plot_equation_of_state_components(
        results,
        save_path=output_path / 'fig1_equation_of_state.png'
    )
    plt.close(fig1)
    print("  ✓ Saved: fig1_equation_of_state.png")

    # Figure 2: Transport coefficients
    print("Creating Figure 2: Transport Coefficients...")
    fig2 = plot_transport_coefficients(
        results,
        save_path=output_path / 'fig2_transport_coefficients.png'
    )
    plt.close(fig2)
    print("  ✓ Saved: fig2_transport_coefficients.png")

    # Figure 3: Section size
    print("Creating Figure 3: Section Size Prediction...")
    fig3 = plot_section_size_prediction(
        results,
        save_path=output_path / 'fig3_section_size.png'
    )
    plt.close(fig3)
    print("  ✓ Saved: fig3_section_size.png")

    # Figure 4: Complexity comparison
    print("Creating Figure 4: Complexity Comparison...")
    fig4 = plot_complexity_comparison(
        results,
        save_path=output_path / 'fig4_complexity_comparison.png'
    )
    plt.close(fig4)
    print("  ✓ Saved: fig4_complexity_comparison.png")

    # Figure 5: Feature detection
    print("Creating Figure 5: Feature Detection Accuracy...")
    fig5 = plot_feature_detection_accuracy(
        save_path=output_path / 'fig5_feature_detection.png'
    )
    plt.close(fig5)
    print("  ✓ Saved: fig5_feature_detection.png")

    print("-" * 60)
    print(f"All figures saved to: {output_path.absolute()}")
    print("Figures ready for publication!")


if __name__ == "__main__":
    # Create all figures
    create_all_validation_figures()
