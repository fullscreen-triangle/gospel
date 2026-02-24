"""
Genomic Validation Visualizations
==================================

Creates publication-quality figures for the paper:
"Derivation of Genomic Structure from Partition Coordinates"

Generates:
- Figure 1: 4-Panel Partition Coordinate Framework
  - Panel A: 2D Coordinate Trajectories
  - Panel B: Symmetry Score Distribution
  - Panel C: 3D Information Landscape
  - Panel D: Pattern Detection Analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class GenomicVisualizer:
    """Create visualizations from validation results."""

    def __init__(self, results_dir: str):
        """Load all validation data."""
        self.results_dir = Path(results_dir)
        self.load_data()

    def load_data(self):
        """Load all JSON result files."""
        with open(self.results_dir / 'palindrome_analysis.json', 'r') as f:
            self.palindrome_data = json.load(f)

        with open(self.results_dir / 'dual_strand_geometry.json', 'r') as f:
            self.geometry_data = json.load(f)

        with open(self.results_dir / 'pattern_detection.json', 'r') as f:
            self.pattern_data = json.load(f)

        with open(self.results_dir / 'hierarchy_analysis.json', 'r') as f:
            self.hierarchy_data = json.load(f)

        print(f"Loaded data:")
        print(f"  Palindromes: {self.palindrome_data['total_palindromes']}")
        print(f"  Geometries: {self.geometry_data['sequences_analyzed']}")
        print(f"  Patterns: {self.pattern_data['total_patterns']}")
        print(f"  Hierarchies: {self.hierarchy_data['sequences_analyzed']}")

    def select_example_sequences(self) -> Tuple[List[str], List[float]]:
        """Select 3 example sequences with different symmetry scores."""
        palindromes = self.palindrome_data['palindromes']

        # Sort by symmetry score
        sorted_palindromes = sorted(palindromes, key=lambda p: p['symmetry_score'])

        # Select high, medium, low
        high_idx = len(sorted_palindromes) - 1
        medium_idx = len(sorted_palindromes) // 2
        low_idx = 0

        sequences = [
            sorted_palindromes[high_idx]['sequence'],
            sorted_palindromes[medium_idx]['sequence'],
            sorted_palindromes[low_idx]['sequence']
        ]

        scores = [
            sorted_palindromes[high_idx]['symmetry_score'],
            sorted_palindromes[medium_idx]['symmetry_score'],
            sorted_palindromes[low_idx]['symmetry_score']
        ]

        return sequences, scores

    def plot_panel_a_trajectories(self, ax):
        """
        Panel A: 2D Coordinate Trajectories

        Shows DNA sequences mapped to coordinate space.
        """
        sequences, scores = self.select_example_sequences()

        # Cardinal direction mapping
        directions = {'A': (0, 1), 'T': (0, -1), 'G': (1, 0), 'C': (-1, 0)}

        colors = ['#2E7D32', '#F57C00', '#C62828']  # Green, Orange, Red
        labels = [
            f'High symmetry ({scores[0]:.3f})',
            f'Medium symmetry ({scores[1]:.3f})',
            f'Low symmetry ({scores[2]:.3f})'
        ]

        for seq, score, color, label in zip(sequences, scores, colors, labels):
            # Compute trajectory
            x, y = [0], [0]
            for base in seq:
                dx, dy = directions.get(base, (0, 0))
                x.append(x[-1] + dx)
                y.append(y[-1] + dy)

            # Plot trajectory
            ax.plot(x, y, '-o', color=color, linewidth=2, markersize=4,
                    label=label, alpha=0.8)

            # Mark start and end
            ax.plot(x[0], y[0], 'o', color=color, markersize=10,
                    markeredgecolor='black', markeredgewidth=2, zorder=10)
            ax.plot(x[-1], y[-1], 's', color=color, markersize=10,
                    markeredgecolor='black', markeredgewidth=2, zorder=10)

        # Origin marker
        ax.plot(0, 0, 'k*', markersize=20, label='Origin', zorder=15)

        # Formatting
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_xlabel('East-West Coordinate (G-C axis)', fontsize=10, fontweight='bold')
        ax.set_ylabel('North-South Coordinate (A-T axis)', fontsize=10, fontweight='bold')
        ax.set_title('Panel A: Coordinate Trajectories by Symmetry',
                     fontsize=11, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    def plot_panel_b_distribution(self, ax1, ax2):
        """
        Panel B: Symmetry Score Distribution

        Histogram and violin plot of symmetry scores.
        """
        symmetry_scores = [p['symmetry_score'] for p in self.palindrome_data['palindromes']]

        # Panel B1: Histogram
        ax1.hist(symmetry_scores, bins=50, color='#1976D2', alpha=0.7,
                 edgecolor='black', linewidth=0.5)

        # Add mean line
        mean_score = np.mean(symmetry_scores)
        ax1.axvline(mean_score, color='red', linewidth=3, linestyle='--',
                    label=f'Mean = {mean_score:.3f}')

        # Add theoretical random (Gaussian centered at 0.5)
        x = np.linspace(0, 1, 100)
        random_dist = len(symmetry_scores) * 0.05 * np.exp(-((x - 0.5)**2) / 0.05)
        ax1.plot(x, random_dist, 'k--', linewidth=2, label='Random expectation')

        ax1.set_xlabel('Symmetry Score', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax1.set_title('Histogram', fontsize=10, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel B2: Violin plot
        parts = ax2.violinplot([symmetry_scores], positions=[0], widths=0.7,
                               showmeans=True, showmedians=True)

        # Color violin
        for pc in parts['bodies']:
            pc.set_facecolor('#1976D2')
            pc.set_alpha(0.7)

        ax2.set_ylabel('Symmetry Score', fontsize=10, fontweight='bold')
        ax2.set_title('Violin Plot', fontsize=10, fontweight='bold')
        ax2.set_xticks([0])
        ax2.set_xticklabels([f'All Palindromes\n(n={len(symmetry_scores):,})'], fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

    def plot_panel_c_3d_landscape(self, ax):
        """
        Panel C: 3D Information Landscape

        3D scatter plot of genomic features.
        """
        # Extract data from geometry
        geometries = self.geometry_data['geometries']

        # Get pattern confidences (match to sequences if possible)
        pattern_confidences = [p['confidence'] for p in self.pattern_data['patterns'][:len(geometries)]]

        # Pad if needed
        while len(pattern_confidences) < len(geometries):
            pattern_confidences.append(0.5)

        # Extract features
        x = [g['geometric_entropy'] for g in geometries]  # Geometric entropy
        y = [g['information_density'] for g in geometries]  # Information density
        z = pattern_confidences[:len(geometries)]  # Pattern confidence

        # Color by confidence
        colors = ['#2E7D32' if conf > 0.5 else '#C62828' for conf in z]

        # Plot
        scatter = ax.scatter(x, y, z, c=colors, s=200, alpha=0.6,
                            edgecolors='black', linewidth=0.5)

        # Labels
        ax.set_xlabel('Geometric Entropy', fontsize=10, fontweight='bold', labelpad=8)
        ax.set_ylabel('Information Density', fontsize=10, fontweight='bold', labelpad=8)
        ax.set_zlabel('Pattern Confidence', fontsize=10, fontweight='bold', labelpad=8)
        ax.set_title('Panel C: 3D Feature Space',
                     fontsize=11, fontweight='bold', pad=15)

        # Legend
        legend_elements = [
            Patch(facecolor='#2E7D32', label='High confidence (>0.5)'),
            Patch(facecolor='#C62828', label='Low confidence (<0.5)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

        # Viewing angle
        ax.view_init(elev=20, azim=45)

        # Grid
        ax.grid(True, alpha=0.3)

    def plot_panel_d_patterns(self, ax1, ax2):
        """
        Panel D: Pattern Detection Analysis

        Stacked bar chart and heatmap of patterns.
        """
        # Panel D1: Stacked bar chart
        pattern_types = self.pattern_data['pattern_types']
        categories = list(pattern_types.keys())
        counts = list(pattern_types.values())
        colors = ['#1976D2', '#F57C00']

        bars = ax1.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.02,
                    f'{count:,}', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

        ax1.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax1.set_title('Pattern Type Distribution', fontsize=10, fontweight='bold')
        ax1.set_ylim(0, max(counts) * 1.15)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', labelsize=9)

        # Panel D2: Heatmap of pattern confidence by type
        # Bin confidences
        patterns = self.pattern_data['patterns']

        # Separate by type
        repeat_confidences = [p['confidence'] for p in patterns if p['type'] == 'repeat']
        motif_confidences = [p['confidence'] for p in patterns if p['type'] == 'motif']

        # Create bins
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

        # Histogram for each type
        repeat_hist, _ = np.histogram(repeat_confidences, bins=bins)
        motif_hist, _ = np.histogram(motif_confidences, bins=bins)

        # Create heatmap data
        heatmap_data = np.array([repeat_hist, motif_hist])

        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd',
                    xticklabels=bin_labels,
                    yticklabels=[f'Repeat\n(n={len(repeat_confidences)})',
                                f'Motif\n(n={len(motif_confidences)})'],
                    cbar_kws={'label': 'Count'}, ax=ax2, linewidths=0.5)

        ax2.set_xlabel('Confidence Score', fontsize=10, fontweight='bold')
        ax2.set_title('Confidence Distribution', fontsize=10, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=8)

    def create_figure_1(self, save_path: str = None):
        """
        Create complete Figure 1 with all 4 panels.
        """
        print("\nGenerating Figure 1: 4-Panel Visualization...")

        # Create figure with GridSpec
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                     height_ratios=[1, 1, 1])

        # Panel A: Top-left
        ax_a = fig.add_subplot(gs[0, 0])
        self.plot_panel_a_trajectories(ax_a)

        # Panel B: Top-right (split into 2 sub-panels)
        ax_b1 = fig.add_subplot(gs[0, 1])
        ax_b2 = fig.add_subplot(gs[1, 1])
        self.plot_panel_b_distribution(ax_b1, ax_b2)

        # Panel C: Middle-left (3D)
        ax_c = fig.add_subplot(gs[1, 0], projection='3d')
        self.plot_panel_c_3d_landscape(ax_c)

        # Panel D: Bottom (split into 2 sub-panels)
        ax_d1 = fig.add_subplot(gs[2, 0])
        ax_d2 = fig.add_subplot(gs[2, 1])
        self.plot_panel_d_patterns(ax_d1, ax_d2)

        # Add panel labels (skip 3D axis which has different text API)
        panel_positions = [
            (ax_a, 'A', (0.02, 0.98)),
            (ax_b1, 'B', (0.02, 0.98)),
            (ax_d1, 'D', (0.02, 0.98))
        ]

        for ax, label, pos in panel_positions:
            ax.text(pos[0], pos[1], label, transform=ax.transAxes,
                   fontsize=18, fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add label for 3D axis (different API)
        ax_c.text2D(0.02, 0.98, 'C', transform=ax_c.transAxes,
                   fontsize=18, fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Overall title
        fig.suptitle('Figure 1: Partition Coordinate Framework for Genomic Analysis',
                    fontsize=16, fontweight='bold', y=0.995)

        # Save
        if save_path is None:
            save_path = self.results_dir / 'figure_1_complete.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight')

        print(f"  Saved: {save_path}")
        print(f"  Saved: {str(save_path).replace('.png', '.pdf')}")

        return fig

    def create_individual_panels(self):
        """Create individual high-resolution panels."""
        print("\nGenerating individual panels...")

        # Panel A
        fig_a, ax_a = plt.subplots(figsize=(8, 6))
        self.plot_panel_a_trajectories(ax_a)
        save_path_a = self.results_dir / 'panel_A_trajectories.png'
        plt.savefig(save_path_a, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path_a}")
        plt.close()

        # Panel B
        fig_b, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(12, 5))
        self.plot_panel_b_distribution(ax_b1, ax_b2)
        fig_b.suptitle('Panel B: Symmetry Score Distribution',
                      fontsize=14, fontweight='bold')
        save_path_b = self.results_dir / 'panel_B_distribution.png'
        plt.savefig(save_path_b, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path_b}")
        plt.close()

        # Panel C
        fig_c = plt.figure(figsize=(10, 8))
        ax_c = fig_c.add_subplot(111, projection='3d')
        self.plot_panel_c_3d_landscape(ax_c)
        save_path_c = self.results_dir / 'panel_C_3d_landscape.png'
        plt.savefig(save_path_c, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path_c}")
        plt.close()

        # Panel D
        fig_d, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(12, 5))
        self.plot_panel_d_patterns(ax_d1, ax_d2)
        fig_d.suptitle('Panel D: Pattern Detection Analysis',
                      fontsize=14, fontweight='bold')
        save_path_d = self.results_dir / 'panel_D_patterns.png'
        plt.savefig(save_path_d, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path_d}")
        plt.close()


def main():
    """Generate all visualizations."""
    import os

    # Get path to results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'genomic_validation_results')

    print("="*80)
    print(" "*20 + "GENOMIC VISUALIZATION GENERATOR")
    print("="*80)
    print(f"\nResults directory: {results_dir}")

    # Create visualizer
    viz = GenomicVisualizer(results_dir)

    # Generate complete figure
    viz.create_figure_1()

    # Generate individual panels
    viz.create_individual_panels()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - figure_1_complete.png (4-panel composite)")
    print("  - figure_1_complete.pdf (vector format)")
    print("  - panel_A_trajectories.png")
    print("  - panel_B_distribution.png")
    print("  - panel_C_3d_landscape.png")
    print("  - panel_D_patterns.png")
    print("="*80)


if __name__ == "__main__":
    main()
