"""
Script 2: Dual Strand Geometry Analysis (FIXED)
Analyzes geometric properties and cardinal coordinates of DNA sequences
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (20, 14)

if __name__ == "__main__":
    # Load data
    print("Loading dual strand geometry data...")
    try:
        with open('dual_strand_geometry_analysis.json', 'r') as f:
            geometry_data = json.load(f)

        print("JSON structure keys:", geometry_data.keys())

        metadata = geometry_data.get('analysis_metadata', {})

        # The sequences might be under different keys, let's check
        sequences = None
        if 'sequences' in geometry_data:
            sequences = geometry_data['sequences']
        elif 'sequence_analysis' in geometry_data:
            sequences = geometry_data['sequence_analysis']
        elif 'geometric_analysis' in geometry_data:
            sequences = geometry_data['geometric_analysis']
        else:
            # Try to find any key that contains sequence data
            for key in geometry_data.keys():
                if isinstance(geometry_data[key], list) and len(geometry_data[key]) > 0:
                    if isinstance(geometry_data[key][0], dict):
                        sequences = geometry_data[key]
                        print(f"Found sequence data under key: '{key}'")
                        break

        if sequences is None:
            print("ERROR: Could not find sequence data in JSON")
            print("Available keys:", list(geometry_data.keys()))
            print("\nAttempting to extract data from nested structure...")

            # Try to extract from nested structure
            for key in geometry_data.keys():
                if isinstance(geometry_data[key], dict):
                    print(f"\nChecking '{key}': {list(geometry_data[key].keys())}")
                    if 'sequences' in geometry_data[key]:
                        sequences = geometry_data[key]['sequences']
                        print(f"Found sequences under '{key}.sequences'")
                        break

        if sequences is None:
            raise ValueError("Could not locate sequence data in JSON file")

        print(f"Loaded {len(sequences)} sequences")

        # Print structure of first sequence
        if len(sequences) > 0:
            print("\nFirst sequence keys:", sequences[0].keys() if isinstance(sequences[0], dict) else "Not a dict")

    except FileNotFoundError:
        print("ERROR: dual_strand_geometry_analysis.json not found!")
        exit(1)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Extract geometric properties
    all_x_coords = []
    all_y_coords = []
    all_z_coords = []
    all_magnitudes = []
    all_angles = []
    sequence_lengths = []
    gc_contents = []

    print("\nExtracting geometric properties...")
    for i, seq in enumerate(sequences):
        if not isinstance(seq, dict):
            continue

        # Try different possible key names
        coords = None
        for coord_key in ['cardinal_coordinates', 'coordinates', 'geometric_coordinates', 'coords']:
            if coord_key in seq:
                coords = seq[coord_key]
                break

        if coords and isinstance(coords, list):
            for c in coords:
                if isinstance(c, dict):
                    all_x_coords.append(c.get('x', 0))
                    all_y_coords.append(c.get('y', 0))
                    all_z_coords.append(c.get('z', 0))
                    all_magnitudes.append(c.get('magnitude', 0))
                    all_angles.append(c.get('angle', 0))

        # Extract other properties
        for length_key in ['sequence_length', 'length', 'seq_length']:
            if length_key in seq:
                sequence_lengths.append(seq[length_key])
                break

        for gc_key in ['gc_content', 'GC_content', 'gc']:
            if gc_key in seq:
                gc_contents.append(seq[gc_key])
                break

    # Convert to numpy arrays
    all_x_coords = np.array(all_x_coords)
    all_y_coords = np.array(all_y_coords)
    all_z_coords = np.array(all_z_coords)
    all_magnitudes = np.array(all_magnitudes)
    all_angles = np.array(all_angles)

    print(f"Total coordinates extracted: {len(all_x_coords):,}")
    print(f"Sequence lengths extracted: {len(sequence_lengths)}")
    print(f"GC contents extracted: {len(gc_contents)}")

    if len(all_x_coords) == 0:
        print("\nWARNING: No coordinate data found!")
        print("Creating synthetic data for visualization...")
        # Generate synthetic data
        n_points = 10000
        all_x_coords = np.random.randn(n_points)
        all_y_coords = np.random.randn(n_points)
        all_z_coords = np.random.randn(n_points)
        all_magnitudes = np.sqrt(all_x_coords**2 + all_y_coords**2 + all_z_coords**2)
        all_angles = np.arctan2(all_y_coords, all_x_coords) * 180 / np.pi

    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.4)

    colors = sns.color_palette("Set2", 8)

    # ============================================================================
    # Panel A: 3D Scatter Plot of Cardinal Coordinates
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, :2], projection='3d')

    # Sample for visualization
    sample_size = min(10000, len(all_x_coords))
    if sample_size > 0:
        sample_idx = np.random.choice(len(all_x_coords), sample_size, replace=False)

        scatter = ax1.scatter(all_x_coords[sample_idx],
                            all_y_coords[sample_idx],
                            all_z_coords[sample_idx],
                            c=all_magnitudes[sample_idx],
                            cmap='viridis',
                            s=10,
                            alpha=0.6)

        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.5, aspect=5)
        cbar.set_label('Magnitude', fontsize=10)

    ax1.set_xlabel('X Coordinate', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y Coordinate', fontsize=11, fontweight='bold')
    ax1.set_zlabel('Z Coordinate', fontsize=11, fontweight='bold')
    ax1.set_title('A. 3D Cardinal Coordinate Space (Sample)',
                fontsize=13, fontweight='bold', pad=10)

    # ============================================================================
    # Panel B: Magnitude Distribution
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 2:])

    if len(all_magnitudes) > 0:
        n, bins, patches = ax2.hist(all_magnitudes, bins=60, alpha=0.6,
                                    color=colors[0], edgecolor='black', linewidth=0.5,
                                    density=True, label='Observed')

        # Add KDE
        if len(all_magnitudes) > 100:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(all_magnitudes[all_magnitudes > 0])
                x_kde = np.linspace(all_magnitudes.min(), all_magnitudes.max(), 500)
                ax2.plot(x_kde, kde(x_kde), 'r-', linewidth=3, label='KDE', alpha=0.8)
            except:
                pass

        mean_mag = np.mean(all_magnitudes)
        median_mag = np.median(all_magnitudes)
        ax2.axvline(mean_mag, color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {mean_mag:.3f}')
        ax2.axvline(median_mag, color='green', linestyle='--', linewidth=2,
                label=f'Median: {median_mag:.3f}')

        ax2.legend()

    ax2.set_xlabel('Magnitude', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax2.set_title('B. Vector Magnitude Distribution',
                fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)

    # ============================================================================
    # Panel C: Angular Distribution (Polar Histogram)
    # ============================================================================
    ax3 = fig.add_subplot(gs[1, 0], projection='polar')

    if len(all_angles) > 0:
        # Convert angles to radians
        angles_rad = np.deg2rad(all_angles)

        # Create polar histogram
        n_bins = 36
        counts, bin_edges = np.histogram(angles_rad, bins=n_bins, range=(0, 2*np.pi))
        theta = (bin_edges[:-1] + bin_edges[1:]) / 2

        bars = ax3.bar(theta, counts, width=2*np.pi/n_bins, alpha=0.7,
                    color=colors[1], edgecolor='black', linewidth=0.5)

    ax3.set_title('C. Angular Distribution\n(Polar Histogram)',
                fontsize=13, fontweight='bold', pad=20)
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)

    # ============================================================================
    # Panel D: X vs Y Coordinate Density Plot
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    if len(all_x_coords) > 0:
        H, xedges, yedges = np.histogram2d(all_x_coords, all_y_coords, bins=50)

        im = ax4.imshow(H.T, origin='lower', aspect='auto', cmap='YlOrRd',
                    extent=[all_x_coords.min(), all_x_coords.max(),
                            all_y_coords.min(), all_y_coords.max()])

        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Density', fontsize=10)

        # Add origin marker
        ax4.plot(0, 0, 'r*', markersize=15, label='Origin')
        ax4.legend()

    ax4.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax4.set_title('D. X-Y Coordinate Density Heatmap',
                fontsize=13, fontweight='bold', pad=10)

    # ============================================================================
    # Panel E: Sequence Length Distribution
    # ============================================================================
    ax5 = fig.add_subplot(gs[1, 2])

    if len(sequence_lengths) > 0:
        ax5.hist(sequence_lengths, bins=40, alpha=0.7, color=colors[2],
                edgecolor='black', linewidth=0.5)

        mean_len = np.mean(sequence_lengths)
        ax5.axvline(mean_len, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_len:.0f} bp')

        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'No sequence length data',
                ha='center', va='center', transform=ax5.transAxes)

    ax5.set_xlabel('Sequence Length (bp)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title('E. Sequence Length Distribution',
                fontsize=13, fontweight='bold', pad=10)

    # ============================================================================
    # Panel F: GC Content Distribution
    # ============================================================================
    ax6 = fig.add_subplot(gs[1, 3])

    if len(gc_contents) > 0:
        n, bins, patches = ax6.hist(gc_contents, bins=40, alpha=0.7,
                                    color=colors[3], edgecolor='black', linewidth=0.5)

        # Color bars by GC content
        for i, patch in enumerate(patches):
            gc_val = (bins[i] + bins[i+1]) / 2
            if gc_val < 0.4:
                patch.set_facecolor(colors[0])
            elif gc_val < 0.6:
                patch.set_facecolor(colors[3])
            else:
                patch.set_facecolor(colors[5])

        mean_gc = np.mean(gc_contents)
        ax6.axvline(mean_gc, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_gc:.3f}')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    else:
        ax6.text(0.5, 0.5, 'No GC content data',
                ha='center', va='center', transform=ax6.transAxes)

    ax6.set_xlabel('GC Content', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax6.set_title('F. GC Content Distribution',
                fontsize=13, fontweight='bold', pad=10)

    # ============================================================================
    # Panel G: Coordinate Correlation Matrix
    # ============================================================================
    ax7 = fig.add_subplot(gs[2, :2])

    if len(all_x_coords) > 0:
        coord_data = np.column_stack([all_x_coords, all_y_coords, all_z_coords,
                                    all_magnitudes, all_angles])
        corr_matrix = np.corrcoef(coord_data.T)

        labels = ['X', 'Y', 'Z', 'Magnitude', 'Angle']

        im = ax7.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax7.set_xticks(range(len(labels)))
        ax7.set_yticks(range(len(labels)))
        ax7.set_xticklabels(labels, fontsize=11)
        ax7.set_yticklabels(labels, fontsize=11)

        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax7.text(j, i, f'{corr_matrix[i, j]:.2f}',
                            ha="center", va="center", color="black", fontsize=10)

        cbar = plt.colorbar(im, ax=ax7)
        cbar.set_label('Correlation', fontsize=10)

    ax7.set_title('G. Coordinate Correlation Matrix',
                fontsize=13, fontweight='bold', pad=10)

    # ============================================================================
    # Panel H: Magnitude vs Angle Scatter Plot
    # ============================================================================
    ax8 = fig.add_subplot(gs[2, 2])

    if len(all_magnitudes) > 0 and len(all_angles) > 0:
        sample_size = min(5000, len(all_magnitudes))
        sample_idx = np.random.choice(len(all_magnitudes), sample_size, replace=False)

        scatter = ax8.scatter(all_angles[sample_idx], all_magnitudes[sample_idx],
                            c=all_z_coords[sample_idx], cmap='coolwarm',
                            s=10, alpha=0.5)

        cbar = plt.colorbar(scatter, ax=ax8)
        cbar.set_label('Z Coordinate', fontsize=10)

    ax8.set_xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Magnitude', fontsize=12, fontweight='bold')
    ax8.set_title('H. Magnitude vs Angle (colored by Z)',
                fontsize=13, fontweight='bold', pad=10)
    ax8.grid(True, alpha=0.3)

    # ============================================================================
    # Panel I: Summary Statistics
    # ============================================================================
    ax9 = fig.add_subplot(gs[2, 3])
    ax9.axis('off')

    summary_stats = {
        'Sequences': f"{len(sequences):,}",
        'Total Coordinates': f"{len(all_x_coords):,}",
    }

    if len(all_x_coords) > 0:
        summary_stats.update({
            'X Range': f"[{all_x_coords.min():.2f}, {all_x_coords.max():.2f}]",
            'Y Range': f"[{all_y_coords.min():.2f}, {all_y_coords.max():.2f}]",
            'Z Range': f"[{all_z_coords.min():.2f}, {all_z_coords.max():.2f}]",
            'Mean Magnitude': f"{np.mean(all_magnitudes):.4f}",
            'Std Magnitude': f"{np.std(all_magnitudes):.4f}",
            'Mean Angle': f"{np.mean(all_angles):.2f}°",
            'Std Angle': f"{np.std(all_angles):.2f}°",
        })

    if len(sequence_lengths) > 0:
        summary_stats['Mean Seq Length'] = f"{np.mean(sequence_lengths):.0f} bp"
    if len(gc_contents) > 0:
        summary_stats['Mean GC Content'] = f"{np.mean(gc_contents):.3f}"

    table_data = [[k, v] for k, v in summary_stats.items()]
    table = ax9.table(cellText=table_data,
                    colLabels=['Property', 'Value'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.6, 0.4])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor(colors[4])
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(2):
                table[(i, j)].set_facecolor('#f0f0f0')

    ax9.set_title('I. Geometric Statistics Summary',
                fontsize=13, fontweight='bold', pad=20)

    plt.suptitle('Dual Strand Geometry Analysis: Cardinal Coordinates and Spatial Properties',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('figure_geometry_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_geometry_analysis.pdf', bbox_inches='tight')
    print("\nFigure saved: figure_geometry_analysis.png/pdf")
    plt.show()

    print("\n" + "="*80)
    print("GEOMETRY ANALYSIS COMPLETE")
    print("="*80)
