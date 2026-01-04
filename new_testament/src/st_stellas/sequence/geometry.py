"""
Script 1: Comprehensive Palindrome Analysis (FIXED)
Analyzes palindrome distribution, length patterns, and geometric properties
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

if __name__ == "__main__":
    # Load data
    print("Loading palindrome data...")
    try:
        with open('palindrome_detection_analysis.json', 'r') as f:
            palindrome_data = json.load(f)

        print("JSON structure keys:", palindrome_data.keys())

        # Extract data based on actual structure
        metadata = palindrome_data.get('analysis_metadata', {})
        palindrome_analysis = palindrome_data.get('palindrome_analysis', {})

        print("Palindrome analysis keys:", palindrome_analysis.keys())

        stats_data = palindrome_analysis.get('population_statistics', {})
        length_analysis = palindrome_analysis.get('palindrome_length_analysis', {})

        print("Length analysis keys:", length_analysis.keys())

        # Extract palindrome lengths
        palindrome_lengths = np.array(length_analysis.get('palindrome_lengths', []))
        length_distribution = length_analysis.get('length_distribution', {})

        print(f"Loaded {len(palindrome_lengths)} palindromes")

        if len(palindrome_lengths) == 0:
            raise ValueError("No palindrome lengths found in data")

        # Convert length distribution to arrays
        if length_distribution:
            unique_lengths = sorted([int(k) for k in length_distribution.keys()])
            length_counts = [length_distribution[str(l)] for l in unique_lengths]
        else:
            # Create distribution from palindrome_lengths array
            unique_vals, counts = np.unique(palindrome_lengths, return_counts=True)
            unique_lengths = unique_vals.tolist()
            length_counts = counts.tolist()

        print(f"Unique lengths: {len(unique_lengths)}")

    except FileNotFoundError:
        print("ERROR: palindrome_detection_analysis.json not found!")
        print("Please ensure the file is in the current directory.")
        exit(1)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        print("Attempting to print JSON structure...")
        try:
            with open('palindrome_detection_analysis.json', 'r') as f:
                data = json.load(f)
                print("\nTop-level keys:", list(data.keys()))
                for key in data.keys():
                    print(f"\n{key}:")
                    if isinstance(data[key], dict):
                        print("  Sub-keys:", list(data[key].keys()))
        except:
            pass
        exit(1)

    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # Color palette
    colors = sns.color_palette("husl", 8)

    # ============================================================================
    # Panel A: Palindrome Length Distribution (Histogram with KDE)
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, :2])

    # Create histogram
    n, bins, patches = ax1.hist(palindrome_lengths, bins=50, alpha=0.6,
                                color=colors[0], edgecolor='black', linewidth=0.5,
                                density=True, label='Observed')

    # Add KDE
    if len(palindrome_lengths) > 100:
        try:
            kde = gaussian_kde(palindrome_lengths)
            x_kde = np.linspace(palindrome_lengths.min(), palindrome_lengths.max(), 500)
            ax1.plot(x_kde, kde(x_kde), 'r-', linewidth=3, label='KDE', alpha=0.8)
        except:
            print("Warning: KDE calculation failed, skipping...")

    # Add mean and median
    mean_len = np.mean(palindrome_lengths)
    median_len = np.median(palindrome_lengths)
    ax1.axvline(mean_len, color='blue', linestyle='--', linewidth=2,
            label=f'Mean: {mean_len:.2f}')
    ax1.axvline(median_len, color='green', linestyle='--', linewidth=2,
            label=f'Median: {median_len:.2f}')

    ax1.set_xlabel('Palindrome Length (bp)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title('A. Palindrome Length Distribution with Kernel Density Estimate',
                fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Add statistics box
    stats_text = f'n = {len(palindrome_lengths):,}\nμ = {mean_len:.2f}\nσ = {np.std(palindrome_lengths):.2f}\nSkew = {stats.skew(palindrome_lengths):.2f}'
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ============================================================================
    # Panel B: Cumulative Distribution Function
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 2:])

    # Sort data for CDF
    sorted_lengths = np.sort(palindrome_lengths)
    cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

    ax2.plot(sorted_lengths, cdf, linewidth=2.5, color=colors[1], alpha=0.8)
    ax2.fill_between(sorted_lengths, 0, cdf, alpha=0.3, color=colors[1])

    # Add percentile lines
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(palindrome_lengths, p)
        ax2.axvline(val, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax2.text(val, 0.02, f'{p}%', rotation=90, fontsize=8,
                verticalalignment='bottom', color='red')

    ax2.set_xlabel('Palindrome Length (bp)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax2.set_title('B. Cumulative Distribution Function with Percentiles',
                fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # ============================================================================
    # Panel C: Length Frequency Bar Chart (Top 20)
    # ============================================================================
    ax3 = fig.add_subplot(gs[1, :2])

    # Get top 20 most common lengths
    top_n = min(20, len(unique_lengths))
    top_lengths = unique_lengths[:top_n]
    top_counts = length_counts[:top_n]

    bars = ax3.bar(range(len(top_lengths)), top_counts,
                color=colors[2], alpha=0.7, edgecolor='black', linewidth=1)

    # Color code by length category
    for i, (bar, length) in enumerate(zip(bars, top_lengths)):
        if length <= 5:
            bar.set_color(colors[0])  # Short
        elif length <= 10:
            bar.set_color(colors[2])  # Medium
        else:
            bar.set_color(colors[4])  # Long

    ax3.set_xlabel('Palindrome Length (bp)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title(f'C. Top {top_n} Most Common Palindrome Lengths',
                fontsize=13, fontweight='bold', pad=10)
    ax3.set_xticks(range(len(top_lengths)))
    ax3.set_xticklabels(top_lengths, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, top_counts)):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}',
                    ha='center', va='bottom', fontsize=7)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Short (≤5 bp)'),
        Patch(facecolor=colors[2], label='Medium (6-10 bp)'),
        Patch(facecolor=colors[4], label='Long (>10 bp)')
    ]
    ax3.legend(handles=legend_elements, loc='upper right')

    # ============================================================================
    # Panel D: Violin Plot by Length Category
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 2:])

    # Categorize palindromes
    categories = []
    for length in palindrome_lengths:
        if length <= 5:
            categories.append('Short\n(≤5 bp)')
        elif length <= 10:
            categories.append('Medium\n(6-10 bp)')
        elif length <= 20:
            categories.append('Long\n(11-20 bp)')
        else:
            categories.append('Very Long\n(>20 bp)')

    df_violin = pd.DataFrame({
        'Length': palindrome_lengths,
        'Category': categories
    })

    # Get data for each category
    cat_names = ['Short\n(≤5 bp)', 'Medium\n(6-10 bp)', 'Long\n(11-20 bp)', 'Very Long\n(>20 bp)']
    cat_data = []
    cat_positions = []

    for i, cat in enumerate(cat_names):
        data = df_violin[df_violin['Category'] == cat]['Length'].values
        if len(data) > 0:
            cat_data.append(data)
            cat_positions.append(i)

    # Create violin plot
    if cat_data:
        parts = ax4.violinplot(cat_data, positions=cat_positions,
                            showmeans=True, showmedians=True, widths=0.7)

        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)

    ax4.set_ylabel('Palindrome Length (bp)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Length Distribution by Category (Violin Plot)',
                fontsize=13, fontweight='bold', pad=10)
    ax4.set_xticks(range(len(cat_names)))
    ax4.set_xticklabels(cat_names)
    ax4.grid(True, alpha=0.3, axis='y')

    # ============================================================================
    # Panel E: Log-Log Plot (Power Law Analysis)
    # ============================================================================
    ax5 = fig.add_subplot(gs[2, 0])

    # Calculate frequency
    unique_vals, counts = np.unique(palindrome_lengths, return_counts=True)
    # Sort by length
    sort_idx = np.argsort(unique_vals)
    unique_vals = unique_vals[sort_idx]
    counts = counts[sort_idx]

    # Filter out zeros for log scale
    mask = (unique_vals > 0) & (counts > 0)
    unique_vals_log = unique_vals[mask]
    counts_log = counts[mask]

    ax5.loglog(unique_vals_log, counts_log, 'o-', color=colors[5],
            markersize=6, linewidth=2, alpha=0.7)

    # Fit power law
    if len(unique_vals_log) > 2:
        try:
            log_x = np.log10(unique_vals_log)
            log_y = np.log10(counts_log)
            slope, intercept = np.polyfit(log_x, log_y, 1)

            fit_x = unique_vals_log
            fit_y = 10**intercept * fit_x**slope
            ax5.loglog(fit_x, fit_y, 'r--', linewidth=2, alpha=0.7,
                    label=f'Power law fit: y ∝ x^{slope:.2f}')
            ax5.legend()
        except:
            print("Warning: Power law fit failed")

    ax5.set_xlabel('Palindrome Length (bp)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title('E. Power Law Analysis (Log-Log Scale)',
                fontsize=13, fontweight='bold', pad=10)
    ax5.grid(True, alpha=0.3, which='both')

    # ============================================================================
    # Panel F: Box Plot with Outliers
    # ============================================================================
    ax6 = fig.add_subplot(gs[2, 1])

    bp = ax6.boxplot([palindrome_lengths], vert=True, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(facecolor=colors[6], alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(color='blue', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor=colors[6],
                                markersize=3, alpha=0.3))

    ax6.set_ylabel('Palindrome Length (bp)', fontsize=12, fontweight='bold')
    ax6.set_title('F. Box Plot with Outlier Detection',
                fontsize=13, fontweight='bold', pad=10)
    ax6.set_xticks([1])
    ax6.set_xticklabels(['All Palindromes'])
    ax6.grid(True, alpha=0.3, axis='y')

    # Add quartile annotations
    q1, median, q3 = np.percentile(palindrome_lengths, [25, 50, 75])
    iqr = q3 - q1
    ax6.text(1.15, q1, f'Q1: {q1:.1f}', fontsize=9, verticalalignment='center')
    ax6.text(1.15, median, f'Median: {median:.1f}', fontsize=9, verticalalignment='center')
    ax6.text(1.15, q3, f'Q3: {q3:.1f}', fontsize=9, verticalalignment='center')

    # ============================================================================
    # Panel G: Heatmap of Length vs Position (binned)
    # ============================================================================
    ax7 = fig.add_subplot(gs[2, 2])

    # Create 2D histogram
    positions = np.arange(len(palindrome_lengths))
    position_bins = 50
    length_bins = 30

    H, xedges, yedges = np.histogram2d(positions, palindrome_lengths,
                                    bins=[position_bins, length_bins])

    im = ax7.imshow(H.T, aspect='auto', origin='lower', cmap='YlOrRd',
                extent=[0, len(palindrome_lengths),
                        palindrome_lengths.min(), palindrome_lengths.max()])

    ax7.set_xlabel('Palindrome Index', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Length (bp)', fontsize=12, fontweight='bold')
    ax7.set_title('G. Length Distribution Heatmap',
                fontsize=13, fontweight='bold', pad=10)

    cbar = plt.colorbar(im, ax=ax7)
    cbar.set_label('Frequency', fontsize=10)

    # ============================================================================
    # Panel H: Summary Statistics Table
    # ============================================================================
    ax8 = fig.add_subplot(gs[2, 3])
    ax8.axis('off')

    # Calculate comprehensive statistics
    summary_stats = {
        'Total Palindromes': f"{len(palindrome_lengths):,}",
        'Unique Lengths': f"{len(unique_lengths)}",
        'Min Length': f"{palindrome_lengths.min()} bp",
        'Max Length': f"{palindrome_lengths.max()} bp",
        'Mean Length': f"{np.mean(palindrome_lengths):.2f} bp",
        'Median Length': f"{np.median(palindrome_lengths):.2f} bp",
        'Std Dev': f"{np.std(palindrome_lengths):.2f} bp",
        'Variance': f"{np.var(palindrome_lengths):.2f}",
        'Skewness': f"{stats.skew(palindrome_lengths):.3f}",
        'Kurtosis': f"{stats.kurtosis(palindrome_lengths):.3f}",
        'Q1 (25%)': f"{np.percentile(palindrome_lengths, 25):.1f} bp",
        'Q3 (75%)': f"{np.percentile(palindrome_lengths, 75):.1f} bp",
        'IQR': f"{np.percentile(palindrome_lengths, 75) - np.percentile(palindrome_lengths, 25):.1f} bp",
        '95th Percentile': f"{np.percentile(palindrome_lengths, 95):.1f} bp",
        '99th Percentile': f"{np.percentile(palindrome_lengths, 99):.1f} bp",
    }

    # Create table
    table_data = [[k, v] for k, v in summary_stats.items()]
    table = ax8.table(cellText=table_data,
                    colLabels=['Statistic', 'Value'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.6, 0.4])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor(colors[7])
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(2):
                table[(i, j)].set_facecolor('#f0f0f0')

    ax8.set_title('H. Comprehensive Statistics Summary',
                fontsize=13, fontweight='bold', pad=20)

    # ============================================================================
    # Main title and save
    # ============================================================================
    plt.suptitle('Palindrome Analysis: Length Distribution and Statistical Properties',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('figure_palindrome_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_palindrome_analysis.pdf', bbox_inches='tight')
    print("\nFigure saved: figure_palindrome_analysis.png/pdf")
    plt.show()

    print("\n" + "="*80)
    print("PALINDROME ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total palindromes analyzed: {len(palindrome_lengths):,}")
    print(f"Length range: {palindrome_lengths.min()} - {palindrome_lengths.max()} bp")
    try:
        mode_val = stats.mode(palindrome_lengths, keepdims=True)[0][0]
        print(f"Most common length: {mode_val} bp")
    except:
        print(f"Most common length: {unique_lengths[np.argmax(length_counts)]} bp")
    print(f"Mean ± SD: {np.mean(palindrome_lengths):.2f} ± {np.std(palindrome_lengths):.2f} bp")
