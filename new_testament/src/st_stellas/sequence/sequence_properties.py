"""
Script 3: Genome Parser Results and Sequence Properties
Analyzes sequence composition, base distributions, and parser statistics
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
import pandas as pd
from collections import Counter

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("tab10")
plt.rcParams['figure.figsize'] = (20, 14)


if __name__ == "__main__":

    # Load data
    print("Loading genome parser results...")
    with open('genome_parser_results.json', 'r') as f:
        parser_data = json.load(f)

    metadata = parser_data['analysis_metadata']
    sequences = parser_data['sequences']

    print(f"Loaded {len(sequences)} sequences")

    # Extract sequence properties
    base_compositions = []
    gc_contents = []
    at_contents = []
    sequence_lengths = []
    n_counts = []

    for seq in sequences:
        if 'base_composition' in seq:
            base_compositions.append(seq['base_composition'])
        if 'gc_content' in seq:
            gc_contents.append(seq['gc_content'])
        if 'at_content' in seq:
            at_contents.append(seq['at_content'])
        if 'sequence_length' in seq:
            sequence_lengths.append(seq['sequence_length'])
        if 'n_count' in seq:
            n_counts.append(seq['n_count'])

    # Aggregate base counts
    total_bases = {'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0}
    for comp in base_compositions:
        for base, count in comp.items():
            if base in total_bases:
                total_bases[base] += count

    print(f"Total bases: {sum(total_bases.values()):,}")

    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.4)

    colors = sns.color_palette("tab10", 10)

    # ============================================================================
    # Panel A: Overall Base Composition (Pie Chart)
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    bases = ['A', 'T', 'G', 'C']
    counts = [total_bases[b] for b in bases]
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    wedges, texts, autotexts = ax1.pie(counts, labels=bases, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90,
                                        textprops={'fontsize': 11, 'weight': 'bold'})

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)

    ax1.set_title('A. Overall Base Composition',
                fontsize=13, fontweight='bold', pad=10)

    # ============================================================================
    # Panel B: Base Composition Bar Chart
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    bars = ax2.bar(bases, counts, color=colors_pie, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('B. Base Frequency Distribution',
                fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/sum(counts)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)

    # ============================================================================
    # Panel C: GC Content Distribution (Histogram with regions)
    # ============================================================================
    ax3 = fig.add_subplot(gs[0, 2:])

    if len(gc_contents) > 0:
        n, bins, patches = ax3.hist(gc_contents, bins=50, alpha=0.7,
                                    color=colors[2], edgecolor='black', linewidth=0.5)

        # Color code by GC content regions
        for i, patch in enumerate(patches):
            gc_val = (bins[i] + bins[i+1]) / 2
            if gc_val < 0.35:
                patch.set_facecolor('#ff6b6b')  # Low GC (AT-rich)
            elif gc_val < 0.45:
                patch.set_facecolor('#feca57')  # Medium-low GC
            elif gc_val < 0.55:
                patch.set_facecolor('#48dbfb')  # Medium GC
            elif gc_val < 0.65:
                patch.set_facecolor('#1dd1a1')  # Medium-high GC
            else:
                patch.set_facecolor('#5f27cd')  # High GC

        mean_gc = np.mean(gc_contents)
        median_gc = np.median(gc_contents)

        ax3.axvline(mean_gc, color='red', linestyle='--', linewidth=2.5,
                label=f'Mean: {mean_gc:.3f}')
        ax3.axvline(median_gc, color='blue', linestyle='--', linewidth=2.5,
                label=f'Median: {median_gc:.3f}')

        # Add region labels
        ax3.axvspan(0, 0.35, alpha=0.1, color='red', label='AT-rich')
        ax3.axvspan(0.45, 0.55, alpha=0.1, color='blue', label='Balanced')
        ax3.axvspan(0.65, 1, alpha=0.1, color='purple', label='GC-rich')

        ax3.set_xlabel('GC Content', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('C. GC Content Distribution with Genomic Regions',
                    fontsize=13, fontweight='bold', pad=10)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

    # ============================================================================
    # Panel D: GC vs AT Content Scatter Plot
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 0])

    if len(gc_contents) > 0 and len(at_contents) > 0:
        scatter = ax4.scatter(gc_contents, at_contents,
                            c=sequence_lengths if len(sequence_lengths) > 0 else 'blue',
                            cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Add diagonal line (GC + AT should = 1 for sequences without N)
        ax4.plot([0, 1], [1, 0], 'r--', linewidth=2, alpha=0.5, label='GC + AT = 1')

        ax4.set_xlabel('GC Content', fontsize=12, fontweight='bold')
        ax4.set_ylabel('AT Content', fontsize=12, fontweight='bold')
        ax4.set_title('D. GC vs AT Content Correlation',
                    fontsize=13, fontweight='bold', pad=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)

        if len(sequence_lengths) > 0:
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('Sequence Length (bp)', fontsize=10)

    # ============================================================================
    # Panel E: Sequence Length Distribution (Log scale)
    # ============================================================================
    ax5 = fig.add_subplot(gs[1, 1])

    if len(sequence_lengths) > 0:
        ax5.hist(sequence_lengths, bins=50, alpha=0.7, color=colors[4],
                edgecolor='black', linewidth=0.5)

        mean_len = np.mean(sequence_lengths)
        median_len = np.median(sequence_lengths)

        ax5.axvline(mean_len, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_len:.0f} bp')
        ax5.axvline(median_len, color='blue', linestyle='--', linewidth=2,
                label=f'Median: {median_len:.0f} bp')

        ax5.set_xlabel('Sequence Length (bp)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax5.set_title('E. Sequence Length Distribution',
                    fontsize=13, fontweight='bold', pad=10)
        ax5.set_yscale('log')
        ax5.legend()
        ax5.grid(True, alpha=0.3, which='both')

    # ============================================================================
    # Panel F: N Content Analysis (Box Plot)
    # ============================================================================
    ax6 = fig.add_subplot(gs[1, 2])

    if len(n_counts) > 0:
        # Calculate N percentage
        n_percentages = [n / length * 100 if length > 0 else 0
                        for n, length in zip(n_counts, sequence_lengths)]

        bp = ax6.boxplot([n_percentages], vert=True, widths=0.6, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(facecolor=colors[5], alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='blue', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        ax6.set_ylabel('N Content (%)', fontsize=12, fontweight='bold')
        ax6.set_title('F. Ambiguous Base (N) Content',
                    fontsize=13, fontweight='bold', pad=10)
        ax6.set_xticks([1])
        ax6.set_xticklabels(['All Sequences'])
        ax6.grid(True, alpha=0.3, axis='y')

        # Add statistics
        mean_n = np.mean(n_percentages)
        median_n = np.median(n_percentages)
        ax6.text(1.15, mean_n, f'Mean: {mean_n:.2f}%', fontsize=9)
        ax6.text(1.15, median_n, f'Median: {median_n:.2f}%', fontsize=9)

    # ============================================================================
    # Panel G: Base Composition Heatmap (per sequence)
    # ============================================================================
    ax7 = fig.add_subplot(gs[1, 3])

    if len(base_compositions) > 0:
        # Create matrix of base percentages
        n_seqs = min(50, len(base_compositions))  # Show first 50 sequences
        base_matrix = np.zeros((n_seqs, 4))

        for i, comp in enumerate(base_compositions[:n_seqs]):
            total = sum(comp.values())
            if total > 0:
                base_matrix[i, 0] = comp.get('A', 0) / total
                base_matrix[i, 1] = comp.get('T', 0) / total
                base_matrix[i, 2] = comp.get('G', 0) / total
                base_matrix[i, 3] = comp.get('C', 0) / total

        im = ax7.imshow(base_matrix.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5)

        ax7.set_xlabel('Sequence Index', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Base', fontsize=12, fontweight='bold')
        ax7.set_yticks([0, 1, 2, 3])
        ax7.set_yticklabels(['A', 'T', 'G', 'C'])
        ax7.set_title(f'G. Base Composition Heatmap\n(First {n_seqs} Sequences)',
                    fontsize=13, fontweight='bold', pad=10)

        cbar = plt.colorbar(im, ax=ax7)
        cbar.set_label('Fraction', fontsize=10)

    # ============================================================================
    # Panel H: Purine vs Pyrimidine Content
    # ============================================================================
    ax8 = fig.add_subplot(gs[2, :2])

    if len(base_compositions) > 0:
        purines = []  # A + G
        pyrimidines = []  # T + C

        for comp in base_compositions:
            total = sum(comp.values())
            if total > 0:
                purine = (comp.get('A', 0) + comp.get('G', 0)) / total
                pyrimidine = (comp.get('T', 0) + comp.get('C', 0)) / total
                purines.append(purine)
                pyrimidines.append(pyrimidine)

        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(purines, pyrimidines, bins=40)

        im = ax8.imshow(H.T, origin='lower', aspect='auto', cmap='Blues',
                    extent=[0, 1, 0, 1])

        # Add diagonal line (should be close to Chargaff's rule)
        ax8.plot([0, 1], [1, 0], 'r--', linewidth=2, alpha=0.7, label='Purine = Pyrimidine')

        ax8.set_xlabel('Purine Content (A+G)', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Pyrimidine Content (T+C)', fontsize=12, fontweight='bold')
        ax8.set_title('H. Purine vs Pyrimidine Content (Chargaff\'s Rule)',
                    fontsize=13, fontweight='bold', pad=10)
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        cbar = plt.colorbar(im, ax=ax8)
        cbar.set_label('Density', fontsize=10)

    # ============================================================================
    # Panel I: Comprehensive Statistics Table
    # ============================================================================
    ax9 = fig.add_subplot(gs[2, 2:])
    ax9.axis('off')

    summary_stats = {
        'Total Sequences': f"{len(sequences):,}",
        'Total Bases': f"{sum(total_bases.values()):,}",
        'A Count': f"{total_bases['A']:,} ({total_bases['A']/sum(total_bases.values())*100:.2f}%)",
        'T Count': f"{total_bases['T']:,} ({total_bases['T']/sum(total_bases.values())*100:.2f}%)",
        'G Count': f"{total_bases['G']:,} ({total_bases['G']/sum(total_bases.values())*100:.2f}%)",
        'C Count': f"{total_bases['C']:,} ({total_bases['C']/sum(total_bases.values())*100:.2f}%)",
        'N Count': f"{total_bases['N']:,} ({total_bases['N']/sum(total_bases.values())*100:.2f}%)",
    }

    if len(gc_contents) > 0:
        summary_stats['Mean GC Content'] = f"{np.mean(gc_contents):.4f}"
        summary_stats['Std GC Content'] = f"{np.std(gc_contents):.4f}"

    if len(sequence_lengths) > 0:
        summary_stats['Mean Seq Length'] = f"{np.mean(sequence_lengths):.0f} bp"
        summary_stats['Median Seq Length'] = f"{np.median(sequence_lengths):.0f} bp"
        summary_stats['Total Genome Size'] = f"{sum(sequence_lengths):,} bp"

    # Chargaff's ratios
    if total_bases['A'] > 0 and total_bases['T'] > 0:
        summary_stats['A/T Ratio'] = f"{total_bases['A']/total_bases['T']:.4f}"
    if total_bases['G'] > 0 and total_bases['C'] > 0:
        summary_stats['G/C Ratio'] = f"{total_bases['G']/total_bases['C']:.4f}"

    table_data = [[k, v] for k, v in summary_stats.items()]
    table = ax9.table(cellText=table_data,
                    colLabels=['Statistic', 'Value'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.5, 0.5])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor(colors[6])
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(2):
                table[(i, j)].set_facecolor('#f0f0f0')

    ax9.set_title('I. Comprehensive Genome Statistics',
                fontsize=13, fontweight='bold', pad=20)

    plt.suptitle('Genome Parser Results: Sequence Composition and Base Distribution Analysis',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('figure_genome_parser.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_genome_parser.pdf', bbox_inches='tight')
    print("\nFigure saved: figure_genome_parser.png/pdf")
    plt.show()

    print("\n" + "="*80)
    print("GENOME PARSER ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total sequences: {len(sequences):,}")
    print(f"Total bases: {sum(total_bases.values()):,}")
    print(f"GC content: {np.mean(gc_contents):.3f}" if len(gc_contents) > 0 else "N/A")
