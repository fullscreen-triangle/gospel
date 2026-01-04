"""
Script 5: Transcriptional Bursting and Charge Avalanches
Simulates stochastic transcription with charge-dependent burst dynamics
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import poisson, expon

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (18, 14)

def burst_probability(Mg_conc, threshold=0.35):
    """
    Calculate probability of transcriptional burst
    Lower [Mg2+] → higher probability (better charge neutralization)
    """
    # Sigmoid function
    p_max = 0.8
    p_min = 0.1
    k = 20  # steepness

    p_burst = p_min + (p_max - p_min) / (1 + np.exp(k * (Mg_conc - threshold)))

    return p_burst

def burst_size_distribution(mean_size=50, alpha=1.5):
    """
    Generate burst size from power-law distribution with exponential cutoff
    (self-organized criticality)
    """
    # Power-law with cutoff
    s_min = 1
    s_max = 500

    # Generate from truncated power-law
    u = np.random.rand()
    if alpha == 1:
        s = s_min * np.exp(u * np.log(s_max / s_min))
    else:
        s = s_min * (1 + u * ((s_max/s_min)**(1-alpha) - 1))**(1/(1-alpha))

    # Apply exponential cutoff
    cutoff = 100
    if s > cutoff:
        s = cutoff * np.exp(-(s - cutoff) / mean_size)

    return int(s)

def simulate_bursting(duration=600, Mg_oscillation=True):
    """
    Simulate stochastic transcriptional bursting
    duration: simulation time (s)
    Returns: time points, burst events, mRNA count
    """
    dt = 0.1  # time step (s)
    time = np.arange(0, duration, dt)

    burst_times = []
    burst_sizes = []
    mRNA_count = np.zeros_like(time)

    gamma_mRNA = 0.001  # mRNA degradation rate (1/s, half-life ~700s)

    for i, t in enumerate(time):
        # Oscillating [Mg2+]
        if Mg_oscillation:
            Mg = 0.3 + 0.15 * np.cos(2 * np.pi * t / 5.0)
        else:
            Mg = 0.3

        # Burst probability
        p_burst = burst_probability(Mg)

        # Check for burst
        if np.random.rand() < p_burst * dt:
            # Burst occurs
            size = burst_size_distribution()
            burst_times.append(t)
            burst_sizes.append(size)

            if i > 0:
                mRNA_count[i] = mRNA_count[i-1] + size
            else:
                mRNA_count[i] = size
        else:
            if i > 0:
                mRNA_count[i] = mRNA_count[i-1]
            else:
                mRNA_count[i] = 0

        # Degradation
        if i > 0:
            mRNA_count[i] -= gamma_mRNA * mRNA_count[i-1] * dt

    return time, np.array(burst_times), np.array(burst_sizes), mRNA_count

def calculate_burst_frequency(burst_times, window=60):
    """
    Calculate time-varying burst frequency
    window: time window for averaging (s)
    """
    if len(burst_times) == 0:
        return np.array([]), np.array([])

    max_time = burst_times[-1]
    time_bins = np.arange(0, max_time, window)
    frequencies = []

    for t in time_bins:
        count = np.sum((burst_times >= t) & (burst_times < t + window))
        freq = count / window  # bursts per second
        frequencies.append(freq)

    return time_bins, np.array(frequencies)

if __name__ == "__main__":
    # Run simulations
    np.random.seed(42)  # For reproducibility
    time_burst, burst_times, burst_sizes, mRNA_osc = simulate_bursting(duration=600, Mg_oscillation=True)
    time_const, burst_times_const, burst_sizes_const, mRNA_const = simulate_bursting(duration=600, Mg_oscillation=False)

    # Multiple realizations for statistics
    n_realizations = 20
    all_burst_sizes = []
    all_burst_frequencies = []

    for _ in range(n_realizations):
        _, bt, bs, _ = simulate_bursting(duration=600, Mg_oscillation=True)
        all_burst_sizes.extend(bs)
        if len(bt) > 0:
            _, freq = calculate_burst_frequency(bt, window=30)
            all_burst_frequencies.extend(freq)

    all_burst_sizes = np.array(all_burst_sizes)
    all_burst_frequencies = np.array(all_burst_frequencies)

    # Calculate burst frequency
    time_freq, freq_osc = calculate_burst_frequency(burst_times, window=30)
    time_freq_const, freq_const = calculate_burst_frequency(burst_times_const, window=30)

    # Get [Mg2+] time series
    Mg_burst = 0.3 + 0.15 * np.cos(2 * np.pi * time_burst / 5.0)

    # Create figure
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: mRNA count over time (oscillating)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(time_burst, mRNA_osc, 'b-', linewidth=1.5, alpha=0.8)

    # Mark burst events
    for bt, bs in zip(burst_times[:50], burst_sizes[:50]):  # First 50 bursts for clarity
        idx = np.argmin(np.abs(time_burst - bt))
        ax1.plot(bt, mRNA_osc[idx], 'ro', markersize=np.sqrt(bs)/2, alpha=0.6)

    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('mRNA Count', fontsize=12, fontweight='bold')
    ax1.set_title('A. Transcriptional Bursting Dynamics (Oscillating [Mg²⁺])', fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 300)

    # Panel B: [Mg2+] and burst events
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(time_burst[:3000], Mg_burst[:3000], 'g-', linewidth=2, alpha=0.7)

    # Overlay burst events
    burst_mask = burst_times < 300
    ax2.scatter(burst_times[burst_mask],
            0.3 + 0.15 * np.cos(2 * np.pi * burst_times[burst_mask] / 5.0),
            c='red', s=50, alpha=0.6, zorder=5)

    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Burst Events vs [Mg²⁺]', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 300)

    # Panel C: Burst size distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(all_burst_sizes, bins=50, color='purple', alpha=0.7, edgecolor='black', density=True)

    # Fit power-law
    bins = np.logspace(np.log10(1), np.log10(max(all_burst_sizes)), 30)
    hist, edges = np.histogram(all_burst_sizes, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    # Plot power-law fit
    ax3.plot(centers, hist, 'r-', linewidth=2, label='Empirical')
    ax3.set_xlabel('Burst Size (mRNA molecules)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax3.set_title('C. Burst Size Distribution', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel D: Burst frequency over time
    ax4 = fig.add_subplot(gs[1, 1])
    if len(time_freq) > 0:
        ax4.plot(time_freq, freq_osc, 'b-', linewidth=2, label='Oscillating [Mg²⁺]')
    if len(time_freq_const) > 0:
        ax4.plot(time_freq_const, freq_const, 'r--', linewidth=2, label='Constant [Mg²⁺]')

    ax4.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Burst Frequency (s⁻¹)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Time-Varying Burst Frequency', fontsize=13, fontweight='bold', pad=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Panel E: Burst probability vs [Mg2+]
    ax5 = fig.add_subplot(gs[1, 2])
    Mg_range = np.linspace(0.1, 0.6, 100)
    p_burst_range = [burst_probability(Mg) for Mg in Mg_range]

    ax5.plot(Mg_range, p_burst_range, 'orange', linewidth=3)
    ax5.axvline(0.3, color='k', linestyle='--', alpha=0.5, label='Baseline [Mg²⁺]')
    ax5.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    ax5.set_xlabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Burst Probability', fontsize=12, fontweight='bold')
    ax5.set_title('E. Burst Probability vs [Mg²⁺]', fontsize=13, fontweight='bold', pad=10)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel F: Inter-burst interval distribution
    ax6 = fig.add_subplot(gs[2, 0])
    if len(burst_times) > 1:
        intervals = np.diff(burst_times)
        ax6.hist(intervals, bins=50, color='brown', alpha=0.7, edgecolor='black', density=True)

        # Fit exponential
        rate = 1 / np.mean(intervals)
        x_fit = np.linspace(0, max(intervals), 100)
        y_fit = rate * np.exp(-rate * x_fit)
        ax6.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Exponential fit (λ={rate:.3f})')

    ax6.set_xlabel('Inter-Burst Interval (s)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax6.set_title('F. Inter-Burst Interval Distribution', fontsize=13, fontweight='bold', pad=10)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Panel G: mRNA count comparison (oscillating vs constant)
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.plot(time_burst, mRNA_osc, 'b-', linewidth=2, alpha=0.8, label='Oscillating [Mg²⁺]')
    ax7.plot(time_const, mRNA_const, 'r--', linewidth=2, alpha=0.8, label='Constant [Mg²⁺]')
    ax7.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('mRNA Count', fontsize=12, fontweight='bold')
    ax7.set_title('G. mRNA Dynamics: Oscillating vs Constant [Mg²⁺]', fontsize=13, fontweight='bold', pad=10)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, 600)

    # Panel H: Correlation between [Mg2+] and burst frequency
    ax8 = fig.add_subplot(gs[3, 0])

    # Calculate correlation
    if len(time_freq) > 0:
        # Interpolate [Mg2+] to burst frequency time points
        Mg_at_freq = 0.3 + 0.15 * np.cos(2 * np.pi * time_freq / 5.0)

        # Plot
        scatter = ax8.scatter(Mg_at_freq, freq_osc, c=time_freq, cmap='viridis', s=50, alpha=0.7)

        # Calculate correlation
        if len(Mg_at_freq) > 1:
            corr = np.corrcoef(Mg_at_freq, freq_osc)[0, 1]
            ax8.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax8.transAxes,
                    fontsize=12, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax8.set_xlabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Burst Frequency (s⁻¹)', fontsize=12, fontweight='bold')
    ax8.set_title('H. Correlation: [Mg²⁺] vs Burst Frequency', fontsize=13, fontweight='bold', pad=10)
    ax8.grid(True, alpha=0.3)
    if len(time_freq) > 0:
        cbar = plt.colorbar(scatter, ax=ax8)
        cbar.set_label('Time (s)', fontsize=10)

    # Panel I: Fano factor analysis
    ax9 = fig.add_subplot(gs[3, 1])

    # Calculate Fano factor (variance/mean) in sliding windows
    window_size = 100
    fano_factors = []
    window_times = []

    for i in range(0, len(mRNA_osc) - window_size, window_size//2):
        window = mRNA_osc[i:i+window_size]
        if np.mean(window) > 0:
            fano = np.var(window) / np.mean(window)
            fano_factors.append(fano)
            window_times.append(time_burst[i + window_size//2])

    if len(fano_factors) > 0:
        ax9.plot(window_times, fano_factors, 'purple', linewidth=2)
        ax9.axhline(1, color='r', linestyle='--', linewidth=2, label='Poisson (Fano=1)')
        ax9.axhline(np.mean(fano_factors), color='k', linestyle=':', alpha=0.5,
                label=f'Mean={np.mean(fano_factors):.2f}')

    ax9.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Fano Factor (Var/Mean)', fontsize=12, fontweight='bold')
    ax9.set_title('I. Fano Factor: Noise Analysis', fontsize=13, fontweight='bold', pad=10)
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # Panel J: Summary statistics
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.axis('off')

    # Calculate statistics
    total_bursts_osc = len(burst_times)
    total_bursts_const = len(burst_times_const)
    mean_burst_size = np.mean(all_burst_sizes)
    median_burst_size = np.median(all_burst_sizes)
    mean_interval = np.mean(np.diff(burst_times)) if len(burst_times) > 1 else 0
    mean_mRNA_osc = np.mean(mRNA_osc)
    mean_mRNA_const = np.mean(mRNA_const)
    mean_fano = np.mean(fano_factors) if len(fano_factors) > 0 else 0

    stats_text = f"""
    BURSTING STATISTICS

    Burst Events:
    • Total (oscillating): {total_bursts_osc}
    • Total (constant): {total_bursts_const}
    • Ratio: {total_bursts_osc/max(total_bursts_const,1):.2f}×

    Burst Size:
    • Mean: {mean_burst_size:.1f} mRNA
    • Median: {median_burst_size:.1f} mRNA
    • Max: {max(all_burst_sizes):.0f} mRNA
    • Distribution: Power-law (α≈1.5)

    Burst Timing:
    • Mean interval: {mean_interval:.2f} s
    • Frequency: {1/mean_interval if mean_interval>0 else 0:.3f} Hz

    mRNA Levels:
    • Mean (osc): {mean_mRNA_osc:.1f}
    • Mean (const): {mean_mRNA_const:.1f}
    • Ratio: {mean_mRNA_osc/max(mean_mRNA_const,1):.2f}×

    Noise:
    • Fano factor: {mean_fano:.2f}
    • Super-Poissonian: {mean_fano > 1}

    Charge Mechanism:
    • Low [Mg²⁺] → high P(burst)
    • Stochastic fluctuations
    • Avalanche dynamics
    • Self-organized criticality
    """

    ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes,
            fontsize=9.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    plt.suptitle('Transcriptional Bursting: Charge Avalanche Dynamics and Stochastic Gene Expression',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('figure5_transcriptional_bursting.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure5_transcriptional_bursting.pdf', bbox_inches='tight')
    print("Figure 5 saved: figure5_transcriptional_bursting.png/pdf")
    plt.show()
