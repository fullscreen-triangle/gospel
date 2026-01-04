"""
Script 4: Alternative Splicing Dynamics
Simulates charge-dependent isoform selection and splicing oscillations
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.integrate import odeint

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (18, 14)

def splicing_rate(Mg_conc, exon_type='A'):
    """
    Calculate exon inclusion rate based on [Mg2+]
    exon_type: 'A' (favored at low [Mg2+]) or 'B' (favored at high [Mg2+])
    """
    # Baseline rates
    k0_A = 1.0  # Exon A inclusion rate (1/min)
    k0_B = 0.5  # Exon B inclusion rate (1/min)

    # [Mg2+] dependence (empirical)
    # Low [Mg2+] favors exon A (e.g., PKM1)
    # High [Mg2+] favors exon B (e.g., PKM2)

    if exon_type == 'A':
        # Exon A: decreases with [Mg2+]
        k = k0_A * np.exp(-2 * (Mg_conc - 0.3))
    else:  # exon_type == 'B'
        # Exon B: increases with [Mg2+]
        k = k0_B * np.exp(2 * (Mg_conc - 0.3))

    return k

def isoform_dynamics(y, t, Mg_func):
    """
    ODE system for isoform concentrations
    y = [IsoformA, IsoformB]
    """
    IsoA, IsoB = y

    # Get current [Mg2+]
    Mg = Mg_func(t)

    # Splicing rates
    kA = splicing_rate(Mg, 'A')
    kB = splicing_rate(Mg, 'B')

    # Degradation rates
    gamma = 0.1  # 1/min (mRNA half-life ~7 min)

    # ODEs
    dIsoA_dt = kA - gamma * IsoA
    dIsoB_dt = kB - gamma * IsoB

    return [dIsoA_dt, dIsoB_dt]

def Mg_oscillating(t, period=5.0):
    """Oscillating [Mg2+] with given period (seconds)"""
    return 0.3 + 0.15 * np.cos(2 * np.pi * t / period)

def Mg_constant(t):
    """Constant [Mg2+]"""
    return 0.3

def Mg_circadian(t, period=86400):
    """Circadian [Mg2+] oscillation (24 hour period)"""
    return 0.3 + 0.06 * np.cos(2 * np.pi * t / period)

if __name__ == "__main__":
    # Simulation parameters
    t_short = np.linspace(0, 30, 3000)  # 30 minutes, high resolution
    t_long = np.linspace(0, 180, 1800)  # 3 hours
    t_circadian = np.linspace(0, 86400*2, 10000)  # 2 days

    # Initial conditions
    y0 = [5.0, 5.0]  # Equal initial isoform concentrations

    # Solve ODEs
    sol_osc = odeint(isoform_dynamics, y0, t_short, args=(Mg_oscillating,))
    sol_const = odeint(isoform_dynamics, y0, t_short, args=(Mg_constant,))
    sol_circ = odeint(isoform_dynamics, y0, t_circadian, args=(Mg_circadian,))

    IsoA_osc, IsoB_osc = sol_osc.T
    IsoA_const, IsoB_const = sol_const.T
    IsoA_circ, IsoB_circ = sol_circ.T

    # Calculate isoform ratios
    ratio_osc = IsoA_osc / (IsoB_osc + 1e-10)
    ratio_const = IsoA_const / (IsoB_const + 1e-10)
    ratio_circ = IsoA_circ / (IsoB_circ + 1e-10)

    # Get [Mg2+] time series
    Mg_osc = np.array([Mg_oscillating(t) for t in t_short])
    Mg_const = np.array([Mg_constant(t) for t in t_short])
    Mg_circ_array = np.array([Mg_circadian(t) for t in t_circadian])

    # Create figure
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: [Mg2+] oscillations (short timescale)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_short, Mg_osc, 'b-', linewidth=2, label='Oscillating')
    ax1.plot(t_short, Mg_const, 'r--', linewidth=2, label='Constant')
    ax1.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax1.set_title('A. [Mg²⁺] Dynamics', fontsize=13, fontweight='bold', pad=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B: Isoform concentrations (oscillating)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_short, IsoA_osc, 'b-', linewidth=2, label='Isoform A')
    ax2.plot(t_short, IsoB_osc, 'r-', linewidth=2, label='Isoform B')
    ax2.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Concentration (a.u.)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Isoform Dynamics (Oscillating [Mg²⁺])', fontsize=13, fontweight='bold', pad=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel C: Isoform concentrations (constant)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t_short, IsoA_const, 'b-', linewidth=2, label='Isoform A')
    ax3.plot(t_short, IsoB_const, 'r-', linewidth=2, label='Isoform B')
    ax3.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Concentration (a.u.)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Isoform Dynamics (Constant [Mg²⁺])', fontsize=13, fontweight='bold', pad=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel D: Isoform ratio (oscillating)
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(t_short, ratio_osc, 'purple', linewidth=2.5)
    ax4.axhline(np.mean(ratio_osc), color='k', linestyle='--', alpha=0.5, label='Mean')
    ax4.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Isoform Ratio (A/B)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Isoform Ratio Oscillations', fontsize=13, fontweight='bold', pad=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Panel E: Phase space (Mg vs ratio)
    ax5 = fig.add_subplot(gs[1, 2])
    scatter = ax5.scatter(Mg_osc[::10], ratio_osc[::10], c=t_short[::10],
                        cmap='viridis', s=30, alpha=0.7)
    ax5.set_xlabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Isoform Ratio (A/B)', fontsize=12, fontweight='bold')
    ax5.set_title('E. Phase Space: [Mg²⁺] vs Ratio', fontsize=13, fontweight='bold', pad=10)
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Time (min)', fontsize=10)

    # Panel F: Splicing rates vs [Mg2+]
    ax6 = fig.add_subplot(gs[2, 0])
    Mg_range = np.linspace(0.1, 0.6, 100)
    kA_range = [splicing_rate(Mg, 'A') for Mg in Mg_range]
    kB_range = [splicing_rate(Mg, 'B') for Mg in Mg_range]

    ax6.plot(Mg_range, kA_range, 'b-', linewidth=2.5, label='Exon A (e.g., PKM1)')
    ax6.plot(Mg_range, kB_range, 'r-', linewidth=2.5, label='Exon B (e.g., PKM2)')
    ax6.axvline(0.3, color='k', linestyle='--', alpha=0.5, label='Baseline [Mg²⁺]')
    ax6.set_xlabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Splicing Rate (min⁻¹)', fontsize=12, fontweight='bold')
    ax6.set_title('F. Splicing Rates vs [Mg²⁺]', fontsize=13, fontweight='bold', pad=10)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')

    # Panel G: Circadian modulation
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7_twin = ax7.twinx()

    line1 = ax7.plot(t_circadian/3600, Mg_circ_array, 'g-', linewidth=2, alpha=0.7, label='[Mg²⁺]')
    line2 = ax7_twin.plot(t_circadian/3600, ratio_circ, 'purple', linewidth=2, label='Isoform Ratio')

    ax7.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('[Mg²⁺] (mM)', color='g', fontsize=12, fontweight='bold')
    ax7_twin.set_ylabel('Isoform Ratio (A/B)', color='purple', fontsize=12, fontweight='bold')
    ax7.tick_params(axis='y', labelcolor='g')
    ax7_twin.tick_params(axis='y', labelcolor='purple')
    ax7.set_title('G. Circadian Modulation of Splicing (24h period)', fontsize=13, fontweight='bold', pad=10)
    ax7.grid(True, alpha=0.3)

    # Add day/night shading
    for day in range(3):
        ax7.axvspan(day*24, day*24+12, alpha=0.1, color='yellow', label='Day' if day==0 else '')
        ax7.axvspan(day*24+12, (day+1)*24, alpha=0.1, color='gray', label='Night' if day==0 else '')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax7.legend(lines, labels, loc='upper left', framealpha=0.9)

    # Panel H: Frequency spectrum of ratio oscillations
    ax8 = fig.add_subplot(gs[3, 0])

    # FFT of isoform ratio
    fft_ratio = np.fft.fft(ratio_osc - np.mean(ratio_osc))
    freqs = np.fft.fftfreq(len(t_short), t_short[1] - t_short[0])
    power_ratio = np.abs(fft_ratio)**2

    # Only positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    power_ratio_pos = power_ratio[pos_mask]

    # Convert to period (minutes)
    periods = 1 / freqs_pos

    ax8.plot(periods, power_ratio_pos, 'b-', linewidth=2)
    ax8.set_xlabel('Period (min)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Power (a.u.)', fontsize=12, fontweight='bold')
    ax8.set_title('H. Frequency Spectrum of Ratio', fontsize=13, fontweight='bold', pad=10)
    ax8.set_xlim(0, 10)
    ax8.grid(True, alpha=0.3)
    ax8.axvline(x=5.0/60, color='r', linestyle='--', linewidth=2, label='Expected (5s = 0.083 min)')
    ax8.legend()
    ax8.set_yscale('log')

    # Panel I: Comparison of oscillation amplitudes
    ax9 = fig.add_subplot(gs[3, 1])

    # Calculate oscillation amplitudes
    amp_Mg = (np.max(Mg_osc) - np.min(Mg_osc)) / (2 * np.mean(Mg_osc)) * 100
    amp_IsoA = (np.max(IsoA_osc) - np.min(IsoA_osc)) / (2 * np.mean(IsoA_osc)) * 100
    amp_IsoB = (np.max(IsoB_osc) - np.min(IsoB_osc)) / (2 * np.mean(IsoB_osc)) * 100
    amp_ratio = (np.max(ratio_osc) - np.min(ratio_osc)) / (2 * np.mean(ratio_osc)) * 100

    categories = ['[Mg²⁺]', 'Isoform A', 'Isoform B', 'Ratio A/B']
    amplitudes = [amp_Mg, amp_IsoA, amp_IsoB, amp_ratio]
    colors_bar = ['green', 'blue', 'red', 'purple']

    bars = ax9.bar(categories, amplitudes, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax9.set_ylabel('Oscillation Amplitude (%)', fontsize=12, fontweight='bold')
    ax9.set_title('I. Oscillation Amplitude Comparison', fontsize=13, fontweight='bold', pad=10)
    ax9.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, amp in zip(bars, amplitudes):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{amp:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel J: Summary statistics
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.axis('off')

    # Calculate statistics
    mean_ratio_osc = np.mean(ratio_osc)
    std_ratio_osc = np.std(ratio_osc)
    mean_ratio_const = np.mean(ratio_const)
    std_ratio_const = np.std(ratio_const)

    # Circadian statistics
    mean_ratio_circ = np.mean(ratio_circ)
    amp_ratio_circ = (np.max(ratio_circ) - np.min(ratio_circ)) / (2 * mean_ratio_circ) * 100

    stats_text = f"""
    SPLICING DYNAMICS STATISTICS

    Oscillating [Mg²⁺] (5s period):
    • Mean ratio: {mean_ratio_osc:.2f}
    • Std ratio: {std_ratio_osc:.2f}
    • Amplitude: {amp_ratio:.1f}%
    • IsoA amplitude: {amp_IsoA:.1f}%
    • IsoB amplitude: {amp_IsoB:.1f}%

    Constant [Mg²⁺]:
    • Mean ratio: {mean_ratio_const:.2f}
    • Std ratio: {std_ratio_const:.2f}
    • Amplitude: {(np.max(ratio_const)-np.min(ratio_const))/(2*mean_ratio_const)*100:.1f}%

    Circadian (24h period):
    • Mean ratio: {mean_ratio_circ:.2f}
    • Amplitude: {amp_ratio_circ:.1f}%

    Charge Mechanism:
    • Low [Mg²⁺] → long λ_D
    • Long λ_D → weak screening
    • Weak screening → Isoform A
    • High [Mg²⁺] → short λ_D
    • Short λ_D → strong screening
    • Strong screening → Isoform B

    Predicted vs Observed:
    • Oscillation period: 5s (ATP)
    • Amplitude modulation: ~{amp_ratio:.0f}%
    • Circadian modulation: ~{amp_ratio_circ:.0f}%
    """

    ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes,
            fontsize=9.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle('Alternative Splicing Dynamics: Charge-Dependent Isoform Selection',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('figure4_splicing_dynamics.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure4_splicing_dynamics.pdf', bbox_inches='tight')
    print("Figure 4 saved: figure4_splicing_dynamics.png/pdf")
    plt.show()
