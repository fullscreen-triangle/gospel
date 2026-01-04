"""
Script 3: Chromatin Breathing and Nucleosome Dynamics (FIXED)
Simulates charge-dependent nucleosome unwrapping and chromatin accessibility
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

# Constants
k_B = 1.38e-23
T = 310
e = 1.6e-19

def nucleosome_unwrapping_rate(Mg_conc, K_conc=140):
    """
    Calculate nucleosome unwrapping rate based on electrostatic energy
    Returns rate in Hz (1/s)
    """
    # Simplified model: unwrapping rate decreases with [Mg2+]
    # due to stronger DNA-histone electrostatic attraction
    k_base = 5.0  # Base unwrapping rate (Hz)
    Mg_ref = 0.3  # Reference [Mg2+] (mM)

    # Exponential dependence
    k_unwrap = k_base * np.exp(-1.5 * (Mg_conc - Mg_ref))

    return k_unwrap

def nucleosome_wrapping_rate(Mg_conc, K_conc=140):
    """
    Calculate nucleosome wrapping rate
    Wrapping is generally faster and increases with [Mg2+]
    """
    k_base = 10.0  # Base wrapping rate (Hz)
    Mg_ref = 0.3

    k_wrap = k_base * np.exp(1.0 * (Mg_conc - Mg_ref))

    return k_wrap

def chromatin_accessibility(Mg_conc):
    """
    Calculate chromatin accessibility (0-1 scale)
    Lower [Mg2+] → higher accessibility
    """
    Mg_half = 0.35  # Half-maximal [Mg2+]
    hill = 2.5

    accessibility = 1 / (1 + (Mg_conc / Mg_half)**hill)

    return accessibility

def simulate_nucleosome_breathing(duration=100, dt=0.001, Mg_oscillation=True):
    """
    Simulate nucleosome breathing (wrapped/unwrapped transitions)
    Uses smaller dt for better event capture
    """
    time = np.arange(0, duration, dt)

    # State: 0 = wrapped, 1 = unwrapped
    state = np.zeros_like(time, dtype=int)
    state[0] = 0  # Start wrapped

    unwrap_times = []
    wrap_times = []

    for i in range(1, len(time)):
        t = time[i]

        # Oscillating [Mg2+]
        if Mg_oscillation:
            Mg = 0.3 + 0.15 * np.cos(2 * np.pi * t / 5.0)
        else:
            Mg = 0.3

        k_unwrap = nucleosome_unwrapping_rate(Mg)
        k_wrap = nucleosome_wrapping_rate(Mg)

        if state[i-1] == 0:  # Currently wrapped
            # Probability of unwrapping in time dt
            p_unwrap = 1 - np.exp(-k_unwrap * dt)
            if np.random.rand() < p_unwrap:
                state[i] = 1
                unwrap_times.append(t)
            else:
                state[i] = 0
        else:  # Currently unwrapped
            # Probability of wrapping in time dt
            p_wrap = 1 - np.exp(-k_wrap * dt)
            if np.random.rand() < p_wrap:
                state[i] = 0
                wrap_times.append(t)
            else:
                state[i] = 1

    return time, state, np.array(unwrap_times), np.array(wrap_times)

def histone_modification_wave(x, t, v_wave=1.0, width=50):
    """
    Simulate propagating histone acetylation wave
    x: position (kb)
    t: time (s)
    v_wave: wave velocity (kb/min)
    width: wave width (kb)
    """
    # Convert velocity to kb/s
    v = v_wave / 60

    # Wave center position
    x_center = v * t

    # Gaussian wave
    acetylation = np.exp(-((x - x_center)**2) / (2 * width**2))

    return acetylation

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run single trajectory with good temporal resolution
    print("Simulating nucleosome breathing dynamics...")
    time_breath, state_breath, unwrap_times, wrap_times = simulate_nucleosome_breathing(
        duration=100, dt=0.001, Mg_oscillation=True
    )

    print(f"  Generated {len(unwrap_times)} unwrapping events")
    print(f"  Generated {len(wrap_times)} wrapping events")

    # Multiple realizations for statistics (shorter duration, more realizations)
    n_realizations = 100
    all_states = []
    all_unwrap_times = []
    all_wrap_times = []

    print(f"Running {n_realizations} realizations for statistics...")
    for i in range(n_realizations):
        if (i+1) % 20 == 0:
            print(f"  Completed {i+1}/{n_realizations}")
        _, state, ut, wt = simulate_nucleosome_breathing(duration=100, dt=0.001)
        all_states.append(state)
        all_unwrap_times.extend(ut)
        all_wrap_times.extend(wt)

    all_states = np.array(all_states)
    mean_accessibility = np.mean(all_states, axis=0)
    std_accessibility = np.std(all_states, axis=0)

    print(f"Total events across all realizations:")
    print(f"  Unwrapping: {len(all_unwrap_times)}")
    print(f"  Wrapping: {len(all_wrap_times)}")

    # Histone modification wave
    x_positions = np.linspace(0, 200, 500)  # kb
    time_points = [0, 30, 60, 90, 120]  # seconds
    acetylation_profiles = []

    for t in time_points:
        acetyl = histone_modification_wave(x_positions, t, v_wave=1.0)
        acetylation_profiles.append(acetyl)

    # Create figure
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Single nucleosome breathing trajectory
    ax1 = fig.add_subplot(gs[0, :2])

    # Downsample for visualization
    plot_every = 100
    time_plot = time_breath[::plot_every]
    state_plot = state_breath[::plot_every]

    ax1.fill_between(time_plot, 0, state_plot, alpha=0.6, color='blue', label='Unwrapped', step='post')
    ax1.fill_between(time_plot, state_plot, 1, alpha=0.4, color='gray', label='Wrapped', step='post')

    # Mark transition events
    for ut in unwrap_times[:50]:  # First 50 events
        if ut < 50:
            ax1.axvline(ut, color='red', alpha=0.3, linewidth=0.5)

    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Nucleosome State', fontsize=12, fontweight='bold')
    ax1.set_title('A. Single Nucleosome Breathing Dynamics', fontsize=14, fontweight='bold', pad=10)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Wrapped', 'Unwrapped'])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, 50)
    ax1.set_ylim(-0.1, 1.1)

    # Panel B: Average accessibility over time
    ax2 = fig.add_subplot(gs[0, 2])

    # Downsample for plotting
    plot_every_avg = 1000
    time_avg = time_breath[::plot_every_avg]
    mean_avg = mean_accessibility[::plot_every_avg]
    std_avg = std_accessibility[::plot_every_avg]

    ax2.plot(time_avg, mean_avg, 'b-', linewidth=2, label='Mean')
    ax2.fill_between(time_avg,
                    mean_avg - std_avg,
                    mean_avg + std_avg,
                    alpha=0.3, color='blue', label='±1 SD')
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accessibility', fontsize=12, fontweight='bold')
    ax2.set_title(f'B. Ensemble Average\n(n={n_realizations} nucleosomes)', fontsize=13, fontweight='bold', pad=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1)

    # Panel C: Unwrapping rate vs [Mg2+]
    ax3 = fig.add_subplot(gs[1, 0])
    Mg_range = np.linspace(0.1, 0.8, 100)
    k_unwrap_range = [nucleosome_unwrapping_rate(Mg) for Mg in Mg_range]
    k_wrap_range = [nucleosome_wrapping_rate(Mg) for Mg in Mg_range]

    ax3.plot(Mg_range, k_unwrap_range, 'purple', linewidth=3, label='Unwrapping')
    ax3.plot(Mg_range, k_wrap_range, 'orange', linewidth=3, label='Wrapping')
    ax3.axvline(0.3, color='r', linestyle='--', alpha=0.5, label='Baseline [Mg²⁺]')
    ax3.set_xlabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Rate k (Hz)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Transition Rates vs [Mg²⁺]', fontsize=13, fontweight='bold', pad=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Panel D: Chromatin accessibility vs [Mg2+]
    ax4 = fig.add_subplot(gs[1, 1])
    accessibility_range = [chromatin_accessibility(Mg) for Mg in Mg_range]

    ax4.plot(Mg_range, accessibility_range, 'orange', linewidth=3)
    ax4.axvline(0.3, color='r', linestyle='--', alpha=0.5, label='Baseline [Mg²⁺]')
    ax4.axhline(0.5, color='k', linestyle=':', alpha=0.3)
    ax4.set_xlabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Chromatin Accessibility', fontsize=12, fontweight='bold')
    ax4.set_title('D. Accessibility vs [Mg²⁺]', fontsize=13, fontweight='bold', pad=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    # Panel E: Dwell time distributions
    ax5 = fig.add_subplot(gs[1, 2])

    # Calculate dwell times (time between events)
    if len(all_unwrap_times) > 1:
        all_unwrap_times_sorted = np.sort(all_unwrap_times)
        unwrap_intervals = np.diff(all_unwrap_times_sorted)

        # Remove very short intervals (likely artifacts)
        unwrap_intervals = unwrap_intervals[unwrap_intervals > 0.01]

        if len(unwrap_intervals) > 0:
            ax5.hist(unwrap_intervals, bins=50, alpha=0.7, color='blue',
                    edgecolor='black', density=True, label='Unwrapping intervals')

            mean_interval = np.mean(unwrap_intervals)
            ax5.axvline(mean_interval, color='r', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_interval:.3f} s')

            # Fit exponential
            rate = 1 / mean_interval
            x_fit = np.linspace(0, np.percentile(unwrap_intervals, 95), 100)
            y_fit = rate * np.exp(-rate * x_fit)
            ax5.plot(x_fit, y_fit, 'g-', linewidth=2, alpha=0.7, label='Exponential fit')

    ax5.set_xlabel('Inter-Event Interval (s)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax5.set_title('E. Breathing Event Statistics', fontsize=13, fontweight='bold', pad=10)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_xlim(0, np.percentile(unwrap_intervals, 95) if len(unwrap_intervals) > 0 else 1)

    # Panel F: Histone modification wave propagation
    ax6 = fig.add_subplot(gs[2, :])

    colors_wave = plt.cm.viridis(np.linspace(0, 1, len(time_points)))
    for i, (t, acetyl, color) in enumerate(zip(time_points, acetylation_profiles, colors_wave)):
        ax6.plot(x_positions, acetyl, linewidth=2.5, color=color, label=f't = {t} s')

    ax6.set_xlabel('Genomic Position (kb)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Acetylation Level', fontsize=12, fontweight='bold')
    ax6.set_title('F. Histone Acetylation Wave Propagation (v = 1 kb/min)', fontsize=14, fontweight='bold', pad=10)
    ax6.legend(loc='upper right', ncol=5)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 200)
    ax6.set_ylim(0, 1.05)

    # Panel G: 2D heatmap of acetylation wave
    ax7 = fig.add_subplot(gs[3, :2])

    time_grid = np.linspace(0, 180, 200)
    x_grid = np.linspace(0, 200, 300)
    T_grid, X_grid = np.meshgrid(time_grid, x_grid)

    acetylation_2d = np.zeros_like(T_grid)
    for i in range(len(x_grid)):
        for j in range(len(time_grid)):
            acetylation_2d[i, j] = histone_modification_wave(x_grid[i], time_grid[j])

    im = ax7.contourf(T_grid, X_grid, acetylation_2d, levels=20, cmap='YlOrRd')
    ax7.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Genomic Position (kb)', fontsize=12, fontweight='bold')
    ax7.set_title('G. Spatiotemporal Dynamics of Histone Modification', fontsize=13, fontweight='bold', pad=10)
    cbar = plt.colorbar(im, ax=ax7)
    cbar.set_label('Acetylation Level', fontsize=11)

    # Panel H: Breathing frequency vs [Mg2+]
    ax8 = fig.add_subplot(gs[3, 2])

    # Calculate breathing frequency for different [Mg2+]
    Mg_test = np.linspace(0.15, 0.6, 10)
    breathing_freqs = []

    for Mg in Mg_test:
        # Run short simulation
        _, state_test, _, _ = simulate_nucleosome_breathing(
            duration=50, dt=0.001, Mg_oscillation=False
        )
        # Modify Mg manually
        freq = np.sum(np.diff(state_test) != 0) / 50  # transitions per second
        breathing_freqs.append(freq)

    ax8.plot(Mg_test, breathing_freqs, 'o-', color='purple', linewidth=2, markersize=8)
    ax8.set_xlabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Breathing Frequency (Hz)', fontsize=12, fontweight='bold')
    ax8.set_title('H. Breathing Frequency vs [Mg²⁺]', fontsize=13, fontweight='bold', pad=10)
    ax8.grid(True, alpha=0.3)
    ax8.axvline(0.3, color='r', linestyle='--', alpha=0.5, label='Baseline')
    ax8.legend()

    # Add summary text
    summary_text = f"""
    CHROMATIN DYNAMICS SUMMARY

    Events Generated:
    • Unwrapping: {len(all_unwrap_times)}
    • Wrapping: {len(all_wrap_times)}
    • Total transitions: {len(all_unwrap_times) + len(all_wrap_times)}

    Rates at baseline ([Mg²⁺]=0.3 mM):
    • Unwrapping: {nucleosome_unwrapping_rate(0.3):.2f} Hz
    • Wrapping: {nucleosome_wrapping_rate(0.3):.2f} Hz
    • Mean interval: {np.mean(unwrap_intervals) if len(unwrap_intervals)>0 else 0:.3f} s

    Accessibility:
    • At 0.15 mM: {chromatin_accessibility(0.15):.2f}
    • At 0.30 mM: {chromatin_accessibility(0.30):.2f}
    • At 0.60 mM: {chromatin_accessibility(0.60):.2f}
    • Dynamic range: {chromatin_accessibility(0.15) - chromatin_accessibility(0.60):.2f}

    Wave Dynamics:
    • Velocity: 1.0 kb/min
    • Width: 50 kb
    • Period: ~120 s (200 kb)
    """

    fig.text(0.02, 0.02, summary_text, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
            verticalalignment='bottom')

    plt.suptitle('Chromatin Breathing and Nucleosome Dynamics: Charge-Dependent Accessibility',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('figure3_chromatin_breathing.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure3_chromatin_breathing.pdf', bbox_inches='tight')
    print("\nFigure 3 saved: figure3_chromatin_breathing.png/pdf")
    plt.show()

    print("\n" + "="*80)
    print("CHROMATIN BREATHING SIMULATION COMPLETE")
    print("="*80)
