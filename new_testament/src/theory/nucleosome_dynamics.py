"""
Script 3: Chromatin Breathing and Nucleosome Dynamics
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
    """
    # Debye length
    I = 0.5 * (K_conc + 4 * Mg_conc) / 1000  # Convert to M
    epsilon_0 = 8.85e-12
    epsilon_r = 78
    N_A = 6.022e23
    lambda_D = np.sqrt((epsilon_0 * epsilon_r * k_B * T) / (2 * N_A * e**2 * I))

    # Electrostatic energy difference (wrapped vs unwrapped)
    Q_DNA = -294 * e  # 147 bp
    Q_histone = 200 * e
    r_wrap = 2e-9  # wrapped distance (m)
    r_unwrap = 10e-9  # unwrapped distance (m)

    E_wrap = (Q_DNA * Q_histone) / (4 * np.pi * epsilon_0 * epsilon_r * r_wrap) * np.exp(-r_wrap/lambda_D)
    E_unwrap = (Q_DNA * Q_histone) / (4 * np.pi * epsilon_0 * epsilon_r * r_unwrap) * np.exp(-r_unwrap/lambda_D)

    Delta_E = E_unwrap - E_wrap

    # Unwrapping rate (Arrhenius)
    k0 = 1.0  # baseline rate (1/s)
    k_unwrap = k0 * np.exp(-abs(Delta_E) / (k_B * T))

    return k_unwrap

def chromatin_accessibility(Mg_conc):
    """
    Calculate chromatin accessibility (0-1 scale)
    Lower [Mg2+] → higher accessibility
    """
    # Empirical sigmoid model
    Mg_half = 0.5  # Half-maximal [Mg2+]
    hill = 2

    accessibility = 1 / (1 + (Mg_conc / Mg_half)**hill)

    return accessibility

def simulate_nucleosome_breathing(duration=100, Mg_oscillation=True):
    """
    Simulate nucleosome breathing (wrapped/unwrapped transitions)
    """
    dt = 0.01
    time = np.arange(0, duration, dt)

    # State: 0 = wrapped, 1 = unwrapped
    state = np.zeros_like(time)
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
        k_wrap = 2.0  # Wrapping is faster (more favorable)

        if state[i-1] == 0:  # Currently wrapped
            # Probability of unwrapping
            p_unwrap = k_unwrap * dt
            if np.random.rand() < p_unwrap:
                state[i] = 1
                unwrap_times.append(t)
            else:
                state[i] = 0
        else:  # Currently unwrapped
            # Probability of wrapping
            p_wrap = k_wrap * dt
            if np.random.rand() < p_wrap:
                state[i] = 0
                wrap_times.append(t)
            else:
                state[i] = 1

    return time, state, unwrap_times, wrap_times

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
    # Run simulations
    time_breath, state_breath, unwrap_times, wrap_times = simulate_nucleosome_breathing(duration=100)

    # Multiple realizations for statistics
    n_realizations = 50
    all_states = []
    for _ in range(n_realizations):
        _, state, _, _ = simulate_nucleosome_breathing(duration=100)
        all_states.append(state)

    mean_accessibility = np.mean(all_states, axis=0)
    std_accessibility = np.std(all_states, axis=0)

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
    ax1.fill_between(time_breath, 0, state_breath, alpha=0.5, color='blue', label='Unwrapped')
    ax1.fill_between(time_breath, state_breath, 1, alpha=0.3, color='gray', label='Wrapped')
    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Nucleosome State', fontsize=12, fontweight='bold')
    ax1.set_title('A. Single Nucleosome Breathing Dynamics', fontsize=14, fontweight='bold', pad=10)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Wrapped', 'Unwrapped'])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, 50)

    # Panel B: Average accessibility over time
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(time_breath, mean_accessibility, 'b-', linewidth=2, label='Mean')
    ax2.fill_between(time_breath,
                    mean_accessibility - std_accessibility,
                    mean_accessibility + std_accessibility,
                    alpha=0.3, color='blue', label='±1 SD')
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accessibility', fontsize=12, fontweight='bold')
    ax2.set_title('B. Ensemble Average\n(n=50 nucleosomes)', fontsize=13, fontweight='bold', pad=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)

    # Panel C: Unwrapping rate vs [Mg2+]
    ax3 = fig.add_subplot(gs[1, 0])
    Mg_range = np.linspace(0.1, 2.0, 100)
    k_unwrap_range = [nucleosome_unwrapping_rate(Mg) for Mg in Mg_range]

    ax3.plot(Mg_range, k_unwrap_range, 'purple', linewidth=3)
    ax3.axvline(0.3, color='r', linestyle='--', alpha=0.5, label='Baseline [Mg²⁺]')
    ax3.set_xlabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Unwrapping Rate k (s⁻¹)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Unwrapping Rate vs [Mg²⁺]', fontsize=13, fontweight='bold', pad=10)
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

    # Panel E: Dwell time distributions
    ax5 = fig.add_subplot(gs[1, 2])

    # Calculate dwell times
    if len(unwrap_times) > 1:
        unwrap_intervals = np.diff(unwrap_times)
        ax5.hist(unwrap_intervals, bins=20, alpha=0.7, color='blue', edgecolor='black', label='Unwrapping intervals')
        ax5.axvline(np.mean(unwrap_intervals), color='r', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(unwrap_intervals):.2f} s')

    ax5.set_xlabel('Interval Between Unwrapping Events (s)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title('E. Breathing Event Statistics', fontsize=13, fontweight='bold', pad=10)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

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

    # Panel H: Summary statistics
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.axis('off')

    # Calculate statistics
    mean_unwrap_rate = nucleosome_unwrapping_rate(0.3)
    mean_accessibility = chromatin_accessibility(0.3)

    if len(unwrap_times) > 1:
        mean_interval = np.mean(np.diff(unwrap_times))
        breathing_freq = 1 / mean_interval
    else:
        mean_interval = 0
        breathing_freq = 0

    stats_text = f"""
    CHROMATIN DYNAMICS STATISTICS

    Nucleosome Breathing:
    • Unwrapping rate: {mean_unwrap_rate:.3f} s⁻¹
    • Mean interval: {mean_interval:.2f} s
    • Breathing frequency: {breathing_freq:.2f} Hz
    • Total unwrap events: {len(unwrap_times)}
    • Total wrap events: {len(wrap_times)}

    Accessibility:
    • Baseline: {mean_accessibility:.2f}
    • [Mg²⁺] = 0.1 mM: {chromatin_accessibility(0.1):.2f}
    • [Mg²⁺] = 1.0 mM: {chromatin_accessibility(1.0):.2f}
    • Dynamic range: {chromatin_accessibility(0.1) - chromatin_accessibility(1.0):.2f}

    Histone Modification Wave:
    • Velocity: 1.0 kb/min
    • Width: 50 kb
    • Period: ~120 s (for 200 kb)

    Charge Dependence:
    • Lower [Mg²⁺] → longer λ_D
    • Longer λ_D → weaker DNA-histone binding
    • Weaker binding → more breathing
    • More breathing → higher accessibility
    """

    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    plt.suptitle('Chromatin Breathing and Nucleosome Dynamics: Charge-Dependent Accessibility',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('figure3_chromatin_breathing.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure3_chromatin_breathing.pdf', bbox_inches='tight')
    print("Figure 3 saved: figure3_chromatin_breathing.png/pdf")
    plt.show()
