"""
Script 1: Metabolic Charge Oscillations
Simulates ATP-driven [Mg2+], [K+], and pH oscillations and their effect on Debye screening
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.integrate import odeint

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

if __name__ == "__main__":
    # Physical constants
    k_B = 1.38e-23  # Boltzmann constant (J/K)
    T = 310  # Temperature (K, 37°C)
    e = 1.6e-19  # Elementary charge (C)
    epsilon_0 = 8.85e-12  # Vacuum permittivity (F/m)
    epsilon_r = 78  # Relative permittivity of water
    N_A = 6.022e23  # Avogadro's number

    def calculate_debye_length(K_conc, Mg_conc):
        """
        Calculate Debye screening length
        K_conc: K+ concentration (mM)
        Mg_conc: Mg2+ concentration (mM)
        Returns: Debye length (nm)
        """
        # Convert mM to M
        K_M = K_conc / 1000
        Mg_M = Mg_conc / 1000

        # Ionic strength (M)
        I = 0.5 * (K_M * 1**2 + Mg_M * 2**2)

        # Debye length (m)
        lambda_D = np.sqrt((epsilon_0 * epsilon_r * k_B * T) / (2 * N_A * e**2 * I))

        # Convert to nm
        return lambda_D * 1e9

    def metabolic_oscillations(t, ATP_period=5.0, NaK_period=0.5, gly_period=60.0):
        """
        Simulate metabolic oscillations
        Returns: [Mg2+], [K+], pH at time t
        """
        # Baseline values
        Mg_base = 0.3  # mM
        K_base = 140   # mM
        pH_base = 7.4

        # Oscillation amplitudes
        Mg_amp = 0.15  # 50% oscillation
        K_amp = 10     # 7% oscillation
        pH_amp = 0.2   # ±0.2 pH units

        # Angular frequencies
        omega_ATP = 2 * np.pi / ATP_period
        omega_NaK = 2 * np.pi / NaK_period
        omega_gly = 2 * np.pi / gly_period

        # Oscillations
        Mg = Mg_base + Mg_amp * np.cos(omega_ATP * t)
        K = K_base + K_amp * np.cos(omega_NaK * t + np.pi/4)
        pH = pH_base + pH_amp * np.cos(omega_gly * t + np.pi/2)

        return Mg, K, pH

    def electrostatic_potential(r, lambda_D, charge_density=-9.4e-10):
        """
        Calculate electrostatic potential around DNA
        r: distance from DNA (nm)
        lambda_D: Debye length (nm)
        charge_density: linear charge density (C/m)
        Returns: potential (mV)
        """
        r_m = r * 1e-9  # Convert to meters
        lambda_D_m = lambda_D * 1e-9

        # Potential (V)
        Phi = (charge_density / (2 * np.pi * epsilon_0 * epsilon_r)) * np.exp(-r_m / lambda_D_m)

        # Convert to mV
        return Phi * 1000

    def binding_energy(z_protein, Phi):
        """
        Calculate protein-DNA binding energy
        z_protein: protein charge (elementary charges)
        Phi: electrostatic potential (mV)
        Returns: binding energy (kT units)
        """
        Phi_V = Phi / 1000  # Convert to V
        E = z_protein * e * Phi_V
        return E / (k_B * T)

    # Simulation parameters
    t_max = 120  # seconds
    dt = 0.01
    time = np.arange(0, t_max, dt)

    # Calculate oscillations
    Mg_array = np.zeros_like(time)
    K_array = np.zeros_like(time)
    pH_array = np.zeros_like(time)
    lambda_D_array = np.zeros_like(time)
    Phi_array = np.zeros_like(time)
    binding_energy_array = np.zeros_like(time)

    for i, t in enumerate(time):
        Mg, K, pH = metabolic_oscillations(t)
        Mg_array[i] = Mg
        K_array[i] = K
        pH_array[i] = pH

        lambda_D = calculate_debye_length(K, Mg)
        lambda_D_array[i] = lambda_D

        # Potential at 2 nm from DNA
        Phi = electrostatic_potential(2.0, lambda_D)
        Phi_array[i] = Phi

        # Binding energy for +10e transcription factor
        E_bind = binding_energy(10, Phi)
        binding_energy_array[i] = E_bind

    # Create figure
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Ion concentrations
    ax1 = fig.add_subplot(gs[0, :])
    ax1_twin = ax1.twinx()

    line1 = ax1.plot(time, Mg_array, 'b-', linewidth=2, label='[Mg²⁺]')
    line2 = ax1_twin.plot(time, K_array, 'r-', linewidth=2, label='[K⁺]')

    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('[Mg²⁺] (mM)', color='b', fontsize=12, fontweight='bold')
    ax1_twin.set_ylabel('[K⁺] (mM)', color='r', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_title('A. Metabolic Ion Oscillations', fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 60)

    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.9)

    # Panel B: pH oscillations
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, pH_array, 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('pH', fontsize=12, fontweight='bold')
    ax2.set_title('B. pH Oscillations (Glycolysis)', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=7.4, color='k', linestyle='--', alpha=0.5, label='Baseline')
    ax2.legend()
    ax2.set_xlim(0, 120)

    # Panel C: Debye length oscillations
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time, lambda_D_array, 'purple', linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Debye Length λ_D (nm)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Charge Screening Length Oscillations', fontsize=13, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=np.mean(lambda_D_array), color='k', linestyle='--', alpha=0.5, label='Mean')
    ax3.legend()
    ax3.set_xlim(0, 60)

    # Panel D: Electrostatic potential
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(time, Phi_array, 'orange', linewidth=2)
    ax4.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Potential Φ (mV)', fontsize=12, fontweight='bold')
    ax4.set_title('D. DNA Surface Potential (r=2nm)', fontsize=13, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 60)

    # Panel E: Binding energy modulation
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(time, binding_energy_array, 'brown', linewidth=2)
    ax5.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Binding Energy (k_B T)', fontsize=12, fontweight='bold')
    ax5.set_title('E. TF Binding Energy Oscillations', fontsize=13, fontweight='bold', pad=10)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.set_xlim(0, 60)

    # Panel F: Phase space (Mg vs lambda_D)
    ax6 = fig.add_subplot(gs[2, 1])
    scatter = ax6.scatter(Mg_array[::100], lambda_D_array[::100],
                        c=time[::100], cmap='viridis', s=50, alpha=0.7)
    ax6.set_xlabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('λ_D (nm)', fontsize=12, fontweight='bold')
    ax6.set_title('F. Phase Space: [Mg²⁺] vs λ_D', fontsize=13, fontweight='bold', pad=10)
    ax6.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Time (s)', fontsize=10)

    # Panel G: Correlation matrix
    ax7 = fig.add_subplot(gs[2, 2])
    variables = np.column_stack([Mg_array, K_array, pH_array, lambda_D_array, Phi_array, binding_energy_array])
    corr_matrix = np.corrcoef(variables.T)
    labels = ['[Mg²⁺]', '[K⁺]', 'pH', 'λ_D', 'Φ', 'E_bind']

    im = ax7.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax7.set_xticks(range(len(labels)))
    ax7.set_yticks(range(len(labels)))
    ax7.set_xticklabels(labels, rotation=45, ha='right')
    ax7.set_yticklabels(labels)
    ax7.set_title('G. Correlation Matrix', fontsize=13, fontweight='bold', pad=10)

    # Add correlation values
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax7.text(j, i, f'{corr_matrix[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=9)

    cbar = plt.colorbar(im, ax=ax7)
    cbar.set_label('Correlation', fontsize=10)

    # Panel H: Frequency spectrum (FFT)
    ax8 = fig.add_subplot(gs[3, :2])

    # Compute FFT for Mg oscillations
    fft_Mg = np.fft.fft(Mg_array - np.mean(Mg_array))
    freqs = np.fft.fftfreq(len(time), dt)
    power_Mg = np.abs(fft_Mg)**2

    # Only positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    power_Mg_pos = power_Mg[pos_mask]

    # Convert to period
    periods = 1 / freqs_pos

    ax8.plot(periods, power_Mg_pos, 'b-', linewidth=2, label='[Mg²⁺] Power Spectrum')
    ax8.set_xlabel('Period (s)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Power (a.u.)', fontsize=12, fontweight='bold')
    ax8.set_title('H. Frequency Analysis: Dominant Oscillation Periods', fontsize=13, fontweight='bold', pad=10)
    ax8.set_xlim(0, 20)
    ax8.grid(True, alpha=0.3)
    ax8.axvline(x=5.0, color='r', linestyle='--', linewidth=2, label='ATP synthesis (5s)')
    ax8.axvline(x=0.5, color='g', linestyle='--', linewidth=2, label='Na⁺/K⁺-ATPase (0.5s)')
    ax8.legend()
    ax8.set_yscale('log')

    # Panel I: Summary statistics
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.axis('off')

    stats_text = f"""
    SUMMARY STATISTICS

    Ion Oscillations:
    • [Mg²⁺]: {np.mean(Mg_array):.3f} ± {np.std(Mg_array):.3f} mM
    • [K⁺]: {np.mean(K_array):.1f} ± {np.std(K_array):.1f} mM
    • pH: {np.mean(pH_array):.2f} ± {np.std(pH_array):.2f}

    Charge Screening:
    • λ_D: {np.mean(lambda_D_array):.3f} ± {np.std(lambda_D_array):.3f} nm
    • Oscillation: {100*np.std(lambda_D_array)/np.mean(lambda_D_array):.1f}%

    Electrostatic Effects:
    • Φ(2nm): {np.mean(Phi_array):.1f} ± {np.std(Phi_array):.1f} mV
    • E_bind: {np.mean(binding_energy_array):.1f} ± {np.std(binding_energy_array):.1f} k_BT
    • Modulation: {100*np.std(binding_energy_array)/np.abs(np.mean(binding_energy_array)):.1f}%

    Dominant Frequencies:
    • ATP synthesis: ~5 s period
    • Na⁺/K⁺-ATPase: ~0.5 s period
    • Glycolysis: ~60 s period

    Correlations:
    • [Mg²⁺] vs λ_D: r = {corr_matrix[0,3]:.3f}
    • λ_D vs Φ: r = {corr_matrix[3,4]:.3f}
    • Φ vs E_bind: r = {corr_matrix[4,5]:.3f}
    """

    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Metabolic Charge Oscillations and Electrostatic Regulation',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('figure1_metabolic_charge_oscillations.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure1_metabolic_charge_oscillations.pdf', bbox_inches='tight')
    print("Figure 1 saved: figure1_metabolic_charge_oscillations.png/pdf")
    plt.show()
