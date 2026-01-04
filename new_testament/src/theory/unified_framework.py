"""
Script 6: Unified Framework - Multi-Scale Charge-Oscillatory Genomics
Integrates all levels: metabolism → charge → chromatin → transcription → splicing
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
plt.rcParams['figure.figsize'] = (20, 16)

# Physical constants
k_B = 1.38e-23
T = 310
e = 1.6e-19
epsilon_0 = 8.85e-12
epsilon_r = 78
N_A = 6.022e23

class ChargeGenomicsModel:
    """Unified charge-oscillatory genomics model"""

    def __init__(self):
        # Metabolic parameters
        self.ATP_period = 5.0  # seconds
        self.NaK_period = 0.5  # seconds
        self.gly_period = 60.0  # seconds

        # Ion baseline concentrations
        self.Mg_base = 0.3  # mM
        self.K_base = 140   # mM
        self.pH_base = 7.4

        # Oscillation amplitudes
        self.Mg_amp = 0.15
        self.K_amp = 10
        self.pH_amp = 0.2

    def metabolic_state(self, t):
        """Calculate metabolic state at time t"""
        omega_ATP = 2 * np.pi / self.ATP_period
        omega_NaK = 2 * np.pi / self.NaK_period
        omega_gly = 2 * np.pi / self.gly_period

        Mg = self.Mg_base + self.Mg_amp * np.cos(omega_ATP * t)
        K = self.K_base + self.K_amp * np.cos(omega_NaK * t + np.pi/4)
        pH = self.pH_base + self.pH_amp * np.cos(omega_gly * t + np.pi/2)

        return {'Mg': Mg, 'K': K, 'pH': pH}

    def debye_length(self, Mg, K):
        """Calculate Debye screening length"""
        I = 0.5 * (K/1000 + 4*Mg/1000)  # Convert to M
        lambda_D = np.sqrt((epsilon_0 * epsilon_r * k_B * T) / (2 * N_A * e**2 * I))
        return lambda_D * 1e9  # Convert to nm

    def chromatin_accessibility(self, Mg):
        """Calculate chromatin accessibility"""
        return 1 / (1 + (Mg / 0.5)**2)

    def transcription_rate(self, accessibility):
        """Calculate transcription rate"""
        k_max = 10.0  # mRNA/min
        return k_max * accessibility

    def splicing_rate(self, Mg, isoform='A'):
        """Calculate splicing rate for isoform"""
        k0 = 1.0 if isoform == 'A' else 0.5
        if isoform == 'A':
            return k0 * np.exp(-2 * (Mg - 0.3))
        else:
            return k0 * np.exp(2 * (Mg - 0.3))

    def replication_velocity(self, Mg, K):
        """Calculate replication fork velocity"""
        I = 0.5 * (K + 4*Mg)
        v_base = 50  # bp/s
        return v_base * (1 + 0.3 * (I - 71) / 71)

    def simulate(self, duration=300):
        """Run full simulation"""
        dt = 0.1
        time = np.arange(0, duration, dt)

        # Initialize arrays
        results = {
            'time': time,
            'Mg': np.zeros_like(time),
            'K': np.zeros_like(time),
            'pH': np.zeros_like(time),
            'lambda_D': np.zeros_like(time),
            'accessibility': np.zeros_like(time),
            'transcription_rate': np.zeros_like(time),
            'splicing_A': np.zeros_like(time),
            'splicing_B': np.zeros_like(time),
            'replication_v': np.zeros_like(time),
            'mRNA': np.zeros_like(time),
            'isoform_A': np.zeros_like(time),
            'isoform_B': np.zeros_like(time),
        }

        # Simulation loop
        for i, t in enumerate(time):
            # Metabolic state
            state = self.metabolic_state(t)
            results['Mg'][i] = state['Mg']
            results['K'][i] = state['K']
            results['pH'][i] = state['pH']

            # Charge screening
            results['lambda_D'][i] = self.debye_length(state['Mg'], state['K'])

            # Chromatin
            results['accessibility'][i] = self.chromatin_accessibility(state['Mg'])

            # Transcription
            results['transcription_rate'][i] = self.transcription_rate(results['accessibility'][i])

            # Splicing
            results['splicing_A'][i] = self.splicing_rate(state['Mg'], 'A')
            results['splicing_B'][i] = self.splicing_rate(state['Mg'], 'B')

            # Replication
            results['replication_v'][i] = self.replication_velocity(state['Mg'], state['K'])

            # mRNA dynamics (simple integration)
            gamma_mRNA = 0.1  # 1/min
            if i > 0:
                results['mRNA'][i] = results['mRNA'][i-1] + (results['transcription_rate'][i] - gamma_mRNA * results['mRNA'][i-1]) * dt/60
                results['isoform_A'][i] = results['isoform_A'][i-1] + (results['splicing_A'][i] - gamma_mRNA * results['isoform_A'][i-1]) * dt/60
                results['isoform_B'][i] = results['isoform_B'][i-1] + (results['splicing_B'][i] - gamma_mRNA * results['isoform_B'][i-1]) * dt/60

        return results

if __name__ == "__main__":
    # Run simulation
    model = ChargeGenomicsModel()
    results = model.simulate(duration=300)

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.35)

    # Color scheme
    colors = {
        'Mg': '#1f77b4',
        'K': '#ff7f0e',
        'pH': '#2ca02c',
        'lambda_D': '#d62728',
        'accessibility': '#9467bd',
        'transcription': '#8c564b',
        'splicing_A': '#e377c2',
        'splicing_B': '#7f7f7f',
        'replication': '#bcbd22',
    }

    # Panel A: Hierarchical flow diagram
    ax_flow = fig.add_subplot(gs[0, :])
    ax_flow.axis('off')

    flow_text = """
    HIERARCHICAL CHARGE-OSCILLATORY GENOMICS FRAMEWORK

    Level 1: METABOLISM  →  Level 2: CHARGE SCREENING  →  Level 3: CHROMATIN  →  Level 4: TRANSCRIPTION  →  Level 5: SPLICING
    [ATP], [Mg²⁺]              Debye Length λ_D           Accessibility          mRNA Production         Isoform Selection
    [K⁺], pH                   Electric Field Φ           Nucleosome Breathing   Burst Dynamics          PKM1/PKM2 Ratio
    """

    ax_flow.text(0.5, 0.5, flow_text, transform=ax_flow.transAxes,
                fontsize=12, ha='center', va='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, pad=1))

    # Panel B: Metabolic oscillations
    ax1 = fig.add_subplot(gs[1, :2])
    ax1_twin1 = ax1.twinx()
    ax1_twin2 = ax1.twinx()
    ax1_twin2.spines['right'].set_position(('outward', 60))

    l1 = ax1.plot(results['time'], results['Mg'], color=colors['Mg'], linewidth=2, label='[Mg²⁺]')
    l2 = ax1_twin1.plot(results['time'], results['K'], color=colors['K'], linewidth=2, label='[K⁺]')
    l3 = ax1_twin2.plot(results['time'], results['pH'], color=colors['pH'], linewidth=2, label='pH')

    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('[Mg²⁺] (mM)', color=colors['Mg'], fontsize=11, fontweight='bold')
    ax1_twin1.set_ylabel('[K⁺] (mM)', color=colors['K'], fontsize=11, fontweight='bold')
    ax1_twin2.set_ylabel('pH', color=colors['pH'], fontsize=11, fontweight='bold')

    ax1.tick_params(axis='y', labelcolor=colors['Mg'])
    ax1_twin1.tick_params(axis='y', labelcolor=colors['K'])
    ax1_twin2.tick_params(axis='y', labelcolor=colors['pH'])

    ax1.set_title('A. Level 1: Metabolic Oscillations', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 60)

    # Combine legends
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    # Panel C: Debye length
    ax2 = fig.add_subplot(gs[1, 2:])
    ax2.plot(results['time'], results['lambda_D'], color=colors['lambda_D'], linewidth=2.5)
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Debye Length λ_D (nm)', fontsize=11, fontweight='bold')
    ax2.set_title('B. Level 2: Charge Screening Length', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(np.mean(results['lambda_D']), color='k', linestyle='--', alpha=0.5, label='Mean')
    ax2.legend()
    ax2.set_xlim(0, 60)

    # Panel D: Chromatin accessibility
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(results['time'], results['accessibility'], color=colors['accessibility'], linewidth=2.5)
    ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accessibility', fontsize=11, fontweight='bold')
    ax3.set_title('C. Level 3: Chromatin\nAccessibility', fontsize=13, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 60)

    # Panel E: Transcription rate
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(results['time'], results['transcription_rate'], color=colors['transcription'], linewidth=2.5)
    ax4.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Rate (mRNA/min)', fontsize=11, fontweight='bold')
    ax4.set_title('D. Level 4: Transcription\nRate', fontsize=13, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 60)

    # Panel F: Splicing rates
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.plot(results['time'], results['splicing_A'], color=colors['splicing_A'], linewidth=2.5, label='Isoform A')
    ax5.plot(results['time'], results['splicing_B'], color=colors['splicing_B'], linewidth=2.5, label='Isoform B')
    ax5.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Splicing Rate (min⁻¹)', fontsize=11, fontweight='bold')
    ax5.set_title('E. Level 5: Splicing\nRates', fontsize=13, fontweight='bold', pad=10)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 60)

    # Panel G: Replication velocity
    ax6 = fig.add_subplot(gs[2, 3])
    ax6.plot(results['time'], results['replication_v'], color=colors['replication'], linewidth=2.5)
    ax6.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Velocity (bp/s)', fontsize=11, fontweight='bold')
    ax6.set_title('F. Replication Fork\nVelocity', fontsize=13, fontweight='bold', pad=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 60)

    # Panel H: mRNA accumulation
    ax7 = fig.add_subplot(gs[3, :2])
    ax7.plot(results['time'], results['mRNA'], color=colors['transcription'], linewidth=2.5)
    ax7.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('mRNA Count', fontsize=11, fontweight='bold')
    ax7.set_title('G. Total mRNA Accumulation', fontsize=13, fontweight='bold', pad=10)
    ax7.grid(True, alpha=0.3)

    # Panel I: Isoform dynamics
    ax8 = fig.add_subplot(gs[3, 2:])
    ax8.plot(results['time'], results['isoform_A'], color=colors['splicing_A'], linewidth=2.5, label='Isoform A')
    ax8.plot(results['time'], results['isoform_B'], color=colors['splicing_B'], linewidth=2.5, label='Isoform B')
    ax8.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Isoform Count', fontsize=11, fontweight='bold')
    ax8.set_title('H. Isoform Dynamics', fontsize=13, fontweight='bold', pad=10)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Panel J: Cross-correlation matrix
    ax9 = fig.add_subplot(gs[4, :2])

    variables = np.column_stack([
        results['Mg'],
        results['lambda_D'],
        results['accessibility'],
        results['transcription_rate'],
        results['splicing_A'],
        results['replication_v']
    ])

    corr_matrix = np.corrcoef(variables.T)
    labels_corr = ['[Mg²⁺]', 'λ_D', 'Access.', 'Transcr.', 'Splicing', 'Replic.']

    im = ax9.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax9.set_xticks(range(len(labels_corr)))
    ax9.set_yticks(range(len(labels_corr)))
    ax9.set_xticklabels(labels_corr, rotation=45, ha='right')
    ax9.set_yticklabels(labels_corr)
    ax9.set_title('I. Cross-Level Correlation Matrix', fontsize=13, fontweight='bold', pad=10)

    for i in range(len(labels_corr)):
        for j in range(len(labels_corr)):
            text = ax9.text(j, i, f'{corr_matrix[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=10)

    cbar = plt.colorbar(im, ax=ax9)
    cbar.set_label('Correlation', fontsize=10)

    # Panel K: Summary statistics
    ax10 = fig.add_subplot(gs[4, 2:])
    ax10.axis('off')

    # Calculate statistics
    oscillation_stats = {
        'Mg': (np.max(results['Mg']) - np.min(results['Mg'])) / (2*np.mean(results['Mg'])) * 100,
        'lambda_D': (np.max(results['lambda_D']) - np.min(results['lambda_D'])) / (2*np.mean(results['lambda_D'])) * 100,
        'accessibility': (np.max(results['accessibility']) - np.min(results['accessibility'])) / (2*np.mean(results['accessibility'])) * 100,
        'transcription': (np.max(results['transcription_rate']) - np.min(results['transcription_rate'])) / (2*np.mean(results['transcription_rate'])) * 100,
        'splicing_A': (np.max(results['splicing_A']) - np.min(results['splicing_A'])) / (2*np.mean(results['splicing_A'])) * 100,
    }

    stats_text = f"""
    UNIFIED FRAMEWORK STATISTICS

    Oscillation Amplitudes:
    • [Mg²⁺]: {oscillation_stats['Mg']:.1f}%
    • λ_D: {oscillation_stats['lambda_D']:.1f}%
    • Accessibility: {oscillation_stats['accessibility']:.1f}%
    • Transcription: {oscillation_stats['transcription']:.1f}%
    • Splicing A: {oscillation_stats['splicing_A']:.1f}%

    Key Correlations:
    • [Mg²⁺] vs λ_D: r = {corr_matrix[0,1]:.3f}
    • λ_D vs Access.: r = {corr_matrix[1,2]:.3f}
    • Access. vs Transcr.: r = {corr_matrix[2,3]:.3f}
    • [Mg²⁺] vs Splicing: r = {corr_matrix[0,4]:.3f}

    Mean Values:
    • [Mg²⁺]: {np.mean(results['Mg']):.3f} mM
    • λ_D: {np.mean(results['lambda_D']):.3f} nm
    • Accessibility: {np.mean(results['accessibility']):.3f}
    • Transcription: {np.mean(results['transcription_rate']):.2f} mRNA/min
    • Replication: {np.mean(results['replication_v']):.1f} bp/s

    Dominant Frequencies:
    • ATP synthesis: ~5 s
    • Na⁺/K⁺-ATPase: ~0.5 s
    • Glycolysis: ~60 s

    Integration:
    All genomic processes coupled through
    charge-dependent electrostatic interactions
    """

    ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.4))

    plt.suptitle('Unified Charge-Oscillatory Genomics Framework: Multi-Scale Integration',
                fontsize=18, fontweight='bold', y=0.998)

    plt.savefig('figure6_unified_framework.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure6_unified_framework.pdf', bbox_inches='tight')
    print("Figure 6 saved: figure6_unified_framework.png/pdf")
    plt.show()

    # Print summary
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"\nOscillation amplitudes:")
    for key, val in oscillation_stats.items():
        print(f"  {key}: {val:.1f}%")

    print(f"\nKey predictions:")
    print(f"  • [Mg²⁺] oscillations drive {oscillation_stats['transcription']:.0f}% transcription modulation")
    print(f"  • Chromatin accessibility varies by {oscillation_stats['accessibility']:.0f}%")
    print(f"  • Splicing rates oscillate by {oscillation_stats['splicing_A']:.0f}%")
    print(f"  • All processes correlated through charge screening (λ_D)")

    print("\n" + "="*80)
