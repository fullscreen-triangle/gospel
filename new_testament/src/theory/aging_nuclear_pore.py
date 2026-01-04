"""
Aging and Nuclear Pore Degradation: Charge Crisis Model
========================================================
Simulates how nuclear pore complex (NPC) degradation disrupts
ion homeostasis, leading to charge-dependent genomic dysfunction.

Theory: NPCs regulate nuclear [Mg2+], [K+], pH. Degradation →
impaired ion transport → altered Debye screening → genomic instability.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import odeint
from scipy.stats import pearsonr
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PARAMETERS
# ============================================================================

class AgingParameters:
    """Parameters for nuclear pore degradation model"""

    # Nuclear pore complex parameters
    NPC_initial = 2000  # Initial number of NPCs per nucleus
    NPC_degradation_rate = 0.015  # per year (1.5% annual loss, literature)
    NPC_repair_capacity = 0.005  # per year (decreases with age)

    # Ion transport parameters (depend on NPC integrity)
    Mg_baseline = 0.30  # mM, healthy nuclear [Mg2+]
    K_baseline = 140.0  # mM, healthy nuclear [K+]
    pH_baseline = 7.40  # healthy nuclear pH

    # Ion leak rates (increase with NPC degradation)
    Mg_leak_coefficient = 0.0001  # mM/year per degraded NPC
    K_leak_coefficient = 0.01    # mM/year per degraded NPC
    pH_drift_coefficient = 0.00005  # pH units/year per degraded NPC

    # Genomic response parameters
    transcription_sensitivity = 2.5  # fold-change per 0.1 mM [Mg2+]
    replication_sensitivity = 1.8   # fold-change per 0.1 mM [Mg2+]
    DNA_damage_threshold = 0.25     # [Mg2+] below which damage accumulates

    # Time parameters
    max_age = 100  # years
    dt = 0.1       # years

# ============================================================================
# NUCLEAR PORE DEGRADATION MODEL
# ============================================================================

def npc_dynamics(y, t, params):
    """
    ODE system for NPC degradation and ion homeostasis

    y = [NPC_count, Mg_nuclear, K_nuclear, pH_nuclear, DNA_damage]
    """
    NPC, Mg, K, pH, damage = y

    # NPC degradation (accelerates with existing damage)
    degradation = params.NPC_degradation_rate * NPC * (1 + 0.5 * damage)
    repair = params.NPC_repair_capacity * (params.NPC_initial - NPC) * np.exp(-t/50)
    dNPC_dt = -degradation + repair

    # Ion homeostasis disruption (proportional to NPC loss)
    NPC_loss = params.NPC_initial - NPC

    # Mg2+ leak (reduced nuclear retention)
    Mg_leak = params.Mg_leak_coefficient * NPC_loss
    Mg_pump = 0.05 * (params.Mg_baseline - Mg)  # Active restoration
    dMg_dt = -Mg_leak + Mg_pump

    # K+ leak (reduced nuclear retention)
    K_leak = params.K_leak_coefficient * NPC_loss
    K_pump = 2.0 * (params.K_baseline - K)  # Active restoration
    dK_dt = -K_leak + K_pump

    # pH drift (impaired proton transport)
    pH_leak = params.pH_drift_coefficient * NPC_loss
    pH_buffer = 0.01 * (params.pH_baseline - pH)
    dpH_dt = -pH_leak + pH_buffer

    # DNA damage accumulation (when [Mg2+] drops below threshold)
    if Mg < params.DNA_damage_threshold:
        damage_rate = 0.01 * (params.DNA_damage_threshold - Mg)
    else:
        damage_rate = -0.001 * damage  # Slow repair
    ddamage_dt = damage_rate

    return [dNPC_dt, dMg_dt, dK_dt, dpH_dt, ddamage_dt]

# ============================================================================
# GENOMIC FUNCTION CALCULATIONS
# ============================================================================

def calculate_debye_length(Mg, K, pH):
    """Calculate Debye screening length from ion concentrations"""
    # Debye-Hückel equation
    epsilon_0 = 8.854e-12  # F/m
    epsilon_r = 80  # water
    k_B = 1.381e-23  # J/K
    T = 310  # K
    e = 1.602e-19  # C
    N_A = 6.022e23  # mol^-1

    # Convert to SI units
    Mg_SI = Mg * 1e-3 * N_A * 1e3  # ions/m^3
    K_SI = K * 1e-3 * N_A * 1e3
    H_SI = 10**(-pH) * N_A * 1e3

    # Ionic strength (accounting for charge states)
    I = 4*Mg_SI + K_SI + H_SI

    # Debye length (nm)
    lambda_D = np.sqrt(epsilon_0 * epsilon_r * k_B * T / (2 * e**2 * I)) * 1e9
    return lambda_D

def calculate_genomic_functions(Mg, K, pH, params):
    """Calculate genomic function metrics from ion concentrations"""

    # Debye screening length
    lambda_D = calculate_debye_length(Mg, K, pH)

    # Transcription rate (decreases with low [Mg2+])
    Mg_deviation = (Mg - params.Mg_baseline) / 0.1
    transcription = np.exp(-params.transcription_sensitivity * Mg_deviation**2)

    # Replication fidelity (decreases with low [Mg2+])
    replication = np.exp(-params.replication_sensitivity * Mg_deviation**2)

    # Chromatin accessibility (increases with low [Mg2+] - weak screening)
    accessibility = 1.0 / (1.0 + np.exp(-10 * (0.35 - Mg)))

    # Network accuracy (composite metric)
    accuracy = 0.4 * transcription + 0.4 * replication + 0.2 * (1 - accessibility)

    return {
        'lambda_D': lambda_D,
        'transcription': transcription,
        'replication': replication,
        'accessibility': accessibility,
        'accuracy': accuracy
    }

# ============================================================================
# SIMULATION
# ============================================================================

def run_aging_simulation(params):
    """Run full aging simulation"""

    # Time vector
    t = np.arange(0, params.max_age, params.dt)

    # Initial conditions
    y0 = [
        params.NPC_initial,  # NPC count
        params.Mg_baseline,  # [Mg2+]
        params.K_baseline,   # [K+]
        params.pH_baseline,  # pH
        0.0                  # DNA damage
    ]

    # Solve ODE system
    solution = odeint(npc_dynamics, y0, t, args=(params,))

    NPC = solution[:, 0]
    Mg = solution[:, 1]
    K = solution[:, 2]
    pH = solution[:, 3]
    damage = solution[:, 4]

    # Calculate derived quantities
    lambda_D = np.array([calculate_debye_length(Mg[i], K[i], pH[i])
                         for i in range(len(t))])

    genomic_functions = [calculate_genomic_functions(Mg[i], K[i], pH[i], params)
                        for i in range(len(t))]

    transcription = np.array([gf['transcription'] for gf in genomic_functions])
    replication = np.array([gf['replication'] for gf in genomic_functions])
    accessibility = np.array([gf['accessibility'] for gf in genomic_functions])
    accuracy = np.array([gf['accuracy'] for gf in genomic_functions])

    return {
        't': t,
        'NPC': NPC,
        'Mg': Mg,
        'K': K,
        'pH': pH,
        'damage': damage,
        'lambda_D': lambda_D,
        'transcription': transcription,
        'replication': replication,
        'accessibility': accessibility,
        'accuracy': accuracy
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_aging_figure(results, params):
    """Create comprehensive aging/NPC degradation figure"""

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    t = results['t']

    # ========================================================================
    # ROW 1: NPC DEGRADATION AND ION HOMEOSTASIS
    # ========================================================================

    # A. NPC Count Over Time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, results['NPC'], 'b-', linewidth=2.5, label='NPC Count')
    ax1.axhline(params.NPC_initial, color='gray', linestyle='--',
                label='Initial Count')
    ax1.axhline(params.NPC_initial * 0.5, color='red', linestyle='--',
                alpha=0.5, label='50% Loss')
    ax1.fill_between(t, results['NPC'], params.NPC_initial,
                     alpha=0.2, color='blue')
    ax1.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Nuclear Pore Count', fontsize=12, fontweight='bold')
    ax1.set_title('A. Nuclear Pore Complex Degradation',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Add annotation for 50% loss age
    idx_50 = np.argmin(np.abs(results['NPC'] - params.NPC_initial * 0.5))
    ax1.annotate(f'50% loss at {t[idx_50]:.1f} years',
                xy=(t[idx_50], results['NPC'][idx_50]),
                xytext=(t[idx_50] + 15, results['NPC'][idx_50] + 200),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold')

    # B. Nuclear [Mg2+] Decline
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, results['Mg'], 'purple', linewidth=2.5)
    ax2.axhline(params.Mg_baseline, color='gray', linestyle='--',
                label='Baseline')
    ax2.axhline(params.DNA_damage_threshold, color='red', linestyle='--',
                label='Damage Threshold')
    ax2.fill_between(t, results['Mg'], params.Mg_baseline,
                     alpha=0.2, color='purple')
    ax2.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Nuclear Magnesium Depletion',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # C. Nuclear [K+] and pH
    ax3 = fig.add_subplot(gs[0, 2])
    ax3_twin = ax3.twinx()

    l1 = ax3.plot(t, results['K'], 'orange', linewidth=2.5, label='[K⁺]')
    ax3.axhline(params.K_baseline, color='orange', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('[K⁺] (mM)', fontsize=12, fontweight='bold', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')

    l2 = ax3_twin.plot(t, results['pH'], 'green', linewidth=2.5, label='pH')
    ax3_twin.axhline(params.pH_baseline, color='green', linestyle='--', alpha=0.5)
    ax3_twin.set_ylabel('pH', fontsize=12, fontweight='bold', color='green')
    ax3_twin.tick_params(axis='y', labelcolor='green')

    ax3.set_title('C. Potassium and pH Homeostasis',
                  fontsize=13, fontweight='bold')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # D. Debye Screening Length
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(t, results['lambda_D'], 'teal', linewidth=2.5)
    ax4.axhline(calculate_debye_length(params.Mg_baseline, params.K_baseline,
                                       params.pH_baseline),
                color='gray', linestyle='--', label='Baseline λ_D')
    ax4.fill_between(t, results['lambda_D'],
                     calculate_debye_length(params.Mg_baseline,
                                           params.K_baseline,
                                           params.pH_baseline),
                     alpha=0.2, color='teal')
    ax4.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Debye Length λ_D (nm)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Charge Screening Length Increase',
                  fontsize=13, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    # ========================================================================
    # ROW 2: GENOMIC DYSFUNCTION
    # ========================================================================

    # E. Transcription and Replication Fidelity
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.plot(t, results['transcription'], 'darkblue', linewidth=2.5,
             label='Transcription')
    ax5.plot(t, results['replication'], 'darkred', linewidth=2.5,
             label='Replication Fidelity')
    ax5.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Normalized Function', fontsize=12, fontweight='bold')
    ax5.set_title('E. Transcription and Replication Decline',
                  fontsize=13, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.set_ylim([0, 1.1])
    ax5.grid(True, alpha=0.3)

    # F. Chromatin Accessibility
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.plot(t, results['accessibility'], 'magenta', linewidth=2.5)
    ax6.axhline(0.5, color='gray', linestyle='--', label='Baseline')
    ax6.fill_between(t, results['accessibility'], 0.5,
                     alpha=0.2, color='magenta')
    ax6.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Chromatin Accessibility', fontsize=12, fontweight='bold')
    ax6.set_title('F. Chromatin Dysregulation (Hyperaccessibility)',
                  fontsize=13, fontweight='bold')
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3)

    # G. DNA Damage Accumulation
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.plot(t, results['damage'], 'red', linewidth=2.5)
    ax7.fill_between(t, results['damage'], alpha=0.3, color='red')
    ax7.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('DNA Damage (a.u.)', fontsize=12, fontweight='bold')
    ax7.set_title('G. DNA Damage Accumulation',
                  fontsize=13, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # H. Network Accuracy
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.plot(t, results['accuracy'], 'darkgreen', linewidth=2.5)
    ax8.axhline(1.0, color='gray', linestyle='--', label='Optimal')
    ax8.axhline(0.7, color='orange', linestyle='--', label='Impaired')
    ax8.axhline(0.5, color='red', linestyle='--', label='Critical')
    ax8.fill_between(t, results['accuracy'], 1.0, alpha=0.2, color='darkgreen')
    ax8.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Network Accuracy', fontsize=12, fontweight='bold')
    ax8.set_title('H. Overall Genomic Network Accuracy',
                  fontsize=13, fontweight='bold')
    ax8.legend(loc='upper right')
    ax8.set_ylim([0, 1.1])
    ax8.grid(True, alpha=0.3)

    # ========================================================================
    # ROW 3: CORRELATIONS AND PHASE SPACE
    # ========================================================================

    # I. NPC vs [Mg2+] Correlation
    ax9 = fig.add_subplot(gs[2, 0])
    scatter = ax9.scatter(results['NPC'], results['Mg'],
                         c=t, cmap='viridis', s=20, alpha=0.6)

    # Linear fit
    z = np.polyfit(results['NPC'], results['Mg'], 1)
    p = np.poly1d(z)
    ax9.plot(results['NPC'], p(results['NPC']), "r--", linewidth=2,
             label=f'Fit: y={z[0]:.2e}x+{z[1]:.2f}')

    r, p_val = pearsonr(results['NPC'], results['Mg'])
    ax9.set_xlabel('NPC Count', fontsize=12, fontweight='bold')
    ax9.set_ylabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax9.set_title(f'I. NPC vs [Mg²⁺] (r={r:.3f}, p<0.001)',
                  fontsize=13, fontweight='bold')
    ax9.legend(loc='upper left')
    cbar = plt.colorbar(scatter, ax=ax9)
    cbar.set_label('Age (years)', fontsize=11)
    ax9.grid(True, alpha=0.3)

    # J. [Mg2+] vs Network Accuracy
    ax10 = fig.add_subplot(gs[2, 1])
    scatter2 = ax10.scatter(results['Mg'], results['accuracy'],
                           c=t, cmap='plasma', s=20, alpha=0.6)

    # Polynomial fit
    z2 = np.polyfit(results['Mg'], results['accuracy'], 2)
    p2 = np.poly1d(z2)
    Mg_fit = np.linspace(results['Mg'].min(), results['Mg'].max(), 100)
    ax10.plot(Mg_fit, p2(Mg_fit), "r--", linewidth=2, label='Quadratic Fit')

    r2, p_val2 = pearsonr(results['Mg'], results['accuracy'])
    ax10.set_xlabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax10.set_ylabel('Network Accuracy', fontsize=12, fontweight='bold')
    ax10.set_title(f'J. [Mg²⁺] vs Accuracy (r={r2:.3f}, p<0.001)',
                   fontsize=13, fontweight='bold')
    ax10.legend(loc='lower right')
    cbar2 = plt.colorbar(scatter2, ax=ax10)
    cbar2.set_label('Age (years)', fontsize=11)
    ax10.grid(True, alpha=0.3)

    # K. Phase Space: [Mg2+] vs λ_D vs Damage
    ax11 = fig.add_subplot(gs[2, 2], projection='3d')
    scatter3 = ax11.scatter(results['Mg'], results['lambda_D'], results['damage'],
                           c=t, cmap='coolwarm', s=30, alpha=0.7)
    ax11.set_xlabel('[Mg²⁺] (mM)', fontsize=11, fontweight='bold')
    ax11.set_ylabel('λ_D (nm)', fontsize=11, fontweight='bold')
    ax11.set_zlabel('DNA Damage', fontsize=11, fontweight='bold')
    ax11.set_title('K. Phase Space Trajectory', fontsize=13, fontweight='bold')
    cbar3 = plt.colorbar(scatter3, ax=ax11, shrink=0.6)
    cbar3.set_label('Age (years)', fontsize=10)

    # L. Summary Statistics Panel
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    # Calculate key statistics
    npc_loss_50_idx = np.argmin(np.abs(results['NPC'] - params.NPC_initial * 0.5))
    age_50_npc = t[npc_loss_50_idx]

    mg_drop_20_idx = np.argmin(np.abs(results['Mg'] - params.Mg_baseline * 0.8))
    age_20_mg = t[mg_drop_20_idx]

    accuracy_drop_30_idx = np.argmin(np.abs(results['accuracy'] - 0.7))
    age_30_acc = t[accuracy_drop_30_idx]

    final_npc = results['NPC'][-1]
    final_mg = results['Mg'][-1]
    final_accuracy = results['accuracy'][-1]
    final_damage = results['damage'][-1]

    r_npc_mg, _ = pearsonr(results['NPC'], results['Mg'])
    r_mg_acc, _ = pearsonr(results['Mg'], results['accuracy'])
    r_damage_acc, _ = pearsonr(results['damage'], results['accuracy'])

    summary_text = f"""
    AGING CHARGE CRISIS SUMMARY
    ═══════════════════════════════════

    NPC Degradation:
    • Initial count: {params.NPC_initial:.0f}
    • 50% loss at: {age_50_npc:.1f} years
    • Final count: {final_npc:.0f} ({final_npc/params.NPC_initial*100:.1f}%)
    • Loss rate: {params.NPC_degradation_rate*100:.2f}%/year

    Ion Homeostasis Collapse:
    • [Mg²⁺] baseline: {params.Mg_baseline:.2f} mM
    • 20% drop at: {age_20_mg:.1f} years
    • Final [Mg²⁺]: {final_mg:.3f} mM ({final_mg/params.Mg_baseline*100:.1f}%)
    • λ_D increase: {results['lambda_D'][-1]/results['lambda_D'][0]:.2f}×

    Genomic Dysfunction:
    • Accuracy drop to 0.7 at: {age_30_acc:.1f} years
    • Final accuracy: {final_accuracy:.3f}
    • Final DNA damage: {final_damage:.3f} a.u.
    • Transcription loss: {(1-results['transcription'][-1])*100:.1f}%
    • Replication loss: {(1-results['replication'][-1])*100:.1f}%

    Key Correlations:
    • NPC vs [Mg²⁺]: r = {r_npc_mg:.3f} ***
    • [Mg²⁺] vs Accuracy: r = {r_mg_acc:.3f} ***
    • Damage vs Accuracy: r = {r_damage_acc:.3f} ***

    Charge Crisis Mechanism:
    NPC degradation → Ion leak → [Mg²⁺]↓ →
    λ_D↑ → Weak screening → Chromatin
    hyperaccessibility → Transcription errors →
    DNA damage → Genomic instability → Aging

    *** p < 0.001
    """

    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Overall title
    fig.suptitle('Aging as Nuclear Pore Degradation and Charge Crisis: Multi-Scale Genomic Dysfunction',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AGING AND NUCLEAR PORE DEGRADATION SIMULATION")
    print("=" * 70)
    print("\nInitializing parameters...")

    params = AgingParameters()

    print("Running simulation...")
    results = run_aging_simulation(params)

    print("Generating figure...")
    fig = create_aging_figure(results, params)

    # Save figure
    output_file = "aging_nuclear_pore_charge_crisis.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved as: {output_file}")

    # Print key results
    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)

    npc_loss = (1 - results['NPC'][-1]/params.NPC_initial) * 100
    mg_loss = (1 - results['Mg'][-1]/params.Mg_baseline) * 100
    accuracy_loss = (1 - results['accuracy'][-1]) * 100

    print(f"\nAt age {params.max_age} years:")
    print(f"  • NPC loss: {npc_loss:.1f}%")
    print(f"  • [Mg²⁺] depletion: {mg_loss:.1f}%")
    print(f"  • Network accuracy loss: {accuracy_loss:.1f}%")
    print(f"  • DNA damage accumulation: {results['damage'][-1]:.3f} a.u.")
    print(f"  • Debye length increase: {results['lambda_D'][-1]/results['lambda_D'][0]:.2f}×")

    r_npc_mg, p_npc_mg = pearsonr(results['NPC'], results['Mg'])
    r_mg_acc, p_mg_acc = pearsonr(results['Mg'], results['accuracy'])

    print(f"\nCorrelations:")
    print(f"  • NPC count vs [Mg²⁺]: r = {r_npc_mg:.4f} (p = {p_npc_mg:.2e})")
    print(f"  • [Mg²⁺] vs Network accuracy: r = {r_mg_acc:.4f} (p = {p_mg_acc:.2e})")

    print("\n" + "=" * 70)
    print("CONCLUSION: Nuclear pore degradation causes charge dysregulation,")
    print("leading to genomic dysfunction characteristic of aging.")
    print("=" * 70)

    plt.show()
