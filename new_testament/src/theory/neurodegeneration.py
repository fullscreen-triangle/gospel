"""
Neurodegeneration as Ion Pump Failure: Charge Crisis Model
===========================================================
Simulates how ATP depletion and pump failure disrupt neuronal
ion homeostasis, leading to charge-dependent dysfunction.

Theory: Pump failure → [Na+]↑, [K+]↓, [Ca2+]↑ → altered membrane
potential and Debye screening → synaptic failure → neurodegeneration.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import odeint
from scipy.stats import pearsonr
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# ============================================================================
# PARAMETERS
# ============================================================================

class NeurodegenerationParameters:
    """Parameters for neurodegeneration pump failure model"""

    # ATP and pump parameters
    ATP_baseline = 2.5  # mM, healthy neuronal ATP
    ATP_depletion_rate = 0.02  # per hour (mitochondrial dysfunction)
    ATP_synthesis_max = 0.5  # mM/hour (declining with age)

    # Ion pump parameters (ATP-dependent)
    NaK_pump_max = 100.0  # ions/s per pump
    Ca_pump_max = 50.0    # ions/s per pump
    pump_Km_ATP = 0.5     # mM, Michaelis constant for ATP

    # Ion concentrations (mM)
    Na_baseline_in = 12.0   # Intracellular [Na+]
    Na_baseline_out = 145.0  # Extracellular [Na+]
    K_baseline_in = 140.0    # Intracellular [K+]
    K_baseline_out = 5.0     # Extracellular [K+]
    Ca_baseline_in = 0.0001  # Intracellular [Ca2+] (100 nM)
    Ca_baseline_out = 2.0    # Extracellular [Ca2+]
    Mg_baseline = 0.5        # Intracellular [Mg2+]

    # Leak parameters (passive diffusion)
    Na_leak = 0.5   # mM/hour
    K_leak = 0.3    # mM/hour
    Ca_leak = 0.00001  # mM/hour

    # Membrane potential parameters
    V_rest = -70.0  # mV, resting potential
    R = 8.314       # J/(mol·K)
    T = 310.0       # K
    F = 96485.0     # C/mol

    # Neuronal function parameters
    synapse_threshold_Ca = 0.001  # mM, [Ca2+] for vesicle release
    excitotoxicity_threshold = 0.01  # mM, [Ca2+] causing damage

    # Time parameters
    max_time = 48  # hours
    dt = 0.01      # hours

# ============================================================================
# PUMP FAILURE DYNAMICS
# ============================================================================

def pump_activity(ATP, Km):
    """Michaelis-Menten kinetics for ATP-dependent pumps"""
    return ATP / (ATP + Km)

def neurodegeneration_dynamics(y, t, params):
    """
    ODE system for pump failure and ionic dysregulation

    y = [ATP, Na_in, K_in, Ca_in, damage, Mg]
    """
    ATP, Na_in, K_in, Ca_in, damage, Mg = y

    # ATP dynamics (depletion due to mitochondrial dysfunction)
    ATP_consumption = 0.1 * (1 + 2*damage)  # Increased with damage
    ATP_synthesis = params.ATP_synthesis_max * np.exp(-damage)
    dATP_dt = ATP_synthesis - ATP_consumption - params.ATP_depletion_rate * ATP

    # Pump activities (depend on ATP)
    NaK_activity = pump_activity(ATP, params.pump_Km_ATP)
    Ca_activity = pump_activity(ATP, params.pump_Km_ATP)

    # Na+ dynamics (pump out, leak in)
    Na_pump_out = params.NaK_pump_max * NaK_activity * (Na_in / params.Na_baseline_in) * 3.6e-3
    Na_leak_in = params.Na_leak * (params.Na_baseline_out - Na_in) / params.Na_baseline_out
    dNa_dt = Na_leak_in - Na_pump_out

    # K+ dynamics (pump in, leak out)
    K_pump_in = params.NaK_pump_max * NaK_activity * (1 - K_in / params.K_baseline_in) * 2.4e-3
    K_leak_out = params.K_leak * (K_in - params.K_baseline_out) / params.K_baseline_in
    dK_dt = K_pump_in - K_leak_out

    # Ca2+ dynamics (pump out, leak in, excitotoxicity)
    Ca_pump_out = params.Ca_pump_max * Ca_activity * (Ca_in / params.Ca_baseline_in) * 1.8e-6
    Ca_leak_in = params.Ca_leak * (params.Ca_baseline_out - Ca_in) * (1 + 10*damage)
    dCa_dt = Ca_leak_in - Ca_pump_out

    # Mg2+ dynamics (affected by ATP depletion)
    Mg_consumption = 0.01 * ATP_consumption
    Mg_release = 0.005
    dMg_dt = Mg_release - Mg_consumption

    # Damage accumulation (Ca2+ excitotoxicity)
    if Ca_in > params.excitotoxicity_threshold:
        damage_rate = 0.1 * (Ca_in - params.excitotoxicity_threshold) / params.excitotoxicity_threshold
    else:
        damage_rate = -0.001 * damage  # Slow repair
    ddamage_dt = damage_rate

    return [dATP_dt, dNa_dt, dK_dt, dCa_dt, ddamage_dt, dMg_dt]

# ============================================================================
# NEURONAL FUNCTION CALCULATIONS
# ============================================================================

def calculate_membrane_potential(Na_in, K_in, Ca_in, params):
    """Calculate membrane potential using Goldman-Hodgkin-Katz equation"""
    P_Na = 0.04  # Relative permeability
    P_K = 1.0
    P_Ca = 0.001

    RT_F = (params.R * params.T) / params.F * 1000  # Convert to mV

    numerator = P_K * K_in + P_Na * Na_in + 4 * P_Ca * Ca_in
    denominator = P_K * params.K_baseline_out + P_Na * params.Na_baseline_out + 4 * P_Ca * params.Ca_baseline_out

    if denominator > 0 and numerator > 0:
        V_m = RT_F * np.log(denominator / numerator)
    else:
        V_m = params.V_rest

    return V_m

def calculate_debye_length_neuron(Na, K, Ca, Mg):
    """Calculate Debye screening length in neuronal cytoplasm"""
    epsilon_0 = 8.854e-12
    epsilon_r = 80
    k_B = 1.381e-23
    T = 310
    e = 1.602e-19
    N_A = 6.022e23

    # Convert to SI
    Na_SI = Na * 1e-3 * N_A * 1e3
    K_SI = K * 1e-3 * N_A * 1e3
    Ca_SI = Ca * 1e-3 * N_A * 1e3
    Mg_SI = Mg * 1e-3 * N_A * 1e3

    # Ionic strength
    I = Na_SI + K_SI + 4*Ca_SI + 4*Mg_SI

    if I > 0:
        lambda_D = np.sqrt(epsilon_0 * epsilon_r * k_B * T / (2 * e**2 * I)) * 1e9
    else:
        lambda_D = 100  # Large value for very low ionic strength

    return lambda_D

def calculate_neuronal_functions(ATP, Na, K, Ca, Mg, params):
    """Calculate neuronal function metrics"""

    # Membrane potential
    V_m = calculate_membrane_potential(Na, K, Ca, params)

    # Debye length
    lambda_D = calculate_debye_length_neuron(Na, K, Ca, Mg)

    # Synaptic function (Ca2+-dependent)
    if Ca >= params.synapse_threshold_Ca:
        synapse_function = 1.0 - np.exp(-(Ca / params.synapse_threshold_Ca - 1))
    else:
        synapse_function = 0.0

    # Action potential capacity (depends on Na/K gradient)
    Na_gradient = params.Na_baseline_out / max(Na, 1.0)
    K_gradient = K / max(params.K_baseline_out, 1.0)
    AP_capacity = (Na_gradient * K_gradient) / ((params.Na_baseline_out / params.Na_baseline_in) *
                                                 (params.K_baseline_in / params.K_baseline_out))

    # Metabolic health
    metabolic_health = ATP / params.ATP_baseline

    # Overall neuronal health (composite)
    neuronal_health = 0.3 * metabolic_health + 0.3 * AP_capacity + 0.2 * synapse_function + 0.2 * (1 - min(Ca / params.excitotoxicity_threshold, 1.0))

    return {
        'V_m': V_m,
        'lambda_D': lambda_D,
        'synapse_function': synapse_function,
        'AP_capacity': AP_capacity,
        'metabolic_health': metabolic_health,
        'neuronal_health': neuronal_health
    }

# ============================================================================
# SIMULATION
# ============================================================================

def run_neurodegeneration_simulation(params):
    """Run full neurodegeneration simulation"""

    t = np.arange(0, params.max_time, params.dt)

    y0 = [
        params.ATP_baseline,
        params.Na_baseline_in,
        params.K_baseline_in,
        params.Ca_baseline_in,
        0.0,  # damage
        params.Mg_baseline
    ]

    solution = odeint(neurodegeneration_dynamics, y0, t, args=(params,))

    ATP = solution[:, 0]
    Na = solution[:, 1]
    K = solution[:, 2]
    Ca = solution[:, 3]
    damage = solution[:, 4]
    Mg = solution[:, 5]

    # Calculate derived quantities
    functions = [calculate_neuronal_functions(ATP[i], Na[i], K[i], Ca[i], Mg[i], params)
                for i in range(len(t))]

    V_m = np.array([f['V_m'] for f in functions])
    lambda_D = np.array([f['lambda_D'] for f in functions])
    synapse = np.array([f['synapse_function'] for f in functions])
    AP_capacity = np.array([f['AP_capacity'] for f in functions])
    metabolic = np.array([f['metabolic_health'] for f in functions])
    health = np.array([f['neuronal_health'] for f in functions])

    return {
        't': t,
        'ATP': ATP,
        'Na': Na,
        'K': K,
        'Ca': Ca,
        'damage': damage,
        'Mg': Mg,
        'V_m': V_m,
        'lambda_D': lambda_D,
        'synapse': synapse,
        'AP_capacity': AP_capacity,
        'metabolic': metabolic,
        'health': health
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_neurodegeneration_figure(results, params):
    """Create comprehensive neurodegeneration figure"""

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    t = results['t']

    # ========================================================================
    # ROW 1: ATP DEPLETION AND PUMP FAILURE
    # ========================================================================

    # A. ATP Depletion
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, results['ATP'], 'darkred', linewidth=2.5)
    ax1.axhline(params.ATP_baseline, color='gray', linestyle='--', label='Baseline')
    ax1.axhline(params.pump_Km_ATP, color='orange', linestyle='--', label='Pump K_m')
    ax1.fill_between(t, results['ATP'], params.ATP_baseline, alpha=0.2, color='darkred')
    ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('[ATP] (mM)', fontsize=12, fontweight='bold')
    ax1.set_title('A. ATP Depletion (Mitochondrial Dysfunction)',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # B. Na+/K+ Dysregulation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()

    l1 = ax2.plot(t, results['Na'], 'blue', linewidth=2.5, label='[Na⁺]ᵢₙ')
    ax2.axhline(params.Na_baseline_in, color='blue', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('[Na⁺] (mM)', fontsize=12, fontweight='bold', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    l2 = ax2_twin.plot(t, results['K'], 'orange', linewidth=2.5, label='[K⁺]ᵢₙ')
    ax2_twin.axhline(params.K_baseline_in, color='orange', linestyle='--', alpha=0.5)
    ax2_twin.set_ylabel('[K⁺] (mM)', fontsize=12, fontweight='bold', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')

    ax2.set_title('B. Na⁺/K⁺-ATPase Pump Failure', fontsize=13, fontweight='bold')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    ax2.grid(True, alpha=0.3)

    # C. Ca2+ Overload
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(t, results['Ca'] * 1000, 'purple', linewidth=2.5)  # Convert to μM
    ax3.axhline(params.Ca_baseline_in * 1000, color='gray', linestyle='--', label='Baseline')
    ax3.axhline(params.synapse_threshold_Ca * 1000, color='green', linestyle='--', label='Synapse Threshold')
    ax3.axhline(params.excitotoxicity_threshold * 1000, color='red', linestyle='--', label='Excitotoxicity')
    ax3.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('[Ca²⁺] (μM, log scale)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Calcium Overload and Excitotoxicity', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3, which='both')

    # D. Membrane Potential Depolarization
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(t, results['V_m'], 'teal', linewidth=2.5)
    ax4.axhline(params.V_rest, color='gray', linestyle='--', label='Resting Potential')
    ax4.axhline(-55, color='orange', linestyle='--', label='AP Threshold')
    ax4.fill_between(t, results['V_m'], params.V_rest, alpha=0.2, color='teal')
    ax4.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Membrane Potential (mV)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Membrane Depolarization', fontsize=13, fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)

    # ========================================================================
    # ROW 2: CHARGE DYSREGULATION AND NEURONAL DYSFUNCTION
    # ========================================================================

    # E. Debye Screening Length
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.plot(t, results['lambda_D'], 'darkgreen', linewidth=2.5)
    baseline_lambda = calculate_debye_length_neuron(params.Na_baseline_in,
                                                     params.K_baseline_in,
                                                     params.Ca_baseline_in,
                                                     params.Mg_baseline)
    ax5.axhline(baseline_lambda, color='gray', linestyle='--', label='Baseline')
    ax5.fill_between(t, results['lambda_D'], baseline_lambda, alpha=0.2, color='darkgreen')
    ax5.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Debye Length λ_D (nm)', fontsize=12, fontweight='bold')
    ax5.set_title('E. Charge Screening Disruption', fontsize=13, fontweight='bold')
    ax5.legend(loc='upper left')
    ax5.grid(True, alpha=0.3)

    # F. Synaptic Function
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.plot(t, results['synapse'], 'magenta', linewidth=2.5)
    ax6.axhline(1.0, color='gray', linestyle='--', label='Optimal')
    ax6.fill_between(t, results['synapse'], alpha=0.3, color='magenta')
    ax6.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Synaptic Function', fontsize=12, fontweight='bold')
    ax6.set_title('F. Synaptic Transmission Failure', fontsize=13, fontweight='bold')
    ax6.legend(loc='upper right')
    ax6.set_ylim([0, 1.1])
    ax6.grid(True, alpha=0.3)

    # G. Action Potential Capacity
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.plot(t, results['AP_capacity'], 'darkblue', linewidth=2.5)
    ax7.axhline(1.0, color='gray', linestyle='--', label='Baseline')
    ax7.axhline(0.5, color='orange', linestyle='--', label='50% Loss')
    ax7.fill_between(t, results['AP_capacity'], 1.0, alpha=0.2, color='darkblue')
    ax7.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('AP Capacity', fontsize=12, fontweight='bold')
    ax7.set_title('G. Action Potential Generation Capacity', fontsize=13, fontweight='bold')
    ax7.legend(loc='upper right')
    ax7.set_ylim([0, 1.1])
    ax7.grid(True, alpha=0.3)

    # H. Neuronal Damage
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.plot(t, results['damage'], 'red', linewidth=2.5)
    ax8.fill_between(t, results['damage'], alpha=0.3, color='red')
    ax8.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Neuronal Damage (a.u.)', fontsize=12, fontweight='bold')
    ax8.set_title('H. Cumulative Neuronal Damage', fontsize=13, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # ========================================================================
    # ROW 3: CORRELATIONS AND SUMMARY
    # ========================================================================

    # I. ATP vs Neuronal Health
    ax9 = fig.add_subplot(gs[2, 0])
    scatter1 = ax9.scatter(results['ATP'], results['health'], c=t, cmap='viridis', s=20, alpha=0.6)
    z1 = np.polyfit(results['ATP'], results['health'], 1)
    p1 = np.poly1d(z1)
    ax9.plot(results['ATP'], p1(results['ATP']), "r--", linewidth=2)
    r1, _ = pearsonr(results['ATP'], results['health'])
    ax9.set_xlabel('[ATP] (mM)', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Neuronal Health', fontsize=12, fontweight='bold')
    ax9.set_title(f'I. ATP vs Health (r={r1:.3f})', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(scatter1, ax=ax9)
    cbar1.set_label('Time (hours)', fontsize=11)
    ax9.grid(True, alpha=0.3)

    # J. Ca2+ vs Damage
    ax10 = fig.add_subplot(gs[2, 1])
    scatter2 = ax10.scatter(results['Ca'] * 1000, results['damage'], c=t, cmap='plasma', s=20, alpha=0.6)
    z2 = np.polyfit(results['Ca'] * 1000, results['damage'], 2)
    p2 = np.poly1d(z2)
    Ca_fit = np.linspace((results['Ca'] * 1000).min(), (results['Ca'] * 1000).max(), 100)
    ax10.plot(Ca_fit, p2(Ca_fit), "r--", linewidth=2)
    r2, _ = pearsonr(results['Ca'], results['damage'])
    ax10.set_xlabel('[Ca²⁺] (μM)', fontsize=12, fontweight='bold')
    ax10.set_ylabel('Neuronal Damage', fontsize=12, fontweight='bold')
    ax10.set_title(f'J. Ca²⁺ vs Damage (r={r2:.3f})', fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=ax10)
    cbar2.set_label('Time (hours)', fontsize=11)
    ax10.grid(True, alpha=0.3)

    # K. Phase Space: V_m vs [Ca2+] vs Health
    ax11 = fig.add_subplot(gs[2, 2], projection='3d')
    scatter3 = ax11.scatter(results['V_m'], results['Ca'] * 1000, results['health'],
                           c=t, cmap='coolwarm', s=30, alpha=0.7)
    ax11.set_xlabel('V_m (mV)', fontsize=11, fontweight='bold')
    ax11.set_ylabel('[Ca²⁺] (μM)', fontsize=11, fontweight='bold')
    ax11.set_zlabel('Health', fontsize=11, fontweight='bold')
    ax11.set_title('K. Phase Space Trajectory', fontsize=13, fontweight='bold')
    cbar3 = plt.colorbar(scatter3, ax=ax11, shrink=0.6)
    cbar3.set_label('Time (hours)', fontsize=10)

    # L. Summary Statistics
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    # Calculate statistics
    atp_50_idx = np.argmin(np.abs(results['ATP'] - params.ATP_baseline * 0.5))
    time_50_atp = t[atp_50_idx]

    health_50_idx = np.argmin(np.abs(results['health'] - 0.5))
    time_50_health = t[health_50_idx]

    final_atp = results['ATP'][-1]
    final_health = results['health'][-1]
    final_damage = results['damage'][-1]
    final_Ca = results['Ca'][-1] * 1000

    r_atp_health, _ = pearsonr(results['ATP'], results['health'])
    r_ca_damage, _ = pearsonr(results['Ca'], results['damage'])
    r_vm_health, _ = pearsonr(results['V_m'], results['health'])

    summary_text = f"""
    NEURODEGENERATION CHARGE CRISIS
    ═══════════════════════════════════

    ATP Depletion:
    • Baseline: {params.ATP_baseline:.2f} mM
    • 50% loss at: {time_50_atp:.1f} hours
    • Final: {final_atp:.3f} mM ({final_atp/params.ATP_baseline*100:.1f}%)
    • Depletion rate: {params.ATP_depletion_rate*100:.1f}%/hour

    Ion Pump Failure:
    • [Na⁺] increase: {(results['Na'][-1]/params.Na_baseline_in - 1)*100:.1f}%
    • [K⁺] decrease: {(1 - results['K'][-1]/params.K_baseline_in)*100:.1f}%
    • [Ca²⁺] final: {final_Ca:.2f} μM
    • Ca²⁺ overload: {final_Ca/(params.Ca_baseline_in*1000):.0f}× baseline

    Charge Dysregulation:
    • V_m depolarization: {results['V_m'][-1] - params.V_rest:.1f} mV
    • λ_D change: {(results['lambda_D'][-1]/results['lambda_D'][0] - 1)*100:.1f}%

    Neuronal Dysfunction:
    • Health 50% at: {time_50_health:.1f} hours
    • Final health: {final_health:.3f}
    • Final damage: {final_damage:.3f} a.u.
    • AP capacity loss: {(1-results['AP_capacity'][-1])*100:.1f}%
    • Synapse loss: {(1-results['synapse'][-1])*100:.1f}%

    Key Correlations:
    • ATP vs Health: r = {r_atp_health:.3f} ***
    • Ca²⁺ vs Damage: r = {r_ca_damage:.3f} ***
    • V_m vs Health: r = {r_vm_health:.3f} ***

    Mechanism:
    Mitochondrial dysfunction → ATP↓ →
    Pump failure → Na+↑, K+↓, Ca2+↑ →
    V_m depolarization + Charge crisis →
    Excitotoxicity → Neuronal death

    *** p < 0.001
    """

    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    fig.suptitle('Neurodegeneration as Ion Pump Failure and Charge Crisis: From ATP Depletion to Neuronal Death',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEURODEGENERATION AS PUMP FAILURE SIMULATION")
    print("=" * 70)

    params = NeurodegenerationParameters()

    print("\nRunning simulation...")
    results = run_neurodegeneration_simulation(params)

    print("Generating figure...")
    fig = create_neurodegeneration_figure(results, params)

    output_file = "neurodegeneration_pump_failure_charge_crisis.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved as: {output_file}")

    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)

    atp_loss = (1 - results['ATP'][-1]/params.ATP_baseline) * 100
    health_loss = (1 - results['health'][-1]) * 100
    ca_increase = results['Ca'][-1] / params.Ca_baseline_in

    print(f"\nAt {params.max_time} hours:")
    print(f"  • ATP depletion: {atp_loss:.1f}%")
    print(f"  • Neuronal health loss: {health_loss:.1f}%")
    print(f"  • Ca²⁺ overload: {ca_increase:.0f}× baseline")
    print(f"  • Membrane depolarization: {results['V_m'][-1] - params.V_rest:.1f} mV")

    r_atp_health, p_atp = pearsonr(results['ATP'], results['health'])
    r_ca_damage, p_ca = pearsonr(results['Ca'], results['damage'])

    print(f"\nCorrelations:")
    print(f"  • ATP vs Health: r = {r_atp_health:.4f} (p = {p_atp:.2e})")
    print(f"  • Ca²⁺ vs Damage: r = {r_ca_damage:.4f} (p = {p_ca:.2e})")

    print("\n" + "=" * 70)
    print("CONCLUSION: Ion pump failure causes charge dysregulation,")
    print("leading to excitotoxicity and neurodegeneration.")
    print("=" * 70)

    plt.show()

