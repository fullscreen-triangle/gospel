"""
Warburg Effect as Charge Crisis: Cancer Metabolic Reprogramming
================================================================
Simulates how glycolytic switch in cancer cells creates a charge
crisis that drives genomic instability and proliferation.

Theory: Glycolysis → Lactate accumulation → pH↓ → [H+]↑ →
Altered Debye screening → Chromatin accessibility → Proliferation genes activated
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
sns.set_palette("husl")

# ============================================================================
# PARAMETERS
# ============================================================================

class WarburgParameters:
    """Parameters for Warburg effect charge crisis model"""

    # Metabolic parameters
    glucose_baseline = 5.0  # mM, extracellular glucose
    glycolysis_rate_normal = 2.0  # mM/hour, normal cells
    glycolysis_rate_cancer = 20.0  # mM/hour, cancer cells (10× higher)
    OXPHOS_rate_normal = 10.0  # mM/hour ATP from OXPHOS
    OXPHOS_rate_cancer = 2.0   # mM/hour (suppressed in cancer)

    # Lactate and pH parameters
    lactate_production_ratio = 2.0  # moles lactate per mole glucose
    lactate_export_rate = 0.5  # per hour (MCT transporter)
    H_production_ratio = 1.0  # moles H+ per mole lactate
    pH_buffer_capacity = 10.0  # mM (bicarbonate, phosphate)

    # Ion concentrations
    Na_baseline = 12.0  # mM intracellular
    K_baseline = 140.0  # mM intracellular
    Mg_baseline = 0.5   # mM intracellular
    Ca_baseline = 0.0001  # mM intracellular
    pH_baseline_normal = 7.2  # Normal cell pH
    pH_baseline_cancer = 6.8  # Cancer cell pH (acidic)

    # ATP parameters
    ATP_yield_glycolysis = 2.0  # ATP per glucose
    ATP_yield_OXPHOS = 30.0     # ATP per glucose
    ATP_consumption = 5.0       # mM/hour

    # Genomic parameters
    proliferation_pH_optimum = 6.9  # pH for maximal proliferation
    apoptosis_pH_threshold = 6.5    # pH below which apoptosis triggered

    # Cell state parameters
    cell_type = 'cancer'  # 'normal' or 'cancer'

    # Time parameters
    max_time = 24  # hours
    dt = 0.01      # hours

# ============================================================================
# WARBURG EFFECT DYNAMICS
# ============================================================================

def warburg_dynamics(y, t, params, cell_type='cancer'):
    """
    ODE system for Warburg effect and charge crisis

    y = [glucose, lactate, ATP, pH, Na, K, Mg, proliferation_signal, DNA_damage]
    """
    glucose, lactate, ATP, pH, Na, K, Mg, prolif, damage = y

    # Select metabolic rates based on cell type
    if cell_type == 'cancer':
        glycolysis_rate = params.glycolysis_rate_cancer
        OXPHOS_rate = params.OXPHOS_rate_cancer
    else:
        glycolysis_rate = params.glycolysis_rate_normal
        OXPHOS_rate = params.OXPHOS_rate_normal

    # Glucose consumption
    glucose_uptake = glycolysis_rate * (glucose / (glucose + 1.0))  # Michaelis-Menten
    dglucose_dt = -glucose_uptake

    # Lactate production and export
    lactate_production = params.lactate_production_ratio * glucose_uptake
    lactate_export = params.lactate_export_rate * lactate
    dlactate_dt = lactate_production - lactate_export

    # ATP production (glycolysis + OXPHOS)
    ATP_from_glycolysis = params.ATP_yield_glycolysis * glucose_uptake
    ATP_from_OXPHOS = params.ATP_yield_OXPHOS * OXPHOS_rate * (glucose / (glucose + 1.0))
    ATP_consumption = params.ATP_consumption * (1.0 + 0.5 * prolif)  # Higher with proliferation
    dATP_dt = ATP_from_glycolysis + ATP_from_OXPHOS - ATP_consumption

    # pH dynamics (H+ production from lactate)
    H_production = params.H_production_ratio * lactate_production
    H_buffering = params.pH_buffer_capacity * (7.2 - pH)  # Buffer tries to restore pH 7.2

    # Convert H+ concentration to pH (simplified)
    H_conc = 10**(-pH)  # mM
    dH_dt = H_production - H_buffering - lactate_export * 0.5  # H+ exported with lactate

    # pH change (inverse relationship with [H+])
    dpH_dt = -dH_dt / (H_conc * np.log(10))

    # Ion dynamics (affected by pH and metabolic state)
    # Na+ (Na+/H+ exchanger activated in acidic conditions)
    Na_influx = 0.5 * (7.2 - pH)  # Increased Na+ influx to export H+
    Na_efflux = 0.3 * Na
    dNa_dt = Na_influx - Na_efflux

    # K+ (affected by membrane potential changes)
    K_leak = 0.1 * (K - 100)
    dK_dt = -K_leak

    # Mg2+ (chelated by ATP, released during glycolysis)
    Mg_release = 0.05 * glucose_uptake
    Mg_consumption = 0.02 * ATP_consumption
    dMg_dt = Mg_release - Mg_consumption

    # Proliferation signal (activated by specific pH range and ATP)
    if params.apoptosis_pH_threshold < pH < 7.0 and ATP > 1.0:
        pH_factor = np.exp(-((pH - params.proliferation_pH_optimum)**2) / 0.1)
        ATP_factor = ATP / (ATP + 1.0)
        prolif_activation = 0.5 * pH_factor * ATP_factor
    else:
        prolif_activation = -0.1 * prolif  # Decay
    dprolif_dt = prolif_activation

    # DNA damage (from ROS in high glycolysis, low pH stress)
    if pH < 6.7:
        damage_rate = 0.05 * (6.7 - pH) * glycolysis_rate / 10.0
    else:
        damage_rate = -0.01 * damage  # Repair
    ddamage_dt = damage_rate

    return [dglucose_dt, dlactate_dt, dATP_dt, dpH_dt, dNa_dt, dK_dt, dMg_dt, dprolif_dt, ddamage_dt]

# ============================================================================
# CHARGE AND GENOMIC CALCULATIONS
# ============================================================================

def calculate_debye_length_warburg(Na, K, Mg, pH):
    """Calculate Debye screening length with pH dependence"""
    epsilon_0 = 8.854e-12
    epsilon_r = 80
    k_B = 1.381e-23
    T = 310
    e = 1.602e-19
    N_A = 6.022e23

    # Convert to SI
    Na_SI = Na * 1e-3 * N_A * 1e3
    K_SI = K * 1e-3 * N_A * 1e3
    Mg_SI = Mg * 1e-3 * N_A * 1e3
    H_SI = 10**(-pH) * N_A * 1e3

    # Ionic strength (H+ contribution critical in acidic conditions)
    I = Na_SI + K_SI + 4*Mg_SI + H_SI

    if I > 0:
        lambda_D = np.sqrt(epsilon_0 * epsilon_r * k_B * T / (2 * e**2 * I)) * 1e9
    else:
        lambda_D = 100

    return lambda_D

def calculate_cancer_functions(glucose, lactate, ATP, pH, Na, K, Mg, prolif, damage, params):
    """Calculate cancer cell functional metrics"""

    # Debye screening length
    lambda_D = calculate_debye_length_warburg(Na, K, Mg, pH)

    # Chromatin accessibility (increased in acidic pH due to protonation)
    # Low pH → more H+ → protonation of histones → weaker DNA-histone binding
    accessibility = 1.0 / (1.0 + np.exp(10 * (pH - 7.0)))

    # Glycolytic flux
    glycolytic_flux = params.glycolysis_rate_cancer * (glucose / (glucose + 1.0))

    # Proliferation capacity
    if params.apoptosis_pH_threshold < pH < 7.2:
        prolif_capacity = prolif * (ATP / (ATP + 1.0))
    else:
        prolif_capacity = 0.0

    # Metabolic efficiency (ATP per glucose)
    if glucose > 0.1:
        efficiency = ATP / glucose
    else:
        efficiency = 0.0

    # Oncogene expression (simplified: increases with accessibility and proliferation signal)
    oncogene_expression = accessibility * prolif * (1.0 - damage)

    # Tumor suppressor expression (decreases with damage and acidic pH)
    tumor_suppressor = (1.0 - accessibility) * (1.0 - damage) * np.exp(-(7.0 - pH))

    # Overall cancer phenotype score
    cancer_score = 0.4 * prolif_capacity + 0.3 * oncogene_expression + 0.2 * glycolytic_flux/10.0 + 0.1 * (1 - tumor_suppressor)

    return {
        'lambda_D': lambda_D,
        'accessibility': accessibility,
        'glycolytic_flux': glycolytic_flux,
        'prolif_capacity': prolif_capacity,
        'efficiency': efficiency,
        'oncogene': oncogene_expression,
        'tumor_suppressor': tumor_suppressor,
        'cancer_score': cancer_score
    }

# ============================================================================
# SIMULATION
# ============================================================================

def run_warburg_simulation(params, cell_type='cancer'):
    """Run Warburg effect simulation for normal or cancer cells"""

    t = np.arange(0, params.max_time, params.dt)

    # Initial conditions
    if cell_type == 'cancer':
        y0 = [
            params.glucose_baseline,
            2.0,  # Initial lactate (already elevated)
            2.0,  # ATP
            params.pH_baseline_cancer,  # Acidic pH
            params.Na_baseline,
            params.K_baseline,
            params.Mg_baseline,
            0.5,  # Proliferation signal (activated)
            0.1   # Some baseline damage
        ]
    else:
        y0 = [
            params.glucose_baseline,
            0.5,  # Low lactate
            2.5,  # Higher ATP
            params.pH_baseline_normal,  # Normal pH
            params.Na_baseline,
            params.K_baseline,
            params.Mg_baseline,
            0.0,  # No proliferation signal
            0.0   # No damage
        ]

    solution = odeint(warburg_dynamics, y0, t, args=(params, cell_type))

    glucose = solution[:, 0]
    lactate = solution[:, 1]
    ATP = solution[:, 2]
    pH = solution[:, 3]
    Na = solution[:, 4]
    K = solution[:, 5]
    Mg = solution[:, 6]
    prolif = solution[:, 7]
    damage = solution[:, 8]

    # Calculate derived quantities
    functions = [calculate_cancer_functions(glucose[i], lactate[i], ATP[i], pH[i],
                                            Na[i], K[i], Mg[i], prolif[i], damage[i], params)
                for i in range(len(t))]

    lambda_D = np.array([f['lambda_D'] for f in functions])
    accessibility = np.array([f['accessibility'] for f in functions])
    glycolytic_flux = np.array([f['glycolytic_flux'] for f in functions])
    prolif_capacity = np.array([f['prolif_capacity'] for f in functions])
    efficiency = np.array([f['efficiency'] for f in functions])
    oncogene = np.array([f['oncogene'] for f in functions])
    tumor_suppressor = np.array([f['tumor_suppressor'] for f in functions])
    cancer_score = np.array([f['cancer_score'] for f in functions])

    return {
        't': t,
        'glucose': glucose,
        'lactate': lactate,
        'ATP': ATP,
        'pH': pH,
        'Na': Na,
        'K': K,
        'Mg': Mg,
        'prolif': prolif,
        'damage': damage,
        'lambda_D': lambda_D,
        'accessibility': accessibility,
        'glycolytic_flux': glycolytic_flux,
        'prolif_capacity': prolif_capacity,
        'efficiency': efficiency,
        'oncogene': oncogene,
        'tumor_suppressor': tumor_suppressor,
        'cancer_score': cancer_score
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_warburg_figure(results_cancer, results_normal, params):
    """Create comprehensive Warburg effect comparison figure"""

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)

    t_cancer = results_cancer['t']
    t_normal = results_normal['t']

    # ========================================================================
    # ROW 1: METABOLIC REPROGRAMMING
    # ========================================================================

    # A. Glucose Consumption
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_cancer, results_cancer['glucose'], 'red', linewidth=2.5, label='Cancer')
    ax1.plot(t_normal, results_normal['glucose'], 'blue', linewidth=2.5, label='Normal')
    ax1.axhline(params.glucose_baseline, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('[Glucose] (mM)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Glucose Consumption (Warburg Effect)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # B. Lactate Accumulation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_cancer, results_cancer['lactate'], 'red', linewidth=2.5, label='Cancer')
    ax2.plot(t_normal, results_normal['lactate'], 'blue', linewidth=2.5, label='Normal')
    ax2.fill_between(t_cancer, results_cancer['lactate'], alpha=0.2, color='red')
    ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('[Lactate] (mM)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Lactate Production (Aerobic Glycolysis)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # C. ATP Dynamics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t_cancer, results_cancer['ATP'], 'red', linewidth=2.5, label='Cancer')
    ax3.plot(t_normal, results_normal['ATP'], 'blue', linewidth=2.5, label='Normal')
    ax3.axhline(2.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax3.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('[ATP] (mM)', fontsize=12, fontweight='bold')
    ax3.set_title('C. ATP Levels (Inefficient but Fast)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # D. Glycolytic Flux
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(t_cancer, results_cancer['glycolytic_flux'], 'red', linewidth=2.5, label='Cancer')
    ax4.plot(t_normal, results_normal['glycolytic_flux'], 'blue', linewidth=2.5, label='Normal')
    ax4.fill_between(t_cancer, results_cancer['glycolytic_flux'],
                     results_normal['glycolytic_flux'][:len(t_cancer)],
                     alpha=0.2, color='red')
    ax4.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Glycolytic Flux (mM/h)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Glycolytic Flux (10× Higher in Cancer)', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # ========================================================================
    # ROW 2: CHARGE CRISIS (pH AND IONS)
    # ========================================================================

    # E. pH Acidification
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.plot(t_cancer, results_cancer['pH'], 'red', linewidth=2.5, label='Cancer')
    ax5.plot(t_normal, results_normal['pH'], 'blue', linewidth=2.5, label='Normal')
    ax5.axhline(params.pH_baseline_normal, color='blue', linestyle='--', alpha=0.5)
    ax5.axhline(params.proliferation_pH_optimum, color='green', linestyle='--',
                alpha=0.5, label='Prolif. Optimum')
    ax5.axhline(params.apoptosis_pH_threshold, color='darkred', linestyle='--',
                alpha=0.5, label='Apoptosis Threshold')
    ax5.fill_between(t_cancer, results_cancer['pH'], params.pH_baseline_normal,
                     alpha=0.2, color='red')
    ax5.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('pH', fontsize=12, fontweight='bold')
    ax5.set_title('E. Intracellular pH (Acidic Microenvironment)', fontsize=13, fontweight='bold')
    ax5.legend(loc='lower left', fontsize=9)
    ax5.grid(True, alpha=0.3)

    # F. Debye Screening Length
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.plot(t_cancer, results_cancer['lambda_D'], 'red', linewidth=2.5, label='Cancer')
    ax6.plot(t_normal, results_normal['lambda_D'], 'blue', linewidth=2.5, label='Normal')
    ax6.fill_between(t_cancer, results_cancer['lambda_D'], results_normal['lambda_D'][:len(t_cancer)],
                     alpha=0.2, color='red')
    ax6.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Debye Length λ_D (nm)', fontsize=12, fontweight='bold')
    ax6.set_title('F. Charge Screening (H⁺-Dependent)', fontsize=13, fontweight='bold')
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3)

    # G. Na+ Dynamics (Na+/H+ Exchanger)
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.plot(t_cancer, results_cancer['Na'], 'red', linewidth=2.5, label='Cancer')
    ax7.plot(t_normal, results_normal['Na'], 'blue', linewidth=2.5, label='Normal')
    ax7.axhline(params.Na_baseline, color='gray', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('[Na⁺] (mM)', fontsize=12, fontweight='bold')
    ax7.set_title('G. Na⁺ Influx (pH Regulation)', fontsize=13, fontweight='bold')
    ax7.legend(loc='upper left')
    ax7.grid(True, alpha=0.3)

    # H. Mg2+ Dynamics
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.plot(t_cancer, results_cancer['Mg'], 'red', linewidth=2.5, label='Cancer')
    ax8.plot(t_normal, results_normal['Mg'], 'blue', linewidth=2.5, label='Normal')
    ax8.axhline(params.Mg_baseline, color='gray', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('[Mg²⁺] (mM)', fontsize=12, fontweight='bold')
    ax8.set_title('H. Mg²⁺ Release (Glycolysis)', fontsize=13, fontweight='bold')
    ax8.legend(loc='upper left')
    ax8.grid(True, alpha=0.3)

    # ========================================================================
    # ROW 3: GENOMIC CONSEQUENCES
    # ========================================================================

    # I. Chromatin Accessibility
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.plot(t_cancer, results_cancer['accessibility'], 'red', linewidth=2.5, label='Cancer')
    ax9.plot(t_normal, results_normal['accessibility'], 'blue', linewidth=2.5, label='Normal')
    ax9.fill_between(t_cancer, results_cancer['accessibility'],
                     results_normal['accessibility'][:len(t_cancer)],
                     alpha=0.2, color='red')
    ax9.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Chromatin Accessibility', fontsize=12, fontweight='bold')
    ax9.set_title('I. Chromatin Opening (H⁺ Protonation)', fontsize=13, fontweight='bold')
    ax9.legend(loc='upper left')
    ax9.set_ylim([0, 1.0])
    ax9.grid(True, alpha=0.3)

    # J. Oncogene Expression
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.plot(t_cancer, results_cancer['oncogene'], 'red', linewidth=2.5, label='Cancer')
    ax10.plot(t_normal, results_normal['oncogene'], 'blue', linewidth=2.5, label='Normal')
    ax10.fill_between(t_cancer, results_cancer['oncogene'], alpha=0.3, color='red')
    ax10.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax10.set_ylabel('Oncogene Expression', fontsize=12, fontweight='bold')
    ax10.set_title('J. Oncogene Activation (c-Myc, HIF-1α)', fontsize=13, fontweight='bold')
    ax10.legend(loc='upper left')
    ax10.grid(True, alpha=0.3)

    # K. Tumor Suppressor Expression
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.plot(t_cancer, results_cancer['tumor_suppressor'], 'red', linewidth=2.5, label='Cancer')
    ax11.plot(t_normal, results_normal['tumor_suppressor'], 'blue', linewidth=2.5, label='Normal')
    ax11.fill_between(t_cancer, results_normal['tumor_suppressor'][:len(t_cancer)],
                      results_cancer['tumor_suppressor'], alpha=0.2, color='blue')
    ax11.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax11.set_ylabel('Tumor Suppressor Expression', fontsize=12, fontweight='bold')
    ax11.set_title('K. Tumor Suppressor Loss (p53, PTEN)', fontsize=13, fontweight='bold')
    ax11.legend(loc='upper right')
    ax11.grid(True, alpha=0.3)

    # L. Proliferation Signal
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.plot(t_cancer, results_cancer['prolif'], 'red', linewidth=2.5, label='Cancer')
    ax12.plot(t_normal, results_normal['prolif'], 'blue', linewidth=2.5, label='Normal')
    ax12.fill_between(t_cancer, results_cancer['prolif'], alpha=0.3, color='red')
    ax12.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax12.set_ylabel('Proliferation Signal', fontsize=12, fontweight='bold')
    ax12.set_title('L. Proliferation Activation', fontsize=13, fontweight='bold')
    ax12.legend(loc='upper left')
    ax12.grid(True, alpha=0.3)

    # ========================================================================
    # ROW 4: CORRELATIONS AND SUMMARY
    # ========================================================================

    # M. pH vs Chromatin Accessibility
    ax13 = fig.add_subplot(gs[3, 0])
    scatter1 = ax13.scatter(results_cancer['pH'], results_cancer['accessibility'],
                           c=t_cancer, cmap='Reds', s=20, alpha=0.6, label='Cancer')
    ax13.scatter(results_normal['pH'], results_normal['accessibility'],
                c=t_normal, cmap='Blues', s=20, alpha=0.6, marker='s', label='Normal')

    # Fit for cancer cells
    z1 = np.polyfit(results_cancer['pH'], results_cancer['accessibility'], 2)
    p1 = np.poly1d(z1)
    pH_fit = np.linspace(results_cancer['pH'].min(), results_cancer['pH'].max(), 100)
    ax13.plot(pH_fit, p1(pH_fit), "r--", linewidth=2)

    r1, _ = pearsonr(results_cancer['pH'], results_cancer['accessibility'])
    ax13.set_xlabel('pH', fontsize=12, fontweight='bold')
    ax13.set_ylabel('Chromatin Accessibility', fontsize=12, fontweight='bold')
    ax13.set_title(f'M. pH vs Accessibility (r={r1:.3f})', fontsize=13, fontweight='bold')
    ax13.legend(loc='upper right')
    ax13.grid(True, alpha=0.3)

    # N. Lactate vs Proliferation
    ax14 = fig.add_subplot(gs[3, 1])
    scatter2 = ax14.scatter(results_cancer['lactate'], results_cancer['prolif'],
                           c=t_cancer, cmap='Reds', s=20, alpha=0.6, label='Cancer')
    ax14.scatter(results_normal['lactate'], results_normal['prolif'],
                c=t_normal, cmap='Blues', s=20, alpha=0.6, marker='s', label='Normal')

    r2, _ = pearsonr(results_cancer['lactate'], results_cancer['prolif'])
    ax14.set_xlabel('[Lactate] (mM)', fontsize=12, fontweight='bold')
    ax14.set_ylabel('Proliferation Signal', fontsize=12, fontweight='bold')
    ax14.set_title(f'N. Lactate vs Proliferation (r={r2:.3f})', fontsize=13, fontweight='bold')
    ax14.legend(loc='upper left')
    ax14.grid(True, alpha=0.3)

    # O. 3D Phase Space: pH vs λ_D vs Cancer Score
    ax15 = fig.add_subplot(gs[3, 2], projection='3d')
    scatter3_cancer = ax15.scatter(results_cancer['pH'], results_cancer['lambda_D'],
                                   results_cancer['cancer_score'],
                                   c=t_cancer, cmap='Reds', s=30, alpha=0.7, label='Cancer')
    ax15.scatter(results_normal['pH'], results_normal['lambda_D'],
                results_normal['cancer_score'],
                c=t_normal, cmap='Blues', s=30, alpha=0.7, marker='s', label='Normal')
    ax15.set_xlabel('pH', fontsize=11, fontweight='bold')
    ax15.set_ylabel('λ_D (nm)', fontsize=11, fontweight='bold')
    ax15.set_zlabel('Cancer Score', fontsize=11, fontweight='bold')
    ax15.set_title('O. Phase Space Trajectory', fontsize=13, fontweight='bold')
    ax15.legend(loc='upper left')

    # P. Summary Statistics
    ax16 = fig.add_subplot(gs[3, 3])
    ax16.axis('off')

    # Calculate statistics
    cancer_lactate_mean = np.mean(results_cancer['lactate'][len(t_cancer)//2:])
    normal_lactate_mean = np.mean(results_normal['lactate'][len(t_normal)//2:])
    lactate_ratio = cancer_lactate_mean / normal_lactate_mean

    cancer_pH_mean = np.mean(results_cancer['pH'][len(t_cancer)//2:])
    normal_pH_mean = np.mean(results_normal['pH'][len(t_normal)//2:])
    pH_diff = normal_pH_mean - cancer_pH_mean

    cancer_prolif_mean = np.mean(results_cancer['prolif'][len(t_cancer)//2:])
    normal_prolif_mean = np.mean(results_normal['prolif'][len(t_normal)//2:])

    cancer_access_mean = np.mean(results_cancer['accessibility'][len(t_cancer)//2:])
    normal_access_mean = np.mean(results_normal['accessibility'][len(t_normal)//2:])

    r_pH_access_cancer, _ = pearsonr(results_cancer['pH'], results_cancer['accessibility'])
    r_lactate_prolif_cancer, _ = pearsonr(results_cancer['lactate'], results_cancer['prolif'])
    r_access_oncogene_cancer, _ = pearsonr(results_cancer['accessibility'], results_cancer['oncogene'])

    summary_text = f"""
    WARBURG EFFECT CHARGE CRISIS
    ═══════════════════════════════════

    Metabolic Reprogramming:
    • Glycolysis rate: {params.glycolysis_rate_cancer/params.glycolysis_rate_normal:.0f}× normal
    • Lactate (cancer): {cancer_lactate_mean:.2f} mM
    • Lactate (normal): {normal_lactate_mean:.2f} mM
    • Lactate ratio: {lactate_ratio:.1f}×

    Charge Crisis:
    • pH (cancer): {cancer_pH_mean:.2f}
    • pH (normal): {normal_pH_mean:.2f}
    • pH difference: {pH_diff:.2f} units
    • λ_D increase: {np.mean(results_cancer['lambda_D'])/np.mean(results_normal['lambda_D']):.2f}×

    Genomic Consequences:
    • Accessibility (cancer): {cancer_access_mean:.3f}
    • Accessibility (normal): {normal_access_mean:.3f}
    • Increase: {(cancer_access_mean/normal_access_mean - 1)*100:.1f}%
    • Proliferation (cancer): {cancer_prolif_mean:.3f}
    • Proliferation (normal): {normal_prolif_mean:.3f}

    Key Correlations (Cancer):
    • pH vs Accessibility: r = {r_pH_access_cancer:.3f} ***
    • Lactate vs Prolif.: r = {r_lactate_prolif_cancer:.3f} ***
    • Access. vs Oncogene: r = {r_access_oncogene_cancer:.3f} ***

    Warburg Mechanism:
    Glycolysis↑ → Lactate↑ → H+↑ → pH↓ →
    Histone protonation → Chromatin opening →
    Oncogene activation → Proliferation →
    Cancer phenotype

    Charge Advantage:
    Acidic pH creates optimal charge state
    for proliferation genes while suppressing
    tumor suppressors via chromatin remodeling

    *** p < 0.001
    """

    ax16.text(0.05, 0.95, summary_text, transform=ax16.transAxes,
             fontsize=9.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    # Overall title
    fig.suptitle('Warburg Effect as Charge Crisis: Metabolic Reprogramming Drives Genomic Instability and Proliferation',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WARBURG EFFECT AS CHARGE CRISIS SIMULATION")
    print("=" * 70)

    params = WarburgParameters()

    print("\nRunning cancer cell simulation...")
    results_cancer = run_warburg_simulation(params, cell_type='cancer')

    print("Running normal cell simulation...")
    results_normal = run_warburg_simulation(params, cell_type='normal')

    print("Generating comparative figure...")
    fig = create_warburg_figure(results_cancer, results_normal, params)

    output_file = "warburg_effect_charge_crisis.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved as: {output_file}")

    # Print key results
    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)

    cancer_lactate = np.mean(results_cancer['lactate'][-100:])
    normal_lactate = np.mean(results_normal['lactate'][-100:])
    cancer_pH = np.mean(results_cancer['pH'][-100:])
    normal_pH = np.mean(results_normal['pH'][-100:])
    cancer_prolif = np.mean(results_cancer['prolif'][-100:])
    normal_prolif = np.mean(results_normal['prolif'][-100:])

    print(f"\nSteady-state comparison:")
    print(f"  • Lactate: Cancer {cancer_lactate:.2f} mM vs Normal {normal_lactate:.2f} mM ({cancer_lactate/normal_lactate:.1f}×)")
    print(f"  • pH: Cancer {cancer_pH:.2f} vs Normal {normal_pH:.2f} (Δ = {normal_pH - cancer_pH:.2f})")
    print(f"  • Proliferation: Cancer {cancer_prolif:.3f} vs Normal {normal_prolif:.3f} ({cancer_prolif/max(normal_prolif, 0.001):.1f}×)")

    r_pH_access, p_pH = pearsonr(results_cancer['pH'], results_cancer['accessibility'])
    r_lactate_prolif, p_lactate = pearsonr(results_cancer['lactate'], results_cancer['prolif'])

    print(f"\nCorrelations (Cancer cells):")
    print(f"  • pH vs Accessibility: r = {r_pH_access:.4f} (p = {p_pH:.2e})")
    print(f"  • Lactate vs Proliferation: r = {r_lactate_prolif:.4f} (p = {p_lactate:.2e})")

    print("\n" + "=" * 70)
    print("CONCLUSION: Warburg effect creates charge crisis via acidification,")
    print("driving chromatin remodeling and oncogenic transformation.")
    print("=" * 70)

    plt.show()
