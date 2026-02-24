"""
Physics modules for molecular oscillator properties

Provides:
- Molecular species database (N2, O2, H+, H2O, CO2)
- Harmonic coincidence detection
- Heisenberg uncertainty bypass proof
- Proton trajectory validation (H-bond oscillators)
- H-bond charge oscillation dynamics
"""

from .molecular_oscillators import (
    MolecularSpecies,
    MolecularOscillatorGenerator,
    MOLECULAR_DATABASE,
    get_species_properties,
    list_available_species,
    compare_species
)

from .harmonic_coincidence import (
    HarmonicCoincidence,
    HarmonicCoincidenceDetector,
    calculate_beat_frequency_precision,
    find_coincidence_chains
)

from .heisenberg_bypass import (
    HeisenbergBypass,
    demonstrate_bypass
)

from .virtual_detectors import (
    VirtualDetectorState,
    VirtualMassSpectrometer,
    VirtualIonDetector,
    VirtualPhotodetector,
    VirtualDetectorFactory
)

from .hardware_harvesting import (
    HardwareOscillator,
    HardwareFrequencyHarvester,
    ScreenLEDHarvester,
    CPUClockHarvester
)

from .genomic_charged_fluid import (
    GenomicChargedFluid,
    EmptyDictionaryValidator,
    GenomicChargedFluidState,
    TransportCoefficients,
    SectionPrediction,
    EmptyDictionaryResult,
    validate_paper_claims
)

from .proton_trajectory_validation import (
    HBondProtonOscillator,
    DNACapacitor,
    TripleEquivalenceValidator,
    ProtonTrajectoryValidator,
    IdealGasGenomicValidator,
    PhaseLockValidator,
    ValidationRunner,
    HBondOscillatorState,
    DNACapacitorState,
    TripleEquivalenceResult,
    ProtonTrajectoryResult,
    IdealGasGenomicResult,
    PhaseLockResult,
)

from .hbond_charge_oscillation import (
    HBondProton,
    BasePair,
    DNAHelixSegment,
    ChargeOscillationValidator,
    ChargeOscillationResult,
    ComplementarityResult,
    run_charge_oscillation_validation,
)

__all__ = [
    # Molecular oscillators
    'MolecularSpecies',
    'MolecularOscillatorGenerator',
    'MOLECULAR_DATABASE',
    'get_species_properties',
    'list_available_species',
    'compare_species',

    # Harmonic coincidence
    'HarmonicCoincidence',
    'HarmonicCoincidenceDetector',
    'calculate_beat_frequency_precision',
    'find_coincidence_chains',

    # Heisenberg bypass
    'HeisenbergBypass',
    'demonstrate_bypass',

    # Virtual detectors
    'VirtualDetectorState',
    'VirtualMassSpectrometer',
    'VirtualIonDetector',
    'VirtualPhotodetector',
    'VirtualDetectorFactory',

    # Hardware harvesting
    'HardwareOscillator',
    'HardwareFrequencyHarvester',
    'ScreenLEDHarvester',
    'CPUClockHarvester',

    # Genomic validation
    'GenomicChargedFluid',
    'EmptyDictionaryValidator',
    'GenomicChargedFluidState',
    'TransportCoefficients',
    'SectionPrediction',
    'EmptyDictionaryResult',
    'validate_paper_claims',

    # Proton trajectory validation
    'HBondProtonOscillator',
    'DNACapacitor',
    'TripleEquivalenceValidator',
    'ProtonTrajectoryValidator',
    'IdealGasGenomicValidator',
    'PhaseLockValidator',
    'ValidationRunner',
    'HBondOscillatorState',
    'DNACapacitorState',
    'TripleEquivalenceResult',
    'ProtonTrajectoryResult',
    'IdealGasGenomicResult',
    'PhaseLockResult',

    # H-bond charge oscillation
    'HBondProton',
    'BasePair',
    'DNAHelixSegment',
    'ChargeOscillationValidator',
    'ChargeOscillationResult',
    'ComplementarityResult',
    'run_charge_oscillation_validation',
]
