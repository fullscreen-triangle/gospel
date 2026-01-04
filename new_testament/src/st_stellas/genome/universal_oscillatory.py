# core/oscillatory_foundation.py

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal, fft
from scipy.optimize import minimize

@dataclass
class OscillatoryScale:
    """Represents one of the 8 oscillatory scales"""
    name: str
    frequency_range: Tuple[float, float]  # Hz
    period_range: Tuple[float, float]     # seconds
    biological_process: str
    coupling_strength: float = 1.0

class UniversalOscillatoryEngine:
    """
    Universal Oscillatory Framework - Foundation for all biological analysis
    From oscillatory genomics paper: 8-scale hierarchy
    """

    def __init__(self):
        self.scales = self._initialize_scales()
        self.s_entropy_calculator = SEntropyCalculator()
        self.resonance_detector = ResonanceDetector()

    def _initialize_scales(self) -> Dict[str, OscillatoryScale]:
        """Initialize the 8 oscillatory scales"""
        return {
            'quantum_genomic': OscillatoryScale(
                name='Quantum Genomic Coherence',
                frequency_range=(1e12, 1e15),
                period_range=(1e-15, 1e-12),
                biological_process='DNA quantum coherence, base pair oscillations'
            ),
            'molecular_base': OscillatoryScale(
                name='Molecular Base Oscillations',
                frequency_range=(1e9, 1e12),
                period_range=(1e-12, 1e-9),
                biological_process='Hydrogen bond vibrations, electron tunneling'
            ),
            'gene_circuit': OscillatoryScale(
                name='Gene Circuit Dynamics',
                frequency_range=(1e-1, 1e2),
                period_range=(0.01, 10),
                biological_process='Gene expression oscillations, transcription bursts'
            ),
            'regulatory_network': OscillatoryScale(
                name='Regulatory Network Integration',
                frequency_range=(1e-2, 1e1),
                period_range=(0.1, 100),
                biological_process='Regulatory feedback loops, signaling cascades'
            ),
            'cellular_info': OscillatoryScale(
                name='Cellular Information Processing',
                frequency_range=(1e-4, 1e-1),
                period_range=(10, 10000),
                biological_process='Cell cycle, metabolic oscillations'
            ),
            'genomic_cellular': OscillatoryScale(
                name='Genomic-Cellular Coordination',
                frequency_range=(1e-5, 1e-2),
                period_range=(100, 100000),
                biological_process='DNA consultation events, chromatin remodeling'
            ),
            'environmental_genomic': OscillatoryScale(
                name='Environmental Genomic Response',
                frequency_range=(1e-6, 1e-3),
                period_range=(1000, 1e6),
                biological_process='Environmental adaptation, stress responses'
            ),
            'evolutionary_genomic': OscillatoryScale(
                name='Evolutionary Genomic Dynamics',
                frequency_range=(1e-8, 1e-5),
                period_range=(1e5, 1e8),
                biological_process='Evolutionary selection, population genetics'
            )
        }

    def extract_oscillatory_signature(self, signal_data: np.ndarray,
                                     sampling_rate: float) -> Dict:
        """
        Extract oscillatory signature from biological signal
        Returns frequencies, amplitudes, and phases across all scales
        """
        # Perform FFT
        frequencies = fft.fftfreq(len(signal_data), 1/sampling_rate)
        fft_values = fft.fft(signal_data)
        power_spectrum = np.abs(fft_values)**2

        # Extract dominant frequencies for each scale
        scale_signatures = {}

        for scale_name, scale in self.scales.items():
            # Find frequencies within this scale's range
            mask = (frequencies >= scale.frequency_range[0]) & \
                   (frequencies <= scale.frequency_range[1])

            scale_freqs = frequencies[mask]
            scale_power = power_spectrum[mask]

            if len(scale_freqs) > 0:
                # Find dominant frequency in this scale
                dominant_idx = np.argmax(scale_power)
                dominant_freq = scale_freqs[dominant_idx]
                dominant_amplitude = np.sqrt(scale_power[dominant_idx])
                dominant_phase = np.angle(fft_values[mask][dominant_idx])

                scale_signatures[scale_name] = {
                    'frequency': dominant_freq,
                    'amplitude': dominant_amplitude,
                    'phase': dominant_phase,
                    'power': scale_power[dominant_idx],
                    'all_frequencies': scale_freqs.tolist(),
                    'all_powers': scale_power.tolist()
                }

        return scale_signatures

    def calculate_cross_scale_coupling(self, signature1: Dict,
                                      signature2: Dict) -> Dict:
        """
        Calculate coupling strength between two oscillatory signatures
        From oscillatory paper: resonance condition |ω_i - n·ω_j| < γ
        """
        couplings = {}

        for scale_name in signature1.keys():
            if scale_name in signature2:
                freq1 = signature1[scale_name]['frequency']
                freq2 = signature2[scale_name]['frequency']
                amp1 = signature1[scale_name]['amplitude']
                amp2 = signature2[scale_name]['amplitude']
                phase1 = signature1[scale_name]['phase']
                phase2 = signature2[scale_name]['phase']

                # Check for resonance (including harmonics)
                resonances = []
                for n in range(1, 6):  # Check first 5 harmonics
                    freq_diff = abs(freq1 - n * freq2)
                    gamma_coupling = 0.1 * freq2  # 10% tolerance

                    if freq_diff < gamma_coupling:
                        resonance_strength = 1.0 / (1.0 + freq_diff / gamma_coupling)
                        resonances.append({
                            'harmonic': n,
                            'frequency_difference': freq_diff,
                            'resonance_strength': resonance_strength
                        })

                # Calculate phase coherence
                phase_diff = abs(phase1 - phase2)
                phase_coherence = np.cos(phase_diff)

                # Calculate amplitude coupling
                amplitude_coupling = (amp1 * amp2) / (amp1 + amp2)

                couplings[scale_name] = {
                    'resonances': resonances,
                    'phase_coherence': phase_coherence,
                    'amplitude_coupling': amplitude_coupling,
                    'overall_coupling': (
                        len(resonances) * phase_coherence * amplitude_coupling
                    )
                }

        return couplings

class SEntropyCalculator:
    """
    Calculate S-entropy coordinates for genomic compression
    From oscillatory paper: O(log N) complexity
    """

    def calculate_s_entropy_coordinates(self, oscillatory_signature: Dict,
                                       genomic_context: Dict) -> Dict:
        """
        Calculate tri-dimensional S-entropy coordinates
        S_knowledge, S_time, S_entropy
        """
        # S_knowledge: Information content
        s_knowledge = self._calculate_knowledge_entropy(
            oscillatory_signature, genomic_context
        )

        # S_time: Temporal dynamics
        s_time = self._calculate_temporal_entropy(oscillatory_signature)

        # S_entropy: Thermodynamic entropy
        s_entropy = self._calculate_thermodynamic_entropy(
            oscillatory_signature, genomic_context
        )

        return {
            'S_knowledge': s_knowledge,
            'S_time': s_time,
            'S_entropy': s_entropy,
            'coordinates': (s_knowledge, s_time, s_entropy),
            'compression_ratio': self._calculate_compression_ratio(
                oscillatory_signature
            )
        }

    def _calculate_knowledge_entropy(self, signature: Dict,
                                    context: Dict) -> float:
        """
        H(S) + Σ I(variant_i, phenotype)
        """
        # Shannon entropy of oscillatory signature
        frequencies = []
        amplitudes = []

        for scale_data in signature.values():
            frequencies.append(scale_data['frequency'])
            amplitudes.append(scale_data['amplitude'])

        # Normalize amplitudes to probabilities
        total_amplitude = sum(amplitudes)
        probabilities = [a / total_amplitude for a in amplitudes]

        # Shannon entropy
        shannon_entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

        # Mutual information with phenotype (from context)
        mutual_info = context.get('phenotype_correlation', 0.0)

        return shannon_entropy + mutual_info

    def _calculate_temporal_entropy(self, signature: Dict) -> float:
        """
        Σ τ_oscillatory(scale) · w_genomic(scale)
        """
        temporal_entropy = 0.0

        for scale_name, scale_data in signature.items():
            # Period = 1 / frequency
            period = 1.0 / scale_data['frequency']

            # Weight by amplitude (importance)
            weight = scale_data['amplitude']

            temporal_entropy += period * weight

        return temporal_entropy

    def _calculate_thermodynamic_entropy(self, signature: Dict,
                                        context: Dict) -> float:
        """
        H(V|S,R,E) - H_baseline(genomic_equilibrium)
        """
        # Current entropy
        current_entropy = self._calculate_knowledge_entropy(signature, context)

        # Baseline equilibrium entropy (from reference)
        baseline_entropy = context.get('baseline_entropy', 1.0)

        # Deviation from equilibrium
        return current_entropy - baseline_entropy

    def _calculate_compression_ratio(self, signature: Dict) -> float:
        """
        Calculate compression achieved through S-entropy
        From oscillatory paper: O(N·L·D) → O(log(N·L))
        """
        # Original complexity
        n_variants = len(signature)
        l_sequence = 1000  # Typical sequence length
        d_features = 10    # Feature dimensions

        original_complexity = n_variants * l_sequence * d_features

        # Compressed complexity
        compressed_complexity = np.log2(n_variants * l_sequence)

        return original_complexity / compressed_complexity

class ResonanceDetector:
    """
    Detect resonance patterns across oscillatory scales
    """

    def detect_resonances(self, signatures: List[Dict]) -> List[Dict]:
        """
        Find resonance patterns across multiple signatures
        """
        resonances = []

        for i, sig1 in enumerate(signatures):
            for j, sig2 in enumerate(signatures[i+1:], start=i+1):
                for scale_name in sig1.keys():
                    if scale_name in sig2:
                        freq1 = sig1[scale_name]['frequency']
                        freq2 = sig2[scale_name]['frequency']

                        # Check harmonics
                        for n in range(1, 6):
                            freq_diff = abs(freq1 - n * freq2)
                            tolerance = 0.1 * freq2

                            if freq_diff < tolerance:
                                resonances.append({
                                    'signature1_idx': i,
                                    'signature2_idx': j,
                                    'scale': scale_name,
                                    'harmonic': n,
                                    'frequency1': freq1,
                                    'frequency2': freq2,
                                    'frequency_difference': freq_diff,
                                    'resonance_strength': 1.0 / (1.0 + freq_diff/tolerance)
                                })

        return resonances
