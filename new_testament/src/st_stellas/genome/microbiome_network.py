# core/microbiome_coupling.py

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MicrobiomeCouplingState:
    """State of microbiome multi-scale coupling"""
    C_12: float  # Cellular-population coupling
    C_23: float  # Population-community coupling
    C_34: float  # Community-host coupling
    C_45: float  # Host-environment coupling
    dysbiosis_score: float
    coupling_matrix: np.ndarray

class MultiScaleMicrobiomeNetwork:
    """
    Microbiome as multi-scale oscillatory evidence network
    From microbiome paper: 5 temporal scales
    """

    def __init__(self):
        # Healthy reference couplings
        self.healthy_reference = {
            'C_12': 0.67,
            'C_23': 0.72,
            'C_34': 0.84,
            'C_45': 0.59
        }

        # Clinical importance weights
        self.weights = {
            'C_12': 0.25,
            'C_23': 0.30,
            'C_34': 0.35,
            'C_45': 0.10
        }

    def calculate_microbiome_coupling(self,
                                     intracellular: Dict,
                                     metabolome: Dict,
                                     genome_data: Dict) -> Dict:
        """
        Calculate microbiome coupling across 5 temporal scales
        """
        print("  → Calculating microbiome multi-scale coupling...")

        # Scale 1: Cellular metabolic (T1 ~ 0.1-10 hours)
        cellular_efficiency = self._calculate_cellular_efficiency(metabolome)

        # Scale 2: Population growth (T2 ~ 1-100 hours)
        population_modulation = self._calculate_population_modulation(genome_data)

        # Scale 3: Community dynamics (T3 ~ 10-1000 hours)
        community_stability = self._calculate_community_stability(metabolome)

        # Scale 4: Host circadian (T4 ~ 24 hours)
        circadian_strength = self._calculate_circadian_strength(genome_data)

        # Scale 5: Environmental (T5 ~ 100-10000 hours)
        environmental_adaptation = self._calculate_environmental_adaptation(genome_data)

        # Calculate coupling matrix
        C_matrix = self._build_coupling_matrix(
            cellular_efficiency,
            population_modulation,
            community_stability,
            circadian_strength,
            environmental_adaptation
        )

        # Calculate dysbiosis score
        dysbiosis_score = self._calculate_dysbiosis_score(C_matrix)

        coupling_state = MicrobiomeCouplingState(
            C_12=C_matrix[0, 1],
            C_23=C_matrix[1, 2],
            C_34=C_matrix[2, 3],
            C_45=C_matrix[3, 4],
            dysbiosis_score=dysbiosis_score,
            coupling_matrix=C_matrix
        )

        print(f"    ✓ C_12 (cellular-population): {C_matrix[0,1]:.3f}")
        print(f"    ✓ C_23 (population-community): {C_matrix[1,2]:.3f}")
        print(f"    ✓ C_34 (community-host): {C_matrix[2,3]:.3f}")
        print(f"    ✓ C_45 (host-environment): {C_matrix[3,4]:.3f}")
        print(f"    ✓ Dysbiosis score: {dysbiosis_score:.3f}")

        return {
            'coupling_state': coupling_state,
            'C_12': C_matrix[0, 1],
            'C_23': C_matrix[1, 2],
            'C_34': C_matrix[2, 3],
            'C_45': C_matrix[3, 4],
            'dysbiosis_score': dysbiosis_score,
            'scale_health': self._analyze_scale_health(C_matrix)
        }

    def _calculate_cellular_efficiency(self, metabolome: Dict) -> float:
        """Calculate cellular metabolic efficiency"""
        # ATP oscillation amplitude
        atp_amplitude = metabolome.get('atp_amplitude', 1.284)
        baseline_amplitude = 1.284  # From microbiome paper

        return min(1.0, atp_amplitude / baseline_amplitude)

    def _calculate_population_modulation(self, genome_data: Dict) -> float:
        """Calculate population growth modulation from immune genes"""
        # Find immune-related variants
        immune_genes = ['IL6', 'TNF', 'IFNG', 'IL10', 'IL1B']

        if 'variants' not in genome_data:
            return 0.7  # Default

        immune_variants = [
            v for v in genome_data['variants']
            if v.get('gene') in immune_genes
        ]

        # Strong immune = better population control
        modulation = 0.5
        for variant in immune_variants:
            if not variant.get('pathogenic', False):
                modulation += 0.1

        return min(1.0, modulation)

    def _calculate_community_stability(self, metabolome: Dict) -> float:
        """Calculate community dynamics stability"""
        # Metabolic diversity indicates community stability
        n_metabolites = len(metabolome.get('metabolites', []))
        baseline_metabolites = 1000

        stability = 0.5 + 0.5 * min(1.0, n_metabolites / baseline_metabolites)
        return stability

    def _calculate_circadian_strength(self, genome_data: Dict) -> float:
        """Calculate circadian rhythm strength from clock genes"""
        clock_genes = ['CLOCK', 'PER1', 'PER2', 'PER3', 'CRY1', 'CRY2', 'ARNTL']

        if 'variants' not in genome_data:
            return 0.8  # Default

        clock_variants = [
            v for v in genome_data['variants']
            if v.get('gene') in clock_genes
        ]

        # Fewer pathogenic variants = stronger circadian
        strength = 1.0
        for variant in clock_variants:
            if variant.get('pathogenic', False):
                strength *= 0.9

        return max(0.5, strength)

    def _calculate_environmental_adaptation(self, genome_data: Dict) -> float:
        """Calculate environmental adaptation capacity"""
        stress_genes = ['NR3C1', 'FKBP5', 'CRHR1', 'HSP90AA1']

        if 'variants' not in genome_data:
            return 0.6  # Default

        stress_variants = [
            v for v in genome_data['variants']
            if v.get('gene') in stress_genes
        ]

        adaptation = 0.6
        for variant in stress_variants:
            if not variant.get('pathogenic', False):
                adaptation += 0.1

        return min(1.0, adaptation)

    def _build_coupling_matrix(self, cellular: float, population: float,
                               community: float, circadian: float,
                               environmental: float) -> np.ndarray:
        """
        Build 5x5 coupling matrix
        From microbiome paper Eq 41
        """
        C_matrix = np.zeros((5, 5))

        # Diagonal = 1.0 (self-coupling)
        np.fill_diagonal(C_matrix, 1.0)

        # Off-diagonal couplings
        # From microbiome paper Table 3
        C_matrix[0, 1] = C_matrix[1, 0] = 0.67 * cellular * population
        C_matrix[1, 2] = C_matrix[2, 1] = 0.72 * population * community
        C_matrix[2, 3] = C_matrix[3, 2] = 0.84 * community * circadian
        C_matrix[3, 4] = C_matrix[4, 3] = 0.59 * circadian * environmental

        return C_matrix

    def _calculate_dysbiosis_score(self, C_matrix: np.ndarray) -> float:
        """
        Calculate dysbiosis score
        From microbiome paper Eq 91
        """
        dysbiosis = 0.0

        # C_12: Cellular-population
        dysbiosis += self.weights['C_12'] * abs(1 - C_matrix[0,1] / self.healthy_reference['C_12'])

        # C_23: Population-community
        dysbiosis += self.weights['C_23'] * abs(1 - C_matrix[1,2] / self.healthy_reference['C_23'])

        # C_34: Community-host (most important)
        dysbiosis += self.weights['C_34'] * abs(1 - C_matrix[2,3] / self.healthy_reference['C_34'])

        # C_45: Host-environment
        dysbiosis += self.weights['C_45'] * abs(1 - C_matrix[3,4] / self.healthy_reference['C_45'])

        return dysbiosis

    def _analyze_scale_health(self, C_matrix: np.ndarray) -> Dict:
        """Analyze health of each temporal scale"""
        return {
            'cellular_metabolic': {
                'coupling': C_matrix[0, 1],
                'health': 'healthy' if C_matrix[0,1] > 0.6 else 'impaired'
            },
            'population_growth': {
                'coupling': C_matrix[1, 2],
                'health': 'healthy' if C_matrix[1,2] > 0.65 else 'impaired'
            },
            'community_dynamics': {
                'coupling': C_matrix[2, 3],
                'health': 'healthy' if C_matrix[2,3] > 0.75 else 'impaired'
            },
            'host_environment': {
                'coupling': C_matrix[3, 4],
                'health': 'healthy' if C_matrix[3,4] > 0.5 else 'impaired'
            }
        }
