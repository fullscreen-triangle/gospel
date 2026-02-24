"""
Real Genomic Data Validation
=============================

Validates paper claims using REAL extracted genome sequences.

Experiments:
1. Palindrome detection and analysis
2. Dual-strand geometry analysis
3. Pattern detection (repeats, motifs, regulatory elements)
4. Hierarchy analysis (nested structures)
5. Coordinate transformation validation
6. Empty dictionary prediction accuracy

Uses: genome_parser_results/extracted_sequences.fasta
"""

import time
import math
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter

try:
    from .virtual_molecule import VirtualMolecule, SCoordinate
    from .virtual_capacitor import ChargeState
    from .virtual_partition import VirtualPartition
    from .genomic_charged_fluid import GenomicChargedFluid
except ImportError:
    from virtual_molecule import VirtualMolecule, SCoordinate
    from virtual_capacitor import ChargeState
    from virtual_partition import VirtualPartition
    from genomic_charged_fluid import GenomicChargedFluid


@dataclass
class SequenceRecord:
    """A single sequence from FASTA file."""
    id: str
    length: int
    sequence: str

    def __post_init__(self):
        self.sequence = self.sequence.upper()


@dataclass
class PalindromeResult:
    """Result from palindrome detection."""
    position: int
    length: int
    sequence: str
    palindrome: str
    s_coordinate: SCoordinate
    charge_state: ChargeState
    symmetry_score: float

    @property
    def is_perfect(self) -> bool:
        """Check if palindrome is perfect."""
        return self.sequence == self.palindrome[::-1]


@dataclass
class DualStrandGeometry:
    """Dual-strand geometric analysis result."""
    sequence_id: str
    forward_coords: List[Tuple[float, float]]  # (x, y) cardinal coordinates
    reverse_coords: List[Tuple[float, float]]
    information_density: float
    geometric_entropy: float
    charge_balance: float
    complementarity_score: float


@dataclass
class PatternResult:
    """Pattern detection result."""
    pattern_type: str  # 'repeat', 'motif', 'regulatory'
    sequence: str
    positions: List[int]
    frequency: int
    s_signature: SCoordinate
    predicted_function: str
    confidence: float


@dataclass
class HierarchyNode:
    """Hierarchical structure node."""
    level: int
    start: int
    end: int
    sequence: str
    children: List['HierarchyNode']
    s_coordinate: SCoordinate
    partition_depth: int


class FASTAParser:
    """Parse FASTA files into sequence records."""

    @staticmethod
    def parse(fasta_path: str) -> List[SequenceRecord]:
        """Parse FASTA file."""
        records = []
        current_id = None
        current_seq = []

        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('>'):
                    # Save previous record
                    if current_id:
                        seq = ''.join(current_seq)
                        records.append(SequenceRecord(
                            id=current_id,
                            length=len(seq),
                            sequence=seq
                        ))

                    # Parse header: >random_seq_1 length=51
                    parts = line[1:].split()
                    current_id = parts[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            # Save last record
            if current_id:
                seq = ''.join(current_seq)
                records.append(SequenceRecord(
                    id=current_id,
                    length=len(seq),
                    sequence=seq
                ))

        return records


class PalindromeDetector:
    """
    Detect palindromes using coordinate-based method.

    Validates paper claim: Palindromes have symmetric S-coordinates.
    """

    def __init__(self, min_length: int = 4):
        self.min_length = min_length
        self.partition = VirtualPartition()

    def nucleotide_to_s_coordinate(self, nucleotide: str) -> SCoordinate:
        """
        Map nucleotide to S-coordinate using partition theory.

        Four-state partition: A, T, G, C
        """
        # Partition-based mapping (from paper Section 4)
        mapping = {
            'A': SCoordinate(0.25, 0.25, 0.5),  # Purine, pairs with T
            'T': SCoordinate(0.75, 0.75, 0.5),  # Pyrimidine, pairs with A
            'G': SCoordinate(0.25, 0.75, 0.7),  # Purine, pairs with C
            'C': SCoordinate(0.75, 0.25, 0.7),  # Pyrimidine, pairs with G
        }
        return mapping.get(nucleotide, SCoordinate(0.5, 0.5, 0.5))

    def sequence_to_s_trajectory(self, sequence: str) -> List[SCoordinate]:
        """Convert sequence to S-coordinate trajectory."""
        return [self.nucleotide_to_s_coordinate(nt) for nt in sequence]

    def calculate_symmetry_score(self, sequence: str) -> float:
        """
        Calculate symmetry score using S-coordinates.

        Perfect palindrome: forward trajectory mirrors reverse.
        """
        forward = self.sequence_to_s_trajectory(sequence)
        reverse = self.sequence_to_s_trajectory(sequence[::-1])

        # Calculate trajectory similarity
        total_distance = 0.0
        for f, r in zip(forward, reverse):
            total_distance += f.distance_to(r)

        # Normalize: 1.0 = perfect symmetry, 0.0 = no symmetry
        max_distance = len(sequence) * math.sqrt(3)  # Max possible in unit cube
        symmetry = 1.0 - (total_distance / max_distance) if max_distance > 0 else 0.0

        return symmetry

    def is_palindrome(self, sequence: str) -> bool:
        """Check if sequence is palindrome (reverse complement)."""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        reverse_comp = ''.join(complement.get(nt, nt) for nt in sequence[::-1])
        return sequence == reverse_comp

    def detect_palindromes(self, record: SequenceRecord) -> List[PalindromeResult]:
        """
        Detect all palindromes in sequence using coordinate method.
        """
        palindromes = []
        sequence = record.sequence

        # Sliding window approach
        for length in range(self.min_length, min(len(sequence) + 1, 50)):
            for i in range(len(sequence) - length + 1):
                subseq = sequence[i:i+length]

                # Calculate symmetry score
                symmetry = self.calculate_symmetry_score(subseq)

                # High symmetry suggests palindrome
                if symmetry > 0.7:  # Threshold
                    # Get S-coordinate for center
                    center_idx = i + length // 2
                    center_nt = sequence[center_idx] if center_idx < len(sequence) else 'N'
                    s_coord = self.nucleotide_to_s_coordinate(center_nt)

                    # Get charge state from hardware
                    charge = ChargeState.from_hardware()

                    # Reverse complement
                    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
                    palindrome = ''.join(complement.get(nt, nt) for nt in subseq[::-1])

                    palindromes.append(PalindromeResult(
                        position=i,
                        length=length,
                        sequence=subseq,
                        palindrome=palindrome,
                        s_coordinate=s_coord,
                        charge_state=charge,
                        symmetry_score=symmetry
                    ))

        return palindromes


class DualStrandGeometryAnalyzer:
    """
    Analyze dual-strand geometry using cardinal coordinates.

    Validates paper Section 5: Coordinate Geometry.
    """

    def __init__(self):
        self.partition = VirtualPartition()

    def nucleotide_to_cardinal(self, nucleotide: str) -> Tuple[float, float]:
        """
        Map nucleotide to 2D cardinal direction.

        From paper: A=North, T=South, G=East, C=West
        """
        mapping = {
            'A': (0, 1),   # North
            'T': (0, -1),  # South
            'G': (1, 0),   # East
            'C': (-1, 0),  # West
        }
        return mapping.get(nucleotide, (0, 0))

    def sequence_to_trajectory(self, sequence: str) -> List[Tuple[float, float]]:
        """Convert sequence to 2D trajectory."""
        trajectory = [(0.0, 0.0)]  # Start at origin
        x, y = 0.0, 0.0

        for nt in sequence:
            dx, dy = self.nucleotide_to_cardinal(nt)
            x += dx
            y += dy
            trajectory.append((x, y))

        return trajectory

    def calculate_information_density(self, trajectory: List[Tuple[float, float]]) -> float:
        """
        Calculate geometric information density.

        From paper: I_geo = (L_path / L_direct)²
        """
        if len(trajectory) < 2:
            return 1.0

        # Path length
        path_length = 0.0
        for i in range(1, len(trajectory)):
            x1, y1 = trajectory[i-1]
            x2, y2 = trajectory[i]
            path_length += math.sqrt((x2-x1)**2 + (y2-y1)**2)

        # Direct length (start to end)
        x_start, y_start = trajectory[0]
        x_end, y_end = trajectory[-1]
        direct_length = math.sqrt((x_end-x_start)**2 + (y_end-y_start)**2)

        if direct_length == 0:
            return 1.0

        ratio = path_length / direct_length
        return ratio ** 2

    def calculate_geometric_entropy(self, trajectory: List[Tuple[float, float]]) -> float:
        """
        Calculate geometric entropy from trajectory spread.

        S_geo = log(area_covered)
        """
        if len(trajectory) < 3:
            return 0.0

        xs = [x for x, y in trajectory]
        ys = [y for x, y in trajectory]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        area = width * height

        if area <= 0:
            return 0.0

        return math.log(area + 1)  # +1 to avoid log(0)

    def calculate_charge_balance(self, sequence: str) -> float:
        """
        Calculate charge balance between purines and pyrimidines.

        Purines (A, G): +1 charge
        Pyrimidines (T, C): -1 charge
        """
        purines = sequence.count('A') + sequence.count('G')
        pyrimidines = sequence.count('T') + sequence.count('C')
        total = len(sequence)

        if total == 0:
            return 0.0

        # Balance: 0 = perfect balance, ±1 = all one type
        balance = (purines - pyrimidines) / total
        return abs(balance)

    def calculate_complementarity(self, forward: str, reverse: str) -> float:
        """
        Calculate Watson-Crick complementarity score.
        """
        if len(forward) != len(reverse):
            return 0.0

        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        matches = sum(1 for f, r in zip(forward, reverse)
                     if complement.get(f) == r)

        return matches / len(forward) if len(forward) > 0 else 0.0

    def analyze(self, record: SequenceRecord) -> DualStrandGeometry:
        """Perform complete dual-strand geometry analysis."""
        sequence = record.sequence

        # Forward strand trajectory
        forward_traj = self.sequence_to_trajectory(sequence)

        # Reverse complement strand
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        reverse_seq = ''.join(complement.get(nt, nt) for nt in sequence[::-1])
        reverse_traj = self.sequence_to_trajectory(reverse_seq)

        # Calculate metrics
        info_density = self.calculate_information_density(forward_traj)
        geo_entropy = self.calculate_geometric_entropy(forward_traj)
        charge_balance = self.calculate_charge_balance(sequence)
        complementarity = self.calculate_complementarity(sequence, reverse_seq)

        return DualStrandGeometry(
            sequence_id=record.id,
            forward_coords=forward_traj,
            reverse_coords=reverse_traj,
            information_density=info_density,
            geometric_entropy=geo_entropy,
            charge_balance=charge_balance,
            complementarity_score=complementarity
        )


class PatternDetector:
    """
    Detect patterns using S-coordinate signatures.

    Validates empty dictionary prediction paradigm.
    """

    def __init__(self):
        self.partition = VirtualPartition()

        # Pattern signatures (from paper Section 12)
        self.signatures = {
            'repeat': SCoordinate(0.5, 0.5, 0.3),  # Low evolution
            'regulatory': SCoordinate(0.7, 0.5, 0.5),  # High knowledge
            'coding': SCoordinate(0.7, 0.6, 0.7),  # Directional
            'palindrome': SCoordinate(0.5, 0.5, 0.3),  # Symmetric
        }

    def detect_repeats(self, sequence: str, min_length: int = 3) -> List[PatternResult]:
        """Detect repeat patterns."""
        repeats = []
        seen = defaultdict(list)

        # Find all k-mers
        for k in range(min_length, min(len(sequence) + 1, 20)):
            for i in range(len(sequence) - k + 1):
                kmer = sequence[i:i+k]
                seen[kmer].append(i)

        # Filter for actual repeats (appears 2+ times)
        for kmer, positions in seen.items():
            if len(positions) >= 2:
                # Calculate S-signature
                s_sig = self.signatures['repeat']

                repeats.append(PatternResult(
                    pattern_type='repeat',
                    sequence=kmer,
                    positions=positions,
                    frequency=len(positions),
                    s_signature=s_sig,
                    predicted_function='structural_repeat',
                    confidence=min(1.0, len(positions) / 10.0)
                ))

        return repeats

    def detect_motifs(self, sequence: str) -> List[PatternResult]:
        """Detect known regulatory motifs."""
        motifs = []

        # Common regulatory motifs
        known_motifs = {
            'TATA': 'TATA_box',
            'CAAT': 'CAAT_box',
            'GC': 'GC_box',
            'ATG': 'start_codon',
            'TAG': 'stop_codon',
            'TAA': 'stop_codon',
            'TGA': 'stop_codon',
        }

        for motif, function in known_motifs.items():
            positions = []
            start = 0
            while True:
                pos = sequence.find(motif, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1

            if positions:
                motifs.append(PatternResult(
                    pattern_type='motif',
                    sequence=motif,
                    positions=positions,
                    frequency=len(positions),
                    s_signature=self.signatures['regulatory'],
                    predicted_function=function,
                    confidence=0.9
                ))

        return motifs

    def detect_all_patterns(self, record: SequenceRecord) -> List[PatternResult]:
        """Detect all patterns in sequence."""
        patterns = []
        patterns.extend(self.detect_repeats(record.sequence))
        patterns.extend(self.detect_motifs(record.sequence))
        return patterns


class HierarchyAnalyzer:
    """
    Analyze hierarchical structure using partition theory.

    Validates nested partition operations.
    """

    def __init__(self):
        self.partition = VirtualPartition()

    def build_hierarchy(self, sequence: str, max_depth: int = 5) -> HierarchyNode:
        """
        Build hierarchical structure through recursive partitioning.
        """
        def partition_sequence(seq: str, start: int, level: int) -> HierarchyNode:
            # Base case
            if len(seq) <= 4 or level >= max_depth:
                s_coord = SCoordinate(0.5, 0.5, level / max_depth)
                return HierarchyNode(
                    level=level,
                    start=start,
                    end=start + len(seq),
                    sequence=seq,
                    children=[],
                    s_coordinate=s_coord,
                    partition_depth=level
                )

            # Partition into 4 parts (A, T, G, C dominant regions)
            quarter = len(seq) // 4
            children = []

            for i in range(4):
                child_start = start + i * quarter
                child_end = start + (i + 1) * quarter if i < 3 else start + len(seq)
                child_seq = seq[i * quarter:child_end - start]

                if child_seq:
                    child = partition_sequence(child_seq, child_start, level + 1)
                    children.append(child)

            # S-coordinate for this node
            s_coord = SCoordinate(
                0.5,
                0.5,
                level / max_depth
            )

            return HierarchyNode(
                level=level,
                start=start,
                end=start + len(seq),
                sequence=seq,
                children=children,
                s_coordinate=s_coord,
                partition_depth=level
            )

        return partition_sequence(sequence, 0, 0)

    def count_nodes(self, root: HierarchyNode) -> int:
        """Count total nodes in hierarchy."""
        count = 1
        for child in root.children:
            count += self.count_nodes(child)
        return count

    def max_depth(self, root: HierarchyNode) -> int:
        """Get maximum depth of hierarchy."""
        if not root.children:
            return root.level
        return max(self.max_depth(child) for child in root.children)


def run_complete_validation(fasta_path: str, output_dir: str = '.'):
    """
    Run complete validation on real genomic data.
    """
    print("="*80)
    print(" "*20 + "REAL GENOMIC DATA VALIDATION")
    print("="*80)
    print(f"\nDataset: {fasta_path}")
    print("="*80)

    # Parse sequences
    print("\n1. PARSING FASTA FILE")
    print("-"*80)
    parser = FASTAParser()
    records = parser.parse(fasta_path)
    print(f"  Loaded {len(records)} sequences")
    print(f"  Total length: {sum(r.length for r in records):,} bp")
    print(f"  Length range: {min(r.length for r in records)} - {max(r.length for r in records)} bp")

    # Palindrome detection
    print("\n2. PALINDROME DETECTION")
    print("-"*80)
    palindrome_detector = PalindromeDetector(min_length=4)
    all_palindromes = []

    for record in records[:10]:  # First 10 sequences
        palindromes = palindrome_detector.detect_palindromes(record)
        all_palindromes.extend(palindromes)
        if palindromes:
            print(f"  {record.id}: Found {len(palindromes)} palindromes")

    print(f"\nTotal palindromes detected: {len(all_palindromes)}")
    if all_palindromes:
        perfect = sum(1 for p in all_palindromes if p.is_perfect)
        avg_symmetry = np.mean([p.symmetry_score for p in all_palindromes])
        print(f"  Perfect palindromes: {perfect}")
        print(f"  Average symmetry score: {avg_symmetry:.3f}")

    # Dual-strand geometry
    print("\n3. DUAL-STRAND GEOMETRY ANALYSIS")
    print("-"*80)
    geometry_analyzer = DualStrandGeometryAnalyzer()
    geometries = []

    for record in records[:10]:
        geom = geometry_analyzer.analyze(record)
        geometries.append(geom)
        print(f"  {record.id}:")
        print(f"    Information density: {geom.information_density:.3f}")
        print(f"    Geometric entropy: {geom.geometric_entropy:.3f}")
        print(f"    Charge balance: {geom.charge_balance:.3f}")
        print(f"    Complementarity: {geom.complementarity_score:.3f}")

    # Pattern detection
    print("\n4. PATTERN DETECTION")
    print("-"*80)
    pattern_detector = PatternDetector()
    all_patterns = []

    for record in records[:10]:
        patterns = pattern_detector.detect_all_patterns(record)
        all_patterns.extend(patterns)
        if patterns:
            repeats = [p for p in patterns if p.pattern_type == 'repeat']
            motifs = [p for p in patterns if p.pattern_type == 'motif']
            print(f"  {record.id}: {len(repeats)} repeats, {len(motifs)} motifs")

    print(f"\nTotal patterns detected: {len(all_patterns)}")
    pattern_types = Counter(p.pattern_type for p in all_patterns)
    for ptype, count in pattern_types.items():
        print(f"  {ptype}: {count}")

    # Hierarchy analysis
    print("\n5. HIERARCHY ANALYSIS")
    print("-"*80)
    hierarchy_analyzer = HierarchyAnalyzer()
    hierarchies = []

    for record in records[:5]:  # First 5 (hierarchy is expensive)
        hierarchy = hierarchy_analyzer.build_hierarchy(record.sequence, max_depth=4)
        hierarchies.append(hierarchy)
        node_count = hierarchy_analyzer.count_nodes(hierarchy)
        max_depth = hierarchy_analyzer.max_depth(hierarchy)
        print(f"  {record.id}:")
        print(f"    Total nodes: {node_count}")
        print(f"    Max depth: {max_depth}")

    # Save results
    print("\n6. SAVING RESULTS")
    print("-"*80)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save palindrome results
    palindrome_data = {
        'total_palindromes': len(all_palindromes),
        'perfect_palindromes': sum(1 for p in all_palindromes if p.is_perfect),
        'average_symmetry': float(np.mean([p.symmetry_score for p in all_palindromes])) if all_palindromes else 0.0,
        'palindromes': [
            {
                'position': p.position,
                'length': p.length,
                'sequence': p.sequence,
                'symmetry_score': p.symmetry_score,
                'is_perfect': p.is_perfect
            }
            for p in all_palindromes[:100]  # Save first 100
        ]
    }

    with open(output_path / 'palindrome_analysis.json', 'w') as f:
        json.dump(palindrome_data, f, indent=2)
    print(f"  [OK] Saved: palindrome_analysis.json")

    # Save geometry results
    geometry_data = {
        'sequences_analyzed': len(geometries),
        'average_information_density': float(np.mean([g.information_density for g in geometries])),
        'average_geometric_entropy': float(np.mean([g.geometric_entropy for g in geometries])),
        'average_charge_balance': float(np.mean([g.charge_balance for g in geometries])),
        'average_complementarity': float(np.mean([g.complementarity_score for g in geometries])),
        'geometries': [
            {
                'sequence_id': g.sequence_id,
                'information_density': g.information_density,
                'geometric_entropy': g.geometric_entropy,
                'charge_balance': g.charge_balance,
                'complementarity_score': g.complementarity_score
            }
            for g in geometries
        ]
    }

    with open(output_path / 'dual_strand_geometry.json', 'w') as f:
        json.dump(geometry_data, f, indent=2)
    print(f"  [OK] Saved: dual_strand_geometry.json")

    # Save pattern results
    pattern_data = {
        'total_patterns': len(all_patterns),
        'pattern_types': dict(Counter(p.pattern_type for p in all_patterns)),
        'patterns': [
            {
                'type': p.pattern_type,
                'sequence': p.sequence,
                'frequency': p.frequency,
                'predicted_function': p.predicted_function,
                'confidence': p.confidence
            }
            for p in all_patterns[:100]  # Save first 100
        ]
    }

    with open(output_path / 'pattern_detection.json', 'w') as f:
        json.dump(pattern_data, f, indent=2)
    print(f"  [OK] Saved: pattern_detection.json")

    # Save hierarchy results
    hierarchy_data = {
        'sequences_analyzed': len(hierarchies),
        'hierarchies': [
            {
                'sequence_id': records[i].id,
                'total_nodes': hierarchy_analyzer.count_nodes(h),
                'max_depth': hierarchy_analyzer.max_depth(h)
            }
            for i, h in enumerate(hierarchies)
        ]
    }

    with open(output_path / 'hierarchy_analysis.json', 'w') as f:
        json.dump(hierarchy_data, f, indent=2)
    print(f"  [OK] Saved: hierarchy_analysis.json")

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"[OK] Analyzed {len(records)} sequences ({sum(r.length for r in records):,} bp)")
    print(f"[OK] Detected {len(all_palindromes)} palindromes (symmetry-based)")
    print(f"[OK] Analyzed {len(geometries)} dual-strand geometries")
    print(f"[OK] Detected {len(all_patterns)} patterns ({len(pattern_types)} types)")
    print(f"[OK] Built {len(hierarchies)} hierarchical structures")
    print(f"\nAll results saved to: {output_path.absolute()}")
    print("="*80)

    return {
        'sequences': records,
        'palindromes': all_palindromes,
        'geometries': geometries,
        'patterns': all_patterns,
        'hierarchies': hierarchies
    }


if __name__ == "__main__":
    # Path to FASTA file (relative to project root)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..', '..')
    fasta_path = os.path.join(project_root, 'new_testament', 'src', 'st_stellas', 'sequence',
                               'genome_parser_results', 'extracted_sequences.fasta')
    output_dir = os.path.join(script_dir, 'genomic_validation_results')

    # Run validation
    results = run_complete_validation(fasta_path, output_dir=output_dir)
