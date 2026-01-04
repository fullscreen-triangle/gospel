# Genome File Parser for FASTA and VCF Integration
# Parses FASTA files and integrates with VCF data for comprehensive genomic analysis

import os
import gzip
import argparse
import json
from typing import List, Dict, Tuple, Optional, Iterator
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class GenomeParser:
    """
    Parse FASTA files and integrate with VCF data for St. Stella's sequence analysis
    """

    def __init__(self):
        self.sequences = {}
        self.vcf_variants = {}

    def parse_fasta(self, fasta_file: str, max_length: Optional[int] = None) -> Dict[str, str]:
        """
        Parse FASTA file and extract sequences

        Args:
            fasta_file: Path to FASTA file (.fa, .fasta, .fa.gz, .fasta.gz)
            max_length: Maximum sequence length to extract (for memory management)

        Returns:
            Dictionary mapping sequence ID to sequence string
        """

        sequences = {}
        current_id = None
        current_seq = []

        # Handle gzipped files
        if fasta_file.endswith('.gz'):
            file_opener = gzip.open
            mode = 'rt'
        else:
            file_opener = open
            mode = 'r'

        print(f"Parsing FASTA file: {fasta_file}")

        try:
            with file_opener(fasta_file, mode) as f:
                for line_num, line in enumerate(f):
                    line = line.strip()

                    if line.startswith('>'):
                        # Save previous sequence
                        if current_id is not None and current_seq:
                            seq_string = ''.join(current_seq)
                            if max_length:
                                seq_string = seq_string[:max_length]
                            sequences[current_id] = seq_string
                            print(f"  {current_id}: {len(seq_string):,} bp")

                        # Start new sequence
                        current_id = line[1:].split()[0]  # Take first part of header
                        current_seq = []

                    elif line and not line.startswith(';'):  # Skip comment lines
                        current_seq.append(line.upper())

                    # Memory management for large files
                    if line_num % 100000 == 0 and line_num > 0:
                        print(f"  Processed {line_num:,} lines...")

                # Don't forget the last sequence
                if current_id is not None and current_seq:
                    seq_string = ''.join(current_seq)
                    if max_length:
                        seq_string = seq_string[:max_length]
                    sequences[current_id] = seq_string
                    print(f"  {current_id}: {len(seq_string):,} bp")

        except Exception as e:
            print(f"Error parsing FASTA file: {e}")
            return {}

        print(f"Successfully parsed {len(sequences)} sequences")
        self.sequences.update(sequences)
        return sequences

    def extract_chromosome_regions(self, chromosome: str, start: int, end: int) -> str:
        """Extract specific chromosome region"""
        if chromosome in self.sequences:
            return self.sequences[chromosome][start:end]
        return ""

    def get_random_sequences(self, n_sequences: int = 100, min_length: int = 50,
                           max_length: int = 500) -> List[str]:
        """
        Extract random sequences for analysis

        Args:
            n_sequences: Number of sequences to extract
            min_length: Minimum sequence length
            max_length: Maximum sequence length

        Returns:
            List of random genomic sequences
        """

        random_sequences = []

        for seq_id, sequence in self.sequences.items():
            if len(sequence) < min_length:
                continue

            # Extract multiple random regions from this sequence
            sequences_from_this = min(n_sequences // len(self.sequences) + 1, 10)

            for _ in range(sequences_from_this):
                if len(random_sequences) >= n_sequences:
                    break

                # Random length and position
                seq_length = np.random.randint(min_length, min(max_length, len(sequence) - min_length))
                start_pos = np.random.randint(0, len(sequence) - seq_length)

                random_seq = sequence[start_pos:start_pos + seq_length]

                # Filter out sequences with too many N's
                if random_seq.count('N') / len(random_seq) < 0.1:  # Less than 10% N's
                    random_sequences.append(random_seq)

        return random_sequences[:n_sequences]

    def get_chromosome_sequences(self, chromosome: str, chunk_size: int = 1000,
                               n_chunks: int = 100) -> List[str]:
        """
        Get sequences from specific chromosome

        Args:
            chromosome: Chromosome identifier (e.g., '1', 'chr1', 'chromosome.1')
            chunk_size: Size of each sequence chunk
            n_chunks: Number of chunks to extract

        Returns:
            List of sequences from the chromosome
        """

        # Try different chromosome naming conventions
        possible_names = [
            chromosome,
            f"chr{chromosome}",
            f"chromosome.{chromosome}",
            f"{chromosome}",
        ]

        chr_sequence = None
        chr_name = None

        for name in possible_names:
            for seq_id in self.sequences:
                if name in seq_id.lower() or seq_id.lower() in name.lower():
                    chr_sequence = self.sequences[seq_id]
                    chr_name = seq_id
                    break
            if chr_sequence:
                break

        if not chr_sequence:
            print(f"Chromosome {chromosome} not found in sequences")
            return []

        print(f"Extracting {n_chunks} chunks from {chr_name} ({len(chr_sequence):,} bp)")

        sequences = []
        max_start = len(chr_sequence) - chunk_size

        if max_start <= 0:
            return [chr_sequence]  # Return whole sequence if it's smaller than chunk_size

        # Extract evenly spaced chunks
        for i in range(n_chunks):
            start_pos = (i * max_start) // (n_chunks - 1) if n_chunks > 1 else 0
            end_pos = start_pos + chunk_size

            chunk = chr_sequence[start_pos:end_pos]

            # Filter out chunks with too many N's
            if chunk.count('N') / len(chunk) < 0.2:  # Less than 20% N's
                sequences.append(chunk)

        return sequences

    def parse_vcf_variants(self, vcf_file: str, max_variants: int = 10000) -> Dict:
        """
        Parse VCF file to extract variant information

        Args:
            vcf_file: Path to VCF file
            max_variants: Maximum number of variants to process

        Returns:
            Dictionary with variant information
        """

        variants = {
            'snps': [],
            'indels': [],
            'cnvs': [],
            'metadata': {
                'total_variants': 0,
                'chromosomes': set(),
                'file': vcf_file
            }
        }

        # Handle gzipped VCF files
        if vcf_file.endswith('.gz'):
            file_opener = gzip.open
            mode = 'rt'
        else:
            file_opener = open
            mode = 'r'

        print(f"Parsing VCF file: {vcf_file}")

        try:
            with file_opener(vcf_file, mode) as f:
                for line_num, line in enumerate(f):
                    if line.startswith('#'):
                        continue  # Skip header lines

                    if variants['metadata']['total_variants'] >= max_variants:
                        break

                    fields = line.strip().split('\t')
                    if len(fields) < 8:
                        continue

                    chrom, pos, variant_id, ref, alt, qual, filter_field, info = fields[:8]

                    variant = {
                        'chromosome': chrom,
                        'position': int(pos),
                        'id': variant_id,
                        'ref': ref,
                        'alt': alt,
                        'quality': qual,
                        'filter': filter_field,
                        'info': info
                    }

                    # Classify variant type
                    if len(ref) == 1 and len(alt) == 1:
                        variants['snps'].append(variant)
                    elif len(ref) != len(alt):
                        variants['indels'].append(variant)
                    else:
                        variants['cnvs'].append(variant)

                    variants['metadata']['chromosomes'].add(chrom)
                    variants['metadata']['total_variants'] += 1

                    if line_num % 10000 == 0 and line_num > 0:
                        print(f"  Processed {line_num:,} lines, {variants['metadata']['total_variants']} variants...")

        except Exception as e:
            print(f"Error parsing VCF file: {e}")
            return variants

        variants['metadata']['chromosomes'] = list(variants['metadata']['chromosomes'])

        print(f"Parsed {variants['metadata']['total_variants']} variants:")
        print(f"  SNPs: {len(variants['snps'])}")
        print(f"  Indels: {len(variants['indels'])}")
        print(f"  CNVs: {len(variants['cnvs'])}")
        print(f"  Chromosomes: {', '.join(sorted(variants['metadata']['chromosomes']))}")

        self.vcf_variants = variants
        return variants

    def get_variant_context_sequences(self, window_size: int = 100) -> List[Tuple[str, Dict]]:
        """
        Get sequences around variants for analysis

        Args:
            window_size: Size of sequence window around each variant

        Returns:
            List of (sequence, variant_info) tuples
        """

        context_sequences = []

        for variant_type in ['snps', 'indels']:
            for variant in self.vcf_variants.get(variant_type, []):
                chrom = variant['chromosome']
                pos = variant['position']

                # Try to find matching chromosome in sequences
                chr_sequence = None
                for seq_id, sequence in self.sequences.items():
                    if chrom in seq_id.lower() or seq_id.lower().endswith(chrom):
                        chr_sequence = sequence
                        break

                if chr_sequence and pos - window_size > 0 and pos + window_size < len(chr_sequence):
                    # Extract sequence around variant
                    start_pos = pos - window_size
                    end_pos = pos + window_size
                    context_seq = chr_sequence[start_pos:end_pos]

                    context_sequences.append((context_seq, variant))

                if len(context_sequences) >= 1000:  # Limit for memory
                    break

        print(f"Extracted {len(context_sequences)} variant context sequences")
        return context_sequences

def main():
    """
    Standalone genome parser with publication-ready analysis
    """

    parser = argparse.ArgumentParser(description="Genome File Parser for FASTA and VCF Integration")
    parser.add_argument("--fasta", type=str, nargs='+',
                       help="FASTA file(s) to parse")
    parser.add_argument("--vcf", type=str, nargs='+',
                       help="VCF file(s) to parse")
    parser.add_argument("--chromosome", type=str, default="21",
                       help="Chromosome to focus analysis on")
    parser.add_argument("--max-seq-length", type=int, default=10000000,
                       help="Maximum sequence length to load (for memory management)")
    parser.add_argument("--n-sequences", type=int, default=200,
                       help="Number of random sequences to extract for analysis")
    parser.add_argument("--output", type=str, default="./genome_parser_results/",
                       help="Output directory for results")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Generate visualizations")
    args = parser.parse_args()

    # Default to files in public directory if none specified
    if not args.fasta and not args.vcf:
        public_dir = Path(__file__).parent.parent.parent.parent / "public"

        # Auto-detect FASTA files
        fasta_patterns = ["*.fa", "*.fasta", "*.fa.gz", "*.fasta.gz"]
        fasta_files = []
        for pattern in fasta_patterns:
            fasta_files.extend(list(public_dir.glob(f"fasta/{pattern}")))
            fasta_files.extend(list(public_dir.glob(pattern)))

        # Auto-detect VCF files
        vcf_files = list(public_dir.glob("*.vcf")) + list(public_dir.glob("*.vcf.gz"))

        if fasta_files:
            args.fasta = [str(f) for f in fasta_files[:3]]  # Limit to first 3
        if vcf_files:
            args.vcf = [str(f) for f in vcf_files if 'snp' in str(f)][:2]  # Focus on SNP files

        print("Auto-detected files:")
        print(f"  FASTA: {args.fasta}")
        print(f"  VCF: {args.vcf}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("="*80)
    print("GENOME FILE PARSER & SEQUENCE EXTRACTION")
    print("="*80)
    print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output}")

    # Initialize parser
    genome_parser = GenomeParser()

    # Parse FASTA files
    if args.fasta:
        print(f"\n[1/4] PARSING FASTA FILES")
        print("-"*50)

        for fasta_file in args.fasta:
            if os.path.exists(fasta_file):
                sequences = genome_parser.parse_fasta(fasta_file, args.max_seq_length)
                print(f"  Loaded {len(sequences)} sequences from {fasta_file}")

    # Parse VCF files
    if args.vcf:
        print(f"\n[2/4] PARSING VCF FILES")
        print("-"*50)

        all_variants = {'snps': [], 'indels': [], 'cnvs': []}
        for vcf_file in args.vcf:
            if os.path.exists(vcf_file):
                variants = genome_parser.parse_vcf_variants(vcf_file)
                for vtype in all_variants:
                    all_variants[vtype].extend(variants.get(vtype, []))

    # Extract sequences for analysis
    print(f"\n[3/4] EXTRACTING SEQUENCES FOR ANALYSIS")
    print("-"*50)

    analysis_sequences = {
        'random_sequences': [],
        'chromosome_sequences': [],
        'variant_context_sequences': []
    }

    # Get random sequences
    if genome_parser.sequences:
        random_seqs = genome_parser.get_random_sequences(args.n_sequences)
        analysis_sequences['random_sequences'] = random_seqs
        print(f"  Extracted {len(random_seqs)} random sequences")

        # Get chromosome-specific sequences
        chr_seqs = genome_parser.get_chromosome_sequences(args.chromosome, chunk_size=1000, n_chunks=50)
        analysis_sequences['chromosome_sequences'] = chr_seqs
        print(f"  Extracted {len(chr_seqs)} sequences from chromosome {args.chromosome}")

    # Get variant context sequences
    if genome_parser.vcf_variants:
        variant_contexts = genome_parser.get_variant_context_sequences(window_size=100)
        analysis_sequences['variant_context_sequences'] = [(seq, var) for seq, var in variant_contexts]
        print(f"  Extracted {len(variant_contexts)} variant context sequences")

    # Save results
    results_data = {
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'fasta_files': args.fasta if args.fasta else [],
            'vcf_files': args.vcf if args.vcf else [],
            'sequences_extracted': sum(len(seqs) for seqs in analysis_sequences.values() if isinstance(seqs, list)),
            'framework_version': '1.0.0',
            'analysis_type': 'genome_file_parsing'
        },
        'sequence_statistics': {
            'total_genomic_sequences': len(genome_parser.sequences),
            'sequence_lengths': {seq_id: len(seq) for seq_id, seq in genome_parser.sequences.items()},
            'random_sequences_count': len(analysis_sequences['random_sequences']),
            'chromosome_sequences_count': len(analysis_sequences['chromosome_sequences']),
            'variant_contexts_count': len(analysis_sequences['variant_context_sequences'])
        },
        'variant_statistics': genome_parser.vcf_variants.get('metadata', {}),
        'extracted_sequences': {
            'random_sample': analysis_sequences['random_sequences'][:10],  # Sample for JSON
            'chromosome_sample': analysis_sequences['chromosome_sequences'][:5],
        }
    }

    with open(f"{args.output}/genome_parser_results.json", 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    # Generate visualizations
    if args.visualize:
        print(f"\n[4/4] GENERATING VISUALIZATIONS")
        print("-"*50)
        _generate_genome_parser_visualizations(genome_parser, analysis_sequences, args.output)

    # Save sequences for downstream analysis
    _save_sequences_for_analysis(analysis_sequences, args.output)

    print(f"\n✅ Analysis complete! Results saved to: {args.output}")
    print("\n📊 Generated files:")
    print(f"  • genome_parser_results.json - Complete parsing results")
    print(f"  • genome_statistics.png - Genome composition analysis")
    print(f"  • sequence_analysis.png - Sequence length distributions")
    print(f"  • variant_analysis.png - Variant distribution analysis")
    print(f"  • extracted_sequences.fasta - Sequences ready for St. Stella's analysis")

    return results_data

def _generate_genome_parser_visualizations(genome_parser: GenomeParser, sequences: Dict, output_dir: str):
    """Generate publication-ready visualizations"""

    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300
    })

    # 1. Genome Statistics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Genome Parsing Statistics', fontsize=16, fontweight='bold')

    # Sequence length distribution
    if genome_parser.sequences:
        seq_lengths = [len(seq) for seq in genome_parser.sequences.values()]
        seq_names = list(genome_parser.sequences.keys())

        ax1.bar(range(len(seq_names)), seq_lengths, alpha=0.7, color='blue')
        ax1.set_xlabel('Sequence Index')
        ax1.set_ylabel('Sequence Length (bp)')
        ax1.set_title('A. Genomic Sequence Lengths')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # Add labels for significant sequences
        for i, (name, length) in enumerate(zip(seq_names, seq_lengths)):
            if length > np.median(seq_lengths) * 2:  # Label large sequences
                ax1.text(i, length, name.split('.')[0], rotation=90, ha='center', va='bottom', fontsize=8)

    # Base composition analysis
    if sequences['random_sequences']:
        base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0}
        total_bases = 0

        for seq in sequences['random_sequences'][:50]:  # Sample for speed
            for base in seq:
                if base in base_counts:
                    base_counts[base] += 1
                    total_bases += 1

        if total_bases > 0:
            base_percentages = {base: (count/total_bases)*100 for base, count in base_counts.items()}

            ax2.pie(base_percentages.values(), labels=base_percentages.keys(), autopct='%1.1f%%')
            ax2.set_title('B. Genomic Base Composition')

    # Variant distribution
    if genome_parser.vcf_variants and genome_parser.vcf_variants.get('metadata', {}).get('total_variants', 0) > 0:
        variant_counts = [
            len(genome_parser.vcf_variants.get('snps', [])),
            len(genome_parser.vcf_variants.get('indels', [])),
            len(genome_parser.vcf_variants.get('cnvs', []))
        ]
        variant_types = ['SNPs', 'Indels', 'CNVs']

        bars = ax3.bar(variant_types, variant_counts, alpha=0.7, color=['red', 'orange', 'purple'])
        ax3.set_ylabel('Number of Variants')
        ax3.set_title('C. Variant Type Distribution')
        ax3.grid(True, alpha=0.3)

        # Add count labels on bars
        for bar, count in zip(bars, variant_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom')

    # Sequence extraction summary
    extraction_counts = [
        len(sequences['random_sequences']),
        len(sequences['chromosome_sequences']),
        len(sequences['variant_context_sequences'])
    ]
    extraction_types = ['Random', 'Chromosome', 'Variant Context']

    ax4.bar(extraction_types, extraction_counts, alpha=0.7, color='green')
    ax4.set_ylabel('Number of Sequences')
    ax4.set_title('D. Extracted Sequences for Analysis')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/genome_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Sequence Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Extracted Sequence Analysis', fontsize=16, fontweight='bold')

    # Random sequence length distribution
    if sequences['random_sequences']:
        lengths = [len(seq) for seq in sequences['random_sequences']]
        ax1.hist(lengths, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Sequence Length (bp)')
        ax1.set_ylabel('Number of Sequences')
        ax1.set_title('A. Random Sequence Length Distribution')
        ax1.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.0f} bp')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # GC content distribution
    if sequences['random_sequences']:
        gc_contents = []
        for seq in sequences['random_sequences']:
            gc_count = seq.count('G') + seq.count('C')
            gc_content = (gc_count / len(seq)) * 100 if len(seq) > 0 else 0
            gc_contents.append(gc_content)

        ax2.hist(gc_contents, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('GC Content (%)')
        ax2.set_ylabel('Number of Sequences')
        ax2.set_title('B. GC Content Distribution')
        ax2.axvline(np.mean(gc_contents), color='red', linestyle='--',
                   label=f'Mean: {np.mean(gc_contents):.1f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Chromosome sequences analysis
    if sequences['chromosome_sequences']:
        chr_lengths = [len(seq) for seq in sequences['chromosome_sequences']]
        ax3.plot(range(len(chr_lengths)), chr_lengths, 'o-', alpha=0.7, color='purple')
        ax3.set_xlabel('Sequence Index')
        ax3.set_ylabel('Sequence Length (bp)')
        ax3.set_title('C. Chromosome Sequence Lengths')
        ax3.grid(True, alpha=0.3)

    # Variant context analysis
    if sequences['variant_context_sequences']:
        variant_types = []
        for seq, variant in sequences['variant_context_sequences']:
            if len(variant['ref']) == 1 and len(variant['alt']) == 1:
                variant_types.append('SNP')
            elif len(variant['ref']) != len(variant['alt']):
                variant_types.append('Indel')
            else:
                variant_types.append('Other')

        type_counts = {vtype: variant_types.count(vtype) for vtype in set(variant_types)}
        ax4.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('D. Variant Context Types')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sequence_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Visualizations saved to {output_dir}")

def _save_sequences_for_analysis(sequences: Dict, output_dir: str):
    """Save extracted sequences in FASTA format for downstream analysis"""

    fasta_file = f"{output_dir}/extracted_sequences.fasta"

    with open(fasta_file, 'w') as f:
        # Save random sequences
        for i, seq in enumerate(sequences['random_sequences']):
            f.write(f">random_seq_{i+1} length={len(seq)}\n")
            f.write(f"{seq}\n")

        # Save chromosome sequences
        for i, seq in enumerate(sequences['chromosome_sequences']):
            f.write(f">chromosome_seq_{i+1} length={len(seq)}\n")
            f.write(f"{seq}\n")

        # Save variant context sequences (first 100)
        for i, (seq, variant) in enumerate(sequences['variant_context_sequences'][:100]):
            var_id = variant.get('id', f"var_{i+1}")
            chrom = variant.get('chromosome', 'unknown')
            pos = variant.get('position', 0)
            f.write(f">variant_context_{i+1} {var_id} {chrom}:{pos} length={len(seq)}\n")
            f.write(f"{seq}\n")

    print(f"  ✓ Sequences saved to: {fasta_file}")

    # Also save a sample for quick testing
    sample_file = f"{output_dir}/sample_sequences.txt"
    with open(sample_file, 'w') as f:
        f.write("# Sample sequences for St. Stella's analysis\n")
        f.write("# Format: one sequence per line\n\n")

        sample_seqs = sequences['random_sequences'][:20]  # First 20 random sequences
        for seq in sample_seqs:
            f.write(f"{seq}\n")

    print(f"  ✓ Sample sequences saved to: {sample_file}")

if __name__ == "__main__":
    main()
