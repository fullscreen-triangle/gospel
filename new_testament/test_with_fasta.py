#!/usr/bin/env python3
"""
Test St. Stella's Sequence Modules with FASTA Files
Simple script to test the framework with your downloaded FASTA files

Usage:
    python test_with_fasta.py
    python test_with_fasta.py --chromosome 21 --n-sequences 50
"""

import os
import sys
from pathlib import Path

def main():
    """Test sequence analysis with FASTA files"""

    import argparse
    parser = argparse.ArgumentParser(description="Test St. Stella's modules with FASTA files")
    parser.add_argument("--chromosome", type=str, default="21", help="Chromosome to focus on")
    parser.add_argument("--n-sequences", type=int, default=100, help="Number of sequences to analyze")
    parser.add_argument("--output", type=str, default="./fasta_test_results/", help="Output directory")
    args = parser.parse_args()

    print("🧬 Testing St. Stella's Sequence Analysis with FASTA Files")
    print("="*70)

    # Check for FASTA files
    public_dir = Path("public")
    if not public_dir.exists():
        public_dir = Path("new_testament/public")

    print(f"Looking for FASTA files in: {public_dir}")

    if not public_dir.exists():
        print("❌ Public directory not found!")
        print("Expected: new_testament/public/")
        return

    # Find FASTA files
    fasta_files = list(public_dir.glob("fasta/*.fa")) + list(public_dir.glob("fasta/*.fa.gz"))
    if not fasta_files:
        fasta_files = list(public_dir.glob("*.fa")) + list(public_dir.glob("*.fa.gz"))

    print(f"Found {len(fasta_files)} FASTA files:")
    for i, f in enumerate(fasta_files, 1):
        print(f"  {i}. {f.name}")

    if not fasta_files:
        print("❌ No FASTA files found!")
        print("Expected files like: Homo_sapiens.GRCh38.dna.chromosome.*.fa")
        return

    # Check for VCF files
    vcf_files = list(public_dir.glob("*.vcf")) + list(public_dir.glob("*.vcf.gz"))
    print(f"\nFound {len(vcf_files)} VCF files:")
    for i, f in enumerate(vcf_files, 1):
        print(f"  {i}. {f.name}")

    print("\n" + "="*70)
    print("RUNNING TESTS")
    print("="*70)

    # Test 1: Parse Genome
    print("\n1️⃣ Testing Genome Parser...")
    try:
        # Add source directory to path
        src_dir = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_dir))

        # Run genome parser
        from st_stellas.sequence.parse_genome import main as parse_main

        # Set up arguments
        original_argv = sys.argv.copy()
        sys.argv = [
            "parse_genome.py",
            "--fasta", str(fasta_files[0]),  # Use first FASTA file
            "--chromosome", args.chromosome,
            "--n-sequences", str(args.n_sequences),
            "--output", os.path.join(args.output, "genome_parser_test")
        ]

        if vcf_files:
            # Add first VCF file if available
            vcf_snp_files = [f for f in vcf_files if 'snp' in str(f).lower()]
            if vcf_snp_files:
                sys.argv.extend(["--vcf", str(vcf_snp_files[0])])

        print(f"   Running genome parser on: {fasta_files[0].name}")
        parse_main()
        print("   ✅ Genome parser test completed!")

        sys.argv = original_argv

    except Exception as e:
        print(f"   ❌ Genome parser test failed: {e}")
        sys.argv = original_argv

    # Test 2: Coordinate Transform
    print("\n2️⃣ Testing Coordinate Transformation...")
    try:
        from st_stellas.sequence.coordinate_transform import main as coord_main

        # Check if we have parsed sequences
        parsed_sequences = os.path.join(args.output, "genome_parser_test", "sample_sequences.txt")

        original_argv = sys.argv.copy()
        coord_args = [
            "coordinate_transform.py",
            "--output", os.path.join(args.output, "coordinate_test")
        ]

        if os.path.exists(parsed_sequences):
            coord_args.extend(["--input", parsed_sequences])
            print(f"   Using parsed sequences from genome test")
        else:
            coord_args.extend(["--n-sequences", str(min(50, args.n_sequences))])
            print(f"   Generating random sequences for testing")

        sys.argv = coord_args

        coord_main()
        print("   ✅ Coordinate transformation test completed!")

        sys.argv = original_argv

    except Exception as e:
        print(f"   ❌ Coordinate transformation test failed: {e}")
        sys.argv = original_argv

    # Test 3: Check for other modules
    print("\n3️⃣ Checking Available Sequence Modules...")

    sequence_dir = Path(__file__).parent / "src" / "st_stellas" / "sequence"
    if sequence_dir.exists():
        py_files = list(sequence_dir.glob("*.py"))
        py_files = [f for f in py_files if not f.name.startswith("__")]

        print(f"   Found {len(py_files)} sequence analysis modules:")
        for f in py_files:
            has_main = False
            try:
                with open(f, 'r') as file:
                    content = file.read()
                    if 'def main(' in content:
                        has_main = True

                status = "✅ Ready" if has_main else "🔧 Development"
                print(f"     {status} {f.name}")

            except:
                print(f"     ❓ {f.name}")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    print(f"\n📁 Results saved to: {args.output}")
    print(f"🧬 FASTA files: {len(fasta_files)} available")
    print(f"🔬 VCF files: {len(vcf_files)} available")

    print(f"\n🚀 Next Steps:")
    print(f"   • Review results in {args.output}")
    print(f"   • Run comprehensive demo: python comprehensive_sequence_demo.py")
    print(f"   • Use individual modules: python src/st_stellas/sequence/[module].py")
    print(f"   • Integrate with genome analysis: python dante_labs_demo.py")

    print(f"\n📖 For publication-ready analysis:")
    print(f"   • Each module generates high-resolution figures")
    print(f"   • JSON data available for further analysis")
    print(f"   • Comprehensive reports in markdown format")

if __name__ == "__main__":
    main()
