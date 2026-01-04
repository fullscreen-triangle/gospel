#!/usr/bin/env python3
"""
Dante Labs VCF Computational Pharmacology Demo

This script demonstrates the complete pipeline for analyzing Dante Labs
whole genome sequencing data using St. Stella's computational pharmacology theory.

Usage:
    python dante_labs_demo.py --vcf your_genome.vcf.gz --output results/
    python dante_labs_demo.py --demo  # Run with simulated data
"""

import sys
import os
import argparse
from pathlib import Path

from src.st_stellas.genome import run_complete_pipeline

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))



def main():
    """Main demonstration of Dante Labs VCF analysis"""

    parser = argparse.ArgumentParser(
        description="Dante Labs VCF Computational Pharmacology Analysis Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze your Dante Labs VCF file
  python dante_labs_demo.py --vcf snp.vcf.gz --integrate-all --visualize

  # Quick demo with simulated data
  python dante_labs_demo.py --demo --visualize

  # Analysis with specific output directory
  python dante_labs_demo.py --vcf indel.vcf.gz --output ./my_analysis/ --integrate-all
        """
    )

    parser.add_argument(
        "--vcf",
        type=str,
        help="Path to Dante Labs VCF file (SNP or indel VCF)",
        default=None
    )

    parser.add_argument(
        "--bam",
        type=str,
        help="Path to BAM alignment file (optional)",
        default=None
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for analysis results",
        default="./dante_labs_analysis_results/"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration with simulated Dante Labs data"
    )

    parser.add_argument(
        "--integrate-all",
        action="store_true",
        help="Integrate with all frameworks (Nebuchadnezzar, Borgia, Bene Gesserit, Hegel)"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate comprehensive visualizations"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick analysis (VCF analysis only, no framework integration)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.demo and not args.vcf:
        print("❌ Error: Please provide a VCF file with --vcf or use --demo for simulation")
        parser.print_help()
        sys.exit(1)

    # Print welcome message
    print_welcome_message()

    # Set up analysis parameters
    integrate_frameworks = args.integrate_all and not args.quick
    visualize = args.visualize or not args.quick

    # Run the complete pipeline
    try:
        results = run_complete_pipeline(
            vcf_file=args.vcf,
            bam_file=args.bam,
            output_dir=args.output,
            demo=args.demo,
            visualize=visualize,
            integrate_frameworks=integrate_frameworks
        )

        # Print results summary
        print_results_summary(results, args.output)

        # Provide next steps guidance
        print_next_steps_guidance(args.output, integrate_frameworks)

        return results

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure VCF file path is correct")
        print("2. Check that all dependencies are installed")
        print("3. Try running with --demo flag first")
        print("4. Check the installation guide: new_testament/INSTALL.md")
        sys.exit(1)

def print_welcome_message():
    """Print welcome message with theory overview"""

    print("="*80)
    print("🧬 DANTE LABS VCF COMPUTATIONAL PHARMACOLOGY ANALYSIS 🧬")
    print("="*80)
    print("Based on St. Stella's Oscillatory Genomics Theory")
    print()
    print("📖 Theory Overview:")
    print("   • Pharmaceuticals work by filling 'oscillatory holes' in biological pathways")
    print("   • Effective drugs target non-encoded pathways (< 1.1% genomic consultation)")
    print("   • Drug action through information catalysis by Biological Maxwell Demons")
    print("   • Multi-scale integration from genomic to membrane to cellular levels")
    print()
    print("🔬 Analysis Includes:")
    print("   • Pharmacogenomic variant classification (CYP450, receptors, transporters)")
    print("   • Oscillatory signature extraction and therapeutic hole identification")
    print("   • Information catalytic efficiency calculation (ηIC)")
    print("   • Personalized pharmaceutical efficacy predictions")
    print("   • Multi-framework validation and evidence rectification")
    print()

def print_results_summary(results: dict, output_dir: str):
    """Print summary of analysis results"""

    vcf_results = results['vcf_analysis']
    integration_results = results.get('integration_results')

    print("\n" + "="*60)
    print("📊 ANALYSIS RESULTS SUMMARY")
    print("="*60)

    # VCF Analysis Summary
    print("\n🧬 Genomic Analysis:")
    print(f"   • Total variants: {len(vcf_results['variants']['all'])}")
    print(f"   • CYP450 variants: {len(vcf_results['variants']['cyp450'])}")
    print(f"   • Consciousness variants: {len(vcf_results['variants']['consciousness'])}")
    print(f"   • Average consultation rate: {vcf_results['oscillatory_signatures']['avg_consultation_rate']:.4f}")
    print(f"   • Therapeutic holes: {len(vcf_results['oscillatory_holes']['therapeutic_targets'])}")

    # Pharmaceutical Predictions
    print("\n💊 Pharmaceutical Predictions:")
    high_efficacy_drugs = [
        drug for drug in vcf_results['pharma_predictions']['drug_responses']
        if drug['efficacy'] > 0.7
    ]
    moderate_efficacy_drugs = [
        drug for drug in vcf_results['pharma_predictions']['drug_responses']
        if 0.3 < drug['efficacy'] <= 0.7
    ]
    low_efficacy_drugs = [
        drug for drug in vcf_results['pharma_predictions']['drug_responses']
        if drug['efficacy'] <= 0.3
    ]

    print(f"   • High efficacy (>70%): {len(high_efficacy_drugs)} drugs")
    print(f"   • Moderate efficacy (30-70%): {len(moderate_efficacy_drugs)} drugs")
    print(f"   • Low efficacy (<30%): {len(low_efficacy_drugs)} drugs")
    print(f"   • Placebo susceptibility: {vcf_results['pharma_predictions']['placebo_susceptibility']:.3f}")

    # Framework Integration Summary
    if integration_results:
        print("\n🔗 Multi-Framework Integration:")
        print(f"   • ATP efficiency: {integration_results.nebuchadnezzar_analysis.get('atp_efficiency', 0):.3f}")
        print(f"   • Quantum transport: {integration_results.bene_gesserit_membrane_circuits.get('quantum_transport_efficiency', {}).get('enaqt_efficiency', 0):.3f}")
        print(f"   • Evidence strength: {integration_results.hegel_evidence_rectification.get('overall_evidence_strength', 0):.3f}")
        print(f"   • Framework agreement: {integration_results.confidence_scores.get('framework_agreement', 0):.3f}")

    # Top Recommendations
    print("\n🎯 Top Pharmaceutical Recommendations:")
    for drug in sorted(vcf_results['pharma_predictions']['drug_responses'],
                      key=lambda x: x['efficacy'], reverse=True)[:3]:
        efficacy = drug['efficacy']
        recommendation = "🟢 RECOMMENDED" if efficacy > 0.7 else "🟡 MONITOR" if efficacy > 0.3 else "🔴 AVOID"
        print(f"   • {drug['drug'].title()}: {efficacy:.3f} - {recommendation}")

def print_next_steps_guidance(output_dir: str, integrated: bool):
    """Print guidance on next steps and file locations"""

    print("\n" + "="*60)
    print("📁 RESULTS & NEXT STEPS")
    print("="*60)

    print("\n📋 Generated Files:")
    print(f"   • Comprehensive report: {output_dir}/comprehensive_report.md")
    print(f"   • VCF analysis data: {output_dir}/vcf_analysis_results.json")
    print(f"   • Visualizations: {output_dir}/visualizations/")

    if integrated:
        print(f"   • Integration summary: {output_dir}/integration_summary.md")
        print(f"   • Framework results: {output_dir}/*_results.json")

    print("\n🔬 Clinical Next Steps:")
    print("   1. Review pharmaceutical recommendations with healthcare provider")
    print("   2. Consider therapeutic drug monitoring for moderate predictions")
    print("   3. Discuss genetic testing implications for drug metabolism")
    print("   4. Evaluate placebo susceptibility for treatment planning")

    print("\n🔧 Framework Integration:")
    if not integrated:
        print("   • Run with --integrate-all for complete multi-framework analysis")
        print("   • Install external frameworks for enhanced validation:")
        print("     - Nebuchadnezzar: https://github.com/fullscreen-triangle/nebuchadnezzar")
        print("     - Borgia: https://github.com/fullscreen-triangle/borgia")
        print("     - Bene Gesserit: https://github.com/fullscreen-triangle/bene-gesserit")
        print("     - Hegel: https://github.com/fullscreen-triangle/hegel")
    else:
        print("   ✅ Complete multi-framework integration performed")
        print("   • Cross-framework validation provides enhanced confidence")
        print("   • Evidence rectification strengthens predictions")

    print("\n📖 Documentation:")
    print("   • Theory: docs/publication/st-stellas-genome.tex")
    print("   • Experiment plan: docs/experiment-plan.md")
    print("   • Installation guide: new_testament/INSTALL.md")
    print("   • Computational pharmacology: docs/oscillations/computational-pharmacology.tex")

def demo_usage_examples():
    """Print usage examples for different scenarios"""

    print("\n" + "="*60)
    print("💡 USAGE EXAMPLES")
    print("="*60)

    examples = [
        {
            "title": "Quick Demo",
            "command": "python dante_labs_demo.py --demo --visualize",
            "description": "Run with simulated data to test the system"
        },
        {
            "title": "SNP Analysis",
            "command": "python dante_labs_demo.py --vcf snp.vcf.gz --visualize",
            "description": "Analyze your SNP VCF from Dante Labs"
        },
        {
            "title": "Complete Analysis",
            "command": "python dante_labs_demo.py --vcf indel.vcf.gz --integrate-all --visualize",
            "description": "Full analysis with all framework integrations"
        },
        {
            "title": "Custom Output",
            "command": "python dante_labs_demo.py --vcf genome.vcf.gz --output ./my_results/",
            "description": "Specify custom output directory"
        }
    ]

    for example in examples:
        print(f"\n{example['title']}:")
        print(f"  {example['command']}")
        print(f"  → {example['description']}")

if __name__ == "__main__":
    # Check if this is a help request
    if len(sys.argv) == 1 or "--help" in sys.argv or "-h" in sys.argv:
        demo_usage_examples()

    main()
