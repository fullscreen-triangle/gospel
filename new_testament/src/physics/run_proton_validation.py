#!/usr/bin/env python3
"""
Run Proton Trajectory Validation Suite
=======================================

Validates partition theory predictions for DNA H-bond dynamics:
1. Proton oscillation in H-bonds (categorical binary states)
2. DNA capacitor charge/discharge
3. Triple equivalence: T_osc = 2π T_cat
4. Ideal gas laws for genomic systems
5. Phase-locked H-bond network

Results saved to JSON and CSV formats.

Usage:
    python run_proton_validation.py
    python run_proton_validation.py --samples 5000 --trials 200
    python run_proton_validation.py --output-dir ./results
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from proton_trajectory_validation import (
    ValidationRunner,
    HBondProtonOscillator,
    DNACapacitor,
    TripleEquivalenceValidator,
    ProtonTrajectoryValidator,
    IdealGasGenomicValidator,
    PhaseLockValidator,
)


def main():
    parser = argparse.ArgumentParser(
        description='Run proton trajectory validation suite'
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=1000,
        help='Number of samples per test (default: 1000)'
    )
    parser.add_argument(
        '--trials', '-t',
        type=int,
        default=100,
        help='Number of trials for determinism test (default: 100)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode with reduced samples'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    if args.quick:
        args.samples = 100
        args.trials = 10

    print("=" * 80)
    print("PROTON TRAJECTORY VALIDATION SUITE")
    print("=" * 80)
    print(f"Samples: {args.samples}")
    print(f"Trials: {args.trials}")
    print(f"Output: {args.output_dir or 'default'}")
    print("=" * 80)

    # Create runner
    runner = ValidationRunner(output_dir=args.output_dir)

    # Run all validations
    results = runner.run_all_validations(
        n_samples=args.samples,
        n_trials=args.trials
    )

    # Save results
    json_path = runner.save_results_json()
    csv_path = runner.save_results_csv()

    # Generate detailed trajectory
    print("\nGenerating detailed trajectory data...")
    oscillator = HBondProtonOscillator()
    oscillator.measure_trajectory(min(args.samples, 1000))
    trajectory_path = runner.save_trajectory_csv(oscillator)

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    summary = results.get('summary', {})
    passed = summary.get('tests_passed', 0)
    total = summary.get('tests_total', 0)

    print(f"\nTests passed: {passed}/{total}")
    print(f"Pass rate: {passed/total*100 if total > 0 else 0:.1f}%")

    print(f"\nOutput files:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  Trajectory: {trajectory_path}")

    # Return exit code based on test results
    return 0 if summary.get('all_passed', False) else 1


if __name__ == "__main__":
    sys.exit(main())
