"""
Quick Test: Genomic Validation Suite
=====================================

Runs a quick test to verify all validation modules work correctly.
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from genomic_charged_fluid import (
            GenomicChargedFluid,
            EmptyDictionaryValidator,
            validate_paper_claims
        )
        print("  ✓ genomic_charged_fluid imports successfully")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_charged_fluid():
    """Test GenomicChargedFluid basic functionality."""
    print("\nTesting GenomicChargedFluid...")
    try:
        from genomic_charged_fluid import GenomicChargedFluid

        fluid = GenomicChargedFluid('human')
        state = fluid.measure_state(n_samples=50)

        print(f"  ✓ Created fluid for human genome")
        print(f"  ✓ Measured state: T={state.temperature:.1f}K, C={state.capacitance*1e12:.1f}pF")

        eos = fluid.validate_equation_of_state()
        print(f"  ✓ Equation of state: {eos['equation_satisfied']}")

        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transport_coefficients():
    """Test transport coefficient measurement."""
    print("\nTesting transport coefficients...")
    try:
        from genomic_charged_fluid import GenomicChargedFluid

        fluid = GenomicChargedFluid('human')
        coeffs = fluid.measure_transport_coefficients()

        print(f"  ✓ Measured coefficients:")
        print(f"    μ = {coeffs.chromatin_viscosity:.3e} Pa·s")
        print(f"    ρ = {coeffs.charge_resistivity:.3e} Ω·m")
        print(f"    D = {coeffs.diffusivity:.3e} m²/s")

        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_section_prediction():
    """Test optimal section size prediction."""
    print("\nTesting section size prediction...")
    try:
        from genomic_charged_fluid import GenomicChargedFluid

        fluid = GenomicChargedFluid('human')
        section = fluid.predict_optimal_section_size()

        print(f"  ✓ Predicted section size: {section.section_length_bp:,} bp")
        print(f"  ✓ Matches features: {section.matches_features}")

        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_empty_dictionary():
    """Test empty dictionary validator."""
    print("\nTesting empty dictionary validator...")
    try:
        from genomic_charged_fluid import EmptyDictionaryValidator

        validator = EmptyDictionaryValidator(genome_size=3_000_000_000)

        storage = validator.compare_storage_requirements()
        print(f"  ✓ Storage reduction: {storage['reduction_orders_of_magnitude']:.1f} orders")

        complexity = validator.compare_complexity()
        print(f"  ✓ Speedup: {complexity['speedup_orders_of_magnitude']:.1f} orders")

        accuracy = validator.validate_feature_detection('palindrome', n_tests=50)
        print(f"  ✓ Palindrome detection: {accuracy.coordinate_accuracy:.1%} accuracy")

        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cross_organism():
    """Test cross-organism validation."""
    print("\nTesting cross-organism validation...")
    try:
        from genomic_charged_fluid import GenomicChargedFluid

        organisms = ['human', 'mouse', 'e_coli']

        for org in organisms:
            fluid = GenomicChargedFluid(org)
            state = fluid.measure_state(n_samples=50)
            print(f"  ✓ {org}: {fluid.genome_size_bp/1e6:.1f} Mb, C={state.capacitance*1e12:.1f}pF")

        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print(" "*15 + "GENOMIC VALIDATION TEST SUITE")
    print("="*70)

    tests = [
        ("Imports", test_imports),
        ("Charged Fluid", test_charged_fluid),
        ("Transport Coefficients", test_transport_coefficients),
        ("Section Prediction", test_section_prediction),
        ("Empty Dictionary", test_empty_dictionary),
        ("Cross-Organism", test_cross_organism),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("-"*70)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! Validation suite is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
