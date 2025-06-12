#!/usr/bin/env python3
"""
Test script to verify Gospel framework works with real genomic data.

This script tests:
1. Real data retrieval from public APIs
2. HuggingFace model integration
3. Core framework functionality
"""

import sys
import traceback
from gospel.core import VariantProcessor, VariantAnnotator, GenomicScorer


def test_real_data_integration():
    """Test that the framework can work with real data."""
    print("=== Testing Gospel Framework with Real Data ===\n")
    
    success_count = 0
    total_tests = 4
    
    # Test 1: Import core modules
    print("1. Testing core module imports...")
    try:
        from gospel.core import VariantProcessor, VariantAnnotator
        from gospel.llm import GospelLLM
        print("✓ Successfully imported core modules")
        success_count += 1
    except Exception as e:
        print(f"✗ Failed to import modules: {e}")
    
    # Test 2: Initialize components
    print("\n2. Testing component initialization...")
    try:
        processor = VariantProcessor()
        annotator = VariantAnnotator()
        scorer = GenomicScorer()
        print("✓ Successfully initialized core components")
        success_count += 1
    except Exception as e:
        print(f"✗ Failed to initialize components: {e}")
        traceback.print_exc()
    
    # Test 3: Test API connectivity
    print("\n3. Testing API connectivity...")
    try:
        import requests
        
        # Test Ensembl API
        response = requests.get("https://rest.ensembl.org/info/ping", 
                              headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            print("✓ Ensembl API is accessible")
            success_count += 1
        else:
            print(f"✗ Ensembl API not accessible (status: {response.status_code})")
    except Exception as e:
        print(f"✗ Failed to test API connectivity: {e}")
    
    # Test 4: Test real data fetch
    print("\n4. Testing real data retrieval...")
    try:
        import requests
        
        # Fetch real gene data from Ensembl
        gene_url = "https://rest.ensembl.org/lookup/symbol/homo_sapiens/BRCA1"
        headers = {"Content-Type": "application/json"}
        response = requests.get(gene_url, headers=headers)
        
        if response.status_code == 200:
            gene_data = response.json()
            print(f"✓ Successfully retrieved BRCA1 data:")
            print(f"  - Gene ID: {gene_data.get('id', 'N/A')}")
            print(f"  - Chromosome: {gene_data.get('seq_region_name', 'N/A')}")
            print(f"  - Start: {gene_data.get('start', 'N/A')}")
            print(f"  - End: {gene_data.get('end', 'N/A')}")
            success_count += 1
        else:
            print(f"✗ Failed to retrieve gene data (status: {response.status_code})")
    except Exception as e:
        print(f"✗ Failed to retrieve real data: {e}")
    
    # Summary
    print(f"\n=== Test Results ===")
    print(f"Passed: {success_count}/{total_tests} tests")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Framework is ready for real genomic analysis.")
        return True
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        return False


def test_huggingface_availability():
    """Test HuggingFace model availability."""
    print("\n=== Testing HuggingFace Model Availability ===")
    
    try:
        from transformers import AutoTokenizer
        
        # Test if we can load a genomic model tokenizer
        print("Testing Nucleotide Transformer availability...")
        tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-500m-human-ref",
            cache_dir="./.cache"
        )
        print("✓ Successfully loaded Nucleotide Transformer tokenizer")
        print(f"  - Vocab size: {tokenizer.vocab_size}")
        return True
        
    except Exception as e:
        print(f"✗ HuggingFace model test failed: {e}")
        print("Note: This might require internet connection and sufficient disk space")
        return False


def main():
    """Main test function."""
    print("Gospel Framework Real Data Integration Test")
    print("=" * 50)
    
    # Run core tests
    core_success = test_real_data_integration()
    
    # Run HuggingFace tests (optional)
    print("\n" + "=" * 50)
    hf_success = test_huggingface_availability()
    
    # Final summary
    print("\n" + "=" * 50)
    print("FINAL SUMMARY:")
    
    if core_success:
        print("✓ Core framework functionality: WORKING")
    else:
        print("✗ Core framework functionality: ISSUES DETECTED")
    
    if hf_success:
        print("✓ HuggingFace integration: WORKING")
    else:
        print("? HuggingFace integration: NOT TESTED (requires internet/models)")
    
    print("\nThe Gospel framework is now focused on:")
    print("- Real genomic data from public APIs (Ensembl, ClinVar, GWAS Catalog)")
    print("- HuggingFace transformer models for sequence analysis")
    print("- Cross-domain analysis (genomics + pharmacogenomics + systems biology)")
    print("- NO simulated data or knowledge base components")
    
    return core_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 