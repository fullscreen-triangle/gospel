#!/usr/bin/env python3
"""
Enhanced Gospel Framework Demonstration

This script demonstrates the new metacognitive genomic analysis capabilities
including Bayesian optimization, fuzzy logic, and visual verification.
"""

import asyncio
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path

# Import enhanced Gospel components
from gospel import GospelAnalyzer
from gospel.core import (
    MetacognitiveBayesianNetwork,
    GenomicFuzzySystem,
    GenomicCircuitVisualizer,
    VisualUnderstandingVerifier
)


def create_sample_data():
    """Create sample genomic data for demonstration"""
    
    # Sample variant data
    variants = pd.DataFrame({
        'variant_id': [f'rs{i}' for i in range(1000, 1020)],
        'chromosome': ['1', '2', '3', '4', '5'] * 4,
        'position': np.random.randint(1000000, 2000000, 20),
        'ref': ['A', 'T', 'G', 'C'] * 5,
        'alt': ['T', 'A', 'C', 'G'] * 5,
        'cadd_score': np.random.uniform(0, 40, 20),
        'conservation_score': np.random.uniform(0, 1, 20),
        'allele_frequency': np.random.uniform(0, 0.1, 20),
        'log2_fold_change': np.random.normal(0, 2, 20)
    })
    
    # Sample expression data
    genes = [f'GENE_{i}' for i in range(100)]
    samples = [f'SAMPLE_{i}' for i in range(10)]
    expression = pd.DataFrame(
        np.random.normal(0, 1, (100, 10)),
        index=genes,
        columns=samples
    )
    
    # Sample gene network
    network = nx.erdos_renyi_graph(50, 0.1, directed=True)
    # Add gene names as node attributes
    for i, node in enumerate(network.nodes()):
        network.nodes[node]['symbol'] = f'GENE_{i}'
        network.nodes[node]['function'] = f'function_{i % 5}'
        network.nodes[node]['chromosome'] = str((i % 22) + 1)
        network.nodes[node]['position'] = i * 1000000
    
    # Add edge attributes
    for edge in network.edges():
        network.edges[edge]['interaction_type'] = np.random.choice(['activation', 'repression', 'binding'])
        network.edges[edge]['strength'] = np.random.uniform(0.1, 1.0)
        network.edges[edge]['delay'] = np.random.uniform(0.01, 0.5)
    
    return variants, expression, network


async def demonstrate_basic_analysis():
    """Demonstrate basic enhanced Gospel analysis"""
    print("🧬 Gospel Enhanced Framework Demonstration")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample genomic data...")
    variants, expression, network = create_sample_data()
    print(f"   - {len(variants)} variants")
    print(f"   - {len(expression)} genes with expression data")
    print(f"   - {network.number_of_nodes()} nodes, {network.number_of_edges()} edges in network")
    
    # Initialize Gospel Analyzer
    print("\n🤖 Initializing Gospel Analyzer with metacognitive orchestration...")
    
    # Configure external tools (simulated availability)
    external_tools = {
        'autobahn': True,      # Probabilistic reasoning
        'hegel': True,         # Evidence validation
        'borgia': False,       # Not available in demo
        'nebuchadnezzar': True, # Biological circuits
        'bene_gesserit': False, # Not available in demo
        'lavoisier': False     # Not available in demo
    }
    
    # Tool configuration (would normally point to actual services)
    tool_config = {
        'tools': {
            'autobahn': {
                'base_url': 'http://localhost:8001',
                'api_key': None
            },
            'hegel': {
                'base_url': 'http://localhost:8002', 
                'api_key': None
            },
            'nebuchadnezzar': {
                'base_url': 'http://localhost:8003',
                'api_key': None
            }
        }
    }
    
    analyzer = GospelAnalyzer(
        rust_acceleration=True,
        fuzzy_logic=True,
        visual_verification=True,
        external_tools=external_tools,
        tool_config=tool_config
    )
    
    # Define research objective
    research_objective = {
        'primary_goal': 'identify_pathogenic_variants',
        'confidence_threshold': 0.8,
        'computational_budget': '10_minutes',
        'weights': {
            'pathogenicity': 0.6,
            'conservation': 0.3,
            'expression': 0.1
        }
    }
    
    print("🎯 Research Objective:")
    print(f"   - Goal: {research_objective['primary_goal']}")
    print(f"   - Confidence threshold: {research_objective['confidence_threshold']}")
    print(f"   - Budget: {research_objective['computational_budget']}")
    
    # Perform analysis
    print("\n🔬 Performing metacognitive genomic analysis...")
    try:
        results = await analyzer.analyze(
            variants=variants,
            expression=expression,
            networks=network,
            research_objective=research_objective
        )
        
        # Display results
        print("\n📈 Analysis Results:")
        print(f"   - Execution time: {results.execution_time:.2f} seconds")
        print(f"   - Pathogenic variants identified: {len(results.pathogenic_variants)}")
        print(f"   - Mean confidence: {results.mean_confidence:.3f}")
        print(f"   - Visual verification score: {results.verification_score:.3f}")
        print(f"   - Tool responses: {len(results.tool_responses)}")
        
        # Show top pathogenic variants
        if results.pathogenic_variants:
            print("\n🦠 Top Pathogenic Variants:")
            for i, variant in enumerate(results.pathogenic_variants[:5]):
                print(f"   {i+1}. {variant['variant_id']}: {variant['confidence']:.3f} confidence")
        
        # Show fuzzy analysis summary
        if results.fuzzy_analysis:
            print(f"\n🔮 Fuzzy Logic Analysis: {len(results.fuzzy_analysis)} variants analyzed")
            
        # Show visual verification details
        if results.visual_verification:
            print("\n👁️ Visual Understanding Verification:")
            for test_type in ['occlusion', 'reconstruction', 'perturbation', 'context_switch']:
                mean_key = f'{test_type}_mean'
                if mean_key in results.visual_verification:
                    score = results.visual_verification[mean_key]
                    print(f"   - {test_type.title()} test: {score:.3f}")
        
        # Show system status
        print("\n⚙️ System Status:")
        status = analyzer.get_system_status()
        print(f"   - Rust acceleration: {status['rust_acceleration']}")
        print(f"   - Fuzzy logic: {status['fuzzy_logic_enabled']}")
        print(f"   - Visual verification: {status['visual_verification_enabled']}")
        print(f"   - Available tools: {status['tool_orchestrator_status']['available_tools']}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("   (This is expected in demo mode without actual external tools)")


async def demonstrate_fuzzy_logic():
    """Demonstrate fuzzy logic uncertainty quantification"""
    print("\n🔮 Fuzzy Logic Demonstration")
    print("-" * 30)
    
    # Initialize fuzzy system
    fuzzy_system = GenomicFuzzySystem()
    
    # Test variants with different characteristics
    test_variants = [
        {
            'name': 'High pathogenicity variant',
            'cadd_score': 35.0,
            'conservation_score': 0.95,
            'allele_frequency': 0.001,
            'log2_fold_change': 3.0
        },
        {
            'name': 'Moderate pathogenicity variant',
            'cadd_score': 20.0,
            'conservation_score': 0.6,
            'allele_frequency': 0.01,
            'log2_fold_change': 1.5
        },
        {
            'name': 'Low pathogenicity variant',
            'cadd_score': 8.0,
            'conservation_score': 0.2,
            'allele_frequency': 0.3,
            'log2_fold_change': 0.1
        }
    ]
    
    for variant in test_variants:
        print(f"\n📊 {variant['name']}:")
        result = fuzzy_system.compute_fuzzy_confidence(variant)
        print(f"   - CADD score: {variant['cadd_score']}")
        print(f"   - Conservation: {variant['conservation_score']}")
        print(f"   - Allele frequency: {variant['allele_frequency']}")
        print(f"   - Fuzzy confidence: {result['confidence_score']:.3f}")
        print(f"   - Rules activated: {result['rule_activations']}")


async def demonstrate_visual_verification():
    """Demonstrate visual understanding verification"""
    print("\n👁️ Visual Understanding Verification Demonstration")
    print("-" * 50)
    
    # Create sample network and expression data
    _, expression, network = create_sample_data()
    
    # Initialize components
    visualizer = GenomicCircuitVisualizer()
    bayesian_network = MetacognitiveBayesianNetwork()
    verifier = VisualUnderstandingVerifier(bayesian_network)
    
    # Generate circuit
    print("🔌 Generating genomic circuit diagram...")
    expression_dict = {f'GENE_{i}': np.random.uniform(0, 1) for i in range(50)}
    circuit = visualizer.generate_circuit(network, expression_dict)
    
    print(f"   - Circuit components: {len(circuit.components)}")
    print(f"   - Circuit connections: {circuit.graph.number_of_edges()}")
    
    # Run verification tests
    print("\n🧪 Running understanding verification tests...")
    
    # Occlusion test
    occlusion_result = verifier.occlusion_test(circuit, occlusion_ratio=0.3)
    print(f"   - Occlusion test accuracy: {occlusion_result['accuracy']:.3f}")
    
    # Reconstruction test
    reconstruction_result = verifier.reconstruction_test(circuit, completion_ratio=0.4)
    print(f"   - Reconstruction test accuracy: {reconstruction_result['accuracy']:.3f}")
    
    # Perturbation test
    perturbation_result = verifier.perturbation_test(circuit, perturbation_strength=0.5)
    print(f"   - Perturbation test accuracy: {perturbation_result['overall_accuracy']:.3f}")
    
    # Context switch test
    context = {'cell_type': 'neuron', 'condition': 'stressed'}
    context_result = verifier.context_switch_test(circuit, context)
    print(f"   - Context switch test accuracy: {context_result['adaptation_accuracy']:.3f}")


async def main():
    """Main demonstration function"""
    print("🚀 Starting Gospel Enhanced Framework Demonstrations\n")
    
    try:
        # Basic analysis demonstration
        await demonstrate_basic_analysis()
        
        # Fuzzy logic demonstration
        await demonstrate_fuzzy_logic()
        
        # Visual verification demonstration
        await demonstrate_visual_verification()
        
        print("\n✅ All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 