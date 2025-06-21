#!/usr/bin/env python3
"""
Gospel Turbulance DSL Demonstration

This script demonstrates how to use Gospel's Turbulance DSL compiler
to encode scientific hypotheses and validate genomic analysis workflows.

Turbulance enables researchers to:
1. Express scientific hypotheses with semantic validation
2. Validate statistical and logical soundness
3. Generate executable analysis plans
4. Delegate tasks to specialized tools (Gospel, Autobahn, etc.)
"""

import json
from pathlib import Path
from gospel import (
    TurbulanceCompiler, 
    compile_turbulance_script, 
    validate_turbulance_script,
    GospelAnalyzer
)

def main():
    """Demonstrate Turbulance DSL compilation and validation"""
    
    print("🧬 Gospel Turbulance DSL Demonstration")
    print("="*50)
    
    # Initialize Turbulance compiler
    compiler = TurbulanceCompiler(use_rust=True)
    print(f"✅ Initialized Turbulance compiler (Rust backend: {compiler.use_rust})")
    
    # Example Turbulance script for genomic analysis
    turbulance_script = """
    // Scientific hypothesis about variant pathogenicity
    hypothesis VariantPathogenicity:
        claim: "Multi-feature genomic variants predict pathogenicity with accuracy >85%"
        
        semantic_validation:
            biological_understanding: "Pathogenic variants disrupt protein function through structural changes"
            temporal_understanding: "Variant effects manifest across developmental stages"
            clinical_understanding: "Pathogenicity correlates with disease severity"
        
        requires: "statistical_validation"
    
    // Main analysis function
    funxn main():
        // Load genomic data
        item variants = load_vcf("data/variants.vcf")
        
        // Delegate to Gospel for analysis
        delegate_to gospel, task: "variant_analysis", data: {
            variants: variants,
            prediction_threshold: 0.85
        }
        
        return "analysis_complete"
    
    // Validation proposition
    proposition ValidateResults:
        motion CheckAccuracy:
            description: "Prediction accuracy exceeds 85% threshold"
        
        within validation_context:
            given accuracy > 0.85:
                support CheckAccuracy, confidence: 0.9
    """
    
    print("\n📝 Parsing Turbulance script...")
    try:
        # Parse the script
        ast = compiler.parse(turbulance_script)
        print(f"✅ Successfully parsed AST with {len(ast.hypotheses)} hypotheses")
        
        # Display hypotheses
        for hypothesis in ast.hypotheses:
            print(f"   📋 Hypothesis: {hypothesis.name}")
            print(f"   📌 Claim: {hypothesis.claim[:80]}...")
            
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        return
    
    print("\n🔍 Validating scientific soundness...")
    try:
        # Validate the AST
        validation_errors = compiler.validate(ast)
        
        if not validation_errors:
            print("✅ Script passed all validation checks!")
        else:
            print("⚠️  Validation warnings:")
            for error in validation_errors:
                print(f"   - {error}")
                
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return
    
    print("\n⚙️ Compiling to execution plan...")
    try:
        # Compile to execution plan
        execution_plan = compiler.compile(turbulance_script)
        
        print(f"✅ Compilation successful!")
        print(f"   📊 Hypothesis validations: {len(execution_plan.hypothesis_validations)}")
        print(f"   🔧 Tool delegations: {len(execution_plan.tool_delegations)}")
        print(f"   📋 Execution steps: {len(execution_plan.execution_order)}")
        print(f"   🧠 Semantic requirements: {len(execution_plan.semantic_requirements)}")
        
        # Display hypothesis validations
        for validation in execution_plan.hypothesis_validations:
            print(f"\n   🧪 Hypothesis: {validation['hypothesis']}")
            print(f"   ✓ Valid: {validation['is_scientifically_valid']}")
            print(f"   📝 Reason: {validation['validation_reason']}")
            
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return
    
    # Demonstrate file-based compilation
    print("\n📁 Testing file-based compilation...")
    
    script_file = Path("examples/turbulance_genomic_analysis.trb")
    if script_file.exists():
        try:
            # Validate script file
            errors = validate_turbulance_script(script_file)
            if errors:
                print("⚠️  File validation errors:")
                for error in errors:
                    print(f"   - {error}")
            else:
                print("✅ File validation passed!")
                
                # Compile script file
                plan = compile_turbulance_script(script_file)
                print(f"✅ File compilation successful!")
                
                # Save execution plan
                plan_file = Path("examples/genomic_analysis_plan.json")
                plan.save(plan_file)
                print(f"💾 Execution plan saved to {plan_file}")
                
        except Exception as e:
            print(f"❌ File processing failed: {e}")
    else:
        print(f"⚠️  Example file not found: {script_file}")
    
    # Integration with Gospel analyzer
    print("\n🔬 Integration with Gospel analyzer...")
    try:
        # Initialize Gospel analyzer
        gospel = GospelAnalyzer()
        
        # Add Turbulance compilation capability
        if hasattr(gospel, 'compile_turbulance'):
            print("✅ Gospel has native Turbulance support")
        else:
            print("ℹ️  Gospel can be extended with Turbulance compilation")
            
        print("✅ Ready for scientific hypothesis validation and genomic analysis!")
        
    except Exception as e:
        print(f"⚠️  Gospel integration note: {e}")
    
    print("\n🎉 Turbulance DSL demonstration complete!")
    print("\nNext steps:")
    print("1. Create your own .trb scripts with scientific hypotheses")
    print("2. Use Gospel for genomic analysis with validated reasoning")
    print("3. Delegate probabilistic tasks to Autobahn")
    print("4. Integrate with other specialized tools as needed")


def demonstrate_advanced_features():
    """Demonstrate advanced Turbulance features"""
    
    print("\n🚀 Advanced Turbulance Features")
    print("-" * 30)
    
    # Complex scientific reasoning validation
    complex_script = """
    hypothesis ComplexGenomicInteractions:
        claim: "Multi-gene interactions predict disease risk through network effects with 92% accuracy"
        
        semantic_validation:
            biological_understanding: "Gene networks exhibit emergent properties through protein-protein interactions and regulatory cascades"
            temporal_understanding: "Network effects manifest across multiple developmental timepoints with cumulative effects"
            clinical_understanding: "Complex traits result from polygenic architecture with non-additive genetic effects"
        
        requires: "network_topology_validation"
    
    proposition NetworkEffectsValidation:
        motion ValidateNetworkTopology:
            description: "Gene network topology reflects known biological pathways"
        
        motion ValidateEmergentProperties:
            description: "Network-level predictions exceed single-gene predictions"
        
        within network_analysis_context:
            given network_accuracy > single_gene_accuracy + 0.1:
                support ValidateEmergentProperties, confidence: 0.88
                
            given pathway_enrichment_p < 0.001:
                support ValidateNetworkTopology, confidence: 0.92
    """
    
    compiler = TurbulanceCompiler()
    
    try:
        print("🧠 Parsing complex genomic interactions hypothesis...")
        ast = compiler.parse(complex_script)
        
        print("🔍 Validating advanced scientific reasoning...")
        errors = compiler.validate(ast)
        
        if not errors:
            print("✅ Advanced validation passed - complex reasoning is scientifically sound!")
        else:
            print("⚠️  Advanced validation notes:")
            for error in errors:
                print(f"   - {error}")
                
        print("⚙️ Compiling advanced execution plan...")
        plan = compiler.compile(complex_script)
        
        print("✅ Advanced compilation successful!")
        print(f"   🔬 Complex semantic requirements: {len(plan.semantic_requirements)}")
        
    except Exception as e:
        print(f"❌ Advanced features demonstration failed: {e}")


if __name__ == "__main__":
    try:
        main()
        demonstrate_advanced_features()
    except KeyboardInterrupt:
        print("\n👋 Demonstration interrupted by user")
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc() 