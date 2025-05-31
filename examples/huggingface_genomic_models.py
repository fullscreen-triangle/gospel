#!/usr/bin/env python3
"""
Example script demonstrating how to use specialized genomic models from Hugging Face Hub
in the Gospel framework.

This script shows:
1. Loading and using specialized genomic models
2. DNA sequence analysis with Caduceus and Nucleotide Transformer
3. Protein sequence analysis with ESM-2 and ProtBERT
4. Variant effect prediction
5. Model comparison and benchmarking
"""

import os
import argparse
from typing import List, Dict
import time

from gospel.llm import (
    GospelLLM, 
    GenomicModelManager, 
    create_analysis_config,
    GENOMIC_MODELS
)


def demonstrate_model_loading():
    """Demonstrate loading and listing available genomic models."""
    print("=== Available Genomic Models ===")
    
    # Create genomic model manager
    manager = GenomicModelManager()
    
    # List all available models
    models = manager.list_available_models()
    for name, config in models.items():
        print(f"\n{name}:")
        print(f"  Model ID: {config['model_id']}")
        print(f"  Type: {config['type']}")
        print(f"  Task: {config['task']}")
        print(f"  Max Length: {config['max_length']:,}")
        print(f"  Description: {config['description']}")
    
    print(f"\nTotal models available: {len(models)}")
    
    # Show model recommendations
    print("\n=== Model Recommendations ===")
    analysis_types = ["variant_effect", "protein_function", "dna_analysis", "sequence_generation"]
    for analysis_type in analysis_types:
        recommended = manager.recommend_models_for_analysis(analysis_type)
        print(f"{analysis_type}: {recommended}")


def demonstrate_dna_analysis():
    """Demonstrate DNA sequence analysis with specialized models."""
    print("\n=== DNA Sequence Analysis ===")
    
    # Sample DNA sequences (ACTN3 gene fragment)
    dna_sequences = [
        "ATGGCCTCGGCCAGCCCCTGGACCAACCCCGTGGCCCTGGCGACTTCTACCTGAAGGTG",
        "ATGGCCTCGGCCAGCCCCTGGACCAACCCCGTGGCCCTGGCGACTTCTACCTGAAGGTGCTCGAC",
        "ATGGCCTCGGCCAGCCCCTGGACCAACCCCGTGGCCCTGGCGACTTCTACCTGAAGGTGCTCGACAAGTACCTCAAG"
    ]
    
    # Initialize genomic model manager
    manager = GenomicModelManager()
    
    # Load DNA analysis models
    print("Loading DNA analysis models...")
    dna_models = manager.load_models_for_analysis("dna_analysis", device="cpu")
    
    if not dna_models:
        print("Warning: No DNA models loaded. This may be due to missing dependencies or connectivity issues.")
        return
    
    # Analyze each sequence
    for i, sequence in enumerate(dna_sequences):
        print(f"\nAnalyzing DNA sequence {i+1} (length: {len(sequence)})")
        print(f"Sequence: {sequence}")
        
        for model_name, model in dna_models.items():
            print(f"\n  Analysis with {model_name}:")
            try:
                start_time = time.time()
                result = model.predict(sequence)
                end_time = time.time()
                
                if "error" not in result:
                    print(f"    ✓ Success in {end_time - start_time:.2f}s")
                    if "embeddings" in result:
                        embedding_shape = result["embeddings"].shape
                        print(f"    Embedding shape: {embedding_shape}")
                        print(f"    Sequence length: {result['sequence_length']}")
                    if "logits" in result:
                        logits_shape = result["logits"].shape
                        print(f"    Logits shape: {logits_shape}")
                else:
                    print(f"    ✗ Error: {result['error']}")
            except Exception as e:
                print(f"    ✗ Exception: {e}")
    
    # Clean up memory
    manager.unload_all_models()


def demonstrate_protein_analysis():
    """Demonstrate protein sequence analysis."""
    print("\n=== Protein Sequence Analysis ===")
    
    # Sample protein sequences
    protein_sequences = [
        "MDQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMISELDQDGFIDKEDLHDGDGKISFEEFLNLVNKEMTADVDGDGQVNYEEFVTMMTSK",  # Calmodulin
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKL",  # Hemoglobin alpha
        "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPVNGFNKDVDAAAKRFVVYAADLLAETVFLEKLSVEDEEGKSVYNSFVSNGVKEYMEGDLEKFLKELKADMNGQSVDETRPRFLEQQNQVLQTKWELLQQVDTSTRTQNKDFQEGKRKKGCPVMKEKFRGNLLWECVKRQIERGMFSMFTSVLQKEIRKRGEAETEDVIVTVDGFSSGEVDLKSVEEKKKKLVNFAGLSRLMKDFAKLAKEYQRFIEEVASFVDRSLVEEGNVVRYEKCFDYVRFEKGSDVYLMDRIKGDPRDTMVIIQQAEGFCKHVGHTLDYLNSYEAFSKIKVDFIKSEGVLVQQFEGEKLLQIEEIERDLSRTSRRSGAASDSEETEPANKMIQVKQGPEEPAFGGKGKSRLEGEEKQIDKLAEQFKKDGPEAPGASSQPPSTPGDKVRKTTPEEAVAKKDADAYKFYKETDKVYVELLNKGNPYEYNSKTGSLGELVTTIKSEEAEKVNQHYGGKMKDGTESFTKVKYGGEGADLKLDPNYFHPQKFYFGVFSGSSDFPEYSLVLTLNRNSFGGGVAQDHLKVAYKKLVDEKAVGGVRTPKRQVYHHFNYMTSGKGGNIIVTGGSGRGGLDGKRKMIGILVGVTKQVAERIRKDDQEEKKIKEKFGGLYGLAIKTSIIYTAEEANKLFQTAIREGLLRTFQADNLGSEKKVDYEFYKTFGYAKETSKGGSDRRKAKNGQAQNVNRYQQDSSGEDVYIGEKGQVPNGDNASGSMGDRTKGFKPEDKLGKTLAKKTVWKVFLGRVHGRAGTNLGVAAIGEFVRNNIGYKSDSRKYAADMSLLFLRTILSDDDGTGALKIQDDVMGDGTQSWLLHGKGTTEIGPAIPTKIEYLGKEEKPDKVEKYQLYEEIRGSWLTVDKVSLAQGQKFEEYKKNELIKNFNGNPEQGTDQITIQELFPFITYNKHGGEPAPIDPHAPYVNHSPNQTPLFSAYIAKSGSQSTSSHKQTARSISVDDEKLKKSSDGYGKAQQLIAGEGKVILVYKSLDQYFKNPDDLKQEHDPRYKKFIEDFQEFANEHYLRSYEGIVLGLMRGEAVKFHTKDDMEQAKEYLKHHGIKDDDEQRPPIKVAFEPGYRVILYQAADAHGQFQRYQGHDKDDAENEDCDIVLIIKGEDLLPGGIDTVNGKKIRQKDAQNEEILQDLQRIIEQIAGRKQEFLEYDQKLRLGTSHKQNSKELKIEGLKRYKEEEGLKALEKIRLDGTDVKVAENILSGFGTAQEKNMDKSAEEVIEETVDTTKDLIRETQETQAKIGQLERKWQRRKDYLSERKDPNSEIRNWFYINQMNKLKHQFGEKKGGFLEETDVGNFLRGLDLNKLYQGEIDRRQEKGSYQEGYDSGEEQGKGNNKRSGQRCDKKPGYLDPQERQNQTVEYLKKLGIYDKGYIDRLRNKDHAFNLNSSGTLTGDSGLDQNEKKKTKAGVTVVNRLPYGSVHVKEIVKDGEKDLQDMPEILSKNGLDSPDLLKGGGTPQRTAVLDLLSKAGEAGLDSGILKGDVAEQLNAFGDLIKCNKLSSRGLEILQTPTEQLRGGKSIIAESGPRSRVGFGDPQVAVLNFLDKGKEGNQDSLKFGDGSFQSMLNFLDKGKENKGGEDKWFMKVYDAEGGLPNNTSLLKKVQRRCNGSVDVLSEEEGTKNPGQVVLYGNQEELRMCAATLSTFLGDVGDRIRSLLDGGSPGGASLDSLYLSNLQSMDAEEEEYFKGKSRLRFAAQVLDKEEGTEEQLTTEQEETLRKEGSRFSGKAMTVDPGSGDDAGTAHLTFLTVDGGGGSSCFHIKDQLLMADLGGEPIVNLDTNHKNQESQSLLTGDADKAYFRGKSLIEHVEEELIAEAFLALQHPHIGKQAEDLLPSYGPDTDDEFRARKKLAEQRNQYDVEELEEQLGRSGGQRLLAELEEQRKAKEEAGEAKISQGQTKLGVDGLLPGSTKEGQAKKKTILKQFKGGEEEETKKLKTLVTQGEQFAKAFDAELADRLIAEGNPRGLGVAGLQTVNMTLQSRPTGDVIAKITGTGGADYDIAKQFVKYGKQTDVKTTEIGTKQDDTSGGIQSRLEKLEQKEREAKLREEAQRLIETGGAGDKGKQEPLTMFDQALEEILAKFKDAKATDNQGDGRIDYAELSKSFEDVDKTLRSLVDEGLTQLLGEQVKEVLGTFKGLQSQLAEEIEQQLKARKQTMEEETIVVQLDSQDFEKFKGQVPDFQEALKAFDAELKELRPEMVKDFQEKGIYLALLLLAEKQGGEQAAIKLIKEATEKIFNEKKAAEFEIELKEKLLTNQSIITGEGILSAHLGVGVAKSLKPGSSEILQSLKYALVEALSRVDGQLLPEGLASRFQKRSLDYVKQHLTEGQVRRQTRFFGSFDSQKEPPRGRLTLHMERRQSQGQFPLFNLEFADLGQLGSGDLEVYVPPLRFSLEVLGEVTVQDDVDLGQYTRILVGLQEGNQQFVLISTKGQPKFGKELSQRKMGDLSKGLLLEVGAPEHSLLKPAVDLVGQNLVELGLTGGNSITRGPLRGLQVGKKHVDGSDQAGSQGVVASGMALFASGKGAGMKKMGLAGGTALLVNMGSATYLNPAHHLRGSLLQLGGTVLLGFHGFSNHYRFKTHRLALHLDDAERFQLFNFAEVNNHFDKRYDAILAEMTKPEVGGQSQGGISELKKQRRLEEERKQSLRKQQRELTSLLEQRIEREDSQAEKGKESGQNKKEQDFIQEFMHYFNPFNTQFEEAAALEQDLFEKQGAQGQTLEVLHVALGFLDLGQSLGNRLDQVEALMSLAAQDIIQVDYRGLDRMELLQGAEMLDALQLALEDRGAGGGELVQESQAEVLLQIDSLLTFFDPSKRVLQQIAQGIRRKLSEGELKSLGFGQQEAGQHRNQHVQQLRKEFEAFQATMEKTPPEFQAALEMQSQMQQSLQEHGKDNSHLFSGLSRGRQNRRRESFQKKLAEDFEKLRSSDRLQALQVEYLKEFKSALEELQRVDHNNQRLLAEMEEKAKKAQEEEEDGQLMSLNKQYVQQNSQEMSRRRREEFQKKLAEDFEKLRSSDRLQALQVEYLKEFKSALEELQRVDHNNQRLLAEMEEKAKKAQEEEEDGQLMSLNKQYVQQNSQEMSRRR"  # Long protein
    ]
    
    # Initialize genomic model manager
    manager = GenomicModelManager()
    
    # Load protein analysis models
    print("Loading protein analysis models...")
    protein_models = manager.load_models_for_analysis("protein_function", device="cpu")
    
    if not protein_models:
        print("Warning: No protein models loaded. This may be due to missing dependencies or connectivity issues.")
        return
    
    # Analyze each sequence
    for i, sequence in enumerate(protein_sequences[:2]):  # Limit to first 2 for demo
        print(f"\nAnalyzing protein sequence {i+1} (length: {len(sequence)})")
        print(f"Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
        
        for model_name, model in protein_models.items():
            print(f"\n  Analysis with {model_name}:")
            try:
                start_time = time.time()
                
                # Truncate sequence if too long for the model
                max_len = model.max_length - 10  # Leave room for special tokens
                truncated_sequence = sequence[:max_len] if len(sequence) > max_len else sequence
                
                result = model.predict(truncated_sequence)
                end_time = time.time()
                
                if "error" not in result:
                    print(f"    ✓ Success in {end_time - start_time:.2f}s")
                    if "embeddings" in result:
                        embedding_shape = result["embeddings"].shape
                        print(f"    Embedding shape: {embedding_shape}")
                    if len(sequence) > max_len:
                        print(f"    Note: Sequence truncated to {max_len} characters")
                else:
                    print(f"    ✗ Error: {result['error']}")
            except Exception as e:
                print(f"    ✗ Exception: {e}")
    
    # Clean up memory
    manager.unload_all_models()


def demonstrate_variant_effect_prediction():
    """Demonstrate genetic variant effect prediction."""
    print("\n=== Variant Effect Prediction ===")
    
    # Example: ACTN3 R577X variant (common in athletes)
    reference_sequence = "ATGGCCTCGGCCAGCCCCTGGACCAACCCCGTGGCCCTGGCGACTTCTACCTGAAGGTGCTCGACAAGTACCTCAAG"
    variant_sequence = "ATGGCCTCGGCCAGCCCCTGGACCAACCCCGTGGCCCTGGCGACTTCTACCTGAAGATGCTCGACAAGTACCTCAAG"
    #                                                                           ^-- C to T mutation (R577X)
    
    print(f"Reference: {reference_sequence}")
    print(f"Variant:   {variant_sequence}")
    print(f"Mutation:  C→T at position 58 (R577X in ACTN3)")
    
    # Initialize Gospel LLM with genomic models
    try:
        gospel_llm = GospelLLM(
            genomic_models=["caduceus", "nucleotide_transformer"],
            use_ollama=False,
            device="cpu"
        )
        
        # Predict variant effect
        if gospel_llm.genomic_models:
            for model_name in gospel_llm.genomic_models:
                print(f"\nVariant effect prediction with {model_name}:")
                result = gospel_llm.predict_variant_effect(
                    reference_sequence, 
                    variant_sequence, 
                    model_name
                )
                
                if "error" not in result:
                    print(f"  Similarity score: {result.get('similarity_score', 'N/A'):.4f}")
                    print(f"  Variant effect score: {result.get('variant_effect_score', 'N/A'):.4f}")
                    print(f"  Model used: {result['model_used']}")
                else:
                    print(f"  Error: {result['error']}")
        else:
            print("No genomic models loaded for variant prediction.")
            
    except Exception as e:
        print(f"Error initializing Gospel LLM with genomic models: {e}")


def demonstrate_sequence_analysis_with_gospel():
    """Demonstrate using Gospel LLM with integrated genomic models."""
    print("\n=== Gospel LLM with Genomic Models ===")
    
    # Sample sequences for analysis
    dna_sequence = "ATGGCCTCGGCCAGCCCCTGGACCAACCCCGTGGCCCTGGCGACTTCTACCTGAAGGTG"
    protein_sequence = "MDQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMISELDQDGFIDKEDLHDGDGK"
    
    try:
        # Initialize Gospel LLM with multiple genomic models
        gospel_llm = GospelLLM(
            base_model="microsoft/DialoGPT-medium",  # Example HF model for general LLM
            genomic_models=["caduceus", "esm2"],  # Load specialized models
            use_ollama=False,  # Use HuggingFace for main LLM
            device="cpu"
        )
        
        # Analyze DNA sequence
        print("\nAnalyzing DNA sequence:")
        dna_result = gospel_llm.analyze_sequence(dna_sequence, sequence_type="dna")
        print(f"Sequence: {dna_result['sequence']}")
        print(f"Type: {dna_result['sequence_type']}")
        print(f"Length: {dna_result['length']}")
        
        for model_name, analysis in dna_result['analysis'].items():
            if "error" not in analysis:
                print(f"  {model_name}: ✓ Success")
            else:
                print(f"  {model_name}: ✗ {analysis['error']}")
        
        # Analyze protein sequence  
        print("\nAnalyzing protein sequence:")
        protein_result = gospel_llm.analyze_sequence(protein_sequence, sequence_type="protein")
        print(f"Sequence: {protein_result['sequence'][:50]}...")
        print(f"Type: {protein_result['sequence_type']}")
        print(f"Length: {protein_result['length']}")
        
        for model_name, analysis in protein_result['analysis'].items():
            if "error" not in analysis:
                print(f"  {model_name}: ✓ Success")
            else:
                print(f"  {model_name}: ✗ {analysis['error']}")
        
        # Get available models info
        print(f"\nAvailable genomic models: {list(gospel_llm.get_available_genomic_models().keys())}")
        print(f"Loaded genomic models: {list(gospel_llm.genomic_models.keys())}")
        
    except Exception as e:
        print(f"Error demonstrating Gospel LLM integration: {e}")


def demonstrate_model_benchmarking():
    """Demonstrate model benchmarking capabilities."""
    print("\n=== Model Benchmarking ===")
    
    # Test sequences of different lengths
    test_sequences = [
        "ATGGCCTCGGCCAGCCCCTGGACCAACCCC",  # Short
        "ATGGCCTCGGCCAGCCCCTGGACCAACCCCGTGGCCCTGGCGACTTCTACCTGAAGGTG",  # Medium
        "ATGGCCTCGGCCAGCCCCTGGACCAACCCCGTGGCCCTGGCGACTTCTACCTGAAGGTGCTCGACAAGTACCTCAAGAAGCTGTACCTGGCCCGCATGGCC"  # Long
    ]
    
    manager = GenomicModelManager()
    
    # Try to benchmark one model (if available)
    try:
        model = manager.load_model("caduceus", device="cpu")
        if model:
            print("Benchmarking Caduceus model...")
            benchmark_results = manager.benchmark_model("caduceus", test_sequences)
            
            print(f"Benchmark Results:")
            print(f"  Model: {benchmark_results['model_name']}")
            print(f"  Sequences tested: {benchmark_results['num_sequences']}")
            print(f"  Total time: {benchmark_results['timing']['total_time']:.2f}s")
            print(f"  Average time per sequence: {benchmark_results['timing']['average_time_per_sequence']:.2f}s")
            print(f"  Sequences per second: {benchmark_results['timing']['sequences_per_second']:.2f}")
            
            # Show individual results
            for i, pred_result in enumerate(benchmark_results['predictions']):
                success = "✓" if pred_result['success'] else "✗"
                print(f"  Sequence {i+1} (len={pred_result['sequence_length']}): {success} {pred_result['prediction_time']:.3f}s")
        else:
            print("Could not load model for benchmarking")
    except Exception as e:
        print(f"Benchmarking failed: {e}")
    
    # Show memory usage if any models are loaded
    memory_usage = manager.get_memory_usage()
    print(f"\nMemory Usage:")
    print(f"  Loaded models: {memory_usage['loaded_models']}")
    if memory_usage['loaded_models'] > 0:
        for model_name, details in memory_usage['model_details'].items():
            print(f"  {model_name}: {details}")
    
    manager.unload_all_models()


def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(description="Gospel Genomic Models Demo")
    parser.add_argument(
        "--demo", 
        choices=["all", "models", "dna", "protein", "variant", "gospel", "benchmark"],
        default="all",
        help="Which demonstration to run"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face API token (optional, for private models)"
    )
    
    args = parser.parse_args()
    
    print("Gospel Framework - Hugging Face Genomic Models Demo")
    print("=" * 60)
    
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        print("Using provided Hugging Face token")
    
    try:
        if args.demo in ["all", "models"]:
            demonstrate_model_loading()
        
        if args.demo in ["all", "dna"]:
            demonstrate_dna_analysis()
        
        if args.demo in ["all", "protein"]:
            demonstrate_protein_analysis()
        
        if args.demo in ["all", "variant"]:
            demonstrate_variant_effect_prediction()
        
        if args.demo in ["all", "gospel"]:
            demonstrate_sequence_analysis_with_gospel()
        
        if args.demo in ["all", "benchmark"]:
            demonstrate_model_benchmarking()
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main() 