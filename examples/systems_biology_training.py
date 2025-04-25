#!/usr/bin/env python3
"""
Example script demonstrating how to use systems biology data with the Gospel trainer.
"""

import os
import argparse
from typing import List
import json
import matplotlib.pyplot as plt

from gospel.llm.trainer import ModelTrainer
from gospel.llm.network_processor import NetworkDataProcessor
from gospel.llm.network_analysis import NetworkAnalyzer


def main(gene_ids: List[str], model_name: str = "gospel-systems-bio", perform_analysis: bool = True):
    """
    Train a model using systems biology data for a list of genes.
    
    Args:
        gene_ids: List of gene identifiers to process
        model_name: Name for the trained model
        perform_analysis: Whether to perform network analysis
    """
    print(f"Starting systems biology training for genes: {', '.join(gene_ids)}")
    
    # Create output directories
    os.makedirs("network_data", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("network_analysis", exist_ok=True)
    
    # Initialize the network data processor
    processor = NetworkDataProcessor(
        cache_dir="network_data/cache",
        compression="gzip",  # Use compression for smaller files
        max_workers=4,       # Parallel processing
        memory_limit_mb=2048 # 2GB memory limit
    )
    
    # 1. Create training examples from network data
    print("Creating training examples from systems biology data...")
    examples_count = processor.create_connection_training_examples(
        gene_ids, 
        output_file="training_data/network_examples.jsonl",
        batch_size=5  # Process in batches to save memory
    )
    print(f"Created {examples_count} training examples highlighting gene networks")
    
    # 2. Prepare complete network data
    print("Preparing complete network data for training...")
    network_files = processor.prepare_network_for_training(
        gene_ids,
        output_dir="network_data/complete",
        batch_size=5  # Process in batches to save memory
    )
    print(f"Prepared network data files: {network_files}")
    
    # 3. Initialize the model trainer
    trainer = ModelTrainer(
        base_model="llama3",
        model_name=model_name,
        output_dir="trained_models/systems_bio"
    )
    
    # 4. Add training examples
    trainer.add_training_data("training_data/network_examples.jsonl")
    
    # 5. Add network data
    for data_type, file_path in network_files.items():
        trainer.add_network_data(data_type, file_path)
    
    # 6. Create the Ollama model
    modelfile_path = trainer.prepare_ollama_modelfile()
    print(f"Prepared Ollama Modelfile at: {modelfile_path}")
    
    # 7. Train the model (optional)
    print("Training the model (this requires Ollama to be installed)...")
    try:
        trainer.train()
        print(f"Model '{model_name}' trained successfully!")
        
        # Export the model
        export_path = trainer.export_model()
        if export_path:
            print(f"Model exported to: {export_path}")
    except Exception as e:
        print(f"Error during training: {e}")
        print("Note: You need to have Ollama installed and running to train the model")
        print("You can still use the generated Modelfile manually with Ollama")
    
    # 8. Perform network analysis if requested
    if perform_analysis:
        perform_network_analysis(network_files)


def perform_network_analysis(network_files: dict) -> None:
    """
    Perform network analysis on the generated data.
    
    Args:
        network_files: Dictionary with paths to network data files
    """
    print("\n=== Starting Network Analysis ===")
    
    # Initialize the network analyzer
    analyzer = NetworkAnalyzer(output_dir="network_analysis")
    
    # Load protein interaction data
    with open(network_files["interactomes"], 'r') as f:
        interactions_data = json.load(f)
    
    # Load pathway data
    with open(network_files["reactomes"], 'r') as f:
        pathway_data = json.load(f)
    
    # Extract all interactions into a flat list
    all_interactions = []
    for gene_data in interactions_data:
        all_interactions.extend(gene_data.get("interactions", []))
    
    # Extract all pathways into a flat list
    all_pathways = []
    for gene_data in pathway_data:
        all_pathways.extend(gene_data.get("pathways", []))
    
    # Build network from interactions
    print("Building protein interaction network...")
    ppi_network = analyzer.build_network_from_interactions(
        all_interactions,
        min_score=0.4  # Filter low confidence interactions
    )
    print(f"PPI network created with {ppi_network.number_of_nodes()} nodes and {ppi_network.number_of_edges()} edges")
    
    # Build network from pathways
    print("Building pathway-based network...")
    pathway_network = analyzer.build_network_from_pathways(
        all_pathways,
        min_genes_per_pathway=2
    )
    print(f"Pathway network created with {pathway_network.number_of_nodes()} nodes and {pathway_network.number_of_edges()} edges")
    
    # Analyze PPI network
    print("Analyzing protein interaction network...")
    ppi_analysis = analyzer.analyze_network(ppi_network, output_prefix="ppi_network")
    
    # Analyze pathway network
    print("Analyzing pathway-based network...")
    pathway_analysis = analyzer.analyze_network(pathway_network, output_prefix="pathway_network")
    
    # Identify key genes (hub nodes) in PPI network
    print("\nKey genes in the protein interaction network:")
    for i, gene in enumerate(ppi_analysis["key_genes"][:5]):
        print(f"{i+1}. {gene['gene_id']} - Centrality: {gene['centrality_score']:.4f}, Degree: {gene['degree']}")
    
    # Compare networks
    print("\nComparing PPI and pathway networks...")
    comparison = analyzer.compare_networks(
        ppi_network, 
        pathway_network, 
        name1="PPI Network", 
        name2="Pathway Network"
    )
    print(f"Node overlap: {len(comparison['comparison']['common_nodes'])} genes")
    print(f"Node Jaccard similarity: {comparison['comparison']['node_jaccard_similarity']:.4f}")
    
    # Pathway enrichment for top genes
    top_gene_ids = [gene["gene_id"] for gene in ppi_analysis["key_genes"][:10]]
    print(f"\nAnalyzing pathway enrichment for top {len(top_gene_ids)} genes...")
    enrichment = analyzer.analyze_pathway_enrichment(top_gene_ids, all_pathways)
    
    # Print top enriched pathways
    print("\nTop enriched pathways for key genes:")
    for i, pathway in enumerate(enrichment[:5]):
        print(f"{i+1}. {pathway['pathway_name']} - Fold enrichment: {pathway['fold_enrichment']:.2f}")
        print(f"   Genes: {', '.join(pathway['overlap_gene_ids'])}")
    
    print("\nNetwork analysis completed. Results saved in 'network_analysis/' directory")
    print(f"Visualizations available at:")
    print(f"- PPI Network: {ppi_analysis['visualization_path']}")
    print(f"- Pathway Network: {pathway_analysis['visualization_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gospel model with systems biology data")
    parser.add_argument(
        "--genes", 
        type=str, 
        nargs="+",
        default=["ACTN3", "ACE", "PPARGC1A"],
        help="Gene symbols to process"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gospel-systems-bio",
        help="Name for the trained model"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip network analysis step"
    )
    
    args = parser.parse_args()
    main(args.genes, args.model_name, not args.skip_analysis)