"""
Visualize command for the Gospel CLI.

This module implements the 'visualize' command, which generates visual
representations of genomic data analysis results across fitness,
pharmacogenetics, and nutrition domains.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def configure_parser(subparsers) -> argparse.ArgumentParser:
    """Configure the argument parser for the visualize command.
    
    Args:
        subparsers: Subparsers object from the main argument parser
        
    Returns:
        Configured argument parser for the visualize command
    """
    parser = subparsers.add_parser(
        "visualize",
        help="Generate visualizations from genomic analysis",
        description="Create visual representations of genomic data analysis results."
    )
    
    # Input options
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/",
        help="Directory containing analysis results"
    )
    
    # Visualization type
    parser.add_argument(
        "--type",
        type=str,
        choices=["domain-summary", "variant-impact", "gene-scores", "trait-distribution", 
                 "drug-interactions", "nutrient-needs", "network"],
        default="domain-summary",
        help="Type of visualization to generate"
    )
    
    # Filtering options
    parser.add_argument(
        "--domain",
        type=str,
        choices=["fitness", "pharmacogenetics", "nutrition"],
        help="Limit visualization to a specific domain"
    )
    
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Limit to top N results (for applicable visualizations)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="visualization.png",
        help="Output file path (PNG, PDF, SVG, or JPG)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for output image (dots per inch)"
    )
    
    parser.add_argument(
        "--figsize",
        type=str,
        default="10x6",
        help="Figure size in inches (width x height, e.g., '10x6')"
    )
    
    return parser


def load_results(results_dir: str) -> Dict:
    """Load analysis results from output files.
    
    Args:
        results_dir: Path to the directory containing analysis results
        
    Returns:
        Loaded results
    """
    results = {
        "summary": {},
        "variants": [],
        "domains": {}
    }
    
    try:
        # Load summary results
        summary_path = os.path.join(results_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as summary_file:
                results["summary"] = json.load(summary_file)
                logger.debug(f"Loaded summary from {summary_path}")
        
        # Load variant results
        variants_path = os.path.join(results_dir, "variants.json")
        if os.path.exists(variants_path):
            with open(variants_path, 'r') as variants_file:
                results["variants"] = json.load(variants_file)
                logger.debug(f"Loaded variants from {variants_path}")
        
        # Load domain-specific results
        for domain in ["fitness", "pharmacogenetics", "nutrition"]:
            domain_path = os.path.join(results_dir, f"{domain}.json")
            if os.path.exists(domain_path):
                with open(domain_path, 'r') as domain_file:
                    results["domains"][domain] = json.load(domain_file)
                    logger.debug(f"Loaded {domain} results from {domain_path}")
        
        # Load network results
        network_path = os.path.join(results_dir, "network.json")
        if os.path.exists(network_path):
            with open(network_path, 'r') as network_file:
                results["network"] = json.load(network_file)
                logger.debug(f"Loaded network from {network_path}")
        
        return results
    
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading results: {e}")
        raise


def parse_figsize(figsize_str: str) -> Tuple[float, float]:
    """Parse figure size string into tuple of width and height.
    
    Args:
        figsize_str: Figure size string in format "widthxheight"
        
    Returns:
        Tuple of (width, height) in inches
    """
    try:
        width, height = figsize_str.lower().split('x')
        return (float(width), float(height))
    except ValueError:
        logger.warning(f"Invalid figsize format '{figsize_str}', using default (10, 6)")
        return (10, 6)


def visualize_domain_summary(results: Dict, domain: Optional[str] = None, 
                             figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """Generate domain summary visualization.
    
    Args:
        results: Analysis results
        domain: Optional domain to filter results
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure object
    """
    summary = results["summary"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract domain summary data
    domains = ["fitness", "pharmacogenetics", "nutrition"] if domain is None else [domain]
    domain_data = []
    
    for d in domains:
        if d in summary.get("domains", {}):
            domain_info = summary["domains"][d]
            domain_data.append({
                "domain": d.capitalize(),
                "variants": domain_info.get("relevant_variants", 0),
                "impact_score": domain_info.get("impact_score", 0)
            })
    
    if not domain_data:
        raise ValueError(f"No data available for domain summary visualization")
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame(domain_data)
    
    # Plot impact scores
    bars = ax.bar(df["domain"], df["impact_score"], color=sns.color_palette("viridis", len(df)))
    
    # Add variant counts as text
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row["impact_score"] + 0.1, 
                f"{row['variants']} variants", 
                ha='center', va='bottom', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel("Domain", fontsize=12)
    ax.set_ylabel("Impact Score", fontsize=12)
    ax.set_title("Domain Impact Summary", fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Tight layout
    fig.tight_layout()
    
    return fig


def visualize_variant_impact(results: Dict, domain: Optional[str] = None, top: int = 10,
                             figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """Generate variant impact visualization.
    
    Args:
        results: Analysis results
        domain: Optional domain to filter results
        top: Number of top variants to include
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure object
    """
    variants = results["variants"]
    
    # Filter variants by domain if specified
    if domain is not None:
        filtered_variants = [v for v in variants if domain in v.get("domain_scores", {})]
    else:
        filtered_variants = variants
    
    if not filtered_variants:
        raise ValueError(f"No variants found for impact visualization")
    
    # Calculate overall impact score for each variant
    for variant in filtered_variants:
        if domain is not None:
            variant["overall_score"] = variant.get("domain_scores", {}).get(domain, {}).get("score", 0)
        else:
            scores = [ds.get("score", 0) for ds in variant.get("domain_scores", {}).values()]
            variant["overall_score"] = sum(scores) / len(scores) if scores else 0
    
    # Sort variants by overall score and get top N
    top_variants = sorted(filtered_variants, key=lambda v: v["overall_score"], reverse=True)[:top]
    
    # Create variant labels
    variant_labels = []
    for v in top_variants:
        gene = v.get("functional_impact", {}).get("gene_name", "Unknown")
        variant_id = v.get("id", "Unknown")
        variant_labels.append(f"{gene}: {variant_id}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot scores
    scores = [v["overall_score"] for v in top_variants]
    bars = ax.barh(variant_labels, scores, color=sns.color_palette("viridis", len(top_variants)))
    
    # Set labels and title
    title = f"Top {top} Variant Impact Scores"
    if domain:
        title += f" ({domain.capitalize()})"
    
    ax.set_xlabel("Impact Score", fontsize=12)
    ax.set_ylabel("Variant", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Tight layout
    fig.tight_layout()
    
    return fig


def visualize_gene_scores(results: Dict, domain: Optional[str] = None, top: int = 10,
                         figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """Generate gene scores visualization.
    
    Args:
        results: Analysis results
        domain: Optional domain to filter results
        top: Number of top genes to include
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure object
    """
    variants = results["variants"]
    
    # Group variants by gene
    gene_variants = {}
    for variant in variants:
        gene = variant.get("functional_impact", {}).get("gene_name")
        if not gene:
            continue
        
        # Skip if we're filtering by domain and this variant doesn't have that domain
        if domain is not None and domain not in variant.get("domain_scores", {}):
            continue
        
        if gene not in gene_variants:
            gene_variants[gene] = []
        
        gene_variants[gene].append(variant)
    
    if not gene_variants:
        raise ValueError(f"No gene data found for visualization")
    
    # Calculate gene scores
    gene_scores = {}
    for gene, variants in gene_variants.items():
        total_score = 0
        for variant in variants:
            if domain is not None:
                score = variant.get("domain_scores", {}).get(domain, {}).get("score", 0)
            else:
                scores = [ds.get("score", 0) for ds in variant.get("domain_scores", {}).values()]
                score = sum(scores) / len(scores) if scores else 0
            
            total_score += score
        
        gene_scores[gene] = total_score / len(variants) if variants else 0
    
    # Get top genes
    top_genes = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)[:top]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot scores
    gene_names = [g[0] for g in top_genes]
    scores = [g[1] for g in top_genes]
    
    bars = ax.barh(gene_names, scores, color=sns.color_palette("viridis", len(top_genes)))
    
    # Set labels and title
    title = f"Top {top} Gene Impact Scores"
    if domain:
        title += f" ({domain.capitalize()})"
    
    ax.set_xlabel("Impact Score", fontsize=12)
    ax.set_ylabel("Gene", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Tight layout
    fig.tight_layout()
    
    return fig


def visualize_trait_distribution(results: Dict, top: int = 10,
                                figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """Generate trait distribution visualization for fitness domain.
    
    Args:
        results: Analysis results
        top: Number of top traits to include
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure object
    """
    if "fitness" not in results.get("domains", {}):
        raise ValueError("No fitness domain data found for trait visualization")
    
    fitness_data = results["domains"]["fitness"]
    
    # Count traits across all variants
    trait_counts = {}
    
    # Check if we have a traits section directly
    if "traits" in fitness_data:
        for trait, info in fitness_data["traits"].items():
            trait_counts[trait] = info.get("relevant_variants", 0)
    else:
        # Otherwise, extract from variants
        for variant in results["variants"]:
            if "fitness" in variant.get("domain_scores", {}):
                traits = variant["domain_scores"]["fitness"].get("relevant_traits", [])
                for trait in traits:
                    trait_counts[trait] = trait_counts.get(trait, 0) + 1
    
    if not trait_counts:
        raise ValueError("No trait data found for visualization")
    
    # Get top traits
    top_traits = sorted(trait_counts.items(), key=lambda x: x[1], reverse=True)[:top]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot trait counts
    trait_names = [t[0] for t in top_traits]
    counts = [t[1] for t in top_traits]
    
    bars = ax.barh(trait_names, counts, color=sns.color_palette("viridis", len(top_traits)))
    
    # Set labels and title
    ax.set_xlabel("Number of Relevant Variants", fontsize=12)
    ax.set_ylabel("Fitness Trait", fontsize=12)
    ax.set_title(f"Top {top} Fitness Traits by Variant Count", fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Tight layout
    fig.tight_layout()
    
    return fig


def visualize_drug_interactions(results: Dict, top: int = 10,
                               figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """Generate drug interactions visualization for pharmacogenetics domain.
    
    Args:
        results: Analysis results
        top: Number of top drugs to include
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure object
    """
    if "pharmacogenetics" not in results.get("domains", {}):
        raise ValueError("No pharmacogenetics domain data found for drug visualization")
    
    pharma_data = results["domains"]["pharmacogenetics"]
    
    # Count drug interactions
    drug_scores = {}
    drug_counts = {}
    
    # Check if we have a drugs section directly
    if "drugs" in pharma_data:
        for drug, info in pharma_data["drugs"].items():
            drug_counts[drug] = info.get("relevant_variants", 0)
            drug_scores[drug] = info.get("interaction_score", 0)
    else:
        # Otherwise, extract from variants
        for variant in results["variants"]:
            if "pharmacogenetics" in variant.get("domain_scores", {}):
                drugs = variant["domain_scores"]["pharmacogenetics"].get("drug_interactions", [])
                for drug_info in drugs:
                    drug = drug_info.get("drug_name", "Unknown")
                    drug_counts[drug] = drug_counts.get(drug, 0) + 1
                    drug_scores[drug] = drug_scores.get(drug, 0) + drug_info.get("score", 0)
        
        # Normalize scores
        for drug in drug_scores:
            if drug_counts[drug] > 0:
                drug_scores[drug] /= drug_counts[drug]
    
    if not drug_counts:
        raise ValueError("No drug interaction data found for visualization")
    
    # Get top drugs by count
    top_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:top]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for plotting
    drug_names = [d[0] for d in top_drugs]
    counts = [d[1] for d in top_drugs]
    scores = [drug_scores.get(d[0], 0) for d in top_drugs]
    
    # Create index for bar positions
    x = np.arange(len(drug_names))
    width = 0.35
    
    # Plot counts and scores
    ax.bar(x - width/2, counts, width, label='Variant Count', color='steelblue')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, scores, width, label='Interaction Score', color='darkorange')
    
    # Set labels and title
    ax.set_xlabel("Drug", fontsize=12)
    ax.set_ylabel("Number of Variants", fontsize=12, color='steelblue')
    ax2.set_ylabel("Interaction Score", fontsize=12, color='darkorange')
    ax.set_title(f"Top {top} Drug Interactions", fontsize=14, fontweight='bold')
    
    # Set x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(drug_names, rotation=45, ha='right')
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Tight layout
    fig.tight_layout()
    
    return fig


def visualize_nutrient_needs(results: Dict, top: int = 10,
                            figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """Generate nutrient needs visualization for nutrition domain.
    
    Args:
        results: Analysis results
        top: Number of top nutrients to include
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure object
    """
    if "nutrition" not in results.get("domains", {}):
        raise ValueError("No nutrition domain data found for nutrient visualization")
    
    nutrition_data = results["domains"]["nutrition"]
    
    # Extract nutrient data
    nutrient_scores = {}
    nutrient_counts = {}
    
    # Check if we have a nutrients section directly
    if "nutrients" in nutrition_data:
        for nutrient, info in nutrition_data["nutrients"].items():
            nutrient_counts[nutrient] = info.get("relevant_variants", 0)
            nutrient_scores[nutrient] = info.get("recommendation_score", 0)
    else:
        # Otherwise, extract from variants
        for variant in results["variants"]:
            if "nutrition" in variant.get("domain_scores", {}):
                nutrients = variant["domain_scores"]["nutrition"].get("nutrient_effects", [])
                for nutrient_info in nutrients:
                    nutrient = nutrient_info.get("nutrient_name", "Unknown")
                    nutrient_counts[nutrient] = nutrient_counts.get(nutrient, 0) + 1
                    nutrient_scores[nutrient] = nutrient_scores.get(nutrient, 0) + nutrient_info.get("effect_score", 0)
        
        # Normalize scores
        for nutrient in nutrient_scores:
            if nutrient_counts[nutrient] > 0:
                nutrient_scores[nutrient] /= nutrient_counts[nutrient]
    
    if not nutrient_counts:
        raise ValueError("No nutrient data found for visualization")
    
    # Get top nutrients by score
    top_nutrients = sorted(nutrient_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for plotting
    nutrient_names = [n[0] for n in top_nutrients]
    scores = [n[1] for n in top_nutrients]
    
    # Determine colors based on score (positive/negative)
    colors = ['green' if s > 0 else 'red' for s in scores]
    
    # Plot nutrient scores
    bars = ax.barh(nutrient_names, scores, color=colors)
    
    # Set labels and title
    ax.set_xlabel("Recommendation Score", fontsize=12)
    ax.set_ylabel("Nutrient", fontsize=12)
    ax.set_title(f"Top {top} Nutrient Recommendations", fontsize=14, fontweight='bold')
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add legend for interpretation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Increased need'),
        Patch(facecolor='red', label='Decreased need')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Tight layout
    fig.tight_layout()
    
    return fig


def visualize_network(results: Dict, domain: Optional[str] = None,
                     figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """Generate network visualization.
    
    Args:
        results: Analysis results
        domain: Optional domain to filter results
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure object
    """
    if "network" not in results:
        raise ValueError("No network data found for visualization")
    
    network_data = results["network"]
    
    # Filter network data by domain if specified
    if domain is not None:
        if domain not in network_data:
            raise ValueError(f"No network data found for domain '{domain}'")
        graph_data = network_data[domain]
    else:
        # Use combined network data if available, otherwise merge
        if "combined" in network_data:
            graph_data = network_data["combined"]
        else:
            # Basic merging of networks - in a real application this would be more sophisticated
            graph_data = {"nodes": [], "edges": []}
            for d in ["fitness", "pharmacogenetics", "nutrition"]:
                if d in network_data:
                    graph_data["nodes"].extend(network_data[d].get("nodes", []))
                    graph_data["edges"].extend(network_data[d].get("edges", []))
    
    # Skip if no network data
    if not graph_data or "nodes" not in graph_data or "edges" not in graph_data:
        raise ValueError("Invalid network data structure")
    
    # Import networkx for network visualization
    try:
        import networkx as nx
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        for node in graph_data["nodes"]:
            G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
        
        # Add edges
        for edge in graph_data["edges"]:
            G.add_edge(edge["source"], edge["target"], 
                      weight=edge.get("weight", 1.0),
                      **{k: v for k, v in edge.items() if k not in ["source", "target"]})
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get node colors based on type if available
        node_colors = []
        for node in graph_data["nodes"]:
            if "type" in node:
                if node["type"] == "gene":
                    node_colors.append("skyblue")
                elif node["type"] == "variant":
                    node_colors.append("salmon")
                elif node["type"] == "trait":
                    node_colors.append("lightgreen")
                elif node["type"] == "drug":
                    node_colors.append("orange")
                elif node["type"] == "nutrient":
                    node_colors.append("purple")
                else:
                    node_colors.append("gray")
            else:
                node_colors.append("gray")
        
        # Calculate node sizes based on degree
        node_sizes = [50 + 10 * G.degree[node["id"]] for node in graph_data["nodes"]]
        
        # Get node positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        # Set title
        title = "Genomic Interaction Network"
        if domain:
            title += f" ({domain.capitalize()})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Turn off axis
        ax.axis('off')
        
        # Create legend for node types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', label='Gene'),
            Patch(facecolor='salmon', label='Variant'),
            Patch(facecolor='lightgreen', label='Trait'),
            Patch(facecolor='orange', label='Drug'),
            Patch(facecolor='purple', label='Nutrient')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Tight layout
        fig.tight_layout()
        
        return fig
    
    except ImportError:
        logger.error("NetworkX package required for network visualization")
        raise ImportError("NetworkX library is required for network visualization")


def run_visualization(args) -> plt.Figure:
    """Run the visualization based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Matplotlib figure object
    """
    # Load results
    results = load_results(args.results_dir)
    
    # Parse figure size
    figsize = parse_figsize(args.figsize)
    
    # Generate appropriate visualization
    if args.type == "domain-summary":
        fig = visualize_domain_summary(results, args.domain, figsize)
    
    elif args.type == "variant-impact":
        fig = visualize_variant_impact(results, args.domain, args.top, figsize)
    
    elif args.type == "gene-scores":
        fig = visualize_gene_scores(results, args.domain, args.top, figsize)
    
    elif args.type == "trait-distribution":
        fig = visualize_trait_distribution(results, args.top, figsize)
    
    elif args.type == "drug-interactions":
        fig = visualize_drug_interactions(results, args.top, figsize)
    
    elif args.type == "nutrient-needs":
        fig = visualize_nutrient_needs(results, args.top, figsize)
    
    elif args.type == "network":
        fig = visualize_network(results, args.domain, figsize)
    
    else:
        raise ValueError(f"Unsupported visualization type: {args.type}")
    
    return fig


def run(args) -> int:
    """Main entry point for the visualize command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Configure logging
        log_level = logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"Generating visualization: {args.type}")
        
        # Generate visualization
        fig = run_visualization(args)
        
        # Save visualization
        output_path = args.output
        if not any(output_path.lower().endswith(ext) for ext in ['.png', '.pdf', '.svg', '.jpg', '.jpeg']):
            output_path += '.png'
            logger.warning(f"No valid extension detected, using default format: {output_path}")
        
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Visualization saved to: {output_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        if log_level == logging.DEBUG:
            import traceback
            logger.debug(traceback.format_exc())
        return 1
