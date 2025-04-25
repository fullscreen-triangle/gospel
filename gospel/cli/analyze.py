"""
Analyze command for the Gospel CLI.

This module implements the 'analyze' command, which processes genomic data
and extracts actionable insights across fitness, pharmacogenetics,
and nutrition domains.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from ..core.variant import Variant, VariantProcessor, load_vcf
from ..core.annotation import VariantAnnotator
from ..core.scoring import VariantScorer

logger = logging.getLogger(__name__)


def configure_parser(subparsers) -> argparse.ArgumentParser:
    """Configure the argument parser for the analyze command.
    
    Args:
        subparsers: Subparsers object from the main argument parser
        
    Returns:
        Configured argument parser for the analyze command
    """
    parser = subparsers.add_parser(
        "analyze",
        help="Run genomic analysis pipeline",
        description="Process genomic data and extract actionable insights across fitness, pharmacogenetics, and nutrition domains."
    )
    
    # Input options
    parser.add_argument(
        "--vcf",
        type=str,
        required=True,
        help="Path to the VCF file containing variant data"
    )
    
    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.json",
        help="Path to the configuration file"
    )
    
    # Domain filtering
    parser.add_argument(
        "--domain",
        type=str,
        choices=["fitness", "pharmacogenetics", "nutrition"],
        help="Limit analysis to a specific domain"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Directory to store analysis results"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def load_config(config_path: str) -> Dict:
    """Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def setup_output_directory(output_dir: str) -> None:
    """Set up the output directory.
    
    Args:
        output_dir: Path to the output directory
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory set up: {output_dir}")
    except OSError as e:
        logger.error(f"Error setting up output directory: {e}")
        raise


def save_results(results: Dict, output_dir: str, domain: Optional[str] = None) -> None:
    """Save analysis results to output files.
    
    Args:
        results: Analysis results
        output_dir: Path to the output directory
        domain: Optional domain to filter results
    """
    try:
        # Save summary results
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as summary_file:
            json.dump(results["summary"], summary_file, indent=2)
        logger.info(f"Saved summary results to {summary_path}")
        
        # Save variant results
        variants_path = os.path.join(output_dir, "variants.json")
        variant_data = []
        for variant in results["variants"]:
            # Convert Variant objects to dictionaries
            variant_dict = {
                "id": variant.id,
                "chromosome": variant.chromosome,
                "position": variant.position,
                "reference": variant.reference,
                "alternate": variant.alternate,
                "type": variant.type.value,
                "functional_impact": variant.functional_impact,
                "domain_scores": variant.domain_scores
            }
            
            # Filter by domain if specified
            if domain is None or domain in variant_dict["domain_scores"]:
                variant_data.append(variant_dict)
                
        with open(variants_path, 'w') as variants_file:
            json.dump(variant_data, variants_file, indent=2)
        logger.info(f"Saved variant results to {variants_path}")
        
        # Save domain-specific results
        for domain_name, domain_results in results["domains"].items():
            # Skip if filtering by domain and not the selected domain
            if domain is not None and domain != domain_name:
                continue
                
            domain_path = os.path.join(output_dir, f"{domain_name}.json")
            with open(domain_path, 'w') as domain_file:
                json.dump(domain_results, domain_file, indent=2)
            logger.info(f"Saved {domain_name} results to {domain_path}")
            
        # Save network results
        if "network" in results:
            network_path = os.path.join(output_dir, "network.json")
            with open(network_path, 'w') as network_file:
                json.dump(results["network"], network_file, indent=2)
            logger.info(f"Saved network results to {network_path}")
            
    except (OSError, TypeError) as e:
        logger.error(f"Error saving results: {e}")
        raise


def extract_domain_summary(variants: List[Variant], domain: str) -> Dict:
    """Extract a summary of domain-specific results.
    
    Args:
        variants: List of analyzed variants
        domain: Domain to summarize
        
    Returns:
        Domain summary
    """
    # Count variants by relevance
    relevant_count = 0
    high_impact_count = 0
    total_count = 0
    
    # Track relevant traits/drugs/nutrients
    relevant_items = {}
    
    for variant in variants:
        if domain in variant.domain_scores:
            total_count += 1
            domain_score = variant.domain_scores[domain]
            
            if domain_score.get("score", 0) >= 0.4:
                relevant_count += 1
                
            if domain_score.get("score", 0) >= 0.7:
                high_impact_count += 1
                
            # Track relevant items by type
            if domain == "fitness":
                items_key = "relevant_traits"
            elif domain == "pharmacogenetics":
                items_key = "relevant_drugs"
            elif domain == "nutrition":
                items_key = "relevant_nutrients"
            else:
                items_key = "relevant_items"
                
            for item in domain_score.get(items_key, []):
                if item in relevant_items:
                    relevant_items[item] += 1
                else:
                    relevant_items[item] = 1
    
    # Sort items by count
    sorted_items = sorted(
        relevant_items.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return {
        "total_variants": total_count,
        "relevant_variants": relevant_count,
        "high_impact_variants": high_impact_count,
        "relevant_items": dict(sorted_items[:10])  # Top 10 items
    }


def analyze_variants(args) -> Dict:
    """Run the variant analysis pipeline.
    
    Args:
        args: Command line arguments
        
    Returns:
        Analysis results
    """
    logger.info("Starting variant analysis")
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up output directory
    setup_output_directory(args.output)
    
    # Filter configuration by domain if specified
    domain_filter = args.domain
    if domain_filter:
        logger.info(f"Limiting analysis to {domain_filter} domain")
    
    # Record start time
    start_time = time.time()
    
    # Step 1: Load VCF data
    logger.info(f"Loading VCF data from {args.vcf}")
    genome_data = load_vcf(args.vcf)
    
    # Step 2: Process variants
    logger.info("Processing variants")
    variant_processor = VariantProcessor(config.get("variant_processing", {}))
    variants = variant_processor.process_variants(genome_data)
    
    # Step 3: Annotate variants
    logger.info("Annotating variants")
    annotator = VariantAnnotator(config.get("annotation", {}))
    annotated_variants = annotator.annotate_variants(variants)
    
    # Step 4: Score variants
    logger.info("Scoring variants")
    scorer = VariantScorer(config.get("scoring", {}))
    scored_variants = scorer.score_variants(annotated_variants)
    
    # Step 5: Generate summaries
    logger.info("Generating summaries")
    
    # Overall summary
    summary = {
        "total_variants": len(scored_variants),
        "processing_time": time.time() - start_time,
        "domains": {}
    }
    
    # Domain-specific summaries
    domains = {}
    domain_names = ["fitness", "pharmacogenetics", "nutrition"]
    
    for domain in domain_names:
        # Skip if filtering by domain and not the selected domain
        if domain_filter is not None and domain != domain_filter:
            continue
            
        domain_summary = extract_domain_summary(scored_variants, domain)
        summary["domains"][domain] = domain_summary
        
        # More detailed domain results (e.g., top variants, gene network)
        domains[domain] = {
            "summary": domain_summary,
            "top_variants": []
        }
        
        # Find top variants for this domain
        top_variants = []
        for variant in scored_variants:
            if domain in variant.domain_scores:
                domain_score = variant.domain_scores[domain]
                if domain_score.get("score", 0) >= 0.6:  # High-impact variants
                    top_variants.append({
                        "id": variant.id,
                        "chromosome": variant.chromosome,
                        "position": variant.position,
                        "gene": variant.functional_impact.get("gene_name", ""),
                        "score": domain_score.get("score", 0),
                        "details": domain_score
                    })
        
        # Sort by score and take top 20
        top_variants.sort(key=lambda x: x["score"], reverse=True)
        domains[domain]["top_variants"] = top_variants[:20]
    
    # Step 6: Create gene network (placeholder)
    network = {
        "nodes": [],
        "edges": []
    }
    
    # Return results
    results = {
        "summary": summary,
        "variants": scored_variants,
        "domains": domains,
        "network": network
    }
    
    # Save results
    save_results(results, args.output, domain_filter)
    
    logger.info("Variant analysis complete")
    return results


def run(args) -> int:
    """Run the analyze command.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    try:
        # Configure logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Run analysis
        analyze_variants(args)
        return 0
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=args.verbose)
        return 1
