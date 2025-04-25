"""
Query command for the Gospel CLI.

This module implements the 'query' command, which retrieves insights from
genomic data analysis across fitness, pharmacogenetics, and nutrition domains.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def configure_parser(subparsers) -> argparse.ArgumentParser:
    """Configure the argument parser for the query command.
    
    Args:
        subparsers: Subparsers object from the main argument parser
        
    Returns:
        Configured argument parser for the query command
    """
    parser = subparsers.add_parser(
        "query",
        help="Query genomic insights",
        description="Retrieve insights from genomic data analysis across fitness, pharmacogenetics, and nutrition domains."
    )
    
    # Input options
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/",
        help="Directory containing analysis results"
    )
    
    # Query type
    parser.add_argument(
        "--type",
        type=str,
        choices=["variant", "gene", "trait", "drug", "nutrient"],
        default="variant",
        help="Type of entity to query"
    )
    
    # Entity identifier
    parser.add_argument(
        "--id",
        type=str,
        help="Identifier for the entity to query (e.g., variant ID, gene symbol)"
    )
    
    # Domain filtering
    parser.add_argument(
        "--domain",
        type=str,
        choices=["fitness", "pharmacogenetics", "nutrition"],
        help="Limit query to a specific domain"
    )
    
    # Output options
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format"
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


def query_variant(variant_id: str, results: Dict, domain: Optional[str] = None) -> Dict:
    """Query information about a specific variant.
    
    Args:
        variant_id: Variant identifier
        results: Analysis results
        domain: Optional domain to filter results
        
    Returns:
        Variant information
    """
    # Find the variant
    variant = None
    for v in results["variants"]:
        if v["id"] == variant_id:
            variant = v
            break
    
    if variant is None:
        raise ValueError(f"Variant '{variant_id}' not found")
    
    # Filter by domain if specified
    if domain is not None:
        if domain not in variant["domain_scores"]:
            raise ValueError(f"Variant '{variant_id}' not found in domain '{domain}'")
        
        # Return only domain-specific information
        return {
            "id": variant["id"],
            "chromosome": variant["chromosome"],
            "position": variant["position"],
            "reference": variant["reference"],
            "alternate": variant["alternate"],
            "type": variant["type"],
            "functional_impact": variant["functional_impact"],
            "domain_scores": {domain: variant["domain_scores"][domain]}
        }
    
    # Return all variant information
    return variant


def query_gene(gene_symbol: str, results: Dict, domain: Optional[str] = None) -> Dict:
    """Query information about a specific gene.
    
    Args:
        gene_symbol: Gene symbol
        results: Analysis results
        domain: Optional domain to filter results
        
    Returns:
        Gene information
    """
    # Find variants associated with the gene
    gene_variants = []
    for variant in results["variants"]:
        gene_name = variant["functional_impact"].get("gene_name", "").upper()
        if gene_name == gene_symbol.upper():
            # Filter by domain if specified
            if domain is not None:
                if domain in variant["domain_scores"]:
                    gene_variants.append(variant)
            else:
                gene_variants.append(variant)
    
    if not gene_variants:
        raise ValueError(f"Gene '{gene_symbol}' not found")
    
    # Sort variants by position
    gene_variants.sort(key=lambda v: v["position"])
    
    # Calculate total impact score across domains
    total_score = 0
    domain_scores = {}
    
    for variant in gene_variants:
        for d, score_data in variant["domain_scores"].items():
            if domain is None or d == domain:
                score = score_data.get("score", 0)
                total_score += score
                
                if d in domain_scores:
                    domain_scores[d] += score
                else:
                    domain_scores[d] = score
    
    # Normalize domain scores
    num_variants = len(gene_variants)
    for d in domain_scores:
        domain_scores[d] /= num_variants
    
    # Return gene information
    return {
        "gene_symbol": gene_symbol,
        "num_variants": num_variants,
        "total_score": total_score / num_variants if num_variants > 0 else 0,
        "domain_scores": domain_scores,
        "variants": gene_variants
    }


def query_trait(trait: str, results: Dict) -> Dict:
    """Query information about a specific fitness trait.
    
    Args:
        trait: Trait name
        results: Analysis results
        
    Returns:
        Trait information
    """
    # Find variants associated with the trait
    trait_variants = []
    for variant in results["variants"]:
        if "fitness" in variant["domain_scores"]:
            fitness_score = variant["domain_scores"]["fitness"]
            traits = fitness_score.get("relevant_traits", [])
            
            if any(t.lower() == trait.lower() for t in traits):
                trait_variants.append(variant)
    
    if not trait_variants:
        raise ValueError(f"Trait '{trait}' not found")
    
    # Sort variants by impact score
    trait_variants.sort(
        key=lambda v: v["domain_scores"]["fitness"].get("score", 0),
        reverse=True
    )
    
    # Extract genes associated with the trait
    genes = {}
    for variant in trait_variants:
        gene_name = variant["functional_impact"].get("gene_name", "")
        if gene_name:
            if gene_name in genes:
                genes[gene_name] += 1
            else:
                genes[gene_name] = 1
    
    # Sort genes by variant count
    sorted_genes = sorted(
        genes.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Return trait information
    return {
        "trait": trait,
        "num_variants": len(trait_variants),
        "avg_score": sum(v["domain_scores"]["fitness"].get("score", 0) for v in trait_variants) / len(trait_variants),
        "top_genes": dict(sorted_genes[:10]),
        "variants": trait_variants[:20]  # Limit to top 20 variants
    }


def query_drug(drug: str, results: Dict) -> Dict:
    """Query information about a specific drug response.
    
    Args:
        drug: Drug name
        results: Analysis results
        
    Returns:
        Drug information
    """
    # Find variants associated with the drug
    drug_variants = []
    for variant in results["variants"]:
        if "pharmacogenetics" in variant["domain_scores"]:
            pharma_score = variant["domain_scores"]["pharmacogenetics"]
            drugs = pharma_score.get("relevant_drugs", [])
            
            if any(d.lower() == drug.lower() for d in drugs):
                drug_variants.append(variant)
    
    if not drug_variants:
        raise ValueError(f"Drug '{drug}' not found")
    
    # Sort variants by impact score
    drug_variants.sort(
        key=lambda v: v["domain_scores"]["pharmacogenetics"].get("score", 0),
        reverse=True
    )
    
    # Extract genes associated with the drug
    genes = {}
    for variant in drug_variants:
        gene_name = variant["functional_impact"].get("gene_name", "")
        if gene_name:
            if gene_name in genes:
                genes[gene_name] += 1
            else:
                genes[gene_name] = 1
    
    # Sort genes by variant count
    sorted_genes = sorted(
        genes.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Determine response category
    avg_score = sum(v["domain_scores"]["pharmacogenetics"].get("score", 0) for v in drug_variants) / len(drug_variants)
    
    if avg_score >= 0.7:
        response_category = "High impact on drug response"
    elif avg_score >= 0.4:
        response_category = "Moderate impact on drug response"
    else:
        response_category = "Low impact on drug response"
    
    # Return drug information
    return {
        "drug": drug,
        "response_category": response_category,
        "num_variants": len(drug_variants),
        "avg_score": avg_score,
        "top_genes": dict(sorted_genes[:10]),
        "variants": drug_variants[:20]  # Limit to top 20 variants
    }


def query_nutrient(nutrient: str, results: Dict) -> Dict:
    """Query information about a specific nutrient.
    
    Args:
        nutrient: Nutrient identifier
        results: Analysis results
        
    Returns:
        Nutrient information
    """
    # Check if nutrition domain results exist
    if "nutrition" not in results["domains"]:
        raise ValueError("No nutrition analysis results found")
    
    nutrition_results = results["domains"]["nutrition"]
    
    # Check if nutrient needs exist in the results
    if "nutrient_needs" not in nutrition_results:
        raise ValueError("No nutrient information found in nutrition analysis")
    
    # Find the nutrient information
    nutrient_id = nutrient.upper()
    if nutrient_id not in nutrition_results["nutrient_needs"]:
        # Try case insensitive search by name
        for nid, ndata in nutrition_results["nutrient_needs"].items():
            if ndata.get("name", "").lower() == nutrient.lower():
                nutrient_id = nid
                break
                
    if nutrient_id not in nutrition_results["nutrient_needs"]:
        raise ValueError(f"Nutrient '{nutrient}' not found")
    
    nutrient_data = nutrition_results["nutrient_needs"][nutrient_id]
    
    # Find associated variants
    associated_variants = []
    if "significant_variants" in nutrition_results:
        for variant in nutrition_results["significant_variants"]:
            for effect in variant.get("nutrient_effects", []):
                if effect.get("nutrient_id") == nutrient_id:
                    associated_variants.append({
                        "variant_id": variant.get("variant_id"),
                        "gene": variant.get("gene"),
                        "genotype": variant.get("genotype"),
                        "effect_type": effect.get("effect_type"),
                        "magnitude": effect.get("magnitude"),
                        "confidence": effect.get("confidence")
                    })
    
    # Compile nutrient report
    nutrient_report = {
        "id": nutrient_id,
        "name": nutrient_data.get("name", nutrient_id),
        "adjustment_factor": nutrient_data.get("adjustment_factor", 1.0),
        "confidence": nutrient_data.get("confidence", 0),
        "rda": nutrient_data.get("rda", 0),
        "units": nutrient_data.get("units", ""),
        "adjusted_intake": nutrient_data.get("adjusted_intake", 0),
        "food_sources": nutrient_data.get("food_sources", []),
        "supplement_forms": nutrient_data.get("supplement_forms", []),
        "description": nutrient_data.get("description", ""),
        "priority": nutrient_data.get("priority", 0),
        "associated_variants": associated_variants
    }
    
    # Add interpretation
    if nutrient_report["adjustment_factor"] > 1.2:
        nutrient_report["interpretation"] = f"Your genetic profile suggests an increased need for {nutrient_report['name']}."
    elif nutrient_report["adjustment_factor"] < 0.8:
        nutrient_report["interpretation"] = f"Your genetic profile suggests a decreased need for {nutrient_report['name']}."
    else:
        nutrient_report["interpretation"] = f"Your genetic profile suggests a typical need for {nutrient_report['name']}."
    
    # Add recommendations
    recommendations = []
    
    if nutrient_report["adjustment_factor"] > 1.2:
        if nutrient_report["food_sources"]:
            food_list = ", ".join(nutrient_report["food_sources"][:3])
            recommendations.append(f"Consider increasing intake of {nutrient_report['name']} through foods like {food_list}.")
        
        if nutrient_report["supplement_forms"]:
            supp_list = ", ".join(nutrient_report["supplement_forms"][:2])
            recommendations.append(f"Supplementation with {supp_list} may be beneficial.")
    
    elif nutrient_report["adjustment_factor"] < 0.8:
        recommendations.append(f"Your genetic profile suggests you may need less {nutrient_report['name']} than average.")
    
    else:
        if nutrient_report["food_sources"]:
            food_list = ", ".join(nutrient_report["food_sources"][:2])
            recommendations.append(f"Maintain adequate intake through foods like {food_list}.")
    
    nutrient_report["recommendations"] = recommendations
    
    return nutrient_report


def format_result(result: Dict, format_type: str) -> str:
    """Format the query result as a string.
    
    Args:
        result: Query result
        format_type: Format type
        
    Returns:
        Formatted result
    """
    if format_type != "text":
        raise ValueError(f"Unsupported format type: {format_type}")
    
    output = []
    
    # Variant query
    if "id" in result and "domain_scores" in result:
        output.append(f"Variant: {result['id']}")
        output.append(f"Location: {result['chromosome']}:{result['position']}")
        output.append(f"Alleles: {result['reference']}>{result['alternate']}")
        output.append(f"Type: {result['type']}")
        
        impact = result['functional_impact']
        gene_name = impact.get('gene_name', 'Unknown')
        consequence = impact.get('consequence', 'Unknown')
        output.append(f"Gene: {gene_name}")
        output.append(f"Consequence: {consequence}")
        
        output.append("\nDomain Scores:")
        for domain, scores in result["domain_scores"].items():
            output.append(f"- {domain.capitalize()}: {scores.get('score', 0):.2f}")
            
            if domain == "fitness" and "component_scores" in scores:
                components = scores["component_scores"]
                for comp, score in components.items():
                    if score >= 0.2:  # Only show significant components
                        output.append(f"  - {comp}: {score:.2f}")
            
            elif domain == "pharmacogenetics" and "drug_interactions" in scores:
                interactions = scores["drug_interactions"]
                for drug, details in interactions.items():
                    output.append(f"  - {drug}: {details.get('impact', 'Unknown')}")
            
            elif domain == "nutrition" and "relevant_nutrients" in scores:
                nutrients = scores["relevant_nutrients"]
                for nutrient in nutrients:
                    output.append(f"  - {nutrient}")
    
    # Gene query
    elif "gene" in result:
        output.append(f"Gene: {result['gene']}")
        output.append(f"Number of variants: {result['num_variants']}")
        
        if "domain_scores" in result:
            output.append("\nDomain Scores:")
            for domain, score in result["domain_scores"].items():
                output.append(f"- {domain.capitalize()}: {score:.2f}")
        
        output.append("\nTop Variants:")
        for i, variant in enumerate(result["variants"][:5]):
            output.append(f"{i+1}. {variant['id']} ({variant['type']}) - {variant['functional_impact'].get('consequence', 'Unknown')}")
    
    # Trait query
    elif "trait" in result:
        output.append(f"Trait: {result['trait']}")
        output.append(f"Impact category: {result['impact_category']}")
        output.append(f"Number of variants: {result['num_variants']}")
        output.append(f"Average impact score: {result['avg_score']:.2f}")
        
        output.append("\nTop Genes:")
        for gene, count in list(result["top_genes"].items())[:5]:
            output.append(f"- {gene}: {count} variants")
        
        output.append("\nTop Variants:")
        for i, variant in enumerate(result["variants"][:5]):
            score = variant["domain_scores"]["fitness"].get("score", 0)
            output.append(f"{i+1}. {variant['id']} ({variant['functional_impact'].get('gene_name', 'Unknown')}) - Score: {score:.2f}")
    
    # Drug query
    elif "drug" in result:
        output.append(f"Drug: {result['drug']}")
        output.append(f"Response category: {result['response_category']}")
        output.append(f"Number of variants: {result['num_variants']}")
        output.append(f"Average impact score: {result['avg_score']:.2f}")
        
        output.append("\nTop Genes:")
        for gene, count in list(result["top_genes"].items())[:5]:
            output.append(f"- {gene}: {count} variants")
        
        output.append("\nTop Variants:")
        for i, variant in enumerate(result["variants"][:5]):
            score = variant["domain_scores"]["pharmacogenetics"].get("score", 0)
            output.append(f"{i+1}. {variant['id']} ({variant['functional_impact'].get('gene_name', 'Unknown')}) - Score: {score:.2f}")
    
    # Nutrient query (new format)
    elif "id" in result and "name" in result and "adjustment_factor" in result:
        output.append(f"Nutrient: {result['name']} ({result['id']})")
        output.append(f"Description: {result['description']}")

        # RDA and adjusted intake
        if result.get('rda') and result.get('units'):
            output.append(f"Standard daily allowance: {result['rda']} {result['units']}")
            output.append(f"Adjusted intake: {result['adjusted_intake']:.1f} {result['units']} " +
                         f"(Adjustment factor: {result['adjustment_factor']:.2f})")
        
        # Interpretation and confidence
        output.append(f"Interpretation: {result.get('interpretation', '')}")
        output.append(f"Confidence: {result['confidence']:.2f}")
        
        # Recommendations
        if result.get('recommendations'):
            output.append("\nRecommendations:")
            for rec in result['recommendations']:
                output.append(f"- {rec}")
        
        # Food sources
        if result.get('food_sources'):
            output.append("\nFood Sources:")
            for food in result['food_sources'][:5]:  # Show top 5 sources
                output.append(f"- {food}")
        
        # Supplement forms
        if result.get('supplement_forms'):
            output.append("\nSupplement Forms:")
            for supp in result['supplement_forms']:
                output.append(f"- {supp}")
        
        # Associated variants
        if result.get('associated_variants'):
            output.append("\nGenetic Associations:")
            for variant in result['associated_variants']:
                effect = variant.get('effect_type', 'Unknown effect')
                magnitude = variant.get('magnitude', 0)
                output.append(f"- {variant['variant_id']} ({variant['gene']}): {effect} (magnitude: {magnitude:.2f})")
    
    # Old nutrient format (for backward compatibility)
    elif "nutrient" in result:
        output.append(f"Nutrient: {result['nutrient']}")
        output.append(f"Impact category: {result['impact_category']}")
        output.append(f"Number of variants: {result['num_variants']}")
        output.append(f"Average impact score: {result['avg_score']:.2f}")
        
        output.append("\nTop Genes:")
        for gene, count in list(result["top_genes"].items())[:5]:
            output.append(f"- {gene}: {count} variants")
        
        output.append("\nTop Variants:")
        for i, variant in enumerate(result["variants"][:5]):
            score = variant["domain_scores"]["nutrition"].get("score", 0)
            output.append(f"{i+1}. {variant['id']} ({variant['functional_impact'].get('gene_name', 'Unknown')}) - Score: {score:.2f}")
    
    return "\n".join(output)


def run_query(args) -> Union[Dict, str]:
    """Run the query command.
    
    Args:
        args: Command line arguments
        
    Returns:
        Query result (dictionary or formatted string)
    """
    logger.info(f"Running query: type={args.type}, id={args.id}, domain={args.domain}")
    
    # Load results
    results = load_results(args.results_dir)
    
    # Check if results are empty
    if not results["variants"]:
        raise ValueError(f"No results found in {args.results_dir}")
    
    # Run the appropriate query
    if args.type == "variant":
        if not args.id:
            raise ValueError("Variant ID is required for variant queries")
        result = query_variant(args.id, results, args.domain)
    
    elif args.type == "gene":
        if not args.id:
            raise ValueError("Gene symbol is required for gene queries")
        result = query_gene(args.id, results, args.domain)
    
    elif args.type == "trait":
        if not args.id:
            raise ValueError("Trait name is required for trait queries")
        result = query_trait(args.id, results)
    
    elif args.type == "drug":
        if not args.id:
            raise ValueError("Drug name is required for drug queries")
        result = query_drug(args.id, results)
    
    elif args.type == "nutrient":
        if not args.id:
            raise ValueError("Nutrient name is required for nutrient queries")
        result = query_nutrient(args.id, results)
    
    else:
        raise ValueError(f"Unknown query type: {args.type}")
    
    # Format the result
    if args.format == "json":
        return result
    else:
        return format_result(result, args.format)


def run(args) -> int:
    """Run the query command.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    try:
        # Configure logging
        log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Run query
        result = run_query(args)
        
        # Print result
        if isinstance(result, str):
            print(result)
        else:
            print(json.dumps(result, indent=2))
        
        return 0
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=getattr(args, "verbose", False))
        print(f"Error: {e}", file=sys.stderr)
        return 1
