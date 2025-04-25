"""
Pharmacogenetics domain analysis for Gospel.

This module provides analysis functions for pharmacogenetic genomic insights.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

from ...core.variant import Variant
from .models import PharmacogeneticsDomain, DrugProfile, DrugInteraction, MetabolizerStatus
from .constants import PGX_DRUGS, PGX_VARIANTS, METABOLISM_GENES

logger = logging.getLogger(__name__)


def analyze_variant_drug_interactions(variant: Variant, config: Dict = None) -> Dict:
    """Analyze the pharmacogenetic impact of a single variant.
    
    Args:
        variant: The variant to analyze
        config: Optional configuration parameters
        
    Returns:
        Dictionary of pharmacogenetic scores and drug interactions
    """
    # Create a pharmacogenetics domain instance and analyze the variant
    pgx_domain = PharmacogeneticsDomain(config)
    return pgx_domain.analyze_variant(variant)


def analyze_genotype_drug_response(variants: List[Variant], config: Dict = None) -> Dict:
    """Analyze the pharmacogenetic profile of a complete genotype.
    
    Args:
        variants: List of all variants to analyze
        config: Optional configuration parameters
        
    Returns:
        Complete pharmacogenetic profile with drug summaries and recommendations
    """
    # Create a pharmacogenetics domain instance and analyze all variants
    pgx_domain = PharmacogeneticsDomain(config)
    return pgx_domain.analyze_variants(variants)


def identify_key_pgx_variants(variants: List[Variant]) -> List[Variant]:
    """Identify variants with known pharmacogenetic impacts.
    
    Args:
        variants: List of variants to screen
        
    Returns:
        List of variants with known pharmacogenetic impacts
    """
    key_variants = []
    
    # Check for known pharmacogenetic SNPs
    for variant in variants:
        variant_id = variant.id.lower()
        
        # Check if this is a known PGx SNP
        if variant_id in PGX_VARIANTS or variant_id in [snp.lower() for snp in PGX_VARIANTS]:
            key_variants.append(variant)
            continue
        
        # Check gene-based associations
        gene_name = variant.functional_impact.get("gene_name", "")
        if gene_name:
            from .constants import METABOLISM_GENES, DRUG_TARGET_GENES
            from .constants import DRUG_PATHWAY_GENES, ADVERSE_EFFECT_GENES
            
            if (gene_name in METABOLISM_GENES or 
                gene_name in DRUG_TARGET_GENES or 
                gene_name in DRUG_PATHWAY_GENES or
                gene_name in ADVERSE_EFFECT_GENES):
                key_variants.append(variant)
    
    return key_variants


def predict_drug_metabolizer_status(variants: List[Variant], gene: str) -> MetabolizerStatus:
    """Predict metabolizer status for a specific gene based on variants.
    
    Args:
        variants: List of variants to analyze
        gene: Gene symbol to predict metabolizer status for
        
    Returns:
        Predicted metabolizer status
    """
    # Default to unknown status
    metabolizer_status = MetabolizerStatus.UNKNOWN
    
    # Common star alleles and their functional impact for key genes
    gene_alleles = {
        "CYP2D6": {
            "*1": {"function": "normal", "score": 1.0},
            "*2": {"function": "normal", "score": 1.0},
            "*3": {"function": "none", "score": 0.0},
            "*4": {"function": "none", "score": 0.0},
            "*5": {"function": "none", "score": 0.0},
            "*6": {"function": "none", "score": 0.0},
            "*9": {"function": "decreased", "score": 0.5},
            "*10": {"function": "decreased", "score": 0.3},
            "*17": {"function": "decreased", "score": 0.5},
            "*41": {"function": "decreased", "score": 0.5}
        },
        "CYP2C19": {
            "*1": {"function": "normal", "score": 1.0},
            "*2": {"function": "none", "score": 0.0},
            "*3": {"function": "none", "score": 0.0},
            "*4": {"function": "none", "score": 0.0},
            "*17": {"function": "increased", "score": 1.5}
        },
        "CYP2C9": {
            "*1": {"function": "normal", "score": 1.0},
            "*2": {"function": "decreased", "score": 0.5},
            "*3": {"function": "decreased", "score": 0.2},
            "*5": {"function": "decreased", "score": 0.3},
            "*6": {"function": "none", "score": 0.0}
        },
        "TPMT": {
            "*1": {"function": "normal", "score": 1.0},
            "*2": {"function": "none", "score": 0.0},
            "*3A": {"function": "none", "score": 0.0},
            "*3B": {"function": "none", "score": 0.0},
            "*3C": {"function": "none", "score": 0.0}
        },
        "DPYD": {
            "*1": {"function": "normal", "score": 1.0},
            "*2A": {"function": "none", "score": 0.0},
            "*13": {"function": "none", "score": 0.0}
        }
    }
    
    # If gene is not in our database, return unknown
    if gene not in gene_alleles:
        return metabolizer_status
    
    # Identify alleles present in the variants
    detected_alleles = []
    
    for variant in variants:
        # Skip if no gene information
        gene_name = variant.functional_impact.get("gene_name", "")
        if gene_name != gene:
            continue
        
        # Extract star allele from variant if available
        star_allele = variant.functional_impact.get("star_allele")
        if star_allele and star_allele in gene_alleles[gene]:
            detected_alleles.append(star_allele)
    
    # If no alleles detected, return unknown
    if not detected_alleles:
        return metabolizer_status
    
    # Calculate overall function score based on detected alleles
    # This is a simplified approach - real diplotype analysis is more complex
    total_score = sum(gene_alleles[gene][allele]["function_score"] for allele in detected_alleles)
    total_score /= len(detected_alleles)  # Average score
    
    # Determine metabolizer status based on score
    if total_score > 1.3:
        metabolizer_status = MetabolizerStatus.ULTRARAPID
    elif total_score > 0.8:
        metabolizer_status = MetabolizerStatus.NORMAL
    elif total_score > 0.3:
        metabolizer_status = MetabolizerStatus.INTERMEDIATE
    elif total_score > 0:
        metabolizer_status = MetabolizerStatus.POOR
    
    return metabolizer_status


def generate_drug_recommendations(drug_name: str, metabolizer_status: MetabolizerStatus) -> List[str]:
    """Generate drug-specific recommendations based on metabolizer status.
    
    Args:
        drug_name: Name of the drug
        metabolizer_status: Predicted metabolizer status
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Check if we have specific recommendations for this drug
    if drug_name in PGX_DRUGS:
        drug_info = PGX_DRUGS[drug_name]
        recommendation_logic = drug_info.get("recommendation_logic", {})
        
        # Get status-specific recommendation if available
        if str(metabolizer_status.value) in recommendation_logic:
            recommendations.append(recommendation_logic[str(metabolizer_status.value)])
            
        # Add general recommendation if available
        if "general" in recommendation_logic:
            recommendations.append(recommendation_logic["general"])
    
    # If no specific recommendations, use generic ones
    if not recommendations:
        if metabolizer_status == MetabolizerStatus.POOR:
            recommendations.append(f"Consider dose reduction for {drug_name} due to poor metabolism.")
        elif metabolizer_status == MetabolizerStatus.INTERMEDIATE:
            recommendations.append(f"Monitor response to {drug_name} due to intermediate metabolism.")
        elif metabolizer_status == MetabolizerStatus.ULTRARAPID:
            recommendations.append(f"Standard doses of {drug_name} may be insufficient due to ultrarapid metabolism.")
    
    # Always add this caution
    recommendations.append("Always consult with a healthcare provider before making medication changes.")
    
    return recommendations


def generate_pgx_report(variants: List[Variant], config: Dict = None) -> Dict:
    """Generate a comprehensive pharmacogenetic report from genomic data.
    
    Args:
        variants: List of variants to analyze
        config: Optional configuration parameters
        
    Returns:
        Complete pharmacogenetic report with summaries and recommendations
    """
    # Run complete pharmacogenetic analysis
    pgx_domain = PharmacogeneticsDomain(config)
    analysis_results = pgx_domain.analyze_variants(variants)
    
    # Identify key variants
    key_variants = identify_key_pgx_variants(variants)
    
    # Organize drug data by therapeutic area
    therapeutic_areas = {
        "anticoagulants": ["warfarin", "clopidogrel", "prasugrel"],
        "pain_management": ["codeine", "tramadol", "morphine", "oxycodone"],
        "psychiatry": ["escitalopram", "citalopram", "paroxetine", "fluoxetine", 
                      "amitriptyline", "olanzapine", "risperidone"],
        "cardiology": ["metoprolol", "atorvastatin", "simvastatin", "pravastatin"],
        "oncology": ["tamoxifen", "fluorouracil", "capecitabine"],
        "gastroenterology": ["omeprazole", "pantoprazole"],
        "infectious_disease": ["abacavir", "efavirenz", "voriconazole"],
        "transplant": ["tacrolimus", "cyclosporine"]
    }
    
    # Organize drugs by therapeutic area
    drugs_by_area = {}
    for area, drug_list in therapeutic_areas.items():
        area_drugs = {}
        for drug in drug_list:
            if drug in analysis_results["drugs"]:
                area_drugs[drug] = analysis_results["drugs"][drug]
        
        if area_drugs:
            drugs_by_area[area] = area_drugs
    
    # Extract key metabolism genes and their status
    metabolism_statuses = {}
    for gene in ["CYP2D6", "CYP2C19", "CYP2C9", "DPYD", "TPMT"]:
        status = predict_drug_metabolizer_status(variants, gene)
        if status != MetabolizerStatus.UNKNOWN:
            metabolism_statuses[gene] = status.value
    
    # Generate report metadata
    summary = analysis_results["summary"]
    
    # Assemble the report
    report = {
        "summary": summary,
        "metabolism_genes": metabolism_statuses,
        "therapeutic_areas": drugs_by_area,
        "all_drugs": analysis_results["drugs"],
        "key_variants": [
            {
                "id": v.id,
                "gene": v.functional_impact.get("gene_name", "Unknown"),
                "score": v.domain_scores.get("pharmacogenetics", {}).get("score", 0),
                "drugs": [d["drug_name"] for d in 
                         v.domain_scores.get("pharmacogenetics", {}).get("drug_interactions", [])]
            }
            for v in key_variants[:10]  # Include top 10 key variants
        ]
    }
    
    return report


def analyze_drug_interactions(variants: List[Variant], target_drug: str) -> Dict:
    """Analyze how genetic variants might affect response to a specific drug.
    
    Args:
        variants: List of variants to analyze
        target_drug: Name of the drug to analyze
        
    Returns:
        Detailed analysis of genetic factors affecting drug response
    """
    # Run pharmacogenetic analysis
    pgx_domain = PharmacogeneticsDomain()
    pgx_domain.analyze_variants(variants)
    
    # Check if the drug exists in our profiles
    if target_drug not in pgx_domain.drug_profiles:
        # Run a partial analysis to see if we have any data on this drug
        for gene, gene_info in METABOLISM_GENES.items():
            if target_drug in gene_info.get("affected_drugs", []):
                # We know about this drug, but no relevant variants were found
                return {
                    "drug_name": target_drug,
                    "relevant_variants": [],
                    "relevant_genes": [gene],
                    "metabolizer_status": "normal",
                    "recommendations": [
                        f"No significant genetic variants affecting {target_drug} metabolism were found.",
                        "Standard dosing guidelines should apply."
                    ]
                }
        
        # No data on this drug
        return {
            "drug_name": target_drug,
            "relevant_variants": [],
            "relevant_genes": [],
            "metabolizer_status": "unknown",
            "recommendations": [
                f"No data available for {target_drug} in the pharmacogenetic database.",
                "Follow standard prescribing guidelines."
            ]
        }
    
    # Get the drug profile
    drug_profile = pgx_domain.drug_profiles[target_drug]
    
    # Format variant information
    relevant_variants = []
    for variant in drug_profile.contributing_variants:
        relevant_variants.append({
            "id": variant.id,
            "gene": variant.functional_impact.get("gene_name", "Unknown"),
            "impact": variant.domain_scores.get("pharmacogenetics", {}).get("score", 0),
            "consequence": variant.functional_impact.get("consequence", "Unknown")
        })
    
    # Format gene contribution information
    gene_contributions = {}
    for gene, score in drug_profile.gene_contributions.items():
        gene_contributions[gene] = {
            "score": score,
            "description": METABOLISM_GENES.get(gene, {}).get("description", "Unknown gene function")
        }
    
    # Generate detailed recommendations
    detailed_recommendations = []
    
    # Metabolism-specific recommendations
    if drug_profile.metabolism_score > 0.7:
        if drug_profile.metabolizer_status in [MetabolizerStatus.POOR, MetabolizerStatus.INTERMEDIATE]:
            detailed_recommendations.append(
                f"Reduced metabolism of {target_drug} may lead to increased drug concentrations and risk of adverse effects."
            )
        elif drug_profile.metabolizer_status in [MetabolizerStatus.RAPID, MetabolizerStatus.ULTRARAPID]:
            detailed_recommendations.append(
                f"Increased metabolism of {target_drug} may lead to reduced efficacy at standard doses."
            )
    
    # Efficacy-specific recommendations
    if drug_profile.efficacy_score > 0.7:
        detailed_recommendations.append(
            f"Genetic variants may significantly affect the efficacy of {target_drug}."
        )
    
    # Safety-specific recommendations
    if drug_profile.safety_score > 0.7:
        detailed_recommendations.append(
            f"Genetic variants indicate increased risk of adverse effects with {target_drug}."
        )
    
    # Add recommendations from the drug profile
    detailed_recommendations.extend(drug_profile.recommendations)
    
    return {
        "drug_name": target_drug,
        "metabolizer_status": drug_profile.metabolizer_status.value,
        "metabolism_score": drug_profile.metabolism_score,
        "efficacy_score": drug_profile.efficacy_score,
        "safety_score": drug_profile.safety_score,
        "overall_score": drug_profile.overall_score,
        "relevant_genes": gene_contributions,
        "relevant_variants": relevant_variants,
        "recommendations": detailed_recommendations
    }


def prioritize_pgx_genes_for_testing(variants: List[Variant]) -> List[Dict]:
    """Identify which pharmacogenes should be tested based on partial data.
    
    Args:
        variants: List of variants that have been analyzed
        
    Returns:
        Prioritized list of genes for additional testing
    """
    # Create a set of key pharmacogenes
    key_genes = set(METABOLISM_GENES.keys())
    
    # Add other important PGx genes
    from .constants import DRUG_TARGET_GENES, DRUG_PATHWAY_GENES, ADVERSE_EFFECT_GENES
    key_genes.update(DRUG_TARGET_GENES.keys())
    key_genes.update(ADVERSE_EFFECT_GENES.keys())
    
    # Record which genes we have data for
    covered_genes = set()
    partial_genes = set()
    missing_genes = set()
    
    # Analyze gene coverage
    for variant in variants:
        gene_name = variant.functional_impact.get("gene_name", "")
        if gene_name in key_genes:
            # Check if this is a crucial variant for the gene
            crucial_variant = False
            for snp_id, snp_info in PGX_VARIANTS.items():
                if snp_info["gene"] == gene_name and variant.id == snp_id:
                    crucial_variant = True
                    break
            
            if crucial_variant:
                covered_genes.add(gene_name)
            else:
                partial_genes.add(gene_name)
    
    # Identify completely missing genes
    missing_genes = key_genes - covered_genes - partial_genes
    
    # Prioritize genes for testing
    prioritized_genes = []
    
    # First add missing high-impact genes
    for gene in missing_genes:
        if gene in METABOLISM_GENES and METABOLISM_GENES[gene].get("impact_score", 0) >= 0.85:
            prioritized_genes.append({
                "gene": gene,
                "reason": "High-impact metabolism gene with no data",
                "impact": "high",
                "affected_drugs": METABOLISM_GENES[gene].get("affected_drugs", [])
            })
    
    # Then add partial coverage genes
    for gene in partial_genes:
        if gene in METABOLISM_GENES and METABOLISM_GENES[gene].get("impact_score", 0) >= 0.85:
            prioritized_genes.append({
                "gene": gene,
                "reason": "High-impact metabolism gene with partial data",
                "impact": "high",
                "affected_drugs": METABOLISM_GENES[gene].get("affected_drugs", [])
            })
    
    # Add remaining missing genes
    for gene in missing_genes:
        if gene not in [g["gene"] for g in prioritized_genes]:
            if gene in METABOLISM_GENES:
                prioritized_genes.append({
                    "gene": gene,
                    "reason": "Metabolism gene with no data",
                    "impact": "medium",
                    "affected_drugs": METABOLISM_GENES[gene].get("affected_drugs", [])
                })
            elif gene in ADVERSE_EFFECT_GENES:
                prioritized_genes.append({
                    "gene": gene,
                    "reason": "Adverse effect gene with no data",
                    "impact": "medium",
                    "affected_drugs": ADVERSE_EFFECT_GENES[gene].get("affected_drugs", [])
                })
    
    # Sort by impact
    prioritized_genes.sort(key=lambda x: 0 if x["impact"] == "high" else (1 if x["impact"] == "medium" else 2))
    
    return prioritized_genes[:10]  # Return top 10 priority genes 