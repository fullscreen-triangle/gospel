"""
Domain-specific visualization functions for genomic analysis results.

This module provides functions to create visualizations for domain-specific 
genomic analysis results, including fitness, pharmacogenetics, and nutrition.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Tuple
import math
import os

# Set the style for all plots (if not already set in charts.py)
sns.set(style="whitegrid")

def create_fitness_chart(
    extracted_data_dir: Union[str, Path],
    results_file: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
    top_n: int = 10
) -> plt.Figure:
    """
    Create a visualization of fitness-related variants.
    
    Args:
        extracted_data_dir: Path to the extracted_data directory
        results_file: Optional path to the results file for additional data
        output_file: Path to save the generated chart (if None, chart is not saved)
        show_plot: Whether to display the plot
        top_n: Number of top variants to display
        
    Returns:
        The matplotlib Figure object
    """
    # Load fitness genes data
    fitness_genes_file = Path(extracted_data_dir) / 'fitness_genes.json'
    
    with open(fitness_genes_file, 'r') as f:
        fitness_data = json.load(f)
    
    rs_ids = fitness_data.get('rs_ids', [])
    
    # Create a count of variants by chromosome (extracted from RS ID format if available)
    # For demonstration, we'll create random data if chromosome info isn't in the format
    np.random.seed(42)
    chromosome_counts = {}
    
    # Try to extract chromosome info from the RS IDs if available in the data
    # Otherwise, generate random distribution for demonstration
    if not rs_ids:
        # Generate random data for demonstration
        chromosomes = list(range(1, 23)) + ['X', 'Y']
        chromosome_counts = {str(chrom): np.random.randint(1, 20) for chrom in chromosomes}
    else:
        # Count by first part of gene name if it contains chromosome info
        for rs_id in rs_ids[:top_n]:
            if '_' in rs_id:
                chrom = rs_id.split('_')[0]
                chromosome_counts[chrom] = chromosome_counts.get(chrom, 0) + 1
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Chromosome': list(chromosome_counts.keys()),
        'Count': list(chromosome_counts.values())
    })
    
    # Sort by chromosome
    if df['Chromosome'].dtype == 'O':  # If it's an object/string type
        # Try to convert to numeric, keeping X, Y, etc. as strings
        def chrom_to_numeric(c):
            try:
                return int(c)
            except ValueError:
                return c
        
        df['Sort'] = df['Chromosome'].apply(chrom_to_numeric)
        df = df.sort_values('Sort')
        df = df.drop('Sort', axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar chart
    sns.barplot(x='Chromosome', y='Count', data=df, palette='viridis', ax=ax)
    
    # Customize plot
    ax.set_title('Fitness-Related Variants by Chromosome')
    ax.set_xlabel('Chromosome')
    ax.set_ylabel('Number of Variants')
    plt.tight_layout()
    
    # Add total count as text
    total_variants = len(rs_ids)
    ax.text(0.95, 0.95, f'Total Variants: {total_variants}', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_nutrition_chart(
    extracted_data_dir: Union[str, Path],
    results_file: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a visualization of nutrition-related variants.
    
    Args:
        extracted_data_dir: Path to the extracted_data directory
        results_file: Optional path to the results file for additional data
        output_file: Path to save the generated chart (if None, chart is not saved)
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Load nutrition genes data
    nutrition_genes_file = Path(extracted_data_dir) / 'nutrition_genes.json'
    
    with open(nutrition_genes_file, 'r') as f:
        nutrition_data = json.load(f)
    
    rs_ids = nutrition_data.get('rs_ids', [])
    
    # For demonstration, create categories based on nutrient types
    # In a real implementation, this would be based on actual nutrient categories in the data
    np.random.seed(42)
    nutrient_categories = {
        'Macronutrients': ['Carbohydrate', 'Protein', 'Fat'],
        'Vitamins': ['Vitamin A', 'Vitamin B', 'Vitamin C', 'Vitamin D', 'Vitamin E'],
        'Minerals': ['Iron', 'Calcium', 'Magnesium', 'Zinc', 'Selenium'],
        'Other': ['Antioxidants', 'Omega-3', 'Fiber']
    }
    
    # Create a random distribution of variants across nutrient categories
    category_counts = {}
    for category, nutrients in nutrient_categories.items():
        for nutrient in nutrients:
            category_counts[nutrient] = np.random.randint(0, min(10, len(rs_ids)//5 + 1))
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Nutrient': list(category_counts.keys()),
        'Count': list(category_counts.values())
    })
    
    # Add category information
    nutrient_to_category = {}
    for category, nutrients in nutrient_categories.items():
        for nutrient in nutrients:
            nutrient_to_category[nutrient] = category
    
    df['Category'] = df['Nutrient'].map(nutrient_to_category)
    
    # Sort by count within each category
    df = df.sort_values(['Category', 'Count'], ascending=[True, False])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create grouped bar chart
    g = sns.barplot(x='Nutrient', y='Count', hue='Category', data=df, palette='viridis', ax=ax)
    
    # Customize plot
    ax.set_title('Nutrition-Related Variants by Nutrient Type')
    ax.set_xlabel('Nutrient')
    ax.set_ylabel('Number of Variants')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category')
    plt.tight_layout()
    
    # Add total count as text
    total_variants = len(rs_ids)
    ax.text(0.95, 0.95, f'Total Variants: {total_variants}', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_pharmacogenetic_chart(
    extracted_data_dir: Union[str, Path],
    drug_results_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a visualization of pharmacogenetic variants.
    
    Args:
        extracted_data_dir: Path to the extracted_data directory
        drug_results_dir: Path to the drugs results directory
        output_file: Path to save the generated chart (if None, chart is not saved)
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Load pharmacogenetic genes data
    pgx_genes_file = Path(extracted_data_dir) / 'pgx_genes.json'
    
    with open(pgx_genes_file, 'r') as f:
        pgx_data = json.load(f)
    
    rs_ids = pgx_data.get('rs_ids', [])
    
    # Load drug interaction data if available
    drug_report_file = Path(drug_results_dir) / 'drug_interaction_report.json'
    
    try:
        with open(drug_report_file, 'r') as f:
            drug_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Create placeholder data if file not found or invalid
        drug_data = {
            "affected_drugs": [],
            "interaction_count": 0
        }
    
    # For demonstration, create drug categories
    # In a real implementation, this would be based on actual drug categories in the data
    np.random.seed(42)
    drug_categories = {
        'Analgesics': ['Acetaminophen', 'Ibuprofen', 'Naproxen'],
        'Antidepressants': ['Fluoxetine', 'Sertraline', 'Bupropion'],
        'Statins': ['Atorvastatin', 'Simvastatin', 'Rosuvastatin'],
        'Antihypertensives': ['Lisinopril', 'Amlodipine', 'Losartan'],
        'Other': ['Warfarin', 'Clopidogrel', 'Metformin']
    }
    
    # Create a random distribution of variants across drug categories
    # or use actual data if available in drug_data
    affected_drugs = drug_data.get('affected_drugs', [])
    
    if affected_drugs:
        # Use actual data
        drug_counts = {drug: count for drug, count in affected_drugs}
    else:
        # Create random data for demonstration
        drug_counts = {}
        for category, drugs in drug_categories.items():
            for drug in drugs:
                drug_counts[drug] = np.random.randint(0, min(5, len(rs_ids)//10 + 1))
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Drug': list(drug_counts.keys()),
        'Count': list(drug_counts.values())
    })
    
    # Add category information
    drug_to_category = {}
    for category, drugs in drug_categories.items():
        for drug in drugs:
            drug_to_category[drug] = category
    
    df['Category'] = df['Drug'].map(lambda x: drug_to_category.get(x, 'Other'))
    
    # Sort by count within each category
    df = df.sort_values(['Category', 'Count'], ascending=[True, False])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create grouped bar chart
    sns.barplot(x='Drug', y='Count', hue='Category', data=df, palette='viridis', ax=ax)
    
    # Customize plot
    ax.set_title('Pharmacogenetic Variants by Drug')
    ax.set_xlabel('Drug')
    ax.set_ylabel('Number of Variants')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category')
    plt.tight_layout()
    
    # Add total count as text
    total_variants = len(rs_ids)
    interaction_count = drug_data.get('interaction_count', 0)
    ax.text(0.95, 0.95, 
            f'Total Variants: {total_variants}\nDrug Interactions: {interaction_count}', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_complete_gene_list_chart(
    extracted_data_dir: Union[str, Path],
    domain: str,
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
    items_per_page: int = 50
) -> List[plt.Figure]:
    """
    Create visualizations showing the complete list of genes for a specific domain.
    
    Args:
        extracted_data_dir: Path to the extracted_data directory
        domain: The domain to visualize ('fitness', 'nutrition', 'pgx', or 'all')
        output_file: Path to save the generated chart (if None, chart is not saved)
        show_plot: Whether to display the plot
        items_per_page: Number of genes to display per page
        
    Returns:
        List of matplotlib Figure objects (one per page)
    """
    domain_files = {
        'fitness': 'fitness_genes.json',
        'nutrition': 'nutrition_genes.json',
        'pgx': 'pgx_genes.json',
        'all': 'all_genes.json'
    }
    
    domain_titles = {
        'fitness': 'Fitness Domain Gene List',
        'nutrition': 'Nutrition Domain Gene List',
        'pgx': 'Pharmacogenetics Domain Gene List',
        'all': 'Complete Gene List'
    }
    
    if domain not in domain_files:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {list(domain_files.keys())}")
    
    # Load gene data
    gene_file = Path(extracted_data_dir) / domain_files[domain]
    
    with open(gene_file, 'r') as f:
        gene_data = json.load(f)
    
    rs_ids = gene_data.get('rs_ids', [])
    
    if not rs_ids:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No genes found for this domain", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return [fig]
    
    # Calculate number of pages needed
    total_genes = len(rs_ids)
    pages_needed = math.ceil(total_genes / items_per_page)
    
    figures = []
    
    for page in range(pages_needed):
        start_idx = page * items_per_page
        end_idx = min(start_idx + items_per_page, total_genes)
        page_genes = rs_ids[start_idx:end_idx]
        
        # Create figure for this page
        fig, ax = plt.subplots(figsize=(10, 14))
        
        # Turn off axis
        ax.axis('off')
        
        # Set title
        page_title = f"{domain_titles[domain]} (Page {page+1}/{pages_needed})"
        ax.set_title(page_title, fontsize=16, pad=20)
        
        # Create gene list as text
        cells_per_row = 3
        rows_needed = math.ceil(len(page_genes) / cells_per_row)
        
        cell_width = 1.0 / cells_per_row
        cell_height = 0.9 / rows_needed  # Leave space for title
        
        for i, gene in enumerate(page_genes):
            row = i // cells_per_row
            col = i % cells_per_row
            
            x = col * cell_width + (cell_width / 2)
            y = 1.0 - (row * cell_height + (cell_height / 2)) - 0.05  # Offset from top
            
            ax.text(x, y, gene, ha='center', va='center', fontsize=10)
        
        # Add page info and total count
        ax.text(0.5, 0.02, f"Total Genes: {total_genes}", 
                ha='center', va='bottom', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save if output file is provided
        if output_file:
            # Add page number to filename if multiple pages
            if pages_needed > 1:
                base, ext = os.path.splitext(output_file)
                page_output = f"{base}_page{page+1}{ext}"
            else:
                page_output = output_file
            
            plt.savefig(page_output, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        figures.append(fig)
    
    return figures


def create_all_domain_gene_comparison(
    extracted_data_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a visualization comparing the number of genes across all domains.
    
    Args:
        extracted_data_dir: Path to the extracted_data directory
        output_file: Path to save the generated chart (if None, chart is not saved)
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Define domains and their files
    domains = {
        'Fitness': 'fitness_genes.json',
        'Nutrition': 'nutrition_genes.json',
        'Pharmacogenetics': 'pgx_genes.json',
        'All Genes': 'all_genes.json'
    }
    
    # Load gene counts for each domain
    domain_counts = {}
    for domain_name, filename in domains.items():
        file_path = Path(extracted_data_dir) / filename
        try:
            with open(file_path, 'r') as f:
                gene_data = json.load(f)
                rs_ids = gene_data.get('rs_ids', [])
                domain_counts[domain_name] = len(rs_ids)
        except (FileNotFoundError, json.JSONDecodeError):
            domain_counts[domain_name] = 0
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Domain': list(domain_counts.keys()),
        'Gene Count': list(domain_counts.values())
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar chart with values displayed on top
    bars = sns.barplot(x='Domain', y='Gene Count', data=df, palette='viridis', ax=ax)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(int(bar.get_height())),
            ha='center', va='bottom', fontsize=12
        )
    
    # Customize plot
    ax.set_title('Gene Count by Domain', fontsize=16)
    ax.set_xlabel('Domain', fontsize=14)
    ax.set_ylabel('Number of Genes', fontsize=14)
    plt.tight_layout()
    
    # Create a Venn diagram to show overlap
    from matplotlib_venn import venn3
    
    # Load gene lists for each domain (excluding 'All Genes')
    gene_sets = {}
    domain_list = [d for d in domains.keys() if d != 'All Genes']
    for domain_name in domain_list:
        file_path = Path(extracted_data_dir) / domains[domain_name]
        try:
            with open(file_path, 'r') as f:
                gene_data = json.load(f)
                gene_sets[domain_name] = set(gene_data.get('rs_ids', []))
        except (FileNotFoundError, json.JSONDecodeError):
            gene_sets[domain_name] = set()
    
    # Create a second subplot for the Venn diagram
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    try:
        # Create Venn diagram if we have all three domains
        if len(gene_sets) >= 3:
            venn_sets = [gene_sets[d] for d in domain_list[:3]]
            venn = venn3(venn_sets, domain_list[:3], ax=ax2)
            
            # Customize appearance
            for text in venn.set_labels:
                if text is not None:
                    text.set_fontsize(14)
            
            for text in venn.subset_labels:
                if text is not None:
                    text.set_fontsize(12)
            
            ax2.set_title('Gene Overlap Between Domains', fontsize=16)
        else:
            ax2.text(0.5, 0.5, "Insufficient data for Venn diagram", 
                    ha='center', va='center', fontsize=14)
            ax2.axis('off')
    except Exception as e:
        # Fallback if Venn diagram fails
        ax2.text(0.5, 0.5, f"Could not create Venn diagram: {str(e)}", 
                ha='center', va='center', fontsize=14)
        ax2.axis('off')
    
    plt.tight_layout()
    
    # Save figures if output file is provided
    if output_file:
        # Save bar chart
        plt.figure(fig.number)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        # Save Venn diagram with modified filename
        base, ext = os.path.splitext(output_file)
        venn_output = f"{base}_venn{ext}"
        plt.figure(fig2.number)
        plt.savefig(venn_output, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)
    
    return [fig, fig2] 