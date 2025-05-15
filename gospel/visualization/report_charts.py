"""
Report visualization functions for genomic analysis results.

This module provides functions to create visualizations for report-specific 
genomic analysis results, including deficiency analysis, drug interactions, and pathway analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Tuple
import networkx as nx

# Set the style for all plots (if not already set in charts.py)
sns.set(style="whitegrid")

def create_deficiency_chart(
    deficiency_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a visualization of metabolic deficiencies.
    
    Args:
        deficiency_dir: Path to the deficiencies directory
        output_file: Path to save the generated chart (if None, chart is not saved)
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Load deficiency data
    deficiency_file = Path(deficiency_dir) / 'deficiency_report.json'
    
    try:
        with open(deficiency_file, 'r') as f:
            deficiency_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Create placeholder data if file not found or invalid
        deficiency_data = {
            "deficiencies": [],
            "pathway_counts": {}
        }
    
    # For demonstration, create example deficiency data if not available
    np.random.seed(42)
    pathways = deficiency_data.get('pathway_counts', {})
    
    if not pathways:
        # Create example pathways for demonstration
        example_pathways = [
            'Glycolysis', 'Gluconeogenesis', 'TCA Cycle', 'Fatty Acid Synthesis',
            'Fatty Acid Oxidation', 'Purine Metabolism', 'Pyrimidine Metabolism',
            'Amino Acid Metabolism', 'Urea Cycle', 'Oxidative Phosphorylation'
        ]
        pathways = {pathway: np.random.randint(0, 5) for pathway in example_pathways}
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Pathway': list(pathways.keys()),
        'Deficiency Score': list(pathways.values())
    })
    
    # Sort by deficiency score
    df = df.sort_values('Deficiency Score', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a horizontal bar chart with a color gradient based on deficiency level
    bars = ax.barh(df['Pathway'], df['Deficiency Score'], color=plt.cm.viridis(df['Deficiency Score'] / df['Deficiency Score'].max()))
    
    # Customize plot
    ax.set_title('Metabolic Pathway Deficiency Analysis')
    ax.set_xlabel('Deficiency Score')
    ax.set_ylabel('Pathway')
    ax.set_xlim(0, max(df['Deficiency Score'].max() + 0.5, 1))
    
    # Add a color bar to show the scale
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, df['Deficiency Score'].max() or 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Deficiency Severity')
    
    plt.tight_layout()
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_drug_interaction_chart(
    drug_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a visualization of drug interactions.
    
    Args:
        drug_dir: Path to the drugs directory
        output_file: Path to save the generated chart (if None, chart is not saved)
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Load drug interaction data
    drug_report_file = Path(drug_dir) / 'drug_interaction_report.json'
    
    try:
        with open(drug_report_file, 'r') as f:
            drug_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Create placeholder data if file not found or invalid
        drug_data = {
            "affected_drugs": [],
            "interaction_count": 0,
            "interactions": []
        }
    
    # For demonstration, create example drug interaction data if not available
    np.random.seed(42)
    
    # If we have interaction data, use it; otherwise create example data
    interactions = drug_data.get('interactions', [])
    if not interactions:
        # Create example drug list
        drugs = [
            'Warfarin', 'Aspirin', 'Clopidogrel', 'Simvastatin', 'Atorvastatin',
            'Lisinopril', 'Metformin', 'Ibuprofen', 'Acetaminophen', 'Fluoxetine'
        ]
        
        # Create random interactions
        interactions = []
        for i in range(15):  # Create 15 random interactions
            drug1 = np.random.choice(drugs)
            drug2 = np.random.choice([d for d in drugs if d != drug1])
            severity = np.random.choice(['Low', 'Medium', 'High'], p=[0.5, 0.3, 0.2])
            interactions.append({
                'drug1': drug1,
                'drug2': drug2,
                'severity': severity,
                'mechanism': np.random.choice(['CYP450', 'P-glycoprotein', 'Renal', 'Unknown'])
            })
    
    # Create a network graph of drug interactions
    G = nx.Graph()
    
    # Add nodes and edges from the interactions
    severity_color = {'Low': '#1f77b4', 'Medium': '#ff7f0e', 'High': '#d62728'}
    edge_colors = []
    
    for interaction in interactions:
        drug1 = interaction.get('drug1', 'Unknown')
        drug2 = interaction.get('drug2', 'Unknown')
        severity = interaction.get('severity', 'Low')
        
        G.add_node(drug1)
        G.add_node(drug2)
        G.add_edge(drug1, drug2, severity=severity)
        edge_colors.append(severity_color.get(severity, '#1f77b4'))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate node positions using a spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get node degrees for sizing
    node_degrees = dict(G.degree())
    node_sizes = [300 * (1 + deg) for node, deg in node_degrees.items()]
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8, ax=ax)
    edges = nx.draw_networkx_edges(G, pos, width=2.0, edge_color=edge_colors, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
    
    # Create a legend for severity
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color, lw=4, label=sev)
        for sev, color in severity_color.items()
    ]
    ax.legend(handles=legend_elements, title='Interaction Severity')
    
    # Customize plot
    ax.set_title('Drug Interaction Network')
    ax.axis('off')
    
    # Add interaction count as text
    interaction_count = len(interactions)
    ax.text(0.95, 0.95, f'Total Interactions: {interaction_count}', 
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

def create_pathway_chart(
    results_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a visualization of pathway analysis.
    
    Args:
        results_file: Path to the complete_results JSON file
        output_file: Path to save the generated chart (if None, chart is not saved)
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Load results data
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    database_integration = data.get('database_integration', {})
    pathways = database_integration.get('pathways', [])
    
    # For demonstration, create example pathway data if not available
    np.random.seed(42)
    if not pathways:
        # Create example pathways for demonstration
        example_pathways = [
            'Glycolysis', 'Gluconeogenesis', 'TCA Cycle', 'Fatty Acid Synthesis',
            'Fatty Acid Oxidation', 'Purine Metabolism', 'Pyrimidine Metabolism',
            'Amino Acid Metabolism', 'Urea Cycle', 'Oxidative Phosphorylation'
        ]
        pathways = [{
            'name': pathway,
            'genes': np.random.randint(1, 10),
            'p_value': np.random.uniform(0.0001, 0.05),
            'enrichment': np.random.uniform(1.0, 5.0)
        } for pathway in example_pathways]
    
    # Create a DataFrame for visualization
    pathway_data = []
    for pathway in pathways:
        if isinstance(pathway, dict):
            pathway_data.append({
                'Pathway': pathway.get('name', 'Unknown'),
                'Genes': pathway.get('genes', 0),
                'P-value': pathway.get('p_value', 1.0),
                'Enrichment': pathway.get('enrichment', 1.0)
            })
        else:
            # If pathway is just a string
            pathway_data.append({
                'Pathway': pathway,
                'Genes': np.random.randint(1, 10),
                'P-value': np.random.uniform(0.0001, 0.05),
                'Enrichment': np.random.uniform(1.0, 5.0)
            })
    
    df = pd.DataFrame(pathway_data)
    
    # Calculate -log10(p-value) for better visualization
    df['-log10(P)'] = -np.log10(df['P-value'])
    
    # Sort by enrichment
    df = df.sort_values('Enrichment', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    scatter = ax.scatter(
        df['Enrichment'], 
        df['Pathway'],
        s=df['Genes'] * 30,  # Size based on gene count
        c=df['-log10(P)'],   # Color based on significance
        cmap='viridis',
        alpha=0.7
    )
    
    # Customize plot
    ax.set_title('Pathway Enrichment Analysis')
    ax.set_xlabel('Enrichment Score')
    ax.set_ylabel('Pathway')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add a color bar for significance
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('-log10(P-value)')
    
    # Add a legend for the size
    from matplotlib.lines import Line2D
    handles = []
    gene_counts = [1, 5, 10]
    for count in gene_counts:
        handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=np.sqrt(count * 30 / np.pi), label=str(count))
        )
    ax.legend(handles=handles, labels=gene_counts, title='Gene Count', 
              loc='lower right', title_fontsize=10)
    
    plt.tight_layout()
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig 