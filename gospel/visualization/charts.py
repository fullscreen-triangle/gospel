"""
Core visualization functions for genomic analysis results.

This module provides functions to create visualizations for general genomic analysis results,
including genome scoring and network analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Tuple

# Set the style for all plots
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

def create_genome_score_chart(
    results_file: Union[str, Path], 
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a visualization of genome scoring results.
    
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
    
    genome_scoring = data.get('genome_scoring', {})
    total_score = genome_scoring.get('total_score', 0)
    variant_scores = genome_scoring.get('variant_scores', {})
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Gene': list(variant_scores.keys()),
        'Score': list(variant_scores.values())
    })
    
    # Sort by score
    df = df.sort_values('Score', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = sns.barplot(x='Gene', y='Score', data=df, palette='viridis', ax=ax)
    
    # Add total score as text
    ax.text(0.95, 0.95, f'Total Score: {total_score:.2f}', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Customize plot
    ax.set_title('Genome Scoring Results')
    ax.set_xlabel('Gene')
    ax.set_ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_variant_score_distribution(
    results_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a visualization of variant score distribution.
    
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
    
    genome_scoring = data.get('genome_scoring', {})
    summary = genome_scoring.get('summary', {})
    score_distribution = summary.get('score_distribution', {})
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Category': list(score_distribution.keys()),
        'Count': list(score_distribution.values())
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create pie chart
    colors = {'high': '#1e88e5', 'medium': '#ffc107', 'low': '#d81b60'}
    plt.pie(df['Count'], labels=df['Category'], autopct='%1.1f%%', 
            colors=[colors.get(cat.lower(), '#607d8b') for cat in df['Category']],
            textprops={'fontsize': 12}, startangle=90, explode=[0.05] * len(df))
    
    # Add title
    plt.title('Variant Score Distribution')
    plt.axis('equal')
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_centrality_chart(
    results_file: Union[str, Path],
    centrality_type: str = 'degree',
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a visualization of network centrality measures.
    
    Args:
        results_file: Path to the complete_results JSON file
        centrality_type: Type of centrality to visualize ('degree', 'betweenness', or 'eigenvector')
        output_file: Path to save the generated chart (if None, chart is not saved)
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Load results data
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    network_analysis = data.get('network_analysis', {})
    centrality = network_analysis.get('centrality', {})
    
    if centrality_type not in centrality:
        raise ValueError(f"Centrality type '{centrality_type}' not found in results")
    
    centrality_data = centrality[centrality_type]
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Gene': list(centrality_data.keys()),
        'Centrality': list(centrality_data.values())
    })
    
    # Sort by centrality value
    df = df.sort_values('Centrality', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    sns.barplot(x='Centrality', y='Gene', data=df, palette='viridis', ax=ax)
    
    # Customize plot
    ax.set_title(f'Network {centrality_type.capitalize()} Centrality')
    ax.set_xlabel(f'{centrality_type.capitalize()} Centrality')
    ax.set_ylabel('Gene')
    plt.tight_layout()
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_community_chart(
    results_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a visualization of community detection results.
    
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
    
    network_analysis = data.get('network_analysis', {})
    communities = network_analysis.get('communities', {})
    
    # Create a DataFrame for visualization
    community_sizes = {comm: len(genes) for comm, genes in communities.items()}
    df = pd.DataFrame({
        'Community': list(community_sizes.keys()),
        'Size': list(community_sizes.values())
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    sns.barplot(x='Community', y='Size', data=df, palette='viridis', ax=ax)
    
    # Customize plot
    ax.set_title('Gene Communities')
    ax.set_xlabel('Community ID')
    ax.set_ylabel('Number of Genes')
    plt.tight_layout()
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_network_visualization(
    network_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
    node_size_factor: float = 500,
    node_color_map: Optional[Dict[str, str]] = None
) -> plt.Figure:
    """
    Create a visualization of the gene network.
    
    Args:
        network_file: Path to the network file (graphml, nodes/edges CSV, etc.)
        output_file: Path to save the generated chart (if None, chart is not saved)
        show_plot: Whether to display the plot
        node_size_factor: Factor to scale node sizes
        node_color_map: Optional mapping of node types to colors
        
    Returns:
        The matplotlib Figure object
    """
    # Load network data
    if str(network_file).endswith('.graphml'):
        G = nx.read_graphml(network_file)
    else:
        # Assuming separate nodes and edges files
        network_dir = Path(network_file).parent
        nodes_file = network_dir / 'gene_network_nodes.csv'
        edges_file = network_dir / 'gene_network_edges.csv'
        
        # Read nodes and edges
        nodes_df = pd.read_csv(nodes_file)
        edges_df = pd.read_csv(edges_file)
        
        # Create network
        G = nx.Graph()
        for _, row in nodes_df.iterrows():
            G.add_node(row['id'], label=row.get('label', row['id']))
        
        for _, row in edges_df.iterrows():
            G.add_edge(row['source'], row['target'], weight=row.get('weight', 1.0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate node positions using a spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get node degrees for sizing
    node_degrees = dict(G.degree())
    node_sizes = [node_size_factor * (1 + deg) for _, deg in node_degrees.items()]
    
    # Define node colors based on the provided map or default to a single color
    if node_color_map:
        node_colors = [node_color_map.get(node, '#1f77b4') for node in G.nodes()]
    else:
        node_colors = '#1f77b4'
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', ax=ax)
    
    # Customize plot
    ax.set_title('Gene Interaction Network')
    ax.axis('off')
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_network_metrics_comparison(
    results_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a visualization comparing different network metrics for all genes.
    
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
    
    network_analysis = data.get('network_analysis', {})
    centrality = network_analysis.get('centrality', {})
    
    # Extract all centrality measures
    metrics = {}
    for metric_name, metric_data in centrality.items():
        metrics[metric_name] = metric_data
    
    # Create a DataFrame with all genes and their metrics
    all_genes = set()
    for metric_data in metrics.values():
        all_genes.update(metric_data.keys())
    
    # Create DataFrame
    df_data = []
    for gene in all_genes:
        gene_data = {'gene': gene}
        for metric_name, metric_data in metrics.items():
            gene_data[metric_name] = metric_data.get(gene, 0)
        df_data.append(gene_data)
    
    df = pd.DataFrame(df_data)
    
    # Create figure with subplots for each comparison
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)
    
    if n_metrics < 2:
        # Not enough metrics for comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Insufficient metrics for comparison", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
    else:
        # Create a grid of scatter plots for each metric pair
        fig, axes = plt.subplots(n_metrics-1, n_metrics-1, figsize=(12, 12))
        
        # Flatten axes for easier handling if there's only one row
        if n_metrics == 2:
            axes = np.array([[axes]])
        
        # Create scatter plots for each pair of metrics
        for i in range(n_metrics-1):
            for j in range(n_metrics-1):
                ax = axes[i, j]
                
                x_metric = metric_names[j]
                y_metric = metric_names[i+1]
                
                if i == j:
                    # Diagonal: Show histogram of the metric
                    ax.hist(df[y_metric], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_title(f'{y_metric.capitalize()} Distribution')
                else:
                    # Off-diagonal: Show scatter plot between metrics
                    ax.scatter(df[x_metric], df[y_metric], alpha=0.7)
                    ax.set_xlabel(x_metric.capitalize())
                    ax.set_ylabel(y_metric.capitalize())
                    
                    # Add correlation coefficient
                    corr = df[x_metric].corr(df[y_metric])
                    ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes, 
                            ha='left', va='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add title for the figure
    fig.suptitle('Network Metrics Comparison', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_network_edge_weight_distribution(
    network_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a visualization of edge weight distribution in the network.
    
    Args:
        network_dir: Path to the directory containing network files
        output_file: Path to save the generated chart (if None, chart is not saved)
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Try to load the network
    try:
        # Try graphml first
        network_file = Path(network_dir) / 'gene_network.graphml'
        if network_file.exists():
            G = nx.read_graphml(network_file)
        else:
            # Try edges file
            edges_file = Path(network_dir) / 'gene_network_edges.csv'
            if edges_file.exists():
                edges_df = pd.read_csv(edges_file)
                G = nx.Graph()
                for _, row in edges_df.iterrows():
                    G.add_edge(row['source'], row['target'], weight=row.get('weight', 1.0))
            else:
                # Create sample data if no files found
                raise FileNotFoundError("No network files found")
    except (FileNotFoundError, Exception) as e:
        # Create a simple network for demonstration
        G = nx.complete_graph(6)
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.1, 1.0)
    
    # Extract edge weights
    edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram of edge weights
    ax.hist(edge_weights, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Customize plot
    ax.set_title('Edge Weight Distribution in Gene Network')
    ax.set_xlabel('Edge Weight')
    ax.set_ylabel('Frequency')
    
    # Add summary statistics
    weight_mean = np.mean(edge_weights)
    weight_median = np.median(edge_weights)
    weight_std = np.std(edge_weights)
    
    stats_text = (
        f"Mean: {weight_mean:.3f}\n"
        f"Median: {weight_median:.3f}\n"
        f"Std Dev: {weight_std:.3f}\n"
        f"Min: {min(edge_weights):.3f}\n"
        f"Max: {max(edge_weights):.3f}"
    )
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_network_community_analysis(
    results_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a detailed visualization of community structure in the network.
    
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
    
    network_analysis = data.get('network_analysis', {})
    communities = network_analysis.get('communities', {})
    centrality = network_analysis.get('centrality', {})
    
    # Get degree centrality if available
    degree_centrality = centrality.get('degree', {})
    
    # Create a DataFrame with genes, their communities and centrality
    df_data = []
    for community_id, genes in communities.items():
        for gene in genes:
            df_data.append({
                'gene': gene,
                'community': community_id,
                'centrality': degree_centrality.get(gene, 0)
            })
    
    df = pd.DataFrame(df_data)
    
    # Create a figure with multiple plots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Community size bar chart (upper left)
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    community_sizes = df.groupby('community').size()
    community_sizes.sort_values(ascending=False).plot(kind='bar', ax=ax1)
    ax1.set_title('Community Sizes')
    ax1.set_xlabel('Community ID')
    ax1.set_ylabel('Number of Genes')
    
    # 2. Top genes by centrality for each community (upper middle)
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    top_genes = df.sort_values('centrality', ascending=False).groupby('community').head(3)
    sns.barplot(x='centrality', y='gene', hue='community', data=top_genes, ax=ax2)
    ax2.set_title('Top Genes by Centrality in Each Community')
    ax2.set_xlabel('Centrality')
    ax2.set_ylabel('Gene')
    
    # 3. Centrality distribution by community (upper right)
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    sns.boxplot(x='community', y='centrality', data=df, ax=ax3)
    ax3.set_title('Centrality Distribution by Community')
    ax3.set_xlabel('Community ID')
    ax3.set_ylabel('Centrality')
    
    # 4. Network visualization with communities (bottom span)
    ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    
    # Create a network from the communities
    G = nx.Graph()
    
    # Add nodes with community attributes
    for community_id, genes in communities.items():
        for gene in genes:
            G.add_node(gene, community=community_id)
    
    # Add edges (assuming connected genes are in the same community)
    # In a real implementation, you'd use the actual edges
    for community_id, genes in communities.items():
        for i, gene1 in enumerate(genes):
            for gene2 in genes[i+1:]:
                G.add_edge(gene1, gene2)
    
    # Calculate node positions
    pos = nx.spring_layout(G, seed=42)
    
    # Define colors for communities
    community_colors = plt.cm.tab10(np.linspace(0, 1, len(communities)))
    community_colormap = {comm: community_colors[i] for i, comm in enumerate(communities.keys())}
    
    # Color nodes by community
    node_colors = [community_colormap[G.nodes[node]['community']] for node in G.nodes()]
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, ax=ax4)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, ax=ax4)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', ax=ax4)
    
    # Add a legend for communities
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Community {comm}')
        for comm, color in zip(communities.keys(), community_colors)
    ]
    ax4.legend(handles=legend_elements, loc='upper right', title='Communities')
    
    ax4.set_title('Network Community Structure')
    ax4.axis('off')
    
    # Add network summary data
    summary = network_analysis.get('summary', {})
    if summary:
        summary_text = (
            f"Nodes: {summary.get('num_nodes', 'N/A')}\n"
            f"Edges: {summary.get('num_edges', 'N/A')}\n"
            f"Communities: {summary.get('num_communities', 'N/A')}\n"
            f"Density: {summary.get('density', 'N/A'):.3f}\n"
            f"Clustering: {summary.get('avg_clustering', 'N/A'):.3f}"
        )
        ax4.text(0.01, 0.01, summary_text, transform=ax4.transAxes,
                ha='left', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_complete_network_dashboard(
    results_file: Union[str, Path],
    network_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a comprehensive dashboard of all network analysis visualizations.
    
    Args:
        results_file: Path to the complete_results JSON file
        network_dir: Path to the directory containing network files
        output_file: Path to save the generated chart (if None, chart is not saved)
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Load results data
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    network_analysis = data.get('network_analysis', {})
    
    # Create a large figure for the dashboard
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Network visualization
    ax1 = plt.subplot2grid((4, 2), (0, 0))
    try:
        network_file = Path(network_dir) / 'gene_network.graphml'
        if network_file.exists():
            G = nx.read_graphml(network_file)
        else:
            # Fallback to creating a network from communities
            communities = network_analysis.get('communities', {})
            G = nx.Graph()
            for community_id, genes in communities.items():
                for gene in genes:
                    G.add_node(gene, community=community_id)
                for i, gene1 in enumerate(genes):
                    for gene2 in genes[i+1:]:
                        G.add_edge(gene1, gene2)
        
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(G, pos, node_size=300, node_color='skyblue', width=0.5, 
                        font_size=8, alpha=0.8, ax=ax1)
        ax1.set_title('Gene Interaction Network')
        ax1.axis('off')
    except Exception as e:
        ax1.text(0.5, 0.5, f"Could not create network visualization: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax1.axis('off')
    
    # 2. Degree centrality bar chart
    ax2 = plt.subplot2grid((4, 2), (0, 1))
    try:
        centrality = network_analysis.get('centrality', {})
        degree_centrality = centrality.get('degree', {})
        
        df = pd.DataFrame({
            'Gene': list(degree_centrality.keys()),
            'Centrality': list(degree_centrality.values())
        }).sort_values('Centrality', ascending=False).head(10)
        
        sns.barplot(x='Centrality', y='Gene', data=df, palette='viridis', ax=ax2)
        ax2.set_title('Top 10 Genes by Degree Centrality')
        ax2.set_xlabel('Degree Centrality')
        ax2.set_ylabel('Gene')
    except Exception as e:
        ax2.text(0.5, 0.5, f"Could not create centrality chart: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax2.axis('off')
    
    # 3. Community sizes
    ax3 = plt.subplot2grid((4, 2), (1, 0))
    try:
        communities = network_analysis.get('communities', {})
        community_sizes = {comm: len(genes) for comm, genes in communities.items()}
        
        df = pd.DataFrame({
            'Community': list(community_sizes.keys()),
            'Size': list(community_sizes.values())
        }).sort_values('Size', ascending=False)
        
        sns.barplot(x='Community', y='Size', data=df, palette='viridis', ax=ax3)
        ax3.set_title('Community Sizes')
        ax3.set_xlabel('Community ID')
        ax3.set_ylabel('Number of Genes')
    except Exception as e:
        ax3.text(0.5, 0.5, f"Could not create community size chart: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax3.axis('off')
    
    # 4. Betweenness centrality bar chart
    ax4 = plt.subplot2grid((4, 2), (1, 1))
    try:
        centrality = network_analysis.get('centrality', {})
        betweenness_centrality = centrality.get('betweenness', {})
        
        df = pd.DataFrame({
            'Gene': list(betweenness_centrality.keys()),
            'Centrality': list(betweenness_centrality.values())
        }).sort_values('Centrality', ascending=False).head(10)
        
        sns.barplot(x='Centrality', y='Gene', data=df, palette='viridis', ax=ax4)
        ax4.set_title('Top 10 Genes by Betweenness Centrality')
        ax4.set_xlabel('Betweenness Centrality')
        ax4.set_ylabel('Gene')
    except Exception as e:
        ax4.text(0.5, 0.5, f"Could not create centrality chart: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax4.axis('off')
    
    # 5. Community network visualization
    ax5 = plt.subplot2grid((4, 2), (2, 0), colspan=2, rowspan=1)
    try:
        communities = network_analysis.get('communities', {})
        
        # Create a network with community attributes
        G = nx.Graph()
        for community_id, genes in communities.items():
            for gene in genes:
                G.add_node(gene, community=community_id)
            for i, gene1 in enumerate(genes):
                for gene2 in genes[i+1:]:
                    G.add_edge(gene1, gene2)
        
        # Calculate node positions
        pos = nx.spring_layout(G, seed=42)
        
        # Define colors for communities
        community_colors = plt.cm.tab10(np.linspace(0, 1, len(communities)))
        
        # Color nodes by community
        node_colors = [community_colors[int(G.nodes[node]['community'])] 
                      for node in G.nodes()]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, ax=ax5)
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, ax=ax5)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', ax=ax5)
        
        # Add a legend for communities
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, 
                   label=f'Community {comm}')
            for comm, color in zip(communities.keys(), community_colors)
        ]
        ax5.legend(handles=legend_elements, loc='upper right', title='Communities')
        
        ax5.set_title('Community Structure in Gene Network')
        ax5.axis('off')
    except Exception as e:
        ax5.text(0.5, 0.5, f"Could not create community network visualization: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax5.axis('off')
    
    # 6. Network metrics summary
    ax6 = plt.subplot2grid((4, 2), (3, 0))
    try:
        summary = network_analysis.get('summary', {})
        
        metrics = {
            'Number of Nodes': summary.get('num_nodes', 'N/A'),
            'Number of Edges': summary.get('num_edges', 'N/A'),
            'Number of Communities': summary.get('num_communities', 'N/A'),
            'Network Density': f"{summary.get('density', 'N/A'):.4f}",
            'Average Clustering': f"{summary.get('avg_clustering', 'N/A'):.4f}"
        }
        
        # Create a table
        ax6.axis('tight')
        ax6.axis('off')
        table = ax6.table(
            cellText=[[k, v] for k, v in metrics.items()],
            colLabels=['Metric', 'Value'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        ax6.set_title('Network Summary Metrics')
    except Exception as e:
        ax6.text(0.5, 0.5, f"Could not create network metrics summary: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax6.axis('off')
    
    # 7. Eigenvector centrality bar chart
    ax7 = plt.subplot2grid((4, 2), (3, 1))
    try:
        centrality = network_analysis.get('centrality', {})
        eigenvector_centrality = centrality.get('eigenvector', {})
        
        df = pd.DataFrame({
            'Gene': list(eigenvector_centrality.keys()),
            'Centrality': list(eigenvector_centrality.values())
        }).sort_values('Centrality', ascending=False).head(10)
        
        sns.barplot(x='Centrality', y='Gene', data=df, palette='viridis', ax=ax7)
        ax7.set_title('Top 10 Genes by Eigenvector Centrality')
        ax7.set_xlabel('Eigenvector Centrality')
        ax7.set_ylabel('Gene')
    except Exception as e:
        ax7.text(0.5, 0.5, f"Could not create centrality chart: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax7.axis('off')
    
    # Add a title for the entire dashboard
    fig.suptitle('Comprehensive Network Analysis Dashboard', fontsize=20, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig 