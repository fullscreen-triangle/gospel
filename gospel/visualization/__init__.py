"""
Visualization module for Gospel analysis results.

This module provides visualization tools for various types of genomic analysis results
including genome scoring, network analysis, and domain-specific analyses.
"""

from .charts import (
    create_genome_score_chart,
    create_variant_score_distribution,
    create_centrality_chart,
    create_community_chart,
    create_network_visualization,
    create_network_metrics_comparison,
    create_network_edge_weight_distribution,
    create_network_community_analysis,
    create_complete_network_dashboard
)

from .domain_charts import (
    create_fitness_chart,
    create_nutrition_chart,
    create_pharmacogenetic_chart,
    create_complete_gene_list_chart,
    create_all_domain_gene_comparison
)

from .report_charts import (
    create_deficiency_chart,
    create_drug_interaction_chart,
    create_pathway_chart
)

from .dashboard import create_dashboard, generate_readme_visualizations

__all__ = [
    # Core genomic analysis charts
    'create_genome_score_chart',
    'create_variant_score_distribution',
    
    # Network analysis charts
    'create_centrality_chart',
    'create_community_chart',
    'create_network_visualization',
    'create_network_metrics_comparison',
    'create_network_edge_weight_distribution',
    'create_network_community_analysis',
    'create_complete_network_dashboard',
    
    # Domain-specific charts
    'create_fitness_chart',
    'create_nutrition_chart',
    'create_pharmacogenetic_chart',
    'create_complete_gene_list_chart',
    'create_all_domain_gene_comparison',
    
    # Report charts
    'create_deficiency_chart',
    'create_drug_interaction_chart',
    'create_pathway_chart',
    
    # Dashboard utilities
    'create_dashboard',
    'generate_readme_visualizations'
] 