"""
Dashboard creation functions for Gospel analysis results.

This module provides functions to create a comprehensive dashboard of 
visualizations for all Gospel analysis results.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Tuple
import os

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

def create_dashboard(
    results_dir: Union[str, Path],
    output_dir: Union[str, Path],
    timestamp: Optional[str] = None,
    show_plots: bool = False,
    include_all_data: bool = True
) -> Dict[str, str]:
    """
    Create a comprehensive dashboard of visualizations for all Gospel analysis results.
    
    Args:
        results_dir: Path to the directory containing all result files
        output_dir: Path to the directory to save all visualizations
        timestamp: Optional timestamp to use for naming output files
        show_plots: Whether to display the plots
        include_all_data: Whether to include all available data in detailed visualizations
        
    Returns:
        Dictionary mapping chart types to their output file paths
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the latest results file if timestamp is not provided
    if timestamp is None:
        results_files = list(results_dir.glob('complete_results_*.json'))
        if not results_files:
            raise FileNotFoundError("No results files found in the specified directory")
        latest_results_file = max(results_files, key=os.path.getctime)
        timestamp = latest_results_file.stem.split('_', 1)[1]
    else:
        latest_results_file = results_dir / f'complete_results_{timestamp}.json'
    
    # Dictionary to store output file paths
    output_files = {}
    
    # 1. Generate genome scoring visualizations
    output_files['genome_score'] = str(output_dir / f'genome_score_{timestamp}.png')
    create_genome_score_chart(
        results_file=latest_results_file,
        output_file=output_files['genome_score'],
        show_plot=show_plots
    )
    
    output_files['variant_distribution'] = str(output_dir / f'variant_distribution_{timestamp}.png')
    create_variant_score_distribution(
        results_file=latest_results_file,
        output_file=output_files['variant_distribution'],
        show_plot=show_plots
    )
    
    # 2. Generate network analysis visualizations
    # 2.1 Basic centrality charts
    for centrality_type in ['degree', 'betweenness', 'eigenvector']:
        output_files[f'{centrality_type}_centrality'] = str(output_dir / f'{centrality_type}_centrality_{timestamp}.png')
        try:
            create_centrality_chart(
                results_file=latest_results_file,
                centrality_type=centrality_type,
                output_file=output_files[f'{centrality_type}_centrality'],
                show_plot=show_plots
            )
        except (ValueError, KeyError):
            # Skip if this centrality type is not available
            del output_files[f'{centrality_type}_centrality']
    
    # 2.2 Community chart
    output_files['communities'] = str(output_dir / f'communities_{timestamp}.png')
    try:
        create_community_chart(
            results_file=latest_results_file,
            output_file=output_files['communities'],
            show_plot=show_plots
        )
    except (KeyError, ValueError):
        # Skip if communities data is not available
        del output_files['communities']
    
    # 2.3 Network visualization
    network_dir = results_dir.parent / 'networks'
    if network_dir.exists():
        # Basic network visualization
        network_file = network_dir / 'gene_network.graphml'
        if os.path.exists(network_file):
            output_files['network_visualization'] = str(output_dir / f'network_visualization_{timestamp}.png')
            create_network_visualization(
                network_file=network_file,
                output_file=output_files['network_visualization'],
                show_plot=show_plots
            )
        
        # 2.4 Advanced network analysis (if include_all_data is True)
        if include_all_data:
            # Network metrics comparison
            output_files['network_metrics_comparison'] = str(output_dir / f'network_metrics_comparison_{timestamp}.png')
            try:
                create_network_metrics_comparison(
                    results_file=latest_results_file,
                    output_file=output_files['network_metrics_comparison'],
                    show_plot=show_plots
                )
            except Exception as e:
                print(f"Could not create network metrics comparison: {e}")
                del output_files['network_metrics_comparison']
            
            # Edge weight distribution
            output_files['edge_weight_distribution'] = str(output_dir / f'edge_weight_distribution_{timestamp}.png')
            try:
                create_network_edge_weight_distribution(
                    network_dir=network_dir,
                    output_file=output_files['edge_weight_distribution'],
                    show_plot=show_plots
                )
            except Exception as e:
                print(f"Could not create edge weight distribution: {e}")
                del output_files['edge_weight_distribution']
            
            # Community structure analysis
            output_files['community_analysis'] = str(output_dir / f'community_analysis_{timestamp}.png')
            try:
                create_network_community_analysis(
                    results_file=latest_results_file,
                    output_file=output_files['community_analysis'],
                    show_plot=show_plots
                )
            except Exception as e:
                print(f"Could not create community analysis: {e}")
                del output_files['community_analysis']
            
            # Complete network dashboard
            output_files['network_dashboard'] = str(output_dir / f'network_dashboard_{timestamp}.png')
            try:
                create_complete_network_dashboard(
                    results_file=latest_results_file,
                    network_dir=network_dir,
                    output_file=output_files['network_dashboard'],
                    show_plot=show_plots
                )
            except Exception as e:
                print(f"Could not create network dashboard: {e}")
                del output_files['network_dashboard']
    
    # 3. Generate domain-specific visualizations
    extracted_data_dir = results_dir.parent.parent / 'extracted_data'
    if extracted_data_dir.exists():
        # 3.1 Basic domain charts
        # Fitness
        output_files['fitness'] = str(output_dir / f'fitness_{timestamp}.png')
        create_fitness_chart(
            extracted_data_dir=extracted_data_dir,
            results_file=latest_results_file,
            output_file=output_files['fitness'],
            show_plot=show_plots
        )
        
        # Nutrition
        output_files['nutrition'] = str(output_dir / f'nutrition_{timestamp}.png')
        create_nutrition_chart(
            extracted_data_dir=extracted_data_dir,
            results_file=latest_results_file,
            output_file=output_files['nutrition'],
            show_plot=show_plots
        )
        
        # Pharmacogenetics
        drug_dir = results_dir.parent / 'drugs'
        if drug_dir.exists():
            output_files['pharmacogenetics'] = str(output_dir / f'pharmacogenetics_{timestamp}.png')
            create_pharmacogenetic_chart(
                extracted_data_dir=extracted_data_dir,
                drug_results_dir=drug_dir,
                output_file=output_files['pharmacogenetics'],
                show_plot=show_plots
            )
        
        # 3.2 Complete gene lists for each domain (if include_all_data is True)
        if include_all_data:
            # All domains comparison
            output_files['domain_comparison'] = str(output_dir / f'domain_comparison_{timestamp}.png')
            try:
                create_all_domain_gene_comparison(
                    extracted_data_dir=extracted_data_dir,
                    output_file=output_files['domain_comparison'],
                    show_plot=show_plots
                )
            except Exception as e:
                print(f"Could not create domain comparison: {e}")
                del output_files['domain_comparison']
            
            # Complete gene lists for each domain
            for domain in ['fitness', 'nutrition', 'pgx', 'all']:
                output_files[f'{domain}_gene_list'] = str(output_dir / f'{domain}_gene_list_{timestamp}.png')
                try:
                    create_complete_gene_list_chart(
                        extracted_data_dir=extracted_data_dir,
                        domain=domain,
                        output_file=output_files[f'{domain}_gene_list'],
                        show_plot=show_plots
                    )
                except Exception as e:
                    print(f"Could not create {domain} gene list: {e}")
                    del output_files[f'{domain}_gene_list']
    
    # 4. Generate deficiency and drug interaction visualizations
    deficiency_dir = results_dir.parent / 'deficiencies'
    if deficiency_dir.exists():
        output_files['deficiencies'] = str(output_dir / f'deficiencies_{timestamp}.png')
        create_deficiency_chart(
            deficiency_dir=deficiency_dir,
            output_file=output_files['deficiencies'],
            show_plot=show_plots
        )
    
    drug_dir = results_dir.parent / 'drugs'
    if drug_dir.exists():
        output_files['drug_interactions'] = str(output_dir / f'drug_interactions_{timestamp}.png')
        create_drug_interaction_chart(
            drug_dir=drug_dir,
            output_file=output_files['drug_interactions'],
            show_plot=show_plots
        )
    
    # 5. Generate pathway analysis visualization
    output_files['pathways'] = str(output_dir / f'pathways_{timestamp}.png')
    create_pathway_chart(
        results_file=latest_results_file,
        output_file=output_files['pathways'],
        show_plot=show_plots
    )
    
    # 6. Create a comprehensive dashboard with all visualizations
    output_files['dashboard'] = str(output_dir / f'dashboard_{timestamp}.png')
    create_comprehensive_dashboard(
        output_files=output_files,
        output_file=output_files['dashboard'],
        show_plot=show_plots
    )
    
    return output_files

def create_comprehensive_dashboard(
    output_files: Dict[str, str],
    output_file: str,
    show_plot: bool = False
) -> plt.Figure:
    """
    Create a comprehensive dashboard combining all visualizations.
    
    Args:
        output_files: Dictionary mapping chart types to their output file paths
        output_file: Path to save the combined dashboard
        show_plot: Whether to display the plot
        
    Returns:
        The matplotlib Figure object
    """
    # Load all the individual charts
    charts = {}
    for chart_type, file_path in output_files.items():
        if os.path.exists(file_path) and 'gene_list_page' not in file_path and 'dashboard' not in chart_type:
            # Skip the complete gene lists and nested dashboards
            charts[chart_type] = plt.imread(file_path)
    
    # Create a large figure for the dashboard
    fig = plt.figure(figsize=(24, 36))
    
    # Create a grid for the charts
    n_charts = len(charts)
    n_cols = 2
    n_rows = (n_charts + n_cols - 1) // n_cols  # Ceiling division
    
    # Define the layout
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    # Order of charts in the dashboard
    chart_order = [
        'genome_score', 
        'variant_distribution',
        'degree_centrality', 
        'betweenness_centrality',
        'eigenvector_centrality', 
        'communities',
        'network_visualization', 
        'network_metrics_comparison',
        'edge_weight_distribution',
        'community_analysis',
        'fitness',
        'nutrition', 
        'pharmacogenetics',
        'domain_comparison',
        'deficiencies', 
        'drug_interactions',
        'pathways'
    ]
    
    # Add each chart to the dashboard
    row, col = 0, 0
    for chart_type in chart_order:
        if chart_type in charts:
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(charts[chart_type])
            ax.set_title(chart_type.replace('_', ' ').title(), fontsize=16)
            ax.axis('off')
            
            # Move to the next position
            col += 1
            if col >= n_cols:
                col = 0
                row += 1
    
    # Set a title for the entire dashboard
    fig.suptitle('Gospel Genomic Analysis Dashboard', fontsize=24, y=0.98)
    
    # Add some padding
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the dashboard
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def generate_readme_visualizations(
    output_files: Dict[str, str],
    readme_file: Union[str, Path],
    image_prefix: str = ''
) -> str:
    """
    Generate markdown content for README.md showcasing visualizations.
    
    Args:
        output_files: Dictionary mapping chart types to their output file paths
        readme_file: Path to the README.md file to update
        image_prefix: Prefix to add to image paths in markdown (e.g., 'public/visualization/')
        
    Returns:
        The updated README.md content as a string
    """
    # Read the existing README content
    with open(readme_file, 'r') as f:
        readme_content = f.read()
    
    # Create the visualization section
    visualization_section = """
# Visualization Examples

Gospel provides comprehensive visualization of genomic analysis results across multiple domains.

"""
    
    # Order of charts in the README
    chart_order = [
        ('genome_score', 'Genome Scoring Results'),
        ('variant_distribution', 'Variant Score Distribution'),
        ('network_visualization', 'Gene Interaction Network'),
        ('network_dashboard', 'Complete Network Analysis Dashboard'),
        ('fitness', 'Fitness-Related Variants by Chromosome'),
        ('fitness_gene_list', 'Complete Fitness Gene List'),
        ('nutrition', 'Nutrition-Related Variants by Nutrient Type'),
        ('pharmacogenetics', 'Pharmacogenetic Variants by Drug'),
        ('domain_comparison', 'Domain Gene Count Comparison'),
        ('degree_centrality', 'Network Degree Centrality'),
        ('communities', 'Gene Communities'),
        ('deficiencies', 'Metabolic Pathway Deficiency Analysis'),
        ('drug_interactions', 'Drug Interaction Network'),
        ('pathways', 'Pathway Enrichment Analysis')
    ]
    
    # Add each chart to the visualization section
    for chart_type, chart_title in chart_order:
        if chart_type in output_files and os.path.exists(output_files[chart_type]):
            # Get relative path from the repository root
            rel_path = os.path.relpath(output_files[chart_type], os.path.dirname(readme_file))
            # Add the image prefix if provided
            if image_prefix:
                rel_path = f"{image_prefix}{os.path.basename(rel_path)}"
            
            visualization_section += f"""
## {chart_title}

![{chart_title}]({rel_path})

"""
            
            # Special case for gene lists: add links to all pages if there are multiple
            if chart_type.endswith('_gene_list') and chart_type.replace('_gene_list', '') in ['fitness', 'nutrition', 'pgx', 'all']:
                domain = chart_type.replace('_gene_list', '')
                base, ext = os.path.splitext(output_files[chart_type])
                
                # Check if there are multiple pages
                page2_path = f"{base}_page2{ext}"
                if os.path.exists(page2_path):
                    visualization_section += "Additional pages: "
                    page_links = []
                    page_idx = 2
                    while os.path.exists(f"{base}_page{page_idx}{ext}"):
                        if image_prefix:
                            page_rel_path = f"{image_prefix}{os.path.basename(f'{base}_page{page_idx}{ext}')}"
                        else:
                            page_rel_path = os.path.relpath(f"{base}_page{page_idx}{ext}", os.path.dirname(readme_file))
                        
                        page_links.append(f"[Page {page_idx}]({page_rel_path})")
                        page_idx += 1
                    
                    visualization_section += ", ".join(page_links) + "\n\n"
    
    # Add the dashboard as the last item
    dashboard_path = output_files.get('dashboard')
    if dashboard_path and os.path.exists(dashboard_path):
        rel_path = os.path.relpath(dashboard_path, os.path.dirname(readme_file))
        if image_prefix:
            rel_path = f"{image_prefix}{os.path.basename(rel_path)}"
        
        visualization_section += f"""
## Complete Dashboard

![Complete Genomic Analysis Dashboard]({rel_path})
"""
    
    # Check if a visualization section already exists
    vis_start = readme_content.find("# Visualization Examples")
    if vis_start != -1:
        # Find the next section heading
        next_section = readme_content.find("\n# ", vis_start + 1)
        if next_section != -1:
            # Replace the existing visualization section
            readme_content = readme_content[:vis_start] + visualization_section + readme_content[next_section:]
        else:
            # It's the last section, replace until the end
            readme_content = readme_content[:vis_start] + visualization_section
    else:
        # Append the visualization section at the end
        readme_content += visualization_section
    
    return readme_content 