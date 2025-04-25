"""
Network analysis module for genomic networks.
"""

import os
import json
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import io
import base64


class NetworkAnalyzer:
    """
    Analyzes genomic networks using graph theory algorithms.
    """
    
    def __init__(self, output_dir: str = "network_analysis"):
        """
        Initialize a network analyzer.
        
        Args:
            output_dir: Directory to save analysis outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def build_network_from_interactions(
        self, 
        interactions: List[Dict[str, Any]], 
        min_score: float = 0.0,
        include_edges: List[str] = None
    ) -> nx.Graph:
        """
        Build a network from protein-protein interactions.
        
        Args:
            interactions: List of interaction data
            min_score: Minimum confidence score to include (0.0-1.0)
            include_edges: Types of edges to include (None for all)
            
        Returns:
            NetworkX graph object
        """
        G = nx.Graph()
        
        for interaction in interactions:
            gene_a = interaction.get("gene_a")
            gene_b = interaction.get("gene_b")
            score = interaction.get("score", 0)
            edge_type = interaction.get("interaction_type", "")
            
            # Skip if score is below threshold
            if score < min_score:
                continue
            
            # Skip if edge type is not included
            if include_edges and edge_type and edge_type not in include_edges:
                continue
            
            # Add nodes if they don't exist
            if gene_a and not G.has_node(gene_a):
                G.add_node(gene_a)
            
            if gene_b and not G.has_node(gene_b):
                G.add_node(gene_b)
            
            # Add edge with attributes
            if gene_a and gene_b:
                G.add_edge(
                    gene_a, 
                    gene_b, 
                    weight=score,
                    type=edge_type,
                    source=interaction.get("source", "")
                )
        
        return G
    
    def build_network_from_pathways(
        self, 
        pathways: List[Dict[str, Any]],
        min_genes_per_pathway: int = 2
    ) -> nx.Graph:
        """
        Build a network from pathway data where genes in the same pathway are connected.
        
        Args:
            pathways: List of pathway data
            min_genes_per_pathway: Minimum number of genes in a pathway to create connections
            
        Returns:
            NetworkX graph object
        """
        G = nx.Graph()
        
        # Group genes by pathway
        pathway_genes = defaultdict(list)
        
        for pathway in pathways:
            pathway_id = pathway.get("pathway_id")
            gene_id = pathway.get("gene_id")
            
            if pathway_id and gene_id:
                pathway_genes[pathway_id].append(gene_id)
        
        # Create connections between genes in the same pathway
        for pathway_id, genes in pathway_genes.items():
            if len(genes) < min_genes_per_pathway:
                continue
            
            # Add nodes
            for gene in genes:
                if not G.has_node(gene):
                    G.add_node(gene)
            
            # Connect all genes in this pathway
            for i in range(len(genes)):
                for j in range(i+1, len(genes)):
                    G.add_edge(
                        genes[i], 
                        genes[j], 
                        pathway=pathway_id,
                        type="pathway_association"
                    )
        
        return G
    
    def identify_key_genes(self, G: nx.Graph, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Identify key genes in a network using centrality measures.
        
        Args:
            G: NetworkX graph
            top_n: Number of top genes to return
            
        Returns:
            List of key genes with centrality metrics
        """
        result = []
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # Try to calculate eigenvector centrality (may fail for some graphs)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            eigenvector_centrality = {node: 0.0 for node in G.nodes()}
        
        # Combine measures
        genes = list(G.nodes())
        
        for gene in genes:
            result.append({
                "gene_id": gene,
                "degree_centrality": degree_centrality.get(gene, 0.0),
                "betweenness_centrality": betweenness_centrality.get(gene, 0.0),
                "closeness_centrality": closeness_centrality.get(gene, 0.0),
                "eigenvector_centrality": eigenvector_centrality.get(gene, 0.0),
                "degree": G.degree(gene)
            })
        
        # Sort by centrality measures (using a combined score)
        for gene_data in result:
            gene_data["centrality_score"] = (
                gene_data["degree_centrality"] +
                gene_data["betweenness_centrality"] +
                gene_data["closeness_centrality"] +
                gene_data["eigenvector_centrality"]
            )
        
        result.sort(key=lambda x: x["centrality_score"], reverse=True)
        
        return result[:top_n]
    
    def find_communities(self, G: nx.Graph, algorithm: str = "louvain") -> Dict[str, int]:
        """
        Find communities in the network.
        
        Args:
            G: NetworkX graph
            algorithm: Community detection algorithm ("louvain", "label_propagation", "greedy_modularity")
            
        Returns:
            Dictionary mapping node IDs to community IDs
        """
        if algorithm == "louvain":
            try:
                from community import best_partition
                return best_partition(G)
            except ImportError:
                print("Python-louvain package not installed. Using label propagation instead.")
                algorithm = "label_propagation"
        
        if algorithm == "label_propagation":
            communities_generator = nx.algorithms.community.label_propagation_communities(G)
            communities = {node: i for i, comm in enumerate(communities_generator) for node in comm}
            return communities
        
        if algorithm == "greedy_modularity":
            communities_generator = nx.algorithms.community.greedy_modularity_communities(G)
            communities = {node: i for i, comm in enumerate(communities_generator) for node in comm}
            return communities
        
        # Default to label propagation
        communities_generator = nx.algorithms.community.label_propagation_communities(G)
        communities = {node: i for i, comm in enumerate(communities_generator) for node in comm}
        return communities
    
    def analyze_network(self, G: nx.Graph, output_prefix: str = "network") -> Dict[str, Any]:
        """
        Perform comprehensive network analysis.
        
        Args:
            G: NetworkX graph
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "is_connected": nx.is_connected(G),
            "average_clustering": nx.average_clustering(G),
            "density": nx.density(G),
            "avg_shortest_path": float('nan'),  # Will update if connected
            "diameter": float('nan'),  # Will update if connected
            "communities": {},
            "key_genes": [],
            "visualization": ""
        }
        
        # Calculate metrics that require a connected graph
        if results["is_connected"]:
            results["avg_shortest_path"] = nx.average_shortest_path_length(G)
            results["diameter"] = nx.diameter(G)
        else:
            # Use largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            largest_cc_subgraph = G.subgraph(largest_cc)
            results["largest_component_size"] = largest_cc_subgraph.number_of_nodes()
            results["largest_component_avg_path"] = nx.average_shortest_path_length(largest_cc_subgraph)
            results["largest_component_diameter"] = nx.diameter(largest_cc_subgraph)
        
        # Identify communities
        results["communities"] = self.find_communities(G)
        
        # Identify key genes
        results["key_genes"] = self.identify_key_genes(G)
        
        # Generate visualization
        viz_path = os.path.join(self.output_dir, f"{output_prefix}_visualization.png")
        self.visualize_network(G, results["communities"], save_path=viz_path)
        results["visualization_path"] = viz_path
        
        # Save results to file
        result_path = os.path.join(self.output_dir, f"{output_prefix}_analysis.json")
        with open(result_path, 'w') as f:
            # Convert non-serializable parts
            serializable_results = results.copy()
            serializable_results["communities"] = {str(k): v for k, v in results["communities"].items()}
            serializable_results.pop("visualization", None)
            json.dump(serializable_results, f, indent=2)
        
        return results
    
    def visualize_network(
        self, 
        G: nx.Graph, 
        communities: Dict[str, int] = None,
        layout: str = "spring",
        save_path: str = None,
        to_base64: bool = False
    ) -> Optional[str]:
        """
        Visualize a network with optional community coloring.
        
        Args:
            G: NetworkX graph
            communities: Dict mapping nodes to community IDs
            layout: Layout algorithm ("spring", "circular", "kamada_kawai", "spectral")
            save_path: Path to save the visualization image
            to_base64: Whether to return base64 encoded image
            
        Returns:
            Optional Base64 encoded image if to_base64=True
        """
        plt.figure(figsize=(12, 10))
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Set node colors based on communities
        node_colors = ['#1f77b4']  # Default color
        if communities:
            # Get unique community IDs
            community_ids = set(communities.values())
            color_map = plt.cm.get_cmap('tab20', len(community_ids))
            
            # Map nodes to colors
            node_colors = [color_map(communities.get(node, 0)) for node in G.nodes()]
        
        # Set node sizes based on degree
        node_sizes = [20 + 100 * G.degree(node) for node in G.nodes()]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
        
        # Add labels to important nodes (top 10% by degree)
        node_degrees = dict(G.degree())
        threshold = np.percentile(list(node_degrees.values()), 90)
        labels = {node: node for node, degree in node_degrees.items() if degree >= threshold}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        plt.title(f"Network Visualization (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})")
        plt.axis('off')
        
        # Save or return
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if to_base64:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return img_base64
        
        plt.close()
        return None
    
    def compare_networks(self, G1: nx.Graph, G2: nx.Graph, name1: str = "Network 1", name2: str = "Network 2") -> Dict[str, Any]:
        """
        Compare two networks to identify similarities and differences.
        
        Args:
            G1: First NetworkX graph
            G2: Second NetworkX graph
            name1: Name of first network
            name2: Name of second network
            
        Returns:
            Dictionary with comparison results
        """
        results = {
            "networks": {
                name1: {
                    "nodes": G1.number_of_nodes(),
                    "edges": G1.number_of_edges(),
                    "density": nx.density(G1),
                    "avg_clustering": nx.average_clustering(G1)
                },
                name2: {
                    "nodes": G2.number_of_nodes(),
                    "edges": G2.number_of_edges(),
                    "density": nx.density(G2),
                    "avg_clustering": nx.average_clustering(G2)
                }
            },
            "comparison": {
                "common_nodes": [],
                "unique_nodes_1": [],
                "unique_nodes_2": [],
                "common_edges": [],
                "unique_edges_1": [],
                "unique_edges_2": []
            }
        }
        
        # Find common and unique nodes
        nodes1 = set(G1.nodes())
        nodes2 = set(G2.nodes())
        
        common_nodes = nodes1.intersection(nodes2)
        unique_nodes1 = nodes1 - nodes2
        unique_nodes2 = nodes2 - nodes1
        
        results["comparison"]["common_nodes"] = list(common_nodes)
        results["comparison"]["unique_nodes_1"] = list(unique_nodes1)
        results["comparison"]["unique_nodes_2"] = list(unique_nodes2)
        
        # Find common and unique edges
        edges1 = set(e for e in G1.edges())
        edges2 = set(e for e in G2.edges())
        
        # Check both directions for undirected graphs
        edges1_both = edges1.union({(b, a) for a, b in edges1})
        edges2_both = edges2.union({(b, a) for a, b in edges2})
        
        common_edges = edges1_both.intersection(edges2_both)
        unique_edges1 = edges1 - edges2_both
        unique_edges2 = edges2 - edges1_both
        
        results["comparison"]["common_edges"] = [list(e) for e in common_edges]
        results["comparison"]["unique_edges_1"] = [list(e) for e in unique_edges1]
        results["comparison"]["unique_edges_2"] = [list(e) for e in unique_edges2]
        
        # Calculate Jaccard similarity for nodes and edges
        if nodes1 or nodes2:
            results["comparison"]["node_jaccard_similarity"] = len(common_nodes) / len(nodes1.union(nodes2))
        else:
            results["comparison"]["node_jaccard_similarity"] = 0
            
        if edges1 or edges2:
            results["comparison"]["edge_jaccard_similarity"] = len(common_edges) / len(edges1.union(edges2))
        else:
            results["comparison"]["edge_jaccard_similarity"] = 0
        
        return results
    
    def network_time_series_analysis(self, networks: List[nx.Graph], time_points: List[str]) -> Dict[str, Any]:
        """
        Analyze how a network evolves over time.
        
        Args:
            networks: List of networks at different time points
            time_points: Labels for each time point
            
        Returns:
            Dictionary with time series analysis
        """
        if len(networks) != len(time_points):
            raise ValueError("Number of networks must match number of time points")
            
        results = {
            "time_points": time_points,
            "metrics": {
                "nodes": [],
                "edges": [],
                "density": [],
                "avg_clustering": [],
                "key_genes_by_time": {}
            },
            "node_persistence": {},
            "edge_persistence": {}
        }
        
        # Calculate metrics for each time point
        for i, G in enumerate(networks):
            results["metrics"]["nodes"].append(G.number_of_nodes())
            results["metrics"]["edges"].append(G.number_of_edges())
            results["metrics"]["density"].append(nx.density(G))
            results["metrics"]["avg_clustering"].append(nx.average_clustering(G))
            
            # Get key genes at each time point
            key_genes = self.identify_key_genes(G, top_n=5)
            results["metrics"]["key_genes_by_time"][time_points[i]] = key_genes
        
        # Track node persistence (how long nodes stay in the network)
        all_nodes = set()
        for G in networks:
            all_nodes.update(G.nodes())
            
        for node in all_nodes:
            presence = [1 if node in G else 0 for G in networks]
            if sum(presence) > 0:
                results["node_persistence"][node] = {
                    "presence": presence,
                    "persistence_rate": sum(presence) / len(networks)
                }
        
        # Track edge persistence
        all_edges = set()
        for G in networks:
            all_edges.update(G.edges())
            
        for edge in all_edges:
            presence = [1 if edge in G.edges() or (edge[1], edge[0]) in G.edges() else 0 for G in networks]
            if sum(presence) > 0:
                results["edge_persistence"][str(edge)] = {
                    "presence": presence,
                    "persistence_rate": sum(presence) / len(networks)
                }
        
        return results
    
    def analyze_pathway_enrichment(
        self, 
        gene_list: List[str], 
        pathways: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze pathway enrichment for a list of genes.
        
        Args:
            gene_list: List of gene identifiers
            pathways: List of pathway data
            
        Returns:
            List of enriched pathways with statistics
        """
        results = []
        
        # Group genes by pathway
        pathway_genes = defaultdict(list)
        pathway_info = {}
        
        for pathway in pathways:
            pathway_id = pathway.get("pathway_id")
            pathway_name = pathway.get("pathway_name")
            gene_id = pathway.get("gene_id")
            
            if pathway_id and gene_id:
                pathway_genes[pathway_id].append(gene_id)
                
                if pathway_id not in pathway_info and pathway_name:
                    pathway_info[pathway_id] = {
                        "pathway_id": pathway_id,
                        "pathway_name": pathway_name,
                        "source": pathway.get("source", "")
                    }
        
        # Get all unique genes in our pathway database
        all_genes = set()
        for genes in pathway_genes.values():
            all_genes.update(genes)
        
        # Calculate enrichment for each pathway
        gene_set = set(gene_list)
        for pathway_id, pathway_gene_list in pathway_genes.items():
            pathway_gene_set = set(pathway_gene_list)
            
            # Calculate statistics
            n_genes_in_pathway = len(pathway_gene_set)
            n_input_genes = len(gene_set)
            n_pathway_genes_in_input = len(gene_set.intersection(pathway_gene_set))
            n_background_genes = len(all_genes)
            
            # Skip if no genes from this pathway in input
            if n_pathway_genes_in_input == 0:
                continue
            
            # Calculate fold enrichment
            expected = (n_genes_in_pathway / n_background_genes) * n_input_genes
            fold_enrichment = n_pathway_genes_in_input / expected if expected > 0 else float('inf')
            
            # Create result
            pathway_result = {
                "pathway_id": pathway_id,
                "pathway_name": pathway_info.get(pathway_id, {}).get("pathway_name", pathway_id),
                "genes_in_pathway": n_genes_in_pathway,
                "genes_in_input": n_input_genes,
                "overlap_genes": n_pathway_genes_in_input,
                "overlap_gene_ids": list(gene_set.intersection(pathway_gene_set)),
                "fold_enrichment": fold_enrichment,
                "source": pathway_info.get(pathway_id, {}).get("source", "")
            }
            
            results.append(pathway_result)
        
        # Sort by fold enrichment
        results.sort(key=lambda x: x["fold_enrichment"], reverse=True)
        
        return results 