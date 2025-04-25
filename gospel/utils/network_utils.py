"""
Utility functions for network manipulation and creation.
"""

import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx

from gospel.utils.gene_utils import normalize_gene_id


def create_gene_network(
    gene_id: str, 
    interactions: List[str] = None,
    pathways: List[str] = None,
    variants: List[str] = None,
    attributes: Dict = None
) -> nx.Graph:
    """
    Create a network for a gene.

    Args:
        gene_id: Gene identifier
        interactions: List of interacting genes
        pathways: List of pathways
        variants: List of variants
        attributes: Dictionary of gene attributes

    Returns:
        NetworkX graph representing the gene network
    """
    G = nx.Graph()
    
    # Normalize gene ID
    canonical_id = normalize_gene_id(gene_id)
    
    # Add main gene node
    G.add_node(canonical_id, type='gene', is_main=True, **attributes or {})
    
    # Add variant nodes
    if variants:
        for variant in variants:
            G.add_node(variant, type='variant')
            G.add_edge(canonical_id, variant, relationship='has_variant')
    
    # Add pathway nodes
    if pathways:
        for pathway in pathways:
            pathway_id = pathway if isinstance(pathway, str) else pathway.get('id', 'unknown')
            pathway_name = pathway if isinstance(pathway, str) else pathway.get('name', pathway_id)
            
            G.add_node(pathway_id, type='pathway', name=pathway_name)
            G.add_edge(canonical_id, pathway_id, relationship='participates_in')
    
    # Add interaction nodes
    if interactions:
        for interaction in interactions:
            interaction_id = normalize_gene_id(interaction if isinstance(interaction, str) else interaction.get('gene_id', 'unknown'))
            
            G.add_node(interaction_id, type='gene')
            G.add_edge(canonical_id, interaction_id, relationship='interacts_with')
    
    return G


def merge_networks(networks: List[nx.Graph]) -> nx.Graph:
    """
    Merge multiple gene networks into a single network.

    Args:
        networks: List of NetworkX graphs

    Returns:
        Merged NetworkX graph
    """
    if not networks:
        return nx.Graph()
    
    # Start with the first network
    merged = networks[0].copy()
    
    # Merge in the remaining networks
    for network in networks[1:]:
        # Add nodes from the current network
        for node, attrs in network.nodes(data=True):
            if node not in merged:
                merged.add_node(node, **attrs)
            else:
                # Merge attributes for existing nodes
                for key, value in attrs.items():
                    if key not in merged.nodes[node]:
                        merged.nodes[node][key] = value
        
        # Add edges from the current network
        for u, v, attrs in network.edges(data=True):
            if not merged.has_edge(u, v):
                merged.add_edge(u, v, **attrs)
            else:
                # Merge attributes for existing edges
                for key, value in attrs.items():
                    if key not in merged.edges[u, v]:
                        merged.edges[u, v][key] = value
    
    return merged


def save_network(network: nx.Graph, output_path: str) -> None:
    """
    Save a network to a file.

    Args:
        network: NetworkX graph
        output_path: Output file path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Determine file type from extension
    _, ext = os.path.splitext(output_path)
    
    if ext.lower() == '.pkl':
        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(network, f)
    elif ext.lower() == '.json':
        # Save as JSON
        data = {
            'nodes': [{'id': n, **network.nodes[n]} for n in network.nodes()],
            'edges': [{'source': u, 'target': v, **network.edges[u, v]} for u, v in network.edges()]
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif ext.lower() == '.graphml':
        # Save as GraphML
        nx.write_graphml(network, output_path)
    elif ext.lower() == '.gexf':
        # Save as GEXF
        nx.write_gexf(network, output_path)
    else:
        # Default to pickle
        with open(output_path, 'wb') as f:
            pickle.dump(network, f)


def load_network(input_path: str) -> nx.Graph:
    """
    Load a network from a file.

    Args:
        input_path: Input file path

    Returns:
        NetworkX graph
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Network file not found: {input_path}")
    
    # Determine file type from extension
    _, ext = os.path.splitext(input_path)
    
    if ext.lower() == '.pkl':
        # Load from pickle
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    elif ext.lower() == '.json':
        # Load from JSON
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        G = nx.Graph()
        
        # Add nodes
        for node in data.get('nodes', []):
            node_id = node.pop('id')
            G.add_node(node_id, **node)
        
        # Add edges
        for edge in data.get('edges', []):
            source = edge.pop('source')
            target = edge.pop('target')
            G.add_edge(source, target, **edge)
        
        return G
    elif ext.lower() == '.graphml':
        # Load from GraphML
        return nx.read_graphml(input_path)
    elif ext.lower() == '.gexf':
        # Load from GEXF
        return nx.read_gexf(input_path)
    else:
        # Try to load as pickle (default)
        try:
            with open(input_path, 'rb') as f:
                return pickle.load(f)
        except:
            raise ValueError(f"Unsupported network file format: {ext}")


def find_shortest_path(network: nx.Graph, source: str, target: str) -> List[str]:
    """
    Find the shortest path between two nodes in a network.

    Args:
        network: NetworkX graph
        source: Source node
        target: Target node

    Returns:
        List of nodes in the shortest path
    """
    try:
        return nx.shortest_path(network, source=source, target=target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def find_common_neighbors(network: nx.Graph, nodes: List[str]) -> Set[str]:
    """
    Find common neighbors of multiple nodes in a network.

    Args:
        network: NetworkX graph
        nodes: List of nodes

    Returns:
        Set of common neighbors
    """
    if not nodes:
        return set()
    
    # Initialize with neighbors of the first node
    common = set(network.neighbors(nodes[0]))
    
    # Intersect with neighbors of remaining nodes
    for node in nodes[1:]:
        common.intersection_update(network.neighbors(node))
    
    return common


def calculate_network_metrics(network: nx.Graph) -> Dict:
    """
    Calculate various metrics for a network.

    Args:
        network: NetworkX graph

    Returns:
        Dictionary of network metrics
    """
    metrics = {
        'node_count': network.number_of_nodes(),
        'edge_count': network.number_of_edges(),
        'density': nx.density(network),
        'connected_components': nx.number_connected_components(network),
    }
    
    # Add degree metrics
    degrees = [d for _, d in network.degree()]
    if degrees:
        metrics['avg_degree'] = sum(degrees) / len(degrees)
        metrics['max_degree'] = max(degrees)
    
    # Add centrality metrics (for small-to-medium networks)
    if network.number_of_nodes() <= 1000:
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(network)
            metrics['max_degree_centrality'] = max(degree_centrality.values()) if degree_centrality else 0
            metrics['avg_degree_centrality'] = sum(degree_centrality.values()) / len(degree_centrality) if degree_centrality else 0
            
            # Betweenness centrality (computationally expensive for large networks)
            if network.number_of_nodes() <= 500:
                betweenness_centrality = nx.betweenness_centrality(network)
                metrics['max_betweenness_centrality'] = max(betweenness_centrality.values()) if betweenness_centrality else 0
                metrics['avg_betweenness_centrality'] = sum(betweenness_centrality.values()) / len(betweenness_centrality) if betweenness_centrality else 0
        except:
            # Skip centrality metrics if there's an error
            pass
    
    return metrics


def create_reaction_network(reactions: List[Dict]) -> nx.DiGraph:
    """
    Create a directed network representing biochemical reactions.

    Args:
        reactions: List of reaction dictionaries

    Returns:
        Directed NetworkX graph representing the reaction network
    """
    G = nx.DiGraph()
    
    for reaction in reactions:
        reaction_id = reaction.get('id', f"reaction_{len(G.nodes)}")
        
        # Add reaction node
        G.add_node(reaction_id, 
                 type='reaction',
                 name=reaction.get('name', ''),
                 enzyme=reaction.get('enzyme', ''),
                 reversible=reaction.get('reversible', False))
        
        # Add substrate nodes and edges
        for substrate in reaction.get('substrates', []):
            substrate_id = substrate if isinstance(substrate, str) else substrate.get('id', substrate)
            
            if substrate_id not in G:
                G.add_node(substrate_id, type='metabolite')
            
            # Substrate -> Reaction edge
            G.add_edge(substrate_id, reaction_id, type='substrate')
        
        # Add product nodes and edges
        for product in reaction.get('products', []):
            product_id = product if isinstance(product, str) else product.get('id', product)
            
            if product_id not in G:
                G.add_node(product_id, type='metabolite')
            
            # Reaction -> Product edge
            G.add_edge(reaction_id, product_id, type='product')
        
        # Add enzyme as a node if provided
        enzyme = reaction.get('enzyme')
        if enzyme:
            enzyme_id = normalize_gene_id(enzyme)
            
            if enzyme_id not in G:
                G.add_node(enzyme_id, type='enzyme')
            
            # Enzyme -> Reaction edge
            G.add_edge(enzyme_id, reaction_id, type='catalyzes')
    
    return G


def get_network_statistics(network: nx.Graph) -> Dict:
    """
    Get comprehensive statistics for a network.

    Args:
        network: NetworkX graph

    Returns:
        Dictionary of network statistics
    """
    stats = {
        'nodes': network.number_of_nodes(),
        'edges': network.number_of_edges(),
    }
    
    # Node type counts
    node_types = defaultdict(int)
    for node, attrs in network.nodes(data=True):
        node_type = attrs.get('type', 'unknown')
        node_types[node_type] += 1
    
    stats['node_types'] = dict(node_types)
    
    # Edge type counts
    edge_types = defaultdict(int)
    for u, v, attrs in network.edges(data=True):
        edge_type = attrs.get('relationship', attrs.get('type', 'unknown'))
        edge_types[edge_type] += 1
    
    stats['edge_types'] = dict(edge_types)
    
    # Basic network statistics
    stats['density'] = nx.density(network)
    
    if nx.is_connected(network):
        stats['diameter'] = nx.diameter(network)
        stats['average_shortest_path_length'] = nx.average_shortest_path_length(network)
    else:
        # For disconnected graphs, calculate for the largest connected component
        largest_cc = max(nx.connected_components(network), key=len)
        subgraph = network.subgraph(largest_cc)
        
        stats['largest_component_size'] = len(largest_cc)
        stats['largest_component_diameter'] = nx.diameter(subgraph)
        stats['largest_component_avg_path'] = nx.average_shortest_path_length(subgraph)
    
    # Centrality for the top 10 nodes
    if network.number_of_nodes() <= 1000:
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(network)
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            stats['top_degree_centrality'] = {node: round(value, 4) for node, value in top_degree}
            
            # Betweenness centrality (for smaller networks)
            if network.number_of_nodes() <= 500:
                betweenness = nx.betweenness_centrality(network)
                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
                stats['top_betweenness_centrality'] = {node: round(value, 4) for node, value in top_betweenness}
        except:
            # Skip if there's an error
            pass
    
    return stats 