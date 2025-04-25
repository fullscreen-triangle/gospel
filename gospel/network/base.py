"""
Base network module for genomic analysis.
"""

import json
import os
import pickle
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

from gospel.utils.gene_utils import normalize_gene_id
from gospel.utils.network_utils import (
    create_gene_network,
    merge_networks,
    save_network,
    load_network,
    calculate_network_metrics,
)


class GeneNetwork:
    """
    Gene network representation and analysis.
    """

    def __init__(self, name: str = "gene_network"):
        """
        Initialize a gene network.

        Args:
            name: Network name
        """
        self.name = name
        self.graph = nx.Graph()
        self.main_genes = set()  # Set of main genes in the network
        self.node_attributes = {}  # Additional node attributes

    def add_gene(
        self,
        gene_id: str,
        interactions: List[str] = None,
        pathways: List[str] = None,
        variants: List[str] = None,
        attributes: Dict = None,
    ) -> None:
        """
        Add a gene to the network.

        Args:
            gene_id: Gene identifier
            interactions: List of interacting genes
            pathways: List of pathways
            variants: List of variants
            attributes: Dictionary of gene attributes
        """
        # Normalize gene ID
        canonical_id = normalize_gene_id(gene_id)
        
        # Create a network for this gene
        gene_network = create_gene_network(
            canonical_id,
            interactions,
            pathways,
            variants,
            attributes,
        )
        
        # Add the gene network to the main graph
        self.graph = nx.compose(self.graph, gene_network)
        
        # Add to main genes
        self.main_genes.add(canonical_id)
        
        # Store attributes
        if attributes:
            self.node_attributes[canonical_id] = attributes

    def add_interaction(
        self,
        gene1_id: str,
        gene2_id: str,
        relationship: str = "interacts_with",
        attributes: Dict = None,
    ) -> None:
        """
        Add an interaction between two genes.

        Args:
            gene1_id: First gene identifier
            gene2_id: Second gene identifier
            relationship: Relationship type
            attributes: Dictionary of edge attributes
        """
        # Normalize gene IDs
        canonical_id1 = normalize_gene_id(gene1_id)
        canonical_id2 = normalize_gene_id(gene2_id)
        
        # Add nodes if they don't exist
        if canonical_id1 not in self.graph:
            self.graph.add_node(canonical_id1, type="gene")
        
        if canonical_id2 not in self.graph:
            self.graph.add_node(canonical_id2, type="gene")
        
        # Add edge
        self.graph.add_edge(
            canonical_id1,
            canonical_id2,
            relationship=relationship,
            **(attributes or {}),
        )

    def add_pathway(
        self,
        gene_id: str,
        pathway_id: str,
        pathway_name: Optional[str] = None,
        attributes: Dict = None,
    ) -> None:
        """
        Add a pathway connection to a gene.

        Args:
            gene_id: Gene identifier
            pathway_id: Pathway identifier
            pathway_name: Pathway name
            attributes: Dictionary of edge attributes
        """
        # Normalize gene ID
        canonical_id = normalize_gene_id(gene_id)
        
        # Add gene node if it doesn't exist
        if canonical_id not in self.graph:
            self.graph.add_node(canonical_id, type="gene")
        
        # Add pathway node if it doesn't exist
        if pathway_id not in self.graph:
            self.graph.add_node(pathway_id, type="pathway", name=pathway_name or pathway_id)
        
        # Add edge
        self.graph.add_edge(
            canonical_id,
            pathway_id,
            relationship="participates_in",
            **(attributes or {}),
        )

    def add_variant(
        self,
        gene_id: str,
        variant_id: str,
        variant_type: Optional[str] = None,
        attributes: Dict = None,
    ) -> None:
        """
        Add a variant connection to a gene.

        Args:
            gene_id: Gene identifier
            variant_id: Variant identifier
            variant_type: Variant type
            attributes: Dictionary of edge attributes
        """
        # Normalize gene ID
        canonical_id = normalize_gene_id(gene_id)
        
        # Add gene node if it doesn't exist
        if canonical_id not in self.graph:
            self.graph.add_node(canonical_id, type="gene")
        
        # Add variant node if it doesn't exist
        if variant_id not in self.graph:
            self.graph.add_node(variant_id, type="variant", variant_type=variant_type)
        
        # Add edge
        self.graph.add_edge(
            canonical_id,
            variant_id,
            relationship="has_variant",
            **(attributes or {}),
        )

    def merge(self, other_network: "GeneNetwork") -> "GeneNetwork":
        """
        Merge another gene network into this one.

        Args:
            other_network: Another GeneNetwork instance

        Returns:
            Self, with the merged network
        """
        # Merge graphs
        self.graph = nx.compose(self.graph, other_network.graph)
        
        # Merge main genes
        self.main_genes.update(other_network.main_genes)
        
        # Merge node attributes
        for node, attrs in other_network.node_attributes.items():
            if node in self.node_attributes:
                # Update existing attributes
                self.node_attributes[node].update(attrs)
            else:
                # Add new attributes
                self.node_attributes[node] = attrs.copy()
        
        return self

    def save(self, output_path: str) -> None:
        """
        Save the network to a file.

        Args:
            output_path: Output file path
        """
        # Create a dictionary representation of the network
        data = {
            "name": self.name,
            "main_genes": list(self.main_genes),
            "node_attributes": self.node_attributes,
        }
        
        # Determine file type and save appropriately
        _, ext = os.path.splitext(output_path)
        
        if ext.lower() == ".pkl":
            # Save as pickle (including the graph)
            data["graph"] = self.graph
            with open(output_path, "wb") as f:
                pickle.dump(data, f)
        else:
            # Save as JSON (graph separately)
            graph_path = output_path + ".graph"
            save_network(self.graph, graph_path)
            
            data["graph_path"] = graph_path
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls, input_path: str) -> "GeneNetwork":
        """
        Load a network from a file.

        Args:
            input_path: Input file path

        Returns:
            GeneNetwork instance
        """
        _, ext = os.path.splitext(input_path)
        
        if ext.lower() == ".pkl":
            # Load from pickle
            with open(input_path, "rb") as f:
                data = pickle.load(f)
                
                instance = cls(name=data.get("name", "gene_network"))
                instance.graph = data.get("graph", nx.Graph())
                instance.main_genes = set(data.get("main_genes", []))
                instance.node_attributes = data.get("node_attributes", {})
        else:
            # Load from JSON
            with open(input_path, "r") as f:
                data = json.load(f)
                
                instance = cls(name=data.get("name", "gene_network"))
                instance.main_genes = set(data.get("main_genes", []))
                instance.node_attributes = data.get("node_attributes", {})
                
                # Load graph from separate file
                graph_path = data.get("graph_path")
                if graph_path and os.path.exists(graph_path):
                    instance.graph = load_network(graph_path)
                else:
                    instance.graph = nx.Graph()
        
        return instance

    def get_subnetwork(self, gene_ids: List[str]) -> "GeneNetwork":
        """
        Get a subnetwork containing only the specified genes and their connections.

        Args:
            gene_ids: List of gene identifiers

        Returns:
            GeneNetwork instance containing the subnetwork
        """
        # Normalize gene IDs
        canonical_ids = [normalize_gene_id(gene_id) for gene_id in gene_ids]
        
        # Create nodes to include
        nodes_to_include = set(canonical_ids)
        
        # Add neighbors of these genes
        for gene_id in canonical_ids:
            if gene_id in self.graph:
                nodes_to_include.update(self.graph.neighbors(gene_id))
        
        # Create subgraph
        subgraph = self.graph.subgraph(nodes_to_include).copy()
        
        # Create new network
        subnetwork = GeneNetwork(name=f"{self.name}_sub")
        subnetwork.graph = subgraph
        subnetwork.main_genes = set(canonical_ids) & self.main_genes
        
        # Copy relevant node attributes
        for node in subnetwork.graph.nodes():
            if node in self.node_attributes:
                subnetwork.node_attributes[node] = self.node_attributes[node].copy()
        
        return subnetwork

    def get_metrics(self) -> Dict:
        """
        Get network metrics.

        Returns:
            Dictionary of network metrics
        """
        return calculate_network_metrics(self.graph)

    def get_gene_degree(self, gene_id: str) -> int:
        """
        Get the degree of a gene.

        Args:
            gene_id: Gene identifier

        Returns:
            Degree of the gene
        """
        canonical_id = normalize_gene_id(gene_id)
        
        if canonical_id in self.graph:
            return self.graph.degree(canonical_id)
        else:
            return 0

    def get_gene_neighbors(self, gene_id: str) -> List[str]:
        """
        Get the neighbors of a gene.

        Args:
            gene_id: Gene identifier

        Returns:
            List of neighbor node IDs
        """
        canonical_id = normalize_gene_id(gene_id)
        
        if canonical_id in self.graph:
            return list(self.graph.neighbors(canonical_id))
        else:
            return []

    def get_gene_centrality(self, gene_id: str) -> Dict:
        """
        Get centrality metrics for a gene.

        Args:
            gene_id: Gene identifier

        Returns:
            Dictionary of centrality metrics
        """
        canonical_id = normalize_gene_id(gene_id)
        
        if canonical_id not in self.graph:
            return {}
        
        centrality = {}
        
        # Degree centrality
        try:
            degree_centrality = nx.degree_centrality(self.graph)
            centrality["degree"] = degree_centrality.get(canonical_id, 0)
        except:
            centrality["degree"] = 0
        
        # Betweenness centrality (for smaller networks)
        if self.graph.number_of_nodes() <= 500:
            try:
                betweenness_centrality = nx.betweenness_centrality(self.graph)
                centrality["betweenness"] = betweenness_centrality.get(canonical_id, 0)
            except:
                centrality["betweenness"] = 0
        
        # Closeness centrality
        try:
            closeness_centrality = nx.closeness_centrality(self.graph)
            centrality["closeness"] = closeness_centrality.get(canonical_id, 0)
        except:
            centrality["closeness"] = 0
        
        return centrality

    def get_shortest_path(self, gene1_id: str, gene2_id: str) -> List[str]:
        """
        Get the shortest path between two genes.

        Args:
            gene1_id: First gene identifier
            gene2_id: Second gene identifier

        Returns:
            List of node IDs in the shortest path
        """
        canonical_id1 = normalize_gene_id(gene1_id)
        canonical_id2 = normalize_gene_id(gene2_id)
        
        try:
            return nx.shortest_path(self.graph, source=canonical_id1, target=canonical_id2)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_common_neighbors(self, gene_ids: List[str]) -> Set[str]:
        """
        Get common neighbors of multiple genes.

        Args:
            gene_ids: List of gene identifiers

        Returns:
            Set of common neighbor node IDs
        """
        if not gene_ids:
            return set()
        
        canonical_ids = [normalize_gene_id(gene_id) for gene_id in gene_ids]
        
        # Filter out genes not in the graph
        valid_ids = [gene_id for gene_id in canonical_ids if gene_id in self.graph]
        
        if not valid_ids:
            return set()
        
        # Initialize with neighbors of the first gene
        common = set(self.graph.neighbors(valid_ids[0]))
        
        # Intersect with neighbors of remaining genes
        for gene_id in valid_ids[1:]:
            common.intersection_update(self.graph.neighbors(gene_id))
        
        return common

    def get_node_types(self) -> Dict[str, int]:
        """
        Get counts of different node types in the network.

        Returns:
            Dictionary mapping node types to counts
        """
        type_counts = {}
        
        for _, data in self.graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        return type_counts

    def get_edge_types(self) -> Dict[str, int]:
        """
        Get counts of different edge types in the network.

        Returns:
            Dictionary mapping edge types to counts
        """
        type_counts = {}
        
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get("relationship", "unknown")
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        
        return type_counts

    def get_network_density(self) -> float:
        """
        Get the density of the network.

        Returns:
            Network density
        """
        return nx.density(self.graph)

    def get_component_sizes(self) -> List[int]:
        """
        Get the sizes of connected components in the network.

        Returns:
            List of component sizes, sorted in descending order
        """
        components = list(nx.connected_components(self.graph))
        return sorted([len(component) for component in components], reverse=True)

    def get_gene_pathways(self, gene_id: str) -> List[str]:
        """
        Get pathways associated with a gene.

        Args:
            gene_id: Gene identifier

        Returns:
            List of pathway IDs
        """
        canonical_id = normalize_gene_id(gene_id)
        
        if canonical_id not in self.graph:
            return []
        
        pathways = []
        
        for neighbor in self.graph.neighbors(canonical_id):
            # Check if the neighbor is a pathway
            if self.graph.nodes[neighbor].get("type") == "pathway":
                pathways.append(neighbor)
        
        return pathways

    def get_gene_variants(self, gene_id: str) -> List[str]:
        """
        Get variants associated with a gene.

        Args:
            gene_id: Gene identifier

        Returns:
            List of variant IDs
        """
        canonical_id = normalize_gene_id(gene_id)
        
        if canonical_id not in self.graph:
            return []
        
        variants = []
        
        for neighbor in self.graph.neighbors(canonical_id):
            # Check if the neighbor is a variant
            if self.graph.nodes[neighbor].get("type") == "variant":
                variants.append(neighbor)
        
        return variants

    def get_gene_interactions(self, gene_id: str) -> List[str]:
        """
        Get genes that interact with a gene.

        Args:
            gene_id: Gene identifier

        Returns:
            List of interacting gene IDs
        """
        canonical_id = normalize_gene_id(gene_id)
        
        if canonical_id not in self.graph:
            return []
        
        interactions = []
        
        for neighbor in self.graph.neighbors(canonical_id):
            # Check if the neighbor is a gene
            if self.graph.nodes[neighbor].get("type") == "gene":
                interactions.append(neighbor)
        
        return interactions

    def __len__(self) -> int:
        """
        Get the number of nodes in the network.

        Returns:
            Number of nodes
        """
        return self.graph.number_of_nodes()

    def __contains__(self, gene_id: str) -> bool:
        """
        Check if a gene is in the network.

        Args:
            gene_id: Gene identifier

        Returns:
            True if the gene is in the network, False otherwise
        """
        canonical_id = normalize_gene_id(gene_id)
        return canonical_id in self.graph

    def __str__(self) -> str:
        """
        Get a string representation of the network.

        Returns:
            String representation
        """
        return f"GeneNetwork(name={self.name}, nodes={len(self.graph)}, edges={self.graph.number_of_edges()}, main_genes={len(self.main_genes)})" 