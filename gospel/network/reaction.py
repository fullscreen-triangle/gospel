"""
Reaction network module for metabolic and biochemical pathways.
"""

import json
import os
import pickle
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

from gospel.utils.gene_utils import normalize_gene_id
from gospel.utils.network_utils import (
    create_reaction_network,
    save_network,
    load_network,
)


class ReactionNetwork:
    """
    Reaction network for metabolic and biochemical pathways.
    """

    def __init__(self, name: str = "reaction_network"):
        """
        Initialize a reaction network.

        Args:
            name: Network name
        """
        self.name = name
        self.graph = nx.DiGraph()  # Directed graph for reactions
        self.reactions = {}  # Map of reaction IDs to reaction data
        self.metabolites = {}  # Map of metabolite IDs to metabolite data
        self.enzymes = {}  # Map of enzyme IDs to enzyme data
    
    def add_reaction(
        self,
        reaction_id: str,
        substrates: List[str],
        products: List[str],
        enzyme: Optional[str] = None,
        reversible: bool = False,
        attributes: Optional[Dict] = None,
    ) -> None:
        """
        Add a reaction to the network.

        Args:
            reaction_id: Reaction identifier
            substrates: List of substrate metabolite IDs
            products: List of product metabolite IDs
            enzyme: Enzyme (gene) identifier
            reversible: Whether the reaction is reversible
            attributes: Additional reaction attributes
        """
        # Store reaction data
        reaction_data = {
            "id": reaction_id,
            "substrates": substrates,
            "products": products,
            "enzyme": enzyme,
            "reversible": reversible,
            **(attributes or {}),
        }
        self.reactions[reaction_id] = reaction_data
        
        # Add reaction node
        self.graph.add_node(
            reaction_id,
            type="reaction",
            reversible=reversible,
            **(attributes or {}),
        )
        
        # Add substrate nodes and edges
        for substrate in substrates:
            if substrate not in self.graph:
                self.graph.add_node(substrate, type="metabolite")
                self.metabolites[substrate] = {"id": substrate}
            
            # Substrate -> Reaction edge
            self.graph.add_edge(substrate, reaction_id, type="substrate")
        
        # Add product nodes and edges
        for product in products:
            if product not in self.graph:
                self.graph.add_node(product, type="metabolite")
                self.metabolites[product] = {"id": product}
            
            # Reaction -> Product edge
            self.graph.add_edge(reaction_id, product, type="product")
        
        # Add enzyme if provided
        if enzyme:
            enzyme_id = normalize_gene_id(enzyme)
            
            if enzyme_id not in self.graph:
                self.graph.add_node(enzyme_id, type="enzyme")
                self.enzymes[enzyme_id] = {"id": enzyme_id}
            
            # Enzyme -> Reaction edge
            self.graph.add_edge(enzyme_id, reaction_id, type="catalyzes")
    
    def add_metabolite(
        self,
        metabolite_id: str,
        name: Optional[str] = None,
        formula: Optional[str] = None,
        attributes: Optional[Dict] = None,
    ) -> None:
        """
        Add or update a metabolite in the network.

        Args:
            metabolite_id: Metabolite identifier
            name: Metabolite name
            formula: Chemical formula
            attributes: Additional metabolite attributes
        """
        # Create metabolite data
        metabolite_data = {
            "id": metabolite_id,
            "name": name or metabolite_id,
            "formula": formula,
            **(attributes or {}),
        }
        
        # Update existing metabolite or add new one
        if metabolite_id in self.metabolites:
            self.metabolites[metabolite_id].update(metabolite_data)
            
            # Update node attributes
            for key, value in metabolite_data.items():
                self.graph.nodes[metabolite_id][key] = value
        else:
            self.metabolites[metabolite_id] = metabolite_data
            
            # Add node if it doesn't exist
            if metabolite_id not in self.graph:
                self.graph.add_node(
                    metabolite_id,
                    type="metabolite",
                    **metabolite_data,
                )
    
    def add_enzyme(
        self,
        enzyme_id: str,
        name: Optional[str] = None,
        function: Optional[str] = None,
        attributes: Optional[Dict] = None,
    ) -> None:
        """
        Add or update an enzyme in the network.

        Args:
            enzyme_id: Enzyme (gene) identifier
            name: Enzyme name
            function: Enzyme function
            attributes: Additional enzyme attributes
        """
        # Normalize enzyme ID
        canonical_id = normalize_gene_id(enzyme_id)
        
        # Create enzyme data
        enzyme_data = {
            "id": canonical_id,
            "name": name or canonical_id,
            "function": function,
            **(attributes or {}),
        }
        
        # Update existing enzyme or add new one
        if canonical_id in self.enzymes:
            self.enzymes[canonical_id].update(enzyme_data)
            
            # Update node attributes
            for key, value in enzyme_data.items():
                self.graph.nodes[canonical_id][key] = value
        else:
            self.enzymes[canonical_id] = enzyme_data
            
            # Add node if it doesn't exist
            if canonical_id not in self.graph:
                self.graph.add_node(
                    canonical_id,
                    type="enzyme",
                    **enzyme_data,
                )
    
    def add_metabolite_link(
        self,
        source_id: str,
        target_id: str,
        relationship: str = "related_to",
        attributes: Optional[Dict] = None,
    ) -> None:
        """
        Add a link between two metabolites.

        Args:
            source_id: Source metabolite identifier
            target_id: Target metabolite identifier
            relationship: Relationship type
            attributes: Additional edge attributes
        """
        # Ensure metabolites exist
        if source_id not in self.graph:
            self.add_metabolite(source_id)
        
        if target_id not in self.graph:
            self.add_metabolite(target_id)
        
        # Add edge
        self.graph.add_edge(
            source_id,
            target_id,
            type=relationship,
            **(attributes or {}),
        )
    
    def get_reaction_substrates(self, reaction_id: str) -> List[str]:
        """
        Get substrates for a reaction.

        Args:
            reaction_id: Reaction identifier

        Returns:
            List of substrate metabolite IDs
        """
        if reaction_id not in self.graph:
            return []
        
        substrates = []
        
        # Find incoming edges with type "substrate"
        for source, target, data in self.graph.in_edges(reaction_id, data=True):
            if data.get("type") == "substrate":
                substrates.append(source)
        
        return substrates
    
    def get_reaction_products(self, reaction_id: str) -> List[str]:
        """
        Get products for a reaction.

        Args:
            reaction_id: Reaction identifier

        Returns:
            List of product metabolite IDs
        """
        if reaction_id not in self.graph:
            return []
        
        products = []
        
        # Find outgoing edges with type "product"
        for source, target, data in self.graph.out_edges(reaction_id, data=True):
            if data.get("type") == "product":
                products.append(target)
        
        return products
    
    def get_reaction_enzyme(self, reaction_id: str) -> Optional[str]:
        """
        Get the enzyme for a reaction.

        Args:
            reaction_id: Reaction identifier

        Returns:
            Enzyme ID or None if no enzyme
        """
        if reaction_id not in self.graph:
            return None
        
        # Find incoming edges with type "catalyzes"
        for source, target, data in self.graph.in_edges(reaction_id, data=True):
            if data.get("type") == "catalyzes":
                return source
        
        return None
    
    def get_reactions_by_enzyme(self, enzyme_id: str) -> List[str]:
        """
        Get reactions catalyzed by an enzyme.

        Args:
            enzyme_id: Enzyme identifier

        Returns:
            List of reaction IDs
        """
        canonical_id = normalize_gene_id(enzyme_id)
        
        if canonical_id not in self.graph:
            return []
        
        reactions = []
        
        # Find outgoing edges with type "catalyzes"
        for source, target, data in self.graph.out_edges(canonical_id, data=True):
            if data.get("type") == "catalyzes":
                reactions.append(target)
        
        return reactions
    
    def get_metabolite_reactions(self, metabolite_id: str) -> Dict[str, List[str]]:
        """
        Get reactions where a metabolite is involved.

        Args:
            metabolite_id: Metabolite identifier

        Returns:
            Dictionary with "as_substrate" and "as_product" lists of reaction IDs
        """
        if metabolite_id not in self.graph:
            return {"as_substrate": [], "as_product": []}
        
        as_substrate = []
        as_product = []
        
        # Find outgoing edges (metabolite is substrate)
        for source, target, data in self.graph.out_edges(metabolite_id, data=True):
            if data.get("type") == "substrate":
                as_substrate.append(target)
        
        # Find incoming edges (metabolite is product)
        for source, target, data in self.graph.in_edges(metabolite_id, data=True):
            if data.get("type") == "product":
                as_product.append(source)
        
        return {
            "as_substrate": as_substrate,
            "as_product": as_product,
        }
    
    def find_metabolic_path(
        self, source_metabolite: str, target_metabolite: str, max_length: int = 10
    ) -> List[List[str]]:
        """
        Find metabolic paths between two metabolites.

        Args:
            source_metabolite: Source metabolite identifier
            target_metabolite: Target metabolite identifier
            max_length: Maximum path length

        Returns:
            List of paths, where each path is a list of node IDs
        """
        if source_metabolite not in self.graph or target_metabolite not in self.graph:
            return []
        
        # Create a simplified graph with direct metabolite-to-metabolite connections
        simplified = nx.DiGraph()
        
        for reaction_id, reaction_data in self.reactions.items():
            for substrate in reaction_data["substrates"]:
                for product in reaction_data["products"]:
                    # Add edge from substrate to product
                    if not simplified.has_edge(substrate, product):
                        simplified.add_edge(substrate, product, reactions=[reaction_id])
                    else:
                        # Add this reaction to the existing edge
                        simplified.edges[substrate, product]["reactions"].append(reaction_id)
        
        # Find paths
        paths = []
        try:
            for path in nx.all_simple_paths(
                simplified, source=source_metabolite, target=target_metabolite, cutoff=max_length
            ):
                paths.append(path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        
        return paths
    
    def save(self, output_path: str) -> None:
        """
        Save the reaction network to a file.

        Args:
            output_path: Output file path
        """
        # Create data dictionary
        data = {
            "name": self.name,
            "reactions": self.reactions,
            "metabolites": self.metabolites,
            "enzymes": self.enzymes,
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
    def load(cls, input_path: str) -> "ReactionNetwork":
        """
        Load a reaction network from a file.

        Args:
            input_path: Input file path

        Returns:
            ReactionNetwork instance
        """
        _, ext = os.path.splitext(input_path)
        
        if ext.lower() == ".pkl":
            # Load from pickle
            with open(input_path, "rb") as f:
                data = pickle.load(f)
                
                instance = cls(name=data.get("name", "reaction_network"))
                instance.graph = data.get("graph", nx.DiGraph())
                instance.reactions = data.get("reactions", {})
                instance.metabolites = data.get("metabolites", {})
                instance.enzymes = data.get("enzymes", {})
        else:
            # Load from JSON
            with open(input_path, "r") as f:
                data = json.load(f)
                
                instance = cls(name=data.get("name", "reaction_network"))
                instance.reactions = data.get("reactions", {})
                instance.metabolites = data.get("metabolites", {})
                instance.enzymes = data.get("enzymes", {})
                
                # Load graph from separate file
                graph_path = data.get("graph_path")
                if graph_path and os.path.exists(graph_path):
                    instance.graph = load_network(graph_path)
                else:
                    # Rebuild graph from reaction data
                    instance._rebuild_graph()
        
        return instance
    
    def _rebuild_graph(self) -> None:
        """Rebuild the graph from reaction data."""
        self.graph = nx.DiGraph()
        
        # Add all nodes
        for metabolite_id, metabolite_data in self.metabolites.items():
            self.graph.add_node(metabolite_id, type="metabolite", **metabolite_data)
        
        for enzyme_id, enzyme_data in self.enzymes.items():
            self.graph.add_node(enzyme_id, type="enzyme", **enzyme_data)
        
        for reaction_id, reaction_data in self.reactions.items():
            self.graph.add_node(reaction_id, type="reaction", **reaction_data)
            
            # Add edges
            for substrate in reaction_data["substrates"]:
                self.graph.add_edge(substrate, reaction_id, type="substrate")
            
            for product in reaction_data["products"]:
                self.graph.add_edge(reaction_id, product, type="product")
            
            if reaction_data.get("enzyme"):
                self.graph.add_edge(reaction_data["enzyme"], reaction_id, type="catalyzes")
    
    def add_from_json(self, json_data: Union[str, Dict]) -> None:
        """
        Add reactions from JSON data.

        Args:
            json_data: JSON string or dictionary of reaction data
        """
        # Parse JSON if string
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        # Add metabolites
        for metabolite_data in data.get("metabolites", []):
            metabolite_id = metabolite_data.pop("id")
            self.add_metabolite(metabolite_id, **metabolite_data)
        
        # Add enzymes
        for enzyme_data in data.get("enzymes", []):
            enzyme_id = enzyme_data.pop("id")
            self.add_enzyme(enzyme_id, **enzyme_data)
        
        # Add reactions
        for reaction_data in data.get("reactions", []):
            reaction_id = reaction_data.pop("id")
            substrates = reaction_data.pop("substrates")
            products = reaction_data.pop("products")
            enzyme = reaction_data.pop("enzyme", None)
            reversible = reaction_data.pop("reversible", False)
            
            self.add_reaction(
                reaction_id,
                substrates,
                products,
                enzyme,
                reversible,
                reaction_data,
            )
    
    def get_enzyme_metabolic_scope(self, enzyme_id: str) -> Dict[str, List[str]]:
        """
        Get the metabolic scope of an enzyme.

        Args:
            enzyme_id: Enzyme identifier

        Returns:
            Dictionary with "substrates" and "products" lists
        """
        canonical_id = normalize_gene_id(enzyme_id)
        
        if canonical_id not in self.graph:
            return {"substrates": [], "products": []}
        
        substrates = set()
        products = set()
        
        # Get reactions catalyzed by this enzyme
        reactions = self.get_reactions_by_enzyme(canonical_id)
        
        # Get substrates and products for each reaction
        for reaction_id in reactions:
            substrates.update(self.get_reaction_substrates(reaction_id))
            products.update(self.get_reaction_products(reaction_id))
        
        return {
            "substrates": sorted(list(substrates)),
            "products": sorted(list(products)),
        }
    
    def get_subnetwork_by_reactions(self, reaction_ids: List[str]) -> "ReactionNetwork":
        """
        Get a subnetwork containing only the specified reactions.

        Args:
            reaction_ids: List of reaction identifiers

        Returns:
            ReactionNetwork instance containing the subnetwork
        """
        # Create new network
        subnetwork = ReactionNetwork(name=f"{self.name}_sub")
        
        # Add relevant reactions
        for reaction_id in reaction_ids:
            if reaction_id in self.reactions:
                reaction_data = self.reactions[reaction_id].copy()
                
                # Add reaction with all its components
                subnetwork.add_reaction(
                    reaction_id,
                    reaction_data["substrates"],
                    reaction_data["products"],
                    reaction_data.get("enzyme"),
                    reaction_data.get("reversible", False),
                    {k: v for k, v in reaction_data.items() 
                     if k not in ["id", "substrates", "products", "enzyme", "reversible"]},
                )
                
                # Add metabolite data
                for metabolite_id in reaction_data["substrates"] + reaction_data["products"]:
                    if metabolite_id in self.metabolites:
                        subnetwork.add_metabolite(
                            metabolite_id,
                            **{k: v for k, v in self.metabolites[metabolite_id].items() if k != "id"},
                        )
                
                # Add enzyme data
                if reaction_data.get("enzyme") and reaction_data["enzyme"] in self.enzymes:
                    subnetwork.add_enzyme(
                        reaction_data["enzyme"],
                        **{k: v for k, v in self.enzymes[reaction_data["enzyme"]].items() if k != "id"},
                    )
        
        return subnetwork
    
    def __len__(self) -> int:
        """
        Get the number of reactions in the network.

        Returns:
            Number of reactions
        """
        return len(self.reactions)
    
    def __str__(self) -> str:
        """
        Get a string representation of the network.

        Returns:
            String representation
        """
        return (
            f"ReactionNetwork(name={self.name}, reactions={len(self.reactions)}, "
            f"metabolites={len(self.metabolites)}, enzymes={len(self.enzymes)})"
        ) 