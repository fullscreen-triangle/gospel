"""
Visualizer for gene and reaction networks.
"""

import io
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class NetworkVisualizer:
    """
    Visualizer for gene and reaction networks.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 10), style: str = "light"):
        """
        Initialize a network visualizer.

        Args:
            figsize: Figure size
            style: Visualization style ('light', 'dark', or 'custom')
        """
        self.figsize = figsize
        self.style = style
        
        # Set style
        if HAS_SEABORN:
            if style == "light":
                sns.set_style("whitegrid")
            elif style == "dark":
                sns.set_style("darkgrid")
            else:
                # Custom style
                plt.style.use("default")
        
        # Define color schemes
        self.color_schemes = {
            "light": {
                "gene": "#4C72B0",  # Blue
                "variant": "#DD8452",  # Orange
                "pathway": "#55A868",  # Green
                "metabolite": "#C44E52",  # Red
                "enzyme": "#8172B3",  # Purple
                "reaction": "#937860",  # Brown
                "edge": "#CCCCCC",  # Light gray
                "highlight": "#FFC107",  # Yellow
                "background": "#FFFFFF",  # White
                "text": "#333333",  # Dark gray
            },
            "dark": {
                "gene": "#729ECF",  # Light blue
                "variant": "#FCAF3E",  # Light orange
                "pathway": "#8AE234",  # Light green
                "metabolite": "#EF2929",  # Light red
                "enzyme": "#AD7FA8",  # Light purple
                "reaction": "#C4A000",  # Light brown
                "edge": "#555555",  # Dark gray
                "highlight": "#FFD54F",  # Light yellow
                "background": "#333333",  # Dark gray
                "text": "#EEEEEE",  # Light gray
            },
        }
        
        # Use selected color scheme
        self.colors = self.color_schemes.get(style, self.color_schemes["light"])

    def visualize_gene_network(
        self,
        graph: nx.Graph,
        highlight_nodes: Optional[List[str]] = None,
        highlight_edges: Optional[List[Tuple[str, str]]] = None,
        node_size_attr: Optional[str] = None,
        edge_width_attr: Optional[str] = None,
        node_label_attr: Optional[str] = None,
        title: Optional[str] = None,
        show_legend: bool = True,
        output_path: Optional[str] = None,
        layout: str = "spring",
    ) -> None:
        """
        Visualize a gene network.

        Args:
            graph: NetworkX graph to visualize
            highlight_nodes: List of nodes to highlight
            highlight_edges: List of edges to highlight
            node_size_attr: Node attribute to use for node size
            edge_width_attr: Edge attribute to use for edge width
            node_label_attr: Node attribute to use for node labels
            title: Plot title
            show_legend: Whether to show a legend
            output_path: Path to save the figure
            layout: Layout algorithm ('spring', 'circular', 'spectral', 'kamada_kawai')
        """
        if graph.number_of_nodes() == 0:
            print("Empty graph, nothing to visualize")
            return
        
        # Create figure
        plt.figure(figsize=self.figsize)
        
        # Set title
        if title:
            plt.title(title, fontsize=16)
        
        # Compute layout
        if layout == "spring":
            pos = nx.spring_layout(graph, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "spectral":
            pos = nx.spectral_layout(graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42)
        
        # Set node colors based on node type
        node_colors = []
        node_sizes = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get("type", "gene")
            node_colors.append(self.colors.get(node_type, self.colors["gene"]))
            
            # Set node size
            if node_size_attr and node_size_attr in graph.nodes[node]:
                size = graph.nodes[node][node_size_attr]
                # Scale size to reasonable range
                size = max(100, min(2000, 100 + 500 * size))
            else:
                # Default size based on node type
                if node_type == "gene":
                    size = 300
                elif node_type == "variant":
                    size = 150
                elif node_type == "pathway":
                    size = 400
                else:
                    size = 200
            
            node_sizes.append(size)
        
        # Set edge colors and widths
        edge_colors = []
        edge_widths = []
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("relationship", data.get("type", "default"))
            
            # Highlight edge if requested
            if highlight_edges and (u, v) in highlight_edges:
                edge_colors.append(self.colors["highlight"])
                edge_widths.append(2.5)
            else:
                # Color based on edge type
                if edge_type == "has_variant":
                    edge_colors.append(self.colors["variant"])
                elif edge_type == "participates_in":
                    edge_colors.append(self.colors["pathway"])
                elif edge_type == "interacts_with":
                    edge_colors.append(self.colors["gene"])
                else:
                    edge_colors.append(self.colors["edge"])
                
                # Width based on attribute or default
                if edge_width_attr and edge_width_attr in data:
                    width = data[edge_width_attr]
                    # Scale width to reasonable range
                    width = max(0.5, min(5.0, 0.5 + 2.0 * width))
                else:
                    width = 1.0
                
                edge_widths.append(width)
        
        # Draw network
        nx.draw_networkx_edges(
            graph, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.9,
            linewidths=1.0,
            edgecolors="black",
        )
        
        # Highlight nodes if requested
        if highlight_nodes:
            highlight_list = [n for n in highlight_nodes if n in graph]
            if highlight_list:
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=highlight_list,
                    node_color=self.colors["highlight"],
                    alpha=0.9,
                    linewidths=2.0,
                    edgecolors="black",
                    node_size=[node_sizes[list(graph.nodes()).index(n)] for n in highlight_list],
                )
        
        # Draw labels
        if node_label_attr:
            labels = {}
            for node in graph.nodes():
                if node_label_attr in graph.nodes[node]:
                    labels[node] = graph.nodes[node][node_label_attr]
                else:
                    labels[node] = node
        else:
            labels = {node: node for node in graph.nodes()}
        
        nx.draw_networkx_labels(
            graph, pos, labels=labels, font_size=10, font_weight="bold"
        )
        
        # Add legend if requested
        if show_legend:
            handles = []
            labels = []
            
            # Add node type legend
            node_types = set(nx.get_node_attributes(graph, "type").values())
            for node_type in node_types:
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker="o",
                        color="w",
                        markerfacecolor=self.colors.get(node_type, self.colors["gene"]),
                        markersize=10,
                        label=node_type.capitalize(),
                    )
                )
                labels.append(node_type.capitalize())
            
            # Add edge type legend
            edge_types = set()
            for _, _, data in graph.edges(data=True):
                edge_type = data.get("relationship", data.get("type", "default"))
                edge_types.add(edge_type)
            
            for edge_type in edge_types:
                if edge_type == "has_variant":
                    color = self.colors["variant"]
                elif edge_type == "participates_in":
                    color = self.colors["pathway"]
                elif edge_type == "interacts_with":
                    color = self.colors["gene"]
                else:
                    color = self.colors["edge"]
                
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        color=color,
                        lw=2,
                        label=edge_type.replace("_", " ").capitalize(),
                    )
                )
                labels.append(edge_type.replace("_", " ").capitalize())
            
            plt.legend(handles=handles, labels=labels, loc="best")
        
        plt.axis("off")
        plt.tight_layout()
        
        # Save or show the figure
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close()

    def visualize_reaction_network(
        self,
        graph: nx.DiGraph,
        highlight_nodes: Optional[List[str]] = None,
        highlight_paths: Optional[List[List[str]]] = None,
        title: Optional[str] = None,
        show_legend: bool = True,
        output_path: Optional[str] = None,
        layout: str = "dot",
    ) -> None:
        """
        Visualize a reaction network.

        Args:
            graph: NetworkX directed graph to visualize
            highlight_nodes: List of nodes to highlight
            highlight_paths: List of paths to highlight
            title: Plot title
            show_legend: Whether to show a legend
            output_path: Path to save the figure
            layout: Layout algorithm ('dot', 'neato', 'fdp', 'sfdp', 'twopi', 'circo')
        """
        try:
            import pydot
            from networkx.drawing.nx_pydot import graphviz_layout
            has_graphviz = True
        except ImportError:
            has_graphviz = False
            layout = "spring"  # Fallback to spring layout
        
        if graph.number_of_nodes() == 0:
            print("Empty graph, nothing to visualize")
            return
        
        # Create figure
        plt.figure(figsize=self.figsize)
        
        # Set title
        if title:
            plt.title(title, fontsize=16)
        
        # Compute layout
        if has_graphviz and layout in ["dot", "neato", "fdp", "sfdp", "twopi", "circo"]:
            try:
                pos = graphviz_layout(graph, prog=layout)
            except Exception:
                print(f"Error using {layout} layout, falling back to spring layout")
                pos = nx.spring_layout(graph, seed=42)
        else:
            if layout == "spring":
                pos = nx.spring_layout(graph, seed=42)
            elif layout == "circular":
                pos = nx.circular_layout(graph)
            elif layout == "spectral":
                pos = nx.spectral_layout(graph)
            elif layout == "kamada_kawai":
                pos = nx.kamada_kawai_layout(graph)
            else:
                pos = nx.spring_layout(graph, seed=42)
        
        # Set node colors based on node type
        node_colors = []
        node_sizes = []
        node_shapes = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get("type", "metabolite")
            node_colors.append(self.colors.get(node_type, self.colors["metabolite"]))
            
            # Set node size and shape based on type
            if node_type == "metabolite":
                size = 300
                shape = "o"  # Circle
            elif node_type == "reaction":
                size = 200
                shape = "s"  # Square
            elif node_type == "enzyme":
                size = 400
                shape = "^"  # Triangle
            else:
                size = 250
                shape = "o"  # Circle
            
            node_sizes.append(size)
            node_shapes.append(shape)
        
        # Set edge colors and widths
        edge_colors = []
        edge_widths = []
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("type", "default")
            
            # Check if edge is part of a highlighted path
            is_highlighted = False
            if highlight_paths:
                for path in highlight_paths:
                    if len(path) > 1:
                        for i in range(len(path) - 1):
                            if (path[i] == u and path[i + 1] == v) or (path[i] == v and path[i + 1] == u):
                                is_highlighted = True
                                break
                    if is_highlighted:
                        break
            
            if is_highlighted:
                edge_colors.append(self.colors["highlight"])
                edge_widths.append(2.5)
            else:
                # Color based on edge type
                if edge_type == "substrate":
                    edge_colors.append(self.colors["metabolite"])
                elif edge_type == "product":
                    edge_colors.append(self.colors["metabolite"])
                elif edge_type == "catalyzes":
                    edge_colors.append(self.colors["enzyme"])
                else:
                    edge_colors.append(self.colors["edge"])
                
                edge_widths.append(1.0)
        
        # Group nodes by shape for drawing
        node_groups = {}
        for shape in set(node_shapes):
            nodelist = [
                node for i, node in enumerate(graph.nodes()) if node_shapes[i] == shape
            ]
            node_groups[shape] = nodelist
        
        # Draw edges
        nx.draw_networkx_edges(
            graph, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7,
            arrowsize=15, arrowstyle="-|>",
        )
        
        # Draw nodes by shape
        for shape, nodelist in node_groups.items():
            if not nodelist:
                continue
            
            node_color = [node_colors[list(graph.nodes()).index(n)] for n in nodelist]
            node_size = [node_sizes[list(graph.nodes()).index(n)] for n in nodelist]
            
            if shape == "o":  # Circle
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=nodelist,
                    node_color=node_color,
                    node_size=node_size,
                    alpha=0.9,
                    linewidths=1.0,
                    edgecolors="black",
                )
            elif shape == "s":  # Square
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=nodelist,
                    node_color=node_color,
                    node_size=node_size,
                    alpha=0.9,
                    linewidths=1.0,
                    edgecolors="black",
                    node_shape="s",
                )
            elif shape == "^":  # Triangle
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=nodelist,
                    node_color=node_color,
                    node_size=node_size,
                    alpha=0.9,
                    linewidths=1.0,
                    edgecolors="black",
                    node_shape="^",
                )
        
        # Highlight nodes if requested
        if highlight_nodes:
            highlight_list = [n for n in highlight_nodes if n in graph]
            if highlight_list:
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=highlight_list,
                    node_color=self.colors["highlight"],
                    alpha=0.9,
                    linewidths=2.0,
                    edgecolors="black",
                    node_size=[node_sizes[list(graph.nodes()).index(n)] for n in highlight_list],
                )
        
        # Draw labels
        labels = {}
        for node in graph.nodes():
            # Use name attribute if available, otherwise use node ID
            if "name" in graph.nodes[node]:
                labels[node] = graph.nodes[node]["name"]
            else:
                # Truncate long IDs
                if len(str(node)) > 15:
                    labels[node] = str(node)[:12] + "..."
                else:
                    labels[node] = node
        
        nx.draw_networkx_labels(
            graph, pos, labels=labels, font_size=10, font_weight="bold"
        )
        
        # Add legend if requested
        if show_legend:
            handles = []
            labels = []
            
            # Add node type legend
            node_types = set(nx.get_node_attributes(graph, "type").values())
            for node_type in node_types:
                if node_type == "metabolite":
                    shape = "o"
                elif node_type == "reaction":
                    shape = "s"
                elif node_type == "enzyme":
                    shape = "^"
                else:
                    shape = "o"
                
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker=shape,
                        color="w",
                        markerfacecolor=self.colors.get(node_type, self.colors["metabolite"]),
                        markersize=10,
                        label=node_type.capitalize(),
                    )
                )
                labels.append(node_type.capitalize())
            
            # Add edge type legend
            edge_types = set()
            for _, _, data in graph.edges(data=True):
                edge_type = data.get("type", "default")
                edge_types.add(edge_type)
            
            for edge_type in edge_types:
                if edge_type == "substrate":
                    color = self.colors["metabolite"]
                elif edge_type == "product":
                    color = self.colors["metabolite"]
                elif edge_type == "catalyzes":
                    color = self.colors["enzyme"]
                else:
                    color = self.colors["edge"]
                
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        color=color,
                        lw=2,
                        label=edge_type.replace("_", " ").capitalize(),
                    )
                )
                labels.append(edge_type.replace("_", " ").capitalize())
            
            plt.legend(handles=handles, labels=labels, loc="best")
        
        plt.axis("off")
        plt.tight_layout()
        
        # Save or show the figure
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close()

    def visualize_subgraph(
        self,
        graph: nx.Graph,
        center_node: str,
        max_distance: int = 2,
        highlight_paths: Optional[List[List[str]]] = None,
        title: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Visualize a subgraph centered on a node.

        Args:
            graph: NetworkX graph
            center_node: Center node for the subgraph
            max_distance: Maximum distance from center node
            highlight_paths: List of paths to highlight
            title: Plot title
            output_path: Path to save the figure
        """
        if center_node not in graph:
            print(f"Node {center_node} not found in graph")
            return
        
        # Extract subgraph
        nodes = {center_node}
        current_nodes = {center_node}
        
        for _ in range(max_distance):
            new_nodes = set()
            for node in current_nodes:
                new_nodes.update(graph.neighbors(node))
            
            nodes.update(new_nodes)
            current_nodes = new_nodes
        
        subgraph = graph.subgraph(nodes)
        
        # Visualize based on graph type
        if isinstance(graph, nx.DiGraph):
            self.visualize_reaction_network(
                subgraph,
                highlight_nodes=[center_node],
                highlight_paths=highlight_paths,
                title=title or f"Subgraph centered on {center_node}",
                output_path=output_path,
            )
        else:
            self.visualize_gene_network(
                subgraph,
                highlight_nodes=[center_node],
                highlight_edges=self._paths_to_edges(highlight_paths) if highlight_paths else None,
                title=title or f"Subgraph centered on {center_node}",
                output_path=output_path,
            )

    def visualize_pathway(
        self,
        graph: nx.Graph,
        pathway_id: str,
        include_variants: bool = True,
        title: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Visualize a specific pathway.

        Args:
            graph: NetworkX graph
            pathway_id: Pathway identifier
            include_variants: Whether to include variants
            title: Plot title
            output_path: Path to save the figure
        """
        if pathway_id not in graph:
            print(f"Pathway {pathway_id} not found in graph")
            return
        
        # Find all genes in this pathway
        genes = []
        for u, v, data in graph.edges(data=True):
            if (u == pathway_id or v == pathway_id) and data.get("relationship") == "participates_in":
                gene = v if u == pathway_id else u
                genes.append(gene)
        
        if not genes:
            print(f"No genes found for pathway {pathway_id}")
            return
        
        # Create subgraph
        nodes = set(genes + [pathway_id])
        
        # Add variants if requested
        if include_variants:
            for gene in genes:
                for u, v, data in graph.edges(data=True):
                    if (u == gene or v == gene) and data.get("relationship") == "has_variant":
                        variant = v if u == gene else u
                        nodes.add(variant)
        
        subgraph = graph.subgraph(nodes)
        
        # Visualize
        self.visualize_gene_network(
            subgraph,
            highlight_nodes=[pathway_id],
            title=title or f"Pathway: {pathway_id}",
            output_path=output_path,
            layout="circular",
        )

    def _paths_to_edges(self, paths: List[List[str]]) -> List[Tuple[str, str]]:
        """
        Convert paths to a list of edges.

        Args:
            paths: List of paths, where each path is a list of node IDs

        Returns:
            List of edges
        """
        edges = []
        for path in paths:
            if len(path) > 1:
                for i in range(len(path) - 1):
                    edges.append((path[i], path[i + 1]))
        
        return edges 