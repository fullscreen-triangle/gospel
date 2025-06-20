use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use rayon::prelude::*;

/// Network edge representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEdge {
    pub source: String,
    pub target: String,
    pub weight: f64,
    pub edge_type: String,
    pub source_db: String,
}

/// Node centrality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityMetrics {
    pub node_id: String,
    pub degree_centrality: f64,
    pub betweenness_centrality: f64,
    pub closeness_centrality: f64,
    pub eigenvector_centrality: f64,
    pub pagerank: f64,
    pub degree: usize,
    pub centrality_score: f64,
}

/// Community detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityResult {
    pub node_communities: HashMap<String, usize>,
    pub community_sizes: Vec<usize>,
    pub modularity: f64,
    pub num_communities: usize,
}

/// Network analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAnalysisResult {
    pub node_count: usize,
    pub edge_count: usize,
    pub is_connected: bool,
    pub density: f64,
    pub average_clustering: f64,
    pub average_shortest_path: f64,
    pub diameter: usize,
    pub largest_component_size: usize,
    pub top_central_nodes: Vec<CentralityMetrics>,
    pub communities: CommunityResult,
    pub degree_distribution: Vec<(usize, usize)>, // (degree, count)
}

/// Network comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkComparison {
    pub network1_stats: NetworkStats,
    pub network2_stats: NetworkStats,
    pub common_nodes: Vec<String>,
    pub unique_nodes_1: Vec<String>,
    pub unique_nodes_2: Vec<String>,
    pub common_edges: Vec<(String, String)>,
    pub node_jaccard_similarity: f64,
    pub edge_jaccard_similarity: f64,
}

/// Basic network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub nodes: usize,
    pub edges: usize,
    pub density: f64,
    pub average_clustering: f64,
}

/// High-performance network analyzer
pub struct NetworkAnalyzer {
    adjacency_list: HashMap<String, Vec<(String, f64)>>,
    nodes: HashSet<String>,
    edges: Vec<NetworkEdge>,
    is_directed: bool,
}

impl NetworkAnalyzer {
    /// Create a new network analyzer
    pub fn new(is_directed: bool) -> Self {
        Self {
            adjacency_list: HashMap::new(),
            nodes: HashSet::new(),
            edges: Vec::new(),
            is_directed,
        }
    }

    /// Build network from protein-protein interactions
    pub fn build_from_interactions(
        &mut self,
        interactions: &[NetworkEdge],
        min_score: f64,
        include_edge_types: Option<&[String]>,
    ) -> Result<()> {
        self.clear();

        for interaction in interactions {
            // Filter by score
            if interaction.weight < min_score {
                continue;
            }

            // Filter by edge type
            if let Some(types) = include_edge_types {
                if !types.contains(&interaction.edge_type) {
                    continue;
                }
            }

            self.add_edge(
                &interaction.source,
                &interaction.target,
                interaction.weight,
            );
        }

        Ok(())
    }

    /// Build network from pathway data
    pub fn build_from_pathways(
        &mut self,
        pathways: &[(String, Vec<String>)], // (pathway_id, gene_list)
        min_genes_per_pathway: usize,
    ) -> Result<()> {
        self.clear();

        for (pathway_id, genes) in pathways {
            if genes.len() < min_genes_per_pathway {
                continue;
            }

            // Connect all genes in the same pathway
            for i in 0..genes.len() {
                for j in (i + 1)..genes.len() {
                    self.add_edge(&genes[i], &genes[j], 1.0);
                }
            }
        }

        Ok(())
    }

    /// Add an edge to the network
    pub fn add_edge(&mut self, source: &str, target: &str, weight: f64) {
        let source = source.to_string();
        let target = target.to_string();

        self.nodes.insert(source.clone());
        self.nodes.insert(target.clone());

        // Add to adjacency list
        self.adjacency_list
            .entry(source.clone())
            .or_insert_with(Vec::new)
            .push((target.clone(), weight));

        if !self.is_directed {
            self.adjacency_list
                .entry(target.clone())
                .or_insert_with(Vec::new)
                .push((source.clone(), weight));
        }

        // Store edge
        self.edges.push(NetworkEdge {
            source,
            target,
            weight,
            edge_type: "unknown".to_string(),
            source_db: "unknown".to_string(),
        });
    }

    /// Clear the network
    pub fn clear(&mut self) {
        self.adjacency_list.clear();
        self.nodes.clear();
        self.edges.clear();
    }

    /// Calculate degree centrality for all nodes
    pub fn calculate_degree_centrality(&self) -> HashMap<String, f64> {
        let n = self.nodes.len() as f64;
        if n <= 1.0 {
            return HashMap::new();
        }

        self.nodes
            .par_iter()
            .map(|node| {
                let degree = self.adjacency_list.get(node).map_or(0, |neighbors| neighbors.len()) as f64;
                let centrality = degree / (n - 1.0);
                (node.clone(), centrality)
            })
            .collect()
    }

    /// Calculate betweenness centrality using parallel processing
    pub fn calculate_betweenness_centrality(&self) -> HashMap<String, f64> {
        let nodes: Vec<_> = self.nodes.iter().cloned().collect();
        
        // Use parallel iteration for better performance
        let betweenness_values: Vec<_> = nodes
            .par_iter()
            .map(|node| {
                let betweenness = self.single_source_shortest_paths(node);
                (node.clone(), betweenness)
            })
            .collect();

        let mut betweenness_centrality = HashMap::new();
        for (node, value) in betweenness_values {
            betweenness_centrality.insert(node, value);
        }

        // Normalize
        let n = self.nodes.len() as f64;
        let normalization_factor = if self.is_directed {
            (n - 1.0) * (n - 2.0)
        } else {
            (n - 1.0) * (n - 2.0) / 2.0
        };

        if normalization_factor > 0.0 {
            for value in betweenness_centrality.values_mut() {
                *value /= normalization_factor;
            }
        }

        betweenness_centrality
    }

    /// Single-source shortest paths for betweenness calculation
    fn single_source_shortest_paths(&self, source: &str) -> f64 {
        let mut distances = HashMap::new();
        let mut predecessors: HashMap<String, Vec<String>> = HashMap::new();
        let mut sigma = HashMap::new();
        let mut queue = VecDeque::new();

        // Initialize
        for node in &self.nodes {
            distances.insert(node.clone(), f64::INFINITY);
            predecessors.insert(node.clone(), Vec::new());
            sigma.insert(node.clone(), 0.0);
        }

        distances.insert(source.to_string(), 0.0);
        sigma.insert(source.to_string(), 1.0);
        queue.push_back(source.to_string());

        let mut stack = Vec::new();

        // BFS
        while let Some(current) = queue.pop_front() {
            stack.push(current.clone());

            if let Some(neighbors) = self.adjacency_list.get(&current) {
                for (neighbor, _weight) in neighbors {
                    let current_dist = distances[&current];
                    let neighbor_dist = distances[neighbor];

                    if neighbor_dist == f64::INFINITY {
                        queue.push_back(neighbor.clone());
                        distances.insert(neighbor.clone(), current_dist + 1.0);
                    }

                    if (current_dist + 1.0 - neighbor_dist).abs() < f64::EPSILON {
                        sigma.insert(neighbor.clone(), sigma[neighbor] + sigma[&current]);
                        predecessors.get_mut(neighbor).unwrap().push(current.clone());
                    }
                }
            }
        }

        // Accumulation phase
        let mut dependency = HashMap::new();
        for node in &self.nodes {
            dependency.insert(node.clone(), 0.0);
        }

        while let Some(node) = stack.pop() {
            if let Some(preds) = predecessors.get(&node) {
                for pred in preds {
                    let dep_update = (sigma[pred] / sigma[&node]) * (1.0 + dependency[&node]);
                    dependency.insert(pred.clone(), dependency[pred] + dep_update);
                }
            }
        }

        dependency.values().sum::<f64>() / 2.0 // For undirected graphs
    }

    /// Calculate closeness centrality
    pub fn calculate_closeness_centrality(&self) -> HashMap<String, f64> {
        self.nodes
            .par_iter()
            .map(|node| {
                let shortest_paths = self.dijkstra_single_source(node);
                let sum_distances: f64 = shortest_paths.values().sum();
                let reachable_nodes = shortest_paths.len();

                let closeness = if sum_distances > 0.0 && reachable_nodes > 1 {
                    (reachable_nodes - 1) as f64 / sum_distances
                } else {
                    0.0
                };

                (node.clone(), closeness)
            })
            .collect()
    }

    /// Dijkstra's algorithm for single source shortest paths
    fn dijkstra_single_source(&self, source: &str) -> HashMap<String, f64> {
        let mut distances = HashMap::new();
        let mut visited = HashSet::new();
        let mut queue = std::collections::BinaryHeap::new();

        // Initialize distances
        for node in &self.nodes {
            distances.insert(node.clone(), f64::INFINITY);
        }
        distances.insert(source.to_string(), 0.0);

        // Priority queue with negative distances for min-heap behavior
        queue.push(std::cmp::Reverse((0.0, source.to_string())));

        while let Some(std::cmp::Reverse((current_dist, current_node))) = queue.pop() {
            if visited.contains(&current_node) {
                continue;
            }
            visited.insert(current_node.clone());

            if let Some(neighbors) = self.adjacency_list.get(&current_node) {
                for (neighbor, weight) in neighbors {
                    if !visited.contains(neighbor) {
                        let new_dist = current_dist + weight;
                        if new_dist < distances[neighbor] {
                            distances.insert(neighbor.clone(), new_dist);
                            queue.push(std::cmp::Reverse((new_dist, neighbor.clone())));
                        }
                    }
                }
            }
        }

        // Return only reachable nodes
        distances
            .into_iter()
            .filter(|(_, dist)| *dist < f64::INFINITY)
            .collect()
    }

    /// Calculate eigenvector centrality using power iteration
    pub fn calculate_eigenvector_centrality(&self, max_iterations: usize, tolerance: f64) -> HashMap<String, f64> {
        let nodes: Vec<_> = self.nodes.iter().cloned().collect();
        let n = nodes.len();
        
        if n == 0 {
            return HashMap::new();
        }

        // Initialize centrality values
        let mut centrality: HashMap<String, f64> = nodes.iter().map(|node| (node.clone(), 1.0 / n as f64)).collect();
        
        for _ in 0..max_iterations {
            let mut new_centrality = HashMap::new();
            
            // Calculate new centrality values
            for node in &nodes {
                let mut sum = 0.0;
                if let Some(neighbors) = self.adjacency_list.get(node) {
                    for (neighbor, weight) in neighbors {
                        sum += centrality[neighbor] * weight;
                    }
                }
                new_centrality.insert(node.clone(), sum);
            }
            
            // Normalize
            let norm: f64 = new_centrality.values().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for value in new_centrality.values_mut() {
                    *value /= norm;
                }
            }
            
            // Check convergence
            let mut converged = true;
            for node in &nodes {
                if (new_centrality[node] - centrality[node]).abs() > tolerance {
                    converged = false;
                    break;
                }
            }
            
            centrality = new_centrality;
            
            if converged {
                break;
            }
        }
        
        centrality
    }

    /// Community detection using Louvain algorithm (simplified)
    pub fn detect_communities(&self) -> CommunityResult {
        let nodes: Vec<_> = self.nodes.iter().cloned().collect();
        let mut communities: HashMap<String, usize> = nodes.iter().enumerate().map(|(i, node)| (node.clone(), i)).collect();
        
        // Simplified community detection - in practice, would use more sophisticated algorithms
        let mut improved = true;
        let mut iteration = 0;
        const MAX_ITERATIONS: usize = 100;
        
        while improved && iteration < MAX_ITERATIONS {
            improved = false;
            iteration += 1;
            
            for node in &nodes {
                let current_community = communities[node];
                let mut best_community = current_community;
                let mut best_modularity_gain = 0.0;
                
                // Check neighboring communities
                if let Some(neighbors) = self.adjacency_list.get(node) {
                    let neighbor_communities: HashSet<usize> = neighbors
                        .iter()
                        .map(|(neighbor, _)| communities[neighbor])
                        .collect();
                    
                    for &neighbor_community in &neighbor_communities {
                        if neighbor_community != current_community {
                            let modularity_gain = self.calculate_modularity_gain(node, neighbor_community, &communities);
                            if modularity_gain > best_modularity_gain {
                                best_modularity_gain = modularity_gain;
                                best_community = neighbor_community;
                            }
                        }
                    }
                }
                
                if best_community != current_community {
                    communities.insert(node.clone(), best_community);
                    improved = true;
                }
            }
        }
        
        // Relabel communities to be sequential
        let unique_communities: HashSet<usize> = communities.values().cloned().collect();
        let community_mapping: HashMap<usize, usize> = unique_communities
            .into_iter()
            .enumerate()
            .map(|(new_id, old_id)| (old_id, new_id))
            .collect();
        
        for community_id in communities.values_mut() {
            *community_id = community_mapping[community_id];
        }
        
        // Calculate community sizes
        let mut community_sizes = vec![0; community_mapping.len()];
        for &community_id in communities.values() {
            community_sizes[community_id] += 1;
        }
        
        let modularity = self.calculate_modularity(&communities);
        
        CommunityResult {
            node_communities: communities,
            community_sizes,
            modularity,
            num_communities: community_mapping.len(),
        }
    }

    /// Calculate modularity gain for moving a node to a different community
    fn calculate_modularity_gain(&self, node: &str, new_community: usize, communities: &HashMap<String, usize>) -> f64 {
        // Simplified modularity gain calculation
        if let Some(neighbors) = self.adjacency_list.get(node) {
            let mut internal_edges = 0.0;
            let mut external_edges = 0.0;
            
            for (neighbor, weight) in neighbors {
                if communities[neighbor] == new_community {
                    internal_edges += weight;
                } else {
                    external_edges += weight;
                }
            }
            
            internal_edges - external_edges
        } else {
            0.0
        }
    }

    /// Calculate network modularity
    fn calculate_modularity(&self, communities: &HashMap<String, usize>) -> f64 {
        let m = self.edges.len() as f64 * 2.0; // Total edge weight (count each edge twice for undirected)
        
        if m == 0.0 {
            return 0.0;
        }
        
        let mut modularity = 0.0;
        
        for edge in &self.edges {
            let source_community = communities.get(&edge.source).unwrap_or(&0);
            let target_community = communities.get(&edge.target).unwrap_or(&0);
            
            if source_community == target_community {
                let ki = self.adjacency_list.get(&edge.source).map_or(0, |neighbors| neighbors.len()) as f64;
                let kj = self.adjacency_list.get(&edge.target).map_or(0, |neighbors| neighbors.len()) as f64;
                
                modularity += edge.weight - (ki * kj) / m;
            }
        }
        
        modularity / m
    }

    /// Calculate clustering coefficient
    pub fn calculate_clustering_coefficient(&self) -> f64 {
        let clustering_coefficients: Vec<f64> = self.nodes
            .par_iter()
            .map(|node| self.node_clustering_coefficient(node))
            .collect();

        if clustering_coefficients.is_empty() {
            0.0
        } else {
            clustering_coefficients.iter().sum::<f64>() / clustering_coefficients.len() as f64
        }
    }

    /// Calculate clustering coefficient for a single node
    fn node_clustering_coefficient(&self, node: &str) -> f64 {
        if let Some(neighbors) = self.adjacency_list.get(node) {
            let degree = neighbors.len();
            if degree < 2 {
                return 0.0;
            }

            let mut triangles = 0;
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let neighbor1 = &neighbors[i].0;
                    let neighbor2 = &neighbors[j].0;
                    
                    if let Some(neighbor1_neighbors) = self.adjacency_list.get(neighbor1) {
                        if neighbor1_neighbors.iter().any(|(n, _)| n == neighbor2) {
                            triangles += 1;
                        }
                    }
                }
            }

            let possible_triangles = degree * (degree - 1) / 2;
            triangles as f64 / possible_triangles as f64
        } else {
            0.0
        }
    }

    /// Check if network is connected
    pub fn is_connected(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }

        let start_node = self.nodes.iter().next().unwrap();
        let reachable = self.bfs_reachable(start_node);
        reachable.len() == self.nodes.len()
    }

    /// BFS to find reachable nodes
    fn bfs_reachable(&self, start: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start.to_string());
        visited.insert(start.to_string());

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.adjacency_list.get(&current) {
                for (neighbor, _) in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        visited
    }

    /// Calculate network density
    pub fn calculate_density(&self) -> f64 {
        let n = self.nodes.len() as f64;
        if n <= 1.0 {
            return 0.0;
        }

        let max_edges = if self.is_directed {
            n * (n - 1.0)
        } else {
            n * (n - 1.0) / 2.0
        };

        self.edges.len() as f64 / max_edges
    }

    /// Get degree distribution
    pub fn get_degree_distribution(&self) -> Vec<(usize, usize)> {
        let mut degree_counts = HashMap::new();

        for node in &self.nodes {
            let degree = self.adjacency_list.get(node).map_or(0, |neighbors| neighbors.len());
            *degree_counts.entry(degree).or_insert(0) += 1;
        }

        let mut distribution: Vec<_> = degree_counts.into_iter().collect();
        distribution.sort_by_key(|(degree, _)| *degree);
        distribution
    }

    /// Comprehensive network analysis
    pub fn analyze_network(&self, top_nodes: usize) -> NetworkAnalysisResult {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();
        let is_connected = self.is_connected();
        let density = self.calculate_density();
        let average_clustering = self.calculate_clustering_coefficient();

        // Calculate centralities
        let degree_centrality = self.calculate_degree_centrality();
        let betweenness_centrality = self.calculate_betweenness_centrality();
        let closeness_centrality = self.calculate_closeness_centrality();
        let eigenvector_centrality = self.calculate_eigenvector_centrality(100, 1e-6);

        // Combine centrality metrics
        let mut centrality_metrics: Vec<CentralityMetrics> = self.nodes
            .iter()
            .map(|node| {
                let degree = self.adjacency_list.get(node).map_or(0, |neighbors| neighbors.len());
                let dc = degree_centrality.get(node).copied().unwrap_or(0.0);
                let bc = betweenness_centrality.get(node).copied().unwrap_or(0.0);
                let cc = closeness_centrality.get(node).copied().unwrap_or(0.0);
                let ec = eigenvector_centrality.get(node).copied().unwrap_or(0.0);
                
                CentralityMetrics {
                    node_id: node.clone(),
                    degree_centrality: dc,
                    betweenness_centrality: bc,
                    closeness_centrality: cc,
                    eigenvector_centrality: ec,
                    pagerank: ec, // Using eigenvector as proxy for PageRank
                    degree,
                    centrality_score: dc + bc + cc + ec,
                }
            })
            .collect();

        // Sort by centrality score
        centrality_metrics.sort_by(|a, b| b.centrality_score.partial_cmp(&a.centrality_score).unwrap());
        centrality_metrics.truncate(top_nodes);

        // Community detection
        let communities = self.detect_communities();

        // Calculate shortest path metrics
        let (avg_shortest_path, diameter) = if is_connected {
            self.calculate_path_metrics()
        } else {
            (f64::NAN, 0)
        };

        let largest_component_size = if is_connected {
            node_count
        } else {
            self.find_largest_component_size()
        };

        let degree_distribution = self.get_degree_distribution();

        NetworkAnalysisResult {
            node_count,
            edge_count,
            is_connected,
            density,
            average_clustering,
            average_shortest_path: avg_shortest_path,
            diameter,
            largest_component_size,
            top_central_nodes: centrality_metrics,
            communities,
            degree_distribution,
        }
    }

    /// Calculate shortest path metrics for connected networks
    fn calculate_path_metrics(&self) -> (f64, usize) {
        let mut all_distances = Vec::new();
        let mut max_distance = 0;

        for node in &self.nodes {
            let distances = self.dijkstra_single_source(node);
            for distance in distances.values() {
                if distance.is_finite() && *distance > 0.0 {
                    all_distances.push(*distance);
                    max_distance = max_distance.max(*distance as usize);
                }
            }
        }

        let avg_distance = if all_distances.is_empty() {
            f64::NAN
        } else {
            all_distances.iter().sum::<f64>() / all_distances.len() as f64
        };

        (avg_distance, max_distance)
    }

    /// Find the size of the largest connected component
    fn find_largest_component_size(&self) -> usize {
        let mut visited = HashSet::new();
        let mut largest_size = 0;

        for node in &self.nodes {
            if !visited.contains(node) {
                let component = self.bfs_reachable(node);
                largest_size = largest_size.max(component.len());
                visited.extend(component);
            }
        }

        largest_size
    }

    /// Compare two networks
    pub fn compare_networks(&self, other: &NetworkAnalyzer) -> NetworkComparison {
        let network1_stats = NetworkStats {
            nodes: self.nodes.len(),
            edges: self.edges.len(),
            density: self.calculate_density(),
            average_clustering: self.calculate_clustering_coefficient(),
        };

        let network2_stats = NetworkStats {
            nodes: other.nodes.len(),
            edges: other.edges.len(),
            density: other.calculate_density(),
            average_clustering: other.calculate_clustering_coefficient(),
        };

        // Find common and unique nodes
        let common_nodes: Vec<String> = self.nodes.intersection(&other.nodes).cloned().collect();
        let unique_nodes_1: Vec<String> = self.nodes.difference(&other.nodes).cloned().collect();
        let unique_nodes_2: Vec<String> = other.nodes.difference(&self.nodes).cloned().collect();

        // Find common edges (simplified)
        let edges1: HashSet<(String, String)> = self.edges.iter()
            .map(|e| (e.source.clone(), e.target.clone()))
            .collect();
        let edges2: HashSet<(String, String)> = other.edges.iter()
            .map(|e| (e.source.clone(), e.target.clone()))
            .collect();

        let common_edges: Vec<(String, String)> = edges1.intersection(&edges2)
            .cloned()
            .collect();

        // Calculate Jaccard similarities
        let node_jaccard = if self.nodes.is_empty() && other.nodes.is_empty() {
            1.0
        } else {
            common_nodes.len() as f64 / self.nodes.union(&other.nodes).count() as f64
        };

        let edge_jaccard = if edges1.is_empty() && edges2.is_empty() {
            1.0
        } else {
            common_edges.len() as f64 / edges1.union(&edges2).count() as f64
        };

        NetworkComparison {
            network1_stats,
            network2_stats,
            common_nodes,
            unique_nodes_1,
            unique_nodes_2,
            common_edges,
            node_jaccard_similarity: node_jaccard,
            edge_jaccard_similarity: edge_jaccard,
        }
    }

    /// Get basic network statistics
    pub fn get_network_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("node_count".to_string(), self.nodes.len() as f64);
        stats.insert("edge_count".to_string(), self.edges.len() as f64);
        stats.insert("density".to_string(), self.calculate_density());
        stats.insert("average_clustering".to_string(), self.calculate_clustering_coefficient());
        stats.insert("is_connected".to_string(), if self.is_connected() { 1.0 } else { 0.0 });
        
        stats
    }
}

impl Default for NetworkAnalyzer {
    fn default() -> Self {
        Self::new(false) // Default to undirected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_analyzer_creation() {
        let analyzer = NetworkAnalyzer::new(false);
        assert_eq!(analyzer.nodes.len(), 0);
        assert_eq!(analyzer.edges.len(), 0);
        assert!(!analyzer.is_directed);
    }

    #[test]
    fn test_add_edge() {
        let mut analyzer = NetworkAnalyzer::new(false);
        analyzer.add_edge("A", "B", 1.0);
        
        assert_eq!(analyzer.nodes.len(), 2);
        assert_eq!(analyzer.edges.len(), 1);
        assert!(analyzer.nodes.contains("A"));
        assert!(analyzer.nodes.contains("B"));
    }

    #[test]
    fn test_degree_centrality() {
        let mut analyzer = NetworkAnalyzer::new(false);
        analyzer.add_edge("A", "B", 1.0);
        analyzer.add_edge("A", "C", 1.0);
        analyzer.add_edge("B", "C", 1.0);
        
        let centrality = analyzer.calculate_degree_centrality();
        assert_eq!(centrality.len(), 3);
        
        // In a triangle, each node should have degree centrality of 1.0
        for value in centrality.values() {
            assert!((value - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_clustering_coefficient() {
        let mut analyzer = NetworkAnalyzer::new(false);
        
        // Create a triangle (should have clustering coefficient of 1.0)
        analyzer.add_edge("A", "B", 1.0);
        analyzer.add_edge("A", "C", 1.0);
        analyzer.add_edge("B", "C", 1.0);
        
        let clustering = analyzer.calculate_clustering_coefficient();
        assert!((clustering - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_network_connectivity() {
        let mut analyzer = NetworkAnalyzer::new(false);
        
        // Connected network
        analyzer.add_edge("A", "B", 1.0);
        analyzer.add_edge("B", "C", 1.0);
        assert!(analyzer.is_connected());
        
        // Add isolated node
        analyzer.nodes.insert("D".to_string());
        assert!(!analyzer.is_connected());
    }

    #[test]
    fn test_density_calculation() {
        let mut analyzer = NetworkAnalyzer::new(false);
        
        // Complete graph with 3 nodes should have density 1.0
        analyzer.add_edge("A", "B", 1.0);
        analyzer.add_edge("A", "C", 1.0);
        analyzer.add_edge("B", "C", 1.0);
        
        let density = analyzer.calculate_density();
        assert!((density - 1.0).abs() < 1e-6);
    }
} 