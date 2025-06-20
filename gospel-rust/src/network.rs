//! Gene network analysis with graph algorithms
//!
//! This module provides high-performance graph algorithms for analyzing
//! gene regulatory networks, protein-protein interactions, and pathway analysis.

use anyhow::Result;
use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::GospelConfig;

/// Gene network representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneNetwork {
    /// Adjacency matrix (genes × genes)
    pub adjacency_matrix: Array2<f64>,
    /// Gene names/identifiers
    pub gene_names: Vec<String>,
    /// Network statistics
    pub stats: NetworkStats,
    /// Community/module assignments
    pub communities: Vec<NetworkCommunity>,
    /// Hub genes (highly connected)
    pub hub_genes: Vec<String>,
}

/// Network analysis statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Number of nodes (genes)
    pub num_nodes: usize,
    /// Number of edges (interactions)
    pub num_edges: usize,
    /// Network density
    pub density: f64,
    /// Average clustering coefficient
    pub clustering_coefficient: f64,
    /// Average path length
    pub average_path_length: f64,
    /// Network modularity
    pub modularity: f64,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

impl Default for NetworkStats {
    fn default() -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            density: 0.0,
            clustering_coefficient: 0.0,
            average_path_length: 0.0,
            modularity: 0.0,
            processing_time_ms: 0,
        }
    }
}

/// Network community/module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCommunity {
    /// Community ID
    pub id: usize,
    /// Genes in this community
    pub genes: Vec<String>,
    /// Intra-community connectivity
    pub internal_connectivity: f64,
    /// Inter-community connectivity
    pub external_connectivity: f64,
    /// Community size
    pub size: usize,
}

/// High-performance network processor
#[derive(Debug)]
pub struct NetworkProcessor {
    config: GospelConfig,
}

impl NetworkProcessor {
    /// Create new network processor
    pub fn new(config: &GospelConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Analyze gene network with comprehensive graph algorithms
    pub async fn analyze_network(
        &self,
        adjacency_matrix: &[f64],
        gene_names: &[String],
    ) -> Result<GeneNetwork> {
        let start_time = std::time::Instant::now();
        
        let num_genes = gene_names.len();
        
        if adjacency_matrix.len() != num_genes * num_genes {
            return Err(anyhow::anyhow!(
                "Adjacency matrix size mismatch: expected {}, got {}",
                num_genes * num_genes,
                adjacency_matrix.len()
            ));
        }

        // Convert to ndarray
        let adj_matrix = Array2::from_shape_vec((num_genes, num_genes), adjacency_matrix.to_vec())?;
        
        // Calculate network statistics
        let stats = self.calculate_network_stats(&adj_matrix, start_time.elapsed()).await;
        
        // Find communities using Louvain algorithm
        let communities = self.find_communities(&adj_matrix, gene_names).await?;
        
        // Identify hub genes
        let hub_genes = self.identify_hub_genes(&adj_matrix, gene_names, 0.8).await;
        
        Ok(GeneNetwork {
            adjacency_matrix: adj_matrix,
            gene_names: gene_names.to_vec(),
            stats,
            communities,
            hub_genes,
        })
    }

    /// Calculate shortest paths using Floyd-Warshall algorithm
    pub async fn shortest_paths(&self, network: &GeneNetwork) -> Result<Array2<f64>> {
        let paths = tokio::task::spawn_blocking({
            let adj_matrix = network.adjacency_matrix.clone();
            move || {
                let n = adj_matrix.nrows();
                let mut dist = Array2::from_elem((n, n), f64::INFINITY);
                
                // Initialize distances
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            dist[[i, j]] = 0.0;
                        } else if adj_matrix[[i, j]] > 0.0 {
                            dist[[i, j]] = 1.0 / adj_matrix[[i, j]]; // Convert weight to distance
                        }
                    }
                }
                
                // Floyd-Warshall algorithm
                for k in 0..n {
                    for i in 0..n {
                        for j in 0..n {
                            if dist[[i, k]] + dist[[k, j]] < dist[[i, j]] {
                                dist[[i, j]] = dist[[i, k]] + dist[[k, j]];
                            }
                        }
                    }
                }
                
                dist
            }
        }).await?;
        
        Ok(paths)
    }

    /// Calculate centrality measures for all nodes
    pub async fn centrality_measures(&self, network: &GeneNetwork) -> Result<HashMap<String, CentralityMeasures>> {
        let measures = tokio::task::spawn_blocking({
            let adj_matrix = network.adjacency_matrix.clone();
            let gene_names = network.gene_names.clone();
            
            move || {
                let mut centralities = HashMap::new();
                let n = adj_matrix.nrows();
                
                // Calculate degree centrality
                let degrees: Vec<f64> = (0..n)
                    .map(|i| adj_matrix.row(i).sum())
                    .collect();
                
                // Calculate betweenness centrality (simplified)
                let betweenness = Self::calculate_betweenness_centrality(&adj_matrix);
                
                // Calculate eigenvector centrality
                let eigenvector = Self::calculate_eigenvector_centrality(&adj_matrix);
                
                for i in 0..n {
                    centralities.insert(
                        gene_names[i].clone(),
                        CentralityMeasures {
                            degree: degrees[i],
                            betweenness: betweenness[i],
                            eigenvector: eigenvector[i],
                            closeness: Self::calculate_closeness_centrality(&adj_matrix, i),
                        },
                    );
                }
                
                centralities
            }
        }).await?;
        
        Ok(measures)
    }

    /// Find network motifs (small recurring patterns)
    pub async fn find_motifs(&self, network: &GeneNetwork) -> Result<Vec<NetworkMotif>> {
        let motifs = tokio::task::spawn_blocking({
            let adj_matrix = network.adjacency_matrix.clone();
            let gene_names = network.gene_names.clone();
            
            move || {
                let mut motifs = Vec::new();
                let n = adj_matrix.nrows();
                
                // Find 3-node motifs (triangles, feed-forward loops, etc.)
                for i in 0..n {
                    for j in (i + 1)..n {
                        for k in (j + 1)..n {
                            let motif = Self::classify_3node_motif(&adj_matrix, i, j, k);
                            if motif.motif_type != MotifType::None {
                                motifs.push(NetworkMotif {
                                    motif_type: motif.motif_type,
                                    genes: vec![
                                        gene_names[i].clone(),
                                        gene_names[j].clone(),
                                        gene_names[k].clone(),
                                    ],
                                    strength: motif.strength,
                                });
                            }
                        }
                    }
                }
                
                motifs
            }
        }).await?;
        
        Ok(motifs)
    }

    /// Calculate network statistics
    async fn calculate_network_stats(&self, adj_matrix: &Array2<f64>, processing_time: std::time::Duration) -> NetworkStats {
        let stats = tokio::task::spawn_blocking({
            let adj_matrix = adj_matrix.clone();
            move || {
                let n = adj_matrix.nrows();
                let num_nodes = n;
                
                // Count edges
                let num_edges = adj_matrix.iter()
                    .filter(|&&x| x > 0.0)
                    .count() / 2; // Assuming undirected network
                
                // Calculate density
                let max_edges = n * (n - 1) / 2;
                let density = if max_edges > 0 {
                    num_edges as f64 / max_edges as f64
                } else {
                    0.0
                };
                
                // Calculate clustering coefficient
                let clustering_coefficient = Self::calculate_clustering_coefficient(&adj_matrix);
                
                // Calculate average path length (simplified)
                let average_path_length = Self::calculate_average_path_length(&adj_matrix);
                
                NetworkStats {
                    num_nodes,
                    num_edges,
                    density,
                    clustering_coefficient,
                    average_path_length,
                    modularity: 0.0, // Will be calculated with communities
                    processing_time_ms: processing_time.as_millis() as u64,
                }
            }
        }).await.unwrap();
        
        stats
    }

    /// Find communities using simplified Louvain algorithm
    async fn find_communities(&self, adj_matrix: &Array2<f64>, gene_names: &[String]) -> Result<Vec<NetworkCommunity>> {
        let communities = tokio::task::spawn_blocking({
            let adj_matrix = adj_matrix.clone();
            let gene_names = gene_names.to_vec();
            
            move || {
                let n = adj_matrix.nrows();
                let mut community_assignments = (0..n).collect::<Vec<_>>(); // Each node starts in its own community
                
                // Simplified community detection (greedy modularity optimization)
                let mut improved = true;
                while improved {
                    improved = false;
                    
                    for node in 0..n {
                        let current_community = community_assignments[node];
                        let mut best_community = current_community;
                        let mut best_modularity_gain = 0.0;
                        
                        // Try moving node to each neighbor's community
                        for neighbor in 0..n {
                            if adj_matrix[[node, neighbor]] > 0.0 {
                                let neighbor_community = community_assignments[neighbor];
                                if neighbor_community != current_community {
                                    let modularity_gain = Self::calculate_modularity_gain(
                                        &adj_matrix,
                                        &community_assignments,
                                        node,
                                        neighbor_community,
                                    );
                                    
                                    if modularity_gain > best_modularity_gain {
                                        best_modularity_gain = modularity_gain;
                                        best_community = neighbor_community;
                                    }
                                }
                            }
                        }
                        
                        if best_community != current_community {
                            community_assignments[node] = best_community;
                            improved = true;
                        }
                    }
                }
                
                // Convert to community structures
                let mut communities_map: HashMap<usize, Vec<usize>> = HashMap::new();
                for (node, &community) in community_assignments.iter().enumerate() {
                    communities_map.entry(community).or_insert_with(Vec::new).push(node);
                }
                
                communities_map
                    .into_iter()
                    .enumerate()
                    .map(|(id, (_, nodes))| {
                        let genes = nodes.iter().map(|&i| gene_names[i].clone()).collect();
                        let size = nodes.len();
                        
                        // Calculate internal and external connectivity
                        let (internal, external) = Self::calculate_community_connectivity(&adj_matrix, &nodes);
                        
                        NetworkCommunity {
                            id,
                            genes,
                            internal_connectivity: internal,
                            external_connectivity: external,
                            size,
                        }
                    })
                    .collect()
            }
        }).await?;
        
        Ok(communities)
    }

    /// Identify hub genes based on degree centrality
    async fn identify_hub_genes(&self, adj_matrix: &Array2<f64>, gene_names: &[String], percentile: f64) -> Vec<String> {
        let hubs = tokio::task::spawn_blocking({
            let adj_matrix = adj_matrix.clone();
            let gene_names = gene_names.to_vec();
            
            move || {
                let degrees: Vec<(usize, f64)> = (0..adj_matrix.nrows())
                    .map(|i| (i, adj_matrix.row(i).sum()))
                    .collect();
                
                let mut sorted_degrees = degrees;
                sorted_degrees.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                let threshold_index = ((1.0 - percentile) * sorted_degrees.len() as f64) as usize;
                
                sorted_degrees
                    .into_iter()
                    .take(threshold_index.max(1))
                    .map(|(i, _)| gene_names[i].clone())
                    .collect()
            }
        }).await.unwrap();
        
        hubs
    }

    /// Calculate clustering coefficient
    fn calculate_clustering_coefficient(adj_matrix: &Array2<f64>) -> f64 {
        let n = adj_matrix.nrows();
        let mut total_clustering = 0.0;
        let mut valid_nodes = 0;
        
        for i in 0..n {
            let neighbors: Vec<usize> = (0..n)
                .filter(|&j| j != i && adj_matrix[[i, j]] > 0.0)
                .collect();
            
            if neighbors.len() < 2 {
                continue; // Need at least 2 neighbors for clustering
            }
            
            let mut triangles = 0;
            let possible_triangles = neighbors.len() * (neighbors.len() - 1) / 2;
            
            for j in 0..neighbors.len() {
                for k in (j + 1)..neighbors.len() {
                    if adj_matrix[[neighbors[j], neighbors[k]]] > 0.0 {
                        triangles += 1;
                    }
                }
            }
            
            total_clustering += triangles as f64 / possible_triangles as f64;
            valid_nodes += 1;
        }
        
        if valid_nodes > 0 {
            total_clustering / valid_nodes as f64
        } else {
            0.0
        }
    }

    /// Calculate average path length using BFS
    fn calculate_average_path_length(adj_matrix: &Array2<f64>) -> f64 {
        let n = adj_matrix.nrows();
        let mut total_path_length = 0.0;
        let mut path_count = 0;
        
        for start in 0..n {
            let distances = Self::bfs_distances(adj_matrix, start);
            for &dist in &distances {
                if dist > 0.0 && dist != f64::INFINITY {
                    total_path_length += dist;
                    path_count += 1;
                }
            }
        }
        
        if path_count > 0 {
            total_path_length / path_count as f64
        } else {
            0.0
        }
    }

    /// BFS to calculate distances from a source node
    fn bfs_distances(adj_matrix: &Array2<f64>, start: usize) -> Vec<f64> {
        let n = adj_matrix.nrows();
        let mut distances = vec![f64::INFINITY; n];
        let mut queue = VecDeque::new();
        
        distances[start] = 0.0;
        queue.push_back(start);
        
        while let Some(current) = queue.pop_front() {
            for neighbor in 0..n {
                if adj_matrix[[current, neighbor]] > 0.0 && distances[neighbor] == f64::INFINITY {
                    distances[neighbor] = distances[current] + 1.0;
                    queue.push_back(neighbor);
                }
            }
        }
        
        distances
    }

    /// Calculate betweenness centrality (simplified)
    fn calculate_betweenness_centrality(adj_matrix: &Array2<f64>) -> Vec<f64> {
        let n = adj_matrix.nrows();
        let mut betweenness = vec![0.0; n];
        
        // For each pair of nodes, find shortest paths and count how many pass through each node
        for s in 0..n {
            for t in (s + 1)..n {
                let paths = Self::find_shortest_paths(adj_matrix, s, t);
                for path in paths {
                    for &node in &path[1..path.len()-1] { // Exclude source and target
                        betweenness[node] += 1.0;
                    }
                }
            }
        }
        
        // Normalize
        let normalization = (n - 1) * (n - 2) / 2;
        if normalization > 0 {
            for bc in &mut betweenness {
                *bc /= normalization as f64;
            }
        }
        
        betweenness
    }

    /// Find shortest paths between two nodes (simplified)
    fn find_shortest_paths(adj_matrix: &Array2<f64>, start: usize, end: usize) -> Vec<Vec<usize>> {
        // Simplified: return single shortest path using BFS
        let n = adj_matrix.nrows();
        let mut visited = vec![false; n];
        let mut parent = vec![None; n];
        let mut queue = VecDeque::new();
        
        visited[start] = true;
        queue.push_back(start);
        
        while let Some(current) = queue.pop_front() {
            if current == end {
                break;
            }
            
            for neighbor in 0..n {
                if adj_matrix[[current, neighbor]] > 0.0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    parent[neighbor] = Some(current);
                    queue.push_back(neighbor);
                }
            }
        }
        
        // Reconstruct path
        if !visited[end] {
            return vec![]; // No path found
        }
        
        let mut path = Vec::new();
        let mut current = end;
        
        while let Some(p) = parent[current] {
            path.push(current);
            current = p;
        }
        path.push(start);
        path.reverse();
        
        vec![path]
    }

    /// Calculate eigenvector centrality (simplified power iteration)
    fn calculate_eigenvector_centrality(adj_matrix: &Array2<f64>) -> Vec<f64> {
        let n = adj_matrix.nrows();
        let mut centrality = vec![1.0; n];
        
        // Power iteration
        for _ in 0..100 {
            let mut new_centrality = vec![0.0; n];
            
            for i in 0..n {
                for j in 0..n {
                    new_centrality[i] += adj_matrix[[i, j]] * centrality[j];
                }
            }
            
            // Normalize
            let norm: f64 = new_centrality.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for c in &mut new_centrality {
                    *c /= norm;
                }
            }
            
            centrality = new_centrality;
        }
        
        centrality
    }

    /// Calculate closeness centrality for a node
    fn calculate_closeness_centrality(adj_matrix: &Array2<f64>, node: usize) -> f64 {
        let distances = Self::bfs_distances(adj_matrix, node);
        let reachable_distances: Vec<f64> = distances
            .into_iter()
            .filter(|&d| d != f64::INFINITY && d > 0.0)
            .collect();
        
        if reachable_distances.is_empty() {
            0.0
        } else {
            let sum_distances: f64 = reachable_distances.iter().sum();
            reachable_distances.len() as f64 / sum_distances
        }
    }

    /// Calculate modularity gain for moving a node to a community
    fn calculate_modularity_gain(
        adj_matrix: &Array2<f64>,
        communities: &[usize],
        node: usize,
        target_community: usize,
    ) -> f64 {
        // Simplified modularity calculation
        let total_edges: f64 = adj_matrix.iter().sum::<f64>() / 2.0;
        if total_edges == 0.0 {
            return 0.0;
        }
        
        let node_degree: f64 = adj_matrix.row(node).sum();
        
        // Calculate connections to target community
        let mut connections_to_target = 0.0;
        let mut target_community_degree = 0.0;
        
        for (i, &comm) in communities.iter().enumerate() {
            if comm == target_community {
                connections_to_target += adj_matrix[[node, i]];
                target_community_degree += adj_matrix.row(i).sum();
            }
        }
        
        // Simplified modularity gain calculation
        let expected = (node_degree * target_community_degree) / (2.0 * total_edges);
        (connections_to_target - expected) / total_edges
    }

    /// Calculate community connectivity
    fn calculate_community_connectivity(adj_matrix: &Array2<f64>, nodes: &[usize]) -> (f64, f64) {
        let mut internal = 0.0;
        let mut external = 0.0;
        
        for &i in nodes {
            for &j in nodes {
                if i != j {
                    internal += adj_matrix[[i, j]];
                }
            }
            
            for j in 0..adj_matrix.ncols() {
                if !nodes.contains(&j) {
                    external += adj_matrix[[i, j]];
                }
            }
        }
        
        (internal / 2.0, external) // Internal edges counted twice
    }

    /// Classify 3-node motif
    fn classify_3node_motif(adj_matrix: &Array2<f64>, i: usize, j: usize, k: usize) -> MotifClassification {
        let edges = [
            adj_matrix[[i, j]] > 0.0,
            adj_matrix[[j, i]] > 0.0,
            adj_matrix[[i, k]] > 0.0,
            adj_matrix[[k, i]] > 0.0,
            adj_matrix[[j, k]] > 0.0,
            adj_matrix[[k, j]] > 0.0,
        ];
        
        let edge_count = edges.iter().filter(|&&e| e).count();
        
        match edge_count {
            0 => MotifClassification { motif_type: MotifType::None, strength: 0.0 },
            3 => {
                // Check for triangle (all bidirectional)
                if edges[0] && edges[1] && edges[2] && edges[3] && edges[4] && edges[5] {
                    MotifClassification { motif_type: MotifType::Triangle, strength: 1.0 }
                } else {
                    MotifClassification { motif_type: MotifType::FeedForwardLoop, strength: 0.5 }
                }
            },
            _ => MotifClassification { motif_type: MotifType::Other, strength: edge_count as f64 / 6.0 },
        }
    }

    /// Get processor statistics
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("processor_type".to_string(), "NetworkProcessor".to_string());
        stats.insert("simd_enabled".to_string(), self.config.enable_simd.to_string());
        stats.insert("num_threads".to_string(), self.config.num_threads.to_string());
        stats
    }
}

/// Centrality measures for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityMeasures {
    /// Degree centrality
    pub degree: f64,
    /// Betweenness centrality
    pub betweenness: f64,
    /// Eigenvector centrality
    pub eigenvector: f64,
    /// Closeness centrality
    pub closeness: f64,
}

/// Network motif
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMotif {
    /// Type of motif
    pub motif_type: MotifType,
    /// Genes involved in the motif
    pub genes: Vec<String>,
    /// Motif strength/confidence
    pub strength: f64,
}

/// Types of network motifs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MotifType {
    None,
    Triangle,
    FeedForwardLoop,
    Other,
}

/// Motif classification result
struct MotifClassification {
    motif_type: MotifType,
    strength: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GospelConfig;

    #[tokio::test]
    async fn test_network_processor() {
        let config = GospelConfig::default();
        let processor = NetworkProcessor::new(&config).unwrap();
        
        // Create test network: 3 nodes, fully connected
        let adjacency_matrix = vec![
            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
        ];
        let gene_names = vec!["GENE1".to_string(), "GENE2".to_string(), "GENE3".to_string()];
        
        let network = processor.analyze_network(&adjacency_matrix, &gene_names).await.unwrap();
        
        assert_eq!(network.gene_names.len(), 3);
        assert_eq!(network.stats.num_nodes, 3);
        assert_eq!(network.stats.num_edges, 3); // 3 edges in fully connected triangle
        assert!(network.stats.density > 0.9); // Should be close to 1.0
        assert!(network.stats.clustering_coefficient > 0.9); // Perfect clustering
    }

    #[tokio::test]
    async fn test_centrality_measures() {
        let config = GospelConfig::default();
        let processor = NetworkProcessor::new(&config).unwrap();
        
        // Create star network: one central node connected to 3 others
        let adjacency_matrix = vec![
            0.0, 1.0, 1.0, 1.0,
            1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
        ];
        let gene_names = vec!["HUB".to_string(), "GENE1".to_string(), "GENE2".to_string(), "GENE3".to_string()];
        
        let network = processor.analyze_network(&adjacency_matrix, &gene_names).await.unwrap();
        let centralities = processor.centrality_measures(&network).await.unwrap();
        
        // Hub gene should have highest degree centrality
        let hub_centrality = &centralities["HUB"];
        let gene1_centrality = &centralities["GENE1"];
        
        assert!(hub_centrality.degree > gene1_centrality.degree);
        assert!(hub_centrality.betweenness > gene1_centrality.betweenness);
    }

    #[tokio::test]
    async fn test_shortest_paths() {
        let config = GospelConfig::default();
        let processor = NetworkProcessor::new(&config).unwrap();
        
        // Create linear network: 1-2-3-4
        let adjacency_matrix = vec![
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        let gene_names = vec!["GENE1".to_string(), "GENE2".to_string(), "GENE3".to_string(), "GENE4".to_string()];
        
        let network = processor.analyze_network(&adjacency_matrix, &gene_names).await.unwrap();
        let paths = processor.shortest_paths(&network).await.unwrap();
        
        // Distance from node 0 to node 3 should be 3
        assert!((paths[[0, 3]] - 3.0).abs() < 1e-10);
        
        // Distance from node 1 to node 3 should be 2
        assert!((paths[[1, 3]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_clustering_coefficient() {
        // Test triangle (perfect clustering)
        let triangle = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        ).unwrap();
        
        let clustering = NetworkProcessor::calculate_clustering_coefficient(&triangle);
        assert!((clustering - 1.0).abs() < 1e-10);
        
        // Test star (no clustering for peripheral nodes)
        let star = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 1.0, 1.0,
                1.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0,
            ],
        ).unwrap();
        
        let clustering = NetworkProcessor::calculate_clustering_coefficient(&star);
        assert!(clustering < 0.1); // Should be very low
    }
} 