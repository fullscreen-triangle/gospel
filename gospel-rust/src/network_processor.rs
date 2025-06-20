use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use rayon::prelude::*;
use tokio::time::Instant;
use reqwest;
use futures::future::join_all;

/// Protein-protein interaction data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinInteraction {
    pub gene_a: String,
    pub gene_b: String,
    pub score: f64,
    pub interaction_type: String,
    pub evidence: String,
    pub source: String,
    pub publication: Option<String>,
}

/// Pathway data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwayData {
    pub gene_id: String,
    pub pathway_id: String,
    pub pathway_name: String,
    pub species: String,
    pub source: String,
    pub diagram_url: Option<String>,
}

/// Proteomics data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteomicsData {
    pub gene_id: String,
    pub uniprot_id: String,
    pub protein_name: String,
    pub functions: Vec<String>,
    pub subcellular_locations: Vec<String>,
    pub domains: Vec<ProteinDomain>,
    pub source: String,
}

/// Protein domain information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinDomain {
    pub domain_type: String,
    pub description: String,
    pub start_position: usize,
    pub end_position: usize,
}

/// Training example for LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub prompt: String,
    pub response: String,
    pub gene_id: String,
    pub example_type: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Gene network data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneNetworkData {
    pub gene_id: String,
    pub interactions: Vec<ProteinInteraction>,
    pub pathways: Vec<PathwayData>,
    pub proteomics: Option<ProteomicsData>,
    pub network_metrics: NetworkMetrics,
}

/// Network metrics for a gene
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub interaction_count: usize,
    pub pathway_count: usize,
    pub centrality_score: f64,
    pub clustering_coefficient: f64,
}

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub genes_processed: usize,
    pub interactions_found: usize,
    pub pathways_found: usize,
    pub proteomics_found: usize,
    pub training_examples_generated: usize,
    pub processing_time_seconds: f64,
    pub errors: Vec<String>,
}

/// High-performance network data processor
pub struct NetworkDataProcessor {
    client: reqwest::Client,
    cache: Arc<tokio::sync::RwLock<HashMap<String, serde_json::Value>>>,
    max_concurrent_requests: usize,
    request_delay_ms: u64,
    memory_limit_mb: usize,
    current_memory_usage: Arc<std::sync::atomic::AtomicUsize>,
}

impl NetworkDataProcessor {
    /// Create a new network data processor
    pub fn new(
        max_concurrent_requests: usize,
        request_delay_ms: u64,
        memory_limit_mb: usize,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            max_concurrent_requests,
            request_delay_ms,
            memory_limit_mb,
            current_memory_usage: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    /// Fetch protein interactions from STRING database
    pub async fn fetch_string_interactions(&self, gene_id: &str) -> Result<Vec<ProteinInteraction>> {
        let cache_key = format!("string_{}", gene_id);
        
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached_data) = cache.get(&cache_key) {
                if let Ok(interactions) = serde_json::from_value::<Vec<ProteinInteraction>>(cached_data.clone()) {
                    return Ok(interactions);
                }
            }
        }

        let url = "https://string-db.org/api/json/network";
        let params = [
            ("identifiers", gene_id),
            ("species", "9606"), // Human
            ("required_score", "700"), // High confidence
            ("network_type", "physical"),
        ];

        tokio::time::sleep(tokio::time::Duration::from_millis(self.request_delay_ms)).await;

        let response = self.client
            .get(url)
            .query(&params)
            .send()
            .await
            .context("Failed to fetch STRING data")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("STRING API error: {}", response.status()));
        }

        let data: serde_json::Value = response.json().await
            .context("Failed to parse STRING response")?;

        let mut interactions = Vec::new();
        
        if let Some(items) = data.as_array() {
            for item in items {
                if let (Some(gene_a), Some(gene_b), Some(score)) = (
                    item.get("preferredName_A").and_then(|v| v.as_str()),
                    item.get("preferredName_B").and_then(|v| v.as_str()),
                    item.get("score").and_then(|v| v.as_f64())
                ) {
                    interactions.push(ProteinInteraction {
                        gene_a: gene_a.to_string(),
                        gene_b: gene_b.to_string(),
                        score: score / 1000.0, // Normalize to 0-1
                        interaction_type: "physical_interaction".to_string(),
                        evidence: "experimental".to_string(),
                        source: "STRING".to_string(),
                        publication: None,
                    });
                }
            }
        }

        // Cache results
        {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, serde_json::to_value(&interactions)?);
        }

        Ok(interactions)
    }

    /// Fetch pathway data from Reactome
    pub async fn fetch_reactome_pathways(&self, gene_id: &str) -> Result<Vec<PathwayData>> {
        let cache_key = format!("reactome_{}", gene_id);
        
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached_data) = cache.get(&cache_key) {
                if let Ok(pathways) = serde_json::from_value::<Vec<PathwayData>>(cached_data.clone()) {
                    return Ok(pathways);
                }
            }
        }

        let url = format!("https://reactome.org/ContentService/data/pathways/low/entity/{}/allForms", gene_id);

        tokio::time::sleep(tokio::time::Duration::from_millis(self.request_delay_ms)).await;

        let response = self.client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch Reactome data")?;

        if !response.status().is_success() {
            return Ok(Vec::new()); // Return empty if gene not found
        }

        let data: serde_json::Value = response.json().await
            .context("Failed to parse Reactome response")?;

        let mut pathways = Vec::new();

        if let Some(items) = data.as_array() {
            for item in items {
                if let (Some(pathway_id), Some(pathway_name)) = (
                    item.get("stId").and_then(|v| v.as_str()),
                    item.get("displayName").and_then(|v| v.as_str())
                ) {
                    let species = item.get("species")
                        .and_then(|s| s.get("displayName"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("Unknown");

                    pathways.push(PathwayData {
                        gene_id: gene_id.to_string(),
                        pathway_id: pathway_id.to_string(),
                        pathway_name: pathway_name.to_string(),
                        species: species.to_string(),
                        source: "Reactome".to_string(),
                        diagram_url: Some(format!("https://reactome.org/ContentService/diagram/{}.png", pathway_id)),
                    });
                }
            }
        }

        // Cache results
        {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, serde_json::to_value(&pathways)?);
        }

        Ok(pathways)
    }

    /// Fetch proteomics data from UniProt
    pub async fn fetch_uniprot_data(&self, gene_id: &str) -> Result<Option<ProteomicsData>> {
        let cache_key = format!("uniprot_{}", gene_id);
        
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached_data) = cache.get(&cache_key) {
                if let Ok(proteomics) = serde_json::from_value::<Option<ProteomicsData>>(cached_data.clone()) {
                    return Ok(proteomics);
                }
            }
        }

        let url = "https://rest.uniprot.org/uniprotkb/search";
        let params = [
            ("query", &format!("gene:{} AND organism_id:9606", gene_id)),
            ("format", "json"),
            ("size", "1"), // Get only the first result
        ];

        tokio::time::sleep(tokio::time::Duration::from_millis(self.request_delay_ms)).await;

        let response = self.client
            .get(url)
            .query(&params)
            .send()
            .await
            .context("Failed to fetch UniProt data")?;

        if !response.status().is_success() {
            return Ok(None);
        }

        let data: serde_json::Value = response.json().await
            .context("Failed to parse UniProt response")?;

        let proteomics_data = if let Some(results) = data.get("results").and_then(|r| r.as_array()) {
            if let Some(first_result) = results.first() {
                let uniprot_id = first_result.get("primaryAccession")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");

                let protein_name = first_result
                    .get("proteinDescription")
                    .and_then(|pd| pd.get("recommendedName"))
                    .and_then(|rn| rn.get("fullName"))
                    .and_then(|fn_obj| fn_obj.get("value"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");

                // Extract functions
                let mut functions = Vec::new();
                if let Some(comments) = first_result.get("comments").and_then(|c| c.as_array()) {
                    for comment in comments {
                        if comment.get("commentType").and_then(|ct| ct.as_str()) == Some("FUNCTION") {
                            if let Some(texts) = comment.get("texts").and_then(|t| t.as_array()) {
                                for text in texts {
                                    if let Some(value) = text.get("value").and_then(|v| v.as_str()) {
                                        functions.push(value.to_string());
                                    }
                                }
                            }
                        }
                    }
                }

                // Extract subcellular locations
                let mut subcellular_locations = Vec::new();
                if let Some(comments) = first_result.get("comments").and_then(|c| c.as_array()) {
                    for comment in comments {
                        if comment.get("commentType").and_then(|ct| ct.as_str()) == Some("SUBCELLULAR LOCATION") {
                            if let Some(subcellular_locs) = comment.get("subcellularLocations").and_then(|sl| sl.as_array()) {
                                for location in subcellular_locs {
                                    if let Some(loc_value) = location.get("location").and_then(|l| l.get("value")).and_then(|v| v.as_str()) {
                                        subcellular_locations.push(loc_value.to_string());
                                    }
                                }
                            }
                        }
                    }
                }

                // Extract domains
                let mut domains = Vec::new();
                if let Some(features) = first_result.get("features").and_then(|f| f.as_array()) {
                    for feature in features {
                        if feature.get("type").and_then(|t| t.as_str()) == Some("DOMAIN") {
                            let description = feature.get("description").and_then(|d| d.as_str()).unwrap_or("Unknown");
                            let start_pos = feature.get("location")
                                .and_then(|l| l.get("start"))
                                .and_then(|s| s.get("value"))
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as usize;
                            let end_pos = feature.get("location")
                                .and_then(|l| l.get("end"))
                                .and_then(|e| e.get("value"))
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as usize;

                            domains.push(ProteinDomain {
                                domain_type: "DOMAIN".to_string(),
                                description: description.to_string(),
                                start_position: start_pos,
                                end_position: end_pos,
                            });
                        }
                    }
                }

                Some(ProteomicsData {
                    gene_id: gene_id.to_string(),
                    uniprot_id: uniprot_id.to_string(),
                    protein_name: protein_name.to_string(),
                    functions,
                    subcellular_locations,
                    domains,
                    source: "UniProt".to_string(),
                })
            } else {
                None
            }
        } else {
            None
        };

        // Cache results
        {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, serde_json::to_value(&proteomics_data)?);
        }

        Ok(proteomics_data)
    }

    /// Process a single gene to collect all network data
    pub async fn process_gene(&self, gene_id: &str) -> Result<GeneNetworkData> {
        // Fetch all data in parallel
        let (interactions_result, pathways_result, proteomics_result) = tokio::join!(
            self.fetch_string_interactions(gene_id),
            self.fetch_reactome_pathways(gene_id),
            self.fetch_uniprot_data(gene_id)
        );

        let interactions = interactions_result.unwrap_or_else(|_| Vec::new());
        let pathways = pathways_result.unwrap_or_else(|_| Vec::new());
        let proteomics = proteomics_result.unwrap_or(None);

        // Calculate network metrics
        let network_metrics = self.calculate_network_metrics(&interactions, &pathways);

        Ok(GeneNetworkData {
            gene_id: gene_id.to_string(),
            interactions,
            pathways,
            proteomics,
            network_metrics,
        })
    }

    /// Calculate network metrics for a gene
    fn calculate_network_metrics(&self, interactions: &[ProteinInteraction], pathways: &[PathwayData]) -> NetworkMetrics {
        let interaction_count = interactions.len();
        let pathway_count = pathways.len();

        // Simple centrality score based on number of interactions and pathways
        let centrality_score = (interaction_count as f64).ln() + (pathway_count as f64).ln();

        // Simple clustering coefficient approximation
        let clustering_coefficient = if interaction_count > 1 {
            1.0 / (interaction_count as f64).sqrt()
        } else {
            0.0
        };

        NetworkMetrics {
            interaction_count,
            pathway_count,
            centrality_score,
            clustering_coefficient,
        }
    }

    /// Process multiple genes in parallel with memory management
    pub async fn process_genes_batch(
        &self,
        gene_ids: &[String],
        batch_size: usize,
    ) -> Result<(Vec<GeneNetworkData>, ProcessingStats)> {
        let start_time = Instant::now();
        let mut all_results = Vec::new();
        let mut errors = Vec::new();
        let mut total_interactions = 0;
        let mut total_pathways = 0;
        let mut total_proteomics = 0;

        // Process genes in batches to manage memory and API rate limits
        for batch in gene_ids.chunks(batch_size) {
            // Use semaphore to limit concurrent requests
            let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrent_requests));
            
            let batch_futures: Vec<_> = batch
                .iter()
                .map(|gene_id| {
                    let semaphore = semaphore.clone();
                    let gene_id = gene_id.clone();
                    async move {
                        let _permit = semaphore.acquire().await.unwrap();
                        self.process_gene(&gene_id).await
                    }
                })
                .collect();

            let batch_results = join_all(batch_futures).await;

            for (i, result) in batch_results.into_iter().enumerate() {
                match result {
                    Ok(gene_data) => {
                        total_interactions += gene_data.interactions.len();
                        total_pathways += gene_data.pathways.len();
                        if gene_data.proteomics.is_some() {
                            total_proteomics += 1;
                        }
                        all_results.push(gene_data);
                    }
                    Err(e) => {
                        errors.push(format!("Error processing gene {}: {}", batch[i], e));
                    }
                }
            }

            // Small delay between batches
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        let processing_time = start_time.elapsed().as_secs_f64();

        let stats = ProcessingStats {
            genes_processed: all_results.len(),
            interactions_found: total_interactions,
            pathways_found: total_pathways,
            proteomics_found: total_proteomics,
            training_examples_generated: 0,
            processing_time_seconds: processing_time,
            errors,
        };

        Ok((all_results, stats))
    }

    /// Generate training examples from network data
    pub fn generate_training_examples(&self, network_data: &[GeneNetworkData]) -> Vec<TrainingExample> {
        network_data
            .par_iter()
            .flat_map(|gene_data| self.create_training_examples_for_gene(gene_data))
            .collect()
    }

    /// Create training examples for a single gene
    fn create_training_examples_for_gene(&self, gene_data: &GeneNetworkData) -> Vec<TrainingExample> {
        let mut examples = Vec::new();
        let gene_id = &gene_data.gene_id;

        // Example 1: Protein-protein interactions
        if !gene_data.interactions.is_empty() {
            let interaction_partners: Vec<String> = gene_data.interactions
                .iter()
                .map(|i| if i.gene_a == *gene_id { i.gene_b.clone() } else { i.gene_a.clone() })
                .take(5)
                .collect();

            let mut metadata = HashMap::new();
            metadata.insert("interaction_count".to_string(), serde_json::Value::Number(
                serde_json::Number::from(gene_data.interactions.len())
            ));

            examples.push(TrainingExample {
                prompt: format!("What proteins interact with {} and what are their functions?", gene_id),
                response: format!(
                    "{} interacts with {} proteins in the human interactome. Key interaction partners include {}. These interactions are supported by experimental evidence from STRING database, indicating physical and functional relationships that are crucial for understanding {}'s role in cellular networks.",
                    gene_id,
                    gene_data.interactions.len(),
                    interaction_partners.join(", "),
                    gene_id
                ),
                gene_id: gene_id.clone(),
                example_type: "protein_interactions".to_string(),
                metadata,
            });
        }

        // Example 2: Biological pathways
        if !gene_data.pathways.is_empty() {
            let pathway_names: Vec<String> = gene_data.pathways
                .iter()
                .map(|p| p.pathway_name.clone())
                .take(5)
                .collect();

            let mut metadata = HashMap::new();
            metadata.insert("pathway_count".to_string(), serde_json::Value::Number(
                serde_json::Number::from(gene_data.pathways.len())
            ));

            examples.push(TrainingExample {
                prompt: format!("What biological pathways involve {}?", gene_id),
                response: format!(
                    "{} participates in {} biological pathways according to Reactome. Key pathways include {}. These pathways highlight {}'s role in cellular processes and provide context for understanding its function in systems biology networks.",
                    gene_id,
                    gene_data.pathways.len(),
                    pathway_names.join(", "),
                    gene_id
                ),
                gene_id: gene_id.clone(),
                example_type: "biological_pathways".to_string(),
                metadata,
            });
        }

        // Example 3: Systems biology context
        if let Some(proteomics) = &gene_data.proteomics {
            if !proteomics.functions.is_empty() {
                let mut metadata = HashMap::new();
                metadata.insert("function_count".to_string(), serde_json::Value::Number(
                    serde_json::Number::from(proteomics.functions.len())
                ));
                metadata.insert("domain_count".to_string(), serde_json::Value::Number(
                    serde_json::Number::from(proteomics.domains.len())
                ));

                let function_summary = if proteomics.functions.len() > 1 {
                    format!("has multiple functions including {}", proteomics.functions[0])
                } else {
                    format!("functions as {}", proteomics.functions[0])
                };

                let location_info = if !proteomics.subcellular_locations.is_empty() {
                    format!(" and is located in the {}", proteomics.subcellular_locations[0])
                } else {
                    String::new()
                };

                examples.push(TrainingExample {
                    prompt: format!("Explain the function of {} in the context of systems biology.", gene_id),
                    response: format!(
                        "From a systems biology perspective, {} {}{}. It interacts with {} proteins and participates in {} pathways, forming a complex network that contributes to cellular function. This highlights how {} doesn't function in isolation, but as part of interconnected biological systems with emergent properties beyond individual molecular interactions. The protein's network centrality score of {:.2} indicates its importance in the cellular interaction network.",
                        gene_id,
                        function_summary,
                        location_info,
                        gene_data.interactions.len(),
                        gene_data.pathways.len(),
                        gene_id,
                        gene_data.network_metrics.centrality_score
                    ),
                    gene_id: gene_id.clone(),
                    example_type: "systems_biology_context".to_string(),
                    metadata,
                });
            }
        }

        // Example 4: Network topology insights
        if gene_data.network_metrics.interaction_count > 0 || gene_data.network_metrics.pathway_count > 0 {
            let mut metadata = HashMap::new();
            metadata.insert("centrality_score".to_string(), serde_json::Value::Number(
                serde_json::Number::from_f64(gene_data.network_metrics.centrality_score).unwrap_or(serde_json::Number::from(0))
            ));

            let network_role = if gene_data.network_metrics.centrality_score > 2.0 {
                "hub protein with high connectivity"
            } else if gene_data.network_metrics.centrality_score > 1.0 {
                "moderately connected protein"
            } else {
                "peripheral protein with specific interactions"
            };

            examples.push(TrainingExample {
                prompt: format!("What is the network topology role of {} in cellular networks?", gene_id),
                response: format!(
                    "{} functions as a {} in cellular networks. With {} direct protein interactions and participation in {} pathways, it has a centrality score of {:.2}. The clustering coefficient of {:.3} indicates its local network density. This network topology suggests that {} plays a {} role in cellular function, with its position in the network determining its influence on cellular processes and disease pathways.",
                    gene_id,
                    network_role,
                    gene_data.network_metrics.interaction_count,
                    gene_data.network_metrics.pathway_count,
                    gene_data.network_metrics.centrality_score,
                    gene_data.network_metrics.clustering_coefficient,
                    gene_id,
                    if gene_data.network_metrics.centrality_score > 1.5 { "critical" } else { "specialized" }
                ),
                gene_id: gene_id.clone(),
                example_type: "network_topology".to_string(),
                metadata,
            });
        }

        examples
    }

    /// Save network data to files for training
    pub async fn save_training_data(
        &self,
        network_data: &[GeneNetworkData],
        output_dir: &Path,
    ) -> Result<HashMap<String, String>> {
        tokio::fs::create_dir_all(output_dir).await
            .context("Failed to create output directory")?;

        let mut output_files = HashMap::new();

        // Save proteomics data
        let proteomics_file = output_dir.join("proteomics_data.json");
        let proteomics_data: Vec<&ProteomicsData> = network_data
            .iter()
            .filter_map(|g| g.proteomics.as_ref())
            .collect();
        
        let proteomics_json = serde_json::to_string_pretty(&proteomics_data)?;
        tokio::fs::write(&proteomics_file, proteomics_json).await?;
        output_files.insert("proteomics".to_string(), proteomics_file.to_string_lossy().to_string());

        // Save pathway data
        let pathways_file = output_dir.join("reactome_pathways.json");
        let all_pathways: Vec<&PathwayData> = network_data
            .iter()
            .flat_map(|g| &g.pathways)
            .collect();
        
        let pathways_json = serde_json::to_string_pretty(&all_pathways)?;
        tokio::fs::write(&pathways_file, pathways_json).await?;
        output_files.insert("pathways".to_string(), pathways_file.to_string_lossy().to_string());

        // Save interaction data
        let interactions_file = output_dir.join("protein_interactions.json");
        let all_interactions: Vec<&ProteinInteraction> = network_data
            .iter()
            .flat_map(|g| &g.interactions)
            .collect();
        
        let interactions_json = serde_json::to_string_pretty(&all_interactions)?;
        tokio::fs::write(&interactions_file, interactions_json).await?;
        output_files.insert("interactions".to_string(), interactions_file.to_string_lossy().to_string());

        // Generate and save training examples
        let training_examples = self.generate_training_examples(network_data);
        let training_file = output_dir.join("training_examples.jsonl");
        
        let mut training_content = String::new();
        for example in &training_examples {
            training_content.push_str(&serde_json::to_string(example)?);
            training_content.push('\n');
        }
        
        tokio::fs::write(&training_file, training_content).await?;
        output_files.insert("training_examples".to_string(), training_file.to_string_lossy().to_string());

        // Save network summary
        let summary_file = output_dir.join("network_summary.json");
        let network_summary: Vec<_> = network_data
            .iter()
            .map(|g| {
                let mut summary = HashMap::new();
                summary.insert("gene_id".to_string(), serde_json::Value::String(g.gene_id.clone()));
                summary.insert("interactions_count".to_string(), serde_json::Value::Number(
                    serde_json::Number::from(g.interactions.len())
                ));
                summary.insert("pathways_count".to_string(), serde_json::Value::Number(
                    serde_json::Number::from(g.pathways.len())
                ));
                summary.insert("has_proteomics".to_string(), serde_json::Value::Bool(g.proteomics.is_some()));
                summary.insert("centrality_score".to_string(), serde_json::Value::Number(
                    serde_json::Number::from_f64(g.network_metrics.centrality_score).unwrap_or(serde_json::Number::from(0))
                ));
                summary
            })
            .collect();

        let summary_json = serde_json::to_string_pretty(&network_summary)?;
        tokio::fs::write(&summary_file, summary_json).await?;
        output_files.insert("summary".to_string(), summary_file.to_string_lossy().to_string());

        Ok(output_files)
    }

    /// Clear cache to free memory
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        self.current_memory_usage.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> HashMap<String, usize> {
        let cache = self.cache.read().await;
        let mut stats = HashMap::new();
        stats.insert("cached_entries".to_string(), cache.len());
        stats.insert("memory_usage_mb".to_string(), 
            self.current_memory_usage.load(std::sync::atomic::Ordering::Relaxed) / (1024 * 1024)
        );
        stats
    }
}

impl Default for NetworkDataProcessor {
    fn default() -> Self {
        Self::new(10, 100, 4096).expect("Failed to create default NetworkDataProcessor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_network_processor_creation() {
        let processor = NetworkDataProcessor::new(5, 50, 2048).unwrap();
        assert_eq!(processor.max_concurrent_requests, 5);
        assert_eq!(processor.request_delay_ms, 50);
        assert_eq!(processor.memory_limit_mb, 2048);
    }

    #[test]
    fn test_training_example_generation() {
        let processor = NetworkDataProcessor::default();
        
        let gene_data = GeneNetworkData {
            gene_id: "BRCA1".to_string(),
            interactions: vec![
                ProteinInteraction {
                    gene_a: "BRCA1".to_string(),
                    gene_b: "TP53".to_string(),
                    score: 0.8,
                    interaction_type: "physical".to_string(),
                    evidence: "experimental".to_string(),
                    source: "STRING".to_string(),
                    publication: None,
                }
            ],
            pathways: vec![
                PathwayData {
                    gene_id: "BRCA1".to_string(),
                    pathway_id: "R-HSA-5685942".to_string(),
                    pathway_name: "HDR through Homologous Recombination".to_string(),
                    species: "Homo sapiens".to_string(),
                    source: "Reactome".to_string(),
                    diagram_url: None,
                }
            ],
            proteomics: None,
            network_metrics: NetworkMetrics {
                interaction_count: 1,
                pathway_count: 1,
                centrality_score: 1.0,
                clustering_coefficient: 0.5,
            },
        };

        let examples = processor.create_training_examples_for_gene(&gene_data);
        assert!(!examples.is_empty());
        assert!(examples.iter().any(|e| e.example_type == "protein_interactions"));
        assert!(examples.iter().any(|e| e.example_type == "biological_pathways"));
    }

    #[test]
    fn test_network_metrics_calculation() {
        let processor = NetworkDataProcessor::default();
        
        let interactions = vec![
            ProteinInteraction {
                gene_a: "GENE1".to_string(),
                gene_b: "GENE2".to_string(),
                score: 0.8,
                interaction_type: "physical".to_string(),
                evidence: "experimental".to_string(),
                source: "STRING".to_string(),
                publication: None,
            }
        ];

        let pathways = vec![
            PathwayData {
                gene_id: "GENE1".to_string(),
                pathway_id: "PATHWAY1".to_string(),
                pathway_name: "Test Pathway".to_string(),
                species: "Homo sapiens".to_string(),
                source: "Reactome".to_string(),
                diagram_url: None,
            }
        ];

        let metrics = processor.calculate_network_metrics(&interactions, &pathways);
        assert_eq!(metrics.interaction_count, 1);
        assert_eq!(metrics.pathway_count, 1);
        assert!(metrics.centrality_score > 0.0);
        assert!(metrics.clustering_coefficient >= 0.0);
    }
} 