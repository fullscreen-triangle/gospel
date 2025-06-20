use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use rayon::prelude::*;
use tokio::time::Instant;

/// Configuration for a genomic model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomicModelConfig {
    pub model_id: String,
    pub model_type: String,
    pub description: String,
    pub task: String,
    pub max_length: usize,
    pub supported_sequences: Vec<String>,
    pub performance_tier: String, // "high", "medium", "low"
}

/// Model performance benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBenchmark {
    pub model_name: String,
    pub sequences_processed: usize,
    pub total_time_ms: f64,
    pub average_time_per_sequence_ms: f64,
    pub throughput_sequences_per_second: f64,
    pub memory_usage_mb: f64,
    pub success_rate: f64,
}

/// Sequence validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub message: String,
    pub sequence_length: usize,
    pub sequence_type: String,
    pub invalid_characters: Vec<char>,
}

/// High-performance genomic models manager
pub struct GenomicModelsManager {
    available_models: HashMap<String, GenomicModelConfig>,
    model_cache: HashMap<String, ModelCacheEntry>,
    benchmarks: HashMap<String, ModelBenchmark>,
    max_cache_size: usize,
    current_cache_size: usize,
}

#[derive(Debug, Clone)]
struct ModelCacheEntry {
    model_name: String,
    last_accessed: std::time::Instant,
    memory_size_mb: f64,
    access_count: usize,
}

impl GenomicModelsManager {
    /// Create a new genomic models manager
    pub fn new(max_cache_size_mb: usize) -> Self {
        let mut manager = Self {
            available_models: HashMap::new(),
            model_cache: HashMap::new(),
            benchmarks: HashMap::new(),
            max_cache_size: max_cache_size_mb,
            current_cache_size: 0,
        };
        
        manager.initialize_default_models();
        manager
    }

    /// Initialize default genomic models
    fn initialize_default_models(&mut self) {
        // Caduceus model
        self.available_models.insert("caduceus".to_string(), GenomicModelConfig {
            model_id: "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16".to_string(),
            model_type: "dna_sequence".to_string(),
            description: "Caduceus: Bi-directional equivariant long-range DNA sequence modeling".to_string(),
            task: "fill-mask".to_string(),
            max_length: 131072,
            supported_sequences: vec!["dna".to_string()],
            performance_tier: "high".to_string(),
        });

        // Nucleotide Transformer
        self.available_models.insert("nucleotide_transformer".to_string(), GenomicModelConfig {
            model_id: "InstaDeepAI/nucleotide-transformer-2.5b-1000g".to_string(),
            model_type: "dna_sequence".to_string(),
            description: "Nucleotide Transformer: Foundation model for human genomics".to_string(),
            task: "fill-mask".to_string(),
            max_length: 1000,
            supported_sequences: vec!["dna".to_string()],
            performance_tier: "high".to_string(),
        });

        // ESM2 Protein Model
        self.available_models.insert("esm2".to_string(), GenomicModelConfig {
            model_id: "facebook/esm2_t33_650M_UR50D".to_string(),
            model_type: "protein_sequence".to_string(),
            description: "ESM-2: Evolutionary Scale Modeling for protein sequences".to_string(),
            task: "fill-mask".to_string(),
            max_length: 1024,
            supported_sequences: vec!["protein".to_string()],
            performance_tier: "high".to_string(),
        });

        // ProtBERT
        self.available_models.insert("protbert".to_string(), GenomicModelConfig {
            model_id: "Rostlab/prot_bert".to_string(),
            model_type: "protein_sequence".to_string(),
            description: "ProtBERT: Pre-trained protein language model".to_string(),
            task: "fill-mask".to_string(),
            max_length: 512,
            supported_sequences: vec!["protein".to_string()],
            performance_tier: "medium".to_string(),
        });

        // Gene42 (hypothetical long-range model)
        self.available_models.insert("gene42".to_string(), GenomicModelConfig {
            model_id: "kuleshov-group/gene42-192k".to_string(),
            model_type: "genomic_foundation".to_string(),
            description: "Gene42: Long-range genomic foundation model with dense attention".to_string(),
            task: "generation".to_string(),
            max_length: 192000,
            supported_sequences: vec!["dna".to_string(), "rna".to_string()],
            performance_tier: "high".to_string(),
        });

        // MAMMAL Multimodal
        self.available_models.insert("mammal_biomed".to_string(), GenomicModelConfig {
            model_id: "ibm/biomed.omics.bl.sm.ma-ted-458m".to_string(),
            model_type: "multimodal_biomed".to_string(),
            description: "MAMMAL: Molecular aligned multi-modal architecture for biomedical data".to_string(),
            task: "generation".to_string(),
            max_length: 2048,
            supported_sequences: vec!["dna".to_string(), "protein".to_string(), "rna".to_string()],
            performance_tier: "medium".to_string(),
        });
    }

    /// Get models filtered by type
    pub fn get_models_by_type(&self, model_type: &str) -> Vec<(String, GenomicModelConfig)> {
        self.available_models
            .iter()
            .filter(|(_, config)| config.model_type == model_type)
            .map(|(name, config)| (name.clone(), config.clone()))
            .collect()
    }

    /// Get models filtered by task
    pub fn get_models_by_task(&self, task: &str) -> Vec<(String, GenomicModelConfig)> {
        self.available_models
            .iter()
            .filter(|(_, config)| config.task == task)
            .map(|(name, config)| (name.clone(), config.clone()))
            .collect()
    }

    /// Recommend models for specific analysis type
    pub fn recommend_models_for_analysis(&self, analysis_type: &str) -> Vec<String> {
        match analysis_type {
            "variant_effect" => vec!["caduceus".to_string(), "nucleotide_transformer".to_string()],
            "protein_function" => vec!["esm2".to_string(), "protbert".to_string()],
            "dna_analysis" => vec!["caduceus".to_string(), "nucleotide_transformer".to_string(), "gene42".to_string()],
            "sequence_generation" => vec!["gene42".to_string(), "mammal_biomed".to_string()],
            "multimodal_analysis" => vec!["mammal_biomed".to_string()],
            "long_range_genomic" => vec!["caduceus".to_string(), "gene42".to_string()],
            _ => self.available_models.keys().cloned().collect(),
        }
    }

    /// Validate sequence for genomic analysis with high performance
    pub fn validate_sequence_batch(&self, sequences: &[(String, String)]) -> Vec<ValidationResult> {
        sequences
            .par_iter()
            .map(|(sequence, sequence_type)| self.validate_single_sequence(sequence, sequence_type))
            .collect()
    }

    /// Validate a single sequence
    fn validate_single_sequence(&self, sequence: &str, sequence_type: &str) -> ValidationResult {
        if sequence.is_empty() {
            return ValidationResult {
                is_valid: false,
                message: "Empty sequence".to_string(),
                sequence_length: 0,
                sequence_type: sequence_type.to_string(),
                invalid_characters: vec![],
            };
        }

        let sequence = sequence.to_uppercase();
        let mut invalid_chars = Vec::new();

        let valid_chars = match sequence_type.to_lowercase().as_str() {
            "dna" => "ATCGN".chars().collect::<std::collections::HashSet<_>>(),
            "rna" => "AUCGN".chars().collect::<std::collections::HashSet<_>>(),
            "protein" => "ACDEFGHIKLMNPQRSTVWYXBZJUO*".chars().collect::<std::collections::HashSet<_>>(),
            _ => return ValidationResult {
                is_valid: false,
                message: format!("Unknown sequence type: {}", sequence_type),
                sequence_length: sequence.len(),
                sequence_type: sequence_type.to_string(),
                invalid_characters: vec![],
            },
        };

        // Check for invalid characters
        for ch in sequence.chars() {
            if !valid_chars.contains(&ch) && !invalid_chars.contains(&ch) {
                invalid_chars.push(ch);
            }
        }

        // Length validation
        if sequence.len() < 10 {
            return ValidationResult {
                is_valid: false,
                message: format!("Sequence too short: {} characters (minimum 10)", sequence.len()),
                sequence_length: sequence.len(),
                sequence_type: sequence_type.to_string(),
                invalid_characters: invalid_chars,
            };
        }

        if sequence.len() > 200000 {
            return ValidationResult {
                is_valid: false,
                message: format!("Sequence too long: {} characters (maximum 200,000)", sequence.len()),
                sequence_length: sequence.len(),
                sequence_type: sequence_type.to_string(),
                invalid_characters: invalid_chars,
            };
        }

        if !invalid_chars.is_empty() {
            return ValidationResult {
                is_valid: false,
                message: format!("Invalid {} characters: {:?}", sequence_type, invalid_chars),
                sequence_length: sequence.len(),
                sequence_type: sequence_type.to_string(),
                invalid_characters: invalid_chars,
            };
        }

        ValidationResult {
            is_valid: true,
            message: "Valid sequence".to_string(),
            sequence_length: sequence.len(),
            sequence_type: sequence_type.to_string(),
            invalid_characters: vec![],
        }
    }

    /// Benchmark model performance with synthetic data
    pub fn benchmark_model_performance(
        &mut self,
        model_name: &str,
        test_sequences: &[String],
    ) -> Result<ModelBenchmark> {
        if !self.available_models.contains_key(model_name) {
            return Err(anyhow::anyhow!("Model {} not found", model_name));
        }

        let start_time = Instant::now();
        let mut success_count = 0;
        let mut total_processing_time = 0.0;

        // Simulate model processing (in real implementation, this would call actual models)
        let processing_results: Vec<_> = test_sequences
            .par_iter()
            .map(|sequence| {
                let seq_start = std::time::Instant::now();
                
                // Simulate processing time based on sequence length and model complexity
                let model_config = &self.available_models[model_name];
                let processing_time = self.simulate_model_processing(sequence, model_config);
                
                let seq_end = std::time::Instant::now();
                let actual_time = seq_end.duration_since(seq_start).as_secs_f64() * 1000.0;

                (processing_time, actual_time, true) // success = true for simulation
            })
            .collect();

        // Aggregate results
        for (proc_time, _actual_time, success) in processing_results {
            total_processing_time += proc_time;
            if success {
                success_count += 1;
            }
        }

        let elapsed = start_time.elapsed();
        let total_time_ms = elapsed.as_secs_f64() * 1000.0;
        let average_time_per_sequence = total_processing_time / test_sequences.len() as f64;
        let throughput = test_sequences.len() as f64 / (total_time_ms / 1000.0);
        let success_rate = success_count as f64 / test_sequences.len() as f64;

        let benchmark = ModelBenchmark {
            model_name: model_name.to_string(),
            sequences_processed: test_sequences.len(),
            total_time_ms,
            average_time_per_sequence_ms: average_time_per_sequence,
            throughput_sequences_per_second: throughput,
            memory_usage_mb: self.estimate_model_memory_usage(model_name),
            success_rate,
        };

        // Cache benchmark results
        self.benchmarks.insert(model_name.to_string(), benchmark.clone());

        Ok(benchmark)
    }

    /// Simulate model processing time for benchmarking
    fn simulate_model_processing(&self, sequence: &str, model_config: &GenomicModelConfig) -> f64 {
        let base_time = match model_config.performance_tier.as_str() {
            "high" => 0.5,      // 0.5ms base
            "medium" => 1.0,    // 1.0ms base
            "low" => 2.0,       // 2.0ms base
            _ => 1.0,
        };

        let length_factor = (sequence.len() as f64 / 1000.0).max(0.1);
        let complexity_factor = match model_config.task.as_str() {
            "fill-mask" => 1.0,
            "generation" => 2.0,
            "sequence-classification" => 0.8,
            _ => 1.0,
        };

        base_time * length_factor * complexity_factor
    }

    /// Estimate memory usage for a model
    fn estimate_model_memory_usage(&self, model_name: &str) -> f64 {
        // Simplified estimation based on model type
        match model_name {
            "caduceus" => 2048.0,           // ~2GB
            "nucleotide_transformer" => 4096.0, // ~4GB
            "esm2" => 3072.0,               // ~3GB
            "protbert" => 1536.0,           // ~1.5GB
            "gene42" => 8192.0,             // ~8GB (large model)
            "mammal_biomed" => 2048.0,      // ~2GB
            _ => 1024.0,                    // Default 1GB
        }
    }

    /// Get compatible models for sequence type
    pub fn get_compatible_models(&self, sequence_type: &str) -> Vec<String> {
        self.available_models
            .iter()
            .filter(|(_, config)| {
                config.supported_sequences.contains(&sequence_type.to_string()) ||
                config.model_type.contains("multimodal")
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Create analysis configuration
    pub fn create_analysis_config(
        &self,
        analysis_type: &str,
        sequence_type: &str,
        models: Option<Vec<String>>,
        device: &str,
    ) -> HashMap<String, serde_json::Value> {
        let recommended_models = models.unwrap_or_else(|| {
            self.recommend_models_for_analysis(analysis_type)
        });

        // Filter by sequence type compatibility
        let compatible_models: Vec<String> = recommended_models
            .into_iter()
            .filter(|model_name| {
                if let Some(config) = self.available_models.get(model_name) {
                    config.supported_sequences.contains(&sequence_type.to_string()) ||
                    config.model_type.contains("multimodal")
                } else {
                    false
                }
            })
            .collect();

        // Calculate max sequence length
        let max_length = compatible_models
            .iter()
            .filter_map(|model_name| self.available_models.get(model_name))
            .map(|config| config.max_length)
            .min()
            .unwrap_or(1000);

        let mut config = HashMap::new();
        config.insert("analysis_type".to_string(), serde_json::Value::String(analysis_type.to_string()));
        config.insert("sequence_type".to_string(), serde_json::Value::String(sequence_type.to_string()));
        config.insert("models".to_string(), serde_json::Value::Array(
            compatible_models.into_iter().map(serde_json::Value::String).collect()
        ));
        config.insert("device".to_string(), serde_json::Value::String(device.to_string()));
        config.insert("recommended_batch_size".to_string(), serde_json::Value::Number(
            serde_json::Number::from(if device == "cpu" { 1 } else { 4 })
        ));
        config.insert("max_sequence_length".to_string(), serde_json::Value::Number(
            serde_json::Number::from(max_length)
        ));

        config
    }

    /// Batch sequence processing for training data generation
    pub fn process_sequences_for_training(
        &self,
        sequences: &[(String, String)], // (sequence, sequence_type)
        model_names: &[String],
        batch_size: usize,
    ) -> Result<Vec<TrainingSequenceResult>> {
        // Process sequences in parallel batches
        let results: Vec<TrainingSequenceResult> = sequences
            .par_chunks(batch_size)
            .flat_map(|batch| {
                batch
                    .par_iter()
                    .map(|(sequence, seq_type)| {
                        self.process_single_sequence_for_training(sequence, seq_type, model_names)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(results)
    }

    /// Process a single sequence for training
    fn process_single_sequence_for_training(
        &self,
        sequence: &str,
        sequence_type: &str,
        model_names: &[String],
    ) -> TrainingSequenceResult {
        let validation = self.validate_single_sequence(sequence, sequence_type);
        
        if !validation.is_valid {
            return TrainingSequenceResult {
                sequence: sequence.to_string(),
                sequence_type: sequence_type.to_string(),
                validation_result: validation,
                compatible_models: vec![],
                processing_estimates: HashMap::new(),
                training_features: HashMap::new(),
            };
        }

        // Find compatible models
        let compatible_models: Vec<String> = model_names
            .iter()
            .filter(|model_name| {
                if let Some(config) = self.available_models.get(*model_name) {
                    config.supported_sequences.contains(&sequence_type.to_string()) &&
                    sequence.len() <= config.max_length
                } else {
                    false
                }
            })
            .cloned()
            .collect();

        // Estimate processing times
        let mut processing_estimates = HashMap::new();
        for model_name in &compatible_models {
            if let Some(config) = self.available_models.get(model_name) {
                let estimated_time = self.simulate_model_processing(sequence, config);
                processing_estimates.insert(model_name.clone(), estimated_time);
            }
        }

        // Generate training features
        let mut training_features = HashMap::new();
        training_features.insert("sequence_length".to_string(), sequence.len() as f64);
        training_features.insert("gc_content".to_string(), self.calculate_gc_content(sequence));
        training_features.insert("complexity_score".to_string(), self.calculate_sequence_complexity(sequence));

        TrainingSequenceResult {
            sequence: sequence.to_string(),
            sequence_type: sequence_type.to_string(),
            validation_result: validation,
            compatible_models,
            processing_estimates,
            training_features,
        }
    }

    /// Calculate GC content for DNA/RNA sequences
    fn calculate_gc_content(&self, sequence: &str) -> f64 {
        let sequence = sequence.to_uppercase();
        let gc_count = sequence.chars().filter(|&c| c == 'G' || c == 'C').count();
        gc_count as f64 / sequence.len() as f64
    }

    /// Calculate sequence complexity
    fn calculate_sequence_complexity(&self, sequence: &str) -> f64 {
        let mut char_counts = HashMap::new();
        for ch in sequence.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
        }

        // Calculate entropy as complexity measure
        let len = sequence.len() as f64;
        let entropy: f64 = char_counts
            .values()
            .map(|&count| {
                let p = count as f64 / len;
                -p * p.log2()
            })
            .sum();

        entropy
    }

    /// Get model information
    pub fn get_model_info(&self, model_name: &str) -> Option<ModelInfo> {
        self.available_models.get(model_name).map(|config| {
            ModelInfo {
                name: model_name.to_string(),
                config: config.clone(),
                benchmark: self.benchmarks.get(model_name).cloned(),
                cache_status: self.model_cache.contains_key(model_name),
                estimated_memory_mb: self.estimate_model_memory_usage(model_name),
            }
        })
    }

    /// Get all available models
    pub fn list_available_models(&self) -> Vec<String> {
        self.available_models.keys().cloned().collect()
    }

    /// Get benchmark results
    pub fn get_benchmark_results(&self, model_name: &str) -> Option<&ModelBenchmark> {
        self.benchmarks.get(model_name)
    }

    /// Clear benchmarks
    pub fn clear_benchmarks(&mut self) {
        self.benchmarks.clear();
    }
}

/// Result of processing a sequence for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSequenceResult {
    pub sequence: String,
    pub sequence_type: String,
    pub validation_result: ValidationResult,
    pub compatible_models: Vec<String>,
    pub processing_estimates: HashMap<String, f64>,
    pub training_features: HashMap<String, f64>,
}

/// Complete model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub config: GenomicModelConfig,
    pub benchmark: Option<ModelBenchmark>,
    pub cache_status: bool,
    pub estimated_memory_mb: f64,
}

impl Default for GenomicModelsManager {
    fn default() -> Self {
        Self::new(8192) // Default 8GB cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genomic_models_manager_creation() {
        let manager = GenomicModelsManager::new(4096);
        assert!(manager.available_models.len() > 0);
        assert!(manager.available_models.contains_key("caduceus"));
    }

    #[test]
    fn test_sequence_validation() {
        let manager = GenomicModelsManager::new(4096);
        
        // Valid DNA sequence
        let result = manager.validate_single_sequence("ATCGATCGATCG", "dna");
        assert!(result.is_valid);
        
        // Invalid DNA sequence
        let result = manager.validate_single_sequence("ATCGATCGXYZ", "dna");
        assert!(!result.is_valid);
        assert!(result.invalid_characters.contains(&'X'));
    }

    #[test]
    fn test_model_recommendations() {
        let manager = GenomicModelsManager::new(4096);
        
        let recommendations = manager.recommend_models_for_analysis("dna_analysis");
        assert!(recommendations.contains(&"caduceus".to_string()));
        assert!(recommendations.contains(&"nucleotide_transformer".to_string()));
    }

    #[test]
    fn test_gc_content_calculation() {
        let manager = GenomicModelsManager::new(4096);
        
        let gc_content = manager.calculate_gc_content("ATCGATCG");
        assert!((gc_content - 0.5).abs() < 0.001); // 4 GC out of 8 = 0.5
    }

    #[test]
    fn test_batch_validation() {
        let manager = GenomicModelsManager::new(4096);
        
        let sequences = vec![
            ("ATCGATCGATCG".to_string(), "dna".to_string()),
            ("AUCGAUCGAUCG".to_string(), "rna".to_string()),
            ("ACDEFGHIKLMN".to_string(), "protein".to_string()),
        ];
        
        let results = manager.validate_sequence_batch(&sequences);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_valid));
    }
} 