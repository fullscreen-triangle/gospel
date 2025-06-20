use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use rayon::prelude::*;
use tokio::time::Instant;
use pyo3::prelude::*;

/// Configuration for a genomic model
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomicModelConfig {
    #[pyo3(get, set)]
    pub model_id: String,
    #[pyo3(get, set)]
    pub model_type: String,
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get, set)]
    pub task: String,
    #[pyo3(get, set)]
    pub max_length: usize,
    #[pyo3(get, set)]
    pub supported_sequences: Vec<String>,
    #[pyo3(get, set)]
    pub performance_tier: String,
}

#[pymethods]
impl GenomicModelConfig {
    #[new]
    fn new(
        model_id: String,
        model_type: String,
        description: String,
        task: String,
        max_length: usize,
        supported_sequences: Vec<String>,
        performance_tier: String,
    ) -> Self {
        Self {
            model_id,
            model_type,
            description,
            task,
            max_length,
            supported_sequences,
            performance_tier,
        }
    }
}

/// Model performance benchmarks
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBenchmark {
    #[pyo3(get, set)]
    pub model_name: String,
    #[pyo3(get, set)]
    pub sequences_processed: usize,
    #[pyo3(get, set)]
    pub total_time_ms: f64,
    #[pyo3(get, set)]
    pub average_time_per_sequence_ms: f64,
    #[pyo3(get, set)]
    pub throughput_sequences_per_second: f64,
    #[pyo3(get, set)]
    pub memory_usage_mb: f64,
    #[pyo3(get, set)]
    pub success_rate: f64,
}

/// Sequence validation result
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    #[pyo3(get, set)]
    pub is_valid: bool,
    #[pyo3(get, set)]
    pub message: String,
    #[pyo3(get, set)]
    pub sequence_length: usize,
    #[pyo3(get, set)]
    pub sequence_type: String,
    #[pyo3(get, set)]
    pub invalid_characters: Vec<char>,
}

/// High-performance genomic models manager
#[pyclass]
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

#[pymethods]
impl GenomicModelsManager {
    #[new]
    fn new(max_cache_size_mb: Option<usize>) -> Self {
        let mut manager = Self {
            available_models: HashMap::new(),
            model_cache: HashMap::new(),
            benchmarks: HashMap::new(),
            max_cache_size: max_cache_size_mb.unwrap_or(8192),
            current_cache_size: 0,
        };
        manager.initialize_default_models();
        manager
    }

    /// Get models filtered by type
    fn get_models_by_type(&self, model_type: &str) -> Vec<(String, GenomicModelConfig)> {
        self.available_models
            .iter()
            .filter(|(_, config)| config.model_type == model_type)
            .map(|(name, config)| (name.clone(), config.clone()))
            .collect()
    }

    /// Get models filtered by task
    fn get_models_by_task(&self, task: &str) -> Vec<(String, GenomicModelConfig)> {
        self.available_models
            .iter()
            .filter(|(_, config)| config.task == task)
            .map(|(name, config)| (name.clone(), config.clone()))
            .collect()
    }

    /// Recommend models for specific analysis type
    fn recommend_models_for_analysis(&self, analysis_type: &str) -> Vec<String> {
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

    /// Validate sequence batch with high performance
    fn validate_sequence_batch(&self, sequences: Vec<(String, String)>) -> Vec<ValidationResult> {
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

    /// Benchmark model performance
    fn benchmark_model_performance(
        &mut self,
        model_name: &str,
        test_sequences: Vec<String>,
    ) -> PyResult<ModelBenchmark> {
        if !self.available_models.contains_key(model_name) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Model {} not found", model_name)
            ));
        }

        let start_time = std::time::Instant::now();
        let mut success_count = 0;
        let mut total_processing_time = 0.0;

        // Simulate model processing with parallel computation
        let processing_results: Vec<_> = test_sequences
            .par_iter()
            .map(|sequence| {
                let model_config = &self.available_models[model_name];
                let processing_time = self.simulate_model_processing(sequence, model_config);
                (processing_time, true) // success = true for simulation
            })
            .collect();

        // Aggregate results
        for (proc_time, success) in processing_results {
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

    /// Get compatible models for sequence type
    fn get_compatible_models(&self, sequence_type: &str) -> Vec<String> {
        self.available_models
            .iter()
            .filter(|(_, config)| {
                config.supported_sequences.contains(&sequence_type.to_string()) ||
                config.model_type.contains("multimodal")
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// List all available models
    fn list_available_models(&self) -> Vec<String> {
        self.available_models.keys().cloned().collect()
    }

    /// Get benchmark results
    fn get_benchmark_results(&self, model_name: &str) -> Option<ModelBenchmark> {
        self.benchmarks.get(model_name).cloned()
    }

    /// Clear benchmarks
    fn clear_benchmarks(&mut self) {
        self.benchmarks.clear();
    }
}

impl GenomicModelsManager {
    /// Initialize default genomic models
    fn initialize_default_models(&mut self) {
        self.available_models.insert("caduceus".to_string(), GenomicModelConfig {
            model_id: "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16".to_string(),
            model_type: "dna_sequence".to_string(),
            description: "Caduceus: Bi-directional equivariant long-range DNA sequence modeling".to_string(),
            task: "fill-mask".to_string(),
            max_length: 131072,
            supported_sequences: vec!["dna".to_string()],
            performance_tier: "high".to_string(),
        });

        self.available_models.insert("nucleotide_transformer".to_string(), GenomicModelConfig {
            model_id: "InstaDeepAI/nucleotide-transformer-2.5b-1000g".to_string(),
            model_type: "dna_sequence".to_string(),
            description: "Nucleotide Transformer: Foundation model for human genomics".to_string(),
            task: "fill-mask".to_string(),
            max_length: 1000,
            supported_sequences: vec!["dna".to_string()],
            performance_tier: "high".to_string(),
        });

        self.available_models.insert("esm2".to_string(), GenomicModelConfig {
            model_id: "facebook/esm2_t33_650M_UR50D".to_string(),
            model_type: "protein_sequence".to_string(),
            description: "ESM-2: Evolutionary Scale Modeling for protein sequences".to_string(),
            task: "fill-mask".to_string(),
            max_length: 1024,
            supported_sequences: vec!["protein".to_string()],
            performance_tier: "high".to_string(),
        });

        self.available_models.insert("protbert".to_string(), GenomicModelConfig {
            model_id: "Rostlab/prot_bert".to_string(),
            model_type: "protein_sequence".to_string(),
            description: "ProtBERT: Pre-trained protein language model".to_string(),
            task: "fill-mask".to_string(),
            max_length: 512,
            supported_sequences: vec!["protein".to_string()],
            performance_tier: "medium".to_string(),
        });

        self.available_models.insert("gene42".to_string(), GenomicModelConfig {
            model_id: "kuleshov-group/gene42-192k".to_string(),
            model_type: "genomic_foundation".to_string(),
            description: "Gene42: Long-range genomic foundation model with dense attention".to_string(),
            task: "generation".to_string(),
            max_length: 192000,
            supported_sequences: vec!["dna".to_string(), "rna".to_string()],
            performance_tier: "high".to_string(),
        });

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genomic_models_manager_creation() {
        let manager = GenomicModelsManager::new(Some(4096));
        assert!(manager.available_models.len() > 0);
        assert!(manager.available_models.contains_key("caduceus"));
    }

    #[test]
    fn test_sequence_validation() {
        let manager = GenomicModelsManager::new(Some(4096));
        
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
        let manager = GenomicModelsManager::new(Some(4096));
        
        let recommendations = manager.recommend_models_for_analysis("dna_analysis");
        assert!(recommendations.contains(&"caduceus".to_string()));
        assert!(recommendations.contains(&"nucleotide_transformer".to_string()));
    }
} 