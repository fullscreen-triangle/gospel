//! Gospel Rust - High-Performance Genomic Analysis Core
//!
//! This crate provides the high-performance Rust implementation for Gospel's
//! genomic analysis framework, offering 40× speedup over Python implementations
//! through SIMD vectorization, parallel processing, and memory-mapped I/O.

#![warn(missing_docs)]
#![warn(clippy::all)]

use mimalloc::MiMalloc;

// Use fast allocator globally
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Core modules
pub mod variant;
pub mod expression;
pub mod network;
pub mod fuzzy;
pub mod circuit;
pub mod experiment;
pub mod turbulance;
pub mod utils;

// Python bindings
#[cfg(feature = "python-bindings")]
pub mod python;

// New high-performance modules
pub mod genomic_models;
pub mod network_analysis;
pub mod network_processor;

// Re-export main types
pub use variant::{Variant, VariantProcessor, VariantStats};
pub use expression::{ExpressionProcessor, ExpressionMatrix};
pub use network::{GeneNetwork, NetworkProcessor};
pub use fuzzy::{FuzzyProcessor, FuzzyResult};
pub use circuit::{GenomicCircuit, CircuitProcessor};
pub use turbulance::{TurbulanceCompiler, TurbulanceAST, ExecutionPlan, TurbulanceError};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Gospel Rust processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GospelConfig {
    /// Enable SIMD vectorization
    pub enable_simd: bool,
    /// Number of parallel threads (0 = auto-detect)
    pub num_threads: usize,
    /// Memory mapping chunk size in bytes
    pub chunk_size: usize,
    /// Enable tracing/logging
    pub enable_tracing: bool,
}

impl Default for GospelConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            num_threads: 0, // Auto-detect
            chunk_size: 64 * 1024 * 1024, // 64MB chunks
            enable_tracing: false,
        }
    }
}

/// Main Gospel processor that coordinates all analysis components
#[derive(Debug)]
pub struct GospelProcessor {
    config: GospelConfig,
    variant_processor: VariantProcessor,
    expression_processor: ExpressionProcessor,
    network_processor: NetworkProcessor,
    fuzzy_processor: FuzzyProcessor,
    circuit_processor: CircuitProcessor,
    turbulance_compiler: TurbulanceCompiler,
}

impl GospelProcessor {
    /// Create a new Gospel processor with the given configuration
    pub fn new(config: GospelConfig) -> Result<Self> {
        // Initialize tracing if enabled
        if config.enable_tracing {
            tracing_subscriber::fmt::init();
        }

        // Set up thread pool
        if config.num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.num_threads)
                .build_global()
                .map_err(|e| anyhow::anyhow!("Failed to initialize thread pool: {}", e))?;
        }

        Ok(Self {
            variant_processor: VariantProcessor::new(&config)?,
            expression_processor: ExpressionProcessor::new(&config)?,
            network_processor: NetworkProcessor::new(&config)?,
            fuzzy_processor: FuzzyProcessor::new(&config)?,
            circuit_processor: CircuitProcessor::new(&config)?,
            turbulance_compiler: TurbulanceCompiler::new(),
            config,
        })
    }

    /// Process VCF file with high-performance Rust implementation
    pub async fn process_vcf(&self, vcf_path: &str) -> Result<VariantStats> {
        tracing::info!("Processing VCF file: {}", vcf_path);
        self.variant_processor.process_vcf_file(vcf_path).await
    }

    /// Process expression data with parallel algorithms
    pub async fn process_expression(&self, expression_data: &[f64], genes: &[String], samples: &[String]) -> Result<ExpressionMatrix> {
        tracing::info!("Processing expression data: {} genes, {} samples", genes.len(), samples.len());
        self.expression_processor.process_matrix(expression_data, genes, samples).await
    }

    /// Analyze gene networks with graph algorithms
    pub async fn analyze_network(&self, adjacency_matrix: &[f64], gene_names: &[String]) -> Result<GeneNetwork> {
        tracing::info!("Analyzing gene network: {} nodes", gene_names.len());
        self.network_processor.analyze_network(adjacency_matrix, gene_names).await
    }

    /// Apply fuzzy logic analysis to variants
    pub async fn fuzzy_analysis(&self, variants: &[Variant]) -> Result<Vec<FuzzyResult>> {
        tracing::info!("Applying fuzzy logic to {} variants", variants.len());
        self.fuzzy_processor.analyze_variants(variants).await
    }

    /// Generate and analyze genomic circuits
    pub async fn circuit_analysis(&self, network: &GeneNetwork, expression: &ExpressionMatrix) -> Result<GenomicCircuit> {
        tracing::info!("Generating genomic circuit");
        self.circuit_processor.generate_circuit(network, expression).await
    }

    /// Compile Turbulance script into execution plan
    pub fn compile_turbulance(&self, source: &str) -> Result<ExecutionPlan, TurbulanceError> {
        tracing::info!("Compiling Turbulance script");
        let ast = self.turbulance_compiler.parse(source)?;
        self.turbulance_compiler.compile(ast)
    }

    /// Parse Turbulance script into AST
    pub fn parse_turbulance(&self, source: &str) -> Result<TurbulanceAST, TurbulanceError> {
        tracing::info!("Parsing Turbulance script");
        self.turbulance_compiler.parse(source)
    }

    /// Validate Turbulance AST for scientific soundness
    pub fn validate_turbulance(&self, ast: &TurbulanceAST) -> Result<(), Vec<TurbulanceError>> {
        tracing::info!("Validating Turbulance AST");
        self.turbulance_compiler.validate(ast)
    }

    /// Get processor statistics
    pub fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        stats.insert("config".to_string(), serde_json::to_value(&self.config).unwrap());
        stats.insert("variant_stats".to_string(), serde_json::to_value(self.variant_processor.get_stats()).unwrap());
        stats.insert("expression_stats".to_string(), serde_json::to_value(self.expression_processor.get_stats()).unwrap());
        stats.insert("network_stats".to_string(), serde_json::to_value(self.network_processor.get_stats()).unwrap());
        
        stats
    }
}

/// Initialize Gospel Rust processor with default configuration
pub fn init() -> Result<GospelProcessor> {
    GospelProcessor::new(GospelConfig::default())
}

/// Initialize Gospel Rust processor with custom configuration
pub fn init_with_config(config: GospelConfig) -> Result<GospelProcessor> {
    GospelProcessor::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gospel_processor_creation() {
        let config = GospelConfig::default();
        let processor = GospelProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = GospelConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: GospelConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(config.enable_simd, deserialized.enable_simd);
    }
}

use pyo3::prelude::*;

/// High-performance genomic analysis functions
#[pymodule]
fn gospel_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // Existing modules
    m.add_function(wrap_pyfunction!(variant::process_variants, m)?)?;
    m.add_function(wrap_pyfunction!(fuzzy::calculate_fuzzy_membership, m)?)?;
    m.add_function(wrap_pyfunction!(fuzzy::fuzzy_and, m)?)?;
    m.add_function(wrap_pyfunction!(fuzzy::fuzzy_or, m)?)?;
    m.add_function(wrap_pyfunction!(fuzzy::fuzzy_not, m)?)?;
    m.add_function(wrap_pyfunction!(expression::process_expression_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(network::calculate_network_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(circuit::simulate_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(experiment::generate_training_data, m)?)?;
    m.add_function(wrap_pyfunction!(utils::parallel_processing_test, m)?)?;

    // New high-performance modules
    m.add_class::<genomic_models::GenomicModelsManager>()?;
    m.add_class::<network_analysis::NetworkAnalyzer>()?;
    m.add_class::<network_processor::NetworkDataProcessor>()?;
    
    // Turbulance DSL compiler
    m.add_class::<turbulance::TurbulanceCompiler>()?;

    Ok(())
} 