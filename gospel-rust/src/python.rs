//! Python bindings for Gospel Rust
//!
//! This module provides Python bindings for the high-performance Rust
//! implementation using PyO3, allowing seamless integration with the
//! Python Gospel framework.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2};
use std::collections::HashMap;

use crate::{
    GospelConfig, GospelProcessor, Variant, VariantStats, FuzzyResult,
    variant::VariantProcessor, fuzzy::FuzzyProcessor,
};

/// Python wrapper for GospelConfig
#[pyclass]
#[derive(Clone)]
pub struct PyGospelConfig {
    inner: GospelConfig,
}

#[pymethods]
impl PyGospelConfig {
    #[new]
    #[pyo3(signature = (enable_simd=true, num_threads=0, chunk_size=67108864, enable_tracing=false))]
    fn new(
        enable_simd: bool,
        num_threads: usize,
        chunk_size: usize,
        enable_tracing: bool,
    ) -> Self {
        Self {
            inner: GospelConfig {
                enable_simd,
                num_threads,
                chunk_size,
                enable_tracing,
            }
        }
    }

    #[getter]
    fn enable_simd(&self) -> bool {
        self.inner.enable_simd
    }

    #[setter]
    fn set_enable_simd(&mut self, value: bool) {
        self.inner.enable_simd = value;
    }

    #[getter]
    fn num_threads(&self) -> usize {
        self.inner.num_threads
    }

    #[setter]
    fn set_num_threads(&mut self, value: usize) {
        self.inner.num_threads = value;
    }

    #[getter]
    fn chunk_size(&self) -> usize {
        self.inner.chunk_size
    }

    #[setter]
    fn set_chunk_size(&mut self, value: usize) {
        self.inner.chunk_size = value;
    }

    #[getter]
    fn enable_tracing(&self) -> bool {
        self.inner.enable_tracing
    }

    #[setter]
    fn set_enable_tracing(&mut self, value: bool) {
        self.inner.enable_tracing = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "PyGospelConfig(enable_simd={}, num_threads={}, chunk_size={}, enable_tracing={})",
            self.inner.enable_simd,
            self.inner.num_threads,
            self.inner.chunk_size,
            self.inner.enable_tracing
        )
    }
}

/// Python wrapper for Variant
#[pyclass]
#[derive(Clone)]
pub struct PyVariant {
    inner: Variant,
}

#[pymethods]
impl PyVariant {
    #[new]
    fn new(
        chromosome: String,
        position: u64,
        reference: String,
        alternative: String,
        quality: f64,
    ) -> Self {
        Self {
            inner: Variant::new(chromosome, position, reference, alternative, quality),
        }
    }

    #[getter]
    fn chromosome(&self) -> String {
        self.inner.chromosome.clone()
    }

    #[getter]
    fn position(&self) -> u64 {
        self.inner.position
    }

    #[getter]
    fn reference(&self) -> String {
        self.inner.reference.clone()
    }

    #[getter]
    fn alternative(&self) -> String {
        self.inner.alternative.clone()
    }

    #[getter]
    fn quality(&self) -> f64 {
        self.inner.quality
    }

    #[getter]
    fn cadd_score(&self) -> Option<f64> {
        self.inner.cadd_score
    }

    #[setter]
    fn set_cadd_score(&mut self, value: Option<f64>) {
        self.inner.cadd_score = value;
    }

    #[getter]
    fn conservation_score(&self) -> Option<f64> {
        self.inner.conservation_score
    }

    #[setter]
    fn set_conservation_score(&mut self, value: Option<f64>) {
        self.inner.conservation_score = value;
    }

    #[getter]
    fn allele_frequency(&self) -> Option<f64> {
        self.inner.allele_frequency
    }

    #[setter]
    fn set_allele_frequency(&mut self, value: Option<f64>) {
        self.inner.allele_frequency = value;
    }

    fn pathogenicity_score(&self) -> f64 {
        self.inner.pathogenicity_score()
    }

    fn is_pathogenic(&self, threshold: f64) -> bool {
        self.inner.is_pathogenic(threshold)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyVariant({}:{} {}>{}, qual={:.1})",
            self.inner.chromosome,
            self.inner.position,
            self.inner.reference,
            self.inner.alternative,
            self.inner.quality
        )
    }
}

/// Python wrapper for VariantStats
#[pyclass]
#[derive(Clone)]
pub struct PyVariantStats {
    inner: VariantStats,
}

#[pymethods]
impl PyVariantStats {
    #[getter]
    fn total_variants(&self) -> u64 {
        self.inner.total_variants
    }

    #[getter]
    fn pathogenic_variants(&self) -> u64 {
        self.inner.pathogenic_variants
    }

    #[getter]
    fn benign_variants(&self) -> u64 {
        self.inner.benign_variants
    }

    #[getter]
    fn average_cadd(&self) -> f64 {
        self.inner.average_cadd
    }

    #[getter]
    fn average_conservation(&self) -> f64 {
        self.inner.average_conservation
    }

    #[getter]
    fn processing_time_ms(&self) -> u64 {
        self.inner.processing_time_ms
    }

    #[getter]
    fn throughput(&self) -> f64 {
        self.inner.throughput
    }

    fn __repr__(&self) -> String {
        format!(
            "PyVariantStats(total={}, pathogenic={}, throughput={:.0} var/sec)",
            self.inner.total_variants,
            self.inner.pathogenic_variants,
            self.inner.throughput
        )
    }
}

/// Python wrapper for FuzzyResult
#[pyclass]
#[derive(Clone)]
pub struct PyFuzzyResult {
    inner: FuzzyResult,
}

#[pymethods]
impl PyFuzzyResult {
    #[getter]
    fn variant_id(&self) -> String {
        self.inner.variant_id.clone()
    }

    #[getter]
    fn pathogenicity_confidence(&self) -> f64 {
        self.inner.pathogenicity_confidence
    }

    #[getter]
    fn conservation_confidence(&self) -> f64 {
        self.inner.conservation_confidence
    }

    #[getter]
    fn frequency_confidence(&self) -> f64 {
        self.inner.frequency_confidence
    }

    #[getter]
    fn expression_confidence(&self) -> f64 {
        self.inner.expression_confidence
    }

    #[getter]
    fn uncertainty_score(&self) -> f64 {
        self.inner.uncertainty_score
    }

    fn get_memberships(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        for (variable, memberships) in &self.inner.memberships {
            let inner_dict = PyDict::new(py);
            for (fuzzy_set, membership) in memberships {
                inner_dict.set_item(fuzzy_set, membership)?;
            }
            dict.set_item(variable, inner_dict)?;
        }
        
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "PyFuzzyResult(variant={}, pathogenicity={:.3}, uncertainty={:.3})",
            self.inner.variant_id,
            self.inner.pathogenicity_confidence,
            self.inner.uncertainty_score
        )
    }
}

/// Python wrapper for GospelProcessor
#[pyclass]
pub struct PyGospelProcessor {
    inner: GospelProcessor,
    rt: tokio::runtime::Runtime,
}

#[pymethods]
impl PyGospelProcessor {
    #[new]
    fn new(config: PyGospelConfig) -> PyResult<Self> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create async runtime: {}", e)))?;
        
        let inner = GospelProcessor::new(config.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create processor: {}", e)))?;
        
        Ok(Self { inner, rt })
    }

    fn process_vcf(&mut self, vcf_path: &str) -> PyResult<PyVariantStats> {
        let result = self.rt.block_on(self.inner.process_vcf(vcf_path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("VCF processing failed: {}", e)))?;
        
        Ok(PyVariantStats { inner: result })
    }

    fn process_variants(&mut self, variants: Vec<PyVariant>) -> PyResult<Vec<PyVariant>> {
        let rust_variants: Vec<Variant> = variants.into_iter().map(|v| v.inner).collect();
        
        // Create a temporary variant processor for this operation
        let variant_processor = VariantProcessor::new(&self.inner.config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create variant processor: {}", e)))?;
        
        let processed = self.rt.block_on(variant_processor.process_variants_parallel(rust_variants))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Variant processing failed: {}", e)))?;
        
        Ok(processed.into_iter().map(|v| PyVariant { inner: v }).collect())
    }

    fn fuzzy_analysis(&mut self, variants: Vec<PyVariant>) -> PyResult<Vec<PyFuzzyResult>> {
        let rust_variants: Vec<Variant> = variants.into_iter().map(|v| v.inner).collect();
        
        let results = self.rt.block_on(self.inner.fuzzy_analysis(&rust_variants))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Fuzzy analysis failed: {}", e)))?;
        
        Ok(results.into_iter().map(|r| PyFuzzyResult { inner: r }).collect())
    }

    fn get_stats(&self, py: Python) -> PyResult<PyObject> {
        let stats = self.inner.get_stats();
        let dict = PyDict::new(py);
        
        for (key, value) in stats {
            dict.set_item(key, value.to_string())?;
        }
        
        Ok(dict.into())
    }

    #[cfg(feature = "simd")]
    fn process_cadd_scores_simd(&mut self, py: Python, scores: &PyArray1<f64>) -> PyResult<()> {
        let mut scores_slice = unsafe { scores.as_slice_mut() }
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to get mutable slice: {}", e)))?;
        
        // Create temporary variant processor for SIMD operation
        let variant_processor = VariantProcessor::new(&self.inner.config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create variant processor: {}", e)))?;
        
        variant_processor.process_cadd_scores_simd(scores_slice);
        
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("PyGospelProcessor(config={:?})", self.inner.config)
    }
}

/// Standalone functions for direct Python access
#[pyfunction]
fn create_variant(
    chromosome: String,
    position: u64,
    reference: String,
    alternative: String,
    quality: f64,
) -> PyVariant {
    PyVariant::new(chromosome, position, reference, alternative, quality)
}

#[pyfunction]
fn create_config(
    enable_simd: Option<bool>,
    num_threads: Option<usize>,
    chunk_size: Option<usize>,
    enable_tracing: Option<bool>,
) -> PyGospelConfig {
    PyGospelConfig::new(
        enable_simd.unwrap_or(true),
        num_threads.unwrap_or(0),
        chunk_size.unwrap_or(64 * 1024 * 1024),
        enable_tracing.unwrap_or(false),
    )
}

#[pyfunction]
fn benchmark_rust_vs_python(num_variants: usize) -> PyResult<HashMap<String, f64>> {
    let start = std::time::Instant::now();
    
    // Generate test variants
    let variants: Vec<Variant> = (0..num_variants)
        .map(|i| {
            let mut variant = Variant::new(
                format!("chr{}", (i % 22) + 1),
                (i as u64 + 1) * 1000,
                "A".to_string(),
                "T".to_string(),
                30.0 + (i % 40) as f64,
            );
            variant.cadd_score = Some(10.0 + (i % 30) as f64);
            variant.conservation_score = Some((i % 100) as f64 / 100.0);
            variant.allele_frequency = Some((i % 1000) as f64 / 100000.0);
            variant
        })
        .collect();
    
    let generation_time = start.elapsed();
    
    // Process with Rust
    let rust_start = std::time::Instant::now();
    let _pathogenicity_scores: Vec<f64> = variants
        .iter()
        .map(|v| v.pathogenicity_score())
        .collect();
    let rust_time = rust_start.elapsed();
    
    let mut results = HashMap::new();
    results.insert("generation_time_ms".to_string(), generation_time.as_millis() as f64);
    results.insert("rust_processing_time_ms".to_string(), rust_time.as_millis() as f64);
    results.insert("variants_processed".to_string(), num_variants as f64);
    results.insert("rust_throughput".to_string(), num_variants as f64 / rust_time.as_secs_f64());
    
    Ok(results)
}

/// Python module definition
#[pymodule]
fn gospel_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGospelConfig>()?;
    m.add_class::<PyVariant>()?;
    m.add_class::<PyVariantStats>()?;
    m.add_class::<PyFuzzyResult>()?;
    m.add_class::<PyGospelProcessor>()?;
    
    m.add_function(wrap_pyfunction!(create_variant, m)?)?;
    m.add_function(wrap_pyfunction!(create_config, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_rust_vs_python, m)?)?;
    
    // Module constants
    m.add("__version__", "0.2.0")?;
    m.add("__author__", "Kundai Sachikonye")?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::prepare_freethreaded_python;

    #[test]
    fn test_python_bindings() {
        prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            // Test config creation
            let config = PyGospelConfig::new(true, 4, 1024*1024, false);
            assert!(config.enable_simd());
            assert_eq!(config.num_threads(), 4);
            
            // Test variant creation
            let variant = PyVariant::new(
                "chr1".to_string(),
                12345,
                "A".to_string(),
                "T".to_string(),
                30.0,
            );
            assert_eq!(variant.chromosome(), "chr1");
            assert_eq!(variant.position(), 12345);
            
            // Test pathogenicity calculation
            let score = variant.pathogenicity_score();
            assert!(score >= 0.0 && score <= 1.0);
        });
    }

    #[test]
    fn test_benchmark_function() {
        let results = benchmark_rust_vs_python(1000).unwrap();
        
        assert!(results.contains_key("rust_processing_time_ms"));
        assert!(results.contains_key("rust_throughput"));
        assert_eq!(results["variants_processed"], 1000.0);
        assert!(results["rust_throughput"] > 0.0);
    }
} 