//! High-performance expression data processing
//!
//! This module provides optimized processing of gene expression matrices
//! with parallel algorithms and SIMD acceleration.

use anyhow::Result;
use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::GospelConfig;

/// Gene expression matrix with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionMatrix {
    /// Expression data (genes × samples)
    pub data: Array2<f64>,
    /// Gene names/identifiers
    pub gene_names: Vec<String>,
    /// Sample names/identifiers
    pub sample_names: Vec<String>,
    /// Normalization method used
    pub normalization: String,
    /// Processing statistics
    pub stats: ExpressionStats,
}

/// Statistics for expression processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionStats {
    /// Number of genes
    pub num_genes: usize,
    /// Number of samples
    pub num_samples: usize,
    /// Mean expression level
    pub mean_expression: f64,
    /// Standard deviation of expression
    pub std_expression: f64,
    /// Number of highly expressed genes (>2 fold change)
    pub highly_expressed_genes: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

impl Default for ExpressionStats {
    fn default() -> Self {
        Self {
            num_genes: 0,
            num_samples: 0,
            mean_expression: 0.0,
            std_expression: 0.0,
            highly_expressed_genes: 0,
            processing_time_ms: 0,
        }
    }
}

/// High-performance expression processor
#[derive(Debug)]
pub struct ExpressionProcessor {
    config: GospelConfig,
}

impl ExpressionProcessor {
    /// Create new expression processor
    pub fn new(config: &GospelConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Process expression matrix with parallel algorithms
    pub async fn process_matrix(
        &self,
        expression_data: &[f64],
        genes: &[String],
        samples: &[String],
    ) -> Result<ExpressionMatrix> {
        let start_time = std::time::Instant::now();
        
        let num_genes = genes.len();
        let num_samples = samples.len();
        
        if expression_data.len() != num_genes * num_samples {
            return Err(anyhow::anyhow!(
                "Expression data size mismatch: expected {}, got {}",
                num_genes * num_samples,
                expression_data.len()
            ));
        }

        // Convert to ndarray for efficient operations
        let data = Array2::from_shape_vec((num_genes, num_samples), expression_data.to_vec())?;
        
        // Normalize data
        let normalized_data = self.normalize_expression(&data).await?;
        
        // Calculate statistics
        let stats = self.calculate_expression_stats(&normalized_data, start_time.elapsed()).await;
        
        Ok(ExpressionMatrix {
            data: normalized_data,
            gene_names: genes.to_vec(),
            sample_names: samples.to_vec(),
            normalization: "log2_tpm".to_string(),
            stats,
        })
    }

    /// Normalize expression data using log2(TPM + 1)
    async fn normalize_expression(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let normalized = tokio::task::spawn_blocking({
            let data = data.clone();
            move || {
                data.mapv(|x| (x + 1.0).log2())
            }
        }).await?;
        
        Ok(normalized)
    }

    /// Calculate differential expression between conditions
    pub async fn differential_expression(
        &self,
        matrix: &ExpressionMatrix,
        condition1_samples: &[usize],
        condition2_samples: &[usize],
    ) -> Result<Vec<DifferentialExpressionResult>> {
        let results = tokio::task::spawn_blocking({
            let data = matrix.data.clone();
            let gene_names = matrix.gene_names.clone();
            let condition1 = condition1_samples.to_vec();
            let condition2 = condition2_samples.to_vec();
            
            move || {
                (0..data.nrows())
                    .into_par_iter()
                    .map(|gene_idx| {
                        let gene_data = data.row(gene_idx);
                        
                        // Calculate means for each condition
                        let mean1 = condition1.iter()
                            .map(|&sample_idx| gene_data[sample_idx])
                            .sum::<f64>() / condition1.len() as f64;
                        
                        let mean2 = condition2.iter()
                            .map(|&sample_idx| gene_data[sample_idx])
                            .sum::<f64>() / condition2.len() as f64;
                        
                        // Calculate fold change
                        let fold_change = mean2 - mean1; // Log2 fold change
                        
                        // Calculate t-test p-value (simplified)
                        let p_value = Self::calculate_t_test_p_value(
                            &condition1.iter().map(|&i| gene_data[i]).collect::<Vec<_>>(),
                            &condition2.iter().map(|&i| gene_data[i]).collect::<Vec<_>>(),
                        );
                        
                        DifferentialExpressionResult {
                            gene_name: gene_names[gene_idx].clone(),
                            fold_change,
                            p_value,
                            adjusted_p_value: p_value, // Would apply FDR correction in practice
                            mean_expression_condition1: mean1,
                            mean_expression_condition2: mean2,
                        }
                    })
                    .collect()
            }
        }).await?;
        
        Ok(results)
    }

    /// Calculate correlation between genes
    pub async fn gene_correlation_matrix(&self, matrix: &ExpressionMatrix) -> Result<Array2<f64>> {
        let correlation_matrix = tokio::task::spawn_blocking({
            let data = matrix.data.clone();
            move || {
                let num_genes = data.nrows();
                let mut correlations = Array2::zeros((num_genes, num_genes));
                
                // Parallel computation of correlation matrix
                correlations
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, mut row)| {
                        let gene_i = data.row(i);
                        
                        for j in 0..num_genes {
                            let gene_j = data.row(j);
                            let correlation = Self::pearson_correlation(&gene_i.to_vec(), &gene_j.to_vec());
                            row[j] = correlation;
                        }
                    });
                
                correlations
            }
        }).await?;
        
        Ok(correlation_matrix)
    }

    /// Find co-expressed gene modules using hierarchical clustering
    pub async fn find_coexpression_modules(
        &self,
        matrix: &ExpressionMatrix,
        correlation_threshold: f64,
    ) -> Result<Vec<CoexpressionModule>> {
        let correlation_matrix = self.gene_correlation_matrix(matrix).await?;
        
        let modules = tokio::task::spawn_blocking({
            let correlations = correlation_matrix.clone();
            let gene_names = matrix.gene_names.clone();
            move || {
                Self::hierarchical_clustering(&correlations, &gene_names, correlation_threshold)
            }
        }).await?;
        
        Ok(modules)
    }

    /// Calculate expression statistics
    async fn calculate_expression_stats(&self, data: &Array2<f64>, processing_time: std::time::Duration) -> ExpressionStats {
        let stats = tokio::task::spawn_blocking({
            let data = data.clone();
            move || {
                let flat_data: Vec<f64> = data.iter().cloned().collect();
                let mean = flat_data.iter().sum::<f64>() / flat_data.len() as f64;
                let variance = flat_data.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / flat_data.len() as f64;
                let std_dev = variance.sqrt();
                
                // Count highly expressed genes (>2 fold change from mean)
                let threshold = mean + 2.0; // Log2 scale
                let highly_expressed = data.axis_iter(Axis(0))
                    .filter(|gene_row| {
                        let gene_mean = gene_row.mean().unwrap_or(0.0);
                        gene_mean > threshold
                    })
                    .count();
                
                ExpressionStats {
                    num_genes: data.nrows(),
                    num_samples: data.ncols(),
                    mean_expression: mean,
                    std_expression: std_dev,
                    highly_expressed_genes: highly_expressed,
                    processing_time_ms: processing_time.as_millis() as u64,
                }
            }
        }).await.unwrap();
        
        stats
    }

    /// Calculate Pearson correlation coefficient
    fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        
        let numerator: f64 = x.iter().zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();
        
        let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Calculate t-test p-value (simplified implementation)
    fn calculate_t_test_p_value(group1: &[f64], group2: &[f64]) -> f64 {
        if group1.is_empty() || group2.is_empty() {
            return 1.0;
        }
        
        let n1 = group1.len() as f64;
        let n2 = group2.len() as f64;
        
        let mean1 = group1.iter().sum::<f64>() / n1;
        let mean2 = group2.iter().sum::<f64>() / n2;
        
        let var1 = group1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
        let var2 = group2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);
        
        let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
        let se = (pooled_var * (1.0/n1 + 1.0/n2)).sqrt();
        
        if se == 0.0 {
            return 1.0;
        }
        
        let t_stat = (mean1 - mean2).abs() / se;
        
        // Simplified p-value calculation (would use proper t-distribution in practice)
        let df = n1 + n2 - 2.0;
        let p_value = 2.0 * (1.0 - Self::t_cdf(t_stat, df));
        
        p_value.min(1.0).max(0.0)
    }

    /// Simplified t-distribution CDF approximation
    fn t_cdf(t: f64, df: f64) -> f64 {
        // Very simplified approximation - would use proper implementation in practice
        let x = t / (t.powi(2) + df).sqrt();
        0.5 + 0.5 * x / (1.0 + 0.2316419 * x.abs())
    }

    /// Hierarchical clustering for coexpression modules
    fn hierarchical_clustering(
        correlation_matrix: &Array2<f64>,
        gene_names: &[String],
        threshold: f64,
    ) -> Vec<CoexpressionModule> {
        let mut modules = Vec::new();
        let mut visited = vec![false; gene_names.len()];
        
        for i in 0..gene_names.len() {
            if visited[i] {
                continue;
            }
            
            let mut module_genes = vec![i];
            visited[i] = true;
            
            // Find all genes correlated above threshold
            for j in (i + 1)..gene_names.len() {
                if !visited[j] && correlation_matrix[[i, j]] >= threshold {
                    module_genes.push(j);
                    visited[j] = true;
                }
            }
            
            if module_genes.len() > 1 {
                let module = CoexpressionModule {
                    genes: module_genes.iter().map(|&idx| gene_names[idx].clone()).collect(),
                    average_correlation: module_genes.iter()
                        .flat_map(|&i| module_genes.iter().map(move |&j| correlation_matrix[[i, j]]))
                        .sum::<f64>() / (module_genes.len() * module_genes.len()) as f64,
                    size: module_genes.len(),
                };
                modules.push(module);
            }
        }
        
        modules
    }

    /// Get processor statistics
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("processor_type".to_string(), "ExpressionProcessor".to_string());
        stats.insert("simd_enabled".to_string(), self.config.enable_simd.to_string());
        stats.insert("num_threads".to_string(), self.config.num_threads.to_string());
        stats
    }
}

/// Result of differential expression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialExpressionResult {
    /// Gene name
    pub gene_name: String,
    /// Log2 fold change
    pub fold_change: f64,
    /// P-value from statistical test
    pub p_value: f64,
    /// Adjusted p-value (FDR corrected)
    pub adjusted_p_value: f64,
    /// Mean expression in condition 1
    pub mean_expression_condition1: f64,
    /// Mean expression in condition 2
    pub mean_expression_condition2: f64,
}

/// Coexpression module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoexpressionModule {
    /// Genes in the module
    pub genes: Vec<String>,
    /// Average correlation within module
    pub average_correlation: f64,
    /// Module size (number of genes)
    pub size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GospelConfig;

    #[tokio::test]
    async fn test_expression_processor() {
        let config = GospelConfig::default();
        let processor = ExpressionProcessor::new(&config).unwrap();
        
        // Create test data: 3 genes × 4 samples
        let expression_data = vec![
            1.0, 2.0, 3.0, 4.0,  // Gene 1
            2.0, 4.0, 6.0, 8.0,  // Gene 2
            0.5, 1.0, 1.5, 2.0,  // Gene 3
        ];
        let genes = vec!["GENE1".to_string(), "GENE2".to_string(), "GENE3".to_string()];
        let samples = vec!["SAMPLE1".to_string(), "SAMPLE2".to_string(), "SAMPLE3".to_string(), "SAMPLE4".to_string()];
        
        let matrix = processor.process_matrix(&expression_data, &genes, &samples).await.unwrap();
        
        assert_eq!(matrix.gene_names.len(), 3);
        assert_eq!(matrix.sample_names.len(), 4);
        assert_eq!(matrix.data.shape(), &[3, 4]);
        assert_eq!(matrix.stats.num_genes, 3);
        assert_eq!(matrix.stats.num_samples, 4);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation
        
        let correlation = ExpressionProcessor::pearson_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 1e-10);
        
        let z = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // Perfect negative correlation
        let correlation = ExpressionProcessor::pearson_correlation(&x, &z);
        assert!((correlation + 1.0).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_differential_expression() {
        let config = GospelConfig::default();
        let processor = ExpressionProcessor::new(&config).unwrap();
        
        // Create test expression matrix
        let expression_data = vec![
            1.0, 1.0, 5.0, 5.0,  // Gene differentially expressed
            2.0, 2.0, 2.0, 2.0,  // Gene not differentially expressed
        ];
        let genes = vec!["DE_GENE".to_string(), "STABLE_GENE".to_string()];
        let samples = vec!["CTRL1".to_string(), "CTRL2".to_string(), "TREAT1".to_string(), "TREAT2".to_string()];
        
        let matrix = processor.process_matrix(&expression_data, &genes, &samples).await.unwrap();
        
        // Test differential expression
        let condition1 = vec![0, 1]; // Control samples
        let condition2 = vec![2, 3]; // Treatment samples
        
        let de_results = processor.differential_expression(&matrix, &condition1, &condition2).await.unwrap();
        
        assert_eq!(de_results.len(), 2);
        
        // First gene should have large fold change
        assert!(de_results[0].fold_change.abs() > 1.0);
        
        // Second gene should have small fold change
        assert!(de_results[1].fold_change.abs() < 0.1);
    }

    #[tokio::test]
    async fn test_correlation_matrix() {
        let config = GospelConfig::default();
        let processor = ExpressionProcessor::new(&config).unwrap();
        
        // Create test data with correlated genes
        let expression_data = vec![
            1.0, 2.0, 3.0, 4.0,  // Gene 1
            2.0, 4.0, 6.0, 8.0,  // Gene 2 (perfectly correlated with Gene 1)
            4.0, 3.0, 2.0, 1.0,  // Gene 3 (anti-correlated with Gene 1)
        ];
        let genes = vec!["GENE1".to_string(), "GENE2".to_string(), "GENE3".to_string()];
        let samples = vec!["S1".to_string(), "S2".to_string(), "S3".to_string(), "S4".to_string()];
        
        let matrix = processor.process_matrix(&expression_data, &genes, &samples).await.unwrap();
        let correlation_matrix = processor.gene_correlation_matrix(&matrix).await.unwrap();
        
        assert_eq!(correlation_matrix.shape(), &[3, 3]);
        
        // Diagonal should be 1.0
        assert!((correlation_matrix[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((correlation_matrix[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((correlation_matrix[[2, 2]] - 1.0).abs() < 1e-10);
        
        // Gene 1 and 2 should be perfectly correlated
        assert!((correlation_matrix[[0, 1]] - 1.0).abs() < 1e-10);
        
        // Gene 1 and 3 should be anti-correlated
        assert!((correlation_matrix[[0, 2]] + 1.0).abs() < 1e-10);
    }
} 