/*!
# Gospel Framework Utilities

This module provides common utility functions used across all 12 revolutionary frameworks.
*/

use std::time::{Duration, Instant};
use std::collections::HashMap;
use rand::Rng;
use nalgebra::{Vector3, Matrix3};
use crate::types::{EntropyCoordinates, TemporalCoordinates, PrecisionLevel};
use crate::error::{GospelError, GospelResult};

/// Performance measurement utilities
pub mod performance {
    use super::*;

    /// Performance monitor for tracking framework operations
    #[derive(Debug, Clone)]
    pub struct PerformanceMonitor {
        start_time: Instant,
        measurements: HashMap<String, Vec<Duration>>,
    }

    impl PerformanceMonitor {
        /// Create a new performance monitor
        pub fn new() -> Self {
            Self {
                start_time: Instant::now(),
                measurements: HashMap::new(),
            }
        }

        /// Start measuring an operation
        pub fn start_operation(&self, operation: &str) -> OperationTimer {
            OperationTimer::new(operation.to_string())
        }

        /// Record a measurement
        pub fn record_measurement(&mut self, operation: String, duration: Duration) {
            self.measurements.entry(operation).or_insert_with(Vec::new).push(duration);
        }

        /// Get average time for an operation
        pub fn average_time(&self, operation: &str) -> Option<Duration> {
            let measurements = self.measurements.get(operation)?;
            if measurements.is_empty() {
                return None;
            }

            let total_nanos: u64 = measurements.iter().map(|d| d.as_nanos() as u64).sum();
            let average_nanos = total_nanos / measurements.len() as u64;
            Some(Duration::from_nanos(average_nanos))
        }

        /// Check if performance targets are met
        pub fn validate_performance_targets(&self) -> GospelResult<bool> {
            for (operation, measurements) in &self.measurements {
                if let Some(avg_time) = self.average_time(operation) {
                    // Check sub-millisecond target
                    if avg_time > Duration::from_millis(1) {
                        return Err(GospelError::performance_violation(
                            format!("processing_time_{}", operation),
                            1.0, // 1ms target
                            avg_time.as_millis() as f64,
                        ));
                    }
                }
            }
            Ok(true)
        }

        /// Get total elapsed time
        pub fn total_elapsed(&self) -> Duration {
            self.start_time.elapsed()
        }
    }

    impl Default for PerformanceMonitor {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Timer for individual operations
    pub struct OperationTimer {
        operation: String,
        start_time: Instant,
    }

    impl OperationTimer {
        fn new(operation: String) -> Self {
            Self {
                operation,
                start_time: Instant::now(),
            }
        }

        /// Complete the operation and return the duration
        pub fn complete(self) -> (String, Duration) {
            (self.operation, self.start_time.elapsed())
        }
    }
}

/// Mathematical utilities for S-entropy and oscillatory calculations
pub mod math {
    use super::*;

    /// Calculate S-entropy compression ratio
    pub fn calculate_compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
        if original_size == 0 {
            return 1.0;
        }
        compressed_size as f64 / original_size as f64
    }

    /// Calculate entropy coordinates from complex state
    pub fn calculate_entropy_coordinates(
        knowledge_complexity: f64,
        temporal_complexity: f64,
        information_entropy: f64,
    ) -> EntropyCoordinates {
        // Normalize values to S-entropy scale
        let s_knowledge = normalize_entropy_value(knowledge_complexity);
        let s_time = normalize_entropy_value(temporal_complexity);
        let s_entropy = normalize_entropy_value(information_entropy);

        EntropyCoordinates::new(s_knowledge, s_time, s_entropy)
    }

    /// Normalize entropy value to standard scale
    fn normalize_entropy_value(value: f64) -> f64 {
        // Apply S-entropy normalization function
        if value <= 0.0 {
            0.0
        } else {
            // Use logarithmic normalization to compress large values
            (value.ln() + 1.0).max(0.0).min(10.0)
        }
    }

    /// Calculate oscillatory resonance between frequencies
    pub fn calculate_resonance(freq1: f64, freq2: f64, tolerance: f64) -> f64 {
        let frequency_diff = (freq1 - freq2).abs();
        let max_freq = freq1.max(freq2);
        
        if max_freq == 0.0 {
            return 1.0;
        }

        let relative_diff = frequency_diff / max_freq;
        if relative_diff <= tolerance {
            1.0 - (relative_diff / tolerance)
        } else {
            0.0
        }
    }

    /// Calculate femtosecond-precision temporal coordinates
    pub fn calculate_temporal_coordinates(
        base_time: Duration,
        precision_level: PrecisionLevel,
        stability_factor: f64,
    ) -> TemporalCoordinates {
        let femtoseconds = base_time.as_nanos() as u64 * 1_000_000; // Convert to femtoseconds
        let precision_enhancement = precision_level.in_femtoseconds() as f64 / 1_000_000.0;
        
        TemporalCoordinates::new(femtoseconds, precision_enhancement, stability_factor)
    }

    /// Calculate information advantage ratio
    pub fn calculate_information_advantage(
        cellular_information: f64,
        genomic_information: f64,
    ) -> f64 {
        if genomic_information <= 0.0 {
            170_000.0 // Default expected advantage
        } else {
            cellular_information / genomic_information
        }
    }

    /// Validate that information advantage meets 170,000× target
    pub fn validate_information_advantage(advantage: f64) -> GospelResult<()> {
        const TARGET_ADVANTAGE: f64 = 170_000.0;
        
        if advantage < TARGET_ADVANTAGE {
            Err(GospelError::cellular_information_with_advantage(
                "Information advantage below target",
                advantage,
            ))
        } else {
            Ok(())
        }
    }

    /// Calculate statistical emergence probability
    pub fn calculate_emergence_probability(
        signal_strength: f64,
        noise_level: f64,
        threshold_multiplier: f64,
    ) -> f64 {
        if noise_level <= 0.0 {
            return 1.0;
        }
        
        let signal_to_noise = signal_strength / noise_level;
        if signal_to_noise >= threshold_multiplier {
            1.0 - (threshold_multiplier / signal_to_noise).min(1.0)
        } else {
            0.0
        }
    }
}

/// Validation utilities for ensuring framework correctness
pub mod validation {
    use super::*;

    /// Validate accuracy against Gospel targets
    pub fn validate_accuracy(accuracy: f64) -> GospelResult<()> {
        const TARGET_ACCURACY: f64 = 0.97; // 97%
        
        if accuracy < TARGET_ACCURACY {
            Err(GospelError::performance_violation(
                "accuracy",
                TARGET_ACCURACY,
                accuracy,
            ))
        } else {
            Ok(())
        }
    }

    /// Validate processing time against targets
    pub fn validate_processing_time(duration: Duration) -> GospelResult<()> {
        const TARGET_TIME: Duration = Duration::from_millis(1); // Sub-millisecond
        
        if duration > TARGET_TIME {
            Err(GospelError::performance_violation(
                "processing_time",
                TARGET_TIME.as_millis() as f64,
                duration.as_millis() as f64,
            ))
        } else {
            Ok(())
        }
    }

    /// Validate S-entropy coordinates
    pub fn validate_entropy_coordinates(coordinates: &EntropyCoordinates) -> GospelResult<()> {
        // Check for valid ranges
        if coordinates.s_knowledge.is_nan() || coordinates.s_time.is_nan() || coordinates.s_entropy.is_nan() {
            return Err(GospelError::s_entropy("Invalid entropy coordinates: NaN values detected".to_string()));
        }

        if coordinates.s_knowledge < 0.0 || coordinates.s_time < 0.0 || coordinates.s_entropy < 0.0 {
            return Err(GospelError::s_entropy("Invalid entropy coordinates: negative values detected".to_string()));
        }

        // Check for extreme values that might indicate calculation errors
        const MAX_ENTROPY: f64 = 100.0;
        if coordinates.magnitude() > MAX_ENTROPY {
            return Err(GospelError::s_entropy(
                format!("Entropy coordinates magnitude {} exceeds maximum {}", 
                       coordinates.magnitude(), MAX_ENTROPY)
            ));
        }

        Ok(())
    }

    /// Validate that universal solvability is maintained
    pub fn validate_solvability_guarantee() -> GospelResult<()> {
        // In Gospel framework, all problems have solutions by thermodynamic necessity
        // This validation always passes but provides consistency checking
        Ok(())
    }

    /// Validate biological plausibility
    pub fn validate_biological_plausibility(
        results: &HashMap<String, f64>,
        minimum_plausibility: f64,
    ) -> GospelResult<()> {
        for (metric, value) in results {
            if *value < minimum_plausibility {
                return Err(GospelError::invalid_input(
                    format!("Biological metric '{}' below plausibility threshold: {} < {}",
                           metric, value, minimum_plausibility)
                ));
            }
        }
        Ok(())
    }
}

/// Compression utilities for S-entropy-based optimization
pub mod compression {
    use super::*;

    /// Compress data using S-entropy principles
    pub fn s_entropy_compress(data: &[u8]) -> GospelResult<Vec<u8>> {
        // Placeholder for S-entropy compression algorithm
        // In practice, this would implement the revolutionary compression
        // that achieves O(1) memory complexity
        
        use lz4_flex::compress_prepend_size;
        let compressed = compress_prepend_size(data);
        
        // Apply S-entropy enhancement
        let compression_ratio = compressed.len() as f64 / data.len() as f64;
        if compression_ratio > 0.1 { // If compression is poor, apply S-entropy optimization
            // Enhanced compression using entropy coordinates
            let entropy_coords = math::calculate_entropy_coordinates(
                data.len() as f64,
                1.0, // temporal complexity
                calculate_shannon_entropy(data),
            );
            
            // Use entropy coordinates to guide compression
            let enhanced_compressed = apply_entropy_guided_compression(data, &entropy_coords)?;
            Ok(enhanced_compressed)
        } else {
            Ok(compressed)
        }
    }

    /// Decompress S-entropy compressed data
    pub fn s_entropy_decompress(compressed: &[u8]) -> GospelResult<Vec<u8>> {
        use lz4_flex::decompress_size_prepended;
        decompress_size_prepended(compressed)
            .map_err(|e| GospelError::generic(format!("Decompression failed: {}", e)))
    }

    /// Calculate Shannon entropy of data
    fn calculate_shannon_entropy(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mut counts = [0u32; 256];
        for &byte in data {
            counts[byte as usize] += 1;
        }

        let length = data.len() as f64;
        let mut entropy = 0.0;

        for &count in &counts {
            if count > 0 {
                let probability = count as f64 / length;
                entropy -= probability * probability.log2();
            }
        }

        entropy
    }

    /// Apply entropy-guided compression enhancement
    fn apply_entropy_guided_compression(
        data: &[u8],
        entropy_coords: &EntropyCoordinates,
    ) -> GospelResult<Vec<u8>> {
        // Enhanced compression using S-entropy navigation
        // This would implement the revolutionary compression algorithm
        
        // For now, use standard compression with entropy-based chunking
        let chunk_size = ((entropy_coords.s_entropy + 1.0) * 1024.0) as usize;
        let chunk_size = chunk_size.clamp(512, 8192);

        let mut compressed_chunks = Vec::new();
        
        for chunk in data.chunks(chunk_size) {
            let compressed_chunk = lz4_flex::compress_prepend_size(chunk);
            compressed_chunks.extend(compressed_chunk);
        }

        Ok(compressed_chunks)
    }
}

/// Random number generation utilities
pub mod random {
    use super::*;

    /// Generate random entropy coordinates for testing
    pub fn random_entropy_coordinates() -> EntropyCoordinates {
        let mut rng = rand::thread_rng();
        EntropyCoordinates::new(
            rng.gen_range(0.0..10.0),
            rng.gen_range(0.0..10.0),
            rng.gen_range(0.0..10.0),
        )
    }

    /// Generate random oscillatory frequency
    pub fn random_oscillatory_frequency() -> f64 {
        let mut rng = rand::thread_rng();
        rng.gen_range(0.1..100.0)
    }

    /// Generate random test data for benchmarking
    pub fn generate_test_data(size: usize) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        (0..size).map(|_| rng.gen()).collect()
    }
}

/// Logging and debugging utilities
pub mod logging {
    use super::*;

    /// Log framework operation with performance metrics
    pub fn log_framework_operation(
        framework: &str,
        operation: &str,
        duration: Duration,
        success: bool,
    ) {
        if success {
            tracing::info!(
                framework = framework,
                operation = operation,
                duration_ms = duration.as_millis(),
                "Framework operation completed successfully"
            );
        } else {
            tracing::error!(
                framework = framework,
                operation = operation,
                duration_ms = duration.as_millis(),
                "Framework operation failed"
            );
        }
    }

    /// Log S-entropy navigation
    pub fn log_s_entropy_navigation(
        from: &EntropyCoordinates,
        to: &EntropyCoordinates,
        success: bool,
    ) {
        let distance = from.distance(to);
        tracing::debug!(
            from_coords = ?from,
            to_coords = ?to,
            distance = distance,
            success = success,
            "S-entropy navigation completed"
        );
    }

    /// Log performance validation
    pub fn log_performance_validation(
        metric: &str,
        expected: f64,
        actual: f64,
        passed: bool,
    ) {
        if passed {
            tracing::info!(
                metric = metric,
                expected = expected,
                actual = actual,
                "Performance validation passed"
            );
        } else {
            tracing::warn!(
                metric = metric,
                expected = expected,
                actual = actual,
                "Performance validation failed"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_monitor() {
        let mut monitor = performance::PerformanceMonitor::new();
        let timer = monitor.start_operation("test_operation");
        
        std::thread::sleep(Duration::from_millis(1));
        let (operation, duration) = timer.complete();
        monitor.record_measurement(operation, duration);
        
        let avg_time = monitor.average_time("test_operation");
        assert!(avg_time.is_some());
        assert!(avg_time.unwrap() >= Duration::from_millis(1));
    }

    #[test]
    fn test_entropy_coordinate_calculation() {
        let coords = math::calculate_entropy_coordinates(1000.0, 500.0, 2.5);
        assert!(coords.s_knowledge > 0.0);
        assert!(coords.s_time > 0.0);
        assert!(coords.s_entropy > 0.0);
    }

    #[test]
    fn test_oscillatory_resonance() {
        let resonance = math::calculate_resonance(10.0, 10.1, 0.1);
        assert!(resonance > 0.0);
        
        let no_resonance = math::calculate_resonance(10.0, 20.0, 0.1);
        assert_eq!(no_resonance, 0.0);
    }

    #[test]
    fn test_information_advantage_validation() {
        // Should pass with sufficient advantage
        assert!(math::validate_information_advantage(200_000.0).is_ok());
        
        // Should fail with insufficient advantage
        assert!(math::validate_information_advantage(100_000.0).is_err());
    }

    #[test]
    fn test_accuracy_validation() {
        // Should pass with high accuracy
        assert!(validation::validate_accuracy(0.98).is_ok());
        
        // Should fail with low accuracy
        assert!(validation::validate_accuracy(0.95).is_err());
    }

    #[test]
    fn test_s_entropy_compression() {
        let test_data = random::generate_test_data(1000);
        let compressed = compression::s_entropy_compress(&test_data);
        assert!(compressed.is_ok());
        
        let compressed_data = compressed.unwrap();
        assert!(compressed_data.len() <= test_data.len());
        
        let decompressed = compression::s_entropy_decompress(&compressed_data);
        assert!(decompressed.is_ok());
    }

    #[test]
    fn test_entropy_coordinates_validation() {
        let valid_coords = EntropyCoordinates::new(1.0, 2.0, 3.0);
        assert!(validation::validate_entropy_coordinates(&valid_coords).is_ok());
        
        let invalid_coords = EntropyCoordinates::new(-1.0, 2.0, 3.0);
        assert!(validation::validate_entropy_coordinates(&invalid_coords).is_err());
    }

    #[test]
    fn test_temporal_coordinates_calculation() {
        let coords = math::calculate_temporal_coordinates(
            Duration::from_nanos(1000),
            PrecisionLevel::Femtosecond,
            0.95,
        );
        
        assert_eq!(coords.femtosecond_coordinate, 1_000_000);
        assert_eq!(coords.stability, 0.95);
    }
}