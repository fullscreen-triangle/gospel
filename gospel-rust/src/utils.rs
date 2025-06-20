//! Utility functions for Gospel Rust
//!
//! This module provides common utility functions used across the Gospel Rust codebase.

use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance timer for benchmarking
#[derive(Debug)]
pub struct Timer {
    start_time: Instant,
    label: String,
}

impl Timer {
    /// Create a new timer with a label
    pub fn new(label: &str) -> Self {
        Self {
            start_time: Instant::now(),
            label: label.to_string(),
        }
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Get elapsed time as Duration
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Print elapsed time
    pub fn print_elapsed(&self) {
        println!("{}: {:.2}ms", self.label, self.elapsed().as_secs_f64() * 1000.0);
    }
}

/// Memory usage tracker
#[derive(Debug, Default)]
pub struct MemoryTracker {
    allocations: HashMap<String, usize>,
    total_allocated: usize,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Track memory allocation
    pub fn allocate(&mut self, label: &str, size: usize) {
        *self.allocations.entry(label.to_string()).or_insert(0) += size;
        self.total_allocated += size;
    }

    /// Track memory deallocation
    pub fn deallocate(&mut self, label: &str, size: usize) {
        if let Some(allocated) = self.allocations.get_mut(label) {
            *allocated = allocated.saturating_sub(size);
            self.total_allocated = self.total_allocated.saturating_sub(size);
        }
    }

    /// Get total allocated memory
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Get allocations by label
    pub fn get_allocations(&self) -> &HashMap<String, usize> {
        &self.allocations
    }

    /// Print memory usage summary
    pub fn print_summary(&self) {
        println!("Memory Usage Summary:");
        println!("Total allocated: {} bytes ({:.2} MB)", 
                 self.total_allocated, 
                 self.total_allocated as f64 / (1024.0 * 1024.0));
        
        for (label, size) in &self.allocations {
            if *size > 0 {
                println!("  {}: {} bytes ({:.2} MB)", 
                         label, 
                         size, 
                         *size as f64 / (1024.0 * 1024.0));
            }
        }
    }
}

/// Statistical utilities
pub struct Stats;

impl Stats {
    /// Calculate mean of a slice
    pub fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    /// Calculate standard deviation
    pub fn std_dev(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = Self::mean(values);
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        variance.sqrt()
    }

    /// Calculate median
    pub fn median(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = sorted.len();
        if len % 2 == 0 {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        }
    }

    /// Calculate percentile
    pub fn percentile(values: &[f64], p: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (p / 100.0) * (sorted.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        
        if lower == upper {
            sorted[lower]
        } else {
            let weight = index - lower as f64;
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        }
    }

    /// Calculate correlation coefficient
    pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        
        let mean_x = Self::mean(x);
        let mean_y = Self::mean(y);
        
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

    /// Calculate z-score
    pub fn z_score(value: f64, mean: f64, std_dev: f64) -> f64 {
        if std_dev == 0.0 {
            0.0
        } else {
            (value - mean) / std_dev
        }
    }
}

/// File utilities
pub struct FileUtils;

impl FileUtils {
    /// Check if file exists and is readable
    pub fn is_readable(path: &str) -> bool {
        std::fs::metadata(path).is_ok()
    }

    /// Get file size in bytes
    pub fn file_size(path: &str) -> Result<u64> {
        let metadata = std::fs::metadata(path)?;
        Ok(metadata.len())
    }

    /// Format file size in human-readable format
    pub fn format_file_size(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;
        
        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }
        
        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }

    /// Create directory if it doesn't exist
    pub fn ensure_directory(path: &str) -> Result<()> {
        std::fs::create_dir_all(path)?;
        Ok(())
    }
}

/// String utilities
pub struct StringUtils;

impl StringUtils {
    /// Convert snake_case to camelCase
    pub fn snake_to_camel(snake_str: &str) -> String {
        let mut camel = String::new();
        let mut capitalize_next = false;
        
        for ch in snake_str.chars() {
            if ch == '_' {
                capitalize_next = true;
            } else if capitalize_next {
                camel.push(ch.to_uppercase().next().unwrap_or(ch));
                capitalize_next = false;
            } else {
                camel.push(ch);
            }
        }
        
        camel
    }

    /// Convert camelCase to snake_case
    pub fn camel_to_snake(camel_str: &str) -> String {
        let mut snake = String::new();
        
        for (i, ch) in camel_str.chars().enumerate() {
            if ch.is_uppercase() && i > 0 {
                snake.push('_');
            }
            snake.push(ch.to_lowercase().next().unwrap_or(ch));
        }
        
        snake
    }

    /// Truncate string to specified length
    pub fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...", &s[..max_len.saturating_sub(3)])
        }
    }

    /// Remove whitespace and convert to lowercase
    pub fn normalize(s: &str) -> String {
        s.trim().to_lowercase().replace(' ', "_")
    }
}

/// Math utilities
pub struct MathUtils;

impl MathUtils {
    /// Clamp value between min and max
    pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
        value.max(min).min(max)
    }

    /// Linear interpolation
    pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + t * (b - a)
    }

    /// Map value from one range to another
    pub fn map_range(value: f64, from_min: f64, from_max: f64, to_min: f64, to_max: f64) -> f64 {
        let normalized = (value - from_min) / (from_max - from_min);
        to_min + normalized * (to_max - to_min)
    }

    /// Calculate sigmoid function
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Calculate log-odds
    pub fn log_odds(probability: f64) -> f64 {
        let p = probability.max(1e-10).min(1.0 - 1e-10); // Avoid division by zero
        (p / (1.0 - p)).ln()
    }

    /// Calculate probability from log-odds
    pub fn odds_to_probability(log_odds: f64) -> f64 {
        let odds = log_odds.exp();
        odds / (1.0 + odds)
    }

    /// Check if value is approximately equal (within epsilon)
    pub fn approx_equal(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    /// Safe division (returns 0 if denominator is 0)
    pub fn safe_divide(numerator: f64, denominator: f64) -> f64 {
        if denominator.abs() < f64::EPSILON {
            0.0
        } else {
            numerator / denominator
        }
    }
}

/// Validation utilities
pub struct Validation;

impl Validation {
    /// Validate that a value is within a range
    pub fn in_range(value: f64, min: f64, max: f64) -> Result<()> {
        if value < min || value > max {
            return Err(anyhow::anyhow!(
                "Value {} is not in range [{}, {}]", value, min, max
            ));
        }
        Ok(())
    }

    /// Validate that a probability is between 0 and 1
    pub fn is_probability(value: f64) -> Result<()> {
        Self::in_range(value, 0.0, 1.0)
    }

    /// Validate that arrays have the same length
    pub fn same_length<T, U>(a: &[T], b: &[U]) -> Result<()> {
        if a.len() != b.len() {
            return Err(anyhow::anyhow!(
                "Arrays have different lengths: {} vs {}", a.len(), b.len()
            ));
        }
        Ok(())
    }

    /// Validate that array is not empty
    pub fn not_empty<T>(array: &[T]) -> Result<()> {
        if array.is_empty() {
            return Err(anyhow::anyhow!("Array cannot be empty"));
        }
        Ok(())
    }

    /// Validate that matrix dimensions are consistent
    pub fn matrix_dimensions(data: &[f64], rows: usize, cols: usize) -> Result<()> {
        if data.len() != rows * cols {
            return Err(anyhow::anyhow!(
                "Matrix data length {} does not match dimensions {}x{} (expected {})",
                data.len(), rows, cols, rows * cols
            ));
        }
        Ok(())
    }
}

/// Logging utilities
pub struct Logger;

impl Logger {
    /// Log info message with timestamp
    pub fn info(message: &str) {
        let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
        println!("[{}] INFO: {}", timestamp, message);
    }

    /// Log warning message with timestamp
    pub fn warn(message: &str) {
        let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
        eprintln!("[{}] WARN: {}", timestamp, message);
    }

    /// Log error message with timestamp
    pub fn error(message: &str) {
        let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
        eprintln!("[{}] ERROR: {}", timestamp, message);
    }

    /// Log debug message with timestamp (only in debug builds)
    pub fn debug(message: &str) {
        #[cfg(debug_assertions)]
        {
            let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
            println!("[{}] DEBUG: {}", timestamp, message);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_timer() {
        let timer = Timer::new("test");
        thread::sleep(Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10);
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new();
        tracker.allocate("test", 1024);
        assert_eq!(tracker.total_allocated(), 1024);
        
        tracker.deallocate("test", 512);
        assert_eq!(tracker.total_allocated(), 512);
    }

    #[test]
    fn test_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_eq!(Stats::mean(&values), 3.0);
        assert_eq!(Stats::median(&values), 3.0);
        assert!((Stats::std_dev(&values) - 1.5811388300841898).abs() < 1e-10);
        assert_eq!(Stats::percentile(&values, 50.0), 3.0);
        
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        assert!((Stats::correlation(&x, &y) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_string_utils() {
        assert_eq!(StringUtils::snake_to_camel("hello_world"), "helloWorld");
        assert_eq!(StringUtils::camel_to_snake("helloWorld"), "hello_world");
        assert_eq!(StringUtils::truncate("hello world", 8), "hello...");
        assert_eq!(StringUtils::normalize("  Hello World  "), "hello_world");
    }

    #[test]
    fn test_math_utils() {
        assert_eq!(MathUtils::clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(MathUtils::clamp(-1.0, 0.0, 10.0), 0.0);
        assert_eq!(MathUtils::clamp(15.0, 0.0, 10.0), 10.0);
        
        assert_eq!(MathUtils::lerp(0.0, 10.0, 0.5), 5.0);
        assert_eq!(MathUtils::map_range(5.0, 0.0, 10.0, 0.0, 100.0), 50.0);
        
        assert!((MathUtils::sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(MathUtils::sigmoid(10.0) > 0.9);
        assert!(MathUtils::sigmoid(-10.0) < 0.1);
        
        assert!(MathUtils::approx_equal(1.0, 1.0000001, 1e-6));
        assert!(!MathUtils::approx_equal(1.0, 1.001, 1e-6));
        
        assert_eq!(MathUtils::safe_divide(10.0, 2.0), 5.0);
        assert_eq!(MathUtils::safe_divide(10.0, 0.0), 0.0);
    }

    #[test]
    fn test_validation() {
        assert!(Validation::in_range(5.0, 0.0, 10.0).is_ok());
        assert!(Validation::in_range(-1.0, 0.0, 10.0).is_err());
        
        assert!(Validation::is_probability(0.5).is_ok());
        assert!(Validation::is_probability(1.5).is_err());
        
        assert!(Validation::same_length(&[1, 2, 3], &[4, 5, 6]).is_ok());
        assert!(Validation::same_length(&[1, 2], &[4, 5, 6]).is_err());
        
        assert!(Validation::not_empty(&[1, 2, 3]).is_ok());
        assert!(Validation::not_empty(&Vec::<i32>::new()).is_err());
        
        assert!(Validation::matrix_dimensions(&[1.0, 2.0, 3.0, 4.0], 2, 2).is_ok());
        assert!(Validation::matrix_dimensions(&[1.0, 2.0, 3.0], 2, 2).is_err());
    }

    #[test]
    fn test_file_utils() {
        assert_eq!(FileUtils::format_file_size(1024), "1.00 KB");
        assert_eq!(FileUtils::format_file_size(1048576), "1.00 MB");
        assert_eq!(FileUtils::format_file_size(512), "512 B");
    }
} 