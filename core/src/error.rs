/*!
# Gospel Framework Error Types

This module defines comprehensive error types for the Gospel framework, providing
detailed error information across all 12 revolutionary frameworks.
*/

use std::fmt;
use thiserror::Error;

/// Main error type for the Gospel framework
#[derive(Error, Debug)]
pub enum GospelError {
    /// Cellular Information Architecture errors
    #[error("Cellular Information Architecture error: {message}")]
    CellularInformation { 
        /// Error message
        message: String,
        /// Information advantage metric (should be ≥170,000)
        information_advantage: Option<f64>,
    },

    /// Environmental Gradient Search errors
    #[error("Environmental Gradient Search error: {message}")]
    EnvironmentalGradient {
        /// Error message
        message: String,
        /// Signal emergence strength
        emergence_strength: Option<f64>,
        /// Noise entropy level
        noise_entropy: Option<f64>,
    },

    /// Fuzzy-Bayesian Network errors
    #[error("Fuzzy-Bayesian Network error: {message}")]
    FuzzyBayesian {
        /// Error message
        message: String,
        /// Uncertainty level
        uncertainty_level: Option<f64>,
        /// Confidence interval
        confidence_interval: Option<(f64, f64)>,
    },

    /// Oscillatory Reality Theory errors
    #[error("Oscillatory Reality error: {message}")]
    OscillatoryReality {
        /// Error message
        message: String,
        /// Resonance frequency
        resonance_frequency: Option<f64>,
        /// Coherence level
        coherence_level: Option<f64>,
    },

    /// S-Entropy Navigation errors
    #[error("S-Entropy Navigation error: {message}")]
    SEntropy {
        /// Error message
        message: String,
        /// Current entropy coordinates
        entropy_coordinates: Option<(f64, f64, f64)>,
        /// Compression ratio achieved
        compression_ratio: Option<f64>,
    },

    /// Universal Solvability errors
    #[error("Universal Solvability error: {message}")]
    UniversalSolvability {
        /// Error message
        message: String,
        /// Whether solution exists (should always be true)
        solution_exists: bool,
        /// Predetermination confidence
        predetermination_confidence: Option<f64>,
    },

    /// Stella-Lorraine Clock errors
    #[error("Stella-Lorraine Clock error: {message}")]
    StellaLorraine {
        /// Error message
        message: String,
        /// Current precision level (femtoseconds)
        precision_level: Option<u64>,
        /// Temporal coordinate
        temporal_coordinate: Option<f64>,
    },

    /// Tributary-Stream Dynamics errors
    #[error("Tributary-Stream Dynamics error: {message}")]
    TributaryStreams {
        /// Error message
        message: String,
        /// Flow rate
        flow_rate: Option<f64>,
        /// Convergence status
        convergence_status: Option<String>,
    },

    /// Harare Algorithm errors
    #[error("Harare Algorithm error: {message}")]
    HarareAlgorithm {
        /// Error message
        message: String,
        /// Failure generation rate
        failure_rate: Option<f64>,
        /// Statistical emergence probability
        emergence_probability: Option<f64>,
    },

    /// Honjo Masamune Engine errors
    #[error("Honjo Masamune Engine error: {message}")]
    HonjoMasamune {
        /// Error message
        message: String,
        /// Truth reconstruction confidence
        truth_confidence: Option<f64>,
        /// Evidence decay factor
        evidence_decay: Option<f64>,
    },

    /// Buhera-East LLM Suite errors
    #[error("Buhera-East LLM Suite error: {message}")]
    BuheraEast {
        /// Error message
        message: String,
        /// LLM consensus accuracy
        consensus_accuracy: Option<f64>,
        /// Domain expertise level
        domain_expertise: Option<f64>,
    },

    /// Mufakose Search errors
    #[error("Mufakose Search error: {message}")]
    MufakoseSearch {
        /// Error message
        message: String,
        /// Confirmation probability
        confirmation_probability: Option<f64>,
        /// Search complexity order
        search_complexity: Option<String>,
    },

    /// Integration and orchestration errors
    #[error("Integration error: {message}")]
    Integration {
        /// Error message
        message: String,
        /// Failed framework
        failed_framework: Option<String>,
        /// Integration success rate
        integration_success_rate: Option<f64>,
    },

    /// Performance target violation errors
    #[error("Performance target violation: {message}")]
    PerformanceViolation {
        /// Error message
        message: String,
        /// Target type (accuracy, speed, memory, etc.)
        target_type: String,
        /// Expected value
        expected: f64,
        /// Actual value
        actual: f64,
    },

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Network errors
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Timeout errors
    #[error("Operation timed out after {seconds} seconds")]
    Timeout { 
        /// Timeout duration in seconds
        seconds: u64 
    },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { 
        /// Error message
        message: String 
    },

    /// Invalid input errors
    #[error("Invalid input: {message}")]
    InvalidInput { 
        /// Error message
        message: String 
    },

    /// Insufficient resources errors
    #[error("Insufficient resources: {message}")]
    InsufficientResources { 
        /// Error message
        message: String 
    },

    /// Framework not available
    #[error("Framework '{framework}' is not available or not enabled")]
    FrameworkUnavailable { 
        /// Framework name
        framework: String 
    },

    /// Generic error
    #[error("Gospel framework error: {message}")]
    Generic { 
        /// Error message
        message: String 
    },
}

impl GospelError {
    /// Create a new Cellular Information Architecture error
    pub fn cellular_information(message: impl Into<String>) -> Self {
        Self::CellularInformation {
            message: message.into(),
            information_advantage: None,
        }
    }

    /// Create a new Cellular Information Architecture error with information advantage
    pub fn cellular_information_with_advantage(
        message: impl Into<String>, 
        information_advantage: f64
    ) -> Self {
        Self::CellularInformation {
            message: message.into(),
            information_advantage: Some(information_advantage),
        }
    }

    /// Create a new S-Entropy Navigation error
    pub fn s_entropy(message: impl Into<String>) -> Self {
        Self::SEntropy {
            message: message.into(),
            entropy_coordinates: None,
            compression_ratio: None,
        }
    }

    /// Create a new S-Entropy Navigation error with coordinates
    pub fn s_entropy_with_coordinates(
        message: impl Into<String>,
        coordinates: (f64, f64, f64),
        compression_ratio: Option<f64>,
    ) -> Self {
        Self::SEntropy {
            message: message.into(),
            entropy_coordinates: Some(coordinates),
            compression_ratio,
        }
    }

    /// Create a new Universal Solvability error
    pub fn universal_solvability(message: impl Into<String>) -> Self {
        Self::UniversalSolvability {
            message: message.into(),
            solution_exists: true, // Solutions always exist by thermodynamic necessity
            predetermination_confidence: None,
        }
    }

    /// Create a new performance violation error
    pub fn performance_violation(
        target_type: impl Into<String>,
        expected: f64,
        actual: f64,
    ) -> Self {
        Self::PerformanceViolation {
            message: format!(
                "{} performance target not met: expected {}, got {}",
                target_type.as_ref(),
                expected,
                actual
            ),
            target_type: target_type.into(),
            expected,
            actual,
        }
    }

    /// Create a new framework unavailable error
    pub fn framework_unavailable(framework: impl Into<String>) -> Self {
        Self::FrameworkUnavailable {
            framework: framework.into(),
        }
    }

    /// Create a new timeout error
    pub fn timeout(seconds: u64) -> Self {
        Self::Timeout { seconds }
    }

    /// Create a new configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create a new invalid input error
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Create a new insufficient resources error
    pub fn insufficient_resources(message: impl Into<String>) -> Self {
        Self::InsufficientResources {
            message: message.into(),
        }
    }

    /// Create a new generic error
    pub fn generic(message: impl Into<String>) -> Self {
        Self::Generic {
            message: message.into(),
        }
    }

    /// Check if this is a performance-related error
    pub fn is_performance_error(&self) -> bool {
        matches!(self, Self::PerformanceViolation { .. })
    }

    /// Check if this is a framework availability error
    pub fn is_framework_error(&self) -> bool {
        matches!(self, Self::FrameworkUnavailable { .. })
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Timeout { .. } => true,
            Self::Network(_) => true,
            Self::InsufficientResources { .. } => true,
            Self::FrameworkUnavailable { .. } => false,
            Self::PerformanceViolation { .. } => false,
            _ => true,
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::PerformanceViolation { .. } => ErrorSeverity::Critical,
            Self::UniversalSolvability { solution_exists: false, .. } => ErrorSeverity::Critical,
            Self::FrameworkUnavailable { .. } => ErrorSeverity::High,
            Self::Integration { .. } => ErrorSeverity::High,
            Self::Timeout { .. } => ErrorSeverity::Medium,
            Self::Network(_) => ErrorSeverity::Medium,
            Self::Configuration { .. } => ErrorSeverity::Medium,
            Self::InvalidInput { .. } => ErrorSeverity::Low,
            _ => ErrorSeverity::Medium,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - minor issues that don't affect core functionality
    Low,
    /// Medium severity - issues that may impact performance or accuracy
    Medium,
    /// High severity - issues that significantly impact functionality
    High,
    /// Critical severity - issues that prevent framework operation
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "LOW"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::High => write!(f, "HIGH"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Result type alias for Gospel framework operations
pub type GospelResult<T> = Result<T, GospelError>;

/// Error context for providing additional debugging information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Framework that generated the error
    pub framework: String,
    /// Operation that was being performed
    pub operation: String,
    /// Additional context information
    pub context: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(framework: impl Into<String>, operation: impl Into<String>) -> Self {
        Self {
            framework: framework.into(),
            operation: operation.into(),
            context: std::collections::HashMap::new(),
        }
    }

    /// Add context information
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = GospelError::cellular_information("Test error");
        assert!(error.to_string().contains("Cellular Information Architecture"));
    }

    #[test]
    fn test_error_severity() {
        let performance_error = GospelError::performance_violation("accuracy", 0.97, 0.85);
        assert_eq!(performance_error.severity(), ErrorSeverity::Critical);
        
        let timeout_error = GospelError::timeout(30);
        assert_eq!(timeout_error.severity(), ErrorSeverity::Medium);
    }

    #[test]
    fn test_error_recoverability() {
        let timeout_error = GospelError::timeout(30);
        assert!(timeout_error.is_recoverable());
        
        let framework_error = GospelError::framework_unavailable("test-framework");
        assert!(!framework_error.is_recoverable());
    }

    #[test]
    fn test_s_entropy_error_with_coordinates() {
        let error = GospelError::s_entropy_with_coordinates(
            "Navigation failed",
            (1.0, 2.0, 3.0),
            Some(0.95),
        );
        
        if let GospelError::SEntropy { entropy_coordinates, compression_ratio, .. } = error {
            assert_eq!(entropy_coordinates, Some((1.0, 2.0, 3.0)));
            assert_eq!(compression_ratio, Some(0.95));
        } else {
            panic!("Wrong error type");
        }
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("s-entropy", "navigation")
            .with_context("coordinates", "(1.0, 2.0, 3.0)")
            .with_context("precision", "femtosecond");
        
        assert_eq!(context.framework, "s-entropy");
        assert_eq!(context.operation, "navigation");
        assert_eq!(context.context.len(), 2);
    }
}