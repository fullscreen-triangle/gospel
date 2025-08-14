/*!
# Gospel Framework Core Types

This module defines the fundamental data types used across all 12 revolutionary frameworks
in the Gospel genomic analysis system.
*/

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use nalgebra::{Vector3, Matrix3};

/// Complexity order for algorithm analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplexityOrder {
    /// Constant complexity O(1)
    Constant,
    /// Logarithmic complexity O(log N)
    Logarithmic,
    /// Linear complexity O(N)
    Linear,
    /// Linearithmic complexity O(N log N)
    Linearithmic,
    /// Quadratic complexity O(N²)
    Quadratic,
    /// Exponential complexity O(2^N)
    Exponential,
}

impl std::fmt::Display for ComplexityOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComplexityOrder::Constant => write!(f, "O(1)"),
            ComplexityOrder::Logarithmic => write!(f, "O(log N)"),
            ComplexityOrder::Linear => write!(f, "O(N)"),
            ComplexityOrder::Linearithmic => write!(f, "O(N log N)"),
            ComplexityOrder::Quadratic => write!(f, "O(N²)"),
            ComplexityOrder::Exponential => write!(f, "O(2^N)"),
        }
    }
}

/// Analysis complexity level for determining processing approach
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnalysisComplexityLevel {
    /// Basic analysis using traditional approaches
    Basic,
    /// Enhanced analysis with selective framework integration
    Enhanced,
    /// Advanced analysis with multiple framework coordination
    Advanced,
    /// Revolutionary analysis using all 12 frameworks
    Revolutionary,
}

/// S-Entropy coordinates for tri-dimensional optimization
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EntropyCoordinates {
    /// Knowledge entropy dimension
    pub s_knowledge: f64,
    /// Temporal entropy dimension
    pub s_time: f64,
    /// Information entropy dimension
    pub s_entropy: f64,
}

impl EntropyCoordinates {
    /// Create new entropy coordinates
    pub fn new(s_knowledge: f64, s_time: f64, s_entropy: f64) -> Self {
        Self {
            s_knowledge,
            s_time,
            s_entropy,
        }
    }

    /// Calculate Euclidean distance between coordinates
    pub fn distance(&self, other: &Self) -> f64 {
        let dx = self.s_knowledge - other.s_knowledge;
        let dy = self.s_time - other.s_time;
        let dz = self.s_entropy - other.s_entropy;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Convert to nalgebra Vector3 for mathematical operations
    pub fn to_vector(&self) -> Vector3<f64> {
        Vector3::new(self.s_knowledge, self.s_time, self.s_entropy)
    }

    /// Create from nalgebra Vector3
    pub fn from_vector(v: Vector3<f64>) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    /// Check if coordinates represent optimal state (near zero)
    pub fn is_optimal(&self, threshold: f64) -> bool {
        self.magnitude() < threshold
    }

    /// Calculate magnitude of coordinate vector
    pub fn magnitude(&self) -> f64 {
        (self.s_knowledge * self.s_knowledge + 
         self.s_time * self.s_time + 
         self.s_entropy * self.s_entropy).sqrt()
    }
}

impl Default for EntropyCoordinates {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

/// Temporal coordinates for femtosecond-precision navigation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TemporalCoordinates {
    /// Primary temporal coordinate (femtoseconds)
    pub femtosecond_coordinate: u64,
    /// Secondary precision enhancement
    pub precision_enhancement: f64,
    /// Temporal stability measure
    pub stability: f64,
}

impl TemporalCoordinates {
    /// Create new temporal coordinates
    pub fn new(femtosecond_coordinate: u64, precision_enhancement: f64, stability: f64) -> Self {
        Self {
            femtosecond_coordinate,
            precision_enhancement,
            stability,
        }
    }

    /// Convert to Duration
    pub fn to_duration(&self) -> Duration {
        Duration::from_nanos(self.femtosecond_coordinate / 1_000_000)
    }

    /// Create from Duration
    pub fn from_duration(duration: Duration) -> Self {
        Self::new(
            duration.as_nanos() as u64 * 1_000_000, // Convert to femtoseconds
            1.0,
            1.0,
        )
    }
}

/// Oscillatory pattern for genomic resonance analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OscillatoryPattern {
    /// Frequency of oscillation
    pub frequency: f64,
    /// Amplitude of oscillation
    pub amplitude: f64,
    /// Phase offset
    pub phase: f64,
    /// Coherence level with other patterns
    pub coherence: f64,
    /// Pattern type identifier
    pub pattern_type: String,
}

impl OscillatoryPattern {
    /// Create new oscillatory pattern
    pub fn new(
        frequency: f64,
        amplitude: f64,
        phase: f64,
        coherence: f64,
        pattern_type: String,
    ) -> Self {
        Self {
            frequency,
            amplitude,
            phase,
            coherence,
            pattern_type,
        }
    }

    /// Calculate resonance with another pattern
    pub fn resonance_with(&self, other: &Self) -> f64 {
        let freq_similarity = 1.0 - (self.frequency - other.frequency).abs() / (self.frequency + other.frequency);
        let amplitude_similarity = 1.0 - (self.amplitude - other.amplitude).abs() / (self.amplitude + other.amplitude);
        let coherence_avg = (self.coherence + other.coherence) / 2.0;
        
        freq_similarity * amplitude_similarity * coherence_avg
    }
}

/// Biological hierarchy levels for multi-scale analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiologicalLevel {
    /// Molecular level (DNA, RNA, proteins)
    Molecular,
    /// Cellular level (organelles, cell systems)
    Cellular,
    /// Tissue level (cell populations, tissue architecture)
    Tissue,
    /// Organ level (organ systems)
    Organ,
    /// Organism level (whole organism)
    Organism,
}

impl std::fmt::Display for BiologicalLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BiologicalLevel::Molecular => write!(f, "Molecular"),
            BiologicalLevel::Cellular => write!(f, "Cellular"),
            BiologicalLevel::Tissue => write!(f, "Tissue"),
            BiologicalLevel::Organ => write!(f, "Organ"),
            BiologicalLevel::Organism => write!(f, "Organism"),
        }
    }
}

/// Genomic data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomicData {
    /// Chromosome identifier
    pub chromosome: String,
    /// Start position
    pub start_position: u64,
    /// End position
    pub end_position: u64,
    /// Reference sequence
    pub reference_sequence: String,
    /// Alternative sequences
    pub alternative_sequences: Vec<String>,
    /// Quality scores
    pub quality_scores: Vec<f64>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Expression data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionData {
    /// Gene identifier
    pub gene_id: String,
    /// Expression level
    pub expression_level: f64,
    /// Tissue type
    pub tissue_type: Option<String>,
    /// Cell type
    pub cell_type: Option<String>,
    /// Experimental condition
    pub condition: Option<String>,
    /// Statistical confidence
    pub confidence: f64,
}

/// Pathway network structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwayNetwork {
    /// Network identifier
    pub id: String,
    /// Network name
    pub name: String,
    /// Nodes (genes/proteins)
    pub nodes: Vec<PathwayNode>,
    /// Edges (interactions)
    pub edges: Vec<PathwayEdge>,
    /// Network-level metrics
    pub metrics: HashMap<String, f64>,
}

/// Pathway node (gene or protein)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwayNode {
    /// Node identifier
    pub id: String,
    /// Node name
    pub name: String,
    /// Node type (gene, protein, metabolite, etc.)
    pub node_type: String,
    /// Functional annotations
    pub annotations: Vec<String>,
    /// Expression level (if applicable)
    pub expression_level: Option<f64>,
}

/// Pathway edge (interaction)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwayEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Interaction type
    pub interaction_type: String,
    /// Interaction strength
    pub strength: f64,
    /// Direction (if applicable)
    pub direction: Option<String>,
}

/// Comprehensive genomic analysis input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomicAnalysisInput {
    /// Primary genomic data
    pub genomic_data: GenomicData,
    /// Expression data
    pub expression_data: Vec<ExpressionData>,
    /// Pathway network
    pub pathway_network: Option<PathwayNetwork>,
    /// Analysis complexity level
    pub complexity_level: AnalysisComplexityLevel,
    /// Research objective
    pub research_objective: String,
    /// Computational budget (in seconds)
    pub computational_budget: Option<u64>,
    /// Confidence threshold
    pub confidence_threshold: f64,
    /// Additional parameters
    pub parameters: HashMap<String, String>,
}

impl GenomicAnalysisInput {
    /// Create a new builder for genomic analysis input
    pub fn builder() -> GenomicAnalysisInputBuilder {
        GenomicAnalysisInputBuilder::default()
    }
}

/// Builder for GenomicAnalysisInput
#[derive(Debug, Default)]
pub struct GenomicAnalysisInputBuilder {
    genomic_data: Option<GenomicData>,
    expression_data: Vec<ExpressionData>,
    pathway_network: Option<PathwayNetwork>,
    complexity_level: AnalysisComplexityLevel,
    research_objective: String,
    computational_budget: Option<u64>,
    confidence_threshold: f64,
    parameters: HashMap<String, String>,
}

impl GenomicAnalysisInputBuilder {
    /// Set genomic data
    pub fn with_genomic_data(mut self, genomic_data: GenomicData) -> Self {
        self.genomic_data = Some(genomic_data);
        self
    }

    /// Add expression data
    pub fn with_expression_data(mut self, expression_data: Vec<ExpressionData>) -> Self {
        self.expression_data = expression_data;
        self
    }

    /// Set pathway network
    pub fn with_pathway_network(mut self, pathway_network: PathwayNetwork) -> Self {
        self.pathway_network = Some(pathway_network);
        self
    }

    /// Set complexity level
    pub fn with_complexity_level(mut self, complexity_level: AnalysisComplexityLevel) -> Self {
        self.complexity_level = complexity_level;
        self
    }

    /// Set research objective
    pub fn with_research_objective(mut self, objective: impl Into<String>) -> Self {
        self.research_objective = objective.into();
        self
    }

    /// Set computational budget
    pub fn with_computational_budget(mut self, budget_seconds: u64) -> Self {
        self.computational_budget = Some(budget_seconds);
        self
    }

    /// Set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Add parameter
    pub fn with_parameter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }

    /// Build the GenomicAnalysisInput
    pub fn build(self) -> Result<GenomicAnalysisInput, String> {
        let genomic_data = self.genomic_data
            .ok_or("Genomic data is required")?;

        Ok(GenomicAnalysisInput {
            genomic_data,
            expression_data: self.expression_data,
            pathway_network: self.pathway_network,
            complexity_level: self.complexity_level,
            research_objective: self.research_objective,
            computational_budget: self.computational_budget,
            confidence_threshold: self.confidence_threshold,
            parameters: self.parameters,
        })
    }
}

impl Default for GenomicAnalysisInputBuilder {
    fn default() -> Self {
        Self {
            genomic_data: None,
            expression_data: Vec::new(),
            pathway_network: None,
            complexity_level: AnalysisComplexityLevel::Revolutionary,
            research_objective: "Comprehensive genomic analysis".to_string(),
            computational_budget: None,
            confidence_threshold: 0.97, // 97%+ accuracy target
            parameters: HashMap::new(),
        }
    }
}

/// Comprehensive genomic analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveGenomicAnalysis {
    /// Overall analysis accuracy
    pub accuracy: f64,
    /// Processing time
    pub processing_time: Duration,
    /// Memory complexity achieved
    pub memory_complexity: ComplexityOrder,
    /// Computational complexity achieved
    pub computational_complexity: ComplexityOrder,
    /// Truth reconstruction confidence
    pub truth_confidence: f64,
    /// S-entropy coordinates of final state
    pub final_entropy_coordinates: EntropyCoordinates,
    /// Temporal coordinates
    pub temporal_coordinates: TemporalCoordinates,
    /// Framework-specific results
    pub framework_results: HashMap<String, serde_json::Value>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Validation results
    pub validation_results: ValidationResults,
}

/// Performance metrics for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total processing time
    pub processing_time: Duration,
    /// Memory usage (bytes)
    pub memory_usage: u64,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Framework efficiency scores
    pub framework_efficiency: HashMap<String, f64>,
    /// Scalability factor achieved
    pub scalability_factor: f64,
}

/// Validation results for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    /// Accuracy validation
    pub accuracy_validated: bool,
    /// Performance validation
    pub performance_validated: bool,
    /// Consistency validation
    pub consistency_validated: bool,
    /// Biological plausibility validation
    pub biological_plausibility_validated: bool,
    /// Overall validation score
    pub overall_validation_score: f64,
}

/// Precision level for temporal operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionLevel {
    /// Nanosecond precision
    Nanosecond,
    /// Picosecond precision
    Picosecond,
    /// Femtosecond precision
    Femtosecond,
    /// Attosecond precision (theoretical)
    Attosecond,
}

impl PrecisionLevel {
    /// Get precision in femtoseconds
    pub fn in_femtoseconds(&self) -> u64 {
        match self {
            PrecisionLevel::Nanosecond => 1_000_000,
            PrecisionLevel::Picosecond => 1_000,
            PrecisionLevel::Femtosecond => 1,
            PrecisionLevel::Attosecond => 0, // Theoretical limit
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_coordinates() {
        let coords1 = EntropyCoordinates::new(1.0, 2.0, 3.0);
        let coords2 = EntropyCoordinates::new(4.0, 5.0, 6.0);
        
        let distance = coords1.distance(&coords2);
        assert!((distance - 5.196152422706632).abs() < 1e-10);
        
        assert_eq!(coords1.magnitude(), (14.0_f64).sqrt());
    }

    #[test]
    fn test_oscillatory_pattern_resonance() {
        let pattern1 = OscillatoryPattern::new(1.0, 0.5, 0.0, 0.8, "test1".to_string());
        let pattern2 = OscillatoryPattern::new(1.1, 0.6, 0.0, 0.9, "test2".to_string());
        
        let resonance = pattern1.resonance_with(&pattern2);
        assert!(resonance > 0.0 && resonance <= 1.0);
    }

    #[test]
    fn test_genomic_analysis_input_builder() {
        let genomic_data = GenomicData {
            chromosome: "chr1".to_string(),
            start_position: 1000,
            end_position: 2000,
            reference_sequence: "ATCG".to_string(),
            alternative_sequences: vec!["ATCG".to_string()],
            quality_scores: vec![0.99],
            metadata: HashMap::new(),
        };

        let input = GenomicAnalysisInput::builder()
            .with_genomic_data(genomic_data)
            .with_complexity_level(AnalysisComplexityLevel::Revolutionary)
            .with_confidence_threshold(0.97)
            .build()
            .expect("Should build successfully");

        assert_eq!(input.complexity_level, AnalysisComplexityLevel::Revolutionary);
        assert_eq!(input.confidence_threshold, 0.97);
    }

    #[test]
    fn test_temporal_coordinates() {
        let coords = TemporalCoordinates::new(1_000_000, 1.0, 1.0);
        let duration = coords.to_duration();
        
        assert_eq!(duration.as_nanos(), 1000);
    }

    #[test]
    fn test_precision_levels() {
        assert_eq!(PrecisionLevel::Femtosecond.in_femtoseconds(), 1);
        assert_eq!(PrecisionLevel::Picosecond.in_femtoseconds(), 1000);
        assert_eq!(PrecisionLevel::Nanosecond.in_femtoseconds(), 1_000_000);
    }
}