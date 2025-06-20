//! Fuzzy logic processing for genomic uncertainty quantification
//!
//! This module implements high-performance fuzzy logic operations for handling
//! uncertainty in genomic variant analysis using trapezoidal, Gaussian, and
//! sigmoid membership functions.

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{GospelConfig, Variant};

/// Fuzzy membership function types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MembershipFunction {
    /// Trapezoidal function: f(x) = max(0, min((x-a)/(b-a), 1, (d-x)/(d-c)))
    Trapezoidal { a: f64, b: f64, c: f64, d: f64 },
    /// Gaussian function: f(x) = exp(-((x-μ)²)/(2σ²))
    Gaussian { mu: f64, sigma: f64 },
    /// Sigmoid function: f(x) = 1/(1 + exp(-k(x-x0)))
    Sigmoid { k: f64, x0: f64 },
    /// Exponential decay: f(x) = exp(-λx)
    Exponential { lambda: f64 },
}

impl MembershipFunction {
    /// Calculate membership degree for given input
    pub fn membership(&self, x: f64) -> f64 {
        match self {
            MembershipFunction::Trapezoidal { a, b, c, d } => {
                if x <= *a || x >= *d {
                    0.0
                } else if x >= *b && x <= *c {
                    1.0
                } else if x > *a && x < *b {
                    (x - a) / (b - a)
                } else {
                    (d - x) / (d - c)
                }
            }
            MembershipFunction::Gaussian { mu, sigma } => {
                let diff = x - mu;
                (-diff * diff / (2.0 * sigma * sigma)).exp()
            }
            MembershipFunction::Sigmoid { k, x0 } => {
                1.0 / (1.0 + (-k * (x - x0)).exp())
            }
            MembershipFunction::Exponential { lambda } => {
                if x >= 0.0 {
                    (-lambda * x).exp()
                } else {
                    0.0
                }
            }
        }
    }
}

/// Fuzzy set with linguistic label and membership function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzySet {
    /// Linguistic label (e.g., "high", "medium", "low")
    pub label: String,
    /// Membership function
    pub function: MembershipFunction,
    /// Weight for aggregation
    pub weight: f64,
}

impl FuzzySet {
    /// Create new fuzzy set
    pub fn new(label: String, function: MembershipFunction, weight: f64) -> Self {
        Self {
            label,
            function,
            weight,
        }
    }

    /// Calculate weighted membership
    pub fn weighted_membership(&self, x: f64) -> f64 {
        self.function.membership(x) * self.weight
    }
}

/// Fuzzy rule for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyRule {
    /// Rule antecedents (conditions)
    pub antecedents: Vec<(String, String)>, // (variable, fuzzy_set)
    /// Rule consequent (conclusion)
    pub consequent: (String, String), // (variable, fuzzy_set)
    /// Rule weight/importance
    pub weight: f64,
}

impl FuzzyRule {
    /// Create new fuzzy rule
    pub fn new(
        antecedents: Vec<(String, String)>,
        consequent: (String, String),
        weight: f64,
    ) -> Self {
        Self {
            antecedents,
            consequent,
            weight,
        }
    }
}

/// Result of fuzzy analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyResult {
    /// Input variant
    pub variant_id: String,
    /// Fuzzy membership degrees for each linguistic variable
    pub memberships: HashMap<String, HashMap<String, f64>>,
    /// Overall pathogenicity confidence
    pub pathogenicity_confidence: f64,
    /// Conservation confidence
    pub conservation_confidence: f64,
    /// Frequency confidence
    pub frequency_confidence: f64,
    /// Expression confidence
    pub expression_confidence: f64,
    /// Aggregated uncertainty score
    pub uncertainty_score: f64,
}

/// High-performance fuzzy processor
#[derive(Debug)]
pub struct FuzzyProcessor {
    config: GospelConfig,
    /// Fuzzy sets for pathogenicity
    pathogenicity_sets: Vec<FuzzySet>,
    /// Fuzzy sets for conservation
    conservation_sets: Vec<FuzzySet>,
    /// Fuzzy sets for frequency
    frequency_sets: Vec<FuzzySet>,
    /// Fuzzy sets for expression
    expression_sets: Vec<FuzzySet>,
    /// Fuzzy rules for inference
    rules: Vec<FuzzyRule>,
}

impl FuzzyProcessor {
    /// Create new fuzzy processor with default genomic fuzzy sets
    pub fn new(config: &GospelConfig) -> Result<Self> {
        let mut processor = Self {
            config: config.clone(),
            pathogenicity_sets: Vec::new(),
            conservation_sets: Vec::new(),
            frequency_sets: Vec::new(),
            expression_sets: Vec::new(),
            rules: Vec::new(),
        };

        processor.initialize_default_sets();
        processor.initialize_default_rules();

        Ok(processor)
    }

    /// Initialize default fuzzy sets for genomic analysis
    fn initialize_default_sets(&mut self) {
        // Pathogenicity fuzzy sets (based on CADD scores 0-40)
        self.pathogenicity_sets = vec![
            FuzzySet::new(
                "benign".to_string(),
                MembershipFunction::Trapezoidal { a: 0.0, b: 0.0, c: 10.0, d: 15.0 },
                1.0,
            ),
            FuzzySet::new(
                "likely_benign".to_string(),
                MembershipFunction::Trapezoidal { a: 10.0, b: 15.0, c: 18.0, d: 22.0 },
                1.0,
            ),
            FuzzySet::new(
                "uncertain".to_string(),
                MembershipFunction::Trapezoidal { a: 18.0, b: 22.0, c: 25.0, d: 28.0 },
                1.0,
            ),
            FuzzySet::new(
                "likely_pathogenic".to_string(),
                MembershipFunction::Trapezoidal { a: 25.0, b: 28.0, c: 32.0, d: 35.0 },
                1.0,
            ),
            FuzzySet::new(
                "pathogenic".to_string(),
                MembershipFunction::Trapezoidal { a: 32.0, b: 35.0, c: 40.0, d: 40.0 },
                1.0,
            ),
        ];

        // Conservation fuzzy sets (0-1 range)
        self.conservation_sets = vec![
            FuzzySet::new(
                "low_conservation".to_string(),
                MembershipFunction::Trapezoidal { a: 0.0, b: 0.0, c: 0.3, d: 0.5 },
                1.0,
            ),
            FuzzySet::new(
                "moderate_conservation".to_string(),
                MembershipFunction::Trapezoidal { a: 0.3, b: 0.5, c: 0.7, d: 0.8 },
                1.0,
            ),
            FuzzySet::new(
                "high_conservation".to_string(),
                MembershipFunction::Trapezoidal { a: 0.7, b: 0.8, c: 1.0, d: 1.0 },
                1.0,
            ),
        ];

        // Frequency fuzzy sets (0-1 range, but typically 0-0.1)
        self.frequency_sets = vec![
            FuzzySet::new(
                "rare".to_string(),
                MembershipFunction::Exponential { lambda: 100.0 }, // Rapid decay
                1.0,
            ),
            FuzzySet::new(
                "uncommon".to_string(),
                MembershipFunction::Trapezoidal { a: 0.001, b: 0.005, c: 0.01, d: 0.05 },
                1.0,
            ),
            FuzzySet::new(
                "common".to_string(),
                MembershipFunction::Sigmoid { k: 50.0, x0: 0.05 },
                1.0,
            ),
        ];

        // Expression fuzzy sets (log2 fold change, typically -5 to +5)
        self.expression_sets = vec![
            FuzzySet::new(
                "strongly_downregulated".to_string(),
                MembershipFunction::Trapezoidal { a: -10.0, b: -5.0, c: -2.0, d: -1.0 },
                1.0,
            ),
            FuzzySet::new(
                "downregulated".to_string(),
                MembershipFunction::Trapezoidal { a: -2.0, b: -1.0, c: -0.5, d: 0.0 },
                1.0,
            ),
            FuzzySet::new(
                "unchanged".to_string(),
                MembershipFunction::Gaussian { mu: 0.0, sigma: 0.5 },
                1.0,
            ),
            FuzzySet::new(
                "upregulated".to_string(),
                MembershipFunction::Trapezoidal { a: 0.0, b: 0.5, c: 1.0, d: 2.0 },
                1.0,
            ),
            FuzzySet::new(
                "strongly_upregulated".to_string(),
                MembershipFunction::Trapezoidal { a: 1.0, b: 2.0, c: 5.0, d: 10.0 },
                1.0,
            ),
        ];
    }

    /// Initialize default fuzzy rules for genomic inference
    fn initialize_default_rules(&mut self) {
        self.rules = vec![
            // High conservation + rare frequency → likely pathogenic
            FuzzyRule::new(
                vec![
                    ("conservation".to_string(), "high_conservation".to_string()),
                    ("frequency".to_string(), "rare".to_string()),
                ],
                ("pathogenicity".to_string(), "likely_pathogenic".to_string()),
                0.9,
            ),
            // Low conservation + common frequency → likely benign
            FuzzyRule::new(
                vec![
                    ("conservation".to_string(), "low_conservation".to_string()),
                    ("frequency".to_string(), "common".to_string()),
                ],
                ("pathogenicity".to_string(), "likely_benign".to_string()),
                0.8,
            ),
            // Strong expression change + high conservation → pathogenic
            FuzzyRule::new(
                vec![
                    ("expression".to_string(), "strongly_upregulated".to_string()),
                    ("conservation".to_string(), "high_conservation".to_string()),
                ],
                ("pathogenicity".to_string(), "pathogenic".to_string()),
                0.9,
            ),
            FuzzyRule::new(
                vec![
                    ("expression".to_string(), "strongly_downregulated".to_string()),
                    ("conservation".to_string(), "high_conservation".to_string()),
                ],
                ("pathogenicity".to_string(), "pathogenic".to_string()),
                0.9,
            ),
        ];
    }

    /// Analyze variants using fuzzy logic
    pub async fn analyze_variants(&self, variants: &[Variant]) -> Result<Vec<FuzzyResult>> {
        let results: Vec<FuzzyResult> = variants
            .par_iter()
            .map(|variant| self.analyze_single_variant(variant))
            .collect::<Result<Vec<_>>>()?;

        Ok(results)
    }

    /// Analyze single variant with fuzzy logic
    fn analyze_single_variant(&self, variant: &Variant) -> Result<FuzzyResult> {
        let variant_id = format!("{}:{}", variant.chromosome, variant.position);
        let mut memberships = HashMap::new();

        // Calculate pathogenicity memberships
        let cadd_score = variant.cadd_score.unwrap_or(10.0);
        let pathogenicity_memberships = self.calculate_memberships(&self.pathogenicity_sets, cadd_score);
        memberships.insert("pathogenicity".to_string(), pathogenicity_memberships);

        // Calculate conservation memberships
        let conservation_score = variant.conservation_score.unwrap_or(0.5);
        let conservation_memberships = self.calculate_memberships(&self.conservation_sets, conservation_score);
        memberships.insert("conservation".to_string(), conservation_memberships);

        // Calculate frequency memberships
        let frequency_score = variant.allele_frequency.unwrap_or(0.01);
        let frequency_memberships = self.calculate_memberships(&self.frequency_sets, frequency_score);
        memberships.insert("frequency".to_string(), frequency_memberships);

        // Expression memberships (placeholder - would need expression data)
        let expression_score = 0.0; // Default to unchanged
        let expression_memberships = self.calculate_memberships(&self.expression_sets, expression_score);
        memberships.insert("expression".to_string(), expression_memberships);

        // Apply fuzzy inference rules
        let pathogenicity_confidence = self.apply_fuzzy_inference(&memberships);

        // Calculate individual confidences
        let conservation_confidence = self.aggregate_memberships(&conservation_memberships);
        let frequency_confidence = self.aggregate_memberships(&frequency_memberships);
        let expression_confidence = self.aggregate_memberships(&expression_memberships);

        // Calculate overall uncertainty
        let uncertainty_score = self.calculate_uncertainty(&memberships);

        Ok(FuzzyResult {
            variant_id,
            memberships,
            pathogenicity_confidence,
            conservation_confidence,
            frequency_confidence,
            expression_confidence,
            uncertainty_score,
        })
    }

    /// Calculate membership degrees for all sets
    fn calculate_memberships(&self, sets: &[FuzzySet], value: f64) -> HashMap<String, f64> {
        sets.iter()
            .map(|set| (set.label.clone(), set.function.membership(value)))
            .collect()
    }

    /// Apply fuzzy inference rules using Mamdani method
    fn apply_fuzzy_inference(&self, memberships: &HashMap<String, HashMap<String, f64>>) -> f64 {
        let mut rule_outputs = Vec::new();

        for rule in &self.rules {
            // Calculate rule activation (minimum of antecedents)
            let mut activation = 1.0;
            
            for (variable, fuzzy_set) in &rule.antecedents {
                if let Some(variable_memberships) = memberships.get(variable) {
                    if let Some(&membership) = variable_memberships.get(fuzzy_set) {
                        activation = activation.min(membership);
                    }
                }
            }

            // Weight the activation
            let weighted_activation = activation * rule.weight;
            rule_outputs.push(weighted_activation);
        }

        // Aggregate rule outputs (maximum)
        rule_outputs.into_iter().fold(0.0, f64::max)
    }

    /// Aggregate membership degrees using centroid defuzzification
    fn aggregate_memberships(&self, memberships: &HashMap<String, f64>) -> f64 {
        let total_membership: f64 = memberships.values().sum();
        let weighted_sum: f64 = memberships
            .iter()
            .enumerate()
            .map(|(i, (_, &membership))| (i as f64 + 1.0) * membership)
            .sum();

        if total_membership > 0.0 {
            weighted_sum / total_membership / memberships.len() as f64
        } else {
            0.5 // Default moderate confidence
        }
    }

    /// Calculate uncertainty based on membership distribution
    fn calculate_uncertainty(&self, memberships: &HashMap<String, HashMap<String, f64>>) -> f64 {
        let mut total_entropy = 0.0;
        let mut count = 0;

        for (_, variable_memberships) in memberships {
            let entropy = self.calculate_entropy(variable_memberships);
            total_entropy += entropy;
            count += 1;
        }

        if count > 0 {
            total_entropy / count as f64
        } else {
            1.0 // Maximum uncertainty
        }
    }

    /// Calculate Shannon entropy for uncertainty quantification
    fn calculate_entropy(&self, memberships: &HashMap<String, f64>) -> f64 {
        let total: f64 = memberships.values().sum();
        
        if total <= 0.0 {
            return 1.0; // Maximum entropy
        }

        let mut entropy = 0.0;
        for &membership in memberships.values() {
            if membership > 0.0 {
                let prob = membership / total;
                entropy -= prob * prob.log2();
            }
        }

        // Normalize by maximum possible entropy
        let max_entropy = (memberships.len() as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Add custom fuzzy set
    pub fn add_pathogenicity_set(&mut self, set: FuzzySet) {
        self.pathogenicity_sets.push(set);
    }

    /// Add custom fuzzy rule
    pub fn add_rule(&mut self, rule: FuzzyRule) {
        self.rules.push(rule);
    }

    /// Get processor statistics
    pub fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("pathogenicity_sets".to_string(), self.pathogenicity_sets.len());
        stats.insert("conservation_sets".to_string(), self.conservation_sets.len());
        stats.insert("frequency_sets".to_string(), self.frequency_sets.len());
        stats.insert("expression_sets".to_string(), self.expression_sets.len());
        stats.insert("rules".to_string(), self.rules.len());
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GospelConfig;

    #[test]
    fn test_membership_functions() {
        // Test trapezoidal function
        let trap = MembershipFunction::Trapezoidal { a: 0.0, b: 2.0, c: 6.0, d: 8.0 };
        assert_eq!(trap.membership(-1.0), 0.0);
        assert_eq!(trap.membership(1.0), 0.5);
        assert_eq!(trap.membership(4.0), 1.0);
        assert_eq!(trap.membership(7.0), 0.5);
        assert_eq!(trap.membership(9.0), 0.0);

        // Test Gaussian function
        let gauss = MembershipFunction::Gaussian { mu: 0.0, sigma: 1.0 };
        assert!((gauss.membership(0.0) - 1.0).abs() < 1e-10);
        assert!(gauss.membership(1.0) > 0.3);
        assert!(gauss.membership(-1.0) > 0.3);

        // Test sigmoid function
        let sigmoid = MembershipFunction::Sigmoid { k: 1.0, x0: 0.0 };
        assert!((sigmoid.membership(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid.membership(5.0) > 0.9);
        assert!(sigmoid.membership(-5.0) < 0.1);
    }

    #[test]
    fn test_fuzzy_set() {
        let set = FuzzySet::new(
            "high".to_string(),
            MembershipFunction::Gaussian { mu: 1.0, sigma: 0.5 },
            0.8,
        );

        let membership = set.weighted_membership(1.0);
        assert!((membership - 0.8).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_fuzzy_processor() {
        let config = GospelConfig::default();
        let processor = FuzzyProcessor::new(&config).unwrap();

        let mut variant = Variant::new(
            "chr1".to_string(),
            12345,
            "A".to_string(),
            "T".to_string(),
            30.0,
        );
        variant.cadd_score = Some(25.0); // Likely pathogenic
        variant.conservation_score = Some(0.9); // High conservation
        variant.allele_frequency = Some(0.001); // Rare

        let results = processor.analyze_variants(&[variant]).await.unwrap();
        assert_eq!(results.len(), 1);

        let result = &results[0];
        assert!(result.pathogenicity_confidence > 0.5);
        assert!(result.conservation_confidence > 0.7);
        assert!(result.uncertainty_score < 0.5); // Low uncertainty
    }

    #[test]
    fn test_entropy_calculation() {
        let config = GospelConfig::default();
        let processor = FuzzyProcessor::new(&config).unwrap();

        // Test uniform distribution (maximum entropy)
        let uniform_memberships: HashMap<String, f64> = [
            ("a".to_string(), 0.25),
            ("b".to_string(), 0.25),
            ("c".to_string(), 0.25),
            ("d".to_string(), 0.25),
        ].iter().cloned().collect();
        
        let entropy = processor.calculate_entropy(&uniform_memberships);
        assert!(entropy > 0.9); // Should be close to 1.0

        // Test concentrated distribution (minimum entropy)
        let concentrated_memberships: HashMap<String, f64> = [
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.0),
            ("c".to_string(), 0.0),
            ("d".to_string(), 0.0),
        ].iter().cloned().collect();
        
        let entropy = processor.calculate_entropy(&concentrated_memberships);
        assert!(entropy < 0.1); // Should be close to 0.0
    }
} 