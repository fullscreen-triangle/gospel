//! Turbulance DSL Parser and Compiler
//! 
//! This module implements a parser and compiler for Turbulance scripts,
//! enabling Gospel to process scientific hypothesis specifications and
//! compile them into executable analysis plans.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TurbulanceError {
    #[error("Syntax error at line {line}, column {column}: {message}")]
    SyntaxError {
        line: usize,
        column: usize,
        message: String,
    },
    #[error("Semantic error: {message}")]
    SemanticError { message: String },
    #[error("Hypothesis validation failed: {reason}")]
    HypothesisValidationError { reason: String },
    #[error("Type error: expected {expected}, found {found}")]
    TypeError { expected: String, found: String },
    #[error("Undefined identifier: {identifier}")]
    UndefinedIdentifier { identifier: String },
    #[error("Scientific reasoning flaw: {flaw}")]
    ScientificReasoningFlaw { flaw: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TurbulanceValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<TurbulanceValue>),
    Object(HashMap<String, TurbulanceValue>),
    Dataset(String), // Path or identifier
    Null,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticValidation {
    pub biological_understanding: String,
    pub temporal_understanding: String,
    pub clinical_understanding: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    pub name: String,
    pub claim: String,
    pub semantic_validation: Option<SemanticValidation>,
    pub requires: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposition {
    pub name: String,
    pub motions: Vec<Motion>,
    pub within_clauses: Vec<WithinClause>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Motion {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithinClause {
    pub context: String,
    pub conditions: Vec<Condition>,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub expression: String,
    pub operator: String,
    pub value: TurbulanceValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    Support { motion: String, confidence: f64 },
    Apply { function: String, args: Vec<TurbulanceValue> },
    Print { message: String },
    Delegate { tool: String, task: String, data: TurbulanceValue },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: String,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub param_type: String,
    pub default_value: Option<TurbulanceValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Statement {
    ItemDeclaration { name: String, value: TurbulanceValue },
    FunctionCall { name: String, args: Vec<TurbulanceValue> },
    ToolDelegation { tool: String, task: String, data: TurbulanceValue },
    ConditionalStatement { condition: String, if_true: Vec<Statement>, if_false: Option<Vec<Statement>> },
    PrintStatement { message: String },
    ReturnStatement { value: Option<TurbulanceValue> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurbulanceAST {
    pub imports: Vec<String>,
    pub hypotheses: Vec<Hypothesis>,
    pub functions: Vec<Function>,
    pub propositions: Vec<Proposition>,
    pub main_function: Option<Function>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub hypothesis_validations: Vec<HypothesisValidation>,
    pub tool_delegations: Vec<ToolDelegation>,
    pub execution_order: Vec<ExecutionStep>,
    pub semantic_requirements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisValidation {
    pub hypothesis: String,
    pub is_scientifically_valid: bool,
    pub validation_reason: String,
    pub required_evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDelegation {
    pub tool: String,
    pub task: String,
    pub data: TurbulanceValue,
    pub expected_output: String,
    pub confidence_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStep {
    LoadData { source: String, destination: String },
    ProcessData { function: String, inputs: Vec<String>, output: String },
    ValidateHypothesis { hypothesis: String },
    DelegateToTool { delegation: ToolDelegation },
    EvaluateProposition { proposition: String },
}

pub struct TurbulanceCompiler {
    scientific_knowledge_base: HashMap<String, Vec<String>>,
    validation_rules: Vec<ValidationRule>,
}

struct ValidationRule {
    name: String,
    check: fn(&TurbulanceAST) -> Result<(), TurbulanceError>,
}

impl TurbulanceCompiler {
    pub fn new() -> Self {
        let mut compiler = Self {
            scientific_knowledge_base: HashMap::new(),
            validation_rules: Vec::new(),
        };
        
        compiler.initialize_scientific_knowledge();
        compiler.initialize_validation_rules();
        compiler
    }

    fn initialize_scientific_knowledge(&mut self) {
        // Add scientific domain knowledge for validation
        self.scientific_knowledge_base.insert(
            "genomics".to_string(),
            vec![
                "pathogenicity_prediction".to_string(),
                "variant_annotation".to_string(),
                "gene_expression_analysis".to_string(),
                "pathway_analysis".to_string(),
            ],
        );
        
        self.scientific_knowledge_base.insert(
            "statistics".to_string(),
            vec![
                "hypothesis_testing".to_string(),
                "confidence_intervals".to_string(),
                "bayesian_inference".to_string(),
                "multiple_testing_correction".to_string(),
            ],
        );
    }

    fn initialize_validation_rules(&mut self) {
        self.validation_rules.push(ValidationRule {
            name: "hypothesis_scientific_validity".to_string(),
            check: Self::validate_hypothesis_scientific_validity,
        });
        
        self.validation_rules.push(ValidationRule {
            name: "semantic_coherence".to_string(),
            check: Self::validate_semantic_coherence,
        });
        
        self.validation_rules.push(ValidationRule {
            name: "statistical_soundness".to_string(),
            check: Self::validate_statistical_soundness,
        });
    }

    pub fn parse(&self, source: &str) -> Result<TurbulanceAST, TurbulanceError> {
        let mut lexer = TurbulanceLexer::new(source);
        let tokens = lexer.tokenize()?;
        
        let mut parser = TurbulanceParser::new(tokens);
        parser.parse()
    }

    pub fn validate(&self, ast: &TurbulanceAST) -> Result<(), Vec<TurbulanceError>> {
        let mut errors = Vec::new();

        // Run all validation rules
        for rule in &self.validation_rules {
            if let Err(error) = (rule.check)(ast) {
                errors.push(error);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn compile(&self, ast: TurbulanceAST) -> Result<ExecutionPlan, TurbulanceError> {
        // Validate AST first
        if let Err(errors) = self.validate(&ast) {
            return Err(errors.into_iter().next().unwrap());
        }

        let mut plan = ExecutionPlan {
            hypothesis_validations: Vec::new(),
            tool_delegations: Vec::new(),
            execution_order: Vec::new(),
            semantic_requirements: Vec::new(),
        };

        // Compile hypotheses into validations
        for hypothesis in &ast.hypotheses {
            let validation = self.compile_hypothesis_validation(hypothesis)?;
            plan.hypothesis_validations.push(validation);
        }

        // Compile functions into execution steps
        for function in &ast.functions {
            let steps = self.compile_function_to_steps(function)?;
            plan.execution_order.extend(steps);
        }

        // Extract tool delegations
        self.extract_tool_delegations(&ast, &mut plan)?;

        // Extract semantic requirements
        self.extract_semantic_requirements(&ast, &mut plan);

        Ok(plan)
    }

    fn compile_hypothesis_validation(&self, hypothesis: &Hypothesis) -> Result<HypothesisValidation, TurbulanceError> {
        // Analyze hypothesis claim for scientific validity
        let is_valid = self.analyze_hypothesis_validity(&hypothesis.claim)?;
        
        let validation_reason = if is_valid {
            "Hypothesis follows scientific methodology and is testable".to_string()
        } else {
            "Hypothesis lacks testable predictions or violates scientific principles".to_string()
        };

        let required_evidence = self.determine_required_evidence(hypothesis)?;

        Ok(HypothesisValidation {
            hypothesis: hypothesis.name.clone(),
            is_scientifically_valid: is_valid,
            validation_reason,
            required_evidence,
        })
    }

    fn analyze_hypothesis_validity(&self, claim: &str) -> Result<bool, TurbulanceError> {
        // Check for testable predictions
        let has_testable_prediction = claim.contains("predict") || 
                                    claim.contains("correlate") || 
                                    claim.contains("associate") ||
                                    claim.contains("cause");

        // Check for scientific terminology
        let has_scientific_terms = self.scientific_knowledge_base
            .values()
            .flatten()
            .any(|term| claim.to_lowercase().contains(&term.to_lowercase()));

        // Check for quantifiable outcomes
        let has_quantifiable_outcome = claim.contains("accuracy") ||
                                     claim.contains("sensitivity") ||
                                     claim.contains("specificity") ||
                                     claim.contains("correlation") ||
                                     claim.contains("percentage");

        if !has_testable_prediction {
            return Err(TurbulanceError::HypothesisValidationError {
                reason: "Hypothesis must contain testable predictions".to_string(),
            });
        }

        if !has_scientific_terms {
            return Err(TurbulanceError::HypothesisValidationError {
                reason: "Hypothesis must use recognized scientific terminology".to_string(),
            });
        }

        Ok(has_testable_prediction && has_scientific_terms && has_quantifiable_outcome)
    }

    fn determine_required_evidence(&self, hypothesis: &Hypothesis) -> Result<Vec<String>, TurbulanceError> {
        let mut evidence = Vec::new();

        // Analyze hypothesis claim to determine required evidence
        if hypothesis.claim.to_lowercase().contains("genomic") {
            evidence.push("genomic_variant_data".to_string());
            evidence.push("population_frequency_data".to_string());
        }

        if hypothesis.claim.to_lowercase().contains("expression") {
            evidence.push("gene_expression_data".to_string());
        }

        if hypothesis.claim.to_lowercase().contains("pathway") {
            evidence.push("pathway_annotation_data".to_string());
        }

        if hypothesis.claim.to_lowercase().contains("clinical") {
            evidence.push("clinical_outcome_data".to_string());
        }

        // Always require statistical validation
        evidence.push("statistical_validation".to_string());

        Ok(evidence)
    }

    fn compile_function_to_steps(&self, function: &Function) -> Result<Vec<ExecutionStep>, TurbulanceError> {
        let mut steps = Vec::new();

        for statement in &function.body {
            match statement {
                Statement::ToolDelegation { tool, task, data } => {
                    let delegation = ToolDelegation {
                        tool: tool.clone(),
                        task: task.clone(),
                        data: data.clone(),
                        expected_output: "analysis_result".to_string(),
                        confidence_threshold: 0.8,
                    };
                    steps.push(ExecutionStep::DelegateToTool { delegation });
                }
                Statement::FunctionCall { name, args: _ } => {
                    if name.contains("analyze") || name.contains("process") {
                        steps.push(ExecutionStep::ProcessData {
                            function: name.clone(),
                            inputs: vec!["input_data".to_string()],
                            output: "processed_data".to_string(),
                        });
                    }
                }
                _ => {
                    // Handle other statement types
                }
            }
        }

        Ok(steps)
    }

    fn extract_tool_delegations(&self, ast: &TurbulanceAST, plan: &mut ExecutionPlan) -> Result<(), TurbulanceError> {
        // Extract all tool delegations from the AST
        for function in &ast.functions {
            for statement in &function.body {
                if let Statement::ToolDelegation { tool, task, data } = statement {
                    let delegation = ToolDelegation {
                        tool: tool.clone(),
                        task: task.clone(),
                        data: data.clone(),
                        expected_output: "tool_analysis_result".to_string(),
                        confidence_threshold: 0.8,
                    };
                    plan.tool_delegations.push(delegation);
                }
            }
        }
        Ok(())
    }

    fn extract_semantic_requirements(&self, ast: &TurbulanceAST, plan: &mut ExecutionPlan) {
        for hypothesis in &ast.hypotheses {
            if let Some(validation) = &hypothesis.semantic_validation {
                plan.semantic_requirements.push(validation.biological_understanding.clone());
                plan.semantic_requirements.push(validation.temporal_understanding.clone());
                plan.semantic_requirements.push(validation.clinical_understanding.clone());
            }
        }
    }

    // Validation rule implementations
    fn validate_hypothesis_scientific_validity(ast: &TurbulanceAST) -> Result<(), TurbulanceError> {
        for hypothesis in &ast.hypotheses {
            if hypothesis.claim.is_empty() {
                return Err(TurbulanceError::HypothesisValidationError {
                    reason: format!("Hypothesis '{}' has empty claim", hypothesis.name),
                });
            }

            // Check for circular reasoning
            if hypothesis.claim.contains(&hypothesis.name) {
                return Err(TurbulanceError::ScientificReasoningFlaw {
                    flaw: format!("Hypothesis '{}' contains circular reasoning", hypothesis.name),
                });
            }
        }
        Ok(())
    }

    fn validate_semantic_coherence(ast: &TurbulanceAST) -> Result<(), TurbulanceError> {
        // Check that semantic validation requirements are coherent
        for hypothesis in &ast.hypotheses {
            if let Some(validation) = &hypothesis.semantic_validation {
                if validation.biological_understanding.is_empty() ||
                   validation.temporal_understanding.is_empty() ||
                   validation.clinical_understanding.is_empty() {
                    return Err(TurbulanceError::SemanticError {
                        message: format!(
                            "Hypothesis '{}' has incomplete semantic validation requirements",
                            hypothesis.name
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_statistical_soundness(ast: &TurbulanceAST) -> Result<(), TurbulanceError> {
        // Check for proper statistical methodology
        for proposition in &ast.propositions {
            for motion in &proposition.motions {
                if motion.description.to_lowercase().contains("correlation") && 
                   !motion.description.to_lowercase().contains("causation") &&
                   motion.description.to_lowercase().contains("cause") {
                    return Err(TurbulanceError::ScientificReasoningFlaw {
                        flaw: "Conflating correlation with causation".to_string(),
                    });
                }
            }
        }
        Ok(())
    }
}

// Lexer and Parser implementations
struct TurbulanceLexer {
    source: String,
    position: usize,
    line: usize,
    column: usize,
}

impl TurbulanceLexer {
    fn new(source: &str) -> Self {
        Self {
            source: source.to_string(),
            position: 0,
            line: 1,
            column: 1,
        }
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, TurbulanceError> {
        let mut tokens = Vec::new();
        
        while self.position < self.source.len() {
            self.skip_whitespace();
            
            if self.position >= self.source.len() {
                break;
            }

            let token = self.next_token()?;
            tokens.push(token);
        }

        Ok(tokens)
    }

    fn skip_whitespace(&mut self) {
        while self.position < self.source.len() {
            let ch = self.current_char();
            if ch.is_whitespace() {
                if ch == '\n' {
                    self.line += 1;
                    self.column = 1;
                } else {
                    self.column += 1;
                }
                self.position += 1;
            } else {
                break;
            }
        }
    }

    fn current_char(&self) -> char {
        self.source.chars().nth(self.position).unwrap_or('\0')
    }

    fn next_token(&mut self) -> Result<Token, TurbulanceError> {
        let ch = self.current_char();
        
        match ch {
            'a'..='z' | 'A'..='Z' | '_' => self.read_identifier(),
            '0'..='9' => self.read_number(),
            '"' => self.read_string(),
            ':' => {
                self.position += 1;
                self.column += 1;
                Ok(Token::Colon)
            }
            '=' => {
                self.position += 1;
                self.column += 1;
                Ok(Token::Equals)
            }
            '(' => {
                self.position += 1;
                self.column += 1;
                Ok(Token::LeftParen)
            }
            ')' => {
                self.position += 1;
                self.column += 1;
                Ok(Token::RightParen)
            }
            '{' => {
                self.position += 1;
                self.column += 1;
                Ok(Token::LeftBrace)
            }
            '}' => {
                self.position += 1;
                self.column += 1;
                Ok(Token::RightBrace)
            }
            '/' if self.peek_char() == '/' => {
                self.skip_comment();
                self.next_token()
            }
            _ => Err(TurbulanceError::SyntaxError {
                line: self.line,
                column: self.column,
                message: format!("Unexpected character: '{}'", ch),
            }),
        }
    }

    fn peek_char(&self) -> char {
        self.source.chars().nth(self.position + 1).unwrap_or('\0')
    }

    fn read_identifier(&mut self) -> Result<Token, TurbulanceError> {
        let start = self.position;
        while self.position < self.source.len() {
            let ch = self.current_char();
            if ch.is_alphanumeric() || ch == '_' {
                self.position += 1;
                self.column += 1;
            } else {
                break;
            }
        }
        
        let value = self.source[start..self.position].to_string();
        
        // Check for keywords
        let token = match value.as_str() {
            "hypothesis" => Token::Hypothesis,
            "proposition" => Token::Proposition,
            "motion" => Token::Motion,
            "within" => Token::Within,
            "given" => Token::Given,
            "funxn" => Token::Function,
            "item" => Token::Item,
            "import" => Token::Import,
            "print" => Token::Print,
            "return" => Token::Return,
            "true" => Token::Boolean(true),
            "false" => Token::Boolean(false),
            _ => Token::Identifier(value),
        };
        
        Ok(token)
    }

    fn read_number(&mut self) -> Result<Token, TurbulanceError> {
        let start = self.position;
        while self.position < self.source.len() {
            let ch = self.current_char();
            if ch.is_numeric() || ch == '.' {
                self.position += 1;
                self.column += 1;
            } else {
                break;
            }
        }
        
        let value_str = &self.source[start..self.position];
        let value = value_str.parse::<f64>().map_err(|_| TurbulanceError::SyntaxError {
            line: self.line,
            column: self.column,
            message: format!("Invalid number: {}", value_str),
        })?;
        
        Ok(Token::Number(value))
    }

    fn read_string(&mut self) -> Result<Token, TurbulanceError> {
        self.position += 1; // Skip opening quote
        self.column += 1;
        
        let start = self.position;
        while self.position < self.source.len() && self.current_char() != '"' {
            if self.current_char() == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            self.position += 1;
        }
        
        if self.position >= self.source.len() {
            return Err(TurbulanceError::SyntaxError {
                line: self.line,
                column: self.column,
                message: "Unterminated string literal".to_string(),
            });
        }
        
        let value = self.source[start..self.position].to_string();
        self.position += 1; // Skip closing quote
        self.column += 1;
        
        Ok(Token::String(value))
    }

    fn skip_comment(&mut self) {
        while self.position < self.source.len() && self.current_char() != '\n' {
            self.position += 1;
            self.column += 1;
        }
    }
}

#[derive(Debug, Clone)]
enum Token {
    // Literals
    Identifier(String),
    String(String),
    Number(f64),
    Boolean(bool),
    
    // Keywords
    Hypothesis,
    Proposition,
    Motion,
    Within,
    Given,
    Function,
    Item,
    Import,
    Print,
    Return,
    
    // Operators and punctuation
    Colon,
    Equals,
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
}

struct TurbulanceParser {
    tokens: Vec<Token>,
    position: usize,
}

impl TurbulanceParser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, position: 0 }
    }

    fn parse(&mut self) -> Result<TurbulanceAST, TurbulanceError> {
        let mut ast = TurbulanceAST {
            imports: Vec::new(),
            hypotheses: Vec::new(),
            functions: Vec::new(),
            propositions: Vec::new(),
            main_function: None,
        };

        while self.position < self.tokens.len() {
            match &self.tokens[self.position] {
                Token::Import => {
                    ast.imports.push(self.parse_import()?);
                }
                Token::Hypothesis => {
                    ast.hypotheses.push(self.parse_hypothesis()?);
                }
                Token::Function => {
                    let func = self.parse_function()?;
                    if func.name == "main" {
                        ast.main_function = Some(func);
                    } else {
                        ast.functions.push(func);
                    }
                }
                Token::Proposition => {
                    ast.propositions.push(self.parse_proposition()?);
                }
                _ => {
                    return Err(TurbulanceError::SyntaxError {
                        line: 0,
                        column: 0,
                        message: format!("Unexpected token: {:?}", self.tokens[self.position]),
                    });
                }
            }
        }

        Ok(ast)
    }

    fn parse_import(&mut self) -> Result<String, TurbulanceError> {
        self.position += 1; // Skip 'import'
        if let Token::Identifier(module) = &self.tokens[self.position] {
            let import = module.clone();
            self.position += 1;
            Ok(import)
        } else {
            Err(TurbulanceError::SyntaxError {
                line: 0,
                column: 0,
                message: "Expected module name after 'import'".to_string(),
            })
        }
    }

    fn parse_hypothesis(&mut self) -> Result<Hypothesis, TurbulanceError> {
        self.position += 1; // Skip 'hypothesis'
        
        if let Token::Identifier(name) = &self.tokens[self.position] {
            let hypothesis_name = name.clone();
            self.position += 1;
            
            // Parse hypothesis body (simplified)
            let claim = "Sample claim".to_string(); // Would parse actual claim from tokens
            
            Ok(Hypothesis {
                name: hypothesis_name,
                claim,
                semantic_validation: None,
                requires: "authentic_semantic_comprehension".to_string(),
            })
        } else {
            Err(TurbulanceError::SyntaxError {
                line: 0,
                column: 0,
                message: "Expected hypothesis name".to_string(),
            })
        }
    }

    fn parse_function(&mut self) -> Result<Function, TurbulanceError> {
        // Simplified function parsing
        self.position += 1; // Skip 'funxn'
        
        if let Token::Identifier(name) = &self.tokens[self.position] {
            let function_name = name.clone();
            self.position += 1;
            
            Ok(Function {
                name: function_name,
                parameters: Vec::new(),
                return_type: "void".to_string(),
                body: Vec::new(),
            })
        } else {
            Err(TurbulanceError::SyntaxError {
                line: 0,
                column: 0,
                message: "Expected function name".to_string(),
            })
        }
    }

    fn parse_proposition(&mut self) -> Result<Proposition, TurbulanceError> {
        // Simplified proposition parsing
        self.position += 1; // Skip 'proposition'
        
        if let Token::Identifier(name) = &self.tokens[self.position] {
            let proposition_name = name.clone();
            self.position += 1;
            
            Ok(Proposition {
                name: proposition_name,
                motions: Vec::new(),
                within_clauses: Vec::new(),
            })
        } else {
            Err(TurbulanceError::SyntaxError {
                line: 0,
                column: 0,
                message: "Expected proposition name".to_string(),
            })
        }
    }
}

impl Default for TurbulanceCompiler {
    fn default() -> Self {
        Self::new()
    }
}

// Python bindings
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
#[pymethods]
impl TurbulanceCompiler {
    #[new]
    pub fn py_new() -> Self {
        Self::new()
    }

    #[pyo3(name = "parse")]
    pub fn py_parse(&self, source: &str) -> PyResult<String> {
        match self.parse(source) {
            Ok(ast) => Ok(serde_json::to_string(&ast).unwrap()),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Parse error: {}", e))),
        }
    }

    #[pyo3(name = "compile")]
    pub fn py_compile(&self, source: &str) -> PyResult<String> {
        match self.parse(source) {
            Ok(ast) => match self.compile(ast) {
                Ok(plan) => Ok(serde_json::to_string(&plan).unwrap()),
                Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Compile error: {}", e))),
            },
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Parse error: {}", e))),
        }
    }

    #[pyo3(name = "validate_script")]
    pub fn py_validate_script(&self, source: &str) -> PyResult<Vec<String>> {
        match self.parse(source) {
            Ok(ast) => match self.validate(&ast) {
                Ok(()) => Ok(vec![]),
                Err(errors) => Ok(errors.into_iter().map(|e| e.to_string()).collect()),
            },
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Parse error: {}", e))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_parsing() {
        let compiler = TurbulanceCompiler::new();
        let source = r#"
            hypothesis TestHypothesis:
                claim: "This is a test hypothesis"
                requires: "scientific_validation"
        "#;

        let result = compiler.parse(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hypothesis_validation() {
        let compiler = TurbulanceCompiler::new();
        
        // Valid hypothesis
        let valid_claim = "Genomic variants predict disease risk with 85% accuracy";
        assert!(compiler.analyze_hypothesis_validity(valid_claim).is_ok());
        
        // Invalid hypothesis (no testable prediction)
        let invalid_claim = "Genes are important for health";
        assert!(compiler.analyze_hypothesis_validity(invalid_claim).is_err());
    }

    #[test]
    fn test_scientific_reasoning_validation() {
        let ast = TurbulanceAST {
            imports: Vec::new(),
            hypotheses: vec![
                Hypothesis {
                    name: "CircularHypothesis".to_string(),
                    claim: "CircularHypothesis is true because CircularHypothesis".to_string(),
                    semantic_validation: None,
                    requires: "validation".to_string(),
                }
            ],
            functions: Vec::new(),
            propositions: Vec::new(),
            main_function: None,
        };

        let result = TurbulanceCompiler::validate_hypothesis_scientific_validity(&ast);
        assert!(result.is_err());
    }
} 