---
layout: page
title: API Reference
permalink: /api-reference/
---

# Gospel API Reference

This comprehensive reference covers Gospel's programmatic interface for integration into custom applications and workflows.

## Table of Contents
1. [Core Classes](#core-classes)
2. [Domain Analyzers](#domain-analyzers)
3. [Data Models](#data-models)
4. [Utilities](#utilities)
5. [Integration Interfaces](#integration-interfaces)
6. [Configuration](#configuration)

## Core Classes

### GospelAnalyzer

Main analysis engine for genomic data processing.

```python
class GospelAnalyzer:
    """Primary interface for Gospel genomic analysis."""
    
    def __init__(self, config: Optional[Dict] = None, domains: List[str] = None):
        """
        Initialize Gospel analyzer.
        
        Args:
            config: Configuration dictionary
            domains: List of domains to analyze ['fitness', 'pharmacogenetics', 'nutrition']
        """
```

#### Methods

##### analyze_vcf()

```python
def analyze_vcf(self, 
               vcf_file: str, 
               output_dir: str = None,
               population: str = 'EUR',
               domains: List[str] = None,
               **kwargs) -> AnalysisResults:
    """
    Analyze VCF file and generate comprehensive genomic insights.
    
    Args:
        vcf_file: Path to VCF file
        output_dir: Output directory for results
        population: Population ancestry code (EUR, AFR, EAS, AMR, SAS)
        domains: Analysis domains to include
        **kwargs: Additional analysis parameters
        
    Returns:
        AnalysisResults object containing comprehensive analysis
        
    Example:
        >>> analyzer = GospelAnalyzer(domains=['fitness', 'pharmacogenetics'])
        >>> results = analyzer.analyze_vcf('genome.vcf', 'results/')
        >>> print(f"Overall fitness score: {results.fitness.overall_score}")
    """
```

##### analyze_variants()

```python
def analyze_variants(self, 
                    variants: List[Variant],
                    domains: List[str] = None) -> AnalysisResults:
    """
    Analyze pre-loaded variants.
    
    Args:
        variants: List of Variant objects
        domains: Analysis domains
        
    Returns:
        AnalysisResults object
    """
```

##### query()

```python
def query(self, 
         question: str, 
         results: AnalysisResults = None,
         context: Dict = None) -> str:
    """
    Query analysis results using AI.
    
    Args:
        question: Natural language question
        results: Analysis results to query
        context: Additional context
        
    Returns:
        AI-generated response
        
    Example:
        >>> response = analyzer.query(
        ...     "What are my genetic advantages for endurance sports?",
        ...     results=analysis_results
        ... )
    """
```

### VariantProcessor

Core variant processing and extraction engine.

```python
class VariantProcessor:
    """Process and extract genetic variants from genomic data."""
    
    def __init__(self, config: Dict):
        """
        Initialize variant processor.
        
        Args:
            config: Processing configuration
        """
```

#### Methods

##### process_vcf()

```python
def process_vcf(self, vcf_file: str) -> List[Variant]:
    """
    Process VCF file and extract variants.
    
    Args:
        vcf_file: Path to VCF file
        
    Returns:
        List of Variant objects
    """
```

##### extract_snps()

```python
def extract_snps(self, genome_data: Dict) -> Set[Variant]:
    """
    Extract SNP variants.
    
    Args:
        genome_data: Genome data dictionary
        
    Returns:
        Set of SNP variants
    """
```

##### detect_cnvs()

```python
def detect_cnvs(self, genome_data: Dict) -> Set[Variant]:
    """
    Detect copy number variations.
    
    Args:
        genome_data: Genome data dictionary
        
    Returns:
        Set of CNV variants
    """
```

### VariantAnnotator

Functional annotation of genetic variants.

```python
class VariantAnnotator:
    """Annotate variants with functional impact predictions."""
    
    def __init__(self, annotation_config: Dict):
        """
        Initialize annotator.
        
        Args:
            annotation_config: Annotation configuration
        """
```

#### Methods

##### annotate_variants()

```python
def annotate_variants(self, variants: List[Variant]) -> List[Variant]:
    """
    Annotate variants with functional predictions.
    
    Args:
        variants: List of variants to annotate
        
    Returns:
        List of annotated variants
    """
```

##### predict_functional_impact()

```python
def predict_functional_impact(self, variant: Variant) -> Dict:
    """
    Predict functional impact of single variant.
    
    Args:
        variant: Variant to analyze
        
    Returns:
        Functional impact predictions
    """
```

## Domain Analyzers

### FitnessDomain

Analyze genetic factors affecting athletic performance.

```python
class FitnessDomain:
    """Fitness and athletic performance genetic analysis."""
    
    def __init__(self, config: Dict = None):
        """Initialize fitness analyzer."""
```

#### Methods

##### analyze_fitness_profile()

```python
def analyze_fitness_profile(self, variants: List[Variant]) -> FitnessProfile:
    """
    Analyze comprehensive fitness genetic profile.
    
    Args:
        variants: List of genetic variants
        
    Returns:
        FitnessProfile with scores and recommendations
        
    Example:
        >>> fitness = FitnessDomain()
        >>> profile = fitness.analyze_fitness_profile(variants)
        >>> print(f"Sprint score: {profile.sprint_score}")
        >>> print(f"Endurance score: {profile.endurance_score}")
    """
```

##### generate_training_recommendations()

```python
def generate_training_recommendations(self, 
                                    fitness_profile: FitnessProfile,
                                    sport_focus: str = None) -> TrainingPlan:
    """
    Generate personalized training recommendations.
    
    Args:
        fitness_profile: Analyzed fitness profile
        sport_focus: Specific sport focus
        
    Returns:
        TrainingPlan with detailed recommendations
    """
```

##### assess_injury_risk()

```python
def assess_injury_risk(self, variants: List[Variant]) -> InjuryRiskAssessment:
    """
    Assess genetic injury risk factors.
    
    Args:
        variants: Genetic variants
        
    Returns:
        InjuryRiskAssessment with risk factors and prevention strategies
    """
```

### PharmacogeneticsDomain

Analyze drug metabolism and response genetics.

```python
class PharmacogeneticsDomain:
    """Pharmacogenetic analysis for drug response prediction."""
    
    def __init__(self, config: Dict = None):
        """Initialize pharmacogenetics analyzer."""
```

#### Methods

##### analyze_cyp_status()

```python
def analyze_cyp_status(self, variants: List[Variant]) -> CYPStatus:
    """
    Analyze cytochrome P450 enzyme status.
    
    Args:
        variants: Genetic variants
        
    Returns:
        CYPStatus with metabolizer phenotypes
        
    Example:
        >>> pharma = PharmacogeneticsDomain()
        >>> cyp_status = pharma.analyze_cyp_status(variants)
        >>> print(f"CYP2D6 status: {cyp_status.cyp2d6_phenotype}")
    """
```

##### predict_drug_response()

```python
def predict_drug_response(self, 
                         drug_name: str, 
                         variants: List[Variant]) -> DrugResponse:
    """
    Predict response to specific drug.
    
    Args:
        drug_name: Name of drug
        variants: Genetic variants
        
    Returns:
        DrugResponse with efficacy and safety predictions
    """
```

##### generate_dosing_recommendations()

```python
def generate_dosing_recommendations(self, 
                                  drug_name: str,
                                  patient_data: Dict) -> DosingRecommendation:
    """
    Generate personalized dosing recommendations.
    
    Args:
        drug_name: Drug name
        patient_data: Patient information including genetics
        
    Returns:
        DosingRecommendation with dose adjustments
    """
```

### NutritionDomain

Analyze nutritional genetics and dietary requirements.

```python
class NutritionDomain:
    """Nutritional genomics analysis."""
    
    def __init__(self, config: Dict = None):
        """Initialize nutrition analyzer."""
```

#### Methods

##### analyze_macronutrient_metabolism()

```python
def analyze_macronutrient_metabolism(self, 
                                   variants: List[Variant]) -> MacronutrientProfile:
    """
    Analyze macronutrient metabolism genetics.
    
    Args:
        variants: Genetic variants
        
    Returns:
        MacronutrientProfile with optimal ratios
    """
```

##### analyze_micronutrient_requirements()

```python
def analyze_micronutrient_requirements(self, 
                                     variants: List[Variant]) -> MicronutrientProfile:
    """
    Analyze micronutrient requirements.
    
    Args:
        variants: Genetic variants
        
    Returns:
        MicronutrientProfile with personalized requirements
    """
```

##### analyze_food_sensitivities()

```python
def analyze_food_sensitivities(self, 
                             variants: List[Variant]) -> FoodSensitivityProfile:
    """
    Analyze genetic food sensitivities.
    
    Args:
        variants: Genetic variants
        
    Returns:
        FoodSensitivityProfile with sensitivity predictions
    """
```

## Data Models

### Variant

Core variant data structure.

```python
@dataclass
class Variant:
    """Representation of a genetic variant."""
    
    id: str                          # Variant identifier (e.g., rs123456)
    chromosome: str                  # Chromosome (1-22, X, Y, MT)
    position: int                    # Genomic position
    reference: str                   # Reference allele
    alternate: str                   # Alternate allele
    quality: float                   # Variant quality score
    genotype: str                    # Individual genotype (e.g., "0/1")
    type: VariantType               # Variant type enum
    functional_impact: Dict          # Functional predictions
    domain_scores: Dict              # Domain-specific scores
    
    # Methods
    def is_heterozygous(self) -> bool:
        """Check if variant is heterozygous."""
        
    def is_homozygous_alt(self) -> bool:
        """Check if variant is homozygous alternate."""
        
    def get_allele_frequency(self, population: str = 'ALL') -> float:
        """Get population allele frequency."""
```

### AnalysisResults

Comprehensive analysis results container.

```python
@dataclass
class AnalysisResults:
    """Container for comprehensive analysis results."""
    
    # Core data
    variants: List[Variant]          # Analyzed variants
    summary_stats: Dict              # Summary statistics
    
    # Domain-specific results
    fitness: Optional[FitnessProfile]
    pharmacogenetics: Optional[PharmaProfile] 
    nutrition: Optional[NutritionProfile]
    
    # Network analysis
    networks: Optional[NetworkAnalysis]
    
    # Metadata
    analysis_date: datetime
    config: Dict
    
    # Methods
    def get_domain_score(self, domain: str) -> float:
        """Get overall score for domain."""
        
    def get_top_variants(self, n: int = 10) -> List[Variant]:
        """Get top N highest-impact variants."""
        
    def export_summary(self) -> Dict:
        """Export summary for external use."""
```

### FitnessProfile

Fitness domain analysis results.

```python
@dataclass
class FitnessProfile:
    """Fitness genetic profile results."""
    
    # Overall scores
    overall_score: float             # 0-10 overall fitness score
    confidence: float                # Confidence in predictions
    
    # Component scores
    sprint_score: float              # Sprint/power performance
    endurance_score: float           # Endurance capacity
    strength_score: float            # Strength potential
    recovery_score: float            # Recovery rate
    injury_risk_score: float         # Injury susceptibility
    
    # Detailed analysis
    key_variants: List[Variant]      # Most impactful variants
    genetic_advantages: List[str]    # Genetic strengths
    genetic_limitations: List[str]   # Areas for improvement
    
    # Recommendations
    optimal_sports: List[str]        # Recommended sports
    training_focus: List[str]        # Training recommendations
    
    # Methods
    def get_dominant_profile(self) -> str:
        """Get dominant fitness profile (sprint/endurance/mixed)."""
        
    def get_injury_prevention_strategies(self) -> List[str]:
        """Get personalized injury prevention recommendations."""
```

### PharmaProfile

Pharmacogenetics analysis results.

```python
@dataclass
class PharmaProfile:
    """Pharmacogenetic profile results."""
    
    # CYP enzyme status
    cyp2d6_phenotype: str           # Poor/Intermediate/Normal/Ultrarapid
    cyp2c19_phenotype: str
    cyp2c9_phenotype: str
    cyp3a4_phenotype: str
    
    # Drug categories
    cardiovascular_drugs: Dict       # CV drug recommendations
    psychiatric_drugs: Dict          # Psychiatric drug recommendations
    pain_medications: Dict           # Pain medication recommendations
    
    # Safety alerts
    high_risk_drugs: List[str]       # Drugs to avoid
    dose_adjustments: Dict           # Required dose modifications
    
    # Methods
    def is_poor_metabolizer(self, enzyme: str) -> bool:
        """Check if poor metabolizer for specific enzyme."""
        
    def get_drug_recommendations(self, drug_class: str) -> Dict:
        """Get recommendations for drug class."""
```

### NutritionProfile

Nutritional genomics analysis results.

```python
@dataclass
class NutritionProfile:
    """Nutritional genetic profile results."""
    
    # Macronutrient metabolism
    carb_metabolism: str            # Fast/Normal/Slow
    fat_metabolism: str
    protein_requirements: str       # Low/Normal/High
    
    # Micronutrient needs
    folate_requirements: str        # Standard/Enhanced
    vitamin_d_requirements: str
    b_vitamin_needs: Dict
    
    # Food sensitivities
    lactose_tolerance: bool
    gluten_sensitivity_risk: str    # Low/Moderate/High
    caffeine_sensitivity: str       # Low/Moderate/High
    
    # Recommendations
    optimal_diet_type: str          # Mediterranean/Low-carb/etc.
    supplement_recommendations: List[str]
    foods_to_emphasize: List[str]
    foods_to_limit: List[str]
    
    # Methods
    def get_macronutrient_ratios(self) -> Dict:
        """Get optimal macronutrient ratios."""
        
    def requires_supplementation(self, nutrient: str) -> bool:
        """Check if supplementation recommended for nutrient."""
```

## Utilities

### NetworkAnalyzer

Protein interaction network analysis.

```python
class NetworkAnalyzer:
    """Analyze protein interaction networks."""
    
    def __init__(self, network_databases: List[str] = None):
        """Initialize network analyzer."""
    
    def build_variant_network(self, variants: List[Variant]) -> nx.Graph:
        """Build protein interaction network from variants."""
    
    def calculate_centrality_measures(self, network: nx.Graph) -> Dict:
        """Calculate network centrality measures."""
    
    def identify_key_pathways(self, 
                             network: nx.Graph, 
                             variants: List[Variant]) -> List[str]:
        """Identify key biological pathways."""
```

### VisualizationEngine

Generate charts and visualizations.

```python
class VisualizationEngine:
    """Generate genomic analysis visualizations."""
    
    def __init__(self, theme: str = 'default'):
        """Initialize visualization engine."""
    
    def create_domain_score_chart(self, 
                                 results: AnalysisResults,
                                 output_file: str):
        """Create domain score visualization."""
    
    def create_fitness_radar(self, 
                           fitness_profile: FitnessProfile,
                           output_file: str):
        """Create fitness profile radar chart."""
    
    def create_pharma_heatmap(self, 
                            pharma_profile: PharmaProfile,
                            output_file: str):
        """Create pharmacogenetic heatmap."""
    
    def create_network_plot(self, 
                          network: nx.Graph,
                          output_file: str):
        """Create interactive network visualization."""
```

### ReportGenerator

Generate comprehensive reports.

```python
class ReportGenerator:
    """Generate analysis reports in multiple formats."""
    
    def __init__(self, template_dir: str = None):
        """Initialize report generator."""
    
    def generate_html_report(self, 
                           results: AnalysisResults,
                           output_file: str) -> str:
        """Generate interactive HTML report."""
    
    def generate_pdf_report(self, 
                          results: AnalysisResults,
                          output_file: str) -> str:
        """Generate PDF report."""
    
    def generate_clinical_report(self, 
                               results: AnalysisResults,
                               patient_info: Dict,
                               output_file: str) -> str:
        """Generate clinical-formatted report."""
```

## Integration Interfaces

### LLMInterface

AI language model integration.

```python
class LLMInterface:
    """Interface for AI language model integration."""
    
    def __init__(self, model_config: Dict):
        """
        Initialize LLM interface.
        
        Args:
            model_config: Model configuration including endpoint, model name, etc.
        """
    
    def query(self, 
             prompt: str, 
             context: Dict = None,
             temperature: float = 0.1) -> str:
        """Query language model with genomic context."""
    
    def generate_explanation(self, 
                           variant: Variant,
                           domain: str) -> str:
        """Generate explanation for variant impact."""
    
    def create_recommendations(self, 
                             results: AnalysisResults) -> List[str]:
        """Generate personalized recommendations."""
```

### DatabaseInterface

Interface for genomic databases.

```python
class DatabaseInterface:
    """Interface for genomic database access."""
    
    def __init__(self, db_config: Dict):
        """Initialize database interface."""
    
    def get_variant_annotation(self, variant_id: str) -> Dict:
        """Get variant annotation from databases."""
    
    def get_population_frequency(self, 
                               variant_id: str,
                               population: str) -> float:
        """Get population allele frequency."""
    
    def get_drug_interactions(self, gene: str) -> List[Dict]:
        """Get drug interaction data for gene."""
```

### EHRInterface

Electronic health record integration.

```python
class EHRInterface:
    """Interface for EHR system integration."""
    
    def __init__(self, ehr_config: Dict):
        """Initialize EHR interface."""
    
    def get_patient_info(self, patient_id: str) -> Dict:
        """Retrieve patient information."""
    
    def update_patient_genetics(self, 
                              patient_id: str,
                              genetic_data: Dict):
        """Update patient genetic profile."""
    
    def send_clinical_alert(self, 
                          patient_id: str,
                          alert: Dict):
        """Send clinical alert to EHR."""
```

## Configuration

### GospelConfig

Main configuration class for Gospel framework.

```python
class GospelConfig:
    """Gospel framework configuration."""
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to YAML configuration file
        """
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'GospelConfig':
        """Create configuration from dictionary."""
    
    @classmethod
    def default(cls) -> 'GospelConfig':
        """Create default configuration."""
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
    
    def save(self, file_path: str):
        """Save configuration to file."""
    
    # Configuration properties
    @property
    def database_config(self) -> Dict:
        """Database configuration."""
    
    @property
    def analysis_config(self) -> Dict:
        """Analysis parameters configuration."""
    
    @property
    def domain_configs(self) -> Dict:
        """Domain-specific configurations."""
    
    @property
    def ai_config(self) -> Dict:
        """AI/LLM configuration."""
```

### Usage Examples

#### Basic Analysis

```python
from gospel import GospelAnalyzer, GospelConfig

# Create analyzer with default configuration
analyzer = GospelAnalyzer(domains=['fitness', 'pharmacogenetics'])

# Analyze VCF file
results = analyzer.analyze_vcf(
    vcf_file='genome.vcf',
    output_dir='analysis_results/',
    population='EUR'
)

# Access results
print(f"Fitness score: {results.fitness.overall_score}")
print(f"CYP2D6 status: {results.pharmacogenetics.cyp2d6_phenotype}")

# Query with AI
response = analyzer.query(
    "What are my genetic advantages for athletic performance?",
    results=results
)
print(response)
```

#### Custom Configuration

```python
from gospel import GospelAnalyzer, GospelConfig

# Create custom configuration
config = GospelConfig.default()
config.analysis_config['min_confidence'] = 0.8
config.domain_configs['fitness']['include_injury_risk'] = True
config.ai_config['model'] = 'llama3'

# Initialize analyzer with custom config
analyzer = GospelAnalyzer(config=config.to_dict())

# Run analysis
results = analyzer.analyze_vcf('genome.vcf')
```

#### Domain-Specific Analysis

```python
from gospel.domains import FitnessDomain, PharmacogeneticsDomain
from gospel.core import VariantProcessor

# Process variants
processor = VariantProcessor(config={})
variants = processor.process_vcf('genome.vcf')

# Fitness analysis
fitness = FitnessDomain()
fitness_profile = fitness.analyze_fitness_profile(variants)
training_plan = fitness.generate_training_recommendations(
    fitness_profile, 
    sport_focus='endurance'
)

# Pharmacogenetics analysis
pharma = PharmacogeneticsDomain()
drug_response = pharma.predict_drug_response('warfarin', variants)
dosing = pharma.generate_dosing_recommendations('warfarin', {
    'age': 45,
    'weight': 70,
    'variants': variants
})
```

#### Visualization and Reporting

```python
from gospel import VisualizationEngine, ReportGenerator

# Create visualizations
viz = VisualizationEngine(theme='publication')
viz.create_fitness_radar(fitness_profile, 'fitness_radar.html')
viz.create_pharma_heatmap(pharma_profile, 'pharma_heatmap.html')

# Generate reports
reporter = ReportGenerator()
reporter.generate_html_report(results, 'comprehensive_report.html')
reporter.generate_pdf_report(results, 'summary_report.pdf')
```

---

This API reference provides comprehensive documentation for integrating Gospel into custom applications and workflows. For practical implementation examples, see the [Examples](examples.html) section.

**Next:** Explore [Contributing](contributing.html) to help improve Gospel or return to the [Getting Started](getting-started.html) guide for implementation. 