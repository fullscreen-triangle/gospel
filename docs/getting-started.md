---
layout: page
title: Getting Started
permalink: /getting-started/
---

# Getting Started with Gospel

This guide will walk you through installing Gospel, setting up your environment, and running your first genomic analysis. By the end of this tutorial, you'll understand how to leverage Gospel's multi-domain capabilities for comprehensive genomic insights.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Basic Configuration](#basic-configuration)
4. [Your First Analysis](#your-first-analysis)
5. [Understanding Results](#understanding-results)
6. [Next Steps](#next-steps)

## Prerequisites

Before installing Gospel, ensure your system meets these requirements:

### System Requirements
- **Python**: 3.9 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended for large genomes)
- **Storage**: At least 10GB free space for databases and results
- **OS**: Linux, macOS, or Windows with WSL

### Required Knowledge
- Basic understanding of genomics and VCF files
- Familiarity with command-line interfaces
- Understanding of genetic variants (SNPs, indels, structural variants)

### Input Data Requirements
Gospel works with standard genomic file formats:
- **VCF files**: Variant Call Format files from genome sequencing
- **FASTA files**: Reference genome sequences
- **Annotation files**: Gene annotations in GTF/GFF format

## Installation

### Option 1: pip Installation (Recommended)

```bash
# Create a virtual environment
python -m venv gospel-env
source gospel-env/bin/activate  # On Windows: gospel-env\Scripts\activate

# Install Gospel
pip install gospel

# Verify installation
gospel --version
```

### Option 2: Development Installation

For developers or users who want the latest features:

```bash
# Clone the repository
git clone https://github.com/fullscreen-triangle/gospel.git
cd gospel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install additional development dependencies
pip install -r requirements-dev.txt
```

### Option 3: Docker Installation

For containerized deployment:

```bash
# Pull the Gospel Docker image
docker pull gospelgenomics/gospel:latest

# Run Gospel in a container
docker run -it --rm -v $(pwd):/data gospelgenomics/gospel:latest
```

## Basic Configuration

### Database Setup

Gospel requires several reference databases for comprehensive analysis:

```bash
# Download and setup reference databases
gospel setup-databases --target-dir ~/.gospel/databases

# This downloads:
# - Human reference genome (GRCh38)
# - ClinVar variant database
# - dbSNP variant annotations
# - Protein interaction networks
# - Pathway databases (KEGG, Reactome)
# - Population frequency data (gnomAD)
```

### Configuration File

Create a configuration file to customize Gospel's behavior:

```bash
# Generate default configuration
gospel config init

# Edit the configuration file
nano ~/.gospel/config.yaml
```

**Sample Configuration (`~/.gospel/config.yaml`):**

```yaml
# Gospel Configuration File
database:
  path: ~/.gospel/databases
  cache_size: 1000MB

analysis:
  default_domains: [fitness, pharmacogenetics, nutrition]
  variant_filters:
    min_quality: 30
    min_depth: 10
    max_allele_frequency: 0.05

scoring:
  weights:
    functional_impact: 0.4
    conservation: 0.3
    population_frequency: 0.2
    literature_evidence: 0.1

output:
  format: html
  include_visualizations: true
  detailed_annotations: true

llm:
  model: ollama/llama2
  temperature: 0.1
  max_tokens: 2000
```

### Environment Variables

Set up environment variables for API access:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export GOSPEL_DATABASE_PATH=~/.gospel/databases
export GOSPEL_CACHE_DIR=~/.gospel/cache
export OLLAMA_HOST=localhost:11434  # If using local Ollama
```

## Your First Analysis

### Preparing Input Data

For this tutorial, we'll use a sample VCF file. If you have your own genomic data, ensure it's in VCF format.

```bash
# Download sample data (if needed)
wget https://github.com/fullscreen-triangle/gospel/raw/main/examples/sample_genome.vcf.gz
gunzip sample_genome.vcf.gz
```

### Basic Analysis Command

Run a comprehensive analysis across all domains:

```bash
gospel analyze \
    --vcf sample_genome.vcf \
    --output results_directory \
    --domains fitness,pharmacogenetics,nutrition \
    --reference GRCh38 \
    --population EUR
```

**Command Breakdown:**
- `--vcf`: Input VCF file containing your genetic variants
- `--output`: Directory for analysis results
- `--domains`: Genomic domains to analyze
- `--reference`: Reference genome version
- `--population`: Population ancestry for frequency comparisons

### Domain-Specific Analysis

You can also run analyses for specific domains:

```bash
# Fitness domain only
gospel analyze --vcf sample_genome.vcf --domains fitness --output fitness_results

# Pharmacogenetics with drug focus
gospel analyze --vcf sample_genome.vcf --domains pharmacogenetics \
    --drugs "warfarin,clopidogrel,codeine" --output pharma_results

# Nutritional genomics
gospel analyze --vcf sample_genome.vcf --domains nutrition \
    --nutrients "vitamin_d,folate,caffeine" --output nutrition_results
```

### Advanced Analysis Options

```bash
# High-sensitivity analysis with custom thresholds
gospel analyze \
    --vcf sample_genome.vcf \
    --domains fitness,pharmacogenetics,nutrition \
    --sensitivity high \
    --min-variant-quality 20 \
    --max-population-frequency 0.01 \
    --include-regulatory-variants \
    --network-analysis \
    --output comprehensive_results
```

## Understanding Results

Gospel generates comprehensive output organized in several sections:

### Output Structure

```
results_directory/
├── summary_report.html          # Interactive HTML report
├── detailed_analysis.json       # Machine-readable results
├── variants/
│   ├── fitness_variants.vcf     # Domain-specific variant subsets
│   ├── pharma_variants.vcf
│   └── nutrition_variants.vcf
├── scores/
│   ├── domain_scores.csv        # Numerical scores for each domain
│   └── variant_impacts.csv      # Individual variant assessments
├── visualizations/
│   ├── network_plots/           # Protein interaction networks
│   ├── pathway_analysis/        # Biological pathway enrichment
│   └── distribution_plots/      # Score distributions
└── annotations/
    ├── functional_predictions.csv
    ├── literature_references.txt
    └── clinical_significance.csv
```

### Key Output Files

#### 1. Summary Report (`summary_report.html`)
Interactive HTML report with:
- Overall genomic profile summary
- Domain-specific insights and recommendations
- Interactive visualizations
- Detailed variant annotations
- Personalized recommendations

#### 2. Domain Scores (`scores/domain_scores.csv`)
```csv
Domain,Overall_Score,Confidence,Top_Genes,Risk_Factors
Fitness,7.2,High,"ACTN3,COL1A1,ACE",Power_vs_Endurance
Pharmacogenetics,6.8,Medium,"CYP2D6,SLCO1B1",Warfarin_Sensitivity
Nutrition,8.1,High,"MTHFR,FTO,VDR",Folate_Metabolism
```

#### 3. Variant Impacts (`scores/variant_impacts.csv`)
```csv
Chromosome,Position,Gene,Variant,Domain,Impact_Score,Functional_Effect,Population_Frequency
1,230845794,AGT,rs699,Fitness,0.85,Moderate_Effect,0.42
19,41009748,CYP2D6,rs3892097,Pharmacogenetics,0.92,High_Effect,0.28
1,219781280,MTHFR,rs1801133,Nutrition,0.78,Moderate_Effect,0.35
```

### Interpreting Scores

Gospel uses a 0-10 scoring system:

| Score Range | Interpretation | Recommendation |
|-------------|----------------|----------------|
| 8.0 - 10.0 | Excellent genetic profile | Maintain current strategies |
| 6.0 - 7.9 | Good profile with optimization potential | Consider targeted interventions |
| 4.0 - 5.9 | Moderate challenges present | Implement specific modifications |
| 2.0 - 3.9 | Significant genetic considerations | Require professional guidance |
| 0.0 - 1.9 | High-impact genetic factors | Essential professional consultation |

### Interactive Query Interface

Explore your results using Gospel's AI-powered query system:

```bash
# Start interactive session
gospel query --interactive --results results_directory

# Example queries:
# "What are my genetic advantages for endurance sports?"
# "Which medications should I be cautious about?"
# "How does my folate metabolism affect my nutrition needs?"
# "Show me the protein networks involved in my fitness profile"
```

## Next Steps

Now that you've completed your first analysis, here are recommended next steps:

### 1. Deep Dive into Domains
- **[Fitness Analysis](domains.html#fitness)**: Understand your athletic genetic profile
- **[Pharmacogenetics](domains.html#pharmacogenetics)**: Learn about drug interactions
- **[Nutritional Genomics](domains.html#nutrition)**: Optimize your nutritional strategy

### 2. Advanced Features
- **[Network Analysis](architecture.html#network-analysis)**: Explore protein interaction networks
- **[Custom Domains](api-reference.html#custom-domains)**: Add your own analysis domains
- **[Batch Processing](cli-reference.html#batch-analysis)**: Analyze multiple genomes

### 3. Integration and Automation
- **[API Usage](api-reference.html)**: Integrate Gospel into your workflows
- **[Pipeline Integration](examples.html#pipeline-integration)**: Automate analysis pipelines
- **[Custom Visualization](examples.html#visualization)**: Create custom reports

### 4. Community and Support
- **[Contributing](contributing.html)**: Help improve Gospel
- **[GitHub Issues](https://github.com/fullscreen-triangle/gospel/issues)**: Report bugs or request features
- **[Discussion Forum](https://github.com/fullscreen-triangle/gospel/discussions)**: Join the community

## Troubleshooting

### Common Issues

#### Installation Problems
```bash
# If pip installation fails
pip install --upgrade pip setuptools wheel
pip install gospel --no-cache-dir

# For dependency conflicts
pip install gospel --force-reinstall
```

#### Database Download Issues
```bash
# Manual database setup
gospel setup-databases --target-dir ~/.gospel/databases --force-update

# Check database integrity
gospel validate-databases
```

#### Memory Issues
```bash
# Reduce memory usage
gospel analyze --vcf input.vcf --memory-limit 4GB --chunk-size 1000
```

#### Performance Optimization
```bash
# Use multiple CPU cores
gospel analyze --vcf input.vcf --threads 8

# Enable caching
gospel analyze --vcf input.vcf --enable-cache --cache-dir ~/.gospel/cache
```

### Getting Help

If you encounter issues:

1. Check the [FAQ](https://github.com/fullscreen-triangle/gospel/wiki/FAQ)
2. Search [existing issues](https://github.com/fullscreen-triangle/gospel/issues)
3. Join our [discussion forum](https://github.com/fullscreen-triangle/gospel/discussions)
4. Create a [new issue](https://github.com/fullscreen-triangle/gospel/issues/new) with:
   - Gospel version (`gospel --version`)
   - Operating system and Python version
   - Complete error message
   - Minimal example to reproduce the issue

---

**Ready for deeper analysis?** Continue to [Architecture Deep Dive](architecture.html) to understand Gospel's technical foundations, or jump to [Domain Analysis](domains.html) to explore specific genomic domains in detail. 