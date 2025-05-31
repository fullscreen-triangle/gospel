---
layout: page
title: CLI Reference
permalink: /cli-reference/
---

# Gospel Command Line Interface Reference

This comprehensive reference covers all Gospel CLI commands, options, and usage patterns for genomic analysis workflows.

## Table of Contents
1. [Command Overview](#command-overview)
2. [Global Options](#global-options)
3. [Analyze Command](#analyze-command)
4. [Query Command](#query-command)
5. [Visualize Command](#visualize-command)
6. [Knowledge Base Commands](#knowledge-base-commands)
7. [LLM Commands](#llm-commands)
8. [Configuration](#configuration)
9. [Workflow Examples](#workflow-examples)

## Command Overview

Gospel provides a comprehensive CLI for genomic analysis and AI-powered interpretation:

```bash
gospel --help
```

### Main Commands

| Command | Purpose | Primary Use Case |
|---------|---------|------------------|
| `analyze` | Process genomic data and extract insights | Core genomic analysis |
| `query` | Interactive AI-powered queries | Explore analysis results |
| `visualize` | Generate charts and network visualizations | Data presentation |
| `kb` | Manage knowledge base | Build and query scientific databases |
| `llm` | Work with language models | Train and query domain-specific AI |

### Quick Start

```bash
# Basic analysis across all domains
gospel analyze --vcf genome.vcf --output results/

# Domain-specific analysis
gospel analyze --vcf genome.vcf --domains fitness --output fitness_results/

# Interactive exploration
gospel query --interactive --results results/

# Generate visualizations
gospel visualize --results results/ --output charts/
```

## Global Options

These options are available for all Gospel commands:

```bash
gospel [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

### Common Global Options

```bash
--version                    # Show Gospel version
--config PATH               # Specify configuration file
--verbose, -v               # Enable verbose output
--quiet, -q                 # Suppress non-error output
--log-level LEVEL           # Set logging level (DEBUG, INFO, WARN, ERROR)
--threads N                 # Number of parallel threads
--memory-limit SIZE         # Memory usage limit (e.g., "8GB")
```

### Configuration File

Specify a custom configuration:

```bash
gospel --config ~/.gospel/custom_config.yaml analyze --vcf genome.vcf
```

## Analyze Command

The `analyze` command is Gospel's core genomic analysis engine.

### Basic Syntax

```bash
gospel analyze --vcf INPUT.vcf [OPTIONS]
```

### Input Options

```bash
# Required
--vcf PATH                  # Input VCF file

# Optional input files
--reference PATH            # Reference genome (default: GRCh38)
--annotation PATH           # Custom annotation file
--pedigree PATH             # Family pedigree file
--phenotype PATH            # Phenotype data file
```

### Domain Selection

```bash
# All domains (default)
--domains all

# Specific domains
--domains fitness
--domains pharmacogenetics  
--domains nutrition

# Multiple domains
--domains fitness,pharmacogenetics
```

### Analysis Parameters

```bash
# Quality filters
--min-quality N             # Minimum variant quality score (default: 30)
--min-depth N               # Minimum read depth (default: 10)
--max-allele-freq FLOAT     # Maximum population frequency (default: 0.05)

# Analysis scope
--include-regulatory        # Include regulatory region variants
--include-structural        # Include structural variants
--include-cnvs              # Include copy number variations

# Population parameters
--population CODE           # Population ancestry (EUR, AFR, AMR, EAS, SAS)
--custom-frequencies PATH   # Custom allele frequency database
```

### Scoring Options

```bash
# Scoring weights
--functional-weight FLOAT   # Weight for functional impact (default: 0.4)
--conservation-weight FLOAT # Weight for conservation (default: 0.3)
--frequency-weight FLOAT    # Weight for population frequency (default: 0.2)
--literature-weight FLOAT   # Weight for literature evidence (default: 0.1)

# Confidence thresholds
--min-confidence FLOAT      # Minimum confidence for reporting (default: 0.6)
--high-confidence FLOAT     # Threshold for high confidence (default: 0.8)
```

### Output Options

```bash
# Output directory and format
--output PATH               # Output directory (default: ./results)
--format FORMAT             # Output format (html, json, csv, all)

# Report customization
--detailed-annotations      # Include detailed variant annotations
--include-networks          # Generate protein interaction networks
--include-pathways          # Include pathway enrichment analysis
--generate-plots            # Create visualization plots

# File naming
--prefix STRING             # Prefix for output files
--timestamp                 # Add timestamp to output files
```

### Performance Options

```bash
# Computational resources
--threads N                 # Number of CPU cores (default: auto)
--memory-limit SIZE         # Memory limit (e.g., "16GB")
--cache-dir PATH            # Directory for caching (default: ~/.gospel/cache)

# Processing mode
--streaming                 # Stream large VCF files
--chunk-size N              # Process variants in chunks of size N
--parallel-domains          # Process domains in parallel
```

### Examples

#### Basic Analysis

```bash
# Simple analysis with default settings
gospel analyze --vcf sample.vcf --output basic_analysis/

# Analysis with quality filters
gospel analyze --vcf sample.vcf \
    --min-quality 50 \
    --min-depth 20 \
    --max-allele-freq 0.01 \
    --output high_quality_analysis/
```

#### Domain-Specific Analysis

```bash
# Fitness domain only
gospel analyze --vcf athlete.vcf \
    --domains fitness \
    --include-regulatory \
    --output fitness_profile/

# Pharmacogenetics with drug focus
gospel analyze --vcf patient.vcf \
    --domains pharmacogenetics \
    --drugs "warfarin,clopidogrel,simvastatin" \
    --output pharma_analysis/

# Nutritional genomics
gospel analyze --vcf genome.vcf \
    --domains nutrition \
    --nutrients "folate,vitamin_d,caffeine" \
    --output nutrition_profile/
```

#### Advanced Analysis

```bash
# Comprehensive analysis with all features
gospel analyze --vcf genome.vcf \
    --domains all \
    --include-regulatory \
    --include-structural \
    --include-cnvs \
    --include-networks \
    --include-pathways \
    --population EUR \
    --min-confidence 0.7 \
    --generate-plots \
    --format all \
    --threads 8 \
    --output comprehensive_analysis/
```

#### Family Analysis

```bash
# Trio analysis (parents + child)
gospel analyze --vcf family.vcf \
    --pedigree family.ped \
    --inheritance-mode recessive \
    --domains all \
    --output family_analysis/
```

## Query Command

The `query` command provides AI-powered exploration of genomic analysis results.

### Basic Syntax

```bash
gospel query [OPTIONS]
```

### Input Sources

```bash
# Query analysis results
--results PATH              # Directory containing analysis results
--vcf PATH                  # Direct VCF file query
--variant VARIANT           # Specific variant (e.g., "rs1234567")

# Knowledge base query
--kb-dir PATH               # Query knowledge base directly
--pubmed-search             # Search PubMed for additional context
```

### Query Modes

```bash
# Interactive mode
--interactive               # Start interactive query session

# Single query mode
--query "QUESTION"          # Ask specific question

# Batch query mode
--query-file PATH           # File containing multiple queries
```

### AI Model Options

```bash
# Model selection
--model MODEL_NAME          # Specify AI model (default: llama3)
--temperature FLOAT         # Model temperature (default: 0.1)
--max-tokens N              # Maximum response tokens (default: 2000)

# Context options
--include-literature        # Include literature context
--include-pathways          # Include pathway information
--include-population        # Include population genetics context
```

### Output Options

```bash
# Response format
--format FORMAT             # Response format (text, json, markdown)
--save-session PATH         # Save query session to file
--export-results PATH       # Export all responses to file
```

### Example Queries

#### Interactive Mode

```bash
# Start interactive session
gospel query --interactive --results analysis_results/

# Example session:
> What are my genetic advantages for endurance sports?
> Which medications should I be cautious about?
> How does my MTHFR variant affect folate metabolism?
> Show me genes connected to ACTN3 in my network
> What supplements might benefit my genetic profile?
```

#### Direct Queries

```bash
# Specific genetic question
gospel query --results analysis/ \
    --query "What does my APOE genotype mean for cardiovascular health?"

# Drug interaction query
gospel query --results analysis/ \
    --query "Is it safe for me to take warfarin based on my genetics?"

# Training optimization
gospel query --results fitness_analysis/ \
    --query "What type of training would be most effective for my genetic profile?"
```

#### Batch Queries

```bash
# Create query file
cat > queries.txt << EOF
What are my top 5 genetic risk factors?
Which domains show the highest scores?
What lifestyle modifications are recommended?
Are there any drug-gene interactions I should know about?
EOF

# Run batch queries
gospel query --results analysis/ --query-file queries.txt
```

## Visualize Command

Generate comprehensive visualizations of genomic analysis results.

### Basic Syntax

```bash
gospel visualize --results RESULTS_DIR [OPTIONS]
```

### Input Options

```bash
# Source data
--results PATH              # Analysis results directory
--variants PATH             # Variant data file
--scores PATH               # Score data file
--networks PATH             # Network data file
```

### Visualization Types

```bash
# Chart types
--score-distributions       # Domain score distributions
--variant-impacts           # Variant impact plots
--pathway-enrichment        # Pathway enrichment charts
--population-comparisons    # Population frequency comparisons

# Network visualizations
--protein-networks          # Protein interaction networks
--pathway-networks          # Biological pathway networks
--cross-domain-networks     # Cross-domain gene networks

# Specialized plots
--fitness-radar             # Fitness profile radar chart
--pharma-heatmap            # Pharmacogenetic heatmap
--nutrition-wheel           # Nutritional requirements wheel
```

### Output Options

```bash
# Output settings
--output PATH               # Output directory
--format FORMAT             # Image format (png, svg, pdf, html)
--resolution N              # Image resolution (DPI)
--theme THEME               # Visualization theme (light, dark, publication)

# Interactive features
--interactive               # Generate interactive HTML plots
--include-tooltips          # Add detailed tooltips
--enable-zoom               # Enable plot zooming
```

### Example Visualizations

```bash
# Basic visualization suite
gospel visualize --results analysis/ \
    --score-distributions \
    --variant-impacts \
    --protein-networks \
    --output charts/

# Publication-quality figures
gospel visualize --results analysis/ \
    --score-distributions \
    --pathway-enrichment \
    --format pdf \
    --resolution 300 \
    --theme publication \
    --output figures/

# Interactive web report
gospel visualize --results analysis/ \
    --interactive \
    --include-tooltips \
    --enable-zoom \
    --format html \
    --output web_report/
```

## Knowledge Base Commands

Manage Gospel's scientific knowledge base for enhanced AI queries.

### Build Knowledge Base

```bash
gospel kb build --pdf-dir PDFS/ --output-dir KB/ [OPTIONS]
```

#### Options

```bash
# Input sources
--pdf-dir PATH              # Directory containing PDF papers
--pubmed-ids FILE           # File with PubMed IDs to download
--text-dir PATH             # Directory with text files

# Processing options
--model MODEL_NAME          # Model for text processing (default: llama3)
--chunk-size N              # Text chunk size (default: 1000)
--overlap N                 # Chunk overlap (default: 200)

# Output options
--output-dir PATH           # Knowledge base output directory
--index-name NAME           # Vector index name
--metadata-format FORMAT    # Metadata format (json, csv)
```

#### Examples

```bash
# Build from PDF collection
gospel kb build \
    --pdf-dir research_papers/ \
    --output-dir knowledge_base/ \
    --model llama3

# Build with custom parameters
gospel kb build \
    --pdf-dir papers/ \
    --pubmed-ids pubmed_list.txt \
    --output-dir kb/ \
    --chunk-size 1500 \
    --overlap 300
```

### Query Knowledge Base

```bash
gospel kb query --kb-dir KB_DIR --query "QUESTION" [OPTIONS]
```

#### Options

```bash
# Query parameters
--kb-dir PATH               # Knowledge base directory
--query STRING              # Query string
--top-k N                   # Number of results (default: 5)
--similarity-threshold FLOAT # Minimum similarity (default: 0.7)

# Output options
--include-sources           # Include source citations
--format FORMAT             # Output format (text, json)
```

#### Examples

```bash
# Query specific topic
gospel kb query \
    --kb-dir knowledge_base/ \
    --query "ACTN3 variants and sprint performance" \
    --top-k 10

# Query with citations
gospel kb query \
    --kb-dir kb/ \
    --query "CYP2D6 pharmacogenetics" \
    --include-sources \
    --format json
```

## LLM Commands

Train and deploy domain-specific language models.

### Train Domain Model

```bash
gospel llm train --kb-dir KB_DIR --output-dir MODEL_DIR [OPTIONS]
```

#### Options

```bash
# Training data
--kb-dir PATH               # Knowledge base directory
--base-model MODEL          # Base model to fine-tune
--training-examples PATH    # Additional training examples

# Training parameters
--epochs N                  # Training epochs (default: 3)
--learning-rate FLOAT       # Learning rate (default: 1e-5)
--batch-size N              # Batch size (default: 4)

# Output options
--output-dir PATH           # Model output directory
--model-name NAME           # Custom model name
```

#### Examples

```bash
# Train fitness-focused model
gospel llm train \
    --kb-dir fitness_kb/ \
    --output-dir fitness_model/ \
    --base-model llama3 \
    --epochs 5

# Train pharmacogenetics model
gospel llm train \
    --kb-dir pharma_kb/ \
    --output-dir pharma_model/ \
    --base-model mistral \
    --learning-rate 2e-5
```

### Query Domain Model

```bash
gospel llm query --model-dir MODEL_DIR [OPTIONS]
```

#### Options

```bash
# Model parameters
--model-dir PATH            # Trained model directory
--temperature FLOAT         # Sampling temperature
--max-tokens N              # Maximum response tokens

# Query options
--query STRING              # Single query
--interactive               # Interactive mode
--context PATH              # Additional context file
```

#### Examples

```bash
# Single query
gospel llm query \
    --model-dir fitness_model/ \
    --query "Optimize training for ACTN3 RX genotype"

# Interactive session
gospel llm query \
    --model-dir pharma_model/ \
    --interactive
```

## Configuration

Gospel uses YAML configuration files for customizing analysis parameters.

### Configuration File Structure

```yaml
# ~/.gospel/config.yaml
database:
  path: ~/.gospel/databases
  cache_size: 1000MB
  update_frequency: weekly

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
    
    thresholds:
      high_confidence: 0.8
      medium_confidence: 0.6
      low_confidence: 0.4

domains:
  fitness:
    focus_traits: [sprint, endurance, power, recovery]
    include_injury_risk: true
    training_recommendations: true
  
  pharmacogenetics:
    drug_classes: [cardiovascular, psychiatric, oncology, pain]
    include_dosing: true
    include_interactions: true
  
  nutrition:
    include_sensitivities: true
    supplement_recommendations: true
    diet_optimization: true

output:
  default_format: html
  include_visualizations: true
  detailed_annotations: true
  compress_results: false

ai:
  default_model: llama3
  temperature: 0.1
  max_tokens: 2000
  include_literature_context: true

performance:
  threads: auto
  memory_limit: 8GB
  cache_enabled: true
  parallel_domains: true
```

### Environment Variables

```bash
# Core settings
export GOSPEL_CONFIG_DIR=~/.gospel
export GOSPEL_DATABASE_PATH=~/.gospel/databases
export GOSPEL_CACHE_DIR=~/.gospel/cache

# AI model settings
export OLLAMA_HOST=localhost:11434
export GOSPEL_MODEL=llama3

# Performance settings
export GOSPEL_THREADS=8
export GOSPEL_MEMORY_LIMIT=16GB
```

## Workflow Examples

### Complete Analysis Workflow

```bash
#!/bin/bash
# complete_analysis.sh

# 1. Run comprehensive analysis
gospel analyze \
    --vcf genome.vcf \
    --domains all \
    --include-regulatory \
    --include-networks \
    --population EUR \
    --output analysis_results/ \
    --format all

# 2. Generate visualizations
gospel visualize \
    --results analysis_results/ \
    --interactive \
    --output visualizations/

# 3. Explore results interactively
gospel query \
    --interactive \
    --results analysis_results/
```

### Athlete Performance Analysis

```bash
#!/bin/bash
# athlete_analysis.sh

# Focus on fitness domain with enhanced features
gospel analyze \
    --vcf athlete_genome.vcf \
    --domains fitness \
    --include-regulatory \
    --include-networks \
    --training-optimization \
    --injury-risk-assessment \
    --output athlete_profile/

# Generate sport-specific recommendations
gospel query \
    --results athlete_profile/ \
    --query "What sports and training methods suit my genetic profile?" \
    --include-literature

# Create athlete report visualizations
gospel visualize \
    --results athlete_profile/ \
    --fitness-radar \
    --training-recommendations \
    --format pdf \
    --output athlete_report/
```

### Clinical Pharmacogenetics Workflow

```bash
#!/bin/bash
# clinical_pharma.sh

# Pharmacogenetic analysis for clinical use
gospel analyze \
    --vcf patient.vcf \
    --domains pharmacogenetics \
    --clinical-guidelines \
    --drug-interactions \
    --dosing-recommendations \
    --output pharma_analysis/

# Generate clinical report
gospel query \
    --results pharma_analysis/ \
    --query "Provide clinical pharmacogenetic recommendations" \
    --format clinical-report \
    --output clinical_pharma_report.pdf

# Create pharmacist reference
gospel visualize \
    --results pharma_analysis/ \
    --pharma-heatmap \
    --drug-response-table \
    --format html \
    --output pharmacist_reference/
```

### Research Cohort Analysis

```bash
#!/bin/bash
# cohort_analysis.sh

# Process multiple samples in parallel
for vcf in cohort/*.vcf; do
    sample=$(basename "$vcf" .vcf)
    gospel analyze \
        --vcf "$vcf" \
        --domains all \
        --population EUR \
        --output "cohort_results/$sample/" &
done
wait

# Aggregate results
gospel aggregate \
    --input-dir cohort_results/ \
    --output cohort_summary/ \
    --generate-statistics

# Perform population analysis
gospel population-analysis \
    --cohort-dir cohort_results/ \
    --output population_genetics/ \
    --include-gwas
```

---

This comprehensive CLI reference provides all the tools needed to leverage Gospel's full genomic analysis capabilities. For specific use cases and detailed examples, see the [Examples](examples.html) section.

**Next:** Explore the [API Reference](api-reference.html) for programmatic usage or check out [Examples](examples.html) for real-world scenarios. 