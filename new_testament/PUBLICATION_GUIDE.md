# Publication Guide: Independent Component Analysis

This guide shows how to run each framework component independently for publication-ready results. Each script is self-contained with its own main function, comprehensive visualizations, and detailed reporting.

## 🧬 Component Overview

The framework consists of **publication-ready modules** that can be executed both independently and as part of the integrated system:

| Component | Script | Focus | Output Files |
|-----------|--------|-------|--------------|
| **Genome File Parser** | `parse_genome.py` | FASTA/VCF parsing and sequence extraction | 2 publication figures + JSON + FASTA |
| **Cardinal Coordinate Transform** | `coordinate_transform.py` | DNA sequences → 2D geometric paths | 3 publication figures + JSON + report |
| **Pharmaceutical Response** | `pharmaceutical_response.py` | Drug efficacy prediction via oscillatory matching | 4 publication figures + JSON + report |
| **Gene Oscillator Circuits** | `genomic_oscillators.py` | Gene regulatory networks as oscillatory circuits | 3 publication figures + JSON + report |
| **Intracellular Bayesian Networks** | `intracellular_bayesian.py` | Cellular dynamics as Bayesian evidence processing | 3 publication figures + JSON + report |
| **VCF Analysis Pipeline** | `dante_labs_vcf_analyzer.py` | Complete genomic variant analysis | Full VCF analysis with multi-panel figures |
| **Multi-Framework Integration** | `multi_framework_integrator.py` | Cross-validation across all frameworks | Integration analysis with confidence scoring |

## 🚀 Quick Start: Independent Execution

### 0. Using Your FASTA and VCF Files

**NEW: Use your actual genomic data!**

```bash
# Quick test with your files
cd new_testament
python test_with_fasta.py

# Complete analysis pipeline
python comprehensive_sequence_demo.py --use-vcf

# Integrated analysis (sequence + VCF + pharmaceutical)
python integrated_genomic_demo.py --deep-analysis
```

### 1. Genome File Parsing

**Purpose**: Extract sequences from your FASTA files and integrate VCF variants

```bash
cd new_testament/src/st_stellas/sequence
python parse_genome.py --output ./genome_results/ --visualize
```

**Auto-detects**:
- FASTA files in `new_testament/public/fasta/`
- VCF files in `new_testament/public/`
- Focuses on chromosome 21 by default

**Generated Files**:
- `genome_statistics.png` - Sequence composition and variant analysis
- `sequence_analysis.png` - Length distributions and GC content
- `genome_parser_results.json` - Complete parsing data
- `extracted_sequences.fasta` - Sequences ready for analysis

### 2. Cardinal Coordinate Transformation

**Purpose**: Transform DNA sequences into 2D coordinate paths for geometric analysis

```bash
cd new_testament/src/st_stellas/sequence
python coordinate_transform.py --benchmark --output ./coordinate_results/
```

**Generated Files**:
- `coordinate_paths_analysis.png` - 2D path visualizations
- `sequence_statistics.png` - Base composition and performance
- `performance_benchmarks.png` - Algorithm scaling analysis
- `coordinate_transform_analysis.json` - Complete transformation data
- `coordinate_transform_report.md` - Comprehensive analysis report

**Key Features**:
- Cardinal mapping: A→North, T→South, G→East, C→West
- Numba JIT acceleration (install with `pip install numba`)
- Performance benchmarking up to 5000+ sequences/second
- Path complexity and geometric analysis

### 3. Pharmaceutical Response Analysis

**Purpose**: Analyze drug efficacy through oscillatory hole-filling theory

```bash
cd new_testament/src/st_stellas/genome
python pharmaceutical_response.py --output ./pharma_results/ --save-json --visualize
```

**Generated Files**:
- `pharmaceutical_efficacy_analysis.png` - 4-panel efficacy analysis
- `oscillatory_mechanism_analysis.png` - Mechanism and frequency analysis
- `drug_resonance_quality_analysis.png` - Performance matrix heatmap
- `pharmaceutical_response_analysis.json` - Complete raw data
- `pharmaceutical_analysis_report.md` - Comprehensive report

**Key Metrics**:
- Information Catalytic Efficiency (ηIC)
- Therapeutic Amplification Factors
- Oscillatory Resonance Quality
- Drug-by-drug recommendations

### 2. Gene Oscillator Circuit Analysis

**Purpose**: Model gene regulatory networks as oscillatory circuits

```bash
cd new_testament/src/st_stellas/genome
python genomic_oscillators.py --n-genes 100 --output ./gene_results/ --save-json --visualize
```

**Generated Files**:
- `gene_oscillator_network.png` - Network graph with frequency/amplitude mapping
- `frequency_distribution_analysis.png` - Statistical distribution analysis
- `circuit_topology_analysis.png` - Network topology and connectivity
- `gene_oscillator_analysis.json` - Complete circuit data
- `gene_oscillator_analysis_report.md` - Circuit analysis report

**Key Metrics**:
- Gene oscillatory frequencies (0.1-100 Hz range)
- Regulatory coupling strengths
- Network topology (clustering coefficient, hub genes)
- Resonance pattern identification

### 3. Intracellular Bayesian Network Analysis

**Purpose**: Model cellular dynamics as Bayesian evidence networks

```bash
cd new_testament/src/st_stellas/genome
python intracellular_bayesian.py --n-cells 200 --output ./cellular_results/ --save-json --visualize
```

**Generated Files**:
- `bayesian_network_performance.png` - Network accuracy and ATP cost analysis
- `atp_efficiency_analysis.png` - Metabolic efficiency by cellular condition
- `glycolysis_pathway_analysis.png` - Pathway-specific Bayesian analysis
- `intracellular_bayesian_analysis.json` - Complete cellular data
- `intracellular_analysis_report.md` - Cellular dynamics report

**Key Metrics**:
- Bayesian network accuracy
- ATP cost per decision (mM)
- Glycolysis pathway efficiency
- Evidence processing capacity (bits/s)

## 📊 Publication-Quality Visualization Features

Each component generates **publication-ready visualizations** with:

### Standard Features
- **300 DPI resolution** for high-quality printing
- **Multi-panel layouts** (2×2, 1×2 formats)
- **Proper axis labeling** with units and statistical annotations
- **Color-coded data** with meaningful legends
- **Grid overlays** for precise reading
- **Statistical summaries** (means, correlations, significance)

### Advanced Features
- **Network visualizations** with NetworkX integration
- **Correlation heatmaps** with significance testing
- **Distribution histograms** with statistical overlays
- **Performance matrices** with normalized scoring
- **Time series analysis** (where applicable)

## 🔬 Academic Research Workflows

### For Drug Discovery Research

```bash
# 1. Analyze pharmaceutical mechanisms
python pharmaceutical_response.py --output ./drug_discovery_results/

# 2. Examine gene circuit impacts
python genomic_oscillators.py --n-genes 200 --output ./gene_circuit_analysis/

# 3. Study cellular ATP efficiency
python intracellular_bayesian.py --n-cells 500 --output ./cellular_efficiency/
```

### For Genomic Research

```bash
# 1. Analyze gene oscillator networks
python genomic_oscillators.py --n-genes 1000 --output ./large_network_study/

# 2. Study pharmaceutical targeting
python pharmaceutical_response.py --output ./pharma_genomics/

# 3. Examine cellular integration
python intracellular_bayesian.py --n-cells 1000 --output ./population_study/
```

### For Systems Biology Research

```bash
# Complete multi-scale analysis
python genomic_oscillators.py --n-genes 500 --output ./systems_biology/genes/
python intracellular_bayesian.py --n-cells 300 --output ./systems_biology/cells/
python pharmaceutical_response.py --output ./systems_biology/drugs/
```

## 📈 Data Export Formats

### JSON Data Structure
Each component exports structured JSON with:
```json
{
  "analysis_metadata": {
    "timestamp": "2024-10-10T15:30:00",
    "framework_version": "1.0.0",
    "analysis_type": "component_specific"
  },
  "results": {
    "primary_metrics": {},
    "statistical_analysis": {},
    "population_statistics": {}
  }
}
```

### Report Structure
Each markdown report includes:
- **Executive Summary** with key findings
- **Statistical Analysis** with population metrics
- **Theoretical Framework** with mathematical foundations
- **Individual Component Analysis** (drug-by-drug, gene-by-gene, etc.)
- **Clinical/Research Implications**
- **Files Generated** summary

## 🔧 Advanced Configuration

### Custom Parameter Analysis

**Pharmaceutical Response**:
```bash
# Analyze specific drug classes
python pharmaceutical_response.py --output ./nsaids_analysis/ --visualize

# Export data for external analysis
python pharmaceutical_response.py --save-json --output ./data_export/
```

**Gene Oscillator Circuits**:
```bash
# Large-scale network analysis
python genomic_oscillators.py --n-genes 2000 --output ./large_networks/

# Focus on specific gene families
python genomic_oscillators.py --n-genes 100 --output ./cyp450_analysis/
```

**Intracellular Bayesian**:
```bash
# Population-scale analysis
python intracellular_bayesian.py --n-cells 1000 --output ./population_study/

# Focus on cellular conditions
python intracellular_bayesian.py --n-cells 500 --output ./disease_conditions/
```

## 🧪 Integration with Complete VCF Pipeline

For complete genomic analysis integration:

```bash
# 1. Run VCF analysis
python dante_labs_demo.py --vcf genome.vcf.gz --integrate-all --output ./complete_analysis/

# 2. Extract specific components for detailed analysis
python genomic_oscillators.py --output ./complete_analysis/detailed_genes/
python pharmaceutical_response.py --output ./complete_analysis/detailed_drugs/
python intracellular_bayesian.py --output ./complete_analysis/detailed_cells/
```

## 📖 Publication Citation

When using individual components in publications, cite as:

```bibtex
@software{st_stellas_component_2024,
  title={St. Stella's [Component Name] Analysis},
  author={Sachikonye, Kundai Farai},
  year={2024},
  institution={Technical University of Munich},
  url={https://github.com/fullscreen-triangle/gospel/tree/main/new_testament},
  note={Component of the St. Stella's Computational Pharmacology Framework}
}
```

## 🔍 Debugging and Validation

### Progress Monitoring
Each script provides detailed progress output:
```
[1/5] Generating simulated data...
[2/5] Analyzing components...
  ✓ Component 1: metric_value
  ✓ Component 2: metric_value
[3/5] Computing statistics...
[4/5] Saving results...
[5/5] Generating visualizations...
✅ Analysis complete!
```

### Output Validation
Verify results with:
```bash
# Check JSON structure
python -m json.tool results.json

# Verify image generation
ls -la *.png

# Review report completeness
wc -l *_report.md
```

## 🎯 Research Applications

### Drug Discovery
- **Pharmaceutical screening** through oscillatory matching
- **Mechanism of action** analysis via frequency patterns
- **Therapeutic index** optimization using ηIC calculations

### Genomics Research
- **Regulatory network** topology analysis
- **Gene expression** oscillation characterization
- **Evolutionary conservation** of oscillatory patterns

### Systems Biology
- **Multi-scale integration** across molecular to cellular levels
- **Information flow** analysis in biological networks
- **Computational efficiency** of biological processes

---

**Each component tells its own scientific story while contributing to the complete computational pharmacology framework. This modular design enables focused research while maintaining theoretical coherence across scales.**

*For complete VCF analysis integration, see the main `dante_labs_demo.py` documentation.*
