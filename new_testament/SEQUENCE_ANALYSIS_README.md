# St. Stella's Sequence Analysis Modules

## 🧬 Using Your FASTA and VCF Files

You now have **publication-ready sequence analysis modules** that work with your downloaded FASTA files and personal VCF data!

### 🚀 Quick Start

#### Test with your files:
```bash
cd new_testament
python test_with_fasta.py
```

#### Complete analysis pipeline:
```bash
python comprehensive_sequence_demo.py --use-vcf
```

#### Individual module analysis:
```bash
# Parse your FASTA files
python src/st_stellas/sequence/parse_genome.py --output ./my_analysis/

# Transform sequences to cardinal coordinates
python src/st_stellas/sequence/coordinate_transform.py --benchmark --output ./coordinates/
```

### 📁 Your Available Data

Based on your `new_testament/public/` directory:

**FASTA Files** (Reference Genome):
- `Homo_sapiens.GRCh38.dna.chromosome.1.fa` - Full chromosome 1
- `Homo_sapiens.GRCh38.dna.chromosome.21.fa` - Chromosome 21
- `Homo_sapiens.GRCh38.dna.chromosome.22.fa` - Chromosome 22
- `Homo_sapiens.GRCh38.dna.alt.fa` - Alternative sequences

**VCF Files** (Your Personal Genome):
- `GFX0436892.filtered.snp.vcf.gz` - Your SNP variants
- `GFX0436892.filtered.indel.vcf.gz` - Your indel variants
- `GFX0436892.cnv.vcf.gz` - Copy number variants

### 🔬 Publication-Ready Components

#### 1. **Genome Parser** (`parse_genome.py`)
- **Auto-detects** your FASTA and VCF files
- **Extracts sequences** from reference genome
- **Integrates variants** from your personal genome
- **Generates visualizations**: Base composition, variant distributions
- **Saves data**: FASTA format for downstream analysis

```bash
python src/st_stellas/sequence/parse_genome.py \
    --fasta public/fasta/Homo_sapiens.GRCh38.dna.chromosome.21.fa \
    --vcf public/GFX0436892.filtered.snp.vcf.gz \
    --chromosome 21 \
    --output ./chr21_analysis/
```

#### 2. **Cardinal Coordinate Transformer** (`coordinate_transform.py`)
- **Transforms DNA → 2D paths**: A→North, T→South, G→East, C→West
- **Performance benchmarks** with Numba acceleration
- **Publication figures**: Path visualizations, statistics
- **Mathematical analysis**: Path complexity, final positions

```bash
python src/st_stellas/sequence/coordinate_transform.py \
    --input ./chr21_analysis/extracted_sequences.fasta \
    --benchmark \
    --output ./coordinate_analysis/
```

### 📊 Generated Outputs

Each module creates **publication-ready results**:

**Data Files:**
- `*.json` - Structured analysis data
- `*.fasta` - Extracted sequences for further analysis
- `*.txt` - Sample sequences for testing

**Visualizations (300 DPI):**
- `genome_statistics.png` - Sequence composition analysis
- `coordinate_paths_analysis.png` - 2D transformation results
- `performance_benchmarks.png` - Algorithm scaling analysis

**Reports:**
- `*_report.md` - Comprehensive analysis documentation

### 🔗 Integration Flow

```
Your FASTA Files → Genome Parser → Sequence Extractor
                      ↓
                Cardinal Coordinates → Path Analysis → Geometric Properties
                      ↓
        [Future Integration with Genome Analysis Modules]
                      ↓
            VCF Variants → Pharmaceutical Predictions → Clinical Insights
```

### 🧪 Example Workflows

#### **Chromosome-Specific Analysis:**
```bash
# Focus on chromosome 21
python comprehensive_sequence_demo.py \
    --chromosome 21 \
    --n-sequences 500 \
    --use-vcf \
    --output ./chr21_complete/
```

#### **Performance Benchmarking:**
```bash
# Test algorithm performance
python src/st_stellas/sequence/coordinate_transform.py \
    --n-sequences 1000 \
    --benchmark \
    --output ./performance_test/
```

#### **VCF-Focused Analysis:**
```bash
# Emphasize your personal variants
python src/st_stellas/sequence/parse_genome.py \
    --vcf public/GFX0436892.filtered.snp.vcf.gz \
    --vcf public/GFX0436892.filtered.indel.vcf.gz \
    --output ./personal_variants/
```

### 🔧 Available Modules

From your `src/st_stellas/sequence/` directory:

| Module | Status | Purpose |
|--------|--------|---------|
| `parse_genome.py` | ✅ **Publication Ready** | FASTA/VCF parsing & integration |
| `coordinate_transform.py` | ✅ **Publication Ready** | Cardinal coordinate transformation |
| `pattern_extractor.py` | 🔧 Development | Extract recurring genomic patterns |
| `genomic_oscillatory_patterns.py` | 🔧 Development | Frequency-based pattern analysis |
| `s_entropy_navigator.py` | 🔧 Development | Entropy-based sequence navigation |
| `dual_strand_analyzer.py` | 🔧 Development | Bidirectional strand analysis |

### 🎯 Research Applications

#### **Immediate Use:**
- **Sequence Pattern Discovery**: Novel motif identification
- **Compositional Bias Analysis**: Directional trends in your genome
- **Performance Optimization**: Algorithm scaling for large datasets
- **Visualization Generation**: Publication-quality figures

#### **Integration with Genome Modules:**
- **Pharmaceutical Response**: Your variants → Drug predictions
- **Gene Networks**: Sequence patterns → Regulatory circuits
- **Cellular Analysis**: Genomic → Intracellular dynamics
- **Multi-Framework**: Connection to Nebuchadnezzar, Borgia, Bene Gesserit, Hegel

### 💡 Tips for Publication

1. **Run with `--benchmark`** for performance analysis
2. **Use `--visualize`** for high-resolution figures
3. **Save JSON data** for reproducibility
4. **Generate reports** for comprehensive documentation
5. **Test different chromosomes** for comparative analysis

### 🚨 Troubleshooting

**No FASTA files found:**
- Check `new_testament/public/fasta/` directory
- Ensure files end with `.fa` or `.fa.gz`

**VCF parsing errors:**
- Use uncompressed `.vcf` files if `.vcf.gz` fails
- Focus on SNP files first: `*snp.vcf`

**Import errors:**
- Run from `new_testament/` directory
- Install missing dependencies: `pip install -e .`

**Performance issues:**
- Install numba for acceleration: `pip install numba`
- Reduce `--n-sequences` for testing

---

**🎉 Your genomic data is now ready for St. Stella's analysis!**

*Use `python test_with_fasta.py` to get started immediately.*
