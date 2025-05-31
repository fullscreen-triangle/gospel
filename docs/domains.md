---
layout: page
title: Domain Analysis
permalink: /domains/
---

# Genomic Domain Analysis

Gospel's multi-domain approach provides comprehensive analysis across three key areas of genomics that directly impact human health and performance. Each domain leverages sophisticated scientific understanding to translate genetic variants into actionable insights.

## Table of Contents
1. [Domain Overview](#domain-overview)
2. [Fitness Domain](#fitness-domain)
3. [Pharmacogenetics Domain](#pharmacogenetics-domain)
4. [Nutritional Genomics Domain](#nutritional-genomics-domain)
5. [Cross-Domain Interactions](#cross-domain-interactions)

## Domain Overview

### Scientific Foundation

Each domain in Gospel is built upon decades of genomic research and validated through extensive literature review:

| Domain | Key Concepts | Primary Applications | Evidence Base |
|--------|--------------|---------------------|---------------|
| **Fitness** | Exercise physiology, muscle composition, energy metabolism | Athletic performance, training optimization, injury prevention | 2,000+ peer-reviewed studies |
| **Pharmacogenetics** | Drug metabolism, receptor sensitivity, transport proteins | Medication selection, dosing optimization, adverse reaction prevention | 5,000+ clinical studies |
| **Nutrition** | Nutrient metabolism, food sensitivities, dietary requirements | Personalized nutrition, supplement optimization, metabolic health | 3,500+ research papers |

### Integration Strategy

Gospel employs a sophisticated integration strategy that considers:

- **Gene-gene interactions**: How variants in multiple genes combine to influence phenotypes
- **Domain overlap**: Genes that affect multiple domains (e.g., inflammation pathways)
- **Environmental modulation**: How lifestyle factors modify genetic effects
- **Population-specific effects**: Ancestry-based variant interpretation

## Fitness Domain

The fitness domain analyzes genetic factors influencing athletic performance, exercise response, and injury susceptibility across multiple physiological systems.

### Core Fitness Components

#### 1. Power and Sprint Performance

**Key Genes and Variants:**

- **ACTN3 (α-Actinin-3)**
  - *Variant*: rs1815739 (R577X)
  - *Mechanism*: Fast-twitch muscle fiber composition
  - *Impact*: RR genotype associated with elite sprint performance
  - *Population frequency*: 18% RR (European), 25% RR (African), 5% RR (Asian)

- **PPARA (Peroxisome Proliferator-Activated Receptor Alpha)**
  - *Variant*: rs4253778 (G/C)
  - *Mechanism*: Fat oxidation and energy metabolism
  - *Impact*: GG genotype enhances power output capacity

- **AGT (Angiotensinogen)**
  - *Variant*: rs699 (M235T)
  - *Mechanism*: Blood pressure regulation during exercise
  - *Impact*: TT genotype associated with power athlete status

**Analysis Framework:**

```python
class SprintAnalyzer:
    def analyze_sprint_potential(self, variants: List[Variant]) -> SprintProfile:
        # Core sprint genes with evidence weights
        sprint_genes = {
            'ACTN3': {'weight': 0.35, 'variants': ['rs1815739']},
            'PPARA': {'weight': 0.25, 'variants': ['rs4253778']}, 
            'AGT': {'weight': 0.20, 'variants': ['rs699']},
            'NOS3': {'weight': 0.20, 'variants': ['rs2070744']}
        }
        
        sprint_score = self.calculate_composite_score(variants, sprint_genes)
        return SprintProfile(
            overall_score=sprint_score,
            fiber_type_bias=self.assess_fiber_type(variants),
            power_potential=self.calculate_power_score(variants),
            recommended_training=self.generate_training_recommendations(variants)
        )
```

#### 2. Endurance Performance

**Key Genes and Variants:**

- **PPARGC1A (PGC-1α)**
  - *Variant*: rs8192678 (Gly482Ser)
  - *Mechanism*: Mitochondrial biogenesis and oxidative metabolism
  - *Impact*: GG genotype associated with enhanced endurance capacity

- **EPAS1 (Endothelial PAS Domain Protein 1)**
  - *Variant*: rs4953354
  - *Mechanism*: High-altitude adaptation and oxygen utilization
  - *Impact*: AA genotype improves oxygen efficiency

- **ACE (Angiotensin-Converting Enzyme)**
  - *Variant*: rs1799752 (I/D polymorphism)
  - *Mechanism*: Cardiovascular efficiency and blood flow
  - *Impact*: II genotype favors endurance performance

**Endurance Analysis Pipeline:**

```python
def analyze_endurance_capacity(variants):
    endurance_pathways = {
        'mitochondrial_function': {
            'genes': ['PPARGC1A', 'NRF1', 'TFAM'],
            'weight': 0.4
        },
        'oxygen_transport': {
            'genes': ['EPAS1', 'HIF1A', 'VEGFA'],
            'weight': 0.3
        },
        'cardiovascular_efficiency': {
            'genes': ['ACE', 'NOS3', 'BDKRB2'],
            'weight': 0.3
        }
    }
    
    return calculate_pathway_scores(variants, endurance_pathways)
```

#### 3. Recovery and Injury Risk

**Recovery Genetics:**

- **IL6 (Interleukin-6)**
  - *Variant*: rs1800795 (-174 G/C)
  - *Mechanism*: Inflammatory response and muscle repair
  - *Impact*: GG genotype associated with faster recovery

- **IGF1 (Insulin-like Growth Factor 1)**
  - *Variant*: rs35767 (CA repeat)
  - *Mechanism*: Muscle growth and repair
  - *Impact*: Longer repeats enhance recovery capacity

**Injury Susceptibility:**

- **COL1A1 (Collagen Type I Alpha 1)**
  - *Variant*: rs1800012 (G1245T)
  - *Mechanism*: Collagen structure and strength
  - *Impact*: TT genotype increases soft tissue injury risk

- **COL5A1 (Collagen Type V Alpha 1)**
  - *Variant*: rs12722 (C/T)
  - *Mechanism*: Tendon and ligament integrity
  - *Impact*: CC genotype associated with reduced injury risk

### Fitness Domain Scoring

The fitness domain employs a multi-component scoring system:

$$Score_{fitness} = \sum_{i=1}^{5} w_i \cdot Component_i$$

**Where components are:**
1. **Sprint/Power** ($w_1 = 0.25$): Fast-twitch muscle capacity
2. **Endurance** ($w_2 = 0.25$): Aerobic capacity and efficiency  
3. **Strength** ($w_3 = 0.20$): Maximum force generation
4. **Recovery** ($w_4 = 0.15$): Post-exercise adaptation rate
5. **Injury Risk** ($w_5 = 0.15$): Susceptibility to exercise-related injury

### Practical Applications

#### Training Optimization

Gospel provides specific training recommendations based on genetic profile:

```bash
# Example output for power-biased genetic profile
gospel analyze --vcf athlete.vcf --domains fitness --training-focus

# Results:
# Genetic Profile: Power/Sprint Biased (Score: 8.2/10)
# Recommendations:
# - High-intensity interval training (HIIT)
# - Plyometric exercises for explosive power
# - Reduced volume, increased intensity
# - Extended recovery periods (genetic slow recovery)
# - Injury prevention focus on connective tissue
```

#### Sport Selection Guidance

- **Sprint Sports**: Track sprints, weightlifting, gymnastics
- **Endurance Sports**: Marathon, cycling, swimming distance events
- **Mixed Sports**: Soccer, basketball, tennis (require both components)

## Pharmacogenetics Domain

The pharmacogenetics domain analyzes how genetic variants affect drug metabolism, efficacy, and adverse reactions across major drug classes.

### Core Pharmacogenetic Systems

#### 1. Drug Metabolism Enzymes

**Cytochrome P450 System:**

- **CYP2D6 (Cytochrome P450 2D6)**
  - *Key Variants*: *1, *2, *4, *10, *17 (among 100+ alleles)
  - *Mechanism*: Metabolizes 25% of all medications
  - *Drugs*: Codeine, tramadol, metoprolol, antidepressants
  - *Phenotypes*: Poor, intermediate, normal, ultrarapid metabolizers

- **CYP2C19 (Cytochrome P450 2C19)**
  - *Key Variants*: *2, *3, *17
  - *Mechanism*: Proton pump inhibitor and antiplatelet drug metabolism
  - *Drugs*: Clopidogrel, omeprazole, diazepam
  - *Clinical Impact*: *2/*2 genotype requires alternative antiplatelet therapy

- **CYP3A4/5 (Cytochrome P450 3A4/5)**
  - *Key Variants*: *1B, *3, *6
  - *Mechanism*: Metabolizes 50% of all drugs
  - *Drugs*: Statins, immunosuppressants, many chemotherapy agents

**Analysis Framework:**

```python
class CYP2D6Analyzer:
    def predict_metabolizer_status(self, diplotype: str) -> MetabolizerStatus:
        # Activity scores for common CYP2D6 alleles
        activity_scores = {
            '*1': 1.0,    # Normal function
            '*2': 1.0,    # Normal function  
            '*4': 0.0,    # No function
            '*10': 0.25,  # Reduced function
            '*17': 0.5,   # Reduced function
            '*1xN': 2.0,  # Gene duplication
        }
        
        total_score = sum(activity_scores.get(allele, 0) 
                         for allele in diplotype.split('/'))
        
        if total_score == 0:
            return MetabolizerStatus.POOR
        elif total_score < 1.25:
            return MetabolizerStatus.INTERMEDIATE  
        elif total_score <= 2.25:
            return MetabolizerStatus.NORMAL
        else:
            return MetabolizerStatus.ULTRARAPID
```

#### 2. Drug Transport Proteins

**SLCO1B1 (Solute Carrier Organic Anion Transporter)**
- *Variant*: rs4149056 (Val174Ala)
- *Mechanism*: Hepatic uptake of drugs
- *Drugs*: Statins (simvastatin, atorvastatin)
- *Impact*: CC genotype increases statin-induced myopathy risk

**ABCB1 (ATP-Binding Cassette B1)**
- *Variant*: rs1045642 (C3435T)
- *Mechanism*: Drug efflux pump
- *Drugs*: Digoxin, immunosuppressants
- *Impact*: Affects drug bioavailability and CNS penetration

#### 3. Drug Target Receptors

**COMT (Catechol-O-Methyltransferase)**
- *Variant*: rs4680 (Val158Met)
- *Mechanism*: Dopamine metabolism
- *Drugs*: L-DOPA, COMT inhibitors
- *Impact*: Met/Met genotype enhances dopaminergic drug sensitivity

**VKORC1 (Vitamin K Epoxide Reductase Complex Subunit 1)**
- *Variant*: rs9923231 (-1639G>A)
- *Mechanism*: Vitamin K recycling
- *Drugs*: Warfarin, other vitamin K antagonists
- *Impact*: AA genotype requires lower warfarin doses

### Clinical Drug Categories

#### 1. Cardiovascular Medications

**Warfarin Dosing Algorithm:**

$$Dose_{warfarin} = 5.6044 - 0.2614 \times Age + 0.0087 \times Height + 0.0128 \times Weight$$
$$- 0.8750 \times VKORC1_{-1639A} - 1.6974 \times CYP2C9*2 - 3.6175 \times CYP2C9*3$$

**Clopidogrel Response:**
- **CYP2C19**: *1/*1 (normal response), *2 carriers (reduced response)
- **Recommendation**: *2/*2 genotype should receive alternative therapy (prasugrel, ticagrelor)

#### 2. Psychiatric Medications

**Antidepressant Selection:**
```python
def recommend_antidepressant(cyp2d6_status, cyp2c19_status):
    recommendations = {
        ('normal', 'normal'): ['sertraline', 'escitalopram', 'paroxetine'],
        ('poor', 'normal'): ['sertraline', 'escitalopram'],  # Avoid CYP2D6 substrates
        ('normal', 'poor'): ['paroxetine', 'fluoxetine'],    # Avoid CYP2C19 substrates
        ('poor', 'poor'): ['sertraline'],                   # Limited options
        ('ultrarapid', 'normal'): ['higher_dose_paroxetine', 'fluoxetine']
    }
    return recommendations.get((cyp2d6_status, cyp2c19_status), ['consult_specialist'])
```

#### 3. Oncology Medications

**DPYD (Dihydropyrimidine Dehydrogenase):**
- *Variants*: rs3918290, rs55886062, rs67376798
- *Drugs*: 5-fluorouracil, capecitabine
- *Impact*: Variants increase severe toxicity risk - dose reduction required

**TPMT (Thiopurine S-Methyltransferase):**
- *Variants*: *2, *3A, *3B, *3C
- *Drugs*: 6-mercaptopurine, azathioprine
- *Impact*: Poor metabolizers require 90% dose reduction

### Pharmacogenetic Scoring

$$Score_{pharma} = \sum_{d=1}^{D} w_d \cdot \left( \frac{Safe_{d}}{Total_{d}} \right) \times 10$$

**Where:**
- $D$ = Number of drug categories
- $w_d$ = Weight for drug category $d$
- $Safe_d$ = Number of safe drugs in category
- $Total_d$ = Total drugs in category

## Nutritional Genomics Domain

The nutritional genomics domain analyzes how genetic variants affect nutrient metabolism, dietary requirements, and food sensitivities.

### Core Nutritional Components

#### 1. Macronutrient Metabolism

**Fat Metabolism:**

- **FTO (Fat Mass and Obesity-Associated)**
  - *Variant*: rs9939609 (A/T)
  - *Mechanism*: Appetite regulation and energy expenditure
  - *Impact*: AA genotype associated with increased obesity risk
  - *Recommendation*: Lower calorie, higher protein diets more effective

- **APOA2 (Apolipoprotein A-II)**
  - *Variant*: rs5082 (-265T>C)
  - *Mechanism*: Lipid metabolism and satiety
  - *Impact*: CC genotype shows greater weight loss on low-fat diets

**Carbohydrate Metabolism:**

- **TCF7L2 (Transcription Factor 7-Like 2)**
  - *Variant*: rs7903146 (C/T)
  - *Mechanism*: Insulin sensitivity and glucose metabolism
  - *Impact*: TT genotype increases type 2 diabetes risk
  - *Recommendation*: Low glycemic index diets, smaller frequent meals

#### 2. Micronutrient Processing

**Folate Metabolism:**

- **MTHFR (Methylenetetrahydrofolate Reductase)**
  - *Variants*: rs1801133 (C677T), rs1801131 (A1298C)
  - *Mechanism*: Folate to active methylfolate conversion
  - *Impact*: TT genotype (677) reduces enzyme activity by 70%
  - *Recommendation*: Methylfolate supplementation, increased folate intake

- **MTRR (Methionine Synthase Reductase)**
  - *Variant*: rs1801394 (A66G)
  - *Mechanism*: B12-dependent methylation
  - *Impact*: GG genotype requires higher B12 intake

**Vitamin D Metabolism:**

- **VDR (Vitamin D Receptor)**
  - *Variants*: rs2228570 (FokI), rs1544410 (BsmI)
  - *Mechanism*: Vitamin D signaling and calcium absorption
  - *Impact*: FF genotype may require higher vitamin D doses

- **GC (Vitamin D Binding Protein)**
  - *Variants*: rs4588, rs7041
  - *Mechanism*: Vitamin D transport and bioavailability
  - *Impact*: Certain haplotypes associated with vitamin D deficiency

#### 3. Food Sensitivities and Intolerances

**Lactose Intolerance:**

- **MCM6 (Minichromosome Maintenance Complex Component 6)**
  - *Variant*: rs4988235 (-13910C>T)
  - *Mechanism*: Lactase persistence regulation
  - *Impact*: CC genotype = lactose intolerant (cannot digest dairy)
  - *Population Distribution*: 
    - Northern European: 5% lactose intolerant
    - East Asian: 90% lactose intolerant
    - African: 80% lactose intolerant

**Celiac Disease Risk:**

- **HLA-DQ2/DQ8 (Human Leukocyte Antigen)**
  - *Variants*: HLA-DQA1, HLA-DQB1 haplotypes
  - *Mechanism*: Gluten peptide presentation to immune system
  - *Impact*: DQ2.5/DQ2.5 highest risk (~13% lifetime risk)
  - *Recommendation*: Genetic risk assessment before gluten-free diet

**Caffeine Metabolism:**

- **CYP1A2 (Cytochrome P450 1A2)**
  - *Variant*: rs762551 (-164A>C)
  - *Mechanism*: Caffeine clearance rate
  - *Impact*: 
    - AA genotype: Slow metabolizer (caffeine sensitivity)
    - CC genotype: Fast metabolizer (may need more caffeine)

### Nutritional Analysis Framework

```python
class NutritionalAnalyzer:
    def analyze_dietary_requirements(self, variants: List[Variant]) -> DietaryProfile:
        # Macronutrient recommendations
        macro_profile = self.analyze_macronutrients(variants)
        
        # Micronutrient needs
        micro_needs = self.assess_micronutrient_requirements(variants)
        
        # Food sensitivities
        sensitivities = self.identify_food_sensitivities(variants)
        
        # Personalized recommendations
        recommendations = self.generate_dietary_recommendations(
            macro_profile, micro_needs, sensitivities
        )
        
        return DietaryProfile(
            macronutrient_ratios=macro_profile,
            micronutrient_requirements=micro_needs,
            food_sensitivities=sensitivities,
            dietary_recommendations=recommendations,
            supplement_suggestions=self.suggest_supplements(variants)
        )
```

### Personalized Nutrition Applications

#### 1. Weight Management

Genetic factors affecting weight management strategies:

```python
def weight_management_strategy(variants):
    fto_genotype = get_genotype(variants, 'FTO', 'rs9939609')
    apoa2_genotype = get_genotype(variants, 'APOA2', 'rs5082')
    
    if fto_genotype == 'AA':
        strategy = {
            'calorie_restriction': 'high_benefit',
            'protein_ratio': 'increase_to_25_percent',
            'meal_frequency': 'smaller_frequent_meals',
            'exercise_importance': 'critical'
        }
    elif fto_genotype == 'TT':
        strategy = {
            'calorie_restriction': 'moderate_benefit', 
            'protein_ratio': 'standard_15_percent',
            'meal_frequency': 'flexible',
            'exercise_importance': 'beneficial'
        }
    
    if apoa2_genotype == 'CC':
        strategy.update({'fat_intake': 'low_fat_diet_preferred'})
    
    return strategy
```

#### 2. Supplement Optimization

Evidence-based supplement recommendations:

| Genetic Variant | Supplement | Dosage Modification | Evidence Level |
|-----------------|------------|-------------------|----------------|
| MTHFR 677TT | Methylfolate | 800-1000 μg/day | High |
| VDR BsmI bb | Vitamin D3 | 2000-4000 IU/day | Moderate |
| CYP1A2 AA | Caffeine | Limit to 100mg/day | High |
| COMT Met/Met | Tyrosine | Consider supplementation | Moderate |

## Cross-Domain Interactions

### Multi-Domain Gene Effects

Many genes influence multiple domains simultaneously:

#### PPARGC1A (PGC-1α)
- **Fitness**: Enhanced endurance capacity and mitochondrial function
- **Nutrition**: Improved fat oxidation and metabolic flexibility
- **Pharmacogenetics**: May affect metformin response in diabetes

#### IL6 (Interleukin-6)
- **Fitness**: Inflammation and recovery rates
- **Nutrition**: Appetite regulation and metabolic health
- **Pharmacogenetics**: Response to anti-inflammatory medications

#### COMT (Catechol-O-Methyltransferase)
- **Fitness**: Pain tolerance and stress response during exercise
- **Nutrition**: Sensitivity to caffeine and need for tyrosine
- **Pharmacogenetics**: Response to dopaminergic medications

### Integration Algorithm

Gospel's cross-domain integration considers gene pleiotropy:

$$Score_{integrated} = \sum_{g=1}^{G} w_g \cdot \left( \sum_{d=1}^{D} \alpha_{g,d} \cdot Impact_{g,d} \right)$$

**Where:**
- $G$ = Total number of genes
- $D$ = Number of domains
- $w_g$ = Gene importance weight
- $\alpha_{g,d}$ = Gene effect size in domain $d$
- $Impact_{g,d}$ = Variant impact for gene $g$ in domain $d$

### Practical Cross-Domain Applications

#### Athlete Health Optimization

```bash
# Comprehensive analysis for elite athlete
gospel analyze --vcf athlete.vcf --domains all --focus athlete-health

# Example results:
# Genetic Profile Summary:
# - Fitness: Power-biased (8.2/10), moderate injury risk
# - Pharmacogenetics: Normal CYP2D6, poor CYP2C19 metabolizer  
# - Nutrition: Higher protein needs, lactose intolerant, slow caffeine metabolism
#
# Integrated Recommendations:
# - Training: High-intensity focus with extended recovery
# - Supplements: Whey protein isolate, methylfolate, limit caffeine
# - Medical: Avoid clopidogrel, consider alternative NSAIDs
```

#### Precision Medicine Integration

Gospel enables comprehensive precision medicine approaches:

1. **Genetic Risk Assessment**: Multi-domain disease risk evaluation
2. **Drug Selection**: Pharmacogenetic-guided medication choices
3. **Lifestyle Optimization**: Genetically-informed diet and exercise plans
4. **Monitoring Strategy**: Risk-based health monitoring protocols

---

This comprehensive domain analysis demonstrates Gospel's ability to translate complex genetic information into actionable insights across multiple aspects of health and performance. Each domain builds upon solid scientific foundations while integrating with others to provide truly personalized genomic medicine.

**Next:** Explore the [CLI Reference](cli-reference.html) for detailed command-line usage, or check out [Examples](examples.html) for real-world application scenarios. 