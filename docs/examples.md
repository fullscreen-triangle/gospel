---
layout: page
title: Real-World Examples
permalink: /examples/
---

# Real-World Examples

This section provides practical examples demonstrating Gospel's capabilities across different use cases, from personal genomics to clinical applications and research.

## Table of Contents
1. [Personal Genomics](#personal-genomics)
2. [Athletic Performance Optimization](#athletic-performance-optimization)
3. [Clinical Pharmacogenetics](#clinical-pharmacogenetics)
4. [Nutritional Optimization](#nutritional-optimization)
5. [Research Applications](#research-applications)
6. [Integration Examples](#integration-examples)

## Personal Genomics

### Complete Personal Genome Analysis

This example demonstrates how to perform a comprehensive personal genome analysis covering all domains.

```bash
#!/bin/bash
# personal_genome_analysis.sh

# Step 1: Comprehensive analysis
echo "Starting comprehensive genome analysis..."
gospel analyze \
    --vcf personal_genome.vcf \
    --domains all \
    --include-regulatory \
    --include-structural \
    --include-networks \
    --population EUR \
    --min-confidence 0.7 \
    --output personal_analysis/ \
    --format all

# Step 2: Generate interactive visualizations
echo "Creating visualizations..."
gospel visualize \
    --results personal_analysis/ \
    --score-distributions \
    --protein-networks \
    --pathway-enrichment \
    --interactive \
    --output personal_visualizations/

# Step 3: Query for specific insights
echo "Generating personalized insights..."
gospel query \
    --results personal_analysis/ \
    --query "What are my top 5 genetic strengths and weaknesses?" \
    --format markdown \
    --output personal_insights.md

gospel query \
    --results personal_analysis/ \
    --query "What lifestyle modifications would benefit my genetic profile?" \
    --format markdown \
    --output lifestyle_recommendations.md

echo "Analysis complete! Check the following files:"
echo "- personal_analysis/summary_report.html"
echo "- personal_visualizations/index.html"
echo "- personal_insights.md"
echo "- lifestyle_recommendations.md"
```

### Family Genomics Analysis

Analyze a family trio to identify inheritance patterns:

```bash
#!/bin/bash
# family_analysis.sh

# Analyze each family member
for member in father mother child; do
    echo "Analyzing $member..."
    gospel analyze \
        --vcf family_${member}.vcf \
        --domains all \
        --population EUR \
        --output family_results/${member}/
done

# Compare family results
gospel compare \
    --input-dirs family_results/father family_results/mother family_results/child \
    --output family_comparison/ \
    --inheritance-analysis \
    --shared-variants

# Interactive family exploration
gospel query \
    --results family_comparison/ \
    --interactive \
    --query "Show me genetic traits inherited from each parent"
```

## Athletic Performance Optimization

### Elite Athlete Profile Analysis

Complete genetic profiling for athletic performance optimization:

```python
#!/usr/bin/env python3
# athlete_profiling.py

from gospel import GospelAnalyzer, FitnessDomain, VisualizationEngine
import json

def analyze_athlete_genome(vcf_file, athlete_name, sport_focus=None):
    """
    Comprehensive athlete genome analysis focusing on performance traits.
    
    Args:
        vcf_file: Path to athlete's VCF file
        athlete_name: Name for output files
        sport_focus: Optional sport focus (sprint, endurance, power, mixed)
    """
    
    # Initialize analyzer with fitness focus
    analyzer = GospelAnalyzer(
        domains=['fitness'],
        include_regulatory=True,
        include_networks=True,
        performance_optimization=True
    )
    
    # Run comprehensive analysis
    print(f"Analyzing {athlete_name}'s genetic profile...")
    results = analyzer.analyze_vcf(
        vcf_file=vcf_file,
        population='EUR',  # Adjust based on athlete's ancestry
        output_dir=f'athlete_profiles/{athlete_name}/'
    )
    
    # Fitness domain deep dive
    fitness_analyzer = FitnessDomain()
    fitness_profile = fitness_analyzer.generate_athlete_profile(
        variants=results.variants,
        sport_focus=sport_focus
    )
    
    # Generate sport-specific recommendations
    recommendations = fitness_analyzer.generate_training_recommendations(
        fitness_profile=fitness_profile,
        sport_focus=sport_focus
    )
    
    # Create visualization suite
    viz_engine = VisualizationEngine()
    
    # Fitness radar chart
    viz_engine.create_fitness_radar(
        fitness_profile,
        output_file=f'athlete_profiles/{athlete_name}/fitness_radar.html'
    )
    
    # Training optimization chart
    viz_engine.create_training_chart(
        recommendations,
        output_file=f'athlete_profiles/{athlete_name}/training_plan.html'
    )
    
    # Injury risk assessment
    injury_analysis = fitness_analyzer.assess_injury_risk(
        variants=results.variants
    )
    
    # Generate comprehensive report
    athlete_report = {
        'athlete_name': athlete_name,
        'overall_fitness_score': fitness_profile.overall_score,
        'component_scores': {
            'sprint_power': fitness_profile.sprint_score,
            'endurance': fitness_profile.endurance_score,
            'strength': fitness_profile.strength_score,
            'recovery': fitness_profile.recovery_score,
            'injury_risk': injury_analysis.overall_risk
        },
        'genetic_advantages': fitness_profile.advantages,
        'genetic_limitations': fitness_profile.limitations,
        'training_recommendations': recommendations.training_plan,
        'recovery_protocols': recommendations.recovery_plan,
        'injury_prevention': injury_analysis.prevention_strategies,
        'optimal_sports': recommendations.sport_recommendations
    }
    
    # Save detailed report
    with open(f'athlete_profiles/{athlete_name}/athlete_report.json', 'w') as f:
        json.dump(athlete_report, f, indent=2)
    
    return athlete_report

# Example usage for different athlete types
if __name__ == "__main__":
    # Sprint athlete analysis
    sprint_athlete = analyze_athlete_genome(
        vcf_file='athletes/sprinter_genome.vcf',
        athlete_name='elite_sprinter',
        sport_focus='sprint'
    )
    
    # Endurance athlete analysis
    endurance_athlete = analyze_athlete_genome(
        vcf_file='athletes/marathoner_genome.vcf',
        athlete_name='elite_marathoner',
        sport_focus='endurance'
    )
    
    # Mixed sport athlete (e.g., soccer, basketball)
    mixed_athlete = analyze_athlete_genome(
        vcf_file='athletes/soccer_player_genome.vcf',
        athlete_name='soccer_player',
        sport_focus='mixed'
    )
    
    print("Athlete profiling complete!")
    print("Check the athlete_profiles/ directory for results.")
```

### Sport-Specific Training Optimization

```bash
#!/bin/bash
# sport_specific_training.sh

# Function to analyze athlete for specific sport
analyze_for_sport() {
    local vcf_file=$1
    local sport=$2
    local athlete_name=$3
    
    echo "Analyzing $athlete_name for $sport performance..."
    
    gospel analyze \
        --vcf "$vcf_file" \
        --domains fitness \
        --sport-focus "$sport" \
        --include-regulatory \
        --training-optimization \
        --injury-risk-assessment \
        --recovery-analysis \
        --output "sport_analysis/${athlete_name}_${sport}/"
    
    # Generate sport-specific recommendations
    gospel query \
        --results "sport_analysis/${athlete_name}_${sport}/" \
        --query "Provide detailed training recommendations for $sport based on my genetic profile" \
        --include-literature \
        --output "sport_analysis/${athlete_name}_${sport}/training_plan.md"
    
    # Create performance prediction
    gospel query \
        --results "sport_analysis/${athlete_name}_${sport}/" \
        --query "What is my performance potential in $sport and which aspects should I focus on?" \
        --output "sport_analysis/${athlete_name}_${sport}/performance_prediction.md"
}

# Analyze for different sports
analyze_for_sport "athlete.vcf" "sprinting" "john_doe"
analyze_for_sport "athlete.vcf" "marathon" "john_doe"
analyze_for_sport "athlete.vcf" "powerlifting" "john_doe"
analyze_for_sport "athlete.vcf" "soccer" "john_doe"

# Compare performance potential across sports
gospel compare-sports \
    --input-dirs sport_analysis/john_doe_*/ \
    --output sport_analysis/john_doe_sport_comparison/ \
    --create-recommendations
```

## Clinical Pharmacogenetics

### Pre-prescription Pharmacogenetic Screening

Clinical workflow for medication safety assessment:

```python
#!/usr/bin/env python3
# clinical_pharma_screening.py

from gospel import PharmacoAnalyzer, ClinicalReportGenerator
import pandas as pd

def clinical_pharma_screening(patient_vcf, patient_id, medication_list=None):
    """
    Clinical pharmacogenetic screening for medication safety.
    
    Args:
        patient_vcf: Patient's VCF file
        patient_id: Unique patient identifier
        medication_list: List of medications to specifically analyze
    """
    
    # Initialize pharmacogenetic analyzer
    pharma_analyzer = PharmacoAnalyzer(
        clinical_mode=True,
        include_dosing_guidelines=True,
        include_drug_interactions=True
    )
    
    # Run pharmacogenetic analysis
    print(f"Running pharmacogenetic analysis for patient {patient_id}...")
    results = pharma_analyzer.analyze_vcf(
        vcf_file=patient_vcf,
        output_dir=f'clinical_reports/{patient_id}/'
    )
    
    # Analyze specific medications if provided
    if medication_list:
        medication_analysis = pharma_analyzer.analyze_medications(
            variants=results.variants,
            medications=medication_list
        )
    else:
        medication_analysis = pharma_analyzer.analyze_all_drug_classes(
            variants=results.variants
        )
    
    # Generate clinical recommendations
    clinical_recommendations = {
        'patient_id': patient_id,
        'analysis_date': pharma_analyzer.get_analysis_date(),
        'cyp_status': {
            'CYP2D6': medication_analysis.cyp2d6_status,
            'CYP2C19': medication_analysis.cyp2c19_status,
            'CYP2C9': medication_analysis.cyp2c9_status,
            'CYP3A4': medication_analysis.cyp3a4_status
        },
        'high_risk_medications': medication_analysis.high_risk_drugs,
        'dosing_adjustments': medication_analysis.dosing_recommendations,
        'alternative_medications': medication_analysis.alternative_drugs,
        'drug_interactions': medication_analysis.potential_interactions,
        'monitoring_recommendations': medication_analysis.monitoring_plans
    }
    
    # Generate clinical report
    report_generator = ClinicalReportGenerator()
    clinical_report = report_generator.generate_pharma_report(
        patient_id=patient_id,
        analysis_results=results,
        medication_analysis=medication_analysis,
        recommendations=clinical_recommendations
    )
    
    # Save clinical report
    report_generator.save_report(
        report=clinical_report,
        output_file=f'clinical_reports/{patient_id}/pharma_report.pdf',
        format='clinical_pdf'
    )
    
    # Generate pharmacist reference sheet
    pharmacist_sheet = report_generator.generate_pharmacist_reference(
        patient_id=patient_id,
        cyp_status=clinical_recommendations['cyp_status'],
        drug_recommendations=clinical_recommendations
    )
    
    return clinical_recommendations

# Example: Pre-surgical screening
def pre_surgical_screening(patient_vcf, patient_id):
    """Screen patient before surgery for pain medication compatibility."""
    
    # Common surgical medications
    surgical_medications = [
        'morphine', 'codeine', 'tramadol', 'fentanyl',
        'warfarin', 'clopidogrel', 'aspirin',
        'midazolam', 'propofol'
    ]
    
    recommendations = clinical_pharma_screening(
        patient_vcf=patient_vcf,
        patient_id=patient_id,
        medication_list=surgical_medications
    )
    
    # Generate surgical-specific recommendations
    surgical_recommendations = {
        'pain_management': recommendations.get('pain_medications', {}),
        'anticoagulation': recommendations.get('anticoagulants', {}),
        'anesthesia': recommendations.get('anesthetics', {}),
        'post_op_monitoring': recommendations.get('monitoring_recommendations', {})
    }
    
    return surgical_recommendations

# Example usage
if __name__ == "__main__":
    # General pharmacogenetic screening
    patient_001 = clinical_pharma_screening(
        patient_vcf='patients/patient_001.vcf',
        patient_id='PATIENT_001'
    )
    
    # Pre-surgical screening
    surgical_patient = pre_surgical_screening(
        patient_vcf='patients/surgical_patient.vcf',
        patient_id='SURGICAL_001'
    )
    
    print("Clinical pharmacogenetic screening complete!")
```

### Drug Development Support

Pharmacogenetic analysis for clinical trials:

```python
#!/usr/bin/env python3
# clinical_trial_stratification.py

from gospel import PharmacoAnalyzer, CohortAnalyzer
import pandas as pd

def stratify_clinical_trial_cohort(cohort_vcfs, drug_target, trial_name):
    """
    Stratify clinical trial participants based on pharmacogenetic profiles.
    
    Args:
        cohort_vcfs: List of VCF files for trial participants
        drug_target: Target drug/pathway for the trial
        trial_name: Name of the clinical trial
    """
    
    pharma_analyzer = PharmacoAnalyzer()
    cohort_analyzer = CohortAnalyzer()
    
    participant_profiles = []
    
    # Analyze each participant
    for i, vcf_file in enumerate(cohort_vcfs):
        participant_id = f"PARTICIPANT_{i+1:03d}"
        
        print(f"Analyzing {participant_id}...")
        
        # Individual pharmacogenetic analysis
        results = pharma_analyzer.analyze_vcf(
            vcf_file=vcf_file,
            drug_focus=drug_target,
            output_dir=f'clinical_trials/{trial_name}/{participant_id}/'
        )
        
        # Extract relevant pharmacogenetic markers
        profile = {
            'participant_id': participant_id,
            'cyp2d6_phenotype': results.cyp2d6_status,
            'cyp2c19_phenotype': results.cyp2c19_status,
            'cyp2c9_phenotype': results.cyp2c9_status,
            'drug_target_variants': results.drug_target_analysis,
            'predicted_response': results.response_prediction,
            'recommended_dose': results.dosing_recommendation,
            'risk_factors': results.risk_assessment
        }
        
        participant_profiles.append(profile)
    
    # Cohort-level analysis
    cohort_summary = cohort_analyzer.analyze_cohort(
        participant_profiles=participant_profiles,
        drug_target=drug_target
    )
    
    # Stratification recommendations
    stratification = cohort_analyzer.recommend_stratification(
        cohort_summary=cohort_summary,
        trial_objectives=['efficacy', 'safety', 'dosing']
    )
    
    # Generate trial design recommendations
    trial_design = {
        'total_participants': len(participant_profiles),
        'stratification_groups': stratification.groups,
        'randomization_strategy': stratification.randomization,
        'dosing_schema': stratification.dosing_groups,
        'monitoring_requirements': stratification.monitoring_plan,
        'expected_outcomes': stratification.outcome_predictions
    }
    
    return trial_design, participant_profiles

# Example usage for different trial types
if __name__ == "__main__":
    # Cardiovascular drug trial
    cv_trial = stratify_clinical_trial_cohort(
        cohort_vcfs=[f'cohorts/cv_trial/participant_{i}.vcf' for i in range(1, 101)],
        drug_target='warfarin',
        trial_name='CV_WARFARIN_TRIAL'
    )
    
    # Oncology trial
    oncology_trial = stratify_clinical_trial_cohort(
        cohort_vcfs=[f'cohorts/oncology/patient_{i}.vcf' for i in range(1, 51)],
        drug_target='5-fluorouracil',
        trial_name='ONCOLOGY_5FU_TRIAL'
    )
```

## Nutritional Optimization

### Personalized Nutrition Plan

Complete nutritional genomics analysis with meal planning:

```python
#!/usr/bin/env python3
# personalized_nutrition.py

from gospel import NutritionAnalyzer, MealPlanner, SupplementOptimizer
import json

def create_personalized_nutrition_plan(vcf_file, user_profile):
    """
    Create comprehensive personalized nutrition plan based on genetics.
    
    Args:
        vcf_file: User's genetic data
        user_profile: Additional user information (age, sex, goals, etc.)
    """
    
    # Initialize nutrition analyzer
    nutrition_analyzer = NutritionAnalyzer(
        include_sensitivities=True,
        include_metabolism=True,
        include_requirements=True
    )
    
    # Run nutritional genomics analysis
    print("Analyzing nutritional genetics...")
    results = nutrition_analyzer.analyze_vcf(
        vcf_file=vcf_file,
        output_dir=f'nutrition_plans/{user_profile["user_id"]}/'
    )
    
    # Analyze specific nutritional components
    macro_analysis = nutrition_analyzer.analyze_macronutrients(
        variants=results.variants,
        user_profile=user_profile
    )
    
    micro_analysis = nutrition_analyzer.analyze_micronutrients(
        variants=results.variants
    )
    
    sensitivity_analysis = nutrition_analyzer.analyze_food_sensitivities(
        variants=results.variants
    )
    
    # Generate personalized recommendations
    nutrition_recommendations = {
        'user_id': user_profile['user_id'],
        'genetic_profile': {
            'metabolism_type': macro_analysis.metabolism_type,
            'caffeine_sensitivity': sensitivity_analysis.caffeine_sensitivity,
            'lactose_tolerance': sensitivity_analysis.lactose_tolerance,
            'gluten_sensitivity_risk': sensitivity_analysis.gluten_risk,
            'alcohol_metabolism': macro_analysis.alcohol_metabolism
        },
        'macronutrient_ratios': {
            'carbohydrates': macro_analysis.optimal_carb_ratio,
            'proteins': macro_analysis.optimal_protein_ratio,
            'fats': macro_analysis.optimal_fat_ratio
        },
        'micronutrient_needs': {
            'folate': micro_analysis.folate_requirements,
            'vitamin_d': micro_analysis.vitamin_d_requirements,
            'vitamin_b12': micro_analysis.b12_requirements,
            'iron': micro_analysis.iron_requirements
        },
        'foods_to_emphasize': macro_analysis.beneficial_foods,
        'foods_to_limit': sensitivity_analysis.problematic_foods,
        'meal_timing': macro_analysis.optimal_meal_timing
    }
    
    # Generate supplement recommendations
    supplement_optimizer = SupplementOptimizer()
    supplement_plan = supplement_optimizer.optimize_supplements(
        genetic_profile=nutrition_recommendations['genetic_profile'],
        micronutrient_needs=nutrition_recommendations['micronutrient_needs'],
        user_profile=user_profile
    )
    
    # Create meal plans
    meal_planner = MealPlanner()
    weekly_meal_plan = meal_planner.create_weekly_plan(
        nutrition_recommendations=nutrition_recommendations,
        user_preferences=user_profile.get('dietary_preferences', {}),
        calorie_target=user_profile.get('calorie_target'),
        meal_count=user_profile.get('meals_per_day', 3)
    )
    
    # Compile complete nutrition plan
    complete_plan = {
        'nutrition_recommendations': nutrition_recommendations,
        'supplement_plan': supplement_plan,
        'weekly_meal_plan': weekly_meal_plan,
        'shopping_list': meal_planner.generate_shopping_list(weekly_meal_plan),
        'recipe_suggestions': meal_planner.get_recipe_suggestions(nutrition_recommendations)
    }
    
    # Save nutrition plan
    with open(f'nutrition_plans/{user_profile["user_id"]}/complete_plan.json', 'w') as f:
        json.dump(complete_plan, f, indent=2)
    
    return complete_plan

# Example user profiles
if __name__ == "__main__":
    # Weight loss focused
    weight_loss_user = {
        'user_id': 'USER_001',
        'age': 35,
        'sex': 'female',
        'goal': 'weight_loss',
        'calorie_target': 1500,
        'activity_level': 'moderate',
        'dietary_preferences': {
            'vegetarian': False,
            'gluten_free': False,
            'dairy_free': False
        },
        'meals_per_day': 4
    }
    
    weight_loss_plan = create_personalized_nutrition_plan(
        vcf_file='users/weight_loss_user.vcf',
        user_profile=weight_loss_user
    )
    
    # Athletic performance focused
    athlete_user = {
        'user_id': 'ATHLETE_001',
        'age': 28,
        'sex': 'male',
        'goal': 'athletic_performance',
        'sport': 'endurance_running',
        'calorie_target': 3500,
        'activity_level': 'high',
        'meals_per_day': 5
    }
    
    athlete_plan = create_personalized_nutrition_plan(
        vcf_file='users/athlete_user.vcf',
        user_profile=athlete_user
    )
    
    print("Personalized nutrition plans created!")
```

## Research Applications

### Population Genomics Study

Large-scale population analysis using Gospel:

```python
#!/usr/bin/env python3
# population_genomics_study.py

from gospel import PopulationAnalyzer, StatisticalAnalyzer
import pandas as pd
from multiprocessing import Pool
import numpy as np

def analyze_population_cohort(cohort_data, study_name, research_question):
    """
    Analyze large population cohort for genomic research.
    
    Args:
        cohort_data: List of dictionaries with VCF files and metadata
        study_name: Name of the research study
        research_question: Specific research focus
    """
    
    population_analyzer = PopulationAnalyzer()
    stats_analyzer = StatisticalAnalyzer()
    
    print(f"Starting population analysis for {study_name}")
    print(f"Cohort size: {len(cohort_data)} individuals")
    
    # Parallel processing of individual genomes
    def analyze_individual(individual_data):
        return population_analyzer.analyze_individual(
            vcf_file=individual_data['vcf_file'],
            metadata=individual_data['metadata'],
            domains=['fitness', 'pharmacogenetics', 'nutrition']
        )
    
    # Process cohort in parallel
    with Pool() as pool:
        individual_results = pool.map(analyze_individual, cohort_data)
    
    # Aggregate population-level results
    population_summary = population_analyzer.aggregate_cohort_results(
        individual_results=individual_results,
        study_focus=research_question
    )
    
    # Statistical analysis
    statistical_results = stats_analyzer.perform_population_analysis(
        population_data=population_summary,
        analysis_type=research_question
    )
    
    # Generate research insights
    research_findings = {
        'study_name': study_name,
        'cohort_characteristics': population_summary.demographics,
        'allele_frequencies': statistical_results.allele_frequencies,
        'domain_distributions': statistical_results.domain_score_distributions,
        'population_comparisons': statistical_results.population_comparisons,
        'significant_associations': statistical_results.gwas_results,
        'pathway_enrichment': statistical_results.pathway_analysis,
        'clinical_implications': statistical_results.clinical_relevance
    }
    
    return research_findings

# Example: Multi-ethnic fitness genomics study
def fitness_genomics_study():
    """Study genetic factors affecting fitness across populations."""
    
    # Simulated cohort data
    cohort_data = []
    
    populations = ['EUR', 'AFR', 'EAS', 'AMR', 'SAS']
    
    for pop in populations:
        for i in range(100):  # 100 individuals per population
            individual = {
                'vcf_file': f'cohorts/fitness_study/{pop}/individual_{i}.vcf',
                'metadata': {
                    'population': pop,
                    'age': np.random.randint(18, 65),
                    'sex': np.random.choice(['male', 'female']),
                    'sport_category': np.random.choice(['sprint', 'endurance', 'power', 'mixed']),
                    'elite_status': np.random.choice([True, False], p=[0.2, 0.8])
                }
            }
            cohort_data.append(individual)
    
    # Run population analysis
    fitness_study_results = analyze_population_cohort(
        cohort_data=cohort_data,
        study_name="Multi-Ethnic Fitness Genomics Study",
        research_question="fitness_population_genetics"
    )
    
    return fitness_study_results

# Example: Pharmacogenetics across populations
def pharma_population_study():
    """Study pharmacogenetic variant frequencies across populations."""
    
    # Focus on key pharmacogenetic genes
    target_genes = ['CYP2D6', 'CYP2C19', 'CYP2C9', 'VKORC1', 'SLCO1B1']
    
    cohort_data = []
    populations = ['EUR', 'AFR', 'EAS', 'AMR', 'SAS']
    
    for pop in populations:
        for i in range(200):  # 200 individuals per population
            individual = {
                'vcf_file': f'cohorts/pharma_study/{pop}/individual_{i}.vcf',
                'metadata': {
                    'population': pop,
                    'age': np.random.randint(18, 80),
                    'sex': np.random.choice(['male', 'female']),
                    'medical_history': np.random.choice(['healthy', 'cardiovascular', 'psychiatric'], 
                                                      p=[0.6, 0.3, 0.1])
                }
            }
            cohort_data.append(individual)
    
    pharma_study_results = analyze_population_cohort(
        cohort_data=cohort_data,
        study_name="Global Pharmacogenetics Frequency Study",
        research_question="pharmacogenetic_population_frequencies"
    )
    
    return pharma_study_results

if __name__ == "__main__":
    # Run fitness genomics study
    fitness_results = fitness_genomics_study()
    print("Fitness genomics study completed!")
    
    # Run pharmacogenetics study
    pharma_results = pharma_population_study()
    print("Pharmacogenetics population study completed!")
```

## Integration Examples

### Healthcare System Integration

Integration with electronic health records (EHR):

```python
#!/usr/bin/env python3
# ehr_integration.py

from gospel import GospelAnalyzer, EHRInterface, ClinicalDecisionSupport
import json
import datetime

class GospelEHRIntegration:
    """Integration layer between Gospel and healthcare systems."""
    
    def __init__(self, ehr_config):
        self.ehr_interface = EHRInterface(ehr_config)
        self.gospel_analyzer = GospelAnalyzer()
        self.clinical_support = ClinicalDecisionSupport()
    
    def process_new_patient_genetics(self, patient_id, vcf_file):
        """Process genetic data for new patient and update EHR."""
        
        # Retrieve patient information from EHR
        patient_info = self.ehr_interface.get_patient_info(patient_id)
        
        # Run Gospel analysis
        genetic_analysis = self.gospel_analyzer.analyze_vcf(
            vcf_file=vcf_file,
            domains=['pharmacogenetics', 'nutrition'],
            clinical_mode=True
        )
        
        # Generate clinical recommendations
        clinical_recommendations = self.clinical_support.generate_recommendations(
            genetic_analysis=genetic_analysis,
            patient_info=patient_info
        )
        
        # Format for EHR integration
        ehr_genetic_data = {
            'patient_id': patient_id,
            'analysis_date': datetime.datetime.now().isoformat(),
            'pharmacogenetic_profile': {
                'cyp2d6_status': genetic_analysis.cyp2d6_phenotype,
                'cyp2c19_status': genetic_analysis.cyp2c19_phenotype,
                'cyp2c9_status': genetic_analysis.cyp2c9_phenotype,
                'high_risk_medications': clinical_recommendations.contraindicated_drugs,
                'dosing_adjustments': clinical_recommendations.dose_modifications
            },
            'nutritional_profile': {
                'folate_metabolism': genetic_analysis.mthfr_status,
                'vitamin_d_requirements': genetic_analysis.vitamin_d_needs,
                'food_sensitivities': genetic_analysis.food_intolerances
            },
            'clinical_alerts': clinical_recommendations.priority_alerts
        }
        
        # Update EHR with genetic data
        self.ehr_interface.update_patient_genetics(
            patient_id=patient_id,
            genetic_data=ehr_genetic_data
        )
        
        return ehr_genetic_data
    
    def check_medication_compatibility(self, patient_id, new_medication):
        """Check medication compatibility against patient's genetic profile."""
        
        # Retrieve genetic profile from EHR
        genetic_profile = self.ehr_interface.get_patient_genetics(patient_id)
        
        if not genetic_profile:
            return {'warning': 'No genetic data available for this patient'}
        
        # Analyze medication compatibility
        compatibility = self.clinical_support.check_drug_compatibility(
            medication=new_medication,
            genetic_profile=genetic_profile
        )
        
        # Generate alerts if needed
        if compatibility.has_contraindications:
            alert = {
                'alert_type': 'GENETIC_CONTRAINDICATION',
                'severity': compatibility.severity,
                'message': compatibility.warning_message,
                'recommendations': compatibility.alternative_medications,
                'references': compatibility.clinical_guidelines
            }
            
            # Send alert to EHR system
            self.ehr_interface.send_clinical_alert(patient_id, alert)
            
            return alert
        
        return {'status': 'compatible', 'recommendations': compatibility.dosing_guidance}

# Example usage in clinical workflow
def clinical_workflow_example():
    """Example of Gospel integration in clinical workflow."""
    
    # Initialize integration
    ehr_config = {
        'ehr_system': 'epic',
        'api_endpoint': 'https://ehr.hospital.com/api',
        'credentials': 'path/to/credentials.json'
    }
    
    gospel_ehr = GospelEHRIntegration(ehr_config)
    
    # Scenario 1: New patient with genetic testing
    patient_genetics = gospel_ehr.process_new_patient_genetics(
        patient_id='PATIENT_12345',
        vcf_file='genetic_tests/patient_12345.vcf'
    )
    
    print("Genetic profile integrated into EHR")
    
    # Scenario 2: Prescribing new medication
    compatibility_check = gospel_ehr.check_medication_compatibility(
        patient_id='PATIENT_12345',
        new_medication='warfarin'
    )
    
    if compatibility_check.get('alert_type'):
        print(f"ALERT: {compatibility_check['message']}")
    else:
        print("Medication compatible with genetic profile")

if __name__ == "__main__":
    clinical_workflow_example()
```

### Research Platform Integration

Integration with research data management platforms:

```python
#!/usr/bin/env python3
# research_platform_integration.py

from gospel import ResearchAnalyzer, DataExporter, QualityController
import pandas as pd

class ResearchPlatformConnector:
    """Connector for integrating Gospel with research platforms."""
    
    def __init__(self, platform_config):
        self.platform_config = platform_config
        self.research_analyzer = ResearchAnalyzer()
        self.data_exporter = DataExporter()
        self.quality_controller = QualityController()
    
    def process_research_cohort(self, study_id, cohort_manifest):
        """Process entire research cohort through Gospel pipeline."""
        
        print(f"Processing research cohort for study {study_id}")
        
        # Load cohort manifest
        cohort_df = pd.read_csv(cohort_manifest)
        
        # Quality control checks
        qc_results = self.quality_controller.assess_cohort_quality(cohort_df)
        
        if not qc_results.passes_qc:
            raise ValueError(f"Cohort failed QC: {qc_results.issues}")
        
        # Process each sample
        cohort_results = []
        
        for _, sample in cohort_df.iterrows():
            sample_id = sample['sample_id']
            vcf_file = sample['vcf_path']
            
            # Individual analysis
            individual_result = self.research_analyzer.analyze_research_sample(
                sample_id=sample_id,
                vcf_file=vcf_file,
                study_metadata=sample.to_dict()
            )
            
            cohort_results.append(individual_result)
        
        # Aggregate cohort-level results
        cohort_summary = self.research_analyzer.summarize_cohort(
            cohort_results=cohort_results,
            study_id=study_id
        )
        
        # Export results in research-friendly formats
        self.export_research_results(
            study_id=study_id,
            cohort_summary=cohort_summary,
            individual_results=cohort_results
        )
        
        return cohort_summary
    
    def export_research_results(self, study_id, cohort_summary, individual_results):
        """Export results in formats suitable for research analysis."""
        
        # PLINK format for GWAS
        self.data_exporter.export_plink_format(
            study_id=study_id,
            results=cohort_summary,
            output_dir=f'research_exports/{study_id}/plink/'
        )
        
        # R data format
        self.data_exporter.export_r_format(
            study_id=study_id,
            results=cohort_summary,
            output_dir=f'research_exports/{study_id}/r_data/'
        )
        
        # JSON for web applications
        self.data_exporter.export_json_format(
            study_id=study_id,
            results=cohort_summary,
            output_dir=f'research_exports/{study_id}/json/'
        )
        
        # CSV for statistical analysis
        self.data_exporter.export_csv_format(
            study_id=study_id,
            results=cohort_summary,
            output_dir=f'research_exports/{study_id}/csv/'
        )

# Example research study
if __name__ == "__main__":
    platform_config = {
        'platform': 'terra',
        'workspace': 'genomics-research',
        'project': 'multi-domain-gwas'
    }
    
    connector = ResearchPlatformConnector(platform_config)
    
    # Process large-scale GWAS study
    gwas_results = connector.process_research_cohort(
        study_id='MULTI_DOMAIN_GWAS_2024',
        cohort_manifest='studies/multi_domain_gwas/cohort_manifest.csv'
    )
    
    print("Research cohort processing complete!")
```

---

These comprehensive examples demonstrate Gospel's versatility across different use cases, from personal genomics to large-scale research applications. Each example includes complete, runnable code that showcases Gospel's capabilities and integration potential.

**Next:** Check out the [API Reference](api-reference.html) for detailed programmatic interface documentation, or explore [Contributing](contributing.html) to help improve Gospel. 