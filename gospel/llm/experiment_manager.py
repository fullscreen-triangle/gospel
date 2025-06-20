"""
Per-Experiment LLM Manager for Gospel.

This module implements experiment-specific LLM creation and management,
allowing Gospel to generate specialized language models for each genomic
analysis experiment based on the research context and objectives.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model

from .trainer import GenomicLLMTrainer
from .genomic_models import GenomicModelManager, GENOMIC_MODELS
from ..core.metacognitive import AnalysisContext, ObjectiveFunction

logger = logging.getLogger(__name__)


@dataclass
class ExperimentContext:
    """Context information for experiment-specific LLM creation"""
    experiment_id: str
    research_objective: str
    genomic_focus: List[str]  # e.g., ['variant_analysis', 'expression_profiling']
    tissue_types: List[str]
    organism: str
    publications: List[str]  # Literature references
    temporal_scope: str  # e.g., 'longitudinal', 'cross_sectional'
    collaboration_partners: List[str]
    expected_sample_size: int
    computational_budget: str  # e.g., '30_minutes', '2_hours'


@dataclass
class LoRAConfiguration:
    """Configuration for LoRA fine-tuning"""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class ExperimentLLMManager:
    """
    Manager for creating and maintaining experiment-specific LLMs.
    
    This class implements the per-experiment LLM architecture described in the
    Gospel framework, where each experiment generates its own specialized
    language model based on research context and genomic data.
    """
    
    def __init__(self,
                 base_model: str = "microsoft/DialoGPT-medium",
                 models_directory: str = "experiment_models",
                 device: str = "auto",
                 use_quantization: bool = True):
        """
        Initialize the Experiment LLM Manager.
        
        Args:
            base_model: Base model for fine-tuning
            models_directory: Directory to store experiment-specific models
            device: Device for training and inference
            use_quantization: Whether to use quantization for efficiency
        """
        self.base_model = base_model
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_quantization = use_quantization
        
        # Component managers
        self.genomic_model_manager = GenomicModelManager()
        self.active_experiments = {}
        
        # Quantization configuration
        if self.use_quantization and self.device == "cuda":
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            self.quantization_config = None
        
        logger.info(f"ExperimentLLMManager initialized with base model: {base_model}")
    
    def create_experiment_llm(self,
                            experiment_context: ExperimentContext,
                            genomic_data: Dict[str, Any],
                            lora_config: Optional[LoRAConfiguration] = None,
                            training_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create specialized LLM for experiment-specific analysis.
        
        Args:
            experiment_context: Context information for the experiment
            genomic_data: Genomic datasets for the experiment
            lora_config: LoRA configuration for efficient fine-tuning
            training_config: Training hyperparameters
            
        Returns:
            Path to the trained experiment-specific model
        """
        logger.info(f"Creating experiment-specific LLM for: {experiment_context.experiment_id}")
        
        # Set default configurations
        if lora_config is None:
            lora_config = LoRAConfiguration()
        
        if training_config is None:
            training_config = {
                "num_epochs": 3,
                "batch_size": 4,
                "learning_rate": 5e-5,
                "warmup_steps": 100,
                "logging_steps": 50,
                "save_steps": 500,
                "max_length": 512
            }
        
        # Generate training dataset from experiment context
        training_data = self.generate_training_dataset(
            experiment_context=experiment_context,
            genomic_data=genomic_data
        )
        
        # Create experiment-specific model directory
        experiment_model_dir = self.models_directory / experiment_context.experiment_id
        experiment_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Fine-tune base model with LoRA
        model_path = self._train_experiment_model(
            training_data=training_data,
            lora_config=lora_config,
            training_config=training_config,
            output_dir=experiment_model_dir,
            experiment_context=experiment_context
        )
        
        # Register active experiment
        self.active_experiments[experiment_context.experiment_id] = {
            "model_path": model_path,
            "context": experiment_context,
            "created_at": pd.Timestamp.now(),
            "genomic_focus": experiment_context.genomic_focus,
            "organism": experiment_context.organism
        }
        
        logger.info(f"Experiment LLM created successfully: {model_path}")
        return model_path
    
    def generate_training_dataset(self,
                                experiment_context: ExperimentContext,
                                genomic_data: Dict[str, Any]) -> Dataset:
        """
        Generate training dataset from experiment context and genomic data.
        
        Args:
            experiment_context: Experiment context information
            genomic_data: Genomic datasets
            
        Returns:
            HuggingFace Dataset for training
        """
        training_examples = []
        
        # Generate examples based on research objective
        objective_examples = self._generate_objective_examples(
            experiment_context.research_objective,
            genomic_data
        )
        training_examples.extend(objective_examples)
        
        # Generate examples based on genomic focus areas
        for focus_area in experiment_context.genomic_focus:
            focus_examples = self._generate_focus_examples(
                focus_area,
                genomic_data,
                experiment_context
            )
            training_examples.extend(focus_examples)
        
        # Generate organism-specific examples
        organism_examples = self._generate_organism_examples(
            experiment_context.organism,
            genomic_data
        )
        training_examples.extend(organism_examples)
        
        # Generate literature-informed examples
        literature_examples = self._generate_literature_examples(
            experiment_context.publications,
            genomic_data
        )
        training_examples.extend(literature_examples)
        
        # Generate temporal analysis examples if applicable
        if experiment_context.temporal_scope == 'longitudinal':
            temporal_examples = self._generate_temporal_examples(genomic_data)
            training_examples.extend(temporal_examples)
        
        logger.info(f"Generated {len(training_examples)} training examples for experiment")
        
        # Convert to HuggingFace Dataset
        return Dataset.from_list(training_examples)
    
    def _generate_objective_examples(self,
                                   research_objective: str,
                                   genomic_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate training examples based on research objective"""
        examples = []
        
        if "pathogenic" in research_objective.lower():
            # Pathogenicity prediction examples
            if "variants" in genomic_data:
                for _, variant in genomic_data["variants"].head(20).iterrows():
                    instruction = f"Assess the pathogenicity of variant {variant.get('id', 'unknown')} in the context of {research_objective}"
                    
                    response = f"""Pathogenicity Assessment for Research Objective: {research_objective}

Variant Details:
- Position: {variant.get('chromosome', 'Unknown')}:{variant.get('position', 'Unknown')}
- Change: {variant.get('reference', 'N')}>{variant.get('alternate', 'N')}
- Gene Context: {variant.get('gene', 'Unknown')}

Analysis Approach:
1. Evaluate variant impact on protein function
2. Consider population frequency data
3. Assess conservation across species
4. Review clinical databases for similar variants

Recommendation: Based on the specific research objective focusing on pathogenic variants, this variant should be prioritized for {research_objective.lower()} analysis."""
                    
                    examples.append({
                        "instruction": instruction,
                        "input": "",
                        "output": response
                    })
        
        elif "expression" in research_objective.lower():
            # Expression analysis examples
            if "expression" in genomic_data:
                for gene in list(genomic_data["expression"].index)[:15]:
                    instruction = f"Interpret expression profile of {gene} for {research_objective}"
                    
                    expression_values = genomic_data["expression"].loc[gene]
                    mean_expr = expression_values.mean()
                    std_expr = expression_values.std()
                    
                    response = f"""Expression Analysis for Research Objective: {research_objective}

Gene: {gene}
Expression Profile:
- Mean Expression: {mean_expr:.2f}
- Standard Deviation: {std_expr:.2f}
- Expression Range: {expression_values.min():.2f} to {expression_values.max():.2f}

Biological Interpretation:
This gene shows {'high' if mean_expr > 5 else 'moderate' if mean_expr > 1 else 'low'} expression levels with {'stable' if std_expr < 1 else 'variable'} patterns.

Research Relevance: In the context of {research_objective}, this expression pattern suggests {gene} may be {'highly relevant' if mean_expr > 3 else 'moderately relevant'} to the study objectives."""
                    
                    examples.append({
                        "instruction": instruction,
                        "input": "",
                        "output": response
                    })
        
        return examples
    
    def _generate_focus_examples(self,
                               focus_area: str,
                               genomic_data: Dict[str, Any],
                               experiment_context: ExperimentContext) -> List[Dict[str, str]]:
        """Generate examples based on genomic focus areas"""
        examples = []
        
        if focus_area == "variant_analysis":
            # Variant-focused examples
            examples.extend(self._create_variant_analysis_examples(genomic_data, experiment_context))
        
        elif focus_area == "expression_profiling":
            # Expression-focused examples
            examples.extend(self._create_expression_profiling_examples(genomic_data, experiment_context))
        
        elif focus_area == "network_analysis":
            # Network-focused examples
            examples.extend(self._create_network_analysis_examples(genomic_data, experiment_context))
        
        elif focus_area == "pathway_enrichment":
            # Pathway-focused examples
            examples.extend(self._create_pathway_examples(genomic_data, experiment_context))
        
        return examples
    
    def _create_variant_analysis_examples(self,
                                        genomic_data: Dict[str, Any],
                                        experiment_context: ExperimentContext) -> List[Dict[str, str]]:
        """Create variant analysis specific examples"""
        examples = []
        
        if "variants" in genomic_data:
            variants_df = genomic_data["variants"]
            
            for _, variant in variants_df.head(10).iterrows():
                instruction = f"Perform comprehensive variant analysis for {experiment_context.organism} study"
                
                response = f"""Comprehensive Variant Analysis

Study Context: {experiment_context.experiment_id}
Organism: {experiment_context.organism}
Research Focus: Variant Analysis

Variant Information:
- ID: {variant.get('id', 'Unknown')}
- Location: {variant.get('chromosome', 'Chr?')}:{variant.get('position', '?')}
- Allelic Change: {variant.get('reference', 'N')} → {variant.get('alternate', 'N')}

Analysis Framework:
1. Functional Impact Assessment
2. Population Genetics Context
3. Clinical Significance Evaluation
4. Evolutionary Conservation Analysis

Experimental Relevance:
This variant is relevant to the {experiment_context.research_objective} because it demonstrates the analytical framework needed for comprehensive variant interpretation in {experiment_context.organism} genomics."""

                examples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": response
                })
        
        return examples
    
    def _create_expression_profiling_examples(self,
                                           genomic_data: Dict[str, Any],
                                           experiment_context: ExperimentContext) -> List[Dict[str, str]]:
        """Create expression profiling specific examples"""
        examples = []
        
        if "expression" in genomic_data:
            expression_df = genomic_data["expression"]
            
            # Sample genes for examples
            sample_genes = list(expression_df.index)[:8]
            
            for gene in sample_genes:
                instruction = f"Profile gene expression patterns for {gene} in {experiment_context.organism}"
                
                gene_expression = expression_df.loc[gene]
                
                response = f"""Gene Expression Profiling Analysis

Study: {experiment_context.experiment_id}
Target Gene: {gene}
Organism: {experiment_context.organism}
Sample Size: {len(gene_expression)} samples

Expression Statistics:
- Mean Expression: {gene_expression.mean():.3f}
- Median Expression: {gene_expression.median():.3f}
- Expression Variance: {gene_expression.var():.3f}
- Dynamic Range: {gene_expression.max() - gene_expression.min():.3f}

Biological Significance:
Gene {gene} shows {'high' if gene_expression.mean() > 5 else 'moderate' if gene_expression.mean() > 1 else 'low'} expression with {'high' if gene_expression.var() > 2 else 'moderate' if gene_expression.var() > 0.5 else 'low'} variability across samples.

Study Relevance:
This expression profile is significant for {experiment_context.research_objective} as it provides insights into {gene} regulation in {experiment_context.organism}."""

                examples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": response
                })
        
        return examples
    
    def _create_network_analysis_examples(self,
                                        genomic_data: Dict[str, Any],
                                        experiment_context: ExperimentContext) -> List[Dict[str, str]]:
        """Create network analysis specific examples"""
        examples = []
        
        # Generate synthetic network examples since network data structure may vary
        network_genes = ["TP53", "BRCA1", "EGFR", "MYC", "PTEN", "RAS", "AKT1", "PIK3CA"]
        
        for i, gene in enumerate(network_genes[:5]):
            instruction = f"Analyze gene regulatory network involving {gene} in {experiment_context.organism}"
            
            # Create synthetic interaction partners
            partners = [g for g in network_genes if g != gene][:3]
            
            response = f"""Gene Regulatory Network Analysis

Primary Gene: {gene}
Study Context: {experiment_context.experiment_id}
Organism: {experiment_context.organism}

Network Topology:
- Central Hub: {gene}
- Direct Interactions: {len(partners)} confirmed
- Interaction Partners: {', '.join(partners)}

Functional Analysis:
Gene {gene} functions as a regulatory hub in the network, with direct interactions to {', '.join(partners)}. This network module is particularly relevant for {experiment_context.research_objective}.

Experimental Design Implications:
For {experiment_context.organism} studies, monitoring {gene} and its network partners provides comprehensive coverage of the regulatory module relevant to your research objectives."""

            examples.append({
                "instruction": instruction,
                "input": "",
                "output": response
            })
        
        return examples
    
    def _create_pathway_examples(self,
                               genomic_data: Dict[str, Any],
                               experiment_context: ExperimentContext) -> List[Dict[str, str]]:
        """Create pathway enrichment examples"""
        examples = []
        
        # Common biological pathways
        pathways = [
            ("DNA Repair", ["TP53", "BRCA1", "ATM", "CHEK2"]),
            ("Cell Cycle", ["CDK1", "CCND1", "RB1", "E2F1"]),
            ("Apoptosis", ["TP53", "BCL2", "BAX", "CASP3"]),
            ("PI3K/AKT Signaling", ["PIK3CA", "AKT1", "PTEN", "mTOR"])
        ]
        
        for pathway_name, pathway_genes in pathways:
            instruction = f"Analyze {pathway_name} pathway enrichment in {experiment_context.organism} study"
            
            response = f"""Pathway Enrichment Analysis

Pathway: {pathway_name}
Study: {experiment_context.experiment_id}
Organism: {experiment_context.organism}

Key Genes in Pathway:
{chr(10).join([f"- {gene}: Core component" for gene in pathway_genes])}

Enrichment Assessment:
The {pathway_name} pathway shows potential enrichment based on the presence of {len(pathway_genes)} key regulatory genes.

Research Relevance:
For {experiment_context.research_objective}, the {pathway_name} pathway provides crucial biological context and may be dysregulated in your experimental conditions.

Experimental Validation:
Consider targeted analysis of {pathway_genes[0]} and {pathway_genes[1]} as representative pathway members for validation studies."""

            examples.append({
                "instruction": instruction,
                "input": "",
                "output": response
            })
        
        return examples
    
    def _generate_organism_examples(self,
                                  organism: str,
                                  genomic_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate organism-specific examples"""
        examples = []
        
        organism_contexts = {
            "human": "clinical relevance and therapeutic implications",
            "mouse": "model organism research and translational studies",
            "zebrafish": "developmental biology and toxicology studies",
            "drosophila": "genetic screening and fundamental biology",
            "arabidopsis": "plant genomics and agricultural applications",
            "yeast": "cell biology and biotechnology applications"
        }
        
        context = organism_contexts.get(organism.lower(), "comparative genomics and evolutionary studies")
        
        instruction = f"Provide {organism}-specific genomic analysis guidance"
        
        response = f"""Organism-Specific Genomic Analysis Guide

Target Organism: {organism}
Analysis Context: {context}

Genomic Characteristics:
- Organism-specific considerations for {organism}
- Relevant databases and resources
- Common analytical approaches
- Experimental design considerations

Analysis Framework:
1. Leverage {organism}-specific genomic databases
2. Consider organism-specific gene nomenclature
3. Apply appropriate statistical models for {organism} data
4. Integrate organism-specific biological pathways

Research Applications:
This analysis framework is optimized for {organism} genomics with focus on {context}."""

        examples.append({
            "instruction": instruction,
            "input": "",
            "output": response
        })
        
        return examples
    
    def _generate_literature_examples(self,
                                    publications: List[str],
                                    genomic_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate literature-informed examples"""
        examples = []
        
        # Create examples based on publication list
        for pub in publications[:3]:  # Limit to first 3 publications
            instruction = f"Integrate findings from {pub} into genomic analysis"
            
            response = f"""Literature Integration Analysis

Reference: {pub}

Genomic Analysis Integration:
- Apply methodological approaches from published study
- Consider findings in context of current dataset
- Identify overlapping analytical frameworks
- Validate results against published benchmarks

Experimental Design:
Incorporate best practices and validated approaches from {pub} to enhance the rigor and reproducibility of current genomic analysis.

Quality Assurance:
Use published study as methodological reference for ensuring analytical consistency and scientific validity."""

            examples.append({
                "instruction": instruction,
                "input": "",
                "output": response
            })
        
        return examples
    
    def _generate_temporal_examples(self, genomic_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate temporal analysis examples for longitudinal studies"""
        examples = []
        
        instruction = "Analyze temporal patterns in genomic data"
        
        response = """Temporal Genomic Analysis

Study Design: Longitudinal Analysis

Temporal Considerations:
1. Time-series expression profiling
2. Temporal variant emergence patterns
3. Dynamic network topology changes
4. Progression markers identification

Analytical Framework:
- Baseline establishment
- Temporal trajectory modeling
- Change point detection
- Predictive modeling for future time points

Statistical Approaches:
- Mixed-effects models for repeated measures
- Time-series analysis for expression data
- Survival analysis for outcome prediction
- Dynamic Bayesian networks for temporal dependencies"""

        examples.append({
            "instruction": instruction,
            "input": "",
            "output": response
        })
        
        return examples
    
    def _train_experiment_model(self,
                              training_data: Dataset,
                              lora_config: LoRAConfiguration,
                              training_config: Dict[str, Any],
                              output_dir: Path,
                              experiment_context: ExperimentContext) -> str:
        """Train the experiment-specific model"""
        
        logger.info("Setting up model and tokenizer for experiment-specific training")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with quantization if available
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=self.quantization_config,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Setup LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
            bias=lora_config.bias
        )
        
        model = get_peft_model(model, peft_config)
        
        # Tokenize dataset
        def tokenize_function(examples):
            texts = []
            for i in range(len(examples["instruction"])):
                text = f"### Instruction:\n{examples['instruction'][i]}\n\n### Response:\n{examples['output'][i]}"
                texts.append(text)
            
            return tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=training_config["max_length"],
                return_tensors="pt"
            )
        
        tokenized_dataset = training_data.map(tokenize_function, batched=True)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=training_config["num_epochs"],
            per_device_train_batch_size=training_config["batch_size"],
            learning_rate=training_config["learning_rate"],
            warmup_steps=training_config["warmup_steps"],
            logging_steps=training_config["logging_steps"],
            save_steps=training_config["save_steps"],
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=self.device == "cuda",
            dataloader_pin_memory=False,
            report_to="none"  # Disable wandb for now
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Train the model
        logger.info("Starting experiment-specific model training")
        trainer.train()
        
        # Save final model
        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        
        # Save experiment metadata
        self._save_experiment_metadata(final_model_path, experiment_context, lora_config, training_config)
        
        logger.info(f"Experiment-specific model training completed: {final_model_path}")
        return str(final_model_path)
    
    def _save_experiment_metadata(self,
                                model_path: Path,
                                experiment_context: ExperimentContext,
                                lora_config: LoRAConfiguration,
                                training_config: Dict[str, Any]):
        """Save experiment metadata with the model"""
        
        metadata = {
            "experiment_id": experiment_context.experiment_id,
            "research_objective": experiment_context.research_objective,
            "genomic_focus": experiment_context.genomic_focus,
            "organism": experiment_context.organism,
            "tissue_types": experiment_context.tissue_types,
            "temporal_scope": experiment_context.temporal_scope,
            "base_model": self.base_model,
            "lora_config": {
                "rank": lora_config.rank,
                "alpha": lora_config.alpha,
                "dropout": lora_config.dropout,
                "target_modules": lora_config.target_modules
            },
            "training_config": training_config,
            "created_at": pd.Timestamp.now().isoformat(),
            "device": self.device,
            "quantization": self.use_quantization
        }
        
        with open(model_path / "experiment_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Experiment metadata saved successfully")
    
    def load_experiment_model(self, experiment_id: str) -> Optional[str]:
        """Load an existing experiment-specific model"""
        
        model_path = self.models_directory / experiment_id / "final_model"
        
        if not model_path.exists():
            logger.error(f"Experiment model not found: {experiment_id}")
            return None
        
        # Load metadata
        metadata_path = model_path / "experiment_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            self.active_experiments[experiment_id] = {
                "model_path": str(model_path),
                "metadata": metadata,
                "loaded_at": pd.Timestamp.now()
            }
        
        logger.info(f"Loaded experiment model: {experiment_id}")
        return str(model_path)
    
    def list_experiment_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available experiment models"""
        
        models = {}
        
        for experiment_dir in self.models_directory.iterdir():
            if experiment_dir.is_dir():
                metadata_path = experiment_dir / "final_model" / "experiment_metadata.json"
                
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    models[experiment_dir.name] = {
                        "experiment_id": metadata.get("experiment_id", experiment_dir.name),
                        "research_objective": metadata.get("research_objective", "Unknown"),
                        "organism": metadata.get("organism", "Unknown"),
                        "created_at": metadata.get("created_at", "Unknown"),
                        "model_path": str(experiment_dir / "final_model")
                    }
        
        return models
    
    def delete_experiment_model(self, experiment_id: str) -> bool:
        """Delete an experiment model"""
        
        model_path = self.models_directory / experiment_id
        
        if not model_path.exists():
            logger.warning(f"Experiment model not found: {experiment_id}")
            return False
        
        try:
            import shutil
            shutil.rmtree(model_path)
            
            # Remove from active experiments
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            
            logger.info(f"Deleted experiment model: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting experiment model {experiment_id}: {e}")
            return False
    
    def get_experiment_stats(self) -> Dict[str, Any]:
        """Get statistics about experiment models"""
        
        stats = {
            "total_experiments": len(self.list_experiment_models()),
            "active_experiments": len(self.active_experiments),
            "models_directory": str(self.models_directory),
            "base_model": self.base_model,
            "device": self.device,
            "quantization_enabled": self.use_quantization
        }
        
        # Add organism distribution
        models = self.list_experiment_models()
        organisms = [model["organism"] for model in models.values()]
        stats["organism_distribution"] = {org: organisms.count(org) for org in set(organisms)}
        
        return stats


# Convenience function for quick experiment LLM creation
def create_experiment_llm(experiment_id: str,
                         research_objective: str,
                         genomic_data: Dict[str, Any],
                         organism: str = "human",
                         genomic_focus: List[str] = None,
                         base_model: str = "microsoft/DialoGPT-medium") -> str:
    """
    Quick function to create an experiment-specific LLM.
    
    Args:
        experiment_id: Unique identifier for the experiment
        research_objective: Research objective description
        genomic_data: Genomic datasets
        organism: Target organism
        genomic_focus: List of genomic focus areas
        base_model: Base model for fine-tuning
        
    Returns:
        Path to trained experiment-specific model
    """
    
    if genomic_focus is None:
        genomic_focus = ["variant_analysis", "expression_profiling"]
    
    # Create experiment context
    experiment_context = ExperimentContext(
        experiment_id=experiment_id,
        research_objective=research_objective,
        genomic_focus=genomic_focus,
        tissue_types=["mixed"],
        organism=organism,
        publications=[],
        temporal_scope="cross_sectional",
        collaboration_partners=[],
        expected_sample_size=100,
        computational_budget="30_minutes"
    )
    
    # Initialize manager
    manager = ExperimentLLMManager(base_model=base_model)
    
    # Create experiment LLM
    return manager.create_experiment_llm(experiment_context, genomic_data) 