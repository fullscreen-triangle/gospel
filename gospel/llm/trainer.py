#!/usr/bin/env python3
"""
Genomic LLM Trainer for Gospel Framework

Produces domain-expert genomic LLMs from real experimental data for integration
with combine-harvester, hegel, four-sided-triangle, and kwasa-kwasa ecosystem.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import wandb

from gospel.core import VariantProcessor, VariantAnnotator, GenomicScorer
from gospel.llm.distiller import SequenceDataDistiller

logger = logging.getLogger(__name__)


class GenomicLLMTrainer:
    """
    Comprehensive genomic LLM trainer for producing domain-expert models
    from real experimental genomic data.
    """
    
    def __init__(
        self,
        base_model: str = "microsoft/DialoGPT-medium",
        output_dir: str = "trained_models",
        use_lora: bool = True,
        device: str = "auto"
    ):
        """
        Initialize the genomic LLM trainer.
        
        Args:
            base_model: Base model for fine-tuning
            output_dir: Directory for saving trained models
            use_lora: Whether to use LoRA for efficient fine-tuning
            device: Device for training (auto, cuda, cpu)
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.use_lora = use_lora
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Core components
        self.variant_processor = VariantProcessor()
        self.annotator = VariantAnnotator()
        self.scorer = GenomicScorer()
        self.distiller = SequenceDataDistiller(None)
        
        # Training components
        self.tokenizer = None
        self.model = None
        self.training_dataset = None
        
        # Ecosystem compatibility
        self.evidence_format = "hegel_compatible"
        self.semantic_format = "kwasa_compatible"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_experimental_data(
        self, 
        data_paths: Dict[str, str],
        data_types: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load real experimental genomic data from various sources.
        
        Args:
            data_paths: Paths to experimental data files
            data_types: Types of data to process
            
        Returns:
            Dictionary of loaded and processed datasets
        """
        if data_types is None:
            data_types = ["sequences", "variants", "expression", "networks"]
            
        datasets = {}
        
        for data_type in data_types:
            if data_type in data_paths and os.path.exists(data_paths[data_type]):
                logger.info(f"Loading {data_type} data from {data_paths[data_type]}")
                
                if data_type == "sequences":
                    datasets[data_type] = self._load_sequence_data(data_paths[data_type])
                elif data_type == "variants":
                    datasets[data_type] = self._load_variant_data(data_paths[data_type])
                elif data_type == "expression":
                    datasets[data_type] = self._load_expression_data(data_paths[data_type])
                elif data_type == "networks":
                    datasets[data_type] = self._load_network_data(data_paths[data_type])
                    
        logger.info(f"Loaded {len(datasets)} experimental datasets")
        return datasets
    
    def _load_sequence_data(self, file_path: str) -> pd.DataFrame:
        """Load sequence data from FASTA, GenBank, or other formats."""
        from Bio import SeqIO
        
        sequences = []
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.fasta', '.fa', '.fas']:
            format_type = 'fasta'
        elif file_ext in ['.gb', '.genbank']:
            format_type = 'genbank'
        else:
            format_type = 'fasta'  # Default
            
        for record in SeqIO.parse(file_path, format_type):
            sequences.append({
                'id': record.id,
                'description': record.description,
                'sequence': str(record.seq),
                'length': len(record.seq),
                'gc_content': (str(record.seq).count('G') + str(record.seq).count('C')) / len(record.seq)
            })
            
        return pd.DataFrame(sequences)
    
    def _load_variant_data(self, file_path: str) -> pd.DataFrame:
        """Load variant data from VCF or CSV files."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.vcf':
            import cyvcf2
            variants = []
            
            vcf = cyvcf2.VCF(file_path)
            for variant in vcf:
                variants.append({
                    'chromosome': variant.CHROM,
                    'position': variant.POS,
                    'id': variant.ID or f"{variant.CHROM}:{variant.POS}",
                    'reference': variant.REF,
                    'alternate': variant.ALT[0] if variant.ALT else '',
                    'quality': variant.QUAL,
                    'filter': variant.FILTER,
                    'info': str(variant.INFO)
                })
                
            return pd.DataFrame(variants)
        else:
            return pd.read_csv(file_path)
    
    def _load_expression_data(self, file_path: str) -> pd.DataFrame:
        """Load gene expression data."""
        return pd.read_csv(file_path, index_col=0)
    
    def _load_network_data(self, file_path: str) -> pd.DataFrame:
        """Load network/interaction data."""
        return pd.read_csv(file_path)
    
    def process_experimental_data(
        self,
        datasets: Dict[str, pd.DataFrame],
        analysis_config: Dict = None
    ) -> Dict[str, Dict]:
        """
        Process experimental data using Gospel's analysis components.
        
        Args:
            datasets: Loaded experimental datasets
            analysis_config: Configuration for analysis
            
        Returns:
            Processed analysis results
        """
        if analysis_config is None:
            analysis_config = {
                "annotate_variants": True,
                "score_genes": True,
                "analyze_networks": True,
                "use_real_apis": True
            }
            
        results = {}
        
        # Process variants
        if "variants" in datasets and analysis_config.get("annotate_variants"):
            logger.info("Processing variants with real annotations")
            variant_results = []
            
            for _, variant_row in datasets["variants"].iterrows():
                # Create variant object
                from gospel.core.variant import Variant, VariantType
                
                variant = Variant(
                    chromosome=str(variant_row["chromosome"]),
                    position=int(variant_row["position"]),
                    reference=variant_row["reference"],
                    alternate=variant_row["alternate"],
                    variant_id=variant_row.get("id", f"{variant_row['chromosome']}:{variant_row['position']}")
                )
                
                # Annotate with real data
                annotated_variant = self.annotator.annotate_variant(variant)
                
                variant_results.append({
                    "variant_id": variant.id,
                    "annotations": annotated_variant.annotations,
                    "clinical_significance": annotated_variant.annotations.get("clinvar", {}).get("clinical_significance", "Unknown"),
                    "functional_impact": annotated_variant.annotations.get("vep", {}).get("impact", "Unknown")
                })
                
            results["variants"] = {"processed_variants": variant_results}
        
        # Process sequences
        if "sequences" in datasets:
            logger.info("Analyzing sequences with HuggingFace models")
            sequence_results = self._analyze_sequences_with_hf(datasets["sequences"])
            results["sequences"] = sequence_results
        
        # Process expression data
        if "expression" in datasets and analysis_config.get("score_genes"):
            logger.info("Scoring genes from expression data")
            gene_scores = self._score_genes_from_expression(datasets["expression"])
            results["expression"] = gene_scores
        
        # Process network data
        if "networks" in datasets and analysis_config.get("analyze_networks"):
            logger.info("Analyzing biological networks")
            network_analysis = self._analyze_networks(datasets["networks"])
            results["networks"] = network_analysis
            
        return results
    
    def _analyze_sequences_with_hf(self, sequences_df: pd.DataFrame) -> Dict:
        """Analyze sequences using HuggingFace genomic models."""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            # Load genomic model
            tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
            model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
            
            analysis_results = []
            
            for _, seq_row in sequences_df.iterrows():
                sequence = seq_row["sequence"]
                
                # Tokenize and analyze
                inputs = tokenizer(sequence[:512], return_tensors="pt", truncation=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                
                analysis_results.append({
                    "sequence_id": seq_row["id"],
                    "length": len(sequence),
                    "gc_content": seq_row["gc_content"],
                    "embedding_mean": float(np.mean(embeddings)),
                    "embedding_std": float(np.std(embeddings)),
                    "functional_prediction": self._predict_function(sequence)
                })
                
            return {"sequence_analyses": analysis_results}
            
        except Exception as e:
            logger.error(f"HuggingFace analysis failed: {e}")
            return {"sequence_analyses": []}
    
    def _predict_function(self, sequence: str) -> Dict:
        """Simple functional prediction based on sequence characteristics."""
        predictions = {}
        
        # Promoter prediction
        if "TATAAA" in sequence:
            predictions["promoter_likelihood"] = 0.8
        else:
            predictions["promoter_likelihood"] = 0.2
            
        # CpG island prediction
        cpg_count = sequence.count("CG")
        cpg_density = cpg_count / len(sequence) if len(sequence) > 0 else 0
        predictions["cpg_island_likelihood"] = min(1.0, cpg_density * 10)
        
        return predictions
    
    def _score_genes_from_expression(self, expression_df: pd.DataFrame) -> Dict:
        """Score genes based on expression patterns."""
        gene_scores = {}
        
        for gene in expression_df.index:
            expression_values = expression_df.loc[gene].values
            
            # Calculate various metrics
            mean_expr = np.mean(expression_values)
            std_expr = np.std(expression_values)
            cv = std_expr / mean_expr if mean_expr > 0 else 0
            
            gene_scores[gene] = {
                "mean_expression": float(mean_expr),
                "expression_variability": float(cv),
                "significance_score": float(mean_expr * (1 - cv))
            }
            
        return {"gene_scores": gene_scores}
    
    def _analyze_networks(self, network_df: pd.DataFrame) -> Dict:
        """Analyze biological networks."""
        import networkx as nx
        
        # Create network graph
        G = nx.from_pandas_edgelist(
            network_df, 
            source=network_df.columns[0], 
            target=network_df.columns[1],
            edge_attr=True
        )
        
        # Calculate network metrics
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        network_analysis = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "top_central_nodes": sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_betweenness_nodes": sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
        return network_analysis
    
    def create_training_dataset(
        self,
        processed_results: Dict[str, Dict],
        dataset_config: Dict = None
    ) -> Dataset:
        """
        Create training dataset from processed experimental results.
        
        Args:
            processed_results: Results from process_experimental_data
            dataset_config: Configuration for dataset creation
            
        Returns:
            HuggingFace Dataset for training
        """
        if dataset_config is None:
            dataset_config = {
                "include_variants": True,
                "include_sequences": True,
                "include_expression": True,
                "include_networks": True,
                "max_examples": 10000
            }
            
        training_examples = []
        
        # Create training examples from variants
        if "variants" in processed_results and dataset_config.get("include_variants"):
            for variant in processed_results["variants"]["processed_variants"]:
                instruction = f"Analyze the clinical significance of variant {variant['variant_id']}"
                
                response = f"""This variant has the following characteristics:
- Clinical significance: {variant['clinical_significance']}
- Functional impact: {variant['functional_impact']}
- Annotation confidence: {variant['annotations'].get('composite_score', {}).get('confidence', 'medium')}

Based on the evidence, this variant should be considered {variant['clinical_significance'].lower()} with {variant['functional_impact'].lower()} functional impact."""

                training_examples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": response
                })
        
        # Create training examples from sequences
        if "sequences" in processed_results and dataset_config.get("include_sequences"):
            for seq_analysis in processed_results["sequences"]["sequence_analyses"]:
                instruction = f"Analyze the functional potential of sequence {seq_analysis['sequence_id']}"
                
                response = f"""Sequence analysis reveals:
- Length: {seq_analysis['length']} bp
- GC content: {seq_analysis['gc_content']:.2%}
- Promoter likelihood: {seq_analysis['functional_prediction']['promoter_likelihood']:.2f}
- CpG island likelihood: {seq_analysis['functional_prediction']['cpg_island_likelihood']:.2f}

This sequence shows characteristics consistent with regulatory elements and may play a role in gene expression control."""

                training_examples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": response
                })
        
        # Create training examples from expression data
        if "expression" in processed_results and dataset_config.get("include_expression"):
            gene_scores = processed_results["expression"]["gene_scores"]
            
            for gene, scores in list(gene_scores.items())[:100]:  # Limit examples
                instruction = f"Interpret the expression profile of gene {gene}"
                
                response = f"""Gene expression analysis for {gene}:
- Mean expression level: {scores['mean_expression']:.2f}
- Expression variability (CV): {scores['expression_variability']:.2f}
- Biological significance score: {scores['significance_score']:.2f}

This gene shows {'high' if scores['mean_expression'] > 5 else 'moderate' if scores['mean_expression'] > 1 else 'low'} expression with {'stable' if scores['expression_variability'] < 0.5 else 'variable'} patterns across conditions."""

                training_examples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": response
                })
        
        # Limit dataset size
        if len(training_examples) > dataset_config.get("max_examples", 10000):
            training_examples = training_examples[:dataset_config["max_examples"]]
            
        logger.info(f"Created {len(training_examples)} training examples")
        
        # Convert to HuggingFace Dataset
        return Dataset.from_list(training_examples)
    
    def setup_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Setup model and tokenizer for training."""
        logger.info(f"Loading base model: {self.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Setup LoRA if requested
        if self.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info("Configured model with LoRA")
        
        return self.model, self.tokenizer
    
    def train_model(
        self,
        training_dataset: Dataset,
        training_config: Dict = None
    ) -> str:
        """
        Train the genomic LLM.
        
        Args:
            training_dataset: Dataset for training
            training_config: Training configuration
            
        Returns:
            Path to trained model
        """
        if training_config is None:
            training_config = {
                "num_epochs": 3,
                "batch_size": 4,
                "learning_rate": 5e-5,
                "warmup_steps": 100,
                "logging_steps": 50,
                "save_steps": 500,
                "eval_steps": 500
            }
            
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Tokenize dataset
        def tokenize_function(examples):
            # Format as instruction-following
            texts = []
            for i in range(len(examples["instruction"])):
                text = f"### Instruction:\n{examples['instruction'][i]}\n\n### Response:\n{examples['output'][i]}"
                texts.append(text)
                
            return self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = training_dataset.map(tokenize_function, batched=True)
        
        # Setup training arguments
        model_output_dir = self.output_dir / "genomic_llm"
        
        training_args = TrainingArguments(
            output_dir=str(model_output_dir),
            num_train_epochs=training_config["num_epochs"],
            per_device_train_batch_size=training_config["batch_size"],
            learning_rate=training_config["learning_rate"],
            warmup_steps=training_config["warmup_steps"],
            logging_steps=training_config["logging_steps"],
            save_steps=training_config["save_steps"],
            evaluation_strategy="steps",
            eval_steps=training_config["eval_steps"],
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=self.device == "cuda",
            dataloader_pin_memory=False,
            report_to="wandb" if wandb.run else "none"
        )
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        logger.info("Starting model training...")
        trainer.train()
        
        # Save final model
        final_model_path = model_output_dir / "final"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        logger.info(f"Model training completed. Saved to: {final_model_path}")
        
        # Save ecosystem compatibility metadata
        self._save_ecosystem_metadata(final_model_path)
        
        return str(final_model_path)
    
    def _save_ecosystem_metadata(self, model_path: Path) -> None:
        """Save metadata for ecosystem integration."""
        metadata = {
            "model_type": "genomic_domain_expert",
            "base_model": self.base_model,
            "framework": "gospel",
            "version": "0.1.0",
            "capabilities": [
                "variant_interpretation",
                "sequence_analysis", 
                "gene_expression_analysis",
                "network_analysis"
            ],
            "compatible_with": [
                "combine-harvester",
                "hegel",
                "four-sided-triangle", 
                "kwasa-kwasa"
            ],
            "evidence_format": self.evidence_format,
            "semantic_format": self.semantic_format
        }
        
        with open(model_path / "ecosystem_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info("Saved ecosystem compatibility metadata")
    
    def export_for_ecosystem(
        self,
        model_path: str,
        export_config: Dict = None
    ) -> Dict[str, str]:
        """
        Export trained model for ecosystem integration.
        
        Args:
            model_path: Path to trained model
            export_config: Export configuration
            
        Returns:
            Dictionary of export paths
        """
        if export_config is None:
            export_config = {
                "combine_harvester": True,
                "hegel": True,
                "ollama": True
            }
            
        export_paths = {}
        model_path = Path(model_path)
        
        # Export for combine-harvester
        if export_config.get("combine_harvester"):
            ch_export_path = model_path / "combine_harvester_export"
            ch_export_path.mkdir(exist_ok=True)
            
            # Create combine-harvester compatible config
            ch_config = {
                "model_path": str(model_path),
                "model_type": "domain_expert",
                "domain": "genomics",
                "capabilities": ["variant_analysis", "sequence_interpretation", "gene_scoring"],
                "router_priority": 0.9,
                "chain_compatibility": ["biomedical", "clinical", "research"]
            }
            
            with open(ch_export_path / "model_config.json", "w") as f:
                json.dump(ch_config, f, indent=2)
                
            export_paths["combine_harvester"] = str(ch_export_path)
        
        # Export for hegel (evidence format)
        if export_config.get("hegel"):
            hegel_export_path = model_path / "hegel_export"
            hegel_export_path.mkdir(exist_ok=True)
            
            hegel_config = {
                "evidence_types": ["genomic_variant", "sequence_functional", "expression_significance"],
                "confidence_scoring": True,
                "bayesian_compatible": True,
                "molecular_types": ["DNA", "RNA", "protein"]
            }
            
            with open(hegel_export_path / "evidence_config.json", "w") as f:
                json.dump(hegel_config, f, indent=2)
                
            export_paths["hegel"] = str(hegel_export_path)
        
        # Export for Ollama
        if export_config.get("ollama"):
            ollama_export_path = model_path / "ollama_export"
            ollama_export_path.mkdir(exist_ok=True)
            
            modelfile_content = f"""FROM {model_path}

# Set parameters for genomic analysis
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# System message for genomic expertise
SYSTEM You are a genomic analysis expert specializing in variant interpretation, sequence analysis, and gene expression. You provide evidence-based insights for genomic research and clinical applications.

# Template for genomic queries
TEMPLATE "### Instruction: {{{{ .Prompt }}}}\\n\\n### Response:"
"""
            
            with open(ollama_export_path / "Modelfile", "w") as f:
                f.write(modelfile_content)
                
            export_paths["ollama"] = str(ollama_export_path)
        
        logger.info(f"Exported model for ecosystem integration: {list(export_paths.keys())}")
        return export_paths


def create_genomic_llm(
    experimental_data_paths: Dict[str, str],
    training_config: Dict = None,
    model_config: Dict = None
) -> str:
    """
    Complete pipeline to create a genomic LLM from experimental data.
    
    Args:
        experimental_data_paths: Paths to experimental data files
        training_config: Training configuration
        model_config: Model configuration
        
    Returns:
        Path to trained genomic LLM
    """
    if model_config is None:
        model_config = {
            "base_model": "microsoft/DialoGPT-medium",
            "use_lora": True,
            "output_dir": "trained_genomic_models"
        }
    
    # Initialize trainer
    trainer = GenomicLLMTrainer(**model_config)
    
    # Load experimental data
    logger.info("Loading experimental genomic data...")
    datasets = trainer.load_experimental_data(experimental_data_paths)
    
    # Process data
    logger.info("Processing experimental data...")
    processed_results = trainer.process_experimental_data(datasets)
    
    # Create training dataset
    logger.info("Creating training dataset...")
    training_dataset = trainer.create_training_dataset(processed_results)
    
    # Train model
    logger.info("Training genomic LLM...")
    model_path = trainer.train_model(training_dataset, training_config)
    
    # Export for ecosystem
    logger.info("Exporting for ecosystem integration...")
    export_paths = trainer.export_for_ecosystem(model_path)
    
    logger.info("Genomic LLM creation complete!")
    logger.info(f"Model available at: {model_path}")
    logger.info(f"Ecosystem exports: {export_paths}")
    
    return model_path 