"""
Core LLM model for Gospel.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import ollama
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from huggingface_hub import login
import logging

# Configure logging for HuggingFace models
logging.getLogger("transformers").setLevel(logging.WARNING)

# Specialized genomic models available on Hugging Face Hub
GENOMIC_MODELS = {
    "caduceus": {
        "model_id": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
        "type": "dna_sequence",
        "description": "Caduceus: Bi-directional equivariant long-range DNA sequence modeling",
        "task": "fill-mask",
        "max_length": 131072
    },
    "nucleotide_transformer": {
        "model_id": "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
        "type": "dna_sequence", 
        "description": "Nucleotide Transformer: Foundation model for human genomics",
        "task": "fill-mask",
        "max_length": 1000
    },
    "gene42": {
        "model_id": "kuleshov-group/gene42-192k",  # This would be the actual model ID when available
        "type": "genomic_foundation",
        "description": "Gene42: Long-range genomic foundation model with dense attention", 
        "task": "generation",
        "max_length": 192000
    },
    "mammal_biomed": {
        "model_id": "ibm/biomed.omics.bl.sm.ma-ted-458m",
        "type": "multimodal_biomed",
        "description": "MAMMAL: Molecular aligned multi-modal architecture for biomedical data",
        "task": "generation",
        "max_length": 2048
    },
    "esm2": {
        "model_id": "facebook/esm2_t33_650M_UR50D",
        "type": "protein_sequence",
        "description": "ESM-2: Protein language model for structure and function prediction",
        "task": "fill-mask", 
        "max_length": 1024
    },
    "protbert": {
        "model_id": "Rostlab/prot_bert",
        "type": "protein_sequence",
        "description": "ProtBERT: Protein language model",
        "task": "fill-mask",
        "max_length": 512
    }
}


class HuggingFaceGenomicModel:
    """
    Wrapper for Hugging Face genomic models.
    """
    
    def __init__(self, model_config: Dict, hf_token: Optional[str] = None, device: str = "auto"):
        """
        Initialize a Hugging Face genomic model.
        
        Args:
            model_config: Configuration dictionary for the model
            hf_token: Hugging Face API token for private models
            device: Device to load the model on
        """
        self.model_config = model_config
        self.model_id = model_config["model_id"]
        self.model_type = model_config["type"]
        self.task = model_config["task"]
        self.max_length = model_config["max_length"]
        
        # Login to HuggingFace if token provided
        if hf_token:
            login(token=hf_token)
        
        # Configure device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load the model and tokenizer from Hugging Face Hub."""
        try:
            print(f"Loading {self.model_id} for task: {self.task}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Configure quantization for large models if on GPU
            quantization_config = None
            if self.device == "cuda" and "2.5b" in self.model_id.lower():
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
            
            # Load model based on task type
            if self.task == "fill-mask":
                self.model = AutoModelForMaskedLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None
                )
            elif self.task == "generation":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None
                )
            elif self.task == "sequence-classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None
                )
            
            if not quantization_config:
                self.model = self.model.to(self.device)
            
            print(f"Successfully loaded {self.model_id}")
            
        except Exception as e:
            print(f"Error loading model {self.model_id}: {e}")
            self.model = None
            self.tokenizer = None
    
    def predict(self, sequence: str, **kwargs) -> Dict:
        """
        Make predictions using the genomic model.
        
        Args:
            sequence: Input genomic/protein sequence
            **kwargs: Additional arguments for prediction
            
        Returns:
            Dictionary containing predictions and metadata
        """
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                sequence,
                return_tensors="pt",
                max_length=min(len(sequence) + 10, self.max_length),
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                if self.task == "fill-mask":
                    outputs = self.model(**inputs)
                    # Get embeddings from last hidden state
                    embeddings = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
                    
                    return {
                        "embeddings": embeddings.cpu().numpy(),
                        "logits": outputs.logits.cpu().numpy(),
                        "sequence_length": len(sequence),
                        "model_type": self.model_type,
                        "model_id": self.model_id
                    }
                    
                elif self.task == "generation":
                    # Generate text/sequences
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=kwargs.get("max_new_tokens", 50),
                        num_return_sequences=kwargs.get("num_return_sequences", 1),
                        temperature=kwargs.get("temperature", 0.7),
                        do_sample=kwargs.get("do_sample", True)
                    )
                    
                    generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                    
                    return {
                        "generated_sequence": generated_text,
                        "input_sequence": sequence,
                        "model_type": self.model_type,
                        "model_id": self.model_id
                    }
                    
                elif self.task == "sequence-classification":
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    
                    return {
                        "predictions": predictions.cpu().numpy(),
                        "logits": outputs.logits.cpu().numpy(),
                        "sequence": sequence,
                        "model_type": self.model_type,
                        "model_id": self.model_id
                    }
                    
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
    
    def get_embeddings(self, sequence: str) -> Optional[torch.Tensor]:
        """
        Get sequence embeddings.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Tensor containing sequence embeddings
        """
        result = self.predict(sequence)
        if "embeddings" in result:
            return torch.tensor(result["embeddings"])
        return None


class GospelLLM:
    """
    Domain-specific LLM for genomic analysis, supporting both Ollama and Hugging Face models.
    """

    def __init__(
        self,
        base_model: str = "llama3",
        model_dir: Optional[str] = None,
        genomic_models: Optional[List[str]] = None,
        hf_token: Optional[str] = None,
        use_ollama: bool = True,
        device: str = "auto"
    ):
        """
        Initialize a Gospel LLM.

        Args:
            base_model: Base Ollama model name or HuggingFace model ID
            model_dir: Directory containing a trained model
            genomic_models: List of specialized genomic models to load
            hf_token: Hugging Face API token
            use_ollama: Whether to use Ollama for the main LLM
            device: Device for HuggingFace models
        """
        self.base_model = base_model
        self.model_dir = model_dir
        self.genomic_networks = {}
        self.fine_tuned = False
        self.use_ollama = use_ollama
        self.hf_token = hf_token
        self.device = device
        
        # Initialize specialized genomic models
        self.genomic_models = {}
        if genomic_models:
            self._load_genomic_models(genomic_models)
        
        # Initialize the base model
        if use_ollama:
            try:
                # Simple Ollama client without LangChain
                self.llm = ollama
                print(f"Initialized Ollama client for model: {base_model}")
            except Exception as e:
                print(f"Error initializing Ollama: {e}")
                print("Please ensure Ollama is installed and running.")
                self.llm = None
        else:
            # Use HuggingFace model as main LLM
            try:
                self.llm = HuggingFaceGenomicModel(
                    {"model_id": base_model, "type": "general", "task": "generation", "max_length": 2048},
                    hf_token=hf_token,
                    device=device
                )
                print(f"Initialized HuggingFace model: {base_model}")
            except Exception as e:
                print(f"Error initializing HuggingFace model: {e}")
                self.llm = None

        # Load model if model_dir is provided
        if model_dir:
            self.load(model_dir)
    
    def _load_genomic_models(self, model_names: List[str]):
        """
        Load specialized genomic models from Hugging Face.
        
        Args:
            model_names: List of model names to load
        """
        for model_name in model_names:
            if model_name in GENOMIC_MODELS:
                try:
                    print(f"Loading genomic model: {model_name}")
                    model_config = GENOMIC_MODELS[model_name]
                    self.genomic_models[model_name] = HuggingFaceGenomicModel(
                        model_config,
                        hf_token=self.hf_token,
                        device=self.device
                    )
                    print(f"Successfully loaded genomic model: {model_name}")
                except Exception as e:
                    print(f"Failed to load genomic model {model_name}: {e}")
            else:
                print(f"Unknown genomic model: {model_name}. Available models: {list(GENOMIC_MODELS.keys())}")
    
    def get_available_genomic_models(self) -> Dict[str, Dict]:
        """
        Get information about available genomic models.
        
        Returns:
            Dictionary of available genomic models and their configurations
        """
        return GENOMIC_MODELS
    
    def analyze_sequence(self, sequence: str, sequence_type: str = "dna", models: Optional[List[str]] = None) -> Dict:
        """
        Analyze a genomic/protein sequence using specialized models.
        
        Args:
            sequence: The sequence to analyze
            sequence_type: Type of sequence ("dna", "protein", "rna")
            models: Specific models to use for analysis
            
        Returns:
            Dictionary containing analysis results from different models
        """
        results = {
            "sequence": sequence,
            "sequence_type": sequence_type,
            "length": len(sequence),
            "analysis": {}
        }
        
        # Determine which models to use based on sequence type
        if models is None:
            if sequence_type.lower() == "dna":
                models = ["caduceus", "nucleotide_transformer"]
            elif sequence_type.lower() == "protein":
                models = ["esm2", "protbert"]
            else:
                models = list(self.genomic_models.keys())
        
        # Run analysis with each available model
        for model_name in models:
            if model_name in self.genomic_models:
                try:
                    model_result = self.genomic_models[model_name].predict(sequence)
                    results["analysis"][model_name] = model_result
                except Exception as e:
                    results["analysis"][model_name] = {"error": str(e)}
            else:
                results["analysis"][model_name] = {"error": f"Model {model_name} not loaded"}
        
        return results
    
    def predict_variant_effect(self, reference_sequence: str, variant_sequence: str, model_name: str = "caduceus") -> Dict:
        """
        Predict the effect of a genetic variant.
        
        Args:
            reference_sequence: Reference genomic sequence
            variant_sequence: Sequence with variant
            model_name: Model to use for prediction
            
        Returns:
            Dictionary containing variant effect predictions
        """
        if model_name not in self.genomic_models:
            return {"error": f"Model {model_name} not available"}
        
        try:
            # Analyze both sequences
            ref_result = self.genomic_models[model_name].predict(reference_sequence)
            var_result = self.genomic_models[model_name].predict(variant_sequence)
            
            # Compare embeddings if available
            if "embeddings" in ref_result and "embeddings" in var_result:
                import numpy as np
                ref_embedding = ref_result["embeddings"].mean(axis=1)  # Average over sequence length
                var_embedding = var_result["embeddings"].mean(axis=1)
                
                # Calculate similarity score
                similarity = np.corrcoef(ref_embedding.flatten(), var_embedding.flatten())[0, 1]
                effect_score = 1 - similarity  # Higher score = more effect
                
                return {
                    "reference_sequence": reference_sequence,
                    "variant_sequence": variant_sequence,
                    "model_used": model_name,
                    "similarity_score": float(similarity),
                    "variant_effect_score": float(effect_score),
                    "reference_analysis": ref_result,
                    "variant_analysis": var_result
                }
            else:
                return {
                    "reference_sequence": reference_sequence,
                    "variant_sequence": variant_sequence,
                    "model_used": model_name,
                    "reference_analysis": ref_result,
                    "variant_analysis": var_result,
                    "note": "Embedding comparison not available for this model"
                }
                
        except Exception as e:
            return {"error": f"Variant effect prediction failed: {e}"}

    def train(self, training_data: List[Dict], output_dir: str) -> None:
        """
        Train the LLM using training data (simplified version without knowledge base).

        Args:
            training_data: List of training examples
            output_dir: Output directory for the trained model
        """
        # Set up output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("Training functionality simplified - knowledge base training removed")
        print(f"Received {len(training_data)} training examples")
        
        # Save the model configuration
        model_config = {
            "base_model": self.base_model,
            "fine_tuned": True,
            "use_ollama": self.use_ollama,
            "genomic_models": list(self.genomic_models.keys()) if self.genomic_models else [],
        }
        
        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)
        
        # Save training data for reference
        with open(os.path.join(output_dir, "training_data.json"), "w") as f:
            json.dump(training_data, f, indent=2)
        
        self.model_dir = output_dir
        self.fine_tuned = True
        print(f"Model configuration saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "GospelLLM":
        """
        Load a trained model configuration from a directory.

        Args:
            model_dir: Directory containing a trained model

        Returns:
            GospelLLM instance
        """
        # Load model configuration
        with open(os.path.join(model_dir, "model_config.json"), "r") as f:
            model_config = json.load(f)
        
        # Create instance with base model and genomic models
        instance = cls(
            base_model=model_config["base_model"], 
            model_dir=model_dir,
            genomic_models=model_config.get("genomic_models", []),
            use_ollama=model_config.get("use_ollama", True)
        )
        
        instance.fine_tuned = model_config.get("fine_tuned", False)
        print(f"Loaded model configuration from {model_dir}")
        return instance

    def query(self, query: str) -> str:
        """
        Query the LLM.

        Args:
            query: Query string

        Returns:
            Response from the LLM
        """
        if self.use_ollama:
            if not self.llm:
                return "Error: LLM not initialized. Please ensure Ollama is installed and running."
            
            # Use Ollama client directly
            try:
                response = self.llm.chat(model=self.base_model, messages=[
                    {"role": "user", "content": query}
                ])
                return response["message"]["content"]
            except Exception as e:
                return f"Error querying Ollama: {e}"
        else:
            # Use HuggingFace model
            if not self.llm:
                return "Error: HuggingFace model not initialized."
            
            try:
                result = self.llm.predict(query, max_new_tokens=200)
                if "generated_sequence" in result:
                    return result["generated_sequence"]
                else:
                    return str(result)
            except Exception as e:
                return f"Error querying HuggingFace model: {e}"

    def solve_genomic_problem(self, problem: str) -> Dict:
        """
        Solve a genomic problem using the LLM and available genomic models.

        Args:
            problem: Problem description

        Returns:
            Dictionary containing the solution, method, and supporting information
        """
        solver_prompt = f"""
        You are a genomics expert with access to specialized genomic models.
        
        Problem: {problem}
        
        Available specialized models: {list(self.genomic_models.keys()) if self.genomic_models else "None"}
        
        Please provide:
        1. A step-by-step solution approach
        2. The genetic factors involved
        3. The specific answer to the problem
        4. Recommendations for specialized model analysis if applicable
        """
        
        # Query the solver
        solution_text = self.query(solver_prompt)
        
        return {
            "problem": problem,
            "solution": solution_text,
            "method": "LLM-based genomic reasoning with specialized models",
            "available_genomic_models": list(self.genomic_models.keys()) if self.genomic_models else []
        }

    def get_gene_network(self, gene_name: str):
        """
        Get the network associated with a gene.

        Args:
            gene_name: Gene name

        Returns:
            NetworkX graph for the gene network or None if not found
        """
        return self.genomic_networks.get(gene_name) 