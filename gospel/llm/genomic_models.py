"""
Genomic model manager for specialized Hugging Face models.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from gospel.llm.model import GENOMIC_MODELS, HuggingFaceGenomicModel


class GenomicModelManager:
    """
    Manager for specialized genomic models from Hugging Face Hub.
    """
    
    def __init__(self, cache_dir: str = "~/.gospel/models", hf_token: Optional[str] = None):
        """
        Initialize the genomic model manager.
        
        Args:
            cache_dir: Directory to cache model information
            hf_token: Hugging Face API token
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token
        self.loaded_models = {}
        
        # Load cached model info
        self.model_info_file = self.cache_dir / "model_info.json"
        self.model_info = self._load_model_info()
    
    def _load_model_info(self) -> Dict:
        """Load cached model information."""
        if self.model_info_file.exists():
            try:
                with open(self.model_info_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load model info cache: {e}")
        return {}
    
    def _save_model_info(self):
        """Save model information to cache."""
        try:
            with open(self.model_info_file, 'w') as f:
                json.dump(self.model_info, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save model info cache: {e}")
    
    def list_available_models(self) -> Dict[str, Dict]:
        """
        List all available genomic models.
        
        Returns:
            Dictionary of available models with their configurations
        """
        return GENOMIC_MODELS
    
    def get_models_by_type(self, model_type: str) -> Dict[str, Dict]:
        """
        Get models filtered by type.
        
        Args:
            model_type: Type of model ("dna_sequence", "protein_sequence", etc.)
            
        Returns:
            Dictionary of models of the specified type
        """
        return {
            name: config for name, config in GENOMIC_MODELS.items() 
            if config["type"] == model_type
        }
    
    def get_models_by_task(self, task: str) -> Dict[str, Dict]:
        """
        Get models filtered by task.
        
        Args:
            task: Task type ("fill-mask", "generation", "sequence-classification")
            
        Returns:
            Dictionary of models for the specified task
        """
        return {
            name: config for name, config in GENOMIC_MODELS.items() 
            if config["task"] == task
        }
    
    def recommend_models_for_analysis(self, analysis_type: str) -> List[str]:
        """
        Recommend models for a specific analysis type.
        
        Args:
            analysis_type: Type of analysis ("variant_effect", "protein_function", 
                          "dna_analysis", "sequence_generation")
            
        Returns:
            List of recommended model names
        """
        recommendations = {
            "variant_effect": ["caduceus", "nucleotide_transformer"],
            "protein_function": ["esm2", "protbert"],
            "dna_analysis": ["caduceus", "nucleotide_transformer", "gene42"],
            "sequence_generation": ["gene42", "mammal_biomed"],
            "multimodal_analysis": ["mammal_biomed"],
            "long_range_genomic": ["caduceus", "gene42"]
        }
        
        return recommendations.get(analysis_type, list(GENOMIC_MODELS.keys()))
    
    def load_model(self, model_name: str, device: str = "auto") -> Optional[HuggingFaceGenomicModel]:
        """
        Load a specific genomic model.
        
        Args:
            model_name: Name of the model to load
            device: Device to load the model on
            
        Returns:
            Loaded HuggingFaceGenomicModel instance or None if failed
        """
        if model_name in self.loaded_models:
            print(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]
        
        if model_name not in GENOMIC_MODELS:
            print(f"Unknown model: {model_name}")
            print(f"Available models: {list(GENOMIC_MODELS.keys())}")
            return None
        
        try:
            model_config = GENOMIC_MODELS[model_name]
            model = HuggingFaceGenomicModel(
                model_config,
                hf_token=self.hf_token,
                device=device
            )
            
            if model.model is not None:
                self.loaded_models[model_name] = model
                
                # Update model info cache
                self.model_info[model_name] = {
                    "status": "loaded",
                    "device": device,
                    "model_config": model_config
                }
                self._save_model_info()
                
                return model
            else:
                print(f"Failed to load model {model_name}")
                return None
                
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def load_models_for_analysis(self, analysis_type: str, device: str = "auto") -> Dict[str, HuggingFaceGenomicModel]:
        """
        Load recommended models for a specific analysis type.
        
        Args:
            analysis_type: Type of analysis
            device: Device to load models on
            
        Returns:
            Dictionary of loaded models
        """
        recommended = self.recommend_models_for_analysis(analysis_type)
        loaded = {}
        
        for model_name in recommended:
            model = self.load_model(model_name, device)
            if model:
                loaded[model_name] = model
        
        return loaded
    
    def unload_model(self, model_name: str):
        """
        Unload a model to free memory.
        
        Args:
            model_name: Name of the model to unload
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            
            # Update cache
            if model_name in self.model_info:
                self.model_info[model_name]["status"] = "unloaded"
                self._save_model_info()
            
            print(f"Unloaded model: {model_name}")
        else:
            print(f"Model {model_name} not currently loaded")
    
    def unload_all_models(self):
        """Unload all models to free memory."""
        model_names = list(self.loaded_models.keys())
        for model_name in model_names:
            self.unload_model(model_name)
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in GENOMIC_MODELS:
            return None
        
        config = GENOMIC_MODELS[model_name]
        info = {
            "name": model_name,
            "model_id": config["model_id"],
            "type": config["type"],
            "description": config["description"],
            "task": config["task"],
            "max_length": config["max_length"],
            "loaded": model_name in self.loaded_models,
            "cached_info": self.model_info.get(model_name, {})
        }
        
        return info
    
    def get_memory_usage(self) -> Dict:
        """
        Get estimated memory usage of loaded models.
        
        Returns:
            Dictionary with memory usage information
        """
        import torch
        
        usage = {
            "loaded_models": len(self.loaded_models),
            "model_details": {}
        }
        
        if torch.cuda.is_available():
            usage["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            usage["gpu_memory_cached"] = torch.cuda.memory_reserved()
        
        for model_name, model in self.loaded_models.items():
            model_info = {
                "device": str(model.device),
                "model_type": model.model_type
            }
            
            # Try to get parameter count
            try:
                if hasattr(model.model, 'num_parameters'):
                    model_info["parameters"] = model.model.num_parameters()
                elif hasattr(model.model, 'config') and hasattr(model.model.config, 'num_parameters'):
                    model_info["parameters"] = model.model.config.num_parameters
            except:
                pass
            
            usage["model_details"][model_name] = model_info
        
        return usage
    
    def benchmark_model(self, model_name: str, test_sequences: List[str]) -> Dict:
        """
        Benchmark a model with test sequences.
        
        Args:
            model_name: Name of the model to benchmark
            test_sequences: List of test sequences
            
        Returns:
            Benchmark results
        """
        if model_name not in self.loaded_models:
            print(f"Model {model_name} not loaded. Loading now...")
            model = self.load_model(model_name)
            if not model:
                return {"error": f"Could not load model {model_name}"}
        else:
            model = self.loaded_models[model_name]
        
        import time
        
        results = {
            "model_name": model_name,
            "num_sequences": len(test_sequences),
            "sequence_lengths": [len(seq) for seq in test_sequences],
            "predictions": [],
            "timing": {}
        }
        
        start_time = time.time()
        
        for i, sequence in enumerate(test_sequences):
            seq_start = time.time()
            prediction = model.predict(sequence)
            seq_end = time.time()
            
            results["predictions"].append({
                "sequence_index": i,
                "sequence_length": len(sequence),
                "prediction_time": seq_end - seq_start,
                "success": "error" not in prediction
            })
        
        end_time = time.time()
        
        results["timing"] = {
            "total_time": end_time - start_time,
            "average_time_per_sequence": (end_time - start_time) / len(test_sequences),
            "sequences_per_second": len(test_sequences) / (end_time - start_time)
        }
        
        return results
    
    def validate_sequence(self, sequence: str, sequence_type: str) -> Tuple[bool, str]:
        """
        Validate a sequence for genomic analysis.
        
        Args:
            sequence: Sequence to validate
            sequence_type: Type of sequence ("dna", "rna", "protein")
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not sequence:
            return False, "Empty sequence"
        
        sequence = sequence.upper().strip()
        
        if sequence_type.lower() == "dna":
            valid_chars = set("ATCGN")
            if not set(sequence).issubset(valid_chars):
                invalid_chars = set(sequence) - valid_chars
                return False, f"Invalid DNA characters: {invalid_chars}"
        
        elif sequence_type.lower() == "rna":
            valid_chars = set("AUCGN")
            if not set(sequence).issubset(valid_chars):
                invalid_chars = set(sequence) - valid_chars
                return False, f"Invalid RNA characters: {invalid_chars}"
        
        elif sequence_type.lower() == "protein":
            # Standard amino acids
            valid_chars = set("ACDEFGHIKLMNPQRSTVWYXBZJUO*")
            if not set(sequence).issubset(valid_chars):
                invalid_chars = set(sequence) - valid_chars
                return False, f"Invalid protein characters: {invalid_chars}"
        
        # Check length constraints
        if len(sequence) < 10:
            return False, f"Sequence too short: {len(sequence)} characters (minimum 10)"
        
        if len(sequence) > 200000:  # Reasonable upper limit
            return False, f"Sequence too long: {len(sequence)} characters (maximum 200,000)"
        
        return True, "Valid sequence"


def create_analysis_config(
    analysis_type: str,
    sequence_type: str = "dna",
    models: Optional[List[str]] = None,
    device: str = "auto"
) -> Dict:
    """
    Create a configuration for genomic analysis.
    
    Args:
        analysis_type: Type of analysis to perform
        sequence_type: Type of sequences to analyze
        models: Specific models to use (optional)
        device: Device to use for computation
        
    Returns:
        Configuration dictionary
    """
    manager = GenomicModelManager()
    
    if models is None:
        models = manager.recommend_models_for_analysis(analysis_type)
    
    # Filter models by sequence type compatibility
    compatible_models = []
    for model_name in models:
        if model_name in GENOMIC_MODELS:
            model_config = GENOMIC_MODELS[model_name]
            if (sequence_type == "dna" and "dna" in model_config["type"]) or \
               (sequence_type == "protein" and "protein" in model_config["type"]) or \
               ("multimodal" in model_config["type"]):
                compatible_models.append(model_name)
    
    return {
        "analysis_type": analysis_type,
        "sequence_type": sequence_type,
        "models": compatible_models,
        "device": device,
        "recommended_batch_size": 1 if device == "cpu" else 4,
        "max_sequence_length": min([GENOMIC_MODELS[m]["max_length"] for m in compatible_models]) if compatible_models else 1000
    } 