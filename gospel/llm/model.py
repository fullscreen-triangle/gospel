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
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
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

from gospel.knowledge_base import KnowledgeBase

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
        kb: Optional[KnowledgeBase] = None,
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
            kb: Knowledge base instance
            model_dir: Directory containing a trained model
            genomic_models: List of specialized genomic models to load
            hf_token: Hugging Face API token
            use_ollama: Whether to use Ollama for the main LLM
            device: Device for HuggingFace models
        """
        self.base_model = base_model
        self.kb = kb
        self.model_dir = model_dir
        self.qa_chain = None
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
                self.llm = Ollama(model=base_model)
                print(f"Initialized Ollama model: {base_model}")
            except Exception as e:
                print(f"Error initializing Ollama model: {e}")
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

    def train(self, kb_dir: str, output_dir: str) -> None:
        """
        Train the LLM using the knowledge base.

        Args:
            kb_dir: Knowledge base directory
            output_dir: Output directory for the trained model
        """
        # Load knowledge base if not already loaded
        if not self.kb:
            self.kb = KnowledgeBase.load(kb_dir)
        
        # Set up output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Build domain-specific prompt templates
        gene_template = """
        You are an expert in genomics specializing in sprint performance analysis.
        Use the following context to answer the question about gene {gene_name}:
        
        {context}
        
        Question: {question}
        Answer:
        """
        
        gene_prompt = PromptTemplate(
            input_variables=["gene_name", "context", "question"],
            template=gene_template,
        )
        
        # Set up QA chain with the knowledge base and prompt
        if self.use_ollama and self.llm:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.kb.get_retriever(),
                chain_type_kwargs={"prompt": gene_prompt},
            )
        
        # Set up genomic networks from knowledge base
        self.genomic_networks = self.kb.get_gene_networks()
        
        # Save the model configuration
        model_config = {
            "base_model": self.base_model,
            "fine_tuned": True,
            "kb_dir": kb_dir,
            "use_ollama": self.use_ollama,
            "genomic_models": list(self.genomic_models.keys()) if self.genomic_models else [],
        }
        
        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)
        
        # Save the QA chain (note: some components might not be serializable)
        if self.qa_chain:
            try:
                with open(os.path.join(output_dir, "qa_chain.pkl"), "wb") as f:
                    pickle.dump(self.qa_chain, f)
            except Exception as e:
                print(f"Warning: Could not save full QA chain: {e}")
                print("Saving components separately...")
        
        # Save genomic networks
        with open(os.path.join(output_dir, "genomic_networks.json"), "w") as f:
            # Convert non-serializable objects to strings if needed
            serializable_networks = {}
            for gene, network in self.genomic_networks.items():
                serializable_networks[gene] = {
                    "nodes": list(network.nodes()),
                    "edges": list(network.edges()),
                    "node_attributes": {str(n): d for n, d in network.nodes(data=True)},
                    "edge_attributes": {str(e): d for e, d in network.edges(data=True)},
                }
            json.dump(serializable_networks, f, indent=2)
        
        # Save knowledge base reference
        if not os.path.exists(os.path.join(output_dir, "kb")):
            self.kb.save(os.path.join(output_dir, "kb"))
        
        self.model_dir = output_dir
        self.fine_tuned = True
        print(f"Model trained and saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "GospelLLM":
        """
        Load a trained model from a directory.

        Args:
            model_dir: Directory containing a trained model

        Returns:
            GospelLLM instance
        """
        # Load model configuration
        with open(os.path.join(model_dir, "model_config.json"), "r") as f:
            model_config = json.load(f)
        
        # Create instance with base model
        instance = cls(base_model=model_config["base_model"], model_dir=model_dir)
        
        # Load knowledge base
        kb_dir = os.path.join(model_dir, "kb")
        if os.path.exists(kb_dir):
            instance.kb = KnowledgeBase.load(kb_dir)
        elif "kb_dir" in model_config and os.path.exists(model_config["kb_dir"]):
            instance.kb = KnowledgeBase.load(model_config["kb_dir"])
        
        # Load QA chain if available
        qa_chain_path = os.path.join(model_dir, "qa_chain.pkl")
        if os.path.exists(qa_chain_path):
            try:
                with open(qa_chain_path, "rb") as f:
                    instance.qa_chain = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load QA chain: {e}")
                print("Reconstructing QA chain...")
                if instance.kb:
                    gene_template = """
                    You are an expert in genomics specializing in sprint performance analysis.
                    Use the following context to answer the question about genes and genomics:
                    
                    {context}
                    
                    Question: {question}
                    Answer:
                    """
                    
                    gene_prompt = PromptTemplate(
                        input_variables=["context", "question"],
                        template=gene_template,
                    )
                    
                    instance.qa_chain = RetrievalQA.from_chain_type(
                        llm=instance.llm,
                        chain_type="stuff",
                        retriever=instance.kb.get_retriever(),
                        chain_type_kwargs={"prompt": gene_prompt},
                    )
        
        # Load genomic networks
        networks_path = os.path.join(model_dir, "genomic_networks.json")
        if os.path.exists(networks_path):
            try:
                import networkx as nx
                
                with open(networks_path, "r") as f:
                    serialized_networks = json.load(f)
                
                instance.genomic_networks = {}
                for gene, data in serialized_networks.items():
                    G = nx.Graph()
                    G.add_nodes_from(data["nodes"])
                    G.add_edges_from(data["edges"])
                    
                    # Add attributes
                    for node, attrs in data["node_attributes"].items():
                        for key, value in attrs.items():
                            G.nodes[node][key] = value
                    
                    for edge, attrs in data["edge_attributes"].items():
                        # Convert edge string back to tuple
                        edge_tuple = eval(edge)
                        for key, value in attrs.items():
                            G.edges[edge_tuple][key] = value
                    
                    instance.genomic_networks[gene] = G
            except Exception as e:
                print(f"Warning: Could not load genomic networks: {e}")
        
        instance.fine_tuned = model_config.get("fine_tuned", False)
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
            
            # If we have a QA chain, use it
            if self.qa_chain:
                try:
                    result = self.qa_chain.run(query)
                    return result
                except Exception as e:
                    print(f"Error using QA chain: {e}")
                    print("Falling back to base model...")
            
            # Fall back to base model
            try:
                response = self.llm(query)
                return response
            except Exception as e:
                return f"Error querying LLM: {e}"
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
        Solve a genomic problem using the LLM and knowledge base.

        Args:
            problem: Problem description

        Returns:
            Dictionary containing the solution, method, and supporting information
        """
        if not self.fine_tuned:
            return {"error": "Model not fine-tuned. Please train the model first."}
        
        # Extract genes mentioned in the problem
        gene_contexts = {}
        for gene, network in self.genomic_networks.items():
            if gene.lower() in problem.lower():
                gene_contexts[gene] = {
                    "network": network,
                    "info": self.kb.get_gene_info(gene) if self.kb else "",
                }
        
        # Prepare a structured prompt
        context_prompt = ""
        for gene, context in gene_contexts.items():
            context_prompt += f"\nGene: {gene}\n"
            context_prompt += f"Information: {context['info']}\n"
            context_prompt += f"Network: {len(context['network'].nodes())} nodes, {len(context['network'].edges())} edges\n"
            context_prompt += f"Connected genes: {', '.join(list(context['network'].nodes())[:10])}\n"
        
        solver_prompt = f"""
        You are a genomics expert solver with access to specialized genomic models.
        
        Problem: {problem}
        
        Context information about relevant genes:
        {context_prompt}
        
        Available specialized models: {list(self.genomic_models.keys()) if self.genomic_models else "None"}
        
        Please provide:
        1. A step-by-step solution approach
        2. The genetic factors involved
        3. The specific answer to the problem
        4. Any relevant network interactions
        5. Recommendations for specialized model analysis if applicable
        """
        
        # Query the solver
        solution_text = self.query(solver_prompt)
        
        return {
            "problem": problem,
            "solution": solution_text,
            "method": "LLM-based genomic reasoning with specialized models",
            "genes": list(gene_contexts.keys()),
            "networks": {gene: list(context["network"].nodes())[:20] for gene, context in gene_contexts.items()},
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