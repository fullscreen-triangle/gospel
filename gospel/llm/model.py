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

from gospel.knowledge_base import KnowledgeBase


class GospelLLM:
    """
    Domain-specific LLM for genomic analysis, built on top of Ollama models.
    """

    def __init__(
        self,
        base_model: str = "llama3",
        kb: Optional[KnowledgeBase] = None,
        model_dir: Optional[str] = None,
    ):
        """
        Initialize a Gospel LLM.

        Args:
            base_model: Base Ollama model name
            kb: Knowledge base instance
            model_dir: Directory containing a trained model
        """
        self.base_model = base_model
        self.kb = kb
        self.model_dir = model_dir
        self.qa_chain = None
        self.genomic_networks = {}
        self.fine_tuned = False
        
        # Initialize the base model
        try:
            self.llm = Ollama(model=base_model)
        except Exception as e:
            print(f"Error initializing Ollama model: {e}")
            print("Please ensure Ollama is installed and running.")
            self.llm = None

        # Load model if model_dir is provided
        if model_dir:
            self.load(model_dir)

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
        }
        
        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)
        
        # Save the QA chain (note: some components might not be serializable)
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
        You are a genomics expert solver.
        
        Problem: {problem}
        
        Context information about relevant genes:
        {context_prompt}
        
        Please provide:
        1. A step-by-step solution approach
        2. The genetic factors involved
        3. The specific answer to the problem
        4. Any relevant network interactions
        """
        
        # Query the solver
        solution_text = self.query(solver_prompt)
        
        return {
            "problem": problem,
            "solution": solution_text,
            "method": "LLM-based genomic reasoning",
            "genes": list(gene_contexts.keys()),
            "networks": {gene: list(context["network"].nodes())[:20] for gene, context in gene_contexts.items()}
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