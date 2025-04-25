"""
Knowledge Base for genomic data.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import networkx as nx
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma

from gospel.knowledge_base.retrieval import PdfTextExtractor
from gospel.utils.gene_utils import normalize_gene_id


class KnowledgeBase:
    """
    Knowledge Base for genomic data, literature, and gene networks.
    """

    def __init__(self, vector_store_path: Optional[str] = None):
        """
        Initialize a Knowledge Base.

        Args:
            vector_store_path: Path to a vector store (optional)
        """
        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.gene_info = {}
        self.gene_networks = {}
        self.gene_variants = {}
        self.id_mapping = {}  # Maps various forms of gene IDs to canonical ones
        
        if vector_store_path and os.path.exists(vector_store_path):
            self._load_vector_store()
    
    def _load_vector_store(self):
        """Load the vector store from disk."""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
            self.vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=embeddings,
            )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.vector_store = None

    def build_from_pdfs(
        self,
        pdf_dir: str,
        output_dir: str,
        model_name: Optional[str] = "llama3",
    ) -> None:
        """
        Build a knowledge base from PDF files.

        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Output directory for the knowledge base
            model_name: Name of the Ollama model to use for information extraction
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        vector_store_path = os.path.join(output_dir, "vector_store")
        os.makedirs(vector_store_path, exist_ok=True)
        
        # Extract text from PDFs
        extractor = PdfTextExtractor()
        documents = extractor.extract_from_directory(pdf_dir)
        
        # Extract gene information and networks
        self._extract_gene_information(documents, model_name)
        
        # Create vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        
        # Convert to langchain documents
        langchain_docs = []
        for doc in documents:
            for chunk in doc.chunks:
                langchain_docs.append(
                    Document(
                        page_content=chunk.text,
                        metadata={
                            "source": doc.source,
                            "page": chunk.page,
                            "genes": chunk.genes,
                            "chunk_id": chunk.id,
                        }
                    )
                )
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=langchain_docs,
            embedding=embeddings,
            persist_directory=vector_store_path,
        )
        self.vector_store_path = vector_store_path
        
        # Save gene information
        with open(os.path.join(output_dir, "gene_info.json"), "w") as f:
            json.dump(self.gene_info, f, indent=2)
        
        # Save ID mapping
        with open(os.path.join(output_dir, "id_mapping.json"), "w") as f:
            json.dump(self.id_mapping, f, indent=2)
        
        # Save gene variants
        with open(os.path.join(output_dir, "gene_variants.json"), "w") as f:
            json.dump(self.gene_variants, f, indent=2)
        
        # Save gene networks (as pickled NetworkX graphs)
        network_dir = os.path.join(output_dir, "networks")
        os.makedirs(network_dir, exist_ok=True)
        
        for gene, network in self.gene_networks.items():
            try:
                with open(os.path.join(network_dir, f"{gene}_network.pkl"), "wb") as f:
                    pickle.dump(network, f)
            except Exception as e:
                print(f"Error saving network for gene {gene}: {e}")
                
                # Fallback: save as JSON
                try:
                    with open(os.path.join(network_dir, f"{gene}_network.json"), "w") as f:
                        # Convert network to serializable format
                        network_data = {
                            "nodes": list(network.nodes()),
                            "edges": list(network.edges()),
                            "node_attributes": {str(n): d for n, d in network.nodes(data=True)},
                            "edge_attributes": {str(e): d for e, d in network.edges(data=True)},
                        }
                        json.dump(network_data, f, indent=2)
                except Exception as e2:
                    print(f"Error saving network as JSON for gene {gene}: {e2}")
        
        # Save knowledge base metadata
        metadata = {
            "vector_store_path": vector_store_path,
            "pdf_source_dir": os.path.abspath(pdf_dir),
            "gene_count": len(self.gene_info),
            "network_count": len(self.gene_networks),
            "variant_count": sum(len(variants) for variants in self.gene_variants.values()),
            "document_count": len(documents),
            "chunk_count": len(langchain_docs),
        }
        
        with open(os.path.join(output_dir, "kb_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Knowledge base built and saved to {output_dir}")
        print(f"Processed {len(documents)} documents")
        print(f"Extracted information for {len(self.gene_info)} genes")
        print(f"Created {len(self.gene_networks)} gene networks")

    def _extract_gene_information(self, documents, model_name: str) -> None:
        """
        Extract gene information from documents using an LLM.

        Args:
            documents: List of document objects with text chunks
            model_name: Name of the Ollama model to use
        """
        from langchain.llms import Ollama
        import re
        
        try:
            # Initialize Ollama
            llm = Ollama(model=model_name)
            
            # Process each document
            for doc in documents:
                print(f"Extracting gene information from {doc.source}")
                
                # Combine chunks with gene mentions for better context
                gene_chunks = [chunk for chunk in doc.chunks if chunk.genes]
                processed_genes = set()
                
                for chunk in gene_chunks:
                    for gene_mention in chunk.genes:
                        # Normalize gene ID to handle different naming conventions
                        canonical_gene_id = normalize_gene_id(gene_mention)
                        
                        # Skip if already processed
                        if canonical_gene_id in processed_genes:
                            continue
                        
                        # Add to ID mapping
                        self.id_mapping[gene_mention] = canonical_gene_id
                        
                        # Extract information about this gene using the LLM
                        context = chunk.text
                        
                        # Find more context from nearby chunks
                        for c in doc.chunks:
                            if c.id != chunk.id and gene_mention in c.genes:
                                context += "\n\n" + c.text
                                if len(context) > 4000:  # Limit context size
                                    break
                        
                        # Query the LLM for gene information
                        gene_prompt = f"""
                        Extract detailed information about the gene {gene_mention} (also known as {canonical_gene_id}) from the following text. 
                        Focus on its function, variants, and role in athletic performance, especially sprint performance.
                        Return the information in JSON format with the following structure:
                        {{
                            "gene_id": "{canonical_gene_id}",
                            "gene_name": "Full gene name",
                            "function": "Detailed description of gene function",
                            "variants": ["rs123456", "rs789012"],
                            "athletic_relevance": "Description of relevance to athletic/sprint performance",
                            "pathways": ["Pathway1", "Pathway2"],
                            "interactions": ["Gene1", "Gene2"]
                        }}
                        
                        Here is the text:
                        {context[:3000]}  # Limit context length
                        """
                        
                        try:
                            response = llm(gene_prompt)
                            
                            # Extract JSON from response
                            json_match = re.search(r'({[\s\S]*})', response)
                            if json_match:
                                gene_data = json.loads(json_match.group(1))
                                
                                # Store gene information
                                self.gene_info[canonical_gene_id] = gene_data
                                
                                # Store variants
                                self.gene_variants[canonical_gene_id] = gene_data.get("variants", [])
                                
                                # Create network
                                self._create_gene_network(canonical_gene_id, gene_data)
                                
                                processed_genes.add(canonical_gene_id)
                                print(f"  Extracted information for {canonical_gene_id}")
                            else:
                                print(f"  Could not extract structured data for {canonical_gene_id}")
                        except Exception as e:
                            print(f"  Error extracting information for {canonical_gene_id}: {e}")
        
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            print("Proceeding with limited gene information extraction...")
            
            # Fallback: Extract basic gene information from text
            for doc in documents:
                for chunk in doc.chunks:
                    for gene in chunk.genes:
                        canonical_gene_id = normalize_gene_id(gene)
                        self.id_mapping[gene] = canonical_gene_id
                        
                        if canonical_gene_id not in self.gene_info:
                            self.gene_info[canonical_gene_id] = {
                                "gene_id": canonical_gene_id,
                                "mentions": [],
                                "contexts": []
                            }
                        
                        # Add context
                        if len(self.gene_info[canonical_gene_id]["contexts"]) < 5:  # Limit contexts
                            self.gene_info[canonical_gene_id]["contexts"].append(chunk.text[:500])
                        
                        # Add mention
                        if gene not in self.gene_info[canonical_gene_id]["mentions"]:
                            self.gene_info[canonical_gene_id]["mentions"].append(gene)

    def _create_gene_network(self, gene_id: str, gene_data: Dict) -> None:
        """
        Create a network for a gene based on extracted data.

        Args:
            gene_id: Gene identifier
            gene_data: Dictionary of gene information
        """
        import networkx as nx
        
        # Create a network if it doesn't exist
        if gene_id not in self.gene_networks:
            self.gene_networks[gene_id] = nx.Graph()
        
        # Add node for the main gene
        self.gene_networks[gene_id].add_node(
            gene_id,
            type="main_gene",
            function=gene_data.get("function", ""),
            athletic_relevance=gene_data.get("athletic_relevance", "")
        )
        
        # Add nodes for variants
        for variant in gene_data.get("variants", []):
            self.gene_networks[gene_id].add_node(
                variant,
                type="variant"
            )
            self.gene_networks[gene_id].add_edge(
                gene_id,
                variant,
                relationship="has_variant"
            )
        
        # Add nodes for pathways
        for pathway in gene_data.get("pathways", []):
            self.gene_networks[gene_id].add_node(
                pathway,
                type="pathway"
            )
            self.gene_networks[gene_id].add_edge(
                gene_id,
                pathway,
                relationship="participates_in"
            )
        
        # Add nodes for interacting genes
        for interaction in gene_data.get("interactions", []):
            interaction_id = normalize_gene_id(interaction)
            self.gene_networks[gene_id].add_node(
                interaction_id,
                type="gene"
            )
            self.gene_networks[gene_id].add_edge(
                gene_id,
                interaction_id,
                relationship="interacts_with"
            )

    def query(self, query: str, k: int = 5) -> str:
        """
        Query the knowledge base.

        Args:
            query: Query string
            k: Number of documents to retrieve

        Returns:
            Answer string
        """
        from langchain.llms import Ollama
        
        # Check if vector store is available
        if not self.vector_store:
            return "Error: Vector store not available"
        
        # Search for relevant documents
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            
            # Extract context from documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Extract mentioned genes
            mentioned_genes = set()
            for doc in docs:
                if "genes" in doc.metadata:
                    mentioned_genes.update(doc.metadata["genes"])
            
            # Add gene information
            gene_context = ""
            for gene in mentioned_genes:
                canonical_id = self.id_mapping.get(gene, gene)
                if canonical_id in self.gene_info:
                    info = self.gene_info[canonical_id]
                    gene_context += f"\nGene: {canonical_id}\n"
                    gene_context += f"Function: {info.get('function', 'Unknown')}\n"
                    gene_context += f"Athletic relevance: {info.get('athletic_relevance', 'Unknown')}\n"
                    
                    # Add variant information
                    if canonical_id in self.gene_variants and self.gene_variants[canonical_id]:
                        gene_context += f"Variants: {', '.join(self.gene_variants[canonical_id])}\n"
            
            # Create response using LLM
            try:
                llm = Ollama(model="llama3")
                
                prompt = f"""
                Answer the following query using the provided context and gene information.
                If the information is not in the context, say so - do not make up information.
                
                Query: {query}
                
                Context from genomic reports:
                {context}
                
                Information about mentioned genes:
                {gene_context}
                
                Answer:
                """
                
                response = llm(prompt)
                return response
            except Exception as e:
                # Fallback if LLM isn't available
                return f"Relevant information:\n\n{context}\n\nGene information:\n{gene_context}"
        
        except Exception as e:
            return f"Error querying knowledge base: {e}"

    def get_retriever(self):
        """
        Get a retriever for the knowledge base.

        Returns:
            Langchain retriever
        """
        if not self.vector_store:
            raise ValueError("Vector store not available")
        
        return self.vector_store.as_retriever()

    def get_gene_info(self, gene_id: str) -> Dict:
        """
        Get information about a gene.

        Args:
            gene_id: Gene identifier

        Returns:
            Dictionary of gene information
        """
        # Normalize gene ID
        canonical_id = self.id_mapping.get(gene_id, gene_id)
        
        # Return gene information
        return self.gene_info.get(canonical_id, {"gene_id": canonical_id, "info": "No information available"})

    def get_gene_network(self, gene_id: str):
        """
        Get the network for a gene.

        Args:
            gene_id: Gene identifier

        Returns:
            NetworkX graph for the gene
        """
        # Normalize gene ID
        canonical_id = self.id_mapping.get(gene_id, gene_id)
        
        # Return gene network
        return self.gene_networks.get(canonical_id)

    def get_gene_networks(self) -> Dict:
        """
        Get all gene networks.

        Returns:
            Dictionary of gene networks
        """
        return self.gene_networks

    def get_all_genes(self) -> Set[str]:
        """
        Get all genes in the knowledge base.

        Returns:
            Set of gene identifiers
        """
        return set(self.gene_info.keys())

    def save(self, output_dir: str) -> None:
        """
        Save the knowledge base to a directory.

        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save gene information
        with open(os.path.join(output_dir, "gene_info.json"), "w") as f:
            json.dump(self.gene_info, f, indent=2)
        
        # Save ID mapping
        with open(os.path.join(output_dir, "id_mapping.json"), "w") as f:
            json.dump(self.id_mapping, f, indent=2)
        
        # Save gene variants
        with open(os.path.join(output_dir, "gene_variants.json"), "w") as f:
            json.dump(self.gene_variants, f, indent=2)
        
        # Save gene networks
        network_dir = os.path.join(output_dir, "networks")
        os.makedirs(network_dir, exist_ok=True)
        
        for gene, network in self.gene_networks.items():
            try:
                with open(os.path.join(network_dir, f"{gene}_network.pkl"), "wb") as f:
                    pickle.dump(network, f)
            except Exception as e:
                print(f"Error saving network for gene {gene}: {e}")
                
                # Fallback: save as JSON
                try:
                    with open(os.path.join(network_dir, f"{gene}_network.json"), "w") as f:
                        # Convert network to serializable format
                        network_data = {
                            "nodes": list(network.nodes()),
                            "edges": list(network.edges()),
                            "node_attributes": {str(n): d for n, d in network.nodes(data=True)},
                            "edge_attributes": {str(e): d for e, d in network.edges(data=True)},
                        }
                        json.dump(network_data, f, indent=2)
                except Exception as e2:
                    print(f"Error saving network as JSON for gene {gene}: {e2}")
        
        # If vector store exists and path is different, save it
        if self.vector_store and (not self.vector_store_path or 
                                  os.path.abspath(self.vector_store_path) != os.path.abspath(os.path.join(output_dir, "vector_store"))):
            vector_store_path = os.path.join(output_dir, "vector_store")
            os.makedirs(vector_store_path, exist_ok=True)
            
            # Persist vector store
            self.vector_store.persist_directory = vector_store_path
            self.vector_store.persist()
            self.vector_store_path = vector_store_path
        
        # Save knowledge base metadata
        metadata = {
            "vector_store_path": self.vector_store_path,
            "gene_count": len(self.gene_info),
            "network_count": len(self.gene_networks),
            "variant_count": sum(len(variants) for variants in self.gene_variants.values()),
        }
        
        with open(os.path.join(output_dir, "kb_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, input_dir: str) -> "KnowledgeBase":
        """
        Load a knowledge base from a directory.

        Args:
            input_dir: Input directory

        Returns:
            KnowledgeBase instance
        """
        # Check if directory exists
        if not os.path.exists(input_dir):
            raise ValueError(f"Directory {input_dir} does not exist")
        
        # Create instance
        vector_store_path = os.path.join(input_dir, "vector_store")
        instance = cls(vector_store_path=vector_store_path if os.path.exists(vector_store_path) else None)
        
        # Load gene information
        gene_info_path = os.path.join(input_dir, "gene_info.json")
        if os.path.exists(gene_info_path):
            with open(gene_info_path, "r") as f:
                instance.gene_info = json.load(f)
        
        # Load ID mapping
        id_mapping_path = os.path.join(input_dir, "id_mapping.json")
        if os.path.exists(id_mapping_path):
            with open(id_mapping_path, "r") as f:
                instance.id_mapping = json.load(f)
        
        # Load gene variants
        gene_variants_path = os.path.join(input_dir, "gene_variants.json")
        if os.path.exists(gene_variants_path):
            with open(gene_variants_path, "r") as f:
                instance.gene_variants = json.load(f)
        
        # Load gene networks
        network_dir = os.path.join(input_dir, "networks")
        if os.path.exists(network_dir):
            import networkx as nx
            
            # First try to load from pickle files
            for file in os.listdir(network_dir):
                if file.endswith("_network.pkl"):
                    gene_id = file[:-12]  # Remove "_network.pkl"
                    try:
                        with open(os.path.join(network_dir, file), "rb") as f:
                            instance.gene_networks[gene_id] = pickle.load(f)
                    except Exception as e:
                        print(f"Error loading network for gene {gene_id}: {e}")
            
            # Then try to load from JSON files (for networks that couldn't be pickled)
            for file in os.listdir(network_dir):
                if file.endswith("_network.json"):
                    gene_id = file[:-13]  # Remove "_network.json"
                    if gene_id not in instance.gene_networks:  # Only load if not already loaded from pickle
                        try:
                            with open(os.path.join(network_dir, file), "r") as f:
                                network_data = json.load(f)
                                
                                # Create network from data
                                G = nx.Graph()
                                G.add_nodes_from(network_data["nodes"])
                                G.add_edges_from(network_data["edges"])
                                
                                # Add attributes
                                for node, attrs in network_data["node_attributes"].items():
                                    for key, value in attrs.items():
                                        G.nodes[node][key] = value
                                
                                for edge, attrs in network_data["edge_attributes"].items():
                                    # Convert edge string back to tuple
                                    edge_tuple = eval(edge)
                                    for key, value in attrs.items():
                                        G.edges[edge_tuple][key] = value
                                
                                instance.gene_networks[gene_id] = G
                        except Exception as e:
                            print(f"Error loading network from JSON for gene {gene_id}: {e}")
        
        return instance
