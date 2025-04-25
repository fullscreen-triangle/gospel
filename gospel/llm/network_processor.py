"""
Network data processor for systems biology data integration.
"""

import json
import os
from typing import Dict, List, Optional, Union, Any
import requests
import concurrent.futures
import gzip
import lzma
import pickle
from functools import lru_cache
import logging
import time
import io
import multiprocessing


class NetworkDataProcessor:
    """
    Processor for systems biology network data to integrate with Gospel LLM training.
    """

    def __init__(
        self, 
        cache_dir: str = "network_data_cache", 
        compression: str = "gzip",
        max_workers: int = None,
        memory_limit_mb: int = 1024
    ):
        """
        Initialize a network data processor.

        Args:
            cache_dir: Directory to cache downloaded network data
            compression: Compression algorithm ("none", "gzip", "lzma")
            max_workers: Maximum number of parallel workers (defaults to CPU count)
            memory_limit_mb: Memory usage limit in MB
        """
        self.cache_dir = cache_dir
        self.compression = compression
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.memory_limit_mb = memory_limit_mb
        self.memory_usage = 0
        self.logger = logging.getLogger("NetworkDataProcessor")
        
        # Create cache directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "protein_interactions"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "reactome"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "proteomics"), exist_ok=True)
        
        # Initialize request session for connection pooling
        self.session = requests.Session()
        
        # Compression handlers
        self.compressors = {
            "none": {"ext": ".json", "open": open, "compress": False},
            "gzip": {"ext": ".json.gz", "open": gzip.open, "compress": True},
            "lzma": {"ext": ".json.xz", "open": lzma.open, "compress": True}
        }
        
        if compression not in self.compressors:
            raise ValueError(f"Unsupported compression: {compression}. Choose from {list(self.compressors.keys())}")
    
    def _check_memory_usage(self, data_size: int) -> bool:
        """
        Check if processing more data would exceed memory limits.
        
        Args:
            data_size: Estimated size of data to add in bytes
            
        Returns:
            Whether operation can proceed
        """
        # Convert MB limit to bytes for comparison
        limit_bytes = self.memory_limit_mb * 1024 * 1024
        
        # Add estimated size
        new_usage = self.memory_usage + data_size
        
        if new_usage > limit_bytes:
            self.logger.warning(f"Memory limit exceeded: {new_usage / (1024*1024):.2f}MB > {self.memory_limit_mb}MB")
            return False
        
        self.memory_usage = new_usage
        return True
    
    def _get_cache_file(self, data_type: str, identifier: str, source: str = None) -> str:
        """Get the appropriate cache file path with compression extension."""
        subfolder = {
            "ppi": "protein_interactions",
            "reactome": "reactome",
            "proteomics": "proteomics"
        }.get(data_type, "")
        
        filename = f"{identifier}_{source}_{data_type}" if source else f"{identifier}_{data_type}"
        ext = self.compressors[self.compression]["ext"]
        
        return os.path.join(self.cache_dir, subfolder, filename + ext)
    
    def _save_cache(self, cache_file: str, data: Any) -> None:
        """Save data to cache with appropriate compression."""
        open_func = self.compressors[self.compression]["open"]
        
        with open_func(cache_file, 'wt' if self.compressors[self.compression]["compress"] else 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_cache(self, cache_file: str) -> Any:
        """Load data from cache with appropriate decompression."""
        open_func = self.compressors[self.compression]["open"]
        
        with open_func(cache_file, 'rt' if self.compressors[self.compression]["compress"] else 'r') as f:
            return json.load(f)
    
    @lru_cache(maxsize=100)
    def fetch_protein_interactions(
        self, gene_id: str, source: str = "string", cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch protein-protein interactions for a gene from a public database.
        
        Args:
            gene_id: Gene identifier (HGNC symbol or Ensembl ID)
            source: Source database ("string", "biogrid", "intact")
            cache: Whether to cache results
            
        Returns:
            List of protein interaction data
        """
        cache_file = self._get_cache_file("ppi", gene_id, source)
        
        # Check cache first
        if cache and os.path.exists(cache_file):
            start_time = time.time()
            data = self._load_cache(cache_file)
            self.logger.debug(f"Cache load took {time.time() - start_time:.2f}s for {gene_id}")
            return data
        
        interactions = []
        start_time = time.time()
        
        if source == "string":
            # STRING API
            url = "https://string-db.org/api/json/network"
            params = {
                "identifiers": gene_id,
                "species": 9606,  # Human
                "required_score": 700,  # High confidence (0-1000)
                "network_type": "physical"
            }
            
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    interactions.append({
                        "gene_a": item.get("preferredName_A", ""),
                        "gene_b": item.get("preferredName_B", ""),
                        "score": item.get("score", 0),
                        "evidence": "physical interaction",
                        "source": "STRING"
                    })
        
        elif source == "biogrid":
            # BioGRID API
            url = "https://webservice.thebiogrid.org/interactions"
            params = {
                "geneList": gene_id,
                "searchNames": "true",
                "includeInteractors": "true",
                "format": "json",
                "taxId": 9606,  # Human
                "accessKey": os.environ.get("BIOGRID_API_KEY", "")
            }
            
            if not params["accessKey"]:
                self.logger.warning("BioGRID API key not found. Set BIOGRID_API_KEY environment variable.")
                return interactions
                
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for _, item in data.items():
                    interactions.append({
                        "gene_a": item.get("OFFICIAL_SYMBOL_A", ""),
                        "gene_b": item.get("OFFICIAL_SYMBOL_B", ""),
                        "interaction_type": item.get("EXPERIMENTAL_SYSTEM", ""),
                        "evidence": item.get("EXPERIMENTAL_SYSTEM_TYPE", ""),
                        "source": "BioGRID",
                        "publication": item.get("PUBMED_ID", "")
                    })
        
        elif source == "intact":
            # IntAct API
            url = "https://www.ebi.ac.uk/intact/ws/interaction/findInteractions"
            params = {
                "query": gene_id,
                "format": "json",
                "species": "human"
            }
            
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("interactions", []):
                    interactions.append({
                        "gene_a": item.get("interactorA", {}).get("symbol", ""),
                        "gene_b": item.get("interactorB", {}).get("symbol", ""),
                        "interaction_type": item.get("type", ""),
                        "score": item.get("confidence", 0),
                        "source": "IntAct"
                    })
        
        self.logger.debug(f"API fetch took {time.time() - start_time:.2f}s for {gene_id}")
        
        # Cache results
        if cache and interactions:
            self._save_cache(cache_file, interactions)
        
        # Update memory usage with estimated size of result
        self._check_memory_usage(len(str(interactions)) * 2)  # Rough estimate
        
        return interactions
    
    @lru_cache(maxsize=100)
    def fetch_reactome_pathways(
        self, gene_id: str, cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch Reactome pathways for a gene.
        
        Args:
            gene_id: Gene identifier (HGNC symbol or Ensembl ID)
            cache: Whether to cache results
            
        Returns:
            List of pathway data
        """
        cache_file = self._get_cache_file("reactome", gene_id)
        
        # Check cache first
        if cache and os.path.exists(cache_file):
            return self._load_cache(cache_file)
        
        pathways = []
        
        # Reactome API
        url = f"https://reactome.org/ContentService/data/pathways/low/entity/{gene_id}/allForms"
        response = self.session.get(url)
        
        if response.status_code == 200:
            data = response.json()
            for pathway in data:
                pathways.append({
                    "gene_id": gene_id,
                    "pathway_id": pathway.get("stId", ""),
                    "pathway_name": pathway.get("displayName", ""),
                    "species": pathway.get("species", {}).get("displayName", ""),
                    "diagram_url": f"https://reactome.org/ContentService/diagram/{pathway.get('stId', '')}.png",
                    "source": "Reactome"
                })
        
        # Cache results
        if cache and pathways:
            self._save_cache(cache_file, pathways)
        
        # Update memory usage with estimated size of result
        self._check_memory_usage(len(str(pathways)) * 2)
        
        return pathways
    
    @lru_cache(maxsize=100)
    def fetch_proteomics_data(
        self, gene_id: str, source: str = "uniprot", cache: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch proteomics data for a gene.
        
        Args:
            gene_id: Gene identifier (HGNC symbol or Ensembl ID)
            source: Source database ("uniprot", "proteomicsdb")
            cache: Whether to cache results
            
        Returns:
            Dictionary with proteomics data
        """
        cache_file = self._get_cache_file("proteomics", gene_id, source)
        
        # Check cache first
        if cache and os.path.exists(cache_file):
            return self._load_cache(cache_file)
        
        result = {
            "gene_id": gene_id,
            "source": source,
            "proteins": []
        }
        
        if source == "uniprot":
            # UniProt API
            url = "https://rest.uniprot.org/uniprotkb/search"
            params = {
                "query": f"gene:{gene_id} AND organism_id:9606",
                "format": "json"
            }
            
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("results", []):
                    protein_data = {
                        "uniprot_id": item.get("primaryAccession", ""),
                        "protein_name": item.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                        "function": [],
                        "subcellular_location": [],
                        "domains": []
                    }
                    
                    # Extract protein functions
                    for comment in item.get("comments", []):
                        if comment.get("commentType") == "FUNCTION":
                            for text in comment.get("texts", []):
                                protein_data["function"].append(text.get("value", ""))
                        elif comment.get("commentType") == "SUBCELLULAR LOCATION":
                            for location in comment.get("subcellularLocations", []):
                                protein_data["subcellular_location"].append(location.get("location", {}).get("value", ""))
                    
                    # Extract protein domains
                    for feature in item.get("features", []):
                        if feature.get("type") == "DOMAIN":
                            protein_data["domains"].append({
                                "type": feature.get("type", ""),
                                "description": feature.get("description", ""),
                                "start": feature.get("location", {}).get("start", {}).get("value", ""),
                                "end": feature.get("location", {}).get("end", {}).get("value", "")
                            })
                    
                    result["proteins"].append(protein_data)
        
        # Cache results
        if cache and result["proteins"]:
            self._save_cache(cache_file, result)
        
        # Update memory usage with estimated size of result
        self._check_memory_usage(len(str(result)) * 2)
        
        return result
    
    def _process_gene(self, gene_id: str) -> Dict[str, Any]:
        """Process a single gene to fetch all its network data."""
        result = {"gene_id": gene_id, "data": {}}
        
        # Fetch proteomics data
        proteomics = self.fetch_proteomics_data(gene_id)
        if proteomics["proteins"]:
            result["data"]["proteomics"] = proteomics
        
        # Fetch reactome pathways
        pathways = self.fetch_reactome_pathways(gene_id)
        if pathways:
            result["data"]["pathways"] = pathways
        
        # Fetch protein interactions
        interactions = self.fetch_protein_interactions(gene_id)
        if interactions:
            result["data"]["interactions"] = interactions
        
        return result
    
    def prepare_network_for_training(
        self, gene_ids: List[str], output_dir: str, batch_size: int = 10
    ) -> Dict[str, str]:
        """
        Prepare complete network data for a list of genes for training.
        Uses parallel processing for faster data fetching.
        
        Args:
            gene_ids: List of gene identifiers
            output_dir: Directory to save output files
            batch_size: Number of genes to process in each batch
            
        Returns:
            Dictionary with paths to output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = {
            "proteomics": os.path.join(output_dir, "proteomics_data.json"),
            "reactomes": os.path.join(output_dir, "reactome_pathways.json"),
            "interactomes": os.path.join(output_dir, "protein_interactions.json"),
            "network_summary": os.path.join(output_dir, "network_summary.json"),
        }
        
        proteomics_data = []
        reactome_data = []
        interaction_data = []
        network_summary = []
        
        # Process genes in batches to control memory usage
        for i in range(0, len(gene_ids), batch_size):
            batch = gene_ids[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(gene_ids)-1)//batch_size + 1} ({len(batch)} genes)")
            
            # Process genes in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_gene = {executor.submit(self._process_gene, gene_id): gene_id for gene_id in batch}
                
                for future in concurrent.futures.as_completed(future_to_gene):
                    gene_id = future_to_gene[future]
                    try:
                        result = future.result()
                        network_summary.append({
                            "gene_id": gene_id,
                            "proteomics_count": len(result["data"].get("proteomics", {}).get("proteins", [])),
                            "pathways_count": len(result["data"].get("pathways", [])),
                            "interactions_count": len(result["data"].get("interactions", []))
                        })
                        
                        # Extract data from result
                        if "proteomics" in result["data"]:
                            proteomics_data.append(result["data"]["proteomics"])
                        
                        if "pathways" in result["data"]:
                            reactome_data.append({
                                "gene_id": gene_id,
                                "pathways": result["data"]["pathways"]
                            })
                        
                        if "interactions" in result["data"]:
                            interaction_data.append({
                                "gene_id": gene_id,
                                "interactions": result["data"]["interactions"]
                            })
                    except Exception as e:
                        self.logger.error(f"Error processing {gene_id}: {e}")
            
            # Save intermediate results to manage memory
            if (i + batch_size) % (batch_size * 5) == 0 or i + batch_size >= len(gene_ids):
                self._save_batch_results(
                    proteomics_data, reactome_data, interaction_data, 
                    output_files, append=(i > 0)
                )
                # Clear memory
                proteomics_data = []
                reactome_data = []
                interaction_data = []
                # Force garbage collection
                import gc
                gc.collect()
                self.memory_usage = 0
        
        # Save network summary
        with open(output_files["network_summary"], 'w') as f:
            json.dump(network_summary, f, indent=2)
        
        return output_files
    
    def _save_batch_results(
        self, proteomics_data, reactome_data, interaction_data, 
        output_files, append=False
    ):
        """Save batch results to files, either creating new files or appending to existing ones."""
        mode = 'a' if append else 'w'
        
        # Helper function to save with streaming
        def save_streaming(data, file_path, mode='w'):
            with open(file_path, mode) as f:
                if mode == 'w':
                    f.write('[\n')
                elif mode == 'a' and os.path.getsize(file_path) <= 2:
                    f.write('[\n')
                
                for i, item in enumerate(data):
                    if i > 0 or (mode == 'a' and os.path.getsize(file_path) > 2):
                        f.write(',\n')
                    json.dump(item, f)
                
                if mode == 'w' or (mode == 'a' and data):
                    f.write('\n]')
        
        # Save proteomics data
        save_streaming(proteomics_data, output_files["proteomics"], mode)
        
        # Save reactome data
        save_streaming(reactome_data, output_files["reactomes"], mode)
        
        # Save interaction data
        save_streaming(interaction_data, output_files["interactomes"], mode)
    
    def create_connection_training_examples(
        self, gene_ids: List[str], output_file: str, batch_size: int = 10
    ) -> int:
        """
        Create training examples that highlight gene connections and networks.
        Uses parallel processing for faster example generation.
        
        Args:
            gene_ids: List of gene identifiers
            output_file: Path to save training examples (JSONL format)
            batch_size: Number of genes to process in each batch
            
        Returns:
            Number of examples created
        """
        example_count = 0
        
        # Create/truncate output file
        with open(output_file, 'w') as f:
            pass
        
        # Process genes in batches
        for i in range(0, len(gene_ids), batch_size):
            batch = gene_ids[i:i+batch_size]
            self.logger.info(f"Generating examples for batch {i//batch_size + 1}/{(len(gene_ids)-1)//batch_size + 1}")
            
            batch_examples = []
            
            # Process genes in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_gene = {executor.submit(self._create_examples_for_gene, gene_id): gene_id for gene_id in batch}
                
                for future in concurrent.futures.as_completed(future_to_gene):
                    gene_id = future_to_gene[future]
                    try:
                        examples = future.result()
                        batch_examples.extend(examples)
                    except Exception as e:
                        self.logger.error(f"Error generating examples for {gene_id}: {e}")
            
            # Write batch examples to file
            with open(output_file, 'a') as f:
                for example in batch_examples:
                    f.write(json.dumps(example) + "\n")
            
            example_count += len(batch_examples)
        
        return example_count
    
    def _create_examples_for_gene(self, gene_id: str) -> List[Dict[str, str]]:
        """Create training examples for a single gene."""
        examples = []
        
        # Get network data
        interactions = self.fetch_protein_interactions(gene_id)
        pathways = self.fetch_reactome_pathways(gene_id)
        proteomics = self.fetch_proteomics_data(gene_id)
        
        # Skip if no network data available
        if not (interactions or pathways or proteomics.get("proteins")):
            return examples
        
        # Create example for protein-protein interactions
        if interactions:
            interaction_partners = [i["gene_b"] for i in interactions if i["gene_a"] == gene_id]
            if interaction_partners:
                example = {
                    "prompt": f"What proteins interact with {gene_id} and what are their functions?",
                    "response": f"{gene_id} interacts with {len(interaction_partners)} proteins in the human interactome. "
                }
                
                # Add top interactions
                top_interactions = interaction_partners[:5]
                example["response"] += f"Key interaction partners include {', '.join(top_interactions)}. "
                
                # Add evidence if available
                evidence_sources = set(i["source"] for i in interactions)
                example["response"] += f"These interactions are supported by evidence from {', '.join(evidence_sources)}."
                
                examples.append(example)
        
        # Create example for pathway involvement
        if pathways:
            pathway_names = [p["pathway_name"] for p in pathways]
            if pathway_names:
                example = {
                    "prompt": f"What biological pathways involve {gene_id}?",
                    "response": f"{gene_id} participates in {len(pathways)} biological pathways according to Reactome. "
                }
                
                # Add top pathways
                top_pathways = pathway_names[:5]
                example["response"] += f"Key pathways include {', '.join(top_pathways)}. "
                
                # Add functional context
                example["response"] += f"These pathways highlight {gene_id}'s role in cellular processes and provide context for understanding its function in systems biology."
                
                examples.append(example)
        
        # Create example for protein function in system context
        if proteomics.get("proteins"):
            protein = proteomics["proteins"][0]  # Use first protein
            if protein.get("function"):
                example = {
                    "prompt": f"Explain the function of {gene_id} in the context of systems biology.",
                    "response": f"From a systems biology perspective, {gene_id} "
                }
                
                # Add function
                example["response"] += f"functions as {protein['function'][0]} "
                
                # Add cellular location if available
                if protein.get("subcellular_location"):
                    example["response"] += f"and is located in the {protein['subcellular_location'][0]}. "
                else:
                    example["response"] += ". "
                
                # Add network context
                if interactions and pathways:
                    example["response"] += f"It interacts with {len(interactions)} proteins and participates in {len(pathways)} pathways, forming a complex network that contributes to cellular function. "
                    
                    # Add systems-level insight
                    example["response"] += f"This highlights how {gene_id} doesn't function in isolation, but as part of interconnected biological systems with emergent properties beyond individual molecular interactions."
                
                examples.append(example)
        
        return examples
        
    def stream_data_processing(self, input_file: str, output_file: str, chunk_size: int = 1000) -> int:
        """
        Process large data files using streaming to minimize memory usage.
        
        Args:
            input_file: Path to input file (JSON or JSONL)
            output_file: Path to output file
            chunk_size: Size of chunks to process at once
            
        Returns:
            Number of records processed
        """
        record_count = 0
        
        # Determine input file type
        is_jsonl = input_file.endswith('.jsonl')
        
        with open(output_file, 'w') as out_f:
            out_f.write('[\n' if not is_jsonl else '')
            
            # Process input file in chunks
            with open(input_file, 'r') as in_f:
                if is_jsonl:
                    # Process JSONL line by line
                    for i, line in enumerate(in_f):
                        if i > 0:
                            out_f.write(',\n' if not is_jsonl else '\n')
                        
                        # Process the record
                        record = json.loads(line.strip())
                        processed_record = self._process_record(record)
                        
                        # Write processed record
                        json.dump(processed_record, out_f)
                        record_count += 1
                        
                        # Log progress
                        if record_count % chunk_size == 0:
                            self.logger.info(f"Processed {record_count} records")
                else:
                    # Process JSON in chunks
                    data = json.load(in_f)
                    if isinstance(data, list):
                        for i, record in enumerate(data):
                            if i > 0:
                                out_f.write(',\n')
                            
                            # Process the record
                            processed_record = self._process_record(record)
                            
                            # Write processed record
                            json.dump(processed_record, out_f)
                            record_count += 1
                            
                            # Log progress
                            if record_count % chunk_size == 0:
                                self.logger.info(f"Processed {record_count} records")
                    else:
                        # Single record
                        processed_record = self._process_record(data)
                        json.dump(processed_record, out_f)
                        record_count = 1
            
            out_f.write('\n]' if not is_jsonl else '')
        
        return record_count
    
    def _process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single record - placeholder for record transformation logic."""
        # This method can be extended with specific processing logic
        return record 