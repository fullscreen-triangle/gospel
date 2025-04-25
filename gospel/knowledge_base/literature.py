"""
Literature extraction and processing for the knowledge base.
"""

import json
import os
import re
from typing import Dict, List, Optional, Set, Union

import requests
from tqdm import tqdm

from gospel.utils.gene_utils import normalize_gene_id


class LiteratureRetriever:
    """
    Retrieve literature related to genes and athletic performance.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize a literature retriever.

        Args:
            cache_dir: Directory for caching results
        """
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def search_pubmed(self, gene_id: str, max_results: int = 10) -> List[Dict]:
        """
        Search PubMed for articles about a gene.

        Args:
            gene_id: Gene identifier
            max_results: Maximum number of results to return

        Returns:
            List of article data
        """
        # Normalize gene ID
        canonical_id = normalize_gene_id(gene_id)
        
        # Check cache
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{canonical_id}_pubmed.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f:
                        return json.load(f)
                except:
                    pass
        
        # Base URL for NCBI E-utilities
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # Search for articles
        search_term = f"{canonical_id}[Gene] AND (athletic[Title/Abstract] OR performance[Title/Abstract] OR sport[Title/Abstract] OR exercise[Title/Abstract] OR fitness[Title/Abstract])"
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={search_term}&retmode=json&retmax={max_results}"
        
        try:
            # Get article IDs
            response = requests.get(search_url)
            response.raise_for_status()
            search_results = response.json()
            
            article_ids = search_results.get("esearchresult", {}).get("idlist", [])
            
            if not article_ids:
                print(f"No articles found for {canonical_id}")
                return []
            
            # Fetch article details
            ids_str = ",".join(article_ids)
            fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids_str}&retmode=xml"
            
            response = requests.get(fetch_url)
            response.raise_for_status()
            xml_content = response.text
            
            # Parse XML (basic parsing, could be improved with a proper XML parser)
            articles = []
            article_blocks = re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml_content, re.DOTALL)
            
            for block in article_blocks:
                # Extract title
                title_match = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", block)
                title = title_match.group(1) if title_match else "Unknown title"
                
                # Extract abstract
                abstract_match = re.search(r"<AbstractText>(.*?)</AbstractText>", block)
                abstract = abstract_match.group(1) if abstract_match else ""
                
                # Extract PMID
                pmid_match = re.search(r"<PMID.*?>(.*?)</PMID>", block)
                pmid = pmid_match.group(1) if pmid_match else "Unknown"
                
                # Extract year
                year_match = re.search(r"<PubDate>.*?<Year>(.*?)</Year>", block)
                year = year_match.group(1) if year_match else "Unknown"
                
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "gene_id": canonical_id
                })
            
            # Cache results
            if self.cache_dir:
                try:
                    with open(cache_file, "w") as f:
                        json.dump(articles, f, indent=2)
                except Exception as e:
                    print(f"Error caching results: {e}")
            
            return articles
        
        except Exception as e:
            print(f"Error searching PubMed for {canonical_id}: {e}")
            return []

    def get_gene_info_from_ncbi(self, gene_id: str) -> Dict:
        """
        Get information about a gene from NCBI.

        Args:
            gene_id: Gene identifier

        Returns:
            Dictionary of gene information
        """
        # Normalize gene ID
        canonical_id = normalize_gene_id(gene_id)
        
        # Check cache
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{canonical_id}_ncbi.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f:
                        return json.load(f)
                except:
                    pass
        
        # Base URL for NCBI E-utilities
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        try:
            # First get Gene ID from symbol
            search_url = f"{base_url}esearch.fcgi?db=gene&term={canonical_id}[Gene%20Symbol]%20AND%20human[Organism]&retmode=json"
            
            response = requests.get(search_url)
            response.raise_for_status()
            search_results = response.json()
            
            gene_ids = search_results.get("esearchresult", {}).get("idlist", [])
            
            if not gene_ids:
                print(f"No NCBI Gene ID found for {canonical_id}")
                return {"gene_id": canonical_id, "error": "Not found in NCBI Gene database"}
            
            # Get gene details
            fetch_url = f"{base_url}efetch.fcgi?db=gene&id={gene_ids[0]}&retmode=xml"
            
            response = requests.get(fetch_url)
            response.raise_for_status()
            xml_content = response.text
            
            # Parse XML (basic parsing, could be improved with a proper XML parser)
            gene_info = {
                "gene_id": canonical_id,
                "ncbi_id": gene_ids[0],
                "aliases": [],
                "description": "",
                "location": "",
                "sequence_length": "",
                "function": ""
            }
            
            # Extract gene description
            description_match = re.search(r"<Gene-ref_desc>(.*?)</Gene-ref_desc>", xml_content)
            if description_match:
                gene_info["description"] = description_match.group(1)
            
            # Extract gene location
            location_match = re.search(r"<Gene-ref_maploc>(.*?)</Gene-ref_maploc>", xml_content)
            if location_match:
                gene_info["location"] = location_match.group(1)
            
            # Extract gene aliases
            alias_matches = re.findall(r"<Gene-ref_syn_E>(.*?)</Gene-ref_syn_E>", xml_content)
            gene_info["aliases"] = alias_matches
            
            # Cache results
            if self.cache_dir:
                try:
                    with open(cache_file, "w") as f:
                        json.dump(gene_info, f, indent=2)
                except Exception as e:
                    print(f"Error caching results: {e}")
            
            return gene_info
        
        except Exception as e:
            print(f"Error retrieving NCBI information for {canonical_id}: {e}")
            return {"gene_id": canonical_id, "error": str(e)}

    def get_pathway_info(self, gene_id: str) -> Dict:
        """
        Get pathway information for a gene from Reactome.

        Args:
            gene_id: Gene identifier

        Returns:
            Dictionary of pathway information
        """
        # Normalize gene ID
        canonical_id = normalize_gene_id(gene_id)
        
        # Check cache
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{canonical_id}_pathway.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f:
                        return json.load(f)
                except:
                    pass
        
        # Reactome API URL
        reactome_url = f"https://reactome.org/ContentService/data/query/{canonical_id}/pathways?species=Homo%20sapiens"
        
        try:
            response = requests.get(reactome_url)
            response.raise_for_status()
            pathways = response.json()
            
            pathway_info = {
                "gene_id": canonical_id,
                "pathways": []
            }
            
            for pathway in pathways:
                pathway_info["pathways"].append({
                    "id": pathway.get("stId", ""),
                    "name": pathway.get("displayName", ""),
                    "url": f"https://reactome.org/content/detail/{pathway.get('stId', '')}"
                })
            
            # Cache results
            if self.cache_dir:
                try:
                    with open(cache_file, "w") as f:
                        json.dump(pathway_info, f, indent=2)
                except Exception as e:
                    print(f"Error caching results: {e}")
            
            return pathway_info
        
        except Exception as e:
            print(f"Error retrieving pathway information for {canonical_id}: {e}")
            return {"gene_id": canonical_id, "pathways": [], "error": str(e)}

    def enrich_gene_info(self, gene_id: str) -> Dict:
        """
        Enrich gene information from multiple sources.

        Args:
            gene_id: Gene identifier

        Returns:
            Enriched gene information
        """
        # Normalize gene ID
        canonical_id = normalize_gene_id(gene_id)
        
        # Get basic gene info from NCBI
        gene_info = self.get_gene_info_from_ncbi(canonical_id)
        
        # Get pathway information
        pathway_info = self.get_pathway_info(canonical_id)
        gene_info["pathways"] = pathway_info.get("pathways", [])
        
        # Get relevant publications
        articles = self.search_pubmed(canonical_id, max_results=5)
        gene_info["literature"] = articles
        
        # Extract key information from abstracts using a simple approach
        athletic_relevance = ""
        for article in articles:
            abstract = article.get("abstract", "").lower()
            
            if "sprint" in abstract or "power" in abstract:
                athletic_relevance += f"Relevant to sprint/power performance. "
            
            if "endurance" in abstract:
                athletic_relevance += f"May affect endurance performance. "
            
            if "muscle" in abstract:
                athletic_relevance += f"Affects muscle function. "
        
        gene_info["athletic_relevance"] = athletic_relevance.strip()
        
        return gene_info

    def batch_enrich_genes(self, gene_ids: List[str]) -> Dict[str, Dict]:
        """
        Enrich information for multiple genes.

        Args:
            gene_ids: List of gene identifiers

        Returns:
            Dictionary mapping gene IDs to enriched information
        """
        results = {}
        
        for gene_id in tqdm(gene_ids, desc="Enriching gene information"):
            results[gene_id] = self.enrich_gene_info(gene_id)
        
        return results
