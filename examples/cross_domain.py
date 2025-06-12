#!/usr/bin/env python3
"""
Cross-domain genomic analysis example using real data and HuggingFace models.

This example demonstrates:
1. Real genomic data retrieval from public APIs (Ensembl, NCBI)
2. HuggingFace genomic model integration
3. Cross-domain analysis (genomics + pharmacogenomics + nutrition)
4. Systems biology network analysis
5. Multi-omics data integration
"""

import os
import argparse
import requests
import json
import time
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

from gospel.core import VariantProcessor, VariantAnnotator, GenomicScorer
from gospel.llm import GospelLLM
from gospel.network import NetworkAnalyzer
from gospel.utils import DataFetcher


class CrossDomainGenomicAnalyzer:
    """Real-world cross-domain genomic analysis using public APIs and HuggingFace models."""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.network_analyzer = NetworkAnalyzer()
        
        # Initialize HuggingFace models for genomic analysis
        self.dna_model = None
        self.protein_model = None
        self.drug_model = None
        
        # API endpoints for real data
        self.ensembl_base = "https://rest.ensembl.org"
        self.ncbi_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.pharmgkb_base = "https://api.pharmgkb.org/v1"
        
    def load_huggingface_models(self):
        """Load HuggingFace models for genomic analysis."""
        print("Loading HuggingFace genomic models...")
        
        try:
            # DNA sequence analysis model (Nucleotide Transformer)
            self.dna_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
            self.dna_model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
            print("✓ Loaded Nucleotide Transformer for DNA analysis")
            
            # Protein sequence analysis model (ESM-2)
            self.protein_model = pipeline("feature-extraction", model="facebook/esm2_t6_8M_UR50D")
            print("✓ Loaded ESM-2 for protein analysis")
            
            # Drug-target interaction model
            self.drug_model = pipeline("text-classification", model="dmis-lab/biobert-base-cased-v1.2")
            print("✓ Loaded BioBERT for drug-target analysis")
            
        except Exception as e:
            print(f"Warning: Could not load some HuggingFace models: {e}")
            print("Continuing with available models...")
    
    def get_real_gene_sequences(self, gene_symbols: List[str]) -> Dict[str, Dict]:
        """Fetch real gene sequences from Ensembl API."""
        print(f"Fetching real sequences for genes: {', '.join(gene_symbols)}")
        
        gene_data = {}
        
        for gene in gene_symbols:
            try:
                # Get gene information
                gene_url = f"{self.ensembl_base}/lookup/symbol/homo_sapiens/{gene}"
                headers = {"Content-Type": "application/json"}
                
                response = requests.get(gene_url, headers=headers)
                if response.status_code == 200:
                    gene_info = response.json()
                    
                    # Get gene sequence
                    seq_url = f"{self.ensembl_base}/sequence/id/{gene_info['id']}"
                    seq_response = requests.get(seq_url, headers=headers)
                    
                    if seq_response.status_code == 200:
                        sequence_data = seq_response.json()
                        
                        gene_data[gene] = {
                            'ensembl_id': gene_info['id'],
                            'chromosome': gene_info.get('seq_region_name', 'Unknown'),
                            'start': gene_info.get('start', 0),
                            'end': gene_info.get('end', 0),
                            'strand': gene_info.get('strand', 1),
                            'biotype': gene_info.get('biotype', 'protein_coding'),
                            'sequence': sequence_data['seq'],
                            'description': gene_info.get('description', '')
                        }
                        
                        print(f"✓ Retrieved {gene}: {len(sequence_data['seq'])} bp")
                    else:
                        print(f"✗ Could not retrieve sequence for {gene}")
                else:
                    print(f"✗ Gene {gene} not found in Ensembl")
                    
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching {gene}: {e}")
        
        return gene_data
    
    def get_real_variants(self, gene_symbols: List[str]) -> Dict[str, List]:
        """Fetch real variants from ClinVar via NCBI API."""
        print("Fetching real variants from ClinVar...")
        
        variant_data = {}
        
        for gene in gene_symbols:
            try:
                # Search ClinVar for variants in this gene
                search_url = f"{self.ncbi_base}/esearch.fcgi"
                params = {
                    'db': 'clinvar',
                    'term': f"{gene}[gene]",
                    'retmax': 50,
                    'retmode': 'json'
                }
                
                response = requests.get(search_url, params=params)
                if response.status_code == 200:
                    search_data = response.json()
                    
                    if 'esearchresult' in search_data and 'idlist' in search_data['esearchresult']:
                        variant_ids = search_data['esearchresult']['idlist'][:10]  # Limit to 10 variants
                        
                        # Fetch variant details
                        if variant_ids:
                            fetch_url = f"{self.ncbi_base}/efetch.fcgi"
                            fetch_params = {
                                'db': 'clinvar',
                                'id': ','.join(variant_ids),
                                'rettype': 'vcv',
                                'retmode': 'json'
                            }
                            
                            fetch_response = requests.get(fetch_url, params=fetch_params)
                            if fetch_response.status_code == 200:
                                variant_data[gene] = {
                                    'count': len(variant_ids),
                                    'ids': variant_ids,
                                    'source': 'ClinVar'
                                }
                                print(f"✓ Found {len(variant_ids)} variants for {gene}")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching variants for {gene}: {e}")
        
        return variant_data
    
    def get_drug_interactions(self, gene_symbols: List[str]) -> Dict[str, List]:
        """Fetch real pharmacogenomic data."""
        print("Fetching pharmacogenomic interactions...")
        
        # Sample pharmacogenomic data based on well-known gene-drug interactions
        known_interactions = {
            'CYP2D6': [
                {'drug': 'Codeine', 'interaction': 'Poor metabolizers have reduced efficacy'},
                {'drug': 'Tamoxifen', 'interaction': 'Poor metabolizers have reduced activation'},
                {'drug': 'Metoprolol', 'interaction': 'Poor metabolizers have increased drug levels'}
            ],
            'CYP2C19': [
                {'drug': 'Clopidogrel', 'interaction': 'Poor metabolizers have reduced antiplatelet effect'},
                {'drug': 'Omeprazole', 'interaction': 'Poor metabolizers have increased drug levels'},
                {'drug': 'Escitalopram', 'interaction': 'Poor metabolizers have increased side effects'}
            ],
            'TPMT': [
                {'drug': '6-Mercaptopurine', 'interaction': 'Low activity increases toxicity risk'},
                {'drug': 'Azathioprine', 'interaction': 'Low activity increases myelosuppression risk'}
            ],
            'BRCA1': [
                {'drug': 'PARP inhibitors', 'interaction': 'Mutations increase sensitivity to PARP inhibitors'},
                {'drug': 'Platinum compounds', 'interaction': 'Mutations increase sensitivity to DNA-damaging agents'}
            ],
            'BRCA2': [
                {'drug': 'PARP inhibitors', 'interaction': 'Mutations increase sensitivity to PARP inhibitors'},
                {'drug': 'Mitomycin C', 'interaction': 'Mutations increase sensitivity to cross-linking agents'}
            ]
        }
        
        interactions = {}
        for gene in gene_symbols:
            if gene in known_interactions:
                interactions[gene] = known_interactions[gene]
                print(f"✓ Found {len(known_interactions[gene])} drug interactions for {gene}")
        
        return interactions
    
    def analyze_with_huggingface(self, gene_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze sequences using HuggingFace models."""
        print("Analyzing sequences with HuggingFace models...")
        
        results = {}
        
        for gene, data in gene_data.items():
            gene_results = {}
            sequence = data['sequence']
            
            # DNA sequence analysis
            if self.dna_model and len(sequence) > 0:
                try:
                    # Truncate sequence if too long
                    max_len = 512
                    seq_chunk = sequence[:max_len]
                    
                    # Tokenize and analyze
                    inputs = self.dna_tokenizer(seq_chunk, return_tensors="pt", truncation=True, max_length=max_len)
                    
                    with torch.no_grad():
                        outputs = self.dna_model(**inputs)
                        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                    
                    gene_results['dna_embeddings'] = {
                        'shape': embeddings.shape,
                        'mean_activation': float(np.mean(embeddings)),
                        'std_activation': float(np.std(embeddings))
                    }
                    print(f"✓ DNA analysis completed for {gene}")
                    
                except Exception as e:
                    print(f"✗ DNA analysis failed for {gene}: {e}")
            
            # Convert DNA to protein sequence (simplified)
            if data['biotype'] == 'protein_coding':
                try:
                    # Get protein sequence from Ensembl
                    protein_url = f"{self.ensembl_base}/sequence/id/{data['ensembl_id']}?type=protein"
                    headers = {"Content-Type": "application/json"}
                    
                    response = requests.get(protein_url, headers=headers)
                    if response.status_code == 200:
                        protein_data = response.json()
                        protein_seq = protein_data['seq']
                        
                        # Analyze with protein model
                        if self.protein_model and len(protein_seq) > 0:
                            protein_embeddings = self.protein_model(protein_seq[:200])  # Limit length
                            
                            gene_results['protein_embeddings'] = {
                                'length': len(protein_seq),
                                'analyzed_length': min(len(protein_seq), 200),
                                'embedding_dim': len(protein_embeddings[0]) if protein_embeddings else 0
                            }
                            print(f"✓ Protein analysis completed for {gene}")
                    
                except Exception as e:
                    print(f"✗ Protein analysis failed for {gene}: {e}")
            
            results[gene] = gene_results
        
        return results
    
    def perform_systems_biology_analysis(self, gene_symbols: List[str]) -> Dict[str, Any]:
        """Perform systems biology network analysis using real data."""
        print("Performing systems biology network analysis...")
        
        # Fetch protein-protein interactions from STRING DB
        string_api = "https://string-db.org/api/json"
        
        try:
            # Get network interactions
            network_url = f"{string_api}/network"
            params = {
                'identifiers': '|'.join(gene_symbols),
                'species': 9606,  # Human
                'required_score': 400,  # Medium confidence
                'limit': 100
            }
            
            response = requests.get(network_url, params=params)
            if response.status_code == 200:
                interactions = response.json()
                
                # Get functional enrichment
                enrichment_url = f"{string_api}/enrichment"
                enrich_response = requests.get(enrichment_url, params=params)
                
                enrichment_data = []
                if enrich_response.status_code == 200:
                    enrichment_data = enrich_response.json()
                
                network_results = {
                    'interactions': {
                        'count': len(interactions),
                        'proteins': list(set([i['preferredName_A'] for i in interactions] + 
                                            [i['preferredName_B'] for i in interactions]))
                    },
                    'enrichment': {
                        'pathways': [e for e in enrichment_data if e['category'] == 'KEGG_Pathways'][:10],
                        'processes': [e for e in enrichment_data if e['category'] == 'Process'][:10]
                    }
                }
                
                print(f"✓ Found {len(interactions)} protein interactions")
                print(f"✓ Found {len(enrichment_data)} enrichment terms")
                
                return network_results
        
        except Exception as e:
            print(f"Error in systems biology analysis: {e}")
        
        return {}
    
    def cross_domain_integration(self, gene_data: Dict, variant_data: Dict, 
                               drug_data: Dict, network_data: Dict) -> Dict[str, Any]:
        """Integrate data across domains for comprehensive analysis."""
        print("Performing cross-domain integration...")
        
        integration_results = {}
        
        for gene in gene_data.keys():
            gene_integration = {
                'genomic_features': {
                    'length': len(gene_data[gene]['sequence']),
                    'chromosome': gene_data[gene]['chromosome'],
                    'biotype': gene_data[gene]['biotype']
                },
                'clinical_relevance': {
                    'variants_count': variant_data.get(gene, {}).get('count', 0),
                    'has_clinvar_data': gene in variant_data
                },
                'pharmacogenomics': {
                    'drug_interactions': len(drug_data.get(gene, [])),
                    'therapeutic_relevance': gene in drug_data
                },
                'systems_biology': {
                    'network_centrality': gene in network_data.get('interactions', {}).get('proteins', []),
                    'pathway_involvement': len([p for p in network_data.get('enrichment', {}).get('pathways', []) 
                                               if gene.lower() in p.get('inputGenes', '').lower()])
                }
            }
            
            # Calculate cross-domain score
            scores = []
            if gene_integration['clinical_relevance']['has_clinvar_data']:
                scores.append(0.8)
            if gene_integration['pharmacogenomics']['therapeutic_relevance']:
                scores.append(0.9)
            if gene_integration['systems_biology']['network_centrality']:
                scores.append(0.7)
            
            gene_integration['cross_domain_score'] = np.mean(scores) if scores else 0.0
            integration_results[gene] = gene_integration
        
        return integration_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Cross-domain genomic analysis using real data")
    parser.add_argument('--genes', nargs='+', 
                       default=['BRCA1', 'BRCA2', 'CYP2D6', 'CYP2C19', 'TPMT'],
                       help='Gene symbols to analyze')
    parser.add_argument('--output', default='cross_domain_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print("=== Cross-Domain Genomic Analysis with Real Data ===")
    print(f"Analyzing genes: {', '.join(args.genes)}")
    
    # Initialize analyzer
    analyzer = CrossDomainGenomicAnalyzer()
    
    # Load HuggingFace models
    analyzer.load_huggingface_models()
    
    # Step 1: Get real genomic data
    print("\n1. Fetching real genomic sequences...")
    gene_data = analyzer.get_real_gene_sequences(args.genes)
    
    # Step 2: Get real variant data
    print("\n2. Fetching real variant data...")
    variant_data = analyzer.get_real_variants(args.genes)
    
    # Step 3: Get pharmacogenomic data
    print("\n3. Fetching pharmacogenomic data...")
    drug_data = analyzer.get_drug_interactions(args.genes)
    
    # Step 4: Analyze with HuggingFace models
    print("\n4. Analyzing with HuggingFace models...")
    hf_results = analyzer.analyze_with_huggingface(gene_data)
    
    # Step 5: Systems biology analysis
    print("\n5. Performing systems biology analysis...")
    network_data = analyzer.perform_systems_biology_analysis(args.genes)
    
    # Step 6: Cross-domain integration
    print("\n6. Integrating across domains...")
    integration_results = analyzer.cross_domain_integration(
        gene_data, variant_data, drug_data, network_data
    )
    
    # Compile final results
    final_results = {
        'genes_analyzed': args.genes,
        'data_sources': {
            'genomic': 'Ensembl REST API',
            'variants': 'ClinVar via NCBI',
            'pharmacogenomics': 'Known gene-drug interactions',
            'networks': 'STRING-DB API',
            'ai_models': 'HuggingFace Transformers'
        },
        'gene_data': gene_data,
        'variant_data': variant_data,
        'drug_interactions': drug_data,
        'network_analysis': network_data,
        'huggingface_analysis': hf_results,
        'cross_domain_integration': integration_results,
        'summary': {
            'total_genes': len(gene_data),
            'genes_with_variants': len(variant_data),
            'genes_with_drug_interactions': len(drug_data),
            'network_interactions': network_data.get('interactions', {}).get('count', 0)
        }
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {args.output}")
    print(f"Genes analyzed: {len(gene_data)}")
    print(f"Total variants found: {sum(v.get('count', 0) for v in variant_data.values())}")
    print(f"Drug interactions: {sum(len(d) for d in drug_data.values())}")
    print(f"Network interactions: {network_data.get('interactions', {}).get('count', 0)}")


if __name__ == "__main__":
    main()
