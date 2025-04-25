"""
Utility functions for gene manipulation and identification.
"""

import re
from typing import List, Set

# Common gene name patterns
GENE_PATTERNS = [
    r'\b[A-Z][A-Z0-9]+\b',  # Uppercase gene symbols like ACTN3
    r'\b[A-Z][A-Za-z0-9]+\d+\b',  # Mixed case with numbers like ACE2
    r'\brs\d+\b',  # SNP IDs like rs1815739
    r'\b[A-Z]{2,4}\d*-\d+\b',  # Gene-variant notation like ACTN3-R577X
]

# Known sprint-related genes
SPRINT_GENES = {
    'ACTN3', 'ACE', 'AMPD1', 'CKM', 'CKMM', 'HFE', 'ADRB2', 'PPARA',
    'PPARGC1A', 'VEGFA', 'NOS3', 'HIF1A', 'EPOR', 'ADRB3', 'UCP2',
    'UCP3', 'MSTN', 'IGF1', 'IGF2', 'GH1', 'COL1A1', 'COL5A1', 'MMP3',
    'TIMP1', 'DIO1', 'BDNF', 'COMT', 'GABPB1', 'IL6', 'AGTR2', 'TRHR',
    'CRP', 'CNTF'
}

# Common gene name aliases
GENE_ALIASES = {
    'ACE1': 'ACE',
    'ACEI': 'ACE',
    'ALPI': 'ACTN3',
    'ACTN-3': 'ACTN3',
    'ACTN 3': 'ACTN3',
    'AMPD': 'AMPD1',
    'MCKM': 'CKM',
    'CKMM': 'CKM',
    'A2BAR': 'ADRB2',
    'ADRB2R': 'ADRB2',
    'NR1C1': 'PPARA',
    'PPAR-ALPHA': 'PPARA',
    'PGC1A': 'PPARGC1A',
    'PGC-1A': 'PPARGC1A',
    'VEGF': 'VEGFA',
    'ENOS': 'NOS3',
    'HIF-1A': 'HIF1A',
    'HIF1': 'HIF1A',
    'EPO-R': 'EPOR',
    'B3AR': 'ADRB3',
    'ADRB3R': 'ADRB3',
    'GDF8': 'MSTN',
    'IGF-I': 'IGF1',
    'IGF-II': 'IGF2',
    'GH': 'GH1',
    'MMP-3': 'MMP3',
    'TIMP-1': 'TIMP1',
}


def normalize_gene_id(gene_id: str) -> str:
    """
    Normalize a gene ID to a canonical form.

    Args:
        gene_id: Gene identifier

    Returns:
        Normalized gene ID
    """
    # Convert to uppercase
    gene_id = gene_id.upper()
    
    # Remove whitespace
    gene_id = gene_id.strip()
    
    # Check aliases
    if gene_id in GENE_ALIASES:
        return GENE_ALIASES[gene_id]
    
    # Remove common prefixes/suffixes
    gene_id = re.sub(r'^GENE[:\-\s]*', '', gene_id)
    gene_id = re.sub(r'^PROTEIN[:\-\s]*', '', gene_id)
    
    # Handle common formatting variations
    gene_id = re.sub(r'[\s\-_]+', '', gene_id)  # Remove spaces, hyphens, underscores
    
    # Special case for SNP IDs
    if re.match(r'^RS\d+$', gene_id):
        return gene_id.lower()  # Convert "RS123" to "rs123"
    
    return gene_id


def find_gene_mentions(text: str) -> List[str]:
    """
    Find gene mentions in text.

    Args:
        text: Text to search

    Returns:
        List of gene mentions
    """
    genes = set()
    
    # Search for genes using patterns
    for pattern in GENE_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            gene = match.group(0)
            
            # Skip if too short (likely not a gene)
            if len(gene) < 3:
                continue
            
            # Skip if not a plausible gene symbol
            if gene.isdigit() or gene.islower():
                continue
            
            # Normalize the gene ID
            gene = normalize_gene_id(gene)
            
            genes.add(gene)
    
    # Search for known sprint genes explicitly
    for gene in SPRINT_GENES:
        # Look for the gene with word boundaries
        if re.search(r'\b' + re.escape(gene) + r'\b', text):
            genes.add(gene)
    
    # Check for common aliases
    for alias, canonical in GENE_ALIASES.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', text):
            genes.add(canonical)
    
    return list(genes)


def get_gene_variants(gene_id: str) -> List[str]:
    """
    Get known variants for a gene.

    Args:
        gene_id: Gene identifier

    Returns:
        List of variant IDs
    """
    # Example mapping of genes to their variants
    gene_variants = {
        'ACTN3': ['rs1815739'],
        'ACE': ['rs4646994', 'rs4340'],
        'AMPD1': ['rs17602729'],
        'CKM': ['rs8111989'],
        'ADRB2': ['rs1042713', 'rs1042714'],
        'PPARA': ['rs4253778'],
        'PPARGC1A': ['rs8192678'],
        'VEGFA': ['rs2010963'],
        'NOS3': ['rs2070744', 'rs1799983'],
        'HIF1A': ['rs11549465'],
        'MSTN': ['rs1805086'],
        'COL5A1': ['rs12722'],
    }
    
    # Normalize the gene ID
    canonical_id = normalize_gene_id(gene_id)
    
    # Return variants if known
    return gene_variants.get(canonical_id, [])


def is_sprint_related_gene(gene_id: str) -> bool:
    """
    Check if a gene is known to be related to sprint performance.

    Args:
        gene_id: Gene identifier

    Returns:
        True if gene is sprint-related, False otherwise
    """
    # Normalize the gene ID
    canonical_id = normalize_gene_id(gene_id)
    
    # Check if in sprint genes set
    return canonical_id in SPRINT_GENES 