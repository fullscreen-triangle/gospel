"""
Utility functions for Gospel.
"""

from gospel.utils.gene_utils import find_gene_mentions, normalize_gene_id
from gospel.utils.network_utils import create_gene_network, merge_networks

__all__ = ["find_gene_mentions", "normalize_gene_id", "create_gene_network", "merge_networks"]
