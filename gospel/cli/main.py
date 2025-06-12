#!/usr/bin/env python3
"""
Command-line interface for the Gospel bioinformatics framework.
"""

import argparse
import sys
import os
from pathlib import Path

from gospel.core import VariantProcessor, VariantAnnotator, GenomicScorer
from gospel.llm import GospelLLM

def main():
    """Main entry point for the Gospel CLI."""
    parser = argparse.ArgumentParser(
        description="Gospel: A comprehensive bioinformatics framework for genomic analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gospel analyze --vcf input.vcf --output results/
  gospel llm --query "What are the effects of BRCA1 mutations?"
  gospel score --genes BRCA1,BRCA2 --output scores.json
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze genomic variants')
    analyze_parser.add_argument('--vcf', required=True, help='Input VCF file')
    analyze_parser.add_argument('--output', required=True, help='Output directory')
    
    # LLM command  
    llm_parser = subparsers.add_parser('llm', help='Query genomic LLM')
    llm_parser.add_argument('--query', required=True, help='Query to ask the LLM')
    
    # Scoring command
    score_parser = subparsers.add_parser('score', help='Score genes/variants')
    score_parser.add_argument('--genes', required=True, help='Comma-separated gene list')
    score_parser.add_argument('--output', required=True, help='Output file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute the appropriate command
    if args.command == 'analyze':
        run_analysis(args.vcf, args.output)
    elif args.command == 'llm':
        run_llm_query(args.query)
    elif args.command == 'score':
        run_scoring(args.genes, args.output)

if __name__ == "__main__":
    main() 