#!/usr/bin/env python3
"""
Main CLI entry point for Gospel.
"""

import argparse
import sys
from pathlib import Path

from gospel import __version__
from gospel.cli import analyze, query, visualize
from gospel.knowledge_base import KnowledgeBase
from gospel.llm import GospelLLM

def main():
    """Main entry point for the Gospel CLI."""
    parser = argparse.ArgumentParser(
        description="Gospel - Genomic analysis with LLM integration"
    )
    parser.add_argument(
        "--version", action="version", version=f"Gospel {__version__}"
    )
    parser.add_argument(
        "--config", type=str, help="Path to configuration file"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze genomic data")
    analyze.setup_parser(analyze_parser)
    
    # Setup query command
    query_parser = subparsers.add_parser("query", help="Query genomic knowledge base")
    query.setup_parser(query_parser)
    
    # Setup visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize genomic data and networks")
    visualize.setup_parser(viz_parser)
    
    # Setup knowledge base command
    kb_parser = subparsers.add_parser("kb", help="Manage knowledge base")
    kb_subparsers = kb_parser.add_subparsers(dest="kb_command", help="Knowledge base command")
    
    # KB build command
    kb_build_parser = kb_subparsers.add_parser("build", help="Build knowledge base from PDFs")
    kb_build_parser.add_argument("--pdf-dir", type=str, required=True, help="Directory containing PDF reports")
    kb_build_parser.add_argument("--output-dir", type=str, required=True, help="Output directory for knowledge base")
    kb_build_parser.add_argument("--model", type=str, default="llama3", help="Ollama model to use")
    
    # KB query command
    kb_query_parser = kb_subparsers.add_parser("query", help="Query knowledge base")
    kb_query_parser.add_argument("--kb-dir", type=str, required=True, help="Knowledge base directory")
    kb_query_parser.add_argument("--query", type=str, required=True, help="Query string")

    # Setup LLM command
    llm_parser = subparsers.add_parser("llm", help="Work with LLM models")
    llm_subparsers = llm_parser.add_subparsers(dest="llm_command", help="LLM command")
    
    # LLM train command
    llm_train_parser = llm_subparsers.add_parser("train", help="Train domain-specific LLM")
    llm_train_parser.add_argument("--base-model", type=str, default="llama3", help="Base Ollama model")
    llm_train_parser.add_argument("--kb-dir", type=str, required=True, help="Knowledge base directory")
    llm_train_parser.add_argument("--output-dir", type=str, required=True, help="Output directory for trained model")
    
    # LLM query command
    llm_query_parser = llm_subparsers.add_parser("query", help="Query domain-specific LLM")
    llm_query_parser.add_argument("--model-dir", type=str, required=True, help="Trained model directory")
    llm_query_parser.add_argument("--query", type=str, help="Query string")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Handle knowledge base commands
    if args.command == "kb":
        if args.kb_command == "build":
            kb = KnowledgeBase()
            kb.build_from_pdfs(args.pdf_dir, args.output_dir, args.model)
        elif args.kb_command == "query":
            kb = KnowledgeBase.load(args.kb_dir)
            result = kb.query(args.query)
            print(result)
        else:
            kb_parser.print_help()
    
    # Handle LLM commands
    elif args.command == "llm":
        if args.llm_command == "train":
            llm = GospelLLM(args.base_model)
            llm.train(args.kb_dir, args.output_dir)
        elif args.llm_command == "query":
            llm = GospelLLM.load(args.model_dir)
            if args.query:
                result = llm.query(args.query)
                print(result)
            else:
                # Interactive mode
                print("Enter queries (Ctrl+D to exit):")
                try:
                    while True:
                        query = input("> ")
                        result = llm.query(query)
                        print(result)
                except EOFError:
                    print("\nExiting.")
        else:
            llm_parser.print_help()
    
    # Handle other commands by delegating to their modules
    elif args.command == "analyze":
        analyze.run(args)
    elif args.command == "query":
        query.run(args)
    elif args.command == "visualize":
        visualize.run(args)

if __name__ == "__main__":
    main() 