"""
Command-line interface for the Gospel visualization module.

This module provides a command-line interface for generating
visualizations from Gospel analysis results.
"""

import argparse
import os
from pathlib import Path
import sys
from datetime import datetime

from gospel.visualization import create_dashboard, generate_readme_visualizations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from Gospel analysis results."
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./public/output/results",
        help="Path to the directory containing result files (default: public/output/results)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./public/visualizations",
        help="Path to the directory to save visualizations (default: public/visualizations)",
    )
    
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Timestamp to use for file naming (default: use latest results)",
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display visualizations during generation",
    )
    
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Update README.md with visualization examples",
    )
    
    parser.add_argument(
        "--readme-file",
        type=str,
        default="README.md",
        help="Path to the README.md file to update (default: README.md)",
    )
    
    parser.add_argument(
        "--image-prefix",
        type=str,
        default="./public/visualisations/",
        help="Prefix to add to image paths in markdown (default: public/visualizations/)",
    )
    
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Generate comprehensive visualizations including all available data",
    )
    
    parser.add_argument(
        "--detailed-networks",
        action="store_true",
        help="Generate detailed network analysis visualizations",
    )
    
    parser.add_argument(
        "--all-genes",
        action="store_true",
        help="Generate complete gene list visualizations for all domains",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the visualization CLI."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine if comprehensive visualization is enabled
    include_all_data = args.comprehensive or args.detailed_networks or args.all_genes
    
    try:
        # Generate all visualizations
        print(f"Generating visualizations from {args.results_dir}...")
        output_files = create_dashboard(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            timestamp=args.timestamp,
            show_plots=args.show,
            include_all_data=include_all_data
        )
        
        print(f"Successfully generated {len(output_files)} visualizations in {args.output_dir}")
        
        # Update README.md if requested
        if args.update_readme:
            print(f"Updating README.md at {args.readme_file}...")
            readme_content = generate_readme_visualizations(
                output_files=output_files,
                readme_file=args.readme_file,
                image_prefix=args.image_prefix
            )
            
            # Write the updated README
            with open(args.readme_file, 'w') as f:
                f.write(readme_content)
            
            print(f"Successfully updated README.md with visualization examples")
        
        return 0
    
    except Exception as e:
        print(f"Error generating visualizations: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
