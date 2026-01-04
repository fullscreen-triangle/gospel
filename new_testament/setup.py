#!/usr/bin/env python3
"""
New Testament: St. Stella's Genomic Analysis Framework
Setup and installation configuration for high-performance genomic analysis.

This framework validates the theoretical foundations described in:
- "St. Stella's Sequence: S-Entropy Coordinate Navigation and Cardinal Direction 
  Transformation for Revolutionary Genomic Pattern Recognition"
- "Genomic Information Architecture Through Precision-by-Difference Observer Networks"
- "S-Entropy Semantic Navigation: Coordinate-Based Text Comprehension"

Author: Kundai Farai Sachikonye
Institution: Technical University of Munich
Email: sachikonye@wzw.tum.de
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# Read long description from README
def read_long_description():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "New Testament: St. Stella's Genomic Analysis Framework for validating theoretical genomic analysis frameworks."

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Version information
VERSION = "1.0.0"
DESCRIPTION = "St. Stella's Genomic Analysis Framework - High Performance Cardinal Direction Coordinate Transformation"

# Python version check
if sys.version_info < (3, 8):
    raise RuntimeError("New Testament requires Python 3.8 or higher")

# Check for Python 3.13 compatibility issues
if sys.version_info >= (3, 13):
    print("WARNING: Python 3.13 detected. Some dependencies may have limited support.")
    print("If you encounter installation issues, consider using Python 3.11 or 3.12.")

setup(
    name="new-testament",
    version=VERSION,
    author="Kundai Farai Sachikonye",
    author_email="sachikonye@wzw.tum.de",
    description=DESCRIPTION,
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/fullscreen-triangle/gospel/tree/main/new_testament",
    project_urls={
        "Documentation": "https://github.com/fullscreen-triangle/gospel/tree/main/new_testament/README.md",
        "Source": "https://github.com/fullscreen-triangle/gospel/tree/main/new_testament",
        "Tracker": "https://github.com/fullscreen-triangle/gospel/issues",
        "Publications": "https://github.com/fullscreen-triangle/gospel/tree/main/docs/publication",
    },
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "st_stellas": ["*.md", "*.txt", "*.json"],
        "st_stellas.sequence": ["*.json", "*.txt"],
        "st_stellas.genome": ["*.json", "*.txt"],
    },
    
    # Dependencies - Python 3.13 compatible versions
    install_requires=read_requirements() or [
        # Core scientific computing - Python 3.13 compatible
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.11.0,<2.0.0",
        
        # Data manipulation and analysis  
        "pandas>=2.0.0,<3.0.0",
        
        # Visualization
        "matplotlib>=3.7.0,<4.0.0",
        "seaborn>=0.12.0,<1.0.0",
        
        # Bioinformatics
        "biopython>=1.81,<2.0.0",
        
        # Performance monitoring
        "psutil>=5.9.0,<6.0.0",
        
        # Configuration and I/O
        "pyyaml>=6.0,<7.0.0",
        "h5py>=3.8.0,<4.0.0",
        
        # Testing and validation
        "pytest>=7.0.0,<8.0.0",
        "pytest-benchmark>=4.0.0,<5.0.0",
        
        # Progress tracking
        "tqdm>=4.64.0,<5.0.0",
        
        # Parallel processing
        "joblib>=1.3.0,<2.0.0",
        
        # Memory optimization
        "memory-profiler>=0.61.0,<1.0.0",
    ] + (["numba>=0.58.0,<1.0.0"] if sys.version_info < (3, 13) else []),
    
    # Optional dependencies for advanced features
    extras_require={
        "numba": [
            "numba>=0.58.0,<1.0.0",  # High-performance JIT compilation
        ],
        "gpu": [
            "cupy>=10.0.0",  # GPU acceleration
            "numba[cuda]>=0.58.0",  # CUDA support
        ],
        "visualization": [
            "plotly>=5.0.0",
            "bokeh>=2.4.0",
            "networkx>=2.6.0",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
            "tensorflow>=2.8.0",
            "torch>=1.11.0",
        ],
        "bioinformatics": [
            "pysam>=0.19.0",
            "pyvcf>=0.6.8",
            "pyranges>=0.0.120",
        ],
        "development": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
            "pytest-cov>=3.0.0",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "all": [
            # Include all optional dependencies
            "cupy>=10.0.0",
            "plotly>=5.0.0", "bokeh>=2.4.0", "networkx>=2.6.0",
            "scikit-learn>=1.0.0", "tensorflow>=2.8.0", "torch>=1.11.0",
            "pysam>=0.19.0", "pyvcf>=0.6.8", "pyranges>=0.0.120",
            "black>=22.0.0", "flake8>=4.0.0", "mypy>=0.950",
        ]
    },
    
    # Entry points for command-line interface
    entry_points={
        "console_scripts": [
            "st-stellas=st_stellas.sequence.coordinate_transform:main",
            "new-testament=st_stellas.sequence:print_framework_info",
            "stella-benchmark=st_stellas.sequence.performance_benchmarks:main",
            "stella-dual-strand=st_stellas.sequence.dual_strand_analyzer:main",
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Keywords for discovery
    keywords=[
        "genomics", "bioinformatics", "sequence-analysis", "coordinate-transformation",
        "cardinal-directions", "s-entropy", "pattern-recognition", "dual-strand-analysis",
        "high-performance-computing", "numba", "parallel-processing", "genomic-patterns",
        "dna-analysis", "sequence-optimization", "genomic-coordinates", "stella-sequence"
    ],
    
    # License
    license="MIT",
    
    # Zip safety
    zip_safe=False,
    
    # Testing
    test_suite="tests",
    tests_require=[
        "pytest>=6.2.0",
        "pytest-benchmark>=3.4.0",
        "pytest-cov>=3.0.0",
    ],
)
