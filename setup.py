from setuptools import setup, find_packages

setup(
    name="gospel",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "biopython>=1.79",
        "networkx>=2.6.0",
        "langchain>=0.0.267",
        "torch>=2.0.0",
        "transformers>=4.15.0",
        "pdfplumber>=0.7.0",
        "chromadb>=0.4.0",
        "PyMuPDF>=1.20.0",
        "pytesseract>=0.3.0",
        "ollama>=0.0.2",
    ],
    entry_points={
        "console_scripts": [
            "gospel=gospel.cli.main:main",
        ],
    },
)
