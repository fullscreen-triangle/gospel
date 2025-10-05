# New Testament Installation Guide

## Quick Installation

### Method 1: Development Installation (Recommended)

```bash
# Navigate to the new_testament directory
cd gospel/new_testament

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Verify installation
python verify_setup.py
```

### Method 2: Standard Installation

```bash
# Navigate to the new_testament directory
cd gospel/new_testament

# Install the package
pip install .

# Verify installation
python verify_setup.py
```

### Method 3: Installation with Optional Features

```bash
# Install with GPU acceleration support
pip install -e ".[gpu]"

# Install with all optional dependencies
pip install -e ".[all]"

# Install for development
pip install -e ".[development]"
```

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, Linux
- **Memory**: 4GB RAM minimum (8GB recommended for large datasets)
- **Storage**: 1GB free space

## Verification

After installation, run the verification script:

```bash
python verify_setup.py
```

This will check:

- ✓ Python version compatibility
- ✓ All required dependencies
- ✓ Package structure integrity
- ✓ Core functionality
- ✓ Performance characteristics
- ✓ Basic benchmark execution

## Quick Start Test

```python
from st_stellas.sequence import StStellaSequenceTransformer

# Initialize transformer
transformer = StStellaSequenceTransformer()

# Transform a DNA sequence to coordinates
sequence = "ATGCGTACGTA"
coordinates = transformer.transform_sequence(sequence)

print(f"Sequence: {sequence}")
print(f"Coordinates shape: {coordinates.shape}")
print("✓ New Testament framework is working!")
```

## Command Line Tools

After installation, these commands are available:

```bash
# Display framework information
new-testament

# Run coordinate transformation
st-stellas --sequences ATGCGTACGTA GCTATCGATGC

# Run performance benchmarks
stella-benchmark --input sequences.fasta

# Run dual-strand analysis
stella-dual-strand --input sequences.fasta --palindromes
```

## Troubleshooting

### Common Issues

1. **Import Error**: `ModuleNotFoundError: No module named 'st_stellas'`

   - **Solution**: Run `pip install -e .` from the new_testament directory

2. **Numba Compilation Warning**: First-time JIT compilation warnings

   - **Solution**: This is normal - subsequent runs will be faster

3. **Memory Issues**: Out of memory with large datasets

   - **Solution**: Process data in smaller batches or install with `.[gpu]` for GPU acceleration

4. **Permission Error**: Package installation failed
   - **Solution**: Use `pip install --user -e .` or virtual environment

### Performance Optimization

For optimal performance:

```bash
# Install with GPU support (if CUDA available)
pip install -e ".[gpu]"

# Install with all optimizations
pip install -e ".[all]"
```

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv new_testament_env

# Activate (Windows)
new_testament_env\Scripts\activate

# Activate (macOS/Linux)
source new_testament_env/bin/activate

# Install
pip install -e .

# Verify
python verify_setup.py
```

## Support

If you encounter issues:

1. Run `python verify_setup.py` for diagnostic information
2. Check that all dependencies in `requirements.txt` are installed
3. Verify Python version is 3.8+
4. Ensure you're in the correct directory (`gospel/new_testament`)

## Next Steps

After successful installation:

1. **Read the documentation**: `README.md`
2. **Run examples**: Explore the sequence analysis capabilities
3. **Run benchmarks**: Validate the performance claims
4. **Integrate with your workflows**: Use the API for your genomic analysis needs

The New Testament framework is now ready to validate St. Stella's genomic analysis theories with high-performance coordinate transformation!
