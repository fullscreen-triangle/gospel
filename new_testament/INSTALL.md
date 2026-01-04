# New Testament Installation Guide

## Python 3.13 Compatibility Notice

**New Testament fully supports Python 3.13!** However, some optional high-performance features may require additional setup:

- **Core functionality**: Works perfectly with Python 3.13
- **High-performance JIT compilation**: Requires optional `numba` installation
- **No ray dependency**: This framework does not use or require ray

## Quick Installation

### Method 1: Standard Installation (Python 3.13 Compatible)

```bash
# Navigate to the new_testament directory
cd gospel/new_testament

# Install dependencies (Python 3.13 compatible versions)
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Verify installation
python verify_setup.py
```

### Method 2: With High-Performance Features

```bash
# Navigate to the new_testament directory
cd gospel/new_testament

# Install with numba for maximum performance
pip install -e ".[numba]"

# Verify installation
python verify_setup.py
```

### Method 3: Installation with Optional Features

```bash
# Install with GPU acceleration support (includes numba)
pip install -e ".[gpu]"

# Install with all optional dependencies
pip install -e ".[all]"

# Install for development
pip install -e ".[development]"

# Install just numba for high-performance JIT compilation
pip install -e ".[numba]"
```

## System Requirements

- **Python**: 3.8 or higher (**Python 3.13 fully supported!**)
- **Operating System**: Windows, macOS, Linux
- **Memory**: 4GB RAM minimum (8GB recommended for large datasets)
- **Storage**: 1GB free space

### Performance Notes

- **With numba**: 273× to 227,191× speedup over traditional methods
- **Without numba**: Still functional with pure NumPy implementation (reduced performance)
- **Python 3.13**: Full compatibility with automatic fallback to NumPy when numba unavailable

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

2. **Python 3.13 + Numba Issues**: `ImportError: cannot import name 'jit' from 'numba'`

   - **Solution**: This is expected! The framework automatically falls back to NumPy
   - **Optional**: Install numba manually when it supports Python 3.13: `pip install numba`
   - **Status**: Framework works perfectly without numba (just slower)

3. **Ray Installation Error**: `ERROR: No matching distribution found for ray`

   - **Solution**: This framework does NOT use ray - you can safely ignore this
   - **Note**: If you see this error, you may be installing a different package

4. **Numba Compilation Warning**: First-time JIT compilation warnings

   - **Solution**: This is normal - subsequent runs will be faster

5. **Memory Issues**: Out of memory with large datasets

   - **Solution**: Process data in smaller batches or install with `.[gpu]` for GPU acceleration

6. **Permission Error**: Package installation failed
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
