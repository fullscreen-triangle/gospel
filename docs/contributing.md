---
layout: page
title: Contributing
permalink: /contributing/
---

# Contributing to Gospel

We welcome contributions to Gospel! This guide will help you get started with contributing to the project, whether you're fixing bugs, adding features, improving documentation, or extending the framework with new domains.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Contributing Guidelines](#contributing-guidelines)
4. [Code Standards](#code-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Extending Gospel](#extending-gospel)
8. [Community](#community)

## Getting Started

### Ways to Contribute

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new capabilities or improvements
- **Code Contributions**: Implement new features or fix bugs
- **Documentation**: Improve guides, examples, and API documentation
- **Domain Extensions**: Add new genomic analysis domains
- **Testing**: Improve test coverage and quality assurance

### Before You Start

1. Check existing [issues](https://github.com/fullscreen-triangle/gospel/issues) and [pull requests](https://github.com/fullscreen-triangle/gospel/pulls)
2. Read our [Code of Conduct](https://github.com/fullscreen-triangle/gospel/blob/main/CODE_OF_CONDUCT.md)
3. Join our [discussion forum](https://github.com/fullscreen-triangle/gospel/discussions) for questions

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, or virtualenv)

### Setup Instructions

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/gospel.git
cd gospel

# 2. Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -e .
pip install -r requirements-dev.txt

# 4. Setup pre-commit hooks
pre-commit install

# 5. Download test databases
python scripts/setup_test_databases.py

# 6. Run tests to verify setup
pytest tests/
```

### Development Dependencies

The development environment includes:

- **Testing**: pytest, pytest-cov, pytest-mock
- **Code Quality**: black, flake8, mypy, pre-commit
- **Documentation**: sphinx, sphinx-rtd-theme
- **Development Tools**: jupyter, ipython, tox

## Contributing Guidelines

### Issue Reporting

When reporting bugs or requesting features:

```markdown
**Bug Report Template:**

**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With input file '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- Gospel version: [e.g. 1.2.3]

**Additional context**
Any other context about the problem.
```

### Pull Request Process

1. **Create a branch** from `main` for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our code standards

3. **Add tests** for new functionality

4. **Update documentation** if needed

5. **Run the test suite**:
   ```bash
   pytest tests/
   black gospel/
   flake8 gospel/
   mypy gospel/
   ```

6. **Commit your changes** with descriptive messages:
   ```bash
   git commit -m "feat: add new pharmacogenetic pathway analysis"
   ```

7. **Push to your fork** and create a pull request

### Commit Message Format

We follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(fitness): add muscle fiber type analysis
fix(cli): resolve VCF parsing error for large files
docs(api): update domain analyzer documentation
```

## Code Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Use black for automatic formatting
black gospel/

# Line length: 88 characters (black default)
# Use type hints for all public functions
def analyze_variants(variants: List[Variant]) -> AnalysisResults:
    """Analyze genetic variants."""
    pass

# Use docstrings for all public functions
def calculate_score(variant: Variant) -> float:
    """
    Calculate variant impact score.
    
    Args:
        variant: Genetic variant to score
        
    Returns:
        Impact score between 0.0 and 1.0
        
    Raises:
        ValueError: If variant data is invalid
    """
    pass
```

### Code Organization

```
gospel/
├── core/                   # Core functionality
│   ├── variant.py         # Variant data structures
│   ├── scoring.py         # Scoring algorithms
│   └── analysis.py        # Main analysis engine
├── domains/               # Domain-specific analyzers
│   ├── fitness.py         # Fitness domain
│   ├── pharmacogenetics.py # Pharmacogenetics domain
│   └── nutrition.py       # Nutrition domain
├── llm/                   # AI/LLM integration
├── cli/                   # Command-line interface
├── utils/                 # Utility functions
└── tests/                 # Test suite
```

### Error Handling

```python
# Use specific exception types
class GospelError(Exception):
    """Base exception for Gospel errors."""
    pass

class VariantProcessingError(GospelError):
    """Error in variant processing."""
    pass

# Provide helpful error messages
def process_vcf(vcf_file: str) -> List[Variant]:
    if not os.path.exists(vcf_file):
        raise FileNotFoundError(f"VCF file not found: {vcf_file}")
    
    try:
        # Process VCF
        pass
    except Exception as e:
        raise VariantProcessingError(
            f"Failed to process VCF file {vcf_file}: {e}"
        ) from e
```

## Testing

### Test Structure

```
tests/
├── unit/                  # Unit tests
│   ├── test_variant.py
│   ├── test_scoring.py
│   └── test_domains/
├── integration/           # Integration tests
│   ├── test_cli.py
│   └── test_analysis.py
├── fixtures/              # Test data
│   ├── sample.vcf
│   └── expected_results.json
└── conftest.py           # Pytest configuration
```

### Writing Tests

```python
import pytest
from gospel.core.variant import Variant
from gospel.domains.fitness import FitnessAnalyzer

class TestFitnessAnalyzer:
    """Test fitness domain analysis."""
    
    def test_sprint_analysis(self):
        """Test sprint performance analysis."""
        # Arrange
        analyzer = FitnessAnalyzer()
        variant = Variant(
            id="rs1815739",
            gene="ACTN3",
            genotype="RR"
        )
        
        # Act
        result = analyzer.analyze_sprint_performance([variant])
        
        # Assert
        assert result.sprint_score > 0.8
        assert "power" in result.advantages
    
    @pytest.mark.parametrize("genotype,expected_score", [
        ("RR", 0.9),
        ("RX", 0.6),
        ("XX", 0.3)
    ])
    def test_actn3_scoring(self, genotype, expected_score):
        """Test ACTN3 variant scoring."""
        analyzer = FitnessAnalyzer()
        variant = Variant(id="rs1815739", genotype=genotype)
        score = analyzer.score_actn3_variant(variant)
        assert abs(score - expected_score) < 0.1
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gospel

# Run specific test file
pytest tests/unit/test_variant.py

# Run tests matching pattern
pytest -k "test_fitness"

# Run tests with verbose output
pytest -v
```

## Documentation

### Documentation Standards

- Use clear, concise language
- Include code examples for all public APIs
- Provide real-world use cases
- Keep documentation up-to-date with code changes

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build HTML documentation
cd docs/
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html/
```

### Documentation Types

1. **API Documentation**: Docstrings in code
2. **User Guides**: Step-by-step tutorials
3. **Reference**: Complete API and CLI reference
4. **Examples**: Real-world use cases

## Extending Gospel

### Adding New Domains

To add a new genomic domain:

1. **Create domain module**:
   ```python
   # gospel/domains/new_domain.py
   from gospel.core.domain import BaseDomain
   
   class NewDomainAnalyzer(BaseDomain):
       """Analyzer for new genomic domain."""
       
       def analyze_variants(self, variants: List[Variant]) -> DomainResults:
           """Implement domain-specific analysis."""
           pass
   ```

2. **Register domain**:
   ```python
   # gospel/domains/__init__.py
   from .new_domain import NewDomainAnalyzer
   
   AVAILABLE_DOMAINS = {
       'fitness': FitnessAnalyzer,
       'pharmacogenetics': PharmacogeneticsAnalyzer,
       'nutrition': NutritionAnalyzer,
       'new_domain': NewDomainAnalyzer,  # Add here
   }
   ```

3. **Add tests**:
   ```python
   # tests/unit/test_domains/test_new_domain.py
   class TestNewDomainAnalyzer:
       def test_analysis(self):
           # Test implementation
           pass
   ```

4. **Update documentation**:
   - Add domain description to `domains.md`
   - Update CLI help text
   - Add examples

### Adding New Scoring Algorithms

```python
# gospel/core/scoring.py
from abc import ABC, abstractmethod

class VariantScorer(ABC):
    """Base class for variant scoring algorithms."""
    
    @abstractmethod
    def score_variant(self, variant: Variant) -> float:
        """Score a genetic variant."""
        pass

class CustomScorer(VariantScorer):
    """Custom scoring algorithm."""
    
    def score_variant(self, variant: Variant) -> float:
        # Implement custom scoring logic
        return score
```

### Adding New Data Sources

```python
# gospel/data/sources.py
class CustomDataSource:
    """Custom genomic data source."""
    
    def fetch_variant_data(self, variant_id: str) -> Dict:
        """Fetch variant data from custom source."""
        pass
    
    def get_population_frequencies(self, variant_id: str) -> Dict:
        """Get population frequency data."""
        pass
```

## Community

### Communication Channels

- **GitHub Discussions**: General questions and community discussions
- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions and reviews

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](https://github.com/fullscreen-triangle/gospel/blob/main/CODE_OF_CONDUCT.md).

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Documentation acknowledgments

## Getting Help

If you need help contributing:

1. Check the [documentation](https://fullscreen-triangle.github.io/gospel/)
2. Search [existing discussions](https://github.com/fullscreen-triangle/gospel/discussions)
3. Ask questions in [GitHub Discussions](https://github.com/fullscreen-triangle/gospel/discussions/new)
4. Join our community calls (announced in discussions)

---

**Thank you for contributing to Gospel!** Your contributions help advance the field of genomic analysis and personalized medicine. 