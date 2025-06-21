"""
Turbulance DSL Integration for Gospel

This module provides Turbulance DSL compilation capabilities for Gospel,
enabling researchers to encode scientific hypotheses and experimental
designs in a domain-specific language that validates scientific reasoning.

Turbulance scripts can express:
- Scientific hypotheses with semantic validation
- Experimental designs with statistical soundness checks
- Tool delegation specifications for analysis workflows
- Semantic requirements for biological understanding

Example:
    >>> from gospel.turbulance import TurbulanceCompiler
    >>> compiler = TurbulanceCompiler()
    >>> script = '''
    ... hypothesis VariantPathogenicity:
    ...     claim: "Genomic variants predict pathogenicity with 85% accuracy"
    ...     requires: "statistical_validation"
    ... '''
    >>> plan = compiler.compile(script)
"""

from .compiler import (
    TurbulanceCompiler,
    TurbulanceAST,
    ExecutionPlan,
    Hypothesis,
    Function,
    Proposition,
    TurbulanceCompilationError,
    HypothesisValidationError,
    ScientificReasoningError,
    compile_turbulance_script,
    validate_turbulance_script,
)

__all__ = [
    'TurbulanceCompiler',
    'TurbulanceAST', 
    'ExecutionPlan',
    'Hypothesis',
    'Function',
    'Proposition',
    'TurbulanceCompilationError',
    'HypothesisValidationError', 
    'ScientificReasoningError',
    'compile_turbulance_script',
    'validate_turbulance_script',
]

__version__ = '0.1.0' 