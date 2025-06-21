"""
Turbulance DSL Compiler for Gospel

This module provides a Python interface for compiling Turbulance scripts,
which encode scientific hypotheses and experimental designs using a
domain-specific language that validates scientific reasoning.
"""

import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

try:
    from gospel_rust import TurbulanceCompiler as RustTurbulanceCompiler
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logging.warning("Rust Turbulance compiler not available, using Python fallback")

logger = logging.getLogger(__name__)


class TurbulanceCompilationError(Exception):
    """Exception raised when Turbulance compilation fails"""
    pass


class HypothesisValidationError(Exception):
    """Exception raised when hypothesis validation fails"""
    pass


class ScientificReasoningError(Exception):
    """Exception raised when scientific reasoning is flawed"""
    pass


class TurbulanceCompiler:
    """
    Compiler for Turbulance DSL scripts.
    
    Turbulance is a domain-specific language for encoding scientific
    hypotheses and experimental designs. The compiler validates:
    
    1. Scientific hypothesis validity
    2. Statistical soundness
    3. Semantic coherence
    4. Logical reasoning patterns
    
    Example:
        >>> compiler = TurbulanceCompiler()
        >>> script = '''
        ... hypothesis TestHypothesis:
        ...     claim: "Genomic variants predict disease with 85% accuracy"
        ...     requires: "statistical_validation"
        ... '''
        >>> plan = compiler.compile(script)
        >>> print(f"Execution plan has {len(plan.tool_delegations)} delegations")
    """
    
    def __init__(self, use_rust: bool = True):
        """
        Initialize Turbulance compiler.
        
        Args:
            use_rust: Whether to use Rust implementation (faster) or Python fallback
        """
        self.use_rust = use_rust and RUST_AVAILABLE
        
        if self.use_rust:
            self._rust_compiler = RustTurbulanceCompiler()
            logger.info("Initialized Turbulance compiler with Rust backend")
        else:
            logger.info("Initialized Turbulance compiler with Python backend")
            # Would implement Python fallback here
    
    def compile(self, source: str) -> 'ExecutionPlan':
        """
        Compile Turbulance script into execution plan.
        
        Args:
            source: Turbulance script source code
            
        Returns:
            ExecutionPlan containing validated analysis steps
            
        Raises:
            TurbulanceCompilationError: If compilation fails
            HypothesisValidationError: If hypothesis is scientifically invalid
            ScientificReasoningError: If reasoning is flawed
        """
        try:
            if self.use_rust:
                # Use Rust compiler
                rust_plan = self._rust_compiler.compile(source)
                return ExecutionPlan.from_rust(rust_plan)
            else:
                # Use Python fallback
                return self._compile_python(source)
                
        except Exception as e:
            if "HypothesisValidationError" in str(e):
                raise HypothesisValidationError(str(e))
            elif "ScientificReasoningFlaw" in str(e):
                raise ScientificReasoningError(str(e))
            else:
                raise TurbulanceCompilationError(f"Compilation failed: {e}")
    
    def parse(self, source: str) -> 'TurbulanceAST':
        """
        Parse Turbulance script into Abstract Syntax Tree.
        
        Args:
            source: Turbulance script source code
            
        Returns:
            TurbulanceAST representing parsed script
            
        Raises:
            TurbulanceCompilationError: If parsing fails
        """
        try:
            if self.use_rust:
                rust_ast = self._rust_compiler.parse(source)
                return TurbulanceAST.from_rust(rust_ast)
            else:
                return self._parse_python(source)
                
        except Exception as e:
            raise TurbulanceCompilationError(f"Parsing failed: {e}")
    
    def validate(self, ast: 'TurbulanceAST') -> List[str]:
        """
        Validate AST for scientific soundness.
        
        Args:
            ast: Turbulance AST to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        try:
            if self.use_rust:
                # Convert to Rust format and validate
                rust_ast = ast.to_rust()
                self._rust_compiler.validate(rust_ast)
                return []  # No errors
            else:
                return self._validate_python(ast)
                
        except Exception as e:
            return [str(e)]
    
    def compile_file(self, file_path: Union[str, Path]) -> 'ExecutionPlan':
        """
        Compile Turbulance script from file.
        
        Args:
            file_path: Path to .trb file
            
        Returns:
            ExecutionPlan containing validated analysis steps
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Turbulance script not found: {file_path}")
        
        if file_path.suffix != '.trb':
            logger.warning(f"File {file_path} does not have .trb extension")
        
        source = file_path.read_text(encoding='utf-8')
        return self.compile(source)
    
    def _compile_python(self, source: str) -> 'ExecutionPlan':
        """Python fallback compiler implementation"""
        # This would be a simplified Python implementation
        # For now, return minimal execution plan
        return ExecutionPlan(
            hypothesis_validations=[],
            tool_delegations=[],
            execution_order=[],
            semantic_requirements=[]
        )
    
    def _parse_python(self, source: str) -> 'TurbulanceAST':
        """Python fallback parser implementation"""
        # This would be a simplified Python implementation
        return TurbulanceAST(
            imports=[],
            hypotheses=[],
            functions=[],
            propositions=[],
            main_function=None
        )
    
    def _validate_python(self, ast: 'TurbulanceAST') -> List[str]:
        """Python fallback validation implementation"""
        errors = []
        
        # Basic validation checks
        for hypothesis in ast.hypotheses:
            if not hypothesis.claim:
                errors.append(f"Hypothesis '{hypothesis.name}' has empty claim")
            
            # Check for testable predictions
            if not any(word in hypothesis.claim.lower() for word in 
                      ['predict', 'correlate', 'associate', 'cause']):
                errors.append(f"Hypothesis '{hypothesis.name}' lacks testable predictions")
        
        return errors


class TurbulanceAST:
    """Abstract Syntax Tree for Turbulance scripts"""
    
    def __init__(self, imports: List[str], hypotheses: List['Hypothesis'], 
                 functions: List['Function'], propositions: List['Proposition'],
                 main_function: Optional['Function'] = None):
        self.imports = imports
        self.hypotheses = hypotheses
        self.functions = functions
        self.propositions = propositions
        self.main_function = main_function
    
    @classmethod
    def from_rust(cls, rust_ast) -> 'TurbulanceAST':
        """Create TurbulanceAST from Rust AST"""
        # Convert Rust data structures to Python
        return cls(
            imports=rust_ast.imports,
            hypotheses=[Hypothesis.from_rust(h) for h in rust_ast.hypotheses],
            functions=[Function.from_rust(f) for f in rust_ast.functions],
            propositions=[Proposition.from_rust(p) for p in rust_ast.propositions],
            main_function=Function.from_rust(rust_ast.main_function) if rust_ast.main_function else None
        )
    
    def to_rust(self):
        """Convert to Rust format"""
        # This would convert Python structures to Rust format
        # Implementation depends on exact Rust binding interface
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'imports': self.imports,
            'hypotheses': [h.to_dict() for h in self.hypotheses],
            'functions': [f.to_dict() for f in self.functions],
            'propositions': [p.to_dict() for p in self.propositions],
            'main_function': self.main_function.to_dict() if self.main_function else None
        }


class Hypothesis:
    """Scientific hypothesis in Turbulance script"""
    
    def __init__(self, name: str, claim: str, requires: str, 
                 semantic_validation: Optional[Dict[str, str]] = None):
        self.name = name
        self.claim = claim
        self.requires = requires
        self.semantic_validation = semantic_validation or {}
    
    @classmethod
    def from_rust(cls, rust_hypothesis) -> 'Hypothesis':
        """Create Hypothesis from Rust structure"""
        return cls(
            name=rust_hypothesis.name,
            claim=rust_hypothesis.claim,
            requires=rust_hypothesis.requires,
            semantic_validation=rust_hypothesis.semantic_validation
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'name': self.name,
            'claim': self.claim,
            'requires': self.requires,
            'semantic_validation': self.semantic_validation
        }


class Function:
    """Function definition in Turbulance script"""
    
    def __init__(self, name: str, parameters: List[Dict], return_type: str, body: List[Dict]):
        self.name = name
        self.parameters = parameters
        self.return_type = return_type
        self.body = body
    
    @classmethod
    def from_rust(cls, rust_function) -> 'Function':
        """Create Function from Rust structure"""
        return cls(
            name=rust_function.name,
            parameters=rust_function.parameters,
            return_type=rust_function.return_type,
            body=rust_function.body
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'return_type': self.return_type,
            'body': self.body
        }


class Proposition:
    """Scientific proposition in Turbulance script"""
    
    def __init__(self, name: str, motions: List[Dict], within_clauses: List[Dict]):
        self.name = name
        self.motions = motions
        self.within_clauses = within_clauses
    
    @classmethod
    def from_rust(cls, rust_proposition) -> 'Proposition':
        """Create Proposition from Rust structure"""
        return cls(
            name=rust_proposition.name,
            motions=rust_proposition.motions,
            within_clauses=rust_proposition.within_clauses
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'name': self.name,
            'motions': self.motions,
            'within_clauses': self.within_clauses
        }


class ExecutionPlan:
    """Compiled execution plan from Turbulance script"""
    
    def __init__(self, hypothesis_validations: List[Dict], tool_delegations: List[Dict],
                 execution_order: List[Dict], semantic_requirements: List[str]):
        self.hypothesis_validations = hypothesis_validations
        self.tool_delegations = tool_delegations
        self.execution_order = execution_order
        self.semantic_requirements = semantic_requirements
    
    @classmethod
    def from_rust(cls, rust_plan) -> 'ExecutionPlan':
        """Create ExecutionPlan from Rust structure"""
        return cls(
            hypothesis_validations=rust_plan.hypothesis_validations,
            tool_delegations=rust_plan.tool_delegations,
            execution_order=rust_plan.execution_order,
            semantic_requirements=rust_plan.semantic_requirements
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'hypothesis_validations': self.hypothesis_validations,
            'tool_delegations': self.tool_delegations,
            'execution_order': self.execution_order,
            'semantic_requirements': self.semantic_requirements
        }
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save execution plan to JSON file"""
        file_path = Path(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'ExecutionPlan':
        """Load execution plan from JSON file"""
        file_path = Path(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            hypothesis_validations=data['hypothesis_validations'],
            tool_delegations=data['tool_delegations'],
            execution_order=data['execution_order'],
            semantic_requirements=data['semantic_requirements']
        )


def compile_turbulance_script(script_path: Union[str, Path], 
                            output_path: Optional[Union[str, Path]] = None) -> ExecutionPlan:
    """
    Convenience function to compile a Turbulance script.
    
    Args:
        script_path: Path to .trb script file
        output_path: Optional path to save execution plan JSON
        
    Returns:
        ExecutionPlan containing compiled analysis steps
    """
    compiler = TurbulanceCompiler()
    plan = compiler.compile_file(script_path)
    
    if output_path:
        plan.save(output_path)
        logger.info(f"Execution plan saved to {output_path}")
    
    return plan


def validate_turbulance_script(script_path: Union[str, Path]) -> List[str]:
    """
    Convenience function to validate a Turbulance script.
    
    Args:
        script_path: Path to .trb script file
        
    Returns:
        List of validation errors (empty if valid)
    """
    compiler = TurbulanceCompiler()
    script_path = Path(script_path)
    source = script_path.read_text(encoding='utf-8')
    
    try:
        ast = compiler.parse(source)
        return compiler.validate(ast)
    except TurbulanceCompilationError as e:
        return [str(e)]


if __name__ == "__main__":
    # CLI interface for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compiler.py <script.trb>")
        sys.exit(1)
    
    script_path = sys.argv[1]
    
    try:
        # Validate script
        errors = validate_turbulance_script(script_path)
        if errors:
            print("Validation errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
        # Compile script
        plan = compile_turbulance_script(script_path)
        print(f"Successfully compiled {script_path}")
        print(f"  - {len(plan.hypothesis_validations)} hypothesis validations")
        print(f"  - {len(plan.tool_delegations)} tool delegations")
        print(f"  - {len(plan.execution_order)} execution steps")
        print(f"  - {len(plan.semantic_requirements)} semantic requirements")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 