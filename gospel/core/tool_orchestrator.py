"""
Tool Orchestrator for External System Integration

This module manages the integration with external tools in the ecosystem
including Autobahn, Hegel, Borgia, Nebuchadnezzar, Bene Gesserit, and Lavoisier.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import subprocess
import tempfile
import os
from pathlib import Path
import time


class ToolStatus(Enum):
    """Status of external tools"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    BUSY = "busy"


@dataclass
class ToolQuery:
    """Query to external tool"""
    tool_name: str
    query_type: str
    data: Dict[str, Any]
    priority: int = 1
    timeout: int = 300  # 5 minutes default
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ToolResponse:
    """Response from external tool"""
    tool_name: str
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class ExternalToolInterface(ABC):
    """Abstract base class for external tool interfaces"""
    
    def __init__(self, tool_name: str, config: Dict[str, Any]):
        self.tool_name = tool_name
        self.config = config
        self.status = ToolStatus.UNAVAILABLE
        self.logger = logging.getLogger(f"{__name__}.{tool_name}")
    
    @abstractmethod
    async def check_availability(self) -> bool:
        """Check if tool is available"""
        pass
    
    @abstractmethod
    async def execute_query(self, query: ToolQuery) -> ToolResponse:
        """Execute query on the tool"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of tool capabilities"""
        pass


class AutobahnInterface(ExternalToolInterface):
    """Interface for Autobahn probabilistic reasoning engine"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("autobahn", config)
        self.base_url = config.get("base_url", "http://localhost:8001")
        self.api_key = config.get("api_key")
        
    async def check_availability(self) -> bool:
        """Check if Autobahn is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=5) as response:
                    if response.status == 200:
                        self.status = ToolStatus.AVAILABLE
                        return True
        except Exception as e:
            self.logger.warning(f"Autobahn unavailable: {e}")
        
        self.status = ToolStatus.UNAVAILABLE
        return False
    
    async def execute_query(self, query: ToolQuery) -> ToolResponse:
        """Execute probabilistic reasoning query"""
        start_time = time.time()
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "query_type": query.query_type,
                "genomic_data": query.data,
                "consciousness_threshold": query.data.get("consciousness_threshold", 0.7),
                "oscillatory_processing": query.data.get("oscillatory_processing", True),
                "bio_metabolic_rag": query.data.get("bio_metabolic_rag", True)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/analyze",
                    json=payload,
                    headers=headers,
                    timeout=query.timeout
                ) as response:
                    
                    if response.status == 200:
                        result_data = await response.json()
                        return ToolResponse(
                            tool_name=self.tool_name,
                            success=True,
                            data=result_data,
                            execution_time=time.time() - start_time
                        )
                    else:
                        error_text = await response.text()
                        return ToolResponse(
                            tool_name=self.tool_name,
                            success=False,
                            data={},
                            error_message=f"HTTP {response.status}: {error_text}",
                            execution_time=time.time() - start_time
                        )
                        
        except Exception as e:
            return ToolResponse(
                tool_name=self.tool_name,
                success=False,
                data={},
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def get_capabilities(self) -> List[str]:
        """Get Autobahn capabilities"""
        return [
            "probabilistic_reasoning",
            "consciousness_aware_processing",
            "oscillatory_bio_metabolic_rag",
            "uncertainty_quantification",
            "biological_intelligence",
            "fire_evolved_cognition"
        ]


class HegelInterface(ExternalToolInterface):
    """Interface for Hegel evidence rectification system"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("hegel", config)
        self.base_url = config.get("base_url", "http://localhost:8002")
        self.api_key = config.get("api_key")
    
    async def check_availability(self) -> bool:
        """Check if Hegel is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/status", timeout=5) as response:
                    if response.status == 200:
                        self.status = ToolStatus.AVAILABLE
                        return True
        except Exception as e:
            self.logger.warning(f"Hegel unavailable: {e}")
        
        self.status = ToolStatus.UNAVAILABLE
        return False
    
    async def execute_query(self, query: ToolQuery) -> ToolResponse:
        """Execute evidence validation query"""
        start_time = time.time()
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "conflicting_annotations": query.data.get("conflicting_annotations", []),
                "confidence_scores": query.data.get("confidence_scores", []),
                "evidence_sources": query.data.get("evidence_sources", []),
                "fuzzy_validation": query.data.get("fuzzy_validation", True),
                "bayesian_rectification": query.data.get("bayesian_rectification", True),
                "federated_learning": query.data.get("federated_learning", False)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/rectify",
                    json=payload,
                    headers=headers,
                    timeout=query.timeout
                ) as response:
                    
                    if response.status == 200:
                        result_data = await response.json()
                        return ToolResponse(
                            tool_name=self.tool_name,
                            success=True,
                            data=result_data,
                            execution_time=time.time() - start_time
                        )
                    else:
                        error_text = await response.text()
                        return ToolResponse(
                            tool_name=self.tool_name,
                            success=False,
                            data={},
                            error_message=f"HTTP {response.status}: {error_text}",
                            execution_time=time.time() - start_time
                        )
                        
        except Exception as e:
            return ToolResponse(
                tool_name=self.tool_name,
                success=False,
                data={},
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def get_capabilities(self) -> List[str]:
        """Get Hegel capabilities"""
        return [
            "evidence_validation",
            "conflict_resolution",
            "fuzzy_bayesian_networks",
            "molecular_identity_rectification",
            "federated_learning",
            "metacognitive_validation"
        ]


class BorgiaInterface(ExternalToolInterface):
    """Interface for Borgia molecular representation engine"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("borgia", config)
        self.executable_path = config.get("executable_path", "borgia")
        self.working_dir = config.get("working_dir", "/tmp/borgia")
    
    async def check_availability(self) -> bool:
        """Check if Borgia is available"""
        try:
            result = subprocess.run(
                [self.executable_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.status = ToolStatus.AVAILABLE
                return True
        except Exception as e:
            self.logger.warning(f"Borgia unavailable: {e}")
        
        self.status = ToolStatus.UNAVAILABLE
        return False
    
    async def execute_query(self, query: ToolQuery) -> ToolResponse:
        """Execute molecular representation query"""
        start_time = time.time()
        
        try:
            # Create temporary input file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(query.data, f)
                input_file = f.name
            
            # Prepare command
            cmd = [
                self.executable_path,
                "--input", input_file,
                "--output-format", "json",
                "--mode", query.query_type
            ]
            
            # Add query-specific parameters
            if query.query_type == "quantum_oscillatory":
                cmd.extend(["--quantum-coherence", "true"])
            elif query.query_type == "temporal_navigation":
                cmd.extend(["--temporal-coordinates", "true"])
            elif query.query_type == "categorical_completion":
                cmd.extend(["--category-analysis", "true"])
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=query.timeout,
                cwd=self.working_dir
            )
            
            # Clean up input file
            os.unlink(input_file)
            
            if result.returncode == 0:
                # Parse output
                try:
                    output_data = json.loads(result.stdout)
                except json.JSONDecodeError:
                    output_data = {"output": result.stdout}
                
                return ToolResponse(
                    tool_name=self.tool_name,
                    success=True,
                    data=output_data,
                    execution_time=time.time() - start_time
                )
            else:
                return ToolResponse(
                    tool_name=self.tool_name,
                    success=False,
                    data={},
                    error_message=result.stderr or "Unknown error",
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            return ToolResponse(
                tool_name=self.tool_name,
                success=False,
                data={},
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def get_capabilities(self) -> List[str]:
        """Get Borgia capabilities"""
        return [
            "quantum_oscillatory_modeling",
            "temporal_coordinate_navigation",
            "categorical_completion_analysis",
            "molecular_representation",
            "deterministic_paradigm",
            "constraint_optimization"
        ]


class NebuchadnezzarInterface(ExternalToolInterface):
    """Interface for Nebuchadnezzar biological circuit system"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("nebuchadnezzar", config)
        self.base_url = config.get("base_url", "http://localhost:8003")
        self.api_key = config.get("api_key")
    
    async def check_availability(self) -> bool:
        """Check if Nebuchadnezzar is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=5) as response:
                    if response.status == 200:
                        self.status = ToolStatus.AVAILABLE
                        return True
        except Exception as e:
            self.logger.warning(f"Nebuchadnezzar unavailable: {e}")
        
        self.status = ToolStatus.UNAVAILABLE
        return False
    
    async def execute_query(self, query: ToolQuery) -> ToolResponse:
        """Execute biological circuit analysis"""
        start_time = time.time()
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "circuit_type": query.query_type,
                "biological_data": query.data,
                "atp_modeling": query.data.get("atp_modeling", True),
                "quantum_membrane": query.data.get("quantum_membrane", True),
                "maxwell_demons": query.data.get("maxwell_demons", True),
                "oscillatory_dynamics": query.data.get("oscillatory_dynamics", True)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/simulate",
                    json=payload,
                    headers=headers,
                    timeout=query.timeout
                ) as response:
                    
                    if response.status == 200:
                        result_data = await response.json()
                        return ToolResponse(
                            tool_name=self.tool_name,
                            success=True,
                            data=result_data,
                            execution_time=time.time() - start_time
                        )
                    else:
                        error_text = await response.text()
                        return ToolResponse(
                            tool_name=self.tool_name,
                            success=False,
                            data={},
                            error_message=f"HTTP {response.status}: {error_text}",
                            execution_time=time.time() - start_time
                        )
                        
        except Exception as e:
            return ToolResponse(
                tool_name=self.tool_name,
                success=False,
                data={},
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def get_capabilities(self) -> List[str]:
        """Get Nebuchadnezzar capabilities"""
        return [
            "atp_based_modeling",
            "quantum_membrane_computation", 
            "biological_maxwells_demons",
            "oscillatory_dynamics",
            "hierarchical_circuits",
            "biological_simulation"
        ]


class BeneGesseritInterface(ExternalToolInterface):
    """Interface for Bene Gesserit membrane quantum computer"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("bene_gesserit", config)
        self.executable_path = config.get("executable_path", "bene-gesserit")
        self.working_dir = config.get("working_dir", "/tmp/bene_gesserit")
    
    async def check_availability(self) -> bool:
        """Check if Bene Gesserit is available"""
        try:
            result = subprocess.run(
                [self.executable_path, "--status"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.status = ToolStatus.AVAILABLE
                return True
        except Exception as e:
            self.logger.warning(f"Bene Gesserit unavailable: {e}")
        
        self.status = ToolStatus.UNAVAILABLE
        return False
    
    async def execute_query(self, query: ToolQuery) -> ToolResponse:
        """Execute membrane quantum computation"""
        start_time = time.time()
        
        try:
            # Create temporary input file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(query.data, f)
                input_file = f.name
            
            # Prepare command
            cmd = [
                self.executable_path,
                "--compute",
                "--input", input_file,
                "--mode", query.query_type,
                "--oscillatory-entropy", "true",
                "--enaqt-transport", "true"
            ]
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=query.timeout,
                cwd=self.working_dir
            )
            
            # Clean up input file
            os.unlink(input_file)
            
            if result.returncode == 0:
                try:
                    output_data = json.loads(result.stdout)
                except json.JSONDecodeError:
                    output_data = {"output": result.stdout}
                
                return ToolResponse(
                    tool_name=self.tool_name,
                    success=True,
                    data=output_data,
                    execution_time=time.time() - start_time
                )
            else:
                return ToolResponse(
                    tool_name=self.tool_name,
                    success=False,
                    data={},
                    error_message=result.stderr or "Unknown error",
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            return ToolResponse(
                tool_name=self.tool_name,
                success=False,
                data={},
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def get_capabilities(self) -> List[str]:
        """Get Bene Gesserit capabilities"""
        return [
            "oscillatory_entropy_control",
            "enaqt_quantum_transport",
            "membrane_quantum_computing",
            "biological_authenticity",
            "atp_constrained_dynamics",
            "endpoint_prediction"
        ]


class LavoisierInterface(ExternalToolInterface):
    """Interface for Lavoisier mass spectrometry analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("lavoisier", config)
        self.base_url = config.get("base_url", "http://localhost:8004")
        self.api_key = config.get("api_key")
    
    async def check_availability(self) -> bool:
        """Check if Lavoisier is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/status", timeout=5) as response:
                    if response.status == 200:
                        self.status = ToolStatus.AVAILABLE
                        return True
        except Exception as e:
            self.logger.warning(f"Lavoisier unavailable: {e}")
        
        self.status = ToolStatus.UNAVAILABLE
        return False
    
    async def execute_query(self, query: ToolQuery) -> ToolResponse:
        """Execute mass spectrometry analysis"""
        start_time = time.time()
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "analysis_type": query.query_type,
                "ms_data": query.data,
                "rust_acceleration": query.data.get("rust_acceleration", True),
                "biological_coherence": query.data.get("biological_coherence", True),
                "molecular_reconstruction": query.data.get("molecular_reconstruction", True)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/analyze",
                    json=payload,
                    headers=headers,
                    timeout=query.timeout
                ) as response:
                    
                    if response.status == 200:
                        result_data = await response.json()
                        return ToolResponse(
                            tool_name=self.tool_name,
                            success=True,
                            data=result_data,
                            execution_time=time.time() - start_time
                        )
                    else:
                        error_text = await response.text()
                        return ToolResponse(
                            tool_name=self.tool_name,
                            success=False,
                            data={},
                            error_message=f"HTTP {response.status}: {error_text}",
                            execution_time=time.time() - start_time
                        )
                        
        except Exception as e:
            return ToolResponse(
                tool_name=self.tool_name,
                success=False,
                data={},
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def get_capabilities(self) -> List[str]:
        """Get Lavoisier capabilities"""
        return [
            "mass_spectrometry_analysis",
            "metabolomics_processing",
            "rust_acceleration",
            "biological_coherence_processing",
            "molecular_reconstruction",
            "3d_visualization"
        ]


class ToolOrchestrator:
    """
    Orchestrates queries to external tools based on Bayesian network decisions.
    
    This class manages the integration with all external tools and handles
    query routing, load balancing, error recovery, and result aggregation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tool orchestrator.
        
        Args:
            config: Configuration dictionary with tool settings
        """
        self.config = config
        self.tools: Dict[str, ExternalToolInterface] = {}
        self.query_queue: List[ToolQuery] = []
        self.active_queries: Dict[str, ToolQuery] = {}
        self.query_history: List[Tuple[ToolQuery, ToolResponse]] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize tool interfaces
        self._initialize_tools()
        
        # Performance tracking
        self.tool_performance: Dict[str, Dict[str, float]] = {}
    
    def _initialize_tools(self):
        """Initialize all tool interfaces"""
        tool_configs = self.config.get("tools", {})
        
        # Initialize each tool interface
        if "autobahn" in tool_configs:
            self.tools["autobahn"] = AutobahnInterface(tool_configs["autobahn"])
        
        if "hegel" in tool_configs:
            self.tools["hegel"] = HegelInterface(tool_configs["hegel"])
        
        if "borgia" in tool_configs:
            self.tools["borgia"] = BorgiaInterface(tool_configs["borgia"])
        
        if "nebuchadnezzar" in tool_configs:
            self.tools["nebuchadnezzar"] = NebuchadnezzarInterface(tool_configs["nebuchadnezzar"])
        
        if "bene_gesserit" in tool_configs:
            self.tools["bene_gesserit"] = BeneGesseritInterface(tool_configs["bene_gesserit"])
        
        if "lavoisier" in tool_configs:
            self.tools["lavoisier"] = LavoisierInterface(tool_configs["lavoisier"])
    
    async def check_tool_availability(self) -> Dict[str, bool]:
        """Check availability of all tools"""
        availability = {}
        
        tasks = []
        for tool_name, tool_interface in self.tools.items():
            tasks.append(self._check_single_tool(tool_name, tool_interface))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (tool_name, _) in enumerate(self.tools.items()):
            if isinstance(results[i], Exception):
                availability[tool_name] = False
                self.logger.error(f"Error checking {tool_name}: {results[i]}")
            else:
                availability[tool_name] = results[i]
        
        return availability
    
    async def _check_single_tool(self, tool_name: str, tool_interface: ExternalToolInterface) -> bool:
        """Check availability of a single tool"""
        try:
            return await tool_interface.check_availability()
        except Exception as e:
            self.logger.error(f"Error checking {tool_name}: {e}")
            return False
    
    async def execute_query(self, query: ToolQuery) -> ToolResponse:
        """
        Execute a query on the specified tool.
        
        Args:
            query: ToolQuery object with query details
            
        Returns:
            ToolResponse with results or error information
        """
        if query.tool_name not in self.tools:
            return ToolResponse(
                tool_name=query.tool_name,
                success=False,
                data={},
                error_message=f"Tool {query.tool_name} not configured"
            )
        
        tool_interface = self.tools[query.tool_name]
        
        # Check if tool is available
        if tool_interface.status != ToolStatus.AVAILABLE:
            is_available = await tool_interface.check_availability()
            if not is_available:
                return ToolResponse(
                    tool_name=query.tool_name,
                    success=False,
                    data={},
                    error_message=f"Tool {query.tool_name} is not available"
                )
        
        # Execute query
        try:
            response = await tool_interface.execute_query(query)
            
            # Update performance tracking
            self._update_performance_tracking(query.tool_name, response)
            
            # Store in history
            self.query_history.append((query, response))
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error executing query on {query.tool_name}: {e}")
            return ToolResponse(
                tool_name=query.tool_name,
                success=False,
                data={},
                error_message=str(e)
            )
    
    async def execute_parallel_queries(self, queries: List[ToolQuery]) -> List[ToolResponse]:
        """
        Execute multiple queries in parallel.
        
        Args:
            queries: List of ToolQuery objects
            
        Returns:
            List of ToolResponse objects
        """
        tasks = []
        for query in queries:
            tasks.append(self.execute_query(query))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                final_responses.append(ToolResponse(
                    tool_name=queries[i].tool_name,
                    success=False,
                    data={},
                    error_message=str(response)
                ))
            else:
                final_responses.append(response)
        
        return final_responses
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        available = []
        for tool_name, tool_interface in self.tools.items():
            if tool_interface.status == ToolStatus.AVAILABLE:
                available.append(tool_name)
        return available
    
    def get_tool_capabilities(self, tool_name: str) -> List[str]:
        """Get capabilities of a specific tool"""
        if tool_name in self.tools:
            return self.tools[tool_name].get_capabilities()
        return []
    
    def get_all_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all tools"""
        capabilities = {}
        for tool_name, tool_interface in self.tools.items():
            capabilities[tool_name] = tool_interface.get_capabilities()
        return capabilities
    
    def _update_performance_tracking(self, tool_name: str, response: ToolResponse):
        """Update performance tracking metrics"""
        if tool_name not in self.tool_performance:
            self.tool_performance[tool_name] = {
                "total_queries": 0,
                "successful_queries": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "success_rate": 0.0
            }
        
        perf = self.tool_performance[tool_name]
        perf["total_queries"] += 1
        perf["total_time"] += response.execution_time
        
        if response.success:
            perf["successful_queries"] += 1
        
        perf["average_time"] = perf["total_time"] / perf["total_queries"]
        perf["success_rate"] = perf["successful_queries"] / perf["total_queries"]
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all tools"""
        return self.tool_performance.copy()
    
    def get_query_history(self, limit: Optional[int] = None) -> List[Tuple[ToolQuery, ToolResponse]]:
        """Get query history"""
        if limit:
            return self.query_history[-limit:]
        return self.query_history.copy()
    
    def clear_history(self):
        """Clear query history"""
        self.query_history.clear()
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current status of the orchestrator"""
        return {
            "configured_tools": list(self.tools.keys()),
            "available_tools": self.get_available_tools(),
            "total_queries": len(self.query_history),
            "performance_metrics": self.get_performance_metrics(),
            "active_queries": len(self.active_queries)
        } 