"""
External Tool Interfaces for Gospel Framework.

This module implements interfaces for integrating Gospel with specialized
external tools in the genomic analysis ecosystem, including Autobahn,
Hegel, Borgia, Nebuchadnezzar, Bene Gesserit, and Lavoisier.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import aiohttp
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ToolQuery:
    """Query structure for external tools"""
    tool_name: str
    query_type: str
    data: Dict[str, Any]
    priority: str = "normal"  # low, normal, high, urgent
    timeout: int = 300  # seconds
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ToolResponse:
    """Response structure from external tools"""
    tool_name: str
    query_id: str
    success: bool
    data: Dict[str, Any]
    confidence: float
    processing_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ExternalToolInterface(ABC):
    """Abstract base class for external tool interfaces"""
    
    def __init__(self, tool_name: str, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.tool_name = tool_name
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
        self.available = False
        
    async def initialize(self) -> bool:
        """Initialize connection to external tool"""
        try:
            self.session = aiohttp.ClientSession()
            self.available = await self.check_availability()
            logger.info(f"{self.tool_name} interface initialized. Available: {self.available}")
            return self.available
        except Exception as e:
            logger.error(f"Failed to initialize {self.tool_name}: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def check_availability(self) -> bool:
        """Check if the external tool is available"""
        pass
    
    @abstractmethod
    async def process_query(self, query: ToolQuery) -> ToolResponse:
        """Process a query using the external tool"""
        pass
    
    def is_available(self) -> bool:
        """Check if tool is available"""
        return self.available


class AutobahnInterface(ExternalToolInterface):
    """
    Interface for Autobahn - Probabilistic reasoning and consciousness-aware processing.
    
    Autobahn is described as a room temperature quantum computer with
    oscillatory bio-metabolic RAG system for handling probabilistic genomic uncertainty.
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__("autobahn", base_url or "http://localhost:8080", api_key)
    
    async def check_availability(self) -> bool:
        """Check if Autobahn is available"""
        if not self.base_url:
            return False
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def process_query(self, query: ToolQuery) -> ToolResponse:
        """Process probabilistic reasoning queries"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if query.query_type == "probabilistic_reasoning":
                result = await self.query_probabilistic_reasoning(query.data)
            elif query.query_type == "consciousness_aware_processing":
                result = await self.query_consciousness_processing(query.data)
            elif query.query_type == "oscillatory_analysis":
                result = await self.query_oscillatory_analysis(query.data)
            else:
                raise ValueError(f"Unknown query type: {query.query_type}")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return ToolResponse(
                tool_name=self.tool_name,
                query_id=query.metadata.get("query_id", "unknown") if query.metadata else "unknown",
                success=True,
                data=result,
                confidence=result.get("confidence", 0.8),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Autobahn query failed: {e}")
            
            return ToolResponse(
                tool_name=self.tool_name,
                query_id=query.metadata.get("query_id", "unknown") if query.metadata else "unknown",
                success=False,
                data={},
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def query_probabilistic_reasoning(self, genomic_uncertainty: Dict[str, Any]) -> Dict[str, Any]:
        """Query Autobahn for consciousness-aware genomic reasoning"""
        
        autobahn_query = {
            "type": "probabilistic_genomic_analysis",
            "variants": genomic_uncertainty.get("variant_list", []),
            "uncertainty_bounds": genomic_uncertainty.get("confidence_intervals", {}),
            "biological_context": genomic_uncertainty.get("pathway_context", {}),
            "consciousness_threshold": 0.7,
            "oscillatory_processing": True
        }
        
        async with self.session.post(
            f"{self.base_url}/probabilistic_reasoning",
            json=autobahn_query,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Autobahn API error: {response.status}")
    
    async def query_consciousness_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Query consciousness-aware processing capabilities"""
        
        consciousness_query = {
            "type": "consciousness_aware_genomics",
            "genomic_data": data.get("genomic_data", {}),
            "awareness_level": data.get("awareness_level", "high"),
            "metabolic_constraints": data.get("metabolic_constraints", {})
        }
        
        async with self.session.post(
            f"{self.base_url}/consciousness_processing",
            json=consciousness_query,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Consciousness processing error: {response.status}")
    
    async def query_oscillatory_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Query oscillatory bio-metabolic analysis"""
        
        oscillatory_query = {
            "type": "oscillatory_genomic_analysis",
            "time_series_data": data.get("time_series", []),
            "oscillation_parameters": data.get("oscillation_params", {}),
            "bio_metabolic_constraints": data.get("metabolic_constraints", {})
        }
        
        async with self.session.post(
            f"{self.base_url}/oscillatory_analysis",
            json=oscillatory_query,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Oscillatory analysis error: {response.status}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Autobahn requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class HegelInterface(ExternalToolInterface):
    """
    Interface for Hegel - Evidence rectification framework for biological molecules.
    
    Hegel implements hybrid fuzzy-Bayesian evidence networks and federated learning
    for multi-institutional genomic collaboration.
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__("hegel", base_url or "http://localhost:8081", api_key)
    
    async def check_availability(self) -> bool:
        """Check if Hegel is available"""
        if not self.base_url:
            return False
        
        try:
            async with self.session.get(f"{self.base_url}/status") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def process_query(self, query: ToolQuery) -> ToolResponse:
        """Process evidence validation queries"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if query.query_type == "evidence_validation":
                result = await self.validate_conflicting_evidence(query.data)
            elif query.query_type == "fuzzy_bayesian_analysis":
                result = await self.fuzzy_bayesian_analysis(query.data)
            elif query.query_type == "federated_learning":
                result = await self.federated_learning_query(query.data)
            else:
                raise ValueError(f"Unknown query type: {query.query_type}")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return ToolResponse(
                tool_name=self.tool_name,
                query_id=query.metadata.get("query_id", "unknown") if query.metadata else "unknown",
                success=True,
                data=result,
                confidence=result.get("confidence", 0.85),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Hegel query failed: {e}")
            
            return ToolResponse(
                tool_name=self.tool_name,
                query_id=query.metadata.get("query_id", "unknown") if query.metadata else "unknown",
                success=False,
                data={},
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def validate_conflicting_evidence(self, evidence_conflicts: Dict[str, Any]) -> Dict[str, Any]:
        """Query Hegel for fuzzy-Bayesian evidence validation"""
        
        validation_request = {
            "conflicting_annotations": evidence_conflicts.get("annotations", {}),
            "confidence_scores": evidence_conflicts.get("confidence_values", {}),
            "evidence_sources": evidence_conflicts.get("databases", []),
            "fuzzy_validation": True,
            "federated_learning": True,
            "validation_modules": {
                "mzekezeke": True,
                "diggiden": True,
                "hatata": True,
                "spectacular": True,
                "nicotine": True
            }
        }
        
        async with self.session.post(
            f"{self.base_url}/evidence_validation",
            json=validation_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Hegel evidence validation error: {response.status}")
    
    async def fuzzy_bayesian_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fuzzy-Bayesian evidence analysis"""
        
        analysis_request = {
            "evidence_data": data.get("evidence", {}),
            "prior_knowledge": data.get("priors", {}),
            "fuzzy_parameters": data.get("fuzzy_params", {}),
            "bayesian_network": data.get("network_structure", {})
        }
        
        async with self.session.post(
            f"{self.base_url}/fuzzy_bayesian",
            json=analysis_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Fuzzy-Bayesian analysis error: {response.status}")
    
    async def federated_learning_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Query federated learning capabilities"""
        
        federated_request = {
            "local_data": data.get("local_genomic_data", {}),
            "collaboration_partners": data.get("partners", []),
            "privacy_level": data.get("privacy_level", "high"),
            "learning_objective": data.get("objective", "evidence_rectification")
        }
        
        async with self.session.post(
            f"{self.base_url}/federated_learning",
            json=federated_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Federated learning error: {response.status}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Hegel requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class BorgiaInterface(ExternalToolInterface):
    """
    Interface for Borgia - Quantum-oscillatory molecular representation engine.
    
    Borgia implements temporal predetermination theory and categorical completion
    principles with Environment-Assisted Quantum Transport (ENAQT).
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__("borgia", base_url or "http://localhost:8082", api_key)
    
    async def check_availability(self) -> bool:
        """Check if Borgia is available"""
        if not self.base_url:
            return False
        
        try:
            async with self.session.get(f"{self.base_url}/health_check") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def process_query(self, query: ToolQuery) -> ToolResponse:
        """Process molecular representation queries"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if query.query_type == "molecular_representation":
                result = await self.molecular_representation_query(query.data)
            elif query.query_type == "quantum_transport":
                result = await self.quantum_transport_analysis(query.data)
            elif query.query_type == "temporal_predetermination":
                result = await self.temporal_predetermination_query(query.data)
            else:
                raise ValueError(f"Unknown query type: {query.query_type}")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return ToolResponse(
                tool_name=self.tool_name,
                query_id=query.metadata.get("query_id", "unknown") if query.metadata else "unknown",
                success=True,
                data=result,
                confidence=result.get("confidence", 0.9),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Borgia query failed: {e}")
            
            return ToolResponse(
                tool_name=self.tool_name,
                query_id=query.metadata.get("query_id", "unknown") if query.metadata else "unknown",
                success=False,
                data={},
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def molecular_representation_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Query molecular representation capabilities"""
        
        representation_request = {
            "molecules": data.get("molecular_structures", []),
            "representation_type": data.get("type", "quantum_oscillatory"),
            "environmental_constraints": data.get("environment", {}),
            "quantum_parameters": data.get("quantum_params", {})
        }
        
        async with self.session.post(
            f"{self.base_url}/molecular_representation",
            json=representation_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Molecular representation error: {response.status}")
    
    async def quantum_transport_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Environment-Assisted Quantum Transport (ENAQT)"""
        
        transport_request = {
            "molecular_system": data.get("system", {}),
            "transport_pathways": data.get("pathways", []),
            "environmental_assistance": data.get("environment_params", {}),
            "quantum_coherence": data.get("coherence_parameters", {})
        }
        
        async with self.session.post(
            f"{self.base_url}/quantum_transport",
            json=transport_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Quantum transport analysis error: {response.status}")
    
    async def temporal_predetermination_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Query temporal predetermination analysis"""
        
        temporal_request = {
            "genomic_sequences": data.get("sequences", []),
            "temporal_constraints": data.get("time_constraints", {}),
            "predetermination_parameters": data.get("pred_params", {}),
            "categorical_completion": data.get("completion_criteria", {})
        }
        
        async with self.session.post(
            f"{self.base_url}/temporal_predetermination",
            json=temporal_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Temporal predetermination error: {response.status}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Borgia requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class NebuchadnezzarInterface(ExternalToolInterface):
    """
    Interface for Nebuchadnezzar - Hierarchical probabilistic electric circuit system.
    
    Nebuchadnezzar implements ATP-based rate modeling and quantum-enhanced circuit 
    modeling with consciousness integration for biological simulation.
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__("nebuchadnezzar", base_url or "http://localhost:8083", api_key)
    
    async def check_availability(self) -> bool:
        """Check if Nebuchadnezzar is available"""
        if not self.base_url:
            return False
        
        try:
            async with self.session.get(f"{self.base_url}/circuit_status") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def process_query(self, query: ToolQuery) -> ToolResponse:
        """Process biological circuit simulation queries"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if query.query_type == "circuit_simulation":
                result = await self.circuit_simulation_query(query.data)
            elif query.query_type == "atp_modeling":
                result = await self.atp_rate_modeling(query.data)
            elif query.query_type == "quantum_circuit":
                result = await self.quantum_circuit_analysis(query.data)
            else:
                raise ValueError(f"Unknown query type: {query.query_type}")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return ToolResponse(
                tool_name=self.tool_name,
                query_id=query.metadata.get("query_id", "unknown") if query.metadata else "unknown",
                success=True,
                data=result,
                confidence=result.get("confidence", 0.88),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Nebuchadnezzar query failed: {e}")
            
            return ToolResponse(
                tool_name=self.tool_name,
                query_id=query.metadata.get("query_id", "unknown") if query.metadata else "unknown",
                success=False,
                data={},
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def circuit_simulation_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Query biological circuit simulation"""
        
        circuit_request = {
            "biological_circuit": data.get("circuit_topology", {}),
            "component_parameters": data.get("components", {}),
            "simulation_time": data.get("time_horizon", 1000),
            "hierarchical_levels": data.get("hierarchy", ["molecular", "cellular", "tissue"])
        }
        
        async with self.session.post(
            f"{self.base_url}/circuit_simulation",
            json=circuit_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Circuit simulation error: {response.status}")
    
    async def atp_rate_modeling(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ATP-based rate modeling"""
        
        atp_request = {
            "metabolic_network": data.get("network", {}),
            "atp_constraints": data.get("atp_params", {}),
            "rate_equations": data.get("kinetics", {}),
            "cellular_context": data.get("cell_type", "generic")
        }
        
        async with self.session.post(
            f"{self.base_url}/atp_modeling",
            json=atp_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"ATP modeling error: {response.status}")
    
    async def quantum_circuit_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quantum-enhanced biological circuits"""
        
        quantum_request = {
            "quantum_components": data.get("quantum_elements", {}),
            "classical_interface": data.get("classical_components", {}),
            "consciousness_integration": data.get("consciousness_params", {}),
            "quantum_coherence": data.get("coherence_time", 100)
        }
        
        async with self.session.post(
            f"{self.base_url}/quantum_circuit",
            json=quantum_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Quantum circuit analysis error: {response.status}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Nebuchadnezzar requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class BeneGesseritInterface(ExternalToolInterface):
    """
    Interface for Bene Gesserit - Biological quantum computer.
    
    Bene Gesserit implements oscillatory entropy control and ATP-constrained 
    dynamics for membrane quantum computing at room temperature.
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__("bene_gesserit", base_url or "http://localhost:8084", api_key)
    
    async def check_availability(self) -> bool:
        """Check if Bene Gesserit is available"""
        if not self.base_url:
            return False
        
        try:
            async with self.session.get(f"{self.base_url}/quantum_status") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def process_query(self, query: ToolQuery) -> ToolResponse:
        """Process quantum biological computing queries"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if query.query_type == "quantum_computation":
                result = await self.quantum_computation_query(query.data)
            elif query.query_type == "entropy_control":
                result = await self.entropy_control_analysis(query.data)
            elif query.query_type == "membrane_computing":
                result = await self.membrane_computing_query(query.data)
            else:
                raise ValueError(f"Unknown query type: {query.query_type}")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return ToolResponse(
                tool_name=self.tool_name,
                query_id=query.metadata.get("query_id", "unknown") if query.metadata else "unknown",
                success=True,
                data=result,
                confidence=result.get("confidence", 0.92),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Bene Gesserit query failed: {e}")
            
            return ToolResponse(
                tool_name=self.tool_name,
                query_id=query.metadata.get("query_id", "unknown") if query.metadata else "unknown",
                success=False,
                data={},
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def quantum_computation_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Query quantum biological computation"""
        
        quantum_request = {
            "biological_qubits": data.get("qubits", {}),
            "quantum_algorithms": data.get("algorithms", []),
            "room_temperature": True,
            "biological_constraints": data.get("bio_constraints", {})
        }
        
        async with self.session.post(
            f"{self.base_url}/quantum_computation",
            json=quantum_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Quantum computation error: {response.status}")
    
    async def entropy_control_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze oscillatory entropy control"""
        
        entropy_request = {
            "oscillation_parameters": data.get("oscillations", {}),
            "entropy_distribution": data.get("entropy_map", {}),
            "control_mechanisms": data.get("control_systems", {}),
            "biological_context": data.get("bio_context", {})
        }
        
        async with self.session.post(
            f"{self.base_url}/entropy_control",
            json=entropy_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Entropy control analysis error: {response.status}")
    
    async def membrane_computing_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Query membrane quantum computing capabilities"""
        
        membrane_request = {
            "membrane_structure": data.get("membrane", {}),
            "quantum_processes": data.get("quantum_ops", []),
            "atp_dynamics": data.get("atp_constraints", {}),
            "metabolic_coupling": data.get("metabolism", {})
        }
        
        async with self.session.post(
            f"{self.base_url}/membrane_computing",
            json=membrane_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Membrane computing error: {response.status}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Bene Gesserit requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class LavoisierInterface(ExternalToolInterface):
    """
    Interface for Lavoisier - Mass spectrometry analysis framework.
    
    Lavoisier implements dual-pipeline processing and six AI modules for 
    comprehensive analytical validation with proteomics integration.
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__("lavoisier", base_url or "http://localhost:8085", api_key)
    
    async def check_availability(self) -> bool:
        """Check if Lavoisier is available"""
        if not self.base_url:
            return False
        
        try:
            async with self.session.get(f"{self.base_url}/ms_status") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def process_query(self, query: ToolQuery) -> ToolResponse:
        """Process mass spectrometry analysis queries"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if query.query_type == "mass_spec_analysis":
                result = await self.mass_spec_analysis_query(query.data)
            elif query.query_type == "proteomics_integration":
                result = await self.proteomics_integration_query(query.data)
            elif query.query_type == "dual_pipeline":
                result = await self.dual_pipeline_analysis(query.data)
            else:
                raise ValueError(f"Unknown query type: {query.query_type}")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return ToolResponse(
                tool_name=self.tool_name,
                query_id=query.metadata.get("query_id", "unknown") if query.metadata else "unknown",
                success=True,
                data=result,
                confidence=result.get("confidence", 0.87),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Lavoisier query failed: {e}")
            
            return ToolResponse(
                tool_name=self.tool_name,
                query_id=query.metadata.get("query_id", "unknown") if query.metadata else "unknown",
                success=False,
                data={},
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def mass_spec_analysis_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Query mass spectrometry analysis"""
        
        ms_request = {
            "spectra_data": data.get("spectra", []),
            "analysis_mode": data.get("mode", "comprehensive"),
            "ai_modules": {
                "module_1": True,
                "module_2": True,
                "module_3": True,
                "module_4": True,
                "module_5": True,
                "module_6": True
            },
            "validation_level": data.get("validation", "high")
        }
        
        async with self.session.post(
            f"{self.base_url}/mass_spec_analysis",
            json=ms_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Mass spec analysis error: {response.status}")
    
    async def proteomics_integration_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Query proteomics integration with InstaNovo and DeepNovo"""
        
        proteomics_request = {
            "protein_sequences": data.get("sequences", []),
            "mass_spec_data": data.get("ms_data", {}),
            "instanovo_params": data.get("instanovo_config", {}),
            "deepnovo_params": data.get("deepnovo_config", {}),
            "integration_mode": data.get("integration", "full")
        }
        
        async with self.session.post(
            f"{self.base_url}/proteomics_integration",
            json=proteomics_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Proteomics integration error: {response.status}")
    
    async def dual_pipeline_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform dual-pipeline processing (numerical + visual)"""
        
        dual_request = {
            "numerical_pipeline": data.get("numerical_data", {}),
            "visual_pipeline": data.get("visual_data", {}),
            "pipeline_fusion": data.get("fusion_method", "weighted_average"),
            "validation_criteria": data.get("validation", {})
        }
        
        async with self.session.post(
            f"{self.base_url}/dual_pipeline",
            json=dual_request,
            headers=self._get_headers()
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Dual pipeline analysis error: {response.status}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Lavoisier requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class ExternalToolManager:
    """
    Manager for all external tool interfaces.
    
    This class coordinates communication with all external tools in the
    Gospel ecosystem and provides a unified interface for tool orchestration.
    """
    
    def __init__(self, config: Optional[Dict[str, Dict[str, Any]]] = None):
        self.config = config or {}
        self.tools = {}
        self.initialized = False
    
    async def initialize_tools(self) -> Dict[str, bool]:
        """Initialize all available external tools"""
        
        tool_classes = {
            "autobahn": AutobahnInterface,
            "hegel": HegelInterface,
            "borgia": BorgiaInterface,
            "nebuchadnezzar": NebuchadnezzarInterface,
            "bene_gesserit": BeneGesseritInterface,
            "lavoisier": LavoisierInterface
        }
        
        initialization_results = {}
        
        for tool_name, tool_class in tool_classes.items():
            try:
                tool_config = self.config.get(tool_name, {})
                base_url = tool_config.get("base_url")
                api_key = tool_config.get("api_key")
                
                tool_interface = tool_class(base_url=base_url, api_key=api_key)
                success = await tool_interface.initialize()
                
                if success:
                    self.tools[tool_name] = tool_interface
                    initialization_results[tool_name] = True
                    logger.info(f"Successfully initialized {tool_name}")
                else:
                    initialization_results[tool_name] = False
                    logger.warning(f"Failed to initialize {tool_name}")
                
            except Exception as e:
                initialization_results[tool_name] = False
                logger.error(f"Error initializing {tool_name}: {e}")
        
        self.initialized = True
        logger.info(f"Initialized {len(self.tools)} external tools")
        return initialization_results
    
    async def query_tool(self, tool_name: str, query: ToolQuery) -> Optional[ToolResponse]:
        """Query a specific external tool"""
        
        if not self.initialized:
            await self.initialize_tools()
        
        if tool_name not in self.tools:
            logger.error(f"Tool {tool_name} not available")
            return None
        
        if not self.tools[tool_name].is_available():
            logger.error(f"Tool {tool_name} is not available")
            return None
        
        try:
            response = await self.tools[tool_name].process_query(query)
            logger.info(f"Query to {tool_name} completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error querying {tool_name}: {e}")
            return None
    
    async def parallel_query(self, queries: List[Tuple[str, ToolQuery]]) -> List[ToolResponse]:
        """Execute multiple tool queries in parallel"""
        
        tasks = []
        for tool_name, query in queries:
            if tool_name in self.tools and self.tools[tool_name].is_available():
                task = asyncio.create_task(self.query_tool(tool_name, query))
                tasks.append(task)
        
        if not tasks:
            logger.warning("No available tools for parallel queries")
            return []
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None responses
        valid_responses = [r for r in responses if isinstance(r, ToolResponse)]
        
        logger.info(f"Completed {len(valid_responses)} parallel queries")
        return valid_responses
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return [name for name, tool in self.tools.items() if tool.is_available()]
    
    async def cleanup(self):
        """Cleanup all tool connections"""
        for tool in self.tools.values():
            await tool.cleanup()
        logger.info("Cleaned up all external tool connections")


# Convenience functions for quick tool access
async def create_tool_manager(config: Optional[Dict[str, Dict[str, Any]]] = None) -> ExternalToolManager:
    """Create and initialize external tool manager"""
    manager = ExternalToolManager(config)
    await manager.initialize_tools()
    return manager


def create_query(tool_name: str, query_type: str, data: Dict[str, Any], 
                priority: str = "normal", timeout: int = 300) -> ToolQuery:
    """Create a tool query with metadata"""
    return ToolQuery(
        tool_name=tool_name,
        query_type=query_type,
        data=data,
        priority=priority,
        timeout=timeout,
        metadata={"query_id": f"{tool_name}_{asyncio.get_event_loop().time()}"}
    ) 