"""
Metacognitive Bayesian Network for Gospel Analysis Orchestration

This module implements the core decision-making engine that autonomously selects
computational tools and analysis strategies to maximize research objective functions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.stats import beta, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class AnalysisState(Enum):
    """Current state of genomic analysis"""
    VARIANT_LOADING = "variant_loading"
    ANNOTATION_REQUIRED = "annotation_required"
    UNCERTAINTY_HIGH = "uncertainty_high"
    EVIDENCE_CONFLICTING = "evidence_conflicting"
    COMPUTATION_LIMITED = "computation_limited"
    MOLECULAR_MODELING_NEEDED = "molecular_modeling_needed"
    PROBABILISTIC_REASONING_NEEDED = "probabilistic_reasoning_needed"


class ToolAction(Enum):
    """Available tool actions"""
    INTERNAL_PROCESSING = "internal_processing"
    QUERY_AUTOBAHN = "query_autobahn"
    QUERY_HEGEL = "query_hegel"
    QUERY_BORGIA = "query_borgia"
    QUERY_NEBUCHADNEZZAR = "query_nebuchadnezzar"
    QUERY_BENE_GESSERIT = "query_bene_gesserit"
    QUERY_LAVOISIER = "query_lavoisier"


@dataclass
class ObjectiveFunction:
    """Research objective function specification"""
    primary_goal: str
    weights: Dict[str, float]
    constraints: Dict[str, Any]
    time_budget: Optional[int] = None
    computational_budget: Optional[float] = None
    confidence_threshold: float = 0.9


@dataclass
class AnalysisContext:
    """Current analysis context and state"""
    variant_count: int
    expression_data_size: int
    network_complexity: float
    uncertainty_level: float
    computational_resources: float
    time_remaining: Optional[int]
    available_tools: List[str]


class MetacognitiveBayesianNetwork:
    """
    Bayesian network for metacognitive analysis orchestration.
    
    This system employs variational Bayes for approximate inference to select
    optimal tools and strategies for genomic analysis based on current state
    and research objectives.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 exploration_rate: float = 0.1,
                 max_iterations: int = 1000,
                 convergence_threshold: float = 1e-6):
        """
        Initialize the metacognitive Bayesian network.
        
        Args:
            learning_rate: Learning rate for variational updates
            exploration_rate: Exploration vs exploitation balance
            max_iterations: Maximum iterations for variational inference
            convergence_threshold: Convergence threshold for ELBO
        """
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Initialize network parameters
        self._initialize_network()
        
        # Gaussian Process for tool utility estimation
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp_utility = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        
        # Experience replay buffer
        self.experience_buffer = []
        self.max_buffer_size = 10000
        
        self.logger = logging.getLogger(__name__)
        
    def _initialize_network(self):
        """Initialize Bayesian network structure and parameters"""
        
        # Prior probabilities for states
        self.state_priors = {
            AnalysisState.VARIANT_LOADING: 0.2,
            AnalysisState.ANNOTATION_REQUIRED: 0.3,
            AnalysisState.UNCERTAINTY_HIGH: 0.15,
            AnalysisState.EVIDENCE_CONFLICTING: 0.1,
            AnalysisState.COMPUTATION_LIMITED: 0.1,
            AnalysisState.MOLECULAR_MODELING_NEEDED: 0.05,
            AnalysisState.PROBABILISTIC_REASONING_NEEDED: 0.1
        }
        
        # Tool effectiveness priors (Beta distributions)
        self.tool_effectiveness = {
            ToolAction.INTERNAL_PROCESSING: beta(a=8, b=2),  # High effectiveness
            ToolAction.QUERY_AUTOBAHN: beta(a=7, b=3),      # Good for probabilistic
            ToolAction.QUERY_HEGEL: beta(a=6, b=4),         # Good for evidence
            ToolAction.QUERY_BORGIA: beta(a=5, b=5),        # Moderate effectiveness
            ToolAction.QUERY_NEBUCHADNEZZAR: beta(a=6, b=4), # Good for circuits
            ToolAction.QUERY_BENE_GESSERIT: beta(a=4, b=6),  # Specialized use
            ToolAction.QUERY_LAVOISIER: beta(a=5, b=5)       # Moderate effectiveness
        }
        
        # State-action compatibility matrix
        self.compatibility_matrix = np.array([
            # States vs Actions compatibility scores
            [0.9, 0.1, 0.2, 0.3, 0.4, 0.2, 0.1],  # VARIANT_LOADING
            [0.8, 0.2, 0.7, 0.4, 0.3, 0.1, 0.2],  # ANNOTATION_REQUIRED
            [0.3, 0.9, 0.6, 0.2, 0.2, 0.3, 0.1],  # UNCERTAINTY_HIGH
            [0.2, 0.7, 0.9, 0.3, 0.2, 0.1, 0.2],  # EVIDENCE_CONFLICTING
            [0.5, 0.3, 0.2, 0.6, 0.8, 0.7, 0.4],  # COMPUTATION_LIMITED
            [0.1, 0.4, 0.2, 0.9, 0.8, 0.6, 0.7],  # MOLECULAR_MODELING_NEEDED
            [0.2, 0.9, 0.5, 0.6, 0.4, 0.7, 0.3]   # PROBABILISTIC_REASONING_NEEDED
        ])
        
    def infer_current_state(self, context: AnalysisContext) -> Dict[AnalysisState, float]:
        """
        Infer current analysis state probabilities using variational Bayes.
        
        Args:
            context: Current analysis context
            
        Returns:
            Dictionary mapping states to probabilities
        """
        # Feature extraction from context
        features = self._extract_state_features(context)
        
        # Variational inference for state estimation
        state_posteriors = {}
        
        for state in AnalysisState:
            # Compute likelihood based on context features
            likelihood = self._compute_state_likelihood(state, features)
            
            # Bayesian update: posterior ∝ likelihood × prior
            posterior = likelihood * self.state_priors[state]
            state_posteriors[state] = posterior
            
        # Normalize probabilities
        total_prob = sum(state_posteriors.values())
        if total_prob > 0:
            state_posteriors = {k: v/total_prob for k, v in state_posteriors.items()}
        
        return state_posteriors
    
    def select_optimal_actions(self, 
                             state_posteriors: Dict[AnalysisState, float],
                             objective: ObjectiveFunction,
                             context: AnalysisContext) -> List[Tuple[ToolAction, float]]:
        """
        Select optimal tool actions using Bayesian decision theory.
        
        Args:
            state_posteriors: Current state probability distribution
            objective: Research objective function
            context: Analysis context
            
        Returns:
            List of (action, utility) tuples sorted by utility
        """
        action_utilities = {}
        
        for action in ToolAction:
            # Skip unavailable tools
            if action.value not in context.available_tools and action != ToolAction.INTERNAL_PROCESSING:
                continue
                
            # Compute expected utility
            expected_utility = self._compute_expected_utility(
                action, state_posteriors, objective, context
            )
            
            action_utilities[action] = expected_utility
            
        # Sort by utility (descending)
        sorted_actions = sorted(action_utilities.items(), 
                              key=lambda x: x[1], reverse=True)
        
        # Apply exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Exploration: sample from utility distribution
            utilities = np.array([u for _, u in sorted_actions])
            probs = self._softmax(utilities / 0.1)  # Temperature = 0.1
            idx = np.random.choice(len(sorted_actions), p=probs)
            selected = [sorted_actions[idx]]
        else:
            # Exploitation: select top actions
            selected = sorted_actions[:3]  # Top 3 actions
            
        return selected
    
    def _extract_state_features(self, context: AnalysisContext) -> np.ndarray:
        """Extract numerical features from analysis context"""
        return np.array([
            np.log10(context.variant_count + 1),
            np.log10(context.expression_data_size + 1),
            context.network_complexity,
            context.uncertainty_level,
            context.computational_resources,
            context.time_remaining or 3600,  # Default 1 hour
            len(context.available_tools)
        ])
    
    def _compute_state_likelihood(self, state: AnalysisState, features: np.ndarray) -> float:
        """Compute likelihood of state given context features"""
        
        # State-specific likelihood functions
        if state == AnalysisState.VARIANT_LOADING:
            return norm.pdf(features[0], loc=4, scale=1)  # log10(10k variants)
            
        elif state == AnalysisState.ANNOTATION_REQUIRED:
            return 1.0 - np.exp(-features[0]/3)  # More variants = more annotation needed
            
        elif state == AnalysisState.UNCERTAINTY_HIGH:
            return features[3]  # Directly use uncertainty level
            
        elif state == AnalysisState.EVIDENCE_CONFLICTING:
            return features[3] * (1 - np.exp(-features[2]))  # Uncertainty × complexity
            
        elif state == AnalysisState.COMPUTATION_LIMITED:
            return 1.0 / (1.0 + features[4])  # Inverse of available resources
            
        elif state == AnalysisState.MOLECULAR_MODELING_NEEDED:
            return features[2] * 0.5  # Network complexity indicator
            
        elif state == AnalysisState.PROBABILISTIC_REASONING_NEEDED:
            return features[3] * features[2] * 0.3  # Uncertainty × complexity
            
        return 0.1  # Default low likelihood
    
    def _compute_expected_utility(self, 
                                action: ToolAction,
                                state_posteriors: Dict[AnalysisState, float],
                                objective: ObjectiveFunction,
                                context: AnalysisContext) -> float:
        """Compute expected utility of action given states and objective"""
        
        expected_utility = 0.0
        
        for state, prob in state_posteriors.items():
            # Get compatibility score
            state_idx = list(AnalysisState).index(state)
            action_idx = list(ToolAction).index(action)
            compatibility = self.compatibility_matrix[state_idx, action_idx]
            
            # Get tool effectiveness
            effectiveness = self.tool_effectiveness[action].mean()
            
            # Compute costs
            cost = self._compute_action_cost(action, context)
            
            # Objective-specific benefits
            benefit = self._compute_objective_benefit(action, objective, state)
            
            # Expected utility for this state
            utility = prob * (compatibility * effectiveness * benefit - cost)
            expected_utility += utility
            
        return expected_utility
    
    def _compute_action_cost(self, action: ToolAction, context: AnalysisContext) -> float:
        """Compute computational/time cost of action"""
        
        base_costs = {
            ToolAction.INTERNAL_PROCESSING: 0.1,
            ToolAction.QUERY_AUTOBAHN: 0.3,
            ToolAction.QUERY_HEGEL: 0.2,
            ToolAction.QUERY_BORGIA: 0.4,
            ToolAction.QUERY_NEBUCHADNEZZAR: 0.5,
            ToolAction.QUERY_BENE_GESSERIT: 0.6,
            ToolAction.QUERY_LAVOISIER: 0.3
        }
        
        base_cost = base_costs.get(action, 0.5)
        
        # Scale by data size
        size_factor = np.log10(context.variant_count + 1) / 5.0
        
        # Scale by resource availability
        resource_factor = 1.0 / (context.computational_resources + 0.1)
        
        return base_cost * size_factor * resource_factor
    
    def _compute_objective_benefit(self, 
                                 action: ToolAction, 
                                 objective: ObjectiveFunction,
                                 state: AnalysisState) -> float:
        """Compute benefit of action for specific objective"""
        
        # Objective-specific benefit matrices
        objective_benefits = {
            "identify_pathogenic_variants": {
                ToolAction.INTERNAL_PROCESSING: 0.8,
                ToolAction.QUERY_HEGEL: 0.9,  # Evidence validation
                ToolAction.QUERY_AUTOBAHN: 0.7,  # Probabilistic reasoning
            },
            "pathway_analysis": {
                ToolAction.QUERY_NEBUCHADNEZZAR: 0.9,  # Circuit modeling
                ToolAction.QUERY_BORGIA: 0.8,  # Molecular representation
                ToolAction.INTERNAL_PROCESSING: 0.6,
            },
            "drug_interaction_prediction": {
                ToolAction.QUERY_BORGIA: 0.9,  # Molecular modeling
                ToolAction.QUERY_LAVOISIER: 0.8,  # Mass spec analysis
                ToolAction.QUERY_HEGEL: 0.7,  # Evidence validation
            }
        }
        
        goal = objective.primary_goal
        if goal in objective_benefits:
            return objective_benefits[goal].get(action, 0.3)
        
        return 0.5  # Default moderate benefit
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def update_experience(self, 
                         context: AnalysisContext,
                         actions_taken: List[ToolAction],
                         outcomes: Dict[str, float]):
        """Update network based on action outcomes"""
        
        # Store experience
        experience = {
            'context': context,
            'actions': actions_taken,
            'outcomes': outcomes,
            'timestamp': pd.Timestamp.now()
        }
        
        self.experience_buffer.append(experience)
        
        # Maintain buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
            
        # Update tool effectiveness priors
        self._update_tool_effectiveness(actions_taken, outcomes)
        
        self.logger.info(f"Updated experience buffer. Size: {len(self.experience_buffer)}")
    
    def _update_tool_effectiveness(self, 
                                 actions: List[ToolAction], 
                                 outcomes: Dict[str, float]):
        """Update tool effectiveness distributions based on outcomes"""
        
        for action in actions:
            if action in self.tool_effectiveness:
                # Get outcome score (0-1)
                score = outcomes.get('success_rate', 0.5)
                
                # Update Beta distribution parameters
                current_dist = self.tool_effectiveness[action]
                
                if score > 0.5:
                    # Success: increase alpha
                    new_alpha = current_dist.args[0] + self.learning_rate
                    new_beta = current_dist.args[1]
                else:
                    # Failure: increase beta
                    new_alpha = current_dist.args[0]
                    new_beta = current_dist.args[1] + self.learning_rate
                    
                self.tool_effectiveness[action] = beta(a=new_alpha, b=new_beta)
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get current network state for debugging/monitoring"""
        return {
            'state_priors': self.state_priors,
            'tool_effectiveness': {
                k: {'mean': v.mean(), 'std': v.std()} 
                for k, v in self.tool_effectiveness.items()
            },
            'experience_count': len(self.experience_buffer),
            'exploration_rate': self.exploration_rate
        } 