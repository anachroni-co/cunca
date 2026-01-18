"""
Chain-of-Thought Module with Dynamic Knowledge Cores

This module implements an advanced Chain-of-Thought reasoning system
with knowledge core management and sub-models.
"""

# Export factories and main module
from .factory import (
    create_enhanced_cot_config,
    create_enhanced_cot_module,
    enhanced_chain_of_thought,
)
from .module import EnhancedChainOfThoughtModule
from .enhanced_cot_module import (
    EnhancedCoTModule,
    CapibaraEnhancedCoT,
    ReasoningConfig,
    ProcessRewardModel,
    MetaCognitionModule,
    SelfReflectionModule,
)


# Minimal ChainOfThought implementation for compatibility
class ChainOfThought:
    def __init__(self):
        self.steps = []

    def solve_problem(self, problem: str):
        self.steps = [
            {"step": 1, "action": "understand"},
            {"step": 2, "action": "reason"},
            {"step": 3, "action": "verify"},
        ]
        return {"problem": problem, "solution": "ok", "steps": self.steps, "confidence": 0.9}


def create_cot_handler() -> ChainOfThought:
    return ChainOfThought()


__all__ = [
    "EnhancedChainOfThoughtModule",
    "create_enhanced_cot_config",
    "create_enhanced_cot_module",
    "enhanced_chain_of_thought",
    "ChainOfThought",
    "create_cot_handler",
    # Real Enhanced CoT
    "EnhancedCoTModule",
    "CapibaraEnhancedCoT",
    "ReasoningConfig",
    "ProcessRewardModel",
    "MetaCognitionModule",
    "SelfReflectionModule",
]
