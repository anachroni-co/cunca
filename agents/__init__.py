"""
Agents Module - CapibaraGPT v3

Multi-agent system with coordination and orchestration:
- capibara_agent: Core agent implementation
- capibara_agent_factory: Agent factory pattern
- ultra_agent_orchestrator: Multi-agent orchestration
- behaviors: Agent behavior implementations
- orchestration_strategies: Strategy implementations
"""

import logging

logger = logging.getLogger(__name__)

# Core agent
try:
    from .capibara_agent import CapibaraAgent, CapibaraTool
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    CapibaraAgent = None
    CapibaraTool = None

# Agent factory
try:
    from .capibara_agent_factory import AgentFactory, create_agent
    FACTORY_AVAILABLE = True
except ImportError:
    FACTORY_AVAILABLE = False
    AgentFactory = None
    create_agent = None

# Ultra orchestrator
try:
    from .ultra_agent_orchestrator import (
        UltraAgentOrchestrator,
        create_ultra_agent_system,
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    UltraAgentOrchestrator = None
    create_ultra_agent_system = None

# Behaviors
try:
    from .behaviors import AgentBehavior
    from .advanced_behaviors import AdvancedBehavior
    BEHAVIORS_AVAILABLE = True
except ImportError:
    BEHAVIORS_AVAILABLE = False
    AgentBehavior = None
    AdvancedBehavior = None

# Orchestration strategies
try:
    from .orchestration_strategies import OrchestrationStrategy
    STRATEGIES_AVAILABLE = True
except ImportError:
    STRATEGIES_AVAILABLE = False
    OrchestrationStrategy = None


__all__ = [
    # Core
    "CapibaraAgent",
    "CapibaraTool",
    # Factory
    "AgentFactory",
    "create_agent",
    # Orchestrator
    "UltraAgentOrchestrator",
    "create_ultra_agent_system",
    # Behaviors
    "AgentBehavior",
    "AdvancedBehavior",
    # Strategies
    "OrchestrationStrategy",
    # Flags
    "CORE_AVAILABLE",
    "FACTORY_AVAILABLE",
    "ORCHESTRATOR_AVAILABLE",
    "BEHAVIORS_AVAILABLE",
    "STRATEGIES_AVAILABLE",
]
