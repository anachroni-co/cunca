"""
Routers Module for CapibaraGPT.

This package contains routing mechanisms for expert selection and token routing:
- SelfModifyingRouter: Nested learning router that learns both what and how to route
- RoutingPolicy: Level 1 policy for immediate routing decisions
- MetaRoutingPolicy: Level 2 policy for optimizing the routing strategy
"""

from .self_modifying_router import (
    SelfModifyingRouter,
    SelfModifyingRouterConfig,
    RoutingPolicy,
    RoutingPolicyConfig,
    MetaRoutingPolicy,
    MetaRoutingPolicyConfig,
    create_self_modifying_router,
    get_global_self_modifying_router,
)

__all__ = [
    "SelfModifyingRouter",
    "SelfModifyingRouterConfig",
    "RoutingPolicy",
    "RoutingPolicyConfig",
    "MetaRoutingPolicy",
    "MetaRoutingPolicyConfig",
    "create_self_modifying_router",
    "get_global_self_modifying_router",
]
