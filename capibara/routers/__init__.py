"""
Routers Module for CapibaraGPT.

This package contains experimental routing mechanisms for expert selection and
token routing (e.g., SelfModifyingRouter). The primary router for the project
is `capibara.core.router.EnhancedRouter`.
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
