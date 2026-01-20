"""
Tests for the Self-Modifying Router.

Tests the nested learning router that learns at two levels:
- Level 1: Routing decisions (fast, every step)
- Level 2: Routing policy optimization (slow, every N steps)
"""

import sys
import os
import importlib.util

import pytest

# Load module directly from file to avoid circular imports through package __init__.py
def _load_module_from_file(module_name: str, file_path: str):
    """Load a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get the project root
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_router_path = os.path.join(_project_root, 'core', 'self_modifying_router.py')

# Load the module directly
_smr = _load_module_from_file('self_modifying_router_direct', _router_path)

# Import classes from loaded module
RoutingPolicyConfig = _smr.RoutingPolicyConfig
MetaRoutingPolicyConfig = _smr.MetaRoutingPolicyConfig
RoutingPolicy = _smr.RoutingPolicy
MetaRoutingPolicy = _smr.MetaRoutingPolicy
SelfModifyingRouterConfig = _smr.SelfModifyingRouterConfig
SelfModifyingRouter = _smr.SelfModifyingRouter
create_self_modifying_router = _smr.create_self_modifying_router
get_global_self_modifying_router = _smr.get_global_self_modifying_router


class TestRoutingPolicyConfig:
    """Tests for RoutingPolicyConfig."""

    def test_create_config(self):
        """Test creating a routing policy configuration."""
        config = RoutingPolicyConfig(
            policy_id='test_policy',
            num_routes=16,
            top_k=2
        )
        assert config.policy_id == 'test_policy'
        assert config.num_routes == 16
        assert config.top_k == 2

    def test_default_values(self):
        """Test default values in config."""
        config = RoutingPolicyConfig(policy_id='default')
        assert config.num_routes == 32
        assert config.hidden_dim == 256
        assert config.temperature == 1.0
        assert config.top_k == 4
        assert config.exploration_rate == 0.1
        assert config.learning_rate == 1e-3
        assert config.update_freq == 1


class TestMetaRoutingPolicyConfig:
    """Tests for MetaRoutingPolicyConfig."""

    def test_create_config(self):
        """Test creating a meta-routing policy configuration."""
        config = MetaRoutingPolicyConfig(
            meta_learning_rate=1e-5,
            meta_update_freq=50
        )
        assert config.meta_learning_rate == 1e-5
        assert config.meta_update_freq == 50

    def test_default_values(self):
        """Test default values."""
        config = MetaRoutingPolicyConfig()
        assert config.meta_learning_rate == 1e-4
        assert config.meta_update_freq == 100
        assert config.poor_performance_threshold == 0.5
        assert config.excellent_performance_threshold == 0.9


class TestRoutingPolicy:
    """Tests for RoutingPolicy."""

    def test_create_policy(self):
        """Test creating a routing policy."""
        config = RoutingPolicyConfig(policy_id='test', num_routes=8)
        policy = RoutingPolicy(config)

        assert len(policy.route_weights) == 8
        assert len(policy.route_biases) == 8
        assert len(policy.recent_decisions) == 0

    def test_select_routes(self):
        """Test selecting routes."""
        config = RoutingPolicyConfig(policy_id='test', num_routes=8, top_k=2)
        policy = RoutingPolicy(config)

        routes, metadata = policy.select_routes(inputs="test_input")

        assert len(routes) == 2
        assert all(0 <= r < 8 for r in routes)
        assert 'probabilities' in metadata
        assert 'selected_routes' in metadata

    def test_select_routes_with_context(self):
        """Test selecting routes with context."""
        config = RoutingPolicyConfig(policy_id='test', num_routes=8, top_k=2)
        policy = RoutingPolicy(config)

        context = {'mode': 'test'}
        routes, metadata = policy.select_routes(inputs="test", context=context)

        assert len(routes) == 2

    def test_update_policy(self):
        """Test updating policy with feedback."""
        config = RoutingPolicyConfig(policy_id='test', num_routes=8)
        policy = RoutingPolicy(config)

        routes, _ = policy.select_routes(inputs="test")
        initial_weights = policy.route_weights.copy()

        # Update with positive feedback
        policy.update(routes, performance_feedback=0.9, step=1)

        # Weights should have changed
        assert policy.route_weights != initial_weights

    def test_update_respects_frequency(self):
        """Test that update respects frequency setting."""
        config = RoutingPolicyConfig(policy_id='test', num_routes=8, update_freq=10)
        policy = RoutingPolicy(config)

        routes, _ = policy.select_routes(inputs="test")
        initial_weights = policy.route_weights.copy()

        # Update at step 5 (should be skipped)
        policy.update(routes, performance_feedback=0.9, step=5)

        # Weights should NOT have changed
        assert policy.route_weights == initial_weights

    def test_route_usage_tracking(self):
        """Test that route usage is tracked."""
        config = RoutingPolicyConfig(policy_id='test', num_routes=8, top_k=2)
        policy = RoutingPolicy(config)

        # Make several routing decisions
        for _ in range(10):
            policy.select_routes(inputs="test")

        # Usage counts should be recorded
        total_usage = sum(policy.route_usage_count.values())
        assert total_usage == 20  # 10 decisions * 2 routes

    def test_performance_tracking(self):
        """Test performance tracking per route."""
        config = RoutingPolicyConfig(policy_id='test', num_routes=8)
        policy = RoutingPolicy(config)

        routes, _ = policy.select_routes(inputs="test")
        policy.update(routes, performance_feedback=0.8, step=1)

        # Performance should be recorded for selected routes
        for route_id in routes:
            assert route_id in policy.route_performance
            assert len(policy.route_performance[route_id]) > 0

    def test_get_statistics(self):
        """Test getting policy statistics."""
        config = RoutingPolicyConfig(policy_id='test_stats', num_routes=8)
        policy = RoutingPolicy(config)

        # Make some decisions
        for i in range(5):
            routes, _ = policy.select_routes(inputs=f"test_{i}")
            policy.update(routes, performance_feedback=0.6, step=i)

        stats = policy.get_statistics()

        assert stats['policy_id'] == 'test_stats'
        assert stats['num_routes'] == 8
        assert stats['total_decisions'] == 5
        assert 'route_usage' in stats
        assert 'avg_route_performance' in stats


class TestMetaRoutingPolicy:
    """Tests for MetaRoutingPolicy."""

    def test_create_meta_policy(self):
        """Test creating a meta-routing policy."""
        config = MetaRoutingPolicyConfig()
        meta_policy = MetaRoutingPolicy(config)

        assert meta_policy.current_strategy == 'balanced'
        assert len(meta_policy.modification_history) == 0

    def test_analyze_routing_trends(self):
        """Test analyzing routing trends."""
        config = MetaRoutingPolicyConfig()
        meta_policy = MetaRoutingPolicy(config)

        # Create a routing policy with some history
        routing_config = RoutingPolicyConfig(policy_id='test', num_routes=8)
        routing_policy = RoutingPolicy(routing_config)

        # Generate some routing history
        for i in range(20):
            routes, _ = routing_policy.select_routes(inputs=f"test_{i}")
            routing_policy.update(routes, performance_feedback=0.5 + i * 0.01, step=i)

        analysis = meta_policy.analyze_routing_trends(routing_policy, current_step=20)

        assert 'step' in analysis
        assert 'trends' in analysis
        assert 'issues' in analysis
        assert 'recommendations' in analysis

    def test_suggest_modifications(self):
        """Test suggesting modifications."""
        config = MetaRoutingPolicyConfig()
        meta_policy = MetaRoutingPolicy(config)

        routing_config = RoutingPolicyConfig(policy_id='test', num_routes=8)
        routing_policy = RoutingPolicy(routing_config)

        # Create an analysis with issues
        analysis = {
            'step': 100,
            'trends': {'performance_trend': -0.01},
            'issues': ['declining_performance'],
            'recommendations': ['increase_temperature']
        }

        modifications = meta_policy.suggest_modifications(analysis, routing_policy)

        assert 'temperature' in modifications
        assert 'top_k' in modifications
        assert 'exploration_rate' in modifications

    def test_apply_temperature_modification(self):
        """Test applying temperature modification."""
        config = MetaRoutingPolicyConfig()
        meta_policy = MetaRoutingPolicy(config)

        routing_config = RoutingPolicyConfig(policy_id='test', num_routes=8, temperature=1.0)
        routing_policy = RoutingPolicy(routing_config)

        modifications = {
            'temperature': 1.2,
            'top_k': None,
            'exploration_rate': None,
            'structural_changes': []
        }

        meta_policy.apply_modifications(modifications, routing_policy, step=100)

        assert routing_policy.config.temperature == 1.2
        assert len(meta_policy.modification_history) == 1

    def test_apply_exploration_modification(self):
        """Test applying exploration rate modification."""
        config = MetaRoutingPolicyConfig()
        meta_policy = MetaRoutingPolicy(config)

        routing_config = RoutingPolicyConfig(policy_id='test', exploration_rate=0.1)
        routing_policy = RoutingPolicy(routing_config)

        modifications = {
            'temperature': None,
            'top_k': None,
            'exploration_rate': 0.15,
            'structural_changes': []
        }

        meta_policy.apply_modifications(modifications, routing_policy, step=100)

        assert routing_policy.config.exploration_rate == 0.15

    def test_structural_changes(self):
        """Test applying structural changes."""
        config = MetaRoutingPolicyConfig()
        meta_policy = MetaRoutingPolicy(config)

        routing_config = RoutingPolicyConfig(policy_id='test', num_routes=8)
        routing_policy = RoutingPolicy(routing_config)

        # Add some poor performance
        routing_policy.route_performance[0] = [0.1, 0.2, 0.15]

        modifications = {
            'temperature': None,
            'top_k': None,
            'exploration_rate': None,
            'structural_changes': ['reset_poor_routes']
        }

        meta_policy.apply_modifications(modifications, routing_policy, step=100)

        # Poor route should be reset
        assert routing_policy.route_weights[0] == 1.0 / 8


class TestSelfModifyingRouterConfig:
    """Tests for SelfModifyingRouterConfig."""

    def test_create_config(self):
        """Test creating router configuration."""
        config = SelfModifyingRouterConfig()

        assert config.enable_self_modification is True
        assert config.track_detailed_stats is True
        assert config.routing_policy_config is not None
        assert config.meta_routing_policy_config is not None

    def test_custom_config(self):
        """Test custom configuration."""
        routing_config = RoutingPolicyConfig(policy_id='custom', num_routes=16)
        config = SelfModifyingRouterConfig(
            routing_policy_config=routing_config,
            enable_self_modification=False
        )

        assert config.routing_policy_config.num_routes == 16
        assert config.enable_self_modification is False


class TestSelfModifyingRouter:
    """Tests for SelfModifyingRouter."""

    def test_create_router(self):
        """Test creating a self-modifying router."""
        router = SelfModifyingRouter()

        assert router.current_step == 0
        assert router.total_routing_decisions == 0
        assert router.meta_updates_performed == 0

    def test_create_with_config(self):
        """Test creating router with custom config."""
        routing_config = RoutingPolicyConfig(policy_id='custom', num_routes=16)
        config = SelfModifyingRouterConfig(routing_policy_config=routing_config)
        router = SelfModifyingRouter(config)

        assert router.routing_policy.config.num_routes == 16

    def test_route(self):
        """Test basic routing."""
        router = SelfModifyingRouter()

        routes, metadata = router.route(inputs="test_input")

        assert len(routes) > 0
        assert router.total_routing_decisions == 1
        assert 'step' in metadata

    def test_route_with_context(self):
        """Test routing with context."""
        router = SelfModifyingRouter()

        context = {'task_type': 'classification'}
        routes, metadata = router.route(inputs="test", context=context)

        assert len(routes) > 0

    def test_update_with_feedback(self):
        """Test updating router with feedback."""
        router = SelfModifyingRouter()

        routes, _ = router.route(inputs="test")
        router.update_with_feedback(routes, performance_feedback=0.8, step=1)

        assert router.current_step == 1

    def test_meta_update_trigger(self):
        """Test that meta-updates are triggered at correct intervals."""
        config = SelfModifyingRouterConfig(
            meta_routing_policy_config=MetaRoutingPolicyConfig(meta_update_freq=10)
        )
        router = SelfModifyingRouter(config)

        # Simulate many steps
        for step in range(25):
            routes, _ = router.route(inputs=f"test_{step}")
            router.update_with_feedback(routes, performance_feedback=0.6, step=step)

        # Should have performed meta-updates at steps 0, 10, and 20
        assert router.meta_updates_performed == 3

    def test_self_modification_disabled(self):
        """Test router with self-modification disabled."""
        config = SelfModifyingRouterConfig(enable_self_modification=False)
        router = SelfModifyingRouter(config)

        # Simulate many steps
        for step in range(25):
            routes, _ = router.route(inputs=f"test_{step}")
            router.update_with_feedback(routes, performance_feedback=0.6, step=step)

        # Should NOT have performed any meta-updates
        assert router.meta_updates_performed == 0

    def test_get_statistics(self):
        """Test getting router statistics."""
        router = SelfModifyingRouter()

        # Make some routing decisions
        for step in range(5):
            routes, _ = router.route(inputs=f"test_{step}")
            router.update_with_feedback(routes, performance_feedback=0.7, step=step)

        stats = router.get_statistics()

        assert stats['current_step'] == 4
        assert stats['total_routing_decisions'] == 5
        assert 'level1_routing_policy' in stats
        assert 'level2_meta_policy' in stats

    def test_get_modification_history(self):
        """Test getting modification history."""
        config = SelfModifyingRouterConfig(
            meta_routing_policy_config=MetaRoutingPolicyConfig(meta_update_freq=5)
        )
        router = SelfModifyingRouter(config)

        # Simulate steps with varying performance
        for step in range(20):
            routes, _ = router.route(inputs=f"test_{step}")
            # Poor performance to trigger modifications
            router.update_with_feedback(routes, performance_feedback=0.3, step=step)

        history = router.get_modification_history()
        assert isinstance(history, list)


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_self_modifying_router(self):
        """Test factory function."""
        router = create_self_modifying_router()
        assert isinstance(router, SelfModifyingRouter)

    def test_create_with_config(self):
        """Test factory with custom config."""
        config = SelfModifyingRouterConfig(enable_self_modification=False)
        router = create_self_modifying_router(config)

        assert router.config.enable_self_modification is False
