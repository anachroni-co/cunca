"""
Self-Modifying Router for CapibaraGPT.

This module implements a self-modifying router inspired by Nested Learning
(Behrouz et al., NeurIPS 2025). The router learns not only routing decisions
but also learns to modify its own routing logic based on observed performance.

Key Features:
    - Two-level nested optimization (routing policy + meta-routing policy)
    - Different update frequencies (every step vs every N steps)
    - Self-modification of routing parameters and strategies
    - Performance trend analysis
    - Automatic adaptation to changing conditions

Architecture:
    Level 1 - Routing Policy (updates every step):
        - Makes routing decisions for input tokens
        - Learns from immediate feedback
        - Maintains route performance statistics

    Level 2 - Meta-Routing Policy (updates every 100 steps):
        - Analyzes routing trends over time
        - Modifies Level 1 policy structure and parameters
        - Optimizes temperature, top-k, exploration rate
        - Triggers structural changes when needed

This implements the key Nested Learning principle of learning at multiple
time scales, where the router learns both WHAT to route AND HOW to route.

Reference:
    Behrouz, A.; Razaviyayn, M.; Mirrokni, V.; Zhong, P. (2025).
    "Nested Learning: The Illusion of Deep Learning Architectures"
    NeurIPS 2025
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import random

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# Numpy fallback for environments without numpy
class NumpyFallback:
    """Minimal numpy-like operations for systems without numpy."""

    @staticmethod
    def mean(x):
        return sum(x) / len(x) if x else 0

    @staticmethod
    def std(x):
        if not x:
            return 0
        mean = sum(x) / len(x)
        variance = sum((val - mean) ** 2 for val in x) / len(x)
        return variance ** 0.5

    @staticmethod
    def polyfit(x, y, deg):
        """Simple linear regression for deg=1."""
        if deg == 1 and len(x) == len(y) and len(x) >= 2:
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_xx = sum(x[i] ** 2 for i in range(n))
            denom = n * sum_xx - sum_x ** 2
            if denom == 0:
                return [0, 0]
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n
            return [slope, intercept]
        return [0, 0]

    @staticmethod
    def clip(x, min_val, max_val):
        return max(min_val, min(max_val, x))


# Use numpy if available, otherwise use fallback
if not NUMPY_AVAILABLE:
    np = NumpyFallback()


@dataclass
class RoutingPolicyConfig:
    """Configuration for a routing policy.

    Attributes:
        policy_id: Unique identifier for this policy.
        num_routes: Number of possible routes (e.g., number of experts).
        hidden_dim: Internal representation size for routing decisions.
        temperature: Softmax temperature for route selection.
        top_k: Number of routes to select per input.
        exploration_rate: Probability of random exploration.
        learning_rate: Learning rate for weight updates.
        update_freq: Frequency of policy updates (every N steps).
        track_performance: Whether to track per-route performance.
        performance_window: Number of recent samples for performance tracking.
    """
    policy_id: str = "default"
    num_routes: int = 32
    hidden_dim: int = 256
    temperature: float = 1.0
    top_k: int = 4
    exploration_rate: float = 0.1
    learning_rate: float = 1e-3
    update_freq: int = 1
    track_performance: bool = True
    performance_window: int = 100


@dataclass
class MetaRoutingPolicyConfig:
    """Configuration for meta-routing policy.

    Attributes:
        meta_learning_rate: Learning rate for meta-level updates.
        meta_update_freq: Frequency of meta updates (every N steps).
        trend_analysis_window: Window size for trend analysis.
        performance_comparison_window: Window for performance comparison.
        poor_performance_threshold: Threshold below which triggers changes.
        excellent_performance_threshold: Threshold above which stabilizes.
        enable_temperature_adaptation: Allow temperature modification.
        enable_top_k_adaptation: Allow top-k modification.
        enable_exploration_adaptation: Allow exploration rate modification.
        enable_structural_changes: Allow structural policy changes.
        temperature_range: Valid range for temperature.
        top_k_range: Valid range for top-k.
        exploration_range: Valid range for exploration rate.
    """
    meta_learning_rate: float = 1e-4
    meta_update_freq: int = 100
    trend_analysis_window: int = 100
    performance_comparison_window: int = 50
    poor_performance_threshold: float = 0.5
    excellent_performance_threshold: float = 0.9
    enable_temperature_adaptation: bool = True
    enable_top_k_adaptation: bool = True
    enable_exploration_adaptation: bool = True
    enable_structural_changes: bool = True
    temperature_range: Tuple[float, float] = (0.1, 2.0)
    top_k_range: Tuple[int, int] = (1, 8)
    exploration_range: Tuple[float, float] = (0.0, 0.3)


class RoutingPolicy:
    """
    Level 1: Routing policy that makes routing decisions.

    This component learns to route inputs based on content and feedback.
    It maintains performance statistics for each route and adjusts
    routing weights based on observed outcomes.

    Attributes:
        config: RoutingPolicyConfig instance.
        route_weights: Current routing weights per route.
        route_biases: Bias terms per route.
        route_performance: Performance history per route.
        route_usage_count: Usage count per route.
        recent_decisions: Recent routing decisions for analysis.
    """

    def __init__(self, config: RoutingPolicyConfig):
        """
        Initialize routing policy.

        Args:
            config: Configuration for the routing policy.
        """
        self.config = config

        # Initialize routing weights uniformly
        self.route_weights = [1.0 / config.num_routes] * config.num_routes
        self.route_biases = [0.0] * config.num_routes

        # Performance tracking
        self.route_performance = defaultdict(list)
        self.route_usage_count = defaultdict(int)
        self.recent_decisions = deque(maxlen=config.performance_window)

        logger.debug(
            f"RoutingPolicy '{config.policy_id}' initialized with "
            f"{config.num_routes} routes"
        )

    def select_routes(
        self,
        inputs: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Select routes for the given input.

        Args:
            inputs: Input data to route.
            context: Optional context information.

        Returns:
            Tuple of (selected_route_ids, routing_metadata).
        """
        # Calculate routing scores
        scores = self._calculate_routing_scores(inputs, context)

        # Apply temperature scaling
        scaled_scores = [s / self.config.temperature for s in scores]

        # Softmax computation
        max_score = max(scaled_scores)
        exp_scores = [2.71828 ** (s - max_score) for s in scaled_scores]
        sum_exp = sum(exp_scores)
        probabilities = [e / sum_exp for e in exp_scores]

        # Select top-k routes with exploration
        is_exploring = random.SystemRandom().random() < self.config.exploration_rate

        if is_exploring:
            # Exploration: random selection
            selected_routes = random.sample(
                range(self.config.num_routes),
                min(self.config.top_k, self.config.num_routes)
            )
        else:
            # Exploitation: top-k selection
            route_prob_pairs = list(enumerate(probabilities))
            route_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            selected_routes = [idx for idx, _ in route_prob_pairs[:self.config.top_k]]

        # Update usage counts
        for route_id in selected_routes:
            self.route_usage_count[route_id] += 1

        # Build metadata
        metadata = {
            'probabilities': probabilities,
            'selected_routes': selected_routes,
            'exploration': is_exploring,
            'temperature': self.config.temperature,
            'top_k': self.config.top_k,
        }

        # Record decision
        self.recent_decisions.append({
            'routes': selected_routes,
            'scores': scores,
            'timestamp': datetime.now().isoformat()
        })

        return selected_routes, metadata

    def _calculate_routing_scores(
        self,
        inputs: Any,
        context: Optional[Dict[str, Any]]
    ) -> List[float]:
        """Calculate routing scores for each route."""
        scores = []
        for i in range(self.config.num_routes):
            # Base score from weights and biases
            score = self.route_weights[i] + self.route_biases[i]

            # Adjust based on recent performance
            if i in self.route_performance and self.route_performance[i]:
                recent_perf = self.route_performance[i][-10:]
                avg_perf = sum(recent_perf) / len(recent_perf)
                score += avg_perf * 0.5  # Performance bonus

            scores.append(score)

        return scores

    def update(
        self,
        selected_routes: List[int],
        performance_feedback: float,
        step: int
    ):
        """
        Update routing policy based on feedback.

        Args:
            selected_routes: Routes that were selected.
            performance_feedback: Performance score in [0, 1].
            step: Current training step.
        """
        if step % self.config.update_freq != 0:
            return

        # Update performance history
        for route_id in selected_routes:
            self.route_performance[route_id].append(performance_feedback)

            # Keep history bounded
            if len(self.route_performance[route_id]) > self.config.performance_window:
                self.route_performance[route_id].pop(0)

        # Update weights with simple gradient-like update
        lr = self.config.learning_rate
        for route_id in selected_routes:
            # Positive adjustment for good performance, negative for poor
            adjustment = lr * (performance_feedback - 0.5)
            self.route_weights[route_id] += adjustment
            self.route_weights[route_id] = max(0.01, self.route_weights[route_id])

        # Normalize weights to sum to 1
        total_weight = sum(self.route_weights)
        if total_weight > 0:
            self.route_weights = [w / total_weight for w in self.route_weights]

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing policy statistics."""
        avg_performance = {}
        for route_id, perfs in self.route_performance.items():
            if perfs:
                if NUMPY_AVAILABLE:
                    avg_performance[route_id] = float(np.mean(perfs))
                else:
                    avg_performance[route_id] = sum(perfs) / len(perfs)

        return {
            'policy_id': self.config.policy_id,
            'num_routes': self.config.num_routes,
            'temperature': self.config.temperature,
            'top_k': self.config.top_k,
            'exploration_rate': self.config.exploration_rate,
            'total_decisions': len(self.recent_decisions),
            'route_usage': dict(self.route_usage_count),
            'avg_route_performance': avg_performance,
        }


class MetaRoutingPolicy:
    """
    Level 2: Meta-routing policy that modifies the routing policy.

    This component learns HOW to optimize the routing policy by analyzing
    long-term trends and making structural changes to the Level 1 policy.

    It operates at a slower timescale (every 100 steps by default) and
    can modify temperature, top-k, exploration rate, and even trigger
    structural resets of poorly performing routes.

    Attributes:
        config: MetaRoutingPolicyConfig instance.
        meta_performance_history: History of meta-level performance.
        modification_history: History of all modifications made.
        current_strategy: Current meta-strategy being employed.
        last_meta_update_step: Step of last meta update.
    """

    def __init__(self, config: MetaRoutingPolicyConfig):
        """
        Initialize meta-routing policy.

        Args:
            config: Configuration for the meta-routing policy.
        """
        self.config = config
        self.meta_performance_history = []
        self.modification_history = []
        self.current_strategy = 'balanced'
        self.last_meta_update_step = 0

        logger.info("MetaRoutingPolicy initialized")

    def analyze_routing_trends(
        self,
        routing_policy: RoutingPolicy,
        current_step: int
    ) -> Dict[str, Any]:
        """
        Analyze routing performance trends.

        Args:
            routing_policy: The routing policy to analyze.
            current_step: Current training step.

        Returns:
            Analysis results with trend information.
        """
        stats = routing_policy.get_statistics()

        analysis = {
            'step': current_step,
            'timestamp': datetime.now().isoformat(),
            'trends': {},
            'issues': [],
            'recommendations': []
        }

        # Analyze route usage distribution
        usage_counts = list(stats['route_usage'].values())
        if usage_counts:
            if NUMPY_AVAILABLE:
                usage_variance = float(np.std(usage_counts)) if len(usage_counts) > 1 else 0
                usage_mean = float(np.mean(usage_counts))
            else:
                usage_mean = sum(usage_counts) / len(usage_counts)
                usage_variance = NumpyFallback.std(usage_counts)

            analysis['trends']['usage_variance'] = usage_variance
            analysis['trends']['usage_mean'] = usage_mean

            # Check for imbalanced usage
            if usage_variance > usage_mean * 0.5:
                analysis['issues'].append('high_usage_imbalance')
                analysis['recommendations'].append('increase_exploration')

        # Analyze performance trends
        all_performances = []
        for route_id, perfs in routing_policy.route_performance.items():
            if perfs:
                all_performances.extend(perfs[-self.config.trend_analysis_window:])

        if len(all_performances) >= 10:
            x = list(range(len(all_performances)))
            y = all_performances

            try:
                if NUMPY_AVAILABLE:
                    trend_coef = float(np.polyfit(x, y, 1)[0])
                else:
                    trend_coef = NumpyFallback.polyfit(x, y, 1)[0]

                analysis['trends']['performance_trend'] = trend_coef

                if trend_coef < -0.001:
                    analysis['issues'].append('declining_performance')
                    analysis['recommendations'].append('increase_temperature')
                elif trend_coef > 0.001:
                    analysis['issues'].append('improving_performance')
                    analysis['recommendations'].append('continue_current_strategy')
            except Exception:
                pass

            # Overall performance level
            if NUMPY_AVAILABLE:
                avg_performance = float(np.mean(all_performances))
            else:
                avg_performance = sum(all_performances) / len(all_performances)

            analysis['trends']['avg_performance'] = avg_performance

            if avg_performance < self.config.poor_performance_threshold:
                analysis['issues'].append('poor_overall_performance')
                analysis['recommendations'].append('major_restructure')
            elif avg_performance > self.config.excellent_performance_threshold:
                analysis['issues'].append('excellent_performance')
                analysis['recommendations'].append('reduce_exploration')

        return analysis

    def suggest_modifications(
        self,
        analysis: Dict[str, Any],
        routing_policy: RoutingPolicy
    ) -> Dict[str, Any]:
        """
        Suggest modifications based on analysis.

        Args:
            analysis: Analysis results from analyze_routing_trends.
            routing_policy: Current routing policy.

        Returns:
            Dictionary of suggested modifications.
        """
        modifications = {
            'temperature': None,
            'top_k': None,
            'exploration_rate': None,
            'structural_changes': [],
        }

        recommendations = analysis.get('recommendations', [])

        # Temperature adaptation
        if self.config.enable_temperature_adaptation:
            if 'increase_temperature' in recommendations:
                new_temp = min(
                    routing_policy.config.temperature * 1.2,
                    self.config.temperature_range[1]
                )
                modifications['temperature'] = new_temp
            elif 'decrease_temperature' in recommendations:
                new_temp = max(
                    routing_policy.config.temperature * 0.8,
                    self.config.temperature_range[0]
                )
                modifications['temperature'] = new_temp

        # Top-k adaptation
        if self.config.enable_top_k_adaptation:
            if 'increase_diversity' in recommendations:
                new_k = min(
                    routing_policy.config.top_k + 1,
                    self.config.top_k_range[1]
                )
                modifications['top_k'] = new_k
            elif 'decrease_diversity' in recommendations:
                new_k = max(
                    routing_policy.config.top_k - 1,
                    self.config.top_k_range[0]
                )
                modifications['top_k'] = new_k

        # Exploration rate adaptation
        if self.config.enable_exploration_adaptation:
            if 'increase_exploration' in recommendations:
                new_rate = min(
                    routing_policy.config.exploration_rate + 0.05,
                    self.config.exploration_range[1]
                )
                modifications['exploration_rate'] = new_rate
            elif 'reduce_exploration' in recommendations:
                new_rate = max(
                    routing_policy.config.exploration_rate - 0.05,
                    self.config.exploration_range[0]
                )
                modifications['exploration_rate'] = new_rate

        # Structural changes
        if self.config.enable_structural_changes:
            if 'major_restructure' in recommendations:
                modifications['structural_changes'].append('reset_poor_routes')

            if 'high_usage_imbalance' in analysis.get('issues', []):
                modifications['structural_changes'].append('rebalance_weights')

        return modifications

    def apply_modifications(
        self,
        modifications: Dict[str, Any],
        routing_policy: RoutingPolicy,
        step: int
    ):
        """
        Apply suggested modifications to the routing policy.

        Args:
            modifications: Modifications to apply.
            routing_policy: Routing policy to modify.
            step: Current step.
        """
        applied = []

        # Apply temperature change
        if modifications['temperature'] is not None:
            old_temp = routing_policy.config.temperature
            routing_policy.config.temperature = modifications['temperature']
            applied.append(f"temperature: {old_temp:.3f} -> {modifications['temperature']:.3f}")

        # Apply top-k change
        if modifications['top_k'] is not None:
            old_k = routing_policy.config.top_k
            routing_policy.config.top_k = modifications['top_k']
            applied.append(f"top_k: {old_k} -> {modifications['top_k']}")

        # Apply exploration rate change
        if modifications['exploration_rate'] is not None:
            old_rate = routing_policy.config.exploration_rate
            routing_policy.config.exploration_rate = modifications['exploration_rate']
            applied.append(f"exploration: {old_rate:.3f} -> {modifications['exploration_rate']:.3f}")

        # Apply structural changes
        for change in modifications['structural_changes']:
            if change == 'reset_poor_routes':
                for route_id, perfs in routing_policy.route_performance.items():
                    if NUMPY_AVAILABLE:
                        avg_perf = np.mean(perfs) if perfs else 0
                    else:
                        avg_perf = sum(perfs) / len(perfs) if perfs else 0

                    if avg_perf < 0.3:
                        routing_policy.route_weights[route_id] = 1.0 / routing_policy.config.num_routes
                        routing_policy.route_biases[route_id] = 0.0
                applied.append("reset_poor_routes")

            elif change == 'rebalance_weights':
                total = sum(routing_policy.route_weights)
                if total > 0:
                    routing_policy.route_weights = [
                        w / total for w in routing_policy.route_weights
                    ]
                applied.append("rebalance_weights")

        # Record modification
        if applied:
            self.modification_history.append({
                'step': step,
                'modifications': applied,
                'timestamp': datetime.now().isoformat()
            })
            logger.info(f"Step {step}: Applied meta-modifications: {', '.join(applied)}")


@dataclass
class SelfModifyingRouterConfig:
    """Configuration for self-modifying router.

    Attributes:
        routing_policy_config: Configuration for Level 1 routing policy.
        meta_routing_policy_config: Configuration for Level 2 meta policy.
        enable_self_modification: Whether to enable meta-level modifications.
        track_detailed_stats: Whether to track detailed statistics.
    """
    routing_policy_config: RoutingPolicyConfig = field(
        default_factory=RoutingPolicyConfig
    )
    meta_routing_policy_config: MetaRoutingPolicyConfig = field(
        default_factory=MetaRoutingPolicyConfig
    )
    enable_self_modification: bool = True
    track_detailed_stats: bool = True


class SelfModifyingRouter:
    """
    Self-Modifying Router implementing Nested Learning.

    This router learns at two levels:
    - Level 1: Routing decisions (fast, every step)
    - Level 2: Routing policy optimization (slow, every N steps)

    The meta-routing policy learns to modify the routing policy's structure
    and parameters based on observed performance trends.

    Example:
        >>> config = SelfModifyingRouterConfig()
        >>> router = SelfModifyingRouter(config)
        >>> routes, metadata = router.route(input_data)
        >>> router.update_with_feedback(routes, performance=0.8, step=100)

    Attributes:
        config: SelfModifyingRouterConfig instance.
        routing_policy: Level 1 routing policy.
        meta_routing_policy: Level 2 meta-routing policy.
        current_step: Current training step.
        total_routing_decisions: Total number of routing decisions made.
        meta_updates_performed: Number of meta-level updates performed.
    """

    def __init__(self, config: Optional[SelfModifyingRouterConfig] = None):
        """
        Initialize self-modifying router.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or SelfModifyingRouterConfig()

        # Level 1: Routing policy
        self.routing_policy = RoutingPolicy(self.config.routing_policy_config)

        # Level 2: Meta-routing policy
        self.meta_routing_policy = MetaRoutingPolicy(
            self.config.meta_routing_policy_config
        )

        # State tracking
        self.current_step = 0
        self.total_routing_decisions = 0
        self.meta_updates_performed = 0

        logger.info("Self-Modifying Router initialized")
        logger.info(
            f"  Level 1 (Routing): Updates every "
            f"{self.config.routing_policy_config.update_freq} steps"
        )
        logger.info(
            f"  Level 2 (Meta-Routing): Updates every "
            f"{self.config.meta_routing_policy_config.meta_update_freq} steps"
        )

    def route(
        self,
        inputs: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Perform routing decision.

        Args:
            inputs: Input data to route.
            context: Optional routing context.

        Returns:
            Tuple of (selected_routes, routing_info).
        """
        self.total_routing_decisions += 1

        # Level 1: Make routing decision
        selected_routes, metadata = self.routing_policy.select_routes(inputs, context)

        # Add meta-information
        metadata['step'] = self.current_step
        metadata['meta_updates_count'] = self.meta_updates_performed

        return selected_routes, metadata

    def update_with_feedback(
        self,
        selected_routes: List[int],
        performance_feedback: float,
        step: int
    ):
        """
        Update router with performance feedback.

        Args:
            selected_routes: Routes that were selected.
            performance_feedback: Performance score in [0, 1].
            step: Current training step.
        """
        self.current_step = step

        # Level 1: Update routing policy
        self.routing_policy.update(selected_routes, performance_feedback, step)

        # Level 2: Meta-update if it's time
        if (self.config.enable_self_modification and
            step % self.config.meta_routing_policy_config.meta_update_freq == 0):
            self._perform_meta_update(step)

    def _perform_meta_update(self, step: int):
        """Perform meta-level update (Level 2)."""
        # Analyze routing trends
        analysis = self.meta_routing_policy.analyze_routing_trends(
            self.routing_policy,
            step
        )

        # Get modification suggestions
        modifications = self.meta_routing_policy.suggest_modifications(
            analysis,
            self.routing_policy
        )

        # Apply modifications
        self.meta_routing_policy.apply_modifications(
            modifications,
            self.routing_policy,
            step
        )

        self.meta_updates_performed += 1
        logger.debug(f"Meta-update {self.meta_updates_performed} completed at step {step}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive router statistics."""
        routing_stats = self.routing_policy.get_statistics()

        return {
            'current_step': self.current_step,
            'total_routing_decisions': self.total_routing_decisions,
            'meta_updates_performed': self.meta_updates_performed,
            'level1_routing_policy': routing_stats,
            'level2_meta_policy': {
                'modification_count': len(self.meta_routing_policy.modification_history),
                'recent_modifications': self.meta_routing_policy.modification_history[-5:],
                'current_strategy': self.meta_routing_policy.current_strategy,
            },
            'self_modification_enabled': self.config.enable_self_modification,
        }

    def get_modification_history(self) -> List[Dict[str, Any]]:
        """Get history of meta-level modifications."""
        return self.meta_routing_policy.modification_history.copy()


def create_self_modifying_router(
    config: Optional[SelfModifyingRouterConfig] = None
) -> SelfModifyingRouter:
    """
    Factory function to create a self-modifying router.

    Args:
        config: Optional configuration.

    Returns:
        SelfModifyingRouter instance.
    """
    return SelfModifyingRouter(config)


# Global instance management
_global_router: Optional[SelfModifyingRouter] = None


def get_global_self_modifying_router() -> SelfModifyingRouter:
    """Get the global self-modifying router instance."""
    global _global_router
    if _global_router is None:
        _global_router = create_self_modifying_router()
    return _global_router


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Self-Modifying Router - Testing Mode")

    router = create_self_modifying_router()

    # Simulate routing with varying performance
    for step in range(500):
        inputs = f"input_{step}"
        selected_routes, metadata = router.route(inputs)

        # Simulate improving performance with noise
        base_performance = 0.3 + step * 0.001
        noise = (hash(str(step)) % 20 - 10) / 100
        performance = max(0, min(1, base_performance + noise))

        router.update_with_feedback(selected_routes, performance, step)

        if step % 100 == 0:
            stats = router.get_statistics()
            logger.info(f"\nStep {step}:")
            logger.info(f"  Routing decisions: {stats['total_routing_decisions']}")
            logger.info(f"  Meta-updates: {stats['meta_updates_performed']}")
            logger.info(f"  Temperature: {stats['level1_routing_policy']['temperature']:.3f}")

    final_stats = router.get_statistics()
    logger.info(f"\nFinal Statistics:")
    logger.info(f"  Total routing decisions: {final_stats['total_routing_decisions']}")
    logger.info(f"  Total meta-updates: {final_stats['meta_updates_performed']}")
