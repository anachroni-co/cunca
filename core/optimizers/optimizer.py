"""Optimizer Module for CapibaraGPT Training.

This module provides optimizer configuration and base optimizer implementations
for training CapibaraGPT models. It supports various optimization algorithms
including Adam, SGD with momentum, and includes gradient clipping functionality.

The optimizer system features:
- Configurable learning rate and hyperparameters
- Multiple optimizer types (Adam, SGD, AdamW)
- Gradient clipping support
- Weight decay (L2 regularization)
- Extensible base class for custom optimizers

Key Components:
    - OptimizerConfig: Configuration dataclass for optimizer hyperparameters
    - BaseOptimizer: Base optimizer implementation
    - create_optimizer: Factory function for optimizer creation

Example:
    Basic optimizer creation:

    >>> from capibara.core.optimizers.optimizer import create_optimizer, OptimizerConfig
    >>>
    >>> # Create with default Adam configuration
    >>> optimizer = create_optimizer()
    >>>
    >>> # Custom configuration
    >>> config = OptimizerConfig(
    ...     learning_rate=0.0001,
    ...     optimizer_type="adam",
    ...     weight_decay=0.01,
    ...     clip_grad_norm=1.0
    ... )
    >>> optimizer = create_optimizer(config)

    Using the optimizer:

    >>> # Perform optimization step
    >>> updated_params = optimizer.step(gradients)
    >>>
    >>> # Zero gradients before next step
    >>> optimizer.zero_grad()

    Configuration dictionary:

    >>> config = OptimizerConfig(learning_rate=0.001)
    >>> config_dict = config.to_dict()
    >>> print(config_dict["learning_rate"])  # 0.001

Note:
    This is a base implementation designed for extensibility. For production
    training, consider integrating with Optax or other JAX optimization libraries.

See Also:
    - capibara.training: Training loop implementations
    - capibara.core.optimization: Advanced optimization utilities
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class OptimizerConfig:
    """Configuration for optimizer hyperparameters.

    This dataclass encapsulates all hyperparameters needed to configure an
    optimizer for training. It supports Adam, SGD, and AdamW optimizers with
    configurable learning rate, momentum, weight decay, and gradient clipping.

    Attributes:
        learning_rate (float): Learning rate for parameter updates. Defaults to 0.001.
        optimizer_type (str): Type of optimizer ("adam", "sgd", "adamw").
            Defaults to "adam".
        momentum (float): Momentum factor for SGD and related optimizers.
            Defaults to 0.9.
        weight_decay (float): L2 regularization coefficient. Defaults to 0.0 (no decay).
        eps (float): Small constant for numerical stability in Adam. Defaults to 1e-8.
        beta1 (float): Exponential decay rate for first moment estimates (Adam).
            Defaults to 0.9.
        beta2 (float): Exponential decay rate for second moment estimates (Adam).
            Defaults to 0.999.
        clip_grad_norm (Optional[float]): Maximum gradient norm for clipping.
            None means no clipping. Defaults to None.

    Example:
        >>> # Default Adam configuration
        >>> config = OptimizerConfig()
        >>> print(config.optimizer_type)  # "adam"
        >>> print(config.learning_rate)  # 0.001
        >>>
        >>> # SGD with momentum
        >>> sgd_config = OptimizerConfig(
        ...     optimizer_type="sgd",
        ...     learning_rate=0.01,
        ...     momentum=0.95
        ... )
        >>>
        >>> # Adam with weight decay and gradient clipping
        >>> adamw_config = OptimizerConfig(
        ...     optimizer_type="adamw",
        ...     learning_rate=0.0001,
        ...     weight_decay=0.01,
        ...     clip_grad_norm=1.0
        ... )

    Note:
        The optimizer_type field determines which hyperparameters are actually
        used. For example, momentum is only used with SGD-based optimizers,
        while beta1 and beta2 are specific to Adam variants.
    """
    learning_rate: float = 0.001
    optimizer_type: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    eps: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999
    clip_grad_norm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary containing all configuration parameters.

        Example:
            >>> config = OptimizerConfig(learning_rate=0.001)
            >>> config_dict = config.to_dict()
            >>> print(config_dict)
            {
                'learning_rate': 0.001,
                'optimizer_type': 'adam',
                'momentum': 0.9,
                'weight_decay': 0.0,
                'eps': 1e-08,
                'beta1': 0.9,
                'beta2': 0.999,
                'clip_grad_norm': None
            }

        Note:
            Useful for serialization, logging, or passing to other configuration
            systems that expect dictionary inputs.
        """
        return {
            "learning_rate": self.learning_rate,
            "optimizer_type": self.optimizer_type,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "eps": self.eps,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "clip_grad_norm": self.clip_grad_norm
        }

class BaseOptimizer:
    """Base optimizer class for training CapibaraGPT models.

    This class provides a foundation for optimizer implementations, handling
    initialization, step updates, and gradient zeroing. It can be extended to
    implement specific optimization algorithms.

    Attributes:
        config (OptimizerConfig): Optimizer configuration with hyperparameters.
        initialized (bool): Whether the optimizer has been initialized.

    Example:
        >>> config = OptimizerConfig(learning_rate=0.001)
        >>> optimizer = BaseOptimizer(config)
        >>>
        >>> # Initialize optimizer
        >>> optimizer.initialize()
        >>>
        >>> # Perform optimization step
        >>> new_grads = optimizer.step(gradients)
        >>>
        >>> # Zero gradients
        >>> optimizer.zero_grad()

    Note:
        This is a base implementation. For production use, consider extending
        this class with actual optimization logic (gradient descent, Adam updates,
        etc.) or integrating with Optax for JAX-optimized implementations.
    """

    def __init__(self, config: OptimizerConfig):
        """Initialize the base optimizer with configuration.

        Args:
            config (OptimizerConfig): Optimizer configuration with hyperparameters.

        Example:
            >>> config = OptimizerConfig(
            ...     learning_rate=0.0001,
            ...     optimizer_type="adam",
            ...     weight_decay=0.01
            ... )
            >>> optimizer = BaseOptimizer(config)
            >>> print(optimizer.config.learning_rate)  # 0.0001
        """
        self.config = config
        self.initialized = False
        self.state: Dict[str, Any] = {}
        self.step_count = 0

    def initialize(self) -> bool:
        """Initialize the optimizer state.

        This method sets up any necessary state for the optimizer. In the base
        implementation, it simply marks the optimizer as initialized and logs
        the confirmation.

        Returns:
            bool: True if initialization successful, False if error occurred.

        Example:
            >>> optimizer = BaseOptimizer(OptimizerConfig())
            >>> success = optimizer.initialize()
            >>> print(success)  # True
            >>> print(optimizer.initialized)  # True

        Note:
            Extended implementations should override this method to initialize
            optimizer-specific state (momentum buffers, adaptive learning rate
            parameters, etc.).
        """
        try:
            self.initialized = True
            logger.info(f"Optimizer {self.config.optimizer_type} initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            return False

    def step(self, gradients: Any) -> Any:
        """Perform a single optimization step with given gradients.

        This method applies the optimization algorithm to update parameters based
        on computed gradients.

        Args:
            gradients (Any): Gradients computed from loss backpropagation. Type
                depends on the model framework (e.g., JAX pytrees, PyTorch tensors).

        Returns:
            Any: Updated gradients or parameters. Type matches input type.

        Example:
            >>> optimizer = BaseOptimizer(OptimizerConfig())
            >>> gradients = {"layer1": [0.1, 0.2], "layer2": [0.3, 0.4]}
            >>> updated = optimizer.step(gradients)
            >>> # In base implementation, updated == gradients

        Note:
            The optimizer can accept gradients only, or a tuple/dict containing
            both params and grads, and applies clipping and the configured algorithm.
        """
        if not self.initialized:
            self.initialize()

        grads, params = self._split_params_and_grads(gradients)
        grads = self._apply_gradient_clipping(grads)

        if params is None:
            logger.debug("Optimizer step performed (gradients only)")
            return grads

        self._ensure_state(params)
        self.step_count += 1

        if self.config.optimizer_type in ("sgd",):
            updated_params = self._sgd_update(params, grads)
        elif self.config.optimizer_type in ("adam", "adamw"):
            updated_params = self._adam_update(params, grads)
        else:
            logger.warning(f"Unknown optimizer type {self.config.optimizer_type}, returning params unchanged")
            updated_params = params

        logger.debug("Optimizer step performed")
        return updated_params

    def zero_grad(self) -> None:
        """Zero out accumulated gradients.

        This method resets gradient accumulators. In frameworks like PyTorch,
        this is essential before computing new gradients. In JAX with functional
        programming, this may be a no-op since gradients are recomputed each step.

        Example:
            >>> optimizer = BaseOptimizer(OptimizerConfig())
            >>> optimizer.zero_grad()  # Gradients zeroed

        Note:
            Extended implementations should clear any accumulated gradient state
            or buffers specific to the optimization algorithm.
        """
        logger.debug("Gradients zeroed")

    def _split_params_and_grads(self, gradients: Any) -> Tuple[Any, Optional[Any]]:
        """Split inputs into grads and params if provided."""
        if isinstance(gradients, dict) and "params" in gradients and "grads" in gradients:
            return gradients.get("grads"), gradients.get("params")
        if isinstance(gradients, (list, tuple)) and len(gradients) == 2:
            params, grads = gradients
            return grads, params
        return gradients, None

    def _ensure_state(self, params: Any) -> None:
        """Initialize optimizer state based on params."""
        if "m" not in self.state:
            self.state["m"] = self._tree_map(self._zeros_like, params)
        if "v" not in self.state:
            self.state["v"] = self._tree_map(self._zeros_like, params)
        if "velocity" not in self.state:
            self.state["velocity"] = self._tree_map(self._zeros_like, params)

    def _apply_gradient_clipping(self, grads: Any) -> Any:
        """Apply global norm gradient clipping if configured."""
        if self.config.clip_grad_norm is None:
            return grads

        total_norm = self._global_norm(grads)
        if total_norm <= 0:
            return grads
        clip_coef = min(1.0, self.config.clip_grad_norm / (total_norm + 1e-6))
        return self._tree_map(lambda g: self._to_array(g) * clip_coef, grads)

    def _global_norm(self, grads: Any) -> float:
        """Compute global norm of gradients."""
        leaves = self._tree_leaves(grads)
        if not leaves:
            return 0.0
        total = 0.0
        for leaf in leaves:
            arr = self._to_array(leaf)
            total += float(np.sum(arr * arr))
        return float(np.sqrt(total))

    def _sgd_update(self, params: Any, grads: Any) -> Any:
        """SGD with momentum and optional weight decay."""
        momentum = self.config.momentum
        weight_decay = self.config.weight_decay

        def update(p, g, v):
            p_arr = self._to_array(p)
            g_arr = self._to_array(g)
            if weight_decay > 0:
                g_arr = g_arr + weight_decay * p_arr
            v_new = momentum * self._to_array(v) + g_arr
            p_new = p_arr - self.config.learning_rate * v_new
            return p_new, v_new

        updated_params = self._tree_map3(lambda p, g, v: update(p, g, v)[0], params, grads, self.state["velocity"])
        updated_velocity = self._tree_map3(lambda p, g, v: update(p, g, v)[1], params, grads, self.state["velocity"])
        self.state["velocity"] = updated_velocity
        return updated_params

    def _adam_update(self, params: Any, grads: Any) -> Any:
        """Adam/AdamW update."""
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        eps = self.config.eps
        weight_decay = self.config.weight_decay if self.config.optimizer_type == "adamw" else 0.0

        def update(p, g, m, v):
            p_arr = self._to_array(p)
            g_arr = self._to_array(g)
            m_new = beta1 * self._to_array(m) + (1 - beta1) * g_arr
            v_new = beta2 * self._to_array(v) + (1 - beta2) * (g_arr ** 2)
            m_hat = m_new / (1 - beta1 ** self.step_count)
            v_hat = v_new / (1 - beta2 ** self.step_count)
            update_term = m_hat / (np.sqrt(v_hat) + eps)
            if weight_decay > 0:
                update_term = update_term + weight_decay * p_arr
            p_new = p_arr - self.config.learning_rate * update_term
            return p_new, m_new, v_new

        updated_params = self._tree_map4(
            lambda p, g, m, v: update(p, g, m, v)[0],
            params, grads, self.state["m"], self.state["v"]
        )
        updated_m = self._tree_map4(
            lambda p, g, m, v: update(p, g, m, v)[1],
            params, grads, self.state["m"], self.state["v"]
        )
        updated_v = self._tree_map4(
            lambda p, g, m, v: update(p, g, m, v)[2],
            params, grads, self.state["m"], self.state["v"]
        )
        self.state["m"] = updated_m
        self.state["v"] = updated_v
        return updated_params

    def _tree_map(self, fn, tree: Any) -> Any:
        if isinstance(tree, dict):
            return {k: self._tree_map(fn, v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [self._tree_map(fn, v) for v in tree]
        if isinstance(tree, tuple):
            return tuple(self._tree_map(fn, v) for v in tree)
        return fn(tree)

    def _tree_map3(self, fn, tree_a: Any, tree_b: Any, tree_c: Any) -> Any:
        if isinstance(tree_a, dict):
            return {
                k: self._tree_map3(fn, tree_a[k], tree_b[k], tree_c[k])
                for k in tree_a
            }
        if isinstance(tree_a, list):
            return [self._tree_map3(fn, a, b, c) for a, b, c in zip(tree_a, tree_b, tree_c)]
        if isinstance(tree_a, tuple):
            return tuple(self._tree_map3(fn, a, b, c) for a, b, c in zip(tree_a, tree_b, tree_c))
        return fn(tree_a, tree_b, tree_c)

    def _tree_map4(self, fn, tree_a: Any, tree_b: Any, tree_c: Any, tree_d: Any) -> Any:
        if isinstance(tree_a, dict):
            return {
                k: self._tree_map4(fn, tree_a[k], tree_b[k], tree_c[k], tree_d[k])
                for k in tree_a
            }
        if isinstance(tree_a, list):
            return [self._tree_map4(fn, a, b, c, d) for a, b, c, d in zip(tree_a, tree_b, tree_c, tree_d)]
        if isinstance(tree_a, tuple):
            return tuple(self._tree_map4(fn, a, b, c, d) for a, b, c, d in zip(tree_a, tree_b, tree_c, tree_d))
        return fn(tree_a, tree_b, tree_c, tree_d)

    def _tree_leaves(self, tree: Any) -> List[Any]:
        if isinstance(tree, dict):
            leaves = []
            for v in tree.values():
                leaves.extend(self._tree_leaves(v))
            return leaves
        if isinstance(tree, list):
            leaves = []
            for v in tree:
                leaves.extend(self._tree_leaves(v))
            return leaves
        if isinstance(tree, tuple):
            leaves = []
            for v in tree:
                leaves.extend(self._tree_leaves(v))
            return leaves
        return [tree]

    def _to_array(self, value: Any) -> np.ndarray:
        try:
            return np.asarray(value, dtype=np.float32)
        except Exception:
            return np.asarray(0.0, dtype=np.float32)

    def _zeros_like(self, value: Any) -> Any:
        try:
            return np.zeros_like(self._to_array(value))
        except Exception:
            return 0.0

def create_optimizer(config: Optional[OptimizerConfig] = None) -> BaseOptimizer:
    """Factory function to create and initialize an optimizer.

    Creates an optimizer instance with the specified configuration, automatically
    initializing it for immediate use.

    Args:
        config (Optional[OptimizerConfig], optional): Optimizer configuration.
            If None, uses default OptimizerConfig(). Defaults to None.

    Returns:
        BaseOptimizer: Initialized optimizer instance ready for training.

    Example:
        >>> # Create with defaults
        >>> optimizer = create_optimizer()
        >>> print(optimizer.config.optimizer_type)  # "adam"
        >>>
        >>> # Create with custom config
        >>> config = OptimizerConfig(
        ...     learning_rate=0.0001,
        ...     optimizer_type="adamw",
        ...     weight_decay=0.01
        ... )
        >>> optimizer = create_optimizer(config)
        >>> print(optimizer.initialized)  # True

    Note:
        This is the recommended way to create optimizers, as it ensures
        proper initialization. The optimizer is ready to use immediately
        after creation.
    """
    if config is None:
        config = OptimizerConfig()

    optimizer = BaseOptimizer(config)
    optimizer.initialize()

    logger.info(f"Created {config.optimizer_type} optimizer")
    return optimizer

def main():
    """Main function for module execution.

    This function is called when the module is run directly. Provides basic
    logging confirmation that the module loaded successfully.

    Returns:
        bool: Always returns True to indicate successful execution.

    Example:
        >>> from capibara.core.optimizers import optimizer
        >>> result = optimizer.main()
        >>> print(result)  # True

    Note:
        This is primarily for testing and verification purposes. Production code
        should use create_optimizer() or instantiate BaseOptimizer directly.
    """
    logger.info("Module optimizer.py starting")
    return True

if __name__ == "__main__":
    main()
