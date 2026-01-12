# Optimizers Module

Neural network optimizer system with configurable parameters, support for multiple algorithms, and factory functions for optimized creation.

## 📋 Description

Module that provides advanced optimizers with support for Adam, momentum, weight decay, gradient clipping, and learning rate scheduling, optimized for training large language models.

## 🏗️ Architecture

```
optimizers/
├── __init__.py     # Optimizer exports
└── optimizer.py    # Base optimizer class with configurations
```

## 🚀 Base Optimizer

```python
from capibara.core.optimizers import Optimizer

# Configure optimizer with advanced parameters
optimizer = Optimizer(
    algorithm="adam",
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.01,
    gradient_clip_norm=1.0,
    use_scheduler=True,
    scheduler_type="cosine_with_warmup"
)

# Configure advanced scheduler
scheduler_config = {
    "warmup_steps": 10000,
    "max_steps": 500000,
    "min_learning_rate": 1e-6,
    "cosine_restarts": True,
    "restart_decay": 0.8,
    "restart_period": 100000
}

optimizer.configure_scheduler(scheduler_config)

# Apply optimization with metrics
optimization_step = optimizer.step(
    gradients=model_gradients,
    parameters=model_parameters,
    step_number=current_step,
    return_metrics=True
)

print(f"Learning rate: {optimization_step.learning_rate:.2e}")
print(f"Gradient norm: {optimization_step.gradient_norm:.4f}")
print(f"Parameter norm: {optimization_step.parameter_norm:.4f}")
print(f"Weight decay applied: {optimization_step.weight_decay_loss:.6f}")
```

## ⚡ Supported Algorithms

### Optimized Adam

```python
# Adam with LLM-specific configurations
adam_config = {
    "algorithm": "adam",
    "learning_rate": 3e-4,
    "beta1": 0.9,
    "beta2": 0.95,  # Typical value for LLMs
    "epsilon": 1e-8,
    "weight_decay": 0.1,
    "amsgrad": False,
    "maximize": False,
    "foreach": True,  # Vectorized optimization
    "differentiable": False
}

adam_optimizer = Optimizer.create_adam(adam_config)
```

### AdamW with Weight Decay

```python
# AdamW optimized for transformers
adamw_config = {
    "algorithm": "adamw",
    "learning_rate": 6e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1e-8,
    "weight_decay": 0.1,
    "correct_bias": True,  # Bias correction
    "no_deprecation_warning": True
}

adamw_optimizer = Optimizer.create_adamw(adamw_config)
```

### Momentum SGD

```python
# SGD with momentum for specific cases
sgd_config = {
    "algorithm": "sgd",
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "dampening": 0.1
}

sgd_optimizer = Optimizer.create_sgd(sgd_config)
```

## 📊 Learning Rate Scheduling

### Cosine Annealing with Warmup

```python
# Cosine scheduler with warmup (standard for LLMs)
cosine_scheduler_config = {
    "scheduler_type": "cosine_with_warmup",
    "warmup_steps": 2000,
    "max_steps": 100000,
    "max_learning_rate": 6e-4,
    "min_learning_rate": 6e-5,
    "cosine_cycles": 1.0,
    "warmup_init_lr": 0.0
}

optimizer.configure_cosine_scheduler(cosine_scheduler_config)

# Get learning rate for specific step
current_lr = optimizer.get_learning_rate(step=50000)
print(f"Learning rate at step 50000: {current_lr:.2e}")
```

### Linear Decay with Warmup

```python
# Linear scheduler with warmup
linear_scheduler_config = {
    "scheduler_type": "linear_with_warmup",
    "warmup_steps": 5000,
    "training_steps": 200000,
    "start_lr": 0.0,
    "peak_lr": 3e-4,
    "end_lr": 1e-5
}

optimizer.configure_linear_scheduler(linear_scheduler_config)
```

### Polynomial Decay

```python
# Polynomial decay scheduler
poly_scheduler_config = {
    "scheduler_type": "polynomial",
    "power": 0.5,
    "warmup_steps": 1000,
    "total_steps": 500000,
    "end_learning_rate": 1e-6
}

optimizer.configure_polynomial_scheduler(poly_scheduler_config)
```

## 🎯 Specialized Configurations

### For Large Models (>1B parameters)

```python
# Optimized configuration for large models
large_model_config = {
    "algorithm": "adamw",
    "learning_rate": 1.5e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1e-8,
    "weight_decay": 0.1,
    "gradient_clip_norm": 1.0,
    "gradient_accumulation_steps": 8,
    "scheduler_type": "cosine_with_warmup",
    "warmup_ratio": 0.03,  # 3% of total steps
    "min_lr_ratio": 0.1    # 10% of peak LR
}

large_model_optimizer = Optimizer.create_for_large_model(large_model_config)
```

### For Fine-tuning

```python
# Configuration for fine-tuning with lower learning rates
finetune_config = {
    "algorithm": "adam",
    "learning_rate": 5e-5,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0.01,
    "gradient_clip_norm": 0.5,  # More conservative for fine-tuning
    "scheduler_type": "linear_with_warmup",
    "warmup_steps": 500,
    "layer_wise_lr_decay": 0.95,  # Different LRs per layer
    "differential_learning_rates": {
        "embedding": 0.1,   # 10% of base LR
        "encoder": 1.0,     # 100% of base LR
        "head": 2.0         # 200% of base LR
    }
}

finetune_optimizer = Optimizer.create_for_finetuning(finetune_config)
```

## 🔧 Gradient Processing

### Advanced Gradient Clipping

```python
# Gradient clipping configuration
clip_config = {
    "clip_method": "norm",  # "norm", "value", "adaptive"
    "clip_value": 1.0,
    "adaptive_clipping": {
        "enable": True,
        "percentile": 10,  # Clip based on historical norm percentile
        "history_length": 1000,
        "min_clip_value": 0.1,
        "max_clip_value": 10.0
    },
    "per_parameter_clipping": False,
    "clip_coeff": 0.01
}

optimizer.configure_gradient_clipping(clip_config)

# Gradient analysis
grad_analysis = optimizer.analyze_gradients(
    gradients=current_gradients,
    include_histogram=True,
    include_layer_wise_stats=True
)

print(f"Gradient norm distribution: {grad_analysis['norm_percentiles']}")
print(f"Layer-wise gradient norms: {grad_analysis['layer_norms']}")
print(f"Clipping frequency: {grad_analysis['clipping_frequency']:.3f}")
```

### Gradient Accumulation

```python
# Configure gradient accumulation for large effective batch sizes
accumulation_config = {
    "accumulation_steps": 16,  # Effective batch size = 16x real batch size
    "normalize_gradients": True,
    "sync_batchnorm": True,
    "scale_loss": True,
    "defer_clip_until_accumulation": True  # Clip after accumulation
}

optimizer.configure_gradient_accumulation(accumulation_config)

# Training step with accumulation
for micro_batch in micro_batches:
    # Forward pass
    loss = model(micro_batch)
    scaled_loss = loss / accumulation_config["accumulation_steps"]

    # Backward pass
    scaled_loss.backward()

    # Accumulate gradients
    optimizer.accumulate_gradients()

# Apply accumulated gradients
optimizer.step_with_accumulated_gradients()
optimizer.zero_grad()
```

## 📈 Metrics and Monitoring

### Optimization Metrics

```python
# Optimization metrics system
optimization_metrics = optimizer.get_optimization_metrics()

metrics_summary = {
    "training_dynamics": {
        "learning_rate": optimization_metrics["current_lr"],
        "gradient_norm": optimization_metrics["grad_norm"],
        "parameter_norm": optimization_metrics["param_norm"],
        "update_norm": optimization_metrics["update_norm"],
        "update_to_param_ratio": optimization_metrics["update_ratio"]
    },

    "convergence_indicators": {
        "loss_smoothness": optimization_metrics["loss_smoothness"],
        "gradient_variance": optimization_metrics["grad_variance"],
        "parameter_change_rate": optimization_metrics["param_change_rate"],
        "optimization_progress": optimization_metrics["progress_score"]
    },

    "stability_metrics": {
        "gradient_explosion_risk": optimization_metrics["explosion_risk"],
        "vanishing_gradient_risk": optimization_metrics["vanishing_risk"],
        "learning_rate_stability": optimization_metrics["lr_stability"],
        "parameter_stability": optimization_metrics["param_stability"]
    }
}

# Optimization alerts
optimization_alerts = optimizer.check_optimization_health()
for alert in optimization_alerts:
    print(f"⚠️ {alert.level}: {alert.message}")
    if alert.suggestion:
        print(f"💡 Suggestion: {alert.suggestion}")
```

### Training Dynamics Visualization

```python
# Generate training dynamics plots
training_visualizations = optimizer.generate_training_plots(
    metrics_history=training_history,
    plot_types=[
        "learning_rate_schedule",
        "gradient_norm_evolution",
        "parameter_updates",
        "loss_landscape_approximation"
    ],
    save_plots=True,
    plot_directory="training_plots/"
)

# Convergence analysis
convergence_analysis = optimizer.analyze_convergence(
    loss_history=loss_values,
    gradient_history=gradient_norms,
    window_size=1000,
    smoothing_factor=0.99
)

print("📈 Convergence Analysis:")
print(f"Convergence rate: {convergence_analysis['rate']:.2e}")
print(f"Estimated steps to convergence: {convergence_analysis['eta_steps']}")
print(f"Training stability score: {convergence_analysis['stability']:.3f}")
```

## 🤖 Hyperparameter Auto-tuning

```python
# Auto-tuning system for hyperparameters
from capibara.core.optimizers import OptimizerAutoTuner

auto_tuner = OptimizerAutoTuner(
    search_space={
        "learning_rate": [1e-5, 1e-3, "log"],
        "beta1": [0.85, 0.95, "uniform"],
        "beta2": [0.9, 0.999, "uniform"],
        "weight_decay": [0.0, 0.2, "uniform"],
        "warmup_steps": [500, 5000, "int"],
        "gradient_clip_norm": [0.1, 2.0, "uniform"]
    },
    optimization_metric="validation_loss",
    search_algorithm="bayesian_optimization",
    max_trials=50
)

# Run auto-tuning
optimal_config = auto_tuner.find_optimal_hyperparameters(
    model=model,
    train_dataset=train_data,
    val_dataset=val_data,
    max_steps_per_trial=5000,
    early_stopping_patience=1000
)

print("🎯 Optimal Optimizer Configuration:")
for param, value in optimal_config.items():
    print(f"  {param}: {value}")

# Create automatically optimized optimizer
auto_optimized_optimizer = Optimizer.from_config(optimal_config)
```

## 🚀 Factory Functions

```python
# Factory functions for common configurations
optimizers_collection = {
    # For large model pre-training
    "pretraining_large": Optimizer.create_pretraining_optimizer(
        model_size="large",
        dataset_size="multi_billion_tokens",
        target_steps=500000
    ),

    # For fast fine-tuning
    "finetuning_fast": Optimizer.create_finetuning_optimizer(
        task_type="text_classification",
        dataset_size="medium",
        target_epochs=3
    ),

    # For resource-constrained training
    "resource_efficient": Optimizer.create_efficient_optimizer(
        memory_budget="8GB",
        compute_budget="medium",
        prioritize="memory"
    ),

    # For research experiments
    "research_flexible": Optimizer.create_research_optimizer(
        experimental_features=True,
        detailed_logging=True,
        custom_schedulers=True
    )
}

# Use factory function
pretraining_opt = optimizers_collection["pretraining_large"]
```

## 📚 References

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)
- [Training Large Language Models](https://arxiv.org/abs/2104.04473)
- [Learning Rate Schedules for Deep Learning](https://arxiv.org/abs/1506.01186)
