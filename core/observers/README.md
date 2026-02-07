# Observer Pattern for Dynamic Expert Activation

This module implements the Observer pattern to dynamically activate experts based on request patterns and inputs. It enables an intelligent routing system that can adapt to different query types and automatically activate the most appropriate experts.

##  Main Features

###  Intelligent Observers
- **RequestPatternObserver**: Detects specific patterns in request text
- **ComplexityObserver**: Analyzes query complexity and activates experts as needed
- **DomainSpecificObserver**: Specialized in detecting specific domains (mathematics, programming, etc.)
- **PerformanceObserver**: Monitors system performance and makes load-based decisions
- **AdaptiveObserver**: Learns from activation patterns and adapts over time

###  Dynamic Expert Management
- **ExpertActivationManager**: Manages expert lifecycle
- **ExpertPool**: Dynamic expert pool that can be activated on demand
- **ActivationStrategy**: Different strategies for expert activation

###  Router Integration
- **ObserverAwareRouter**: Router that integrates the Observer pattern with traditional routing
- **RoutingMode**: Different routing modes (traditional, observer-first, hybrid)
- **DynamicRoutingDecision**: Routing decisions enriched with observer information

##  Use Cases

### 1. Automatic Pattern-Based Activation
```python
from capibara.core.observers import create_observer_aware_router, RoutingMode

# Create router with observer pattern
router = create_observer_aware_router(
    routing_mode=RoutingMode.OBSERVER_ENHANCED
)

# Process request - experts are activated automatically
result = await router.route_request(
    input_data="What would happen if the main server fails during peak traffic?"
)

print(f"Activated experts: {result.experts_activated}")
# Output: ['CSA'] - Counterfactual analysis expert automatically activated
```

### 2. Complexity Detection
```python
# Complex request that activates multiple experts
complex_query = """
Design a machine learning system to predict server failures.
What would happen if the training data is biased?
Include the Python code and calculate the error probability.
"""

result = await router.route_request(input_data=complex_query)
print(f"Activated experts: {result.experts_activated}")
# Output: ['CSA', 'CodeExpert', 'MathExpert'] - Multiple experts for complex query
```

### 3. Adaptive Learning
```python
from capibara.core.observers import ActivationStrategy

# Router with adaptive learning
adaptive_router = create_observer_aware_router(
    routing_mode=RoutingMode.OBSERVER_ENHANCED,
    activation_strategy=ActivationStrategy.ADAPTIVE
)

# Process requests and provide feedback
result = await adaptive_router.route_request(input_data="Calculate derivative of x²")
adaptive_router.provide_feedback("req_001", {"MathExpert": True})  # Positive feedback

# The system learns and improves future activations
```

### 4. Custom Observers
```python
from capibara.core.observers import RequestObserver, create_expert_activation_event

class CustomDomainObserver(RequestObserver):
    async def observe(self, event):
        if "blockchain" in event.request_text.lower():
            return [create_expert_activation_event(
                expert_name="BlockchainExpert",
                reason="Blockchain domain detected",
                confidence=0.8
            )]
        return []

# Add custom observer
router.add_observer(CustomDomainObserver("BlockchainObserver"))
```

## ️ Architecture

### Main Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ObserverAwareRouter                      │
├─────────────────────────────────────────────────────────────┤
│                RouterObserverIntegration                    │
├─────────────────────────────────────────────────────────────┤
│              ExpertActivationManager                        │
├─────────────────────────────────────────────────────────────┤
│    ObserverManager           │         ExpertPool           │
│  ┌─────────────────────────┐ │ ┌─────────────────────────┐  │
│  │ RequestPatternObserver  │ │ │    CSAExpert           │  │
│  │ ComplexityObserver      │ │ │    MathExpert          │  │
│  │ DomainSpecificObserver  │ │ │    CodeExpert          │  │
│  │ PerformanceObserver     │ │ │    SpanishExpert       │  │
│  │ AdaptiveObserver        │ │ │    CustomExperts       │  │
│  └─────────────────────────┘ │ └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Activation Flow

1. **Request Reception**: Router receives a user request
2. **Observer Notification**: All active observers are notified
3. **Pattern Analysis**: Each observer analyzes the request according to its specialty
4. **Event Generation**: Observers Generate expert activation events
5. **Strategy Application**: Configured activation strategy is applied
6. **Expert Activation**: Appropriate experts are activated
7. **Processing**: Experts process the request
8. **Result Combination**: Results from multiple experts are combined

##  Activation Strategies

### ActivationStrategy.IMMEDIATE
Activates experts immediately when observers suggest it.
```python
router = create_observer_aware_router(
    activation_strategy=ActivationStrategy.IMMEDIATE
)
```

### ActivationStrategy.THRESHOLD_BASED
Requires a minimum confidence level to activate experts.
```python
router = create_observer_aware_router(
    activation_strategy=ActivationStrategy.THRESHOLD_BASED
)
# Configure threshold
router.observer_integration.strategy_config["confidence_threshold"] = 0.7
```

### ActivationStrategy.CONSENSUS_REQUIRED
Requires multiple observers to agree before activating an expert.
```python
router = create_observer_aware_router(
    activation_strategy=ActivationStrategy.CONSENSUS_REQUIRED
)
# Configure required votes
router.observer_integration.strategy_config["consensus_required_votes"] = 2
```

### ActivationStrategy.LOAD_BALANCED
Considers current system load when making activation decisions.
```python
router = create_observer_aware_router(
    activation_strategy=ActivationStrategy.LOAD_BALANCED
)
```

### ActivationStrategy.ADAPTIVE
Learns from past activations and adapts automatically.
```python
router = create_observer_aware_router(
    activation_strategy=ActivationStrategy.ADAPTIVE
)
```

## ️ Routing Modes

### RoutingMode.TRADITIONAL
Uses only traditional routing, no observers.

### RoutingMode.OBSERVER_FIRST
Observers take priority over traditional routing.

### RoutingMode.OBSERVER_ENHANCED
Combines traditional routing with observer activation.

### RoutingMode.HYBRID
Dynamically switches between modes based on request complexity.

##  Monitoring and Metrics

### Router Statistics
```python
stats = router.get_statistics()
print(f"Total requests: {stats['integration_metrics']['total_requests']}")
print(f"Observer activations: {stats['integration_metrics']['observer_activations']}")
print(f"Expert activations: {stats['integration_metrics']['expert_activations']}")
```

### Observer Performance
```python
observer_manager = router.observer_integration.activation_manager.observer_manager
for observer in observer_manager.observers:
    performance = observer.get_performance_summary()
    print(f"{observer.name}: {performance['success_rate']:.2%} success")
```

### Expert Metrics
```python
expert_pool = router.observer_integration.activation_manager.expert_pool
pool_stats = expert_pool.get_pool_statistics()
print(f"Pool utilization: {pool_stats['current_utilization']:.2%}")
```

##  Examples and Demos

### Comprehensive Demo
```python
from capibara.core.observers.examples import ObserverPatternDemo

demo = ObserverPatternDemo()
await demo.run_comprehensive_demo()
```

### Interactive Demo
```python
from capibara.core.observers.examples import InteractiveObserverDemo

interactive_demo = InteractiveObserverDemo()
await interactive_demo.run_interactive_demo()
```

### Quick Test
```python
from capibara.core.observers.examples import quick_test

await quick_test()
```

### Performance Benchmark
```python
from capibara.core.observers.examples import benchmark_observer_performance

await benchmark_observer_performance()
```

##  Advanced Configuration

### Observer Configuration
```python
# Create custom pattern observer
pattern_observer = RequestPatternObserver("CustomPatterns", priority=1)
pattern_observer.expert_patterns["MyExpert"] = [
    r"(?i)\b(custom|pattern|detection)\b"
]

router.add_observer(pattern_observer)
```

### Custom Expert Registration
```python
class MyCustomExpert:
    async def process(self, context):
        return {"result": "Custom processing completed"}

router.register_expert("MyExpert", MyCustomExpert, max_concurrent=2)
```

### Strategy Configuration
```python
# Configure strategy parameters
router.observer_integration.strategy_config.update({
    "confidence_threshold": 0.8,
    "consensus_required_votes": 3,
    "max_concurrent_activations": 5,
    "load_balance_threshold": 0.7
})
```

##  Debugging and Logs

### Enable Detailed Logging
```python
import logging
logging.getLogger("capibara.core.observers").setLevel(logging.DEBUG)
```

### Inspect Activation Events
```python
# Access activation history
activation_history = router.observer_integration.activation_manager.activation_history
for activation in activation_history[-5:]:  # Last 5 activations
    print(f"Request: {activation['request_id']}")
    print(f"Experts: {activation['approved_activations']}")
```

##  Best Practices

1. **Priority Configuration**: Assign appropriate priorities to observers
2. **Confidence Thresholds**: Adjust thresholds according to your precision needs
3. **Performance Monitoring**: Regularly monitor system metrics
4. **Learning Feedback**: Provide feedback to improve adaptive learning
5. **Resource Management**: Configure appropriate concurrency limits for experts

##  References

- [Observer Pattern](https://refactoring.guru/design-patterns/observer)
- [Expert Systems](https://en.wikipedia.org/wiki/Expert_system)
- [Dynamic Routing](https://en.wikipedia.org/wiki/Dynamic_routing)

##  Contributing

To contribute to Observer pattern development:

1. Implement new specialized observers
2. Improve existing activation strategies
3. Add additional metrics and monitoring
4. Create examples and use cases
5. Optimize system performance

##  License

This module is part of CapibaraGPT and is subject to the same license as the main project.
