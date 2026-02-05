"""
Observer Pattern Examples and Use Cases
=======================================

This module provides comprehensive examples and test cases for the Observer pattern
implementation in CapibaraGPT. It demonstrates various use cases and scenarios
where dynamic expert activation provides value.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

from .router_integration import (
    ObserverAwareRouter,
    create_observer_aware_router,
    create_simple_observer_router,
    RoutingMode
)
from .expert_activation_manager import ActivationStrategy
from .observers import (
    RequestPatternObserver,
    ComplexityObserver,
    DomainSpecificObserver,
    AdaptiveObserver
)
from .request_observer import (
    RequestEvent,
    RequestEventType,
    create_request_received_event,
    create_expert_activation_event
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObserverPatternDemo:
    """
    Comprehensive demonstration of the Observer pattern for expert activation.
    """
    
    def __init__(self):
        self.demo_requests = self._create_demo_requests()
        self.routers = self._create_demo_routers()
    
    def _create_demo_requests(self) -> List[Dict[str, Any]]:
        """Create a variety of demo requests to test different activation patterns."""
        return [
            # Mathematical requests
            {
                "id": "math_001",
                "text": "Resuelve la ecuación cuadrática: 2x² + 5x - 3 = 0",
                "expected_experts": ["MathExpert"],
                "category": "mathematics"
            },
            {
                "id": "math_002", 
                "text": "Calculate the integral of x² + 3x + 2 from 0 to 5",
                "expected_experts": ["MathExpert"],
                "category": "mathematics"
            },
            
            # Programming requests
            {
                "id": "code_001",
                "text": "Implementa un algoritmo de ordenamiento quicksort en Python",
                "expected_experts": ["CodeExpert"],
                "category": "programming"
            },
            {
                "id": "code_002",
                "text": "Debug this JavaScript function: ```function factorial(n) { return n * factorial(n-1); }```",
                "expected_experts": ["CodeExpert"],
                "category": "programming"
            },
            
            # Counterfactual/Analysis requests
            {
                "id": "csa_001",
                "text": "¿Qué pasaría si el servidor principal falla durante el pico de tráfico? Analiza los posibles escenarios.",
                "expected_experts": ["CSA"],
                "category": "analysis"
            },
            {
                "id": "csa_002",
                "text": "The production system is experiencing intermittent failures. What if the database connection pool is exhausted?",
                "expected_experts": ["CSA"],
                "category": "analysis"
            },
            
            # Translation requests
            {
                "id": "lang_001",
                "text": "Translate this to Spanish: 'The quick brown fox jumps over the lazy dog'",
                "expected_experts": ["SpanishExpert"],
                "category": "language"
            },
            {
                "id": "lang_002",
                "text": "Necesito ayuda con la gramática española en esta oración",
                "expected_experts": ["SpanishExpert"],
                "category": "language"
            },
            
            # Complex multi-domain requests
            {
                "id": "multi_001",
                "text": "Diseña un system de recomendación que use machine learning. ¿Qué pasaría si los datos de entrenamiento están sesgados? Incluye el código en Python.",
                "expected_experts": ["CodeExpert", "CSA"],
                "category": "multi-domain"
            },
            {
                "id": "multi_002",
                "text": "Calculate the probability distribution for user behavior and analyze what if scenarios for system load. Implement the solution.",
                "expected_experts": ["MathExpert", "CSA", "CodeExpert"],
                "category": "multi-domain"
            },
            
            # Simple requests (should use minimal experts)
            {
                "id": "simple_001",
                "text": "Hello, how are you?",
                "expected_experts": [],
                "category": "simple"
            },
            {
                "id": "simple_002",
                "text": "What's the weather like?",
                "expected_experts": [],
                "category": "simple"
            }
        ]
    
    def _create_demo_routers(self) -> Dict[str, ObserverAwareRouter]:
        """Create different router configurations for demonstration."""
        return {
            "observer_first": create_observer_aware_router(
                routing_mode=RoutingMode.OBSERVER_FIRST,
                activation_strategy=ActivationStrategy.THRESHOLD_BASED
            ),
            "observer_enhanced": create_observer_aware_router(
                routing_mode=RoutingMode.OBSERVER_ENHANCED,
                activation_strategy=ActivationStrategy.CONSENSUS_REQUIRED
            ),
            "hybrid": create_observer_aware_router(
                routing_mode=RoutingMode.HYBRID,
                activation_strategy=ActivationStrategy.ADAPTIVE
            ),
            "simple": create_simple_observer_router()
        }
    
    async def run_comprehensive_demo(self):
        """Run a comprehensive demonstration of all features."""
        logger.info(" Observer Pattern Comprehensive Demo")
        logger.info("=" * 50)
        
        # Test each router configuration
        for router_name, router in self.routers.items():
            logger.info(f"\n Testing Router Configuration: {router_name}")
            logger.info("-" * 40)
            
            await self._test_router_with_requests(router, router_name)
            
            # Show statistics
            stats = router.get_statistics()
            logger.info(f"\n Router Statistics for {router_name}:")
            self._print_router_statistics(stats)
        
        # Run specific feature demonstrations
        await self._demo_adaptive_learning()
        await self._demo_performance_monitoring()
        await self._demo_custom_observers()
        
        logger.info("\n Comprehensive demo completed!")
    
    async def _test_router_with_requests(self, router: ObserverAwareRouter, router_name: str):
        """Test a router with various request types."""
        results = []
        
        for request in self.demo_requests[:6]:  # Test with first 6 requests
            logger.info(f"\n Processing: {request['id']} ({request['category']})")
            logger.info(f"   Text: {request['text'][:80]}...")
            
            start_time = time.time()
            result = await router.route_request(
                input_data=request['text'],
                request_id=request['id']
            )
            processing_time = time.time() - start_time
            
            # Analyze results
            activated_experts = result.experts_activated
            expected_experts = request['expected_experts']
            
            logger.info(f"    Experts activated: {activated_experts}")
            logger.info(f"    Expected experts: {expected_experts}")
            logger.info(f"    Processing time: {processing_time:.3f}s")
            logger.info(f"    Confidence: {result.confidence:.2f}")
            
            # Check if activation was appropriate
            if expected_experts:
                overlap = set(activated_experts) & set(expected_experts)
                coverage = len(overlap) / len(expected_experts) if expected_experts else 0
                logger.info(f"    Coverage: {coverage:.2f} ({len(overlap)}/{len(expected_experts)})")
            
            results.append({
                "request": request,
                "result": result,
                "processing_time": processing_time
            })
        
        return results
    
    def _print_router_statistics(self, stats: Dict[str, Any]):
        """Print formatted router statistics."""
        integration_metrics = stats.get("integration_metrics", {})
        
        logger.info(f"   Total requests: {integration_metrics.get('total_requests', 0)}")
        logger.info(f"   Observer activations: {integration_metrics.get('observer_activations', 0)}")
        logger.info(f"   Expert activations: {integration_metrics.get('expert_activations', 0)}")
        logger.info(f"   Routing mode: {stats.get('routing_mode', 'unknown')}")
        logger.info(f"   Observer count: {stats.get('observer_count', 0)}")
        logger.info(f"   Expert pool size: {stats.get('expert_pool_size', 0)}")
    
    async def _demo_adaptive_learning(self):
        """Demonstrate adaptive learning capabilities."""
        logger.info(f"\n Adaptive Learning Demo")
        logger.info("-" * 30)
        
        # Create router with adaptive strategy
        adaptive_router = create_observer_aware_router(
            routing_mode=RoutingMode.OBSERVER_ENHANCED,
            activation_strategy=ActivationStrategy.ADAPTIVE
        )
        
        # Process several requests
        learning_requests = [
            "Calculate the derivative of x³ + 2x² - 5x + 1",
            "¿Qué pasaría si la base de datos se queda sin espacio?",
            "Implement a binary search algorithm in Python",
            "What if the server load increases by 300%?"
        ]
        
        logger.info("Processing requests for adaptive learning...")
        for i, request_text in enumerate(learning_requests):
            result = await adaptive_router.route_request(
                input_data=request_text,
                request_id=f"adaptive_{i}"
            )
            
            # Simulate feedback (in real scenario, this would come from user/system)
            feedback = {expert: True for expert in result.experts_activated}  # Assume success
            adaptive_router.provide_feedback(f"adaptive_{i}", feedback)
            
            logger.info(f"   Request {i+1}: {len(result.experts_activated)} experts activated")
        
        # Show learning statistics
        stats = adaptive_router.get_statistics()
        activation_manager_stats = stats.get("activation_manager_stats", {})
        
        logger.info(f"   Adaptive weights learned: {len(activation_manager_stats.get('adaptive_weights', {}))}")
        logger.info("    Adaptive learning demo completed")
    
    async def _demo_performance_monitoring(self):
        """Demonstrate performance monitoring features."""
        logger.info(f"\n Performance Monitoring Demo")
        logger.info("-" * 35)
        
        # Create router with performance observer
        performance_router = create_observer_aware_router(
            routing_mode=RoutingMode.OBSERVER_ENHANCED,
            activation_strategy=ActivationStrategy.LOAD_BALANCED
        )
        
        # Simulate various load conditions
        load_scenarios = [
            ("Low load scenario", ["Simple calculation: 2 + 2"]),
            ("Medium load scenario", ["Calculate complex integral", "Analyze system performance"]),
            ("High load scenario", [
                "Complex mathematical analysis with multiple steps",
                "Comprehensive system diagnostic with multiple scenarios",
                "Advanced algorithm implementation with optimization"
            ])
        ]
        
        for scenario_name, requests in load_scenarios:
            logger.info(f"\n   {scenario_name}:")
            
            # Process requests concurrently to simulate load
            tasks = []
            for i, request in enumerate(requests):
                task = performance_router.route_request(
                    input_data=request,
                    request_id=f"perf_{scenario_name}_{i}"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Analyze performance
            total_experts = sum(len(r.experts_activated) for r in results)
            avg_processing_time = sum(r.total_processing_time for r in results) / len(results)
            
            logger.info(f"     Requests processed: {len(requests)}")
            logger.info(f"     Total experts activated: {total_experts}")
            logger.info(f"     Average processing time: {avg_processing_time:.3f}s")
        
        logger.info("    Performance monitoring demo completed")
    
    async def _demo_custom_observers(self):
        """Demonstrate creating and using custom observers."""
        logger.info(f"\n Custom Observer Demo")
        logger.info("-" * 25)
        
        # Create a custom observer
        class CustomDomainObserver(RequestPatternObserver):
            def __init__(self):
                super().__init__("CustomDomainObserver", priority=1)
                
                # Add custom patterns for specific domain
                self.expert_patterns["DataScienceExpert"] = [
                    r"(?i)\b(data science|machine learning|neural network|deep learning)\b",
                    r"(?i)\b(pandas|numpy|scikit-learn|tensorflow|pytorch)\b",
                    r"(?i)\b(regression|classification|clustering|prediction)\b"
                ]
                
                # Recompile patterns
                self.compiled_patterns["DataScienceExpert"] = [
                    __import__('re').compile(pattern) 
                    for pattern in self.expert_patterns["DataScienceExpert"]
                ]
        
        # Create router with custom observer
        custom_router = create_observer_aware_router()
        custom_observer = CustomDomainObserver()
        custom_router.add_observer(custom_observer)
        
        # Test with data science requests
        ds_requests = [
            "Build a machine learning model to predict customer churn using pandas and scikit-learn",
            "Implement a neural network for image classification with TensorFlow",
            "Perform clustering analysis on customer data using unsupervised learning"
        ]
        
        logger.info("Testing custom observer with data science requests:")
        for i, request in enumerate(ds_requests):
            result = await custom_router.route_request(
                input_data=request,
                request_id=f"custom_{i}"
            )
            
            logger.info(f"   Request {i+1}: {result.experts_activated}")
            # Note: DataScienceExpert won't actually activate since it's not registered,
            # but the observer would trigger the activation event
        
        logger.info("    Custom observer demo completed")


class InteractiveObserverDemo:
    """
    Interactive demonstration that allows users to test the observer pattern
    with their own inputs.
    """
    
    def __init__(self):
        self.router = create_observer_aware_router(
            routing_mode=RoutingMode.OBSERVER_ENHANCED,
            activation_strategy=ActivationStrategy.THRESHOLD_BASED
        )
    
    async def run_interactive_demo(self):
        """Run interactive demo where users can input their own requests."""
        logger.info(" Interactive Observer Pattern Demo")
        logger.info("=" * 40)
        logger.info("Enter requests to see how the observer pattern activates experts.")
        logger.info("Type 'quit' to exit, 'stats' to see statistics, 'help' for commands.")
        logger.info()
        
        request_count = 0
        
        while True:
            try:
                user_input = input(" Enter your request: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    self._show_statistics()
                    continue
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif not user_input:
                    continue
                
                # Process the request
                request_count += 1
                request_id = f"interactive_{request_count}"
                
                logger.info(f"\n Processing request: {request_id}")
                logger.info(f"   Text: {user_input}")
                
                start_time = time.time()
                result = await self.router.route_request(
                    input_data=user_input,
                    request_id=request_id
                )
                processing_time = time.time() - start_time
                
                # Display results
                logger.info(f"\n Results:")
                logger.info(f"   Experts activated: {result.experts_activated}")
                logger.info(f"   Observers activated: {result.observers_activated}")
                logger.info(f"   Confidence: {result.confidence:.2f}")
                logger.info(f"   Processing time: {processing_time:.3f}s")
                logger.info(f"   Routing mode: {result.routing_mode.value}")
                
                if result.expert_results:
                    logger.info(f"   Expert results: {len(result.expert_results)} results available")
                
                logger.info()
                
            except KeyboardInterrupt:
                logger.info("\n Demo interrupted by user")
                break
            except Exception as e:
                logger.error(f" Error processing request: {e}")
                continue
        
        logger.info(f"\n Final Statistics:")
        self._show_statistics()
        logger.info(" Interactive demo completed!")
    
    def _show_statistics(self):
        """Show current router statistics."""
        stats = self.router.get_statistics()
        
        logger.info("\n Current Statistics:")
        logger.info("-" * 20)
        
        integration_metrics = stats.get("integration_metrics", {})
        logger.info(f"   Total requests: {integration_metrics.get('total_requests', 0)}")
        logger.info(f"   Observer activations: {integration_metrics.get('observer_activations', 0)}")
        logger.info(f"   Expert activations: {integration_metrics.get('expert_activations', 0)}")
        
        # Show routing mode usage
        mode_usage = integration_metrics.get("routing_mode_usage", {})
        if mode_usage:
            logger.info(f"   Routing mode usage:")
            for mode, count in mode_usage.items():
                logger.info(f"     {mode}: {count}")
        
        logger.info(f"   Active observers: {stats.get('observer_count', 0)}")
        logger.info(f"   Expert pool size: {stats.get('expert_pool_size', 0)}")
    
    def _show_help(self):
        """Show help information."""
        logger.info("\n Available Commands:")
        logger.info("-" * 20)
        logger.info("   quit     - Exit the demo")
        logger.info("   stats    - Show current statistics")
        logger.info("   help     - Show this help message")
        logger.info()
        logger.info(" Example Requests:")
        logger.info("   'Calculate the integral of x² from 0 to 5'")
        logger.info("   '¿Qué pasaría si el servidor falla?'")
        logger.info("   'Implement quicksort in Python'")
        logger.info("   'Translate this to Spanish: Hello world'")
        logger.info()


# Utility functions for testing

async def quick_test():
    """Quick test of the observer pattern implementation."""
    logger.info(" Quick Observer Pattern Test")
    logger.info("=" * 30)
    
    # Create simple router
    router = create_simple_observer_router()
    
    # Test requests
    test_requests = [
        "Calculate 2 + 2",
        "¿Qué pasaría si el system falla?",
        "def hello(): print('Hello')",
        "Translate: Good morning"
    ]
    
    for i, request in enumerate(test_requests):
        logger.info(f"\n Request {i+1}: {request}")
        
        result = await router.route_request(
            input_data=request,
            request_id=f"test_{i}"
        )
        
        logger.info(f"    Experts: {result.experts_activated}")
        logger.info(f"    Confidence: {result.confidence:.2f}")
    
    logger.info(f"\n Quick test completed!")


async def benchmark_observer_performance():
    """Benchmark the performance of the observer pattern."""
    logger.info(" Observer Pattern Performance Benchmark")
    logger.info("=" * 45)
    
    # Create different router configurations
    routers = {
        "traditional": create_observer_aware_router(RoutingMode.TRADITIONAL),
        "observer_first": create_observer_aware_router(RoutingMode.OBSERVER_FIRST),
        "observer_enhanced": create_observer_aware_router(RoutingMode.OBSERVER_ENHANCED),
        "hybrid": create_observer_aware_router(RoutingMode.HYBRID)
    }
    
    # Benchmark requests
    benchmark_requests = [
        "Simple calculation: 5 * 3",
        "Complex analysis: What if the database fails during peak hours?",
        "Programming task: Implement binary search in Python",
        "Translation: Translate 'Hello world' to Spanish",
        "Multi-domain: Calculate probability and analyze failure scenarios"
    ] * 10  # 50 total requests
    
    # Run benchmarks
    for router_name, router in routers.items():
        logger.info(f"\n Benchmarking: {router_name}")
        
        start_time = time.time()
        
        # Process all requests
        tasks = []
        for i, request in enumerate(benchmark_requests):
            task = router.route_request(
                input_data=request,
                request_id=f"bench_{router_name}_{i}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_time = total_time / len(benchmark_requests)
        total_experts = sum(len(r.experts_activated) for r in results)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        logger.info(f"   Total time: {total_time:.3f}s")
        logger.info(f"   Average time per request: {avg_time:.4f}s")
        logger.info(f"   Total experts activated: {total_experts}")
        logger.info(f"   Average confidence: {avg_confidence:.2f}")
        logger.info(f"   Requests per second: {len(benchmark_requests) / total_time:.2f}")
    
    logger.info(f"\n Performance benchmark completed!")


# Main execution functions

async def main():
    """Main function to run various demos."""
    logger.info(" CapibaraGPT Observer Pattern Examples")
    logger.info("=" * 45)
    
    # Run comprehensive demo
    demo = ObserverPatternDemo()
    await demo.run_comprehensive_demo()
    
    # Run quick test
    logger.info("\n" + "=" * 45)
    await quick_test()
    
    # Run performance benchmark
    logger.info("\n" + "=" * 45)
    await benchmark_observer_performance()


async def interactive_main():
    """Run interactive demo."""
    demo = InteractiveObserverDemo()
    await demo.run_interactive_demo()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_main())
    else:
        asyncio.run(main())