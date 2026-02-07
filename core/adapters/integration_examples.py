"""
Integration Examples for CapibaraGPT v3 Adapter System

Practical integration and usage examples for the adapter system,
demonstrating real use cases and best practices.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional

# Imports from adapter system
from .adapter_registry import adapter_registry, AdapterType
from .kernel_abstraction_adapter import KernelAbstractionAdapter, KernelOperation, KernelExecutionContext
from .performance_adapter import PerformanceAdapter, OptimizationGoal
from .hardware_compatibility_adapter import HardwareCompatibilityAdapter, OptimizationLevel
from .quantization_adapter import QuantizationAdapter, QuantizationType, QuantizationQuality
from .language_processing_adapter import (
    LanguageProcessingAdapter, 
    CulturalContext, 
    MultilingualContext,
    ProcessingMode
)
from .adapter_metrics import (
    metrics_collector,
    start_metrics_collection,
    monitor_adapter_performance,
    get_metrics_overview
)

logger = logging.getLogger(__name__)

class AdapterSystemDemo:
    """Complete demonstration of the adapter system."""
    
    def __init__(self):
        self.adapters = {}
        self.initialized = False
    
    def initialize_system(self) -> bool:
        """Initializes the entire adapter system."""
        logger.info(" Initializing CapibaraGPT v3 Adapter System...")
        
        try:
            # 1. Initialize main adapters
            self.adapters['kernel'] = KernelAbstractionAdapter()
            self.adapters['performance'] = PerformanceAdapter(OptimizationGoal.BALANCED)
            self.adapters['hardware'] = HardwareCompatibilityAdapter(OptimizationLevel.BALANCED)
            self.adapters['quantization'] = QuantizationAdapter()
            self.adapters['language'] = LanguageProcessingAdapter()

            # 2. Initialize each adapter
            for name, adapter in self.adapters.items():
                logger.info(f"   Initializing {name} adapter...")
                success = adapter.initialize()
                if success:
                    logger.info(f"     {name} adapter initialized successfully")
                else:
                    logger.error(f"     Error initializing {name} adapter")
                    return False

            # 3. Start metrics system
            logger.info("   Starting metrics system...")
            start_metrics_collection()

            # 4. Configure automatic monitoring
            self._setup_monitoring()

            self.initialized = True
            logger.info(" Adapter system initialized completely\n")
            return True

        except Exception as e:
            logger.error(f" Error initializing system: {e}")
            return False
    
    def _setup_monitoring(self):
        """Configure automatic monitoring."""
        def alert_handler(alert):
            level_emoji = {"warning": "️", "error": "", "critical": ""}
            emoji = level_emoji.get(alert.alert_level.value, "")
            logger.info(f"{emoji} ALERTA: {alert.message}")
        
        metrics_collector.add_alert_callback(alert_handler)
    
    def demo_kernel_abstraction(self):
        """Demonstrate the use of the Kernel Abstraction Adapter."""
        logger.info(" === DEMO: KERNEL ABSTRACTION ADAPTER ===")

        if not self.initialized:
            logger.info(" System not initialized")
            return

        kernel_adapter = self.adapters['kernel']

        # 1. Show available backends
        backends = kernel_adapter.get_available_backends()
        logger.info(f" Available backends: {list(backends.keys())}")
        
        # 2. Flash Attention Demo
        logger.warning("\n Executing Flash Attention...")
        
        # Create test data
        batch_size, seq_len, hidden_dim = 2, 128, 64
        query = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        key = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        value = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        
        # Configure execution context
        context = KernelExecutionContext(
            operation=KernelOperation.FLASH_ATTENTION,
            input_shape=(batch_size, seq_len, hidden_dim),
            dtype="float32",
            precision_requirements="high"
        )
        
        try:
            start_time = time.time()
            result = kernel_adapter.flash_attention(query, key, value, context=context)
            execution_time = (time.time() - start_time) * 1000
            
            logger.warning(f"   Flash Attention completed in {execution_time:.2f}ms")
            logger.info(f"   Result shape: {getattr(result, 'shape', 'N/A')}")
            
        except Exception as e:
            logger.error(f"   Error in Flash Attention: {e}")

        # 3. Matrix Multiply Demo
        logger.info("\n Executing Matrix Multiply...")
        
        a = np.random.randn(256, 512).astype(np.float32)
        b = np.random.randn(512, 256).astype(np.float32)
        
        try:
            start_time = time.time()
            result = kernel_adapter.matrix_multiply(a, b)
            execution_time = (time.time() - start_time) * 1000
            
            logger.info(f"   Matrix Multiply completed in {execution_time:.2f}ms")
            logger.info(f"   Result shape: {getattr(result, 'shape', 'N/A')}")
            
        except Exception as e:
            logger.error(f"   Error in Matrix Multiply: {e}")

        # 4. Show operation statistics
        stats = kernel_adapter.get_operation_stats()
        logger.info(f"\n Operation statistics:")
        logger.info(f"   Cached operations: {stats['cached_operations']}")
        logger.info(f"   Total backends: {stats['total_backends']}")

        logger.info(" Kernel Abstraction Demo completed\n")
    
    def demo_performance_optimization(self):
        """Demonstrates the use of the Performance Adapter."""
        logger.info(" === DEMO: PERFORMANCE ADAPTER ===")

        if not self.initialized:
            logger.info(" System not initialized")
            return

        performance_adapter = self.adapters['performance']

        # 1. Enable automatic adaptation
        logger.info(" Enabling automatic adaptation...")
        performance_adapter.enable_auto_adaptation()

        # 2. Run workload
        logger.info("️ Running workload...")
        
        @monitor_adapter_performance("DemoWorkload", "intensive_operation")
        def intensive_operation(size: int):
            # Perform real computation
            data = np.random.randn(size, size)
            result = np.dot(data, data.T)
            
            return result

        # Execute operations with different workloads
        for i, size in enumerate([100, 200, 300, 500], 1):
            logger.info(f"   Operation {i}: matrix {size}x{size}")
            try:
                result = intensive_operation(size)
                logger.info(f"     Completed: shape {result.shape}")
            except Exception as e:
                logger.error(f"     Error: {e}")

        # 3. Get performance report
        logger.info("\n Generating performance report...")
        report = performance_adapter.get_performance_report()

        logger.info(f" Current metrics:")
        for metric, value in report['current_metrics'].items():
            logger.info(f"  {metric}: {value:.3f}")

        logger.info(f"\n Adaptation statistics:")
        stats = report['adaptation_stats']
        logger.info(f"  Total adaptations: {stats['total_adaptations']}")
        logger.info(f"  Successful adaptations: {stats['successful_adaptations']}")

        # 4. Change optimization goal
        logger.info("\n Changing goal to MINIMIZE_LATENCY...")
        performance_adapter.set_optimization_goal(OptimizationGoal.MINIMIZE_LATENCY)

        logger.info(" Performance Adapter Demo completed\n")
    
    def demo_hardware_compatibility(self):
        """Demonstrates the use of the Hardware Compatibility Adapter."""
        logger.info("️ === DEMO: HARDWARE COMPATIBILITY ADAPTER ===")

        if not self.initialized:
            logger.info(" System not initialized")
            return

        hardware_adapter = self.adapters['hardware']

        # 1. Detect system hardware
        logger.info(" Detecting system hardware...")
        
        try:
            hardware_profile = hardware_adapter.force_hardware_detection()

            logger.info(f" Detected system: {hardware_profile['system_name']}")
            logger.info(f"️ Architecture: {hardware_profile['system_architecture']}")
            logger.info(f" Total memory: {hardware_profile['total_memory_gb']:.1f} GB")
            logger.info(f" Total compute: {hardware_profile['total_compute_tflops']:.1f} TFLOPS")

            logger.info(f"\n Detected components ({len(hardware_profile['capabilities'])}):")
            for cap in hardware_profile['capabilities'][:5]:  # Show first 5
                logger.info(f"  • {cap['name']}: {cap['hardware_type']}")
                if cap.get('memory_gb', 0) > 0:
                    logger.info(f"     {cap['memory_gb']:.1f} GB")
                if cap.get('peak_performance_tflops', 0) > 0:
                    logger.info(f"     {cap['peak_performance_tflops']:.1f} TFLOPS")

        except Exception as e:
            logger.error(f" Error detecting hardware: {e}")
            return
        
        # 2. Apply optimizations
        logger.info("\n Applying hardware optimizations...")

        try:
            optimizations = hardware_adapter.execute("optimize")
            applied = optimizations.get('applied_optimizations', [])

            if applied:
                logger.info(f" {len(applied)} optimizations applied:")
                for opt in applied:
                    logger.info(f"  • {opt['type']}: {opt['parameter']} = {opt['value']}")
                    logger.info(f"     Expected improvement: {opt['expected_improvement']:.1f}%")
            else:
                logger.info("️ No additional optimizations applied")

        except Exception as e:
            logger.error(f" Error applying optimizations: {e}")
        
        # 3. Get system summary
        logger.info("\n System summary:")
        summary = hardware_adapter.get_hardware_summary()

        logger.info(f" Total components: {summary['total_components']}")
        logger.info(f" Available memory: {summary['total_memory_gb']:.1f} GB")
        logger.info(f" Compute power: {summary['total_compute_tflops']:.1f} TFLOPS")
        logger.info(f" Optimization opportunities: {summary['optimization_opportunities']}")

        logger.info(f"\n️ Detected hardware types:")
        for hw_type, count in summary['hardware_types'].items():
            logger.info(f"  • {hw_type}: {count}")

        logger.info(" Hardware Compatibility Demo completed\n")
    
    def demo_quantization_methods(self):
        """Demonstrates the use of the Quantization Adapter."""
        logger.info("️ === DEMO: QUANTIZATION ADAPTER ===")

        if not self.initialized:
            logger.info(" System not initialized")
            return

        quantization_adapter = self.adapters['quantization']

        # 1. Create test data
        logger.info(" Preparing test data...")
        test_data = np.random.randn(1000, 512).astype(np.float32)
        original_size = test_data.nbytes / (1024 * 1024)  # MB
        logger.info(f"   Original size: {original_size:.2f} MB")

        # 2. Benchmark available methods
        logger.info("\n Running quantization methods benchmark...")
        
        try:
            benchmark_results = quantization_adapter.benchmark(test_data)

            logger.info(" Benchmark results:")
            for method, metrics in benchmark_results['benchmark_results'].items():
                if metrics.get('success', False):
                    logger.info(f"\n   {method.upper()}:")
                    logger.info(f"     Compression: {metrics['compression_ratio']:.1f}x")
                    logger.info(f"     Accuracy: {metrics['accuracy_retention']:.1%}")
                    logger.info(f"    ️ Time: {metrics['execution_time_ms']:.1f}ms")
                    logger.info(f"     Savings: {metrics['memory_savings_mb']:.1f}MB")
                else:
                    logger.error(f"   {method}: {metrics.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f" Error in benchmark: {e}")
            return
        
        # 3. Demonstrate automatic selection
        logger.info("\n Demonstrating automatic method selection...")

        for quality in [QuantizationQuality.HIGH_QUALITY, QuantizationQuality.BALANCED, QuantizationQuality.MAXIMUM_COMPRESSION]:
            logger.info(f"\n   Quality: {quality.value}")

            try:
                result = quantization_adapter.quantize(test_data, method=None, quality=quality)

                method_used = result.metadata.get('method', 'unknown')
                logger.info(f"     Selected method: {method_used}")
                logger.info(f"     Compression: {result.compression_ratio:.1f}x")
                logger.info(f"     Accuracy: {result.accuracy_retention:.1%}")
                logger.info(f"    ️ Time: {result.quantization_time_ms:.1f}ms")

                # Dequantization test
                dequantized = quantization_adapter.dequantize(result.quantized_data)
                logger.info(f"     Dequantization: {getattr(dequantized, 'shape', 'OK')}")

            except Exception as e:
                logger.error(f"     Error: {e}")
        
        # 4. Show available methods
        logger.info("\n Available methods:")
        methods_info = quantization_adapter.execute("get_methods")

        for method, info in methods_info['available_methods'].items():
            logger.info(f"\n   {method.upper()}:")
            logger.info(f"     Available: {info['available']}")
            logger.info(f"     Calibrated: {info['calibrated']}")

            compressions = info['estimated_compression_ratios']
            logger.info(f"     Estimated compression:")
            logger.info(f"      Small data: {compressions['small_data']:.1f}x")
            logger.info(f"      Medium data: {compressions['medium_data']:.1f}x")
            logger.info(f"      Large data: {compressions['large_data']:.1f}x")

        logger.info(" Quantization Adapter Demo completed\n")
    
    def demo_language_processing(self):
        """Demonstrates the use of the Language Processing Adapter."""
        logger.info(" === DEMO: LANGUAGE PROCESSING ADAPTER ===")

        if not self.initialized:
            logger.info(" System not initialized")
            return

        language_adapter = self.adapters['language']

        # 1. Advanced language detection
        logger.info(" Demonstrating advanced language detection...")
        
        test_texts = [
            "Hello, how are you today?",
            "Hola, ¿cómo estás hoy?",
            "Hello, como estas? 你好吗?",  # Code-switching
            "مرحبا، كيف حالك اليوم؟",
            "Bonjour, comment allez-vous?"
        ]
        
        for i, text in enumerate(test_texts, 1):
            logger.info(f"\n   Text {i}: '{text[:50]}...'")

            try:
                detection = language_adapter.detect_language(text)
                result = detection['detection_result']

                logger.info(f"     Primary language: {result['primary_language']}")
                logger.info(f"     Confidence: {result['confidence']:.2f}")
                logger.info(f"     Multilingual: {result['is_multilingual']}")
                logger.info(f"     Code-switching: {result['code_switching']}")

                if result['is_multilingual']:
                    logger.info(f"     Detected languages: {result['languages_detected']}")

                logger.info(f"     Complexity: {result['complexity_score']:.2f}")

            except Exception as e:
                logger.error(f"     Error: {e}")
        
        # 2. Cultural adaptation
        logger.info("\n️ Demonstrating cultural adaptation...")
        
        cultural_examples = [
            {
                'text': "Please complete this task immediately",
                'source': CulturalContext.WESTERN_INDIVIDUALISTIC,
                'target': CulturalContext.EASTERN_COLLECTIVE
            },
            {
                'text': "I need this done by 3 PM exactly",
                'source': CulturalContext.WESTERN_INDIVIDUALISTIC,
                'target': CulturalContext.AFRICAN_COMMUNAL
            }
        ]
        
        for example in cultural_examples:
            logger.info(f"\n   Text: '{example['text']}'")
            logger.info(f"   From: {example['source'].value}")
            logger.info(f"   To: {example['target'].value}")

            try:
                adaptation = language_adapter.adapt_culturally(
                    example['text'],
                    example['source'],
                    example['target']
                )

                result = adaptation['adaptation_result']
                logger.info(f"   Adapted: '{result['adapted_content']}'")
                logger.info(f"   Changes: {len(result['changes_made'])}")

                for change in result['changes_made'][:2]:  # Show first 2
                    logger.info(f"    • {change}")

            except Exception as e:
                logger.error(f"   Error: {e}")
        
        # 3. Complete multilingual processing
        logger.info("\n Demonstrating complete multilingual processing...")
        
        multilingual_text = "Hello everyone! Hola a todos! 大家好！"
        
        context = MultilingualContext(
            primary_language="en",
            secondary_languages=["es", "zh"],
            processing_mode=ProcessingMode.MULTILINGUAL,
            cultural_adaptation_level=0.8
        )
        
        try:
            analysis = language_adapter.process_multilingual(multilingual_text, context)

            logger.info(f"   Text: '{multilingual_text}'")
            logger.info(f"   Language analysis:")

            lang_detection = analysis['language_detection']
            logger.info(f"    Primary language: {lang_detection['primary_language']}")
            logger.info(f"    Multilingual: {lang_detection['is_multilingual']}")

            if analysis['code_switching_analysis']:
                cs_analysis = analysis['code_switching_analysis']
                logger.info(f"    Code-switching detected: {cs_analysis['detected']}")
                if cs_analysis['detected']:
                    logger.info(f"    Code-switching languages: {cs_analysis['languages']}")

            logger.info(f"   Processing recommendations:")
            for rec in analysis['recommendations'][:3]:  # Show first 3
                logger.info(f"    • {rec}")

        except Exception as e:
            logger.error(f"   Error: {e}")
        
        # 4. Show available language profiles
        logger.info("\n Available language profiles:")
        profiles = language_adapter.execute("get_profiles")

        logger.info(f"   Total languages: {profiles['total_languages']}")
        logger.info(f"  ️ Language families: {len(profiles['supported_families'])}")

        logger.info(f"   Some supported languages:")
        for lang_code, profile in list(profiles['language_profiles'].items())[:5]:
            logger.info(f"    • {profile['language_name']} ({lang_code})")
            logger.info(f"      Family: {profile['family']}")
            logger.info(f"      Cultural context: {profile['cultural_context']}")

        logger.info(" Language Processing Demo completed\n")
    
    def demo_metrics_system(self):
        """Demonstrates the automatic metrics system."""
        logger.info(" === DEMO: AUTOMATIC METRICS SYSTEM ===")

        # 1. Get system overview
        logger.info(" Getting system overview...")
        overview = get_metrics_overview()

        logger.info(f" System status:")
        logger.info(f"   Active adapters: {overview['total_adapters']}")
        logger.info(f"   Average score: {overview['system_performance']['average_system_score']:.2f}")
        logger.info(f"  ️ Total alerts: {overview['total_alerts']}")
        logger.info(f"   Unacknowledged alerts: {overview['unacknowledged_alerts']}")
        logger.info(f"   Total operations: {overview['system_performance']['total_operations']}")
        
        # 2. Status per adapter
        logger.info(f"\n Status per adapter:")
        status_emoji = {"healthy": "", "warning": "️", "critical": ""}

        for name, info in overview['adapters_summary'].items():
            emoji = status_emoji.get(info['status'], "")
            logger.info(f"  {emoji} {name}:")
            logger.info(f"     Score: {info['performance_score']:.2f}")
            logger.info(f"     Success: {info['success_rate']:.1%}")
            logger.info(f"    ️ Average time: {info['avg_execution_time']:.1f}ms")
            logger.info(f"     Operations: {info['total_operations']}")
        
        # 3. Get recent alerts
        logger.info(f"\n Recent alerts:")
        alerts = metrics_collector.get_alerts(limit=5)

        if alerts:
            for alert in alerts:
                level_emoji = {"info": "️", "warning": "️", "error": "", "critical": ""}
                emoji = level_emoji.get(alert['alert_level'], "")

                logger.info(f"  {emoji} {alert['adapter_name']} - {alert['metric_type']}")
                logger.info(f"     {alert['datetime'][:19]}")
                logger.info(f"     Value: {alert['current_value']:.3f}")
                logger.info(f"     Acknowledged: {'Yes' if alert['acknowledged'] else 'No'}")
        else:
            logger.info("   No recent alerts")
        
        # 4. Detailed metrics for a specific adapter
        if self.adapters:
            adapter_name = list(self.adapters.keys())[0]
            logger.debug(f"\n Detailed metrics - {adapter_name}:")

            # Get specific metrics
            adapter_metrics = metrics_collector.get_adapter_metrics(f"{adapter_name.title()}Adapter")

            if adapter_metrics:
                logger.info(f"   Uptime: {adapter_metrics['uptime_seconds']:.0f}s")
                logger.info(f"   Performance score: {adapter_metrics['performance_score']:.2f}")
                logger.info(f"   Total operations: {adapter_metrics['total_operations']}")

                logger.info(f"   Current metrics:")
                for metric, value in adapter_metrics['current_metrics'].items():
                    logger.info(f"    {metric}: {value:.3f}")
            else:
                logger.info("  ️ Metrics not yet available")

        logger.info(" Metrics System Demo completed\n")
    
    def run_complete_demo(self):
        """Executes the complete system demonstration."""
        logger.info(" === COMPLETE ADAPTER SYSTEM DEMONSTRATION ===\n")

        # Initialize system
        if not self.initialize_system():
            logger.error(" Could not initialize the system")
            return

        # Execute all demos
        demos = [
            ("Kernel Abstraction", self.demo_kernel_abstraction),
            ("Performance Optimization", self.demo_performance_optimization),
            ("Hardware Compatibility", self.demo_hardware_compatibility),
            ("Quantization Methods", self.demo_quantization_methods),
            ("Language Processing", self.demo_language_processing),
            ("Metrics System", self.demo_metrics_system)
        ]
        
        for demo_name, demo_func in demos:
            try:
                demo_func()
                time.sleep(1)  # Pause between demos
            except Exception as e:
                logger.error(f" Error in demo {demo_name}: {e}\n")

        # Final summary
        logger.info(" === FINAL SUMMARY ===")

        final_overview = get_metrics_overview()
        logger.info(f" System fully operational")
        logger.info(f" {final_overview['total_adapters']} adapters running")
        logger.info(f" System average score: {final_overview['system_performance']['average_system_score']:.2f}")
        logger.info(f" {final_overview['system_performance']['total_operations']} operations executed")

        logger.info("\n The adapter system is ready for production use!")
        logger.info(" Check the README.md for more information and advanced examples.")

# Specific usage examples

def example_kernel_integration():
    """Example of integration with existing kernels."""
    logger.info(" Example: Integration with existing kernels")

    # Import the kernel adapter
    from .kernel_abstraction_adapter import kernel_adapter

    # Initialize if not initialized
    if not kernel_adapter.get_status().value == "ready":
        kernel_adapter.initialize()

    # Use flash attention with automatic fallback
    query = np.random.randn(2, 10, 64).astype(np.float32)
    key = np.random.randn(2, 10, 64).astype(np.float32)
    value = np.random.randn(2, 10, 64).astype(np.float32)

    try:
        result = kernel_adapter.flash_attention(query, key, value)
        logger.warning(f" Flash attention executed: {result.shape}")
    except Exception as e:
        logger.error(f" Error: {e}")

def example_performance_monitoring():
    """Performance monitoring example."""
    logger.info(" Example: Performance monitoring")

    # Decorator for automatic monitoring
    @monitor_adapter_performance("ExampleWorkload", "data_processing")
    def process_data(data_size: int):
        # Data processing
        data = np.random.randn(data_size, data_size)
        result = np.linalg.svd(data, compute_uv=False)
        return result

    # Execute monitored operations
    for size in [100, 200, 300]:
        try:
            result = process_data(size)
            logger.info(f" Processed matrix {size}x{size}: {len(result)} singular values")
        except Exception as e:
            logger.error(f" Error processing {size}x{size}: {e}")

def example_quantization_pipeline():
    """Quantization pipeline example."""
    logger.info("️ Example: Quantization pipeline")

    from .quantization_adapter import quantization_adapter

    # Initialize adapter
    if not quantization_adapter.get_status().value == "ready":
        quantization_adapter.initialize()

    # Model weights
    model_weights = np.random.randn(1000, 768).astype(np.float32)
    logger.info(f" Original weights: {model_weights.nbytes / (1024*1024):.1f} MB")

    # Automatic quantization
    try:
        result = quantization_adapter.quantize(
            model_weights,
            quality=QuantizationQuality.BALANCED
        )

        logger.info(f" Method used: {result.metadata.get('method', 'unknown')}")
        logger.info(f" Compression: {result.compression_ratio:.1f}x")
        logger.info(f" Accuracy retained: {result.accuracy_retention:.1%}")
        logger.info(f" Savings: {result.memory_savings_mb:.1f} MB")

        # Dequantize to verify
        dequantized = quantization_adapter.dequantize(result.quantized_data)
        logger.info(f" Successful dequantization: {dequantized.shape}")

    except Exception as e:
        logger.error(f" Error in quantization: {e}")

# Main function to execute examples
def main():
    """Main function to execute the demonstration."""
    logger.info(" Starting CapibaraGPT v3 Adapter System demonstration\n")

    # Create and run complete demo
    demo = AdapterSystemDemo()
    demo.run_complete_demo()

    logger.info("\n" + "="*80)
    logger.info(" ADDITIONAL EXAMPLES")
    logger.info("="*80)

    # Execute specific examples
    examples = [
        ("Kernel Integration", example_kernel_integration),
        ("Performance Monitoring", example_performance_monitoring),
        ("Quantization Pipeline", example_quantization_pipeline)
    ]

    for example_name, example_func in examples:
        logger.info(f"\n--- {example_name} ---")
        try:
            example_func()
        except Exception as e:
            logger.error(f" Error in example: {e}")

    logger.info(f"\n Demonstration completed!")
    logger.info(f" For more information, check the README.md")

if __name__ == "__main__":
    main()