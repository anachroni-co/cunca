"""
Integration Examples for CapibaraGPT-v2 Adapter System

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
        print("🚀 Initializing CapibaraGPT-v2 Adapter System...")
        
        try:
            # 1. Initialize main adapters
            self.adapters['kernel'] = KernelAbstractionAdapter()
            self.adapters['performance'] = PerformanceAdapter(OptimizationGoal.BALANCED)
            self.adapters['hardware'] = HardwareCompatibilityAdapter(OptimizationLevel.BALANCED)
            self.adapters['quantization'] = QuantizationAdapter()
            self.adapters['language'] = LanguageProcessingAdapter()

            # 2. Initialize each adapter
            for name, adapter in self.adapters.items():
                print(f"  📦 Initializing {name} adapter...")
                success = adapter.initialize()
                if success:
                    print(f"    ✅ {name} adapter initialized successfully")
                else:
                    print(f"    ❌ Error initializing {name} adapter")
                    return False

            # 3. Start metrics system
            print("  📊 Starting metrics system...")
            start_metrics_collection()

            # 4. Configure automatic monitoring
            self._setup_monitoring()

            self.initialized = True
            print("✅ Adapter system initialized completely\n")
            return True

        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            return False
    
    def _setup_monitoring(self):
        """Configure automatic monitoring."""
        def alert_handler(alert):
            level_emoji = {"warning": "⚠️", "error": "❌", "critical": "🚨"}
            emoji = level_emoji.get(alert.alert_level.value, "❓")
            print(f"{emoji} ALERTA: {alert.message}")
        
        metrics_collector.add_alert_callback(alert_handler)
    
    def demo_kernel_abstraction(self):
        """Demonstrate the use of the Kernel Abstraction Adapter."""
        print("🔧 === DEMO: KERNEL ABSTRACTION ADAPTER ===")

        if not self.initialized:
            print("❌ System not initialized")
            return

        kernel_adapter = self.adapters['kernel']

        # 1. Show available backends
        backends = kernel_adapter.get_available_backends()
        print(f"📋 Available backends: {list(backends.keys())}")
        
        # 2. Flash Attention Demo
        print("\n🧠 Executing Flash Attention...")
        
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
            
            print(f"  ✅ Flash Attention completed in {execution_time:.2f}ms")
            print(f"  📊 Result shape: {getattr(result, 'shape', 'N/A')}")
            
        except Exception as e:
            print(f"  ❌ Error in Flash Attention: {e}")

        # 3. Matrix Multiply Demo
        print("\n🔢 Executing Matrix Multiply...")
        
        a = np.random.randn(256, 512).astype(np.float32)
        b = np.random.randn(512, 256).astype(np.float32)
        
        try:
            start_time = time.time()
            result = kernel_adapter.matrix_multiply(a, b)
            execution_time = (time.time() - start_time) * 1000
            
            print(f"  ✅ Matrix Multiply completed in {execution_time:.2f}ms")
            print(f"  📊 Result shape: {getattr(result, 'shape', 'N/A')}")
            
        except Exception as e:
            print(f"  ❌ Error in Matrix Multiply: {e}")

        # 4. Show operation statistics
        stats = kernel_adapter.get_operation_stats()
        print(f"\n📈 Operation statistics:")
        print(f"  🔄 Cached operations: {stats['cached_operations']}")
        print(f"  🎯 Total backends: {stats['total_backends']}")

        print("✅ Kernel Abstraction Demo completed\n")
    
    def demo_performance_optimization(self):
        """Demonstrates the use of the Performance Adapter."""
        print("⚡ === DEMO: PERFORMANCE ADAPTER ===")

        if not self.initialized:
            print("❌ System not initialized")
            return

        performance_adapter = self.adapters['performance']

        # 1. Enable automatic adaptation
        print("🔄 Enabling automatic adaptation...")
        performance_adapter.enable_auto_adaptation()

        # 2. Simulate workload
        print("🏋️ Simulating workload...")
        
        @monitor_adapter_performance("DemoWorkload", "intensive_operation")
        def intensive_operation(size: int):
            # Simulate intensive operation
            data = np.random.randn(size, size)
            result = np.dot(data, data.T)
            time.sleep(0.1)  # Simulate processing
            return result

        # Execute operations with different workloads
        for i, size in enumerate([100, 200, 300, 500], 1):
            print(f"  🔄 Operation {i}: matrix {size}x{size}")
            try:
                result = intensive_operation(size)
                print(f"    ✅ Completed: shape {result.shape}")
            except Exception as e:
                print(f"    ❌ Error: {e}")

        # 3. Get performance report
        print("\n📊 Generating performance report...")
        report = performance_adapter.get_performance_report()

        print(f"📈 Current metrics:")
        for metric, value in report['current_metrics'].items():
            print(f"  {metric}: {value:.3f}")

        print(f"\n📊 Adaptation statistics:")
        stats = report['adaptation_stats']
        print(f"  Total adaptations: {stats['total_adaptations']}")
        print(f"  Successful adaptations: {stats['successful_adaptations']}")

        # 4. Change optimization goal
        print("\n🎯 Changing goal to MINIMIZE_LATENCY...")
        performance_adapter.set_optimization_goal(OptimizationGoal.MINIMIZE_LATENCY)

        print("✅ Performance Adapter Demo completed\n")
    
    def demo_hardware_compatibility(self):
        """Demonstrates the use of the Hardware Compatibility Adapter."""
        print("🖥️ === DEMO: HARDWARE COMPATIBILITY ADAPTER ===")

        if not self.initialized:
            print("❌ System not initialized")
            return

        hardware_adapter = self.adapters['hardware']

        # 1. Detect system hardware
        print("🔍 Detecting system hardware...")
        
        try:
            hardware_profile = hardware_adapter.force_hardware_detection()

            print(f"💻 Detected system: {hardware_profile['system_name']}")
            print(f"🏗️ Architecture: {hardware_profile['system_architecture']}")
            print(f"💾 Total memory: {hardware_profile['total_memory_gb']:.1f} GB")
            print(f"⚡ Total compute: {hardware_profile['total_compute_tflops']:.1f} TFLOPS")

            print(f"\n🔧 Detected components ({len(hardware_profile['capabilities'])}):")
            for cap in hardware_profile['capabilities'][:5]:  # Show first 5
                print(f"  • {cap['name']}: {cap['hardware_type']}")
                if cap.get('memory_gb', 0) > 0:
                    print(f"    💾 {cap['memory_gb']:.1f} GB")
                if cap.get('peak_performance_tflops', 0) > 0:
                    print(f"    ⚡ {cap['peak_performance_tflops']:.1f} TFLOPS")

        except Exception as e:
            print(f"❌ Error detecting hardware: {e}")
            return
        
        # 2. Apply optimizations
        print("\n🔧 Applying hardware optimizations...")

        try:
            optimizations = hardware_adapter.execute("optimize")
            applied = optimizations.get('applied_optimizations', [])

            if applied:
                print(f"✅ {len(applied)} optimizations applied:")
                for opt in applied:
                    print(f"  • {opt['type']}: {opt['parameter']} = {opt['value']}")
                    print(f"    📈 Expected improvement: {opt['expected_improvement']:.1f}%")
            else:
                print("ℹ️ No additional optimizations applied")

        except Exception as e:
            print(f"❌ Error applying optimizations: {e}")
        
        # 3. Get system summary
        print("\n📊 System summary:")
        summary = hardware_adapter.get_hardware_summary()

        print(f"🔧 Total components: {summary['total_components']}")
        print(f"💾 Available memory: {summary['total_memory_gb']:.1f} GB")
        print(f"⚡ Compute power: {summary['total_compute_tflops']:.1f} TFLOPS")
        print(f"🎯 Optimization opportunities: {summary['optimization_opportunities']}")

        print(f"\n🏷️ Detected hardware types:")
        for hw_type, count in summary['hardware_types'].items():
            print(f"  • {hw_type}: {count}")

        print("✅ Hardware Compatibility Demo completed\n")
    
    def demo_quantization_methods(self):
        """Demonstrates the use of the Quantization Adapter."""
        print("🗜️ === DEMO: QUANTIZATION ADAPTER ===")

        if not self.initialized:
            print("❌ System not initialized")
            return

        quantization_adapter = self.adapters['quantization']

        # 1. Create test data
        print("📊 Preparing test data...")
        test_data = np.random.randn(1000, 512).astype(np.float32)
        original_size = test_data.nbytes / (1024 * 1024)  # MB
        print(f"  📏 Original size: {original_size:.2f} MB")

        # 2. Benchmark available methods
        print("\n🏃 Running quantization methods benchmark...")
        
        try:
            benchmark_results = quantization_adapter.benchmark(test_data)

            print("📈 Benchmark results:")
            for method, metrics in benchmark_results['benchmark_results'].items():
                if metrics.get('success', False):
                    print(f"\n  🔧 {method.upper()}:")
                    print(f"    📦 Compression: {metrics['compression_ratio']:.1f}x")
                    print(f"    🎯 Accuracy: {metrics['accuracy_retention']:.1%}")
                    print(f"    ⏱️ Time: {metrics['execution_time_ms']:.1f}ms")
                    print(f"    💾 Savings: {metrics['memory_savings_mb']:.1f}MB")
                else:
                    print(f"  ❌ {method}: {metrics.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error in benchmark: {e}")
            return
        
        # 3. Demonstrate automatic selection
        print("\n🤖 Demonstrating automatic method selection...")

        for quality in [QuantizationQuality.HIGH_QUALITY, QuantizationQuality.BALANCED, QuantizationQuality.MAXIMUM_COMPRESSION]:
            print(f"\n  🎯 Quality: {quality.value}")

            try:
                result = quantization_adapter.quantize(test_data, method=None, quality=quality)

                method_used = result.metadata.get('method', 'unknown')
                print(f"    🔧 Selected method: {method_used}")
                print(f"    📦 Compression: {result.compression_ratio:.1f}x")
                print(f"    🎯 Accuracy: {result.accuracy_retention:.1%}")
                print(f"    ⏱️ Time: {result.quantization_time_ms:.1f}ms")

                # Dequantization test
                dequantized = quantization_adapter.dequantize(result.quantized_data)
                print(f"    ✅ Dequantization: {getattr(dequantized, 'shape', 'OK')}")

            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        # 4. Show available methods
        print("\n📋 Available methods:")
        methods_info = quantization_adapter.execute("get_methods")

        for method, info in methods_info['available_methods'].items():
            print(f"\n  🔧 {method.upper()}:")
            print(f"    ✅ Available: {info['available']}")
            print(f"    🎓 Calibrated: {info['calibrated']}")

            compressions = info['estimated_compression_ratios']
            print(f"    📦 Estimated compression:")
            print(f"      Small data: {compressions['small_data']:.1f}x")
            print(f"      Medium data: {compressions['medium_data']:.1f}x")
            print(f"      Large data: {compressions['large_data']:.1f}x")

        print("✅ Quantization Adapter Demo completed\n")
    
    def demo_language_processing(self):
        """Demonstrates the use of the Language Processing Adapter."""
        print("🌐 === DEMO: LANGUAGE PROCESSING ADAPTER ===")

        if not self.initialized:
            print("❌ System not initialized")
            return

        language_adapter = self.adapters['language']

        # 1. Advanced language detection
        print("🔍 Demonstrating advanced language detection...")
        
        test_texts = [
            "Hello, how are you today?",
            "Hola, ¿cómo estás hoy?",
            "Hello, como estas? 你好吗?",  # Code-switching
            "مرحبا، كيف حالك اليوم؟",
            "Bonjour, comment allez-vous?"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n  📝 Text {i}: '{text[:50]}...'")

            try:
                detection = language_adapter.detect_language(text)
                result = detection['detection_result']

                print(f"    🌍 Primary language: {result['primary_language']}")
                print(f"    🎯 Confidence: {result['confidence']:.2f}")
                print(f"    🔀 Multilingual: {result['is_multilingual']}")
                print(f"    🔄 Code-switching: {result['code_switching']}")

                if result['is_multilingual']:
                    print(f"    🌐 Detected languages: {result['languages_detected']}")

                print(f"    📊 Complexity: {result['complexity_score']:.2f}")

            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        # 2. Cultural adaptation
        print("\n🏛️ Demonstrating cultural adaptation...")
        
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
            print(f"\n  📝 Text: '{example['text']}'")
            print(f"  🌍 From: {example['source'].value}")
            print(f"  🎯 To: {example['target'].value}")

            try:
                adaptation = language_adapter.adapt_culturally(
                    example['text'],
                    example['source'],
                    example['target']
                )

                result = adaptation['adaptation_result']
                print(f"  ✨ Adapted: '{result['adapted_content']}'")
                print(f"  🔄 Changes: {len(result['changes_made'])}")

                for change in result['changes_made'][:2]:  # Show first 2
                    print(f"    • {change}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # 3. Complete multilingual processing
        print("\n🌐 Demonstrating complete multilingual processing...")
        
        multilingual_text = "Hello everyone! Hola a todos! 大家好！"
        
        context = MultilingualContext(
            primary_language="en",
            secondary_languages=["es", "zh"],
            processing_mode=ProcessingMode.MULTILINGUAL,
            cultural_adaptation_level=0.8
        )
        
        try:
            analysis = language_adapter.process_multilingual(multilingual_text, context)

            print(f"  📝 Text: '{multilingual_text}'")
            print(f"  🌍 Language analysis:")

            lang_detection = analysis['language_detection']
            print(f"    Primary language: {lang_detection['primary_language']}")
            print(f"    Multilingual: {lang_detection['is_multilingual']}")

            if analysis['code_switching_analysis']:
                cs_analysis = analysis['code_switching_analysis']
                print(f"    Code-switching detected: {cs_analysis['detected']}")
                if cs_analysis['detected']:
                    print(f"    Code-switching languages: {cs_analysis['languages']}")

            print(f"  🎯 Processing recommendations:")
            for rec in analysis['recommendations'][:3]:  # Show first 3
                print(f"    • {rec}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # 4. Show available language profiles
        print("\n📚 Available language profiles:")
        profiles = language_adapter.execute("get_profiles")

        print(f"  🌐 Total languages: {profiles['total_languages']}")
        print(f"  🏛️ Language families: {len(profiles['supported_families'])}")

        print(f"  📋 Some supported languages:")
        for lang_code, profile in list(profiles['language_profiles'].items())[:5]:
            print(f"    • {profile['language_name']} ({lang_code})")
            print(f"      Family: {profile['family']}")
            print(f"      Cultural context: {profile['cultural_context']}")

        print("✅ Language Processing Demo completed\n")
    
    def demo_metrics_system(self):
        """Demonstrates the automatic metrics system."""
        print("📊 === DEMO: AUTOMATIC METRICS SYSTEM ===")

        # 1. Get system overview
        print("🔍 Getting system overview...")
        overview = get_metrics_overview()

        print(f"📊 System status:")
        print(f"  🔧 Active adapters: {overview['total_adapters']}")
        print(f"  📈 Average score: {overview['system_performance']['average_system_score']:.2f}")
        print(f"  ⚠️ Total alerts: {overview['total_alerts']}")
        print(f"  🚨 Unacknowledged alerts: {overview['unacknowledged_alerts']}")
        print(f"  🔄 Total operations: {overview['system_performance']['total_operations']}")
        
        # 2. Status per adapter
        print(f"\n📋 Status per adapter:")
        status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}

        for name, info in overview['adapters_summary'].items():
            emoji = status_emoji.get(info['status'], "❓")
            print(f"  {emoji} {name}:")
            print(f"    📊 Score: {info['performance_score']:.2f}")
            print(f"    ✅ Success: {info['success_rate']:.1%}")
            print(f"    ⏱️ Average time: {info['avg_execution_time']:.1f}ms")
            print(f"    🔄 Operations: {info['total_operations']}")
        
        # 3. Get recent alerts
        print(f"\n🚨 Recent alerts:")
        alerts = metrics_collector.get_alerts(limit=5)

        if alerts:
            for alert in alerts:
                level_emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}
                emoji = level_emoji.get(alert['alert_level'], "❓")

                print(f"  {emoji} {alert['adapter_name']} - {alert['metric_type']}")
                print(f"    📅 {alert['datetime'][:19]}")
                print(f"    📊 Value: {alert['current_value']:.3f}")
                print(f"    ✅ Acknowledged: {'Yes' if alert['acknowledged'] else 'No'}")
        else:
            print("  ✅ No recent alerts")
        
        # 4. Detailed metrics for a specific adapter
        if self.adapters:
            adapter_name = list(self.adapters.keys())[0]
            print(f"\n🔍 Detailed metrics - {adapter_name}:")

            # Get specific metrics
            adapter_metrics = metrics_collector.get_adapter_metrics(f"{adapter_name.title()}Adapter")

            if adapter_metrics:
                print(f"  ⏰ Uptime: {adapter_metrics['uptime_seconds']:.0f}s")
                print(f"  📊 Performance score: {adapter_metrics['performance_score']:.2f}")
                print(f"  🔄 Total operations: {adapter_metrics['total_operations']}")

                print(f"  📈 Current metrics:")
                for metric, value in adapter_metrics['current_metrics'].items():
                    print(f"    {metric}: {value:.3f}")
            else:
                print("  ℹ️ Metrics not yet available")

        print("✅ Metrics System Demo completed\n")
    
    def run_complete_demo(self):
        """Executes the complete system demonstration."""
        print("🎬 === COMPLETE ADAPTER SYSTEM DEMONSTRATION ===\n")

        # Initialize system
        if not self.initialize_system():
            print("❌ Could not initialize the system")
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
                print(f"❌ Error in demo {demo_name}: {e}\n")

        # Final summary
        print("🎯 === FINAL SUMMARY ===")

        final_overview = get_metrics_overview()
        print(f"✅ System fully operational")
        print(f"📊 {final_overview['total_adapters']} adapters running")
        print(f"🎯 System average score: {final_overview['system_performance']['average_system_score']:.2f}")
        print(f"🔄 {final_overview['system_performance']['total_operations']} operations executed")

        print("\n🚀 The adapter system is ready for production use!")
        print("📚 Check the README.md for more information and advanced examples.")

# Specific usage examples

def example_kernel_integration():
    """Example of integration with existing kernels."""
    print("🔧 Example: Integration with existing kernels")

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
        print(f"✅ Flash attention executed: {result.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")

def example_performance_monitoring():
    """Performance monitoring example."""
    print("📊 Example: Performance monitoring")

    # Decorator for automatic monitoring
    @monitor_adapter_performance("ExampleWorkload", "data_processing")
    def process_data(data_size: int):
        # Simulate data processing
        data = np.random.randn(data_size, data_size)
        result = np.linalg.svd(data, compute_uv=False)
        return result

    # Execute monitored operations
    for size in [100, 200, 300]:
        try:
            result = process_data(size)
            print(f"✅ Processed matrix {size}x{size}: {len(result)} singular values")
        except Exception as e:
            print(f"❌ Error processing {size}x{size}: {e}")

def example_quantization_pipeline():
    """Quantization pipeline example."""
    print("🗜️ Example: Quantization pipeline")

    from .quantization_adapter import quantization_adapter

    # Initialize adapter
    if not quantization_adapter.get_status().value == "ready":
        quantization_adapter.initialize()

    # Simulate model weights
    model_weights = np.random.randn(1000, 768).astype(np.float32)
    print(f"📊 Original weights: {model_weights.nbytes / (1024*1024):.1f} MB")

    # Automatic quantization
    try:
        result = quantization_adapter.quantize(
            model_weights,
            quality=QuantizationQuality.BALANCED
        )

        print(f"🔧 Method used: {result.metadata.get('method', 'unknown')}")
        print(f"📦 Compression: {result.compression_ratio:.1f}x")
        print(f"🎯 Accuracy retained: {result.accuracy_retention:.1%}")
        print(f"💾 Savings: {result.memory_savings_mb:.1f} MB")

        # Dequantize to verify
        dequantized = quantization_adapter.dequantize(result.quantized_data)
        print(f"✅ Successful dequantization: {dequantized.shape}")

    except Exception as e:
        print(f"❌ Error in quantization: {e}")

# Main function to execute examples
def main():
    """Main function to execute the demonstration."""
    print("🎬 Starting CapibaraGPT-v2 Adapter System demonstration\n")

    # Create and run complete demo
    demo = AdapterSystemDemo()
    demo.run_complete_demo()

    print("\n" + "="*80)
    print("📚 ADDITIONAL EXAMPLES")
    print("="*80)

    # Execute specific examples
    examples = [
        ("Kernel Integration", example_kernel_integration),
        ("Performance Monitoring", example_performance_monitoring),
        ("Quantization Pipeline", example_quantization_pipeline)
    ]

    for example_name, example_func in examples:
        print(f"\n--- {example_name} ---")
        try:
            example_func()
        except Exception as e:
            print(f"❌ Error in example: {e}")

    print(f"\n🎉 Demonstration completed!")
    print(f"📖 For more information, check the README.md")

if __name__ == "__main__":
    main()