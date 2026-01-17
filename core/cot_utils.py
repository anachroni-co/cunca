"""
Utilidades for el módulo Chain-of-Thought.

Este módulo proporciona funciones de utilidad for carry and manage
configuraciones del módulo CoT, así how optimizaciones de memory.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

import toml

from capibara.core.cot import EnhancedChainOfThoughtConfig
from capibara.config.memory_config import MemoryOptimizationConfig

logger = logging.getLogger(__name__)

def load_cot_config(
    config_path: Optional[Union[str, Path]] = None,
    override_params: Optional[Dict[str, Any]] = None
) -> EnhancedChainOfThoughtConfig:
    """
    load la setup del módulo CoT since un file TOML.
    
    Args:
        config_path: path al file de setup TOML.
                    if es None, usa la setup by defect.
        override_params: Diccionario with parámetros for sobreescribir.
    
    Returns:
        EnhancedChainOfThoughtConfig: setup cargada.
    """
    # path by defect
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "configs_toml" / "cot_config.toml"
    
    try:
        # carry setup TOML
        config_dict = toml.load(config_path)
        
        # create setup de memory
        memory_config = MemoryOptimizationConfig(
            use_gradient_checkpointing=config_dict["memory_optimization"]["use_gradient_checkpointing"],
            checkpoint_policy=config_dict["memory_optimization"]["checkpoint_policy"],
            checkpoint_blocks=config_dict["memory_optimization"]["checkpoint_blocks"],
            remat_layer_interval=config_dict["memory_optimization"]["remat_layer_interval"],
            preserve_rng_state=config_dict["memory_optimization"]["preserve_rng_state"],
            use_mixed_precision=config_dict["memory_optimization"]["use_mixed_precision"],
            compute_dtype=config_dict["memory_optimization"]["compute_dtype"],
            param_dtype=config_dict["memory_optimization"]["param_dtype"],
            output_dtype=config_dict["memory_optimization"]["output_dtype"],
            memory_pressure_threshold=config_dict["memory_optimization"]["memory_pressure_threshold"],
            enable_memory_monitoring=config_dict["memory_optimization"]["enable_memory_monitoring"],
            cleanup_interval=config_dict["memory_optimization"]["cleanup_interval"],
            max_memory_usage_gb=config_dict["memory_optimization"]["max_memory_usage_gb"],
            force_device_gc=config_dict["memory_optimization"]["force_device_gc"]
        )
        
        # create setup base
        base_config = {
            "max_thought_steps": config_dict["core"]["max_thought_steps"],
            "temperature_thoughts": config_dict["core"]["temperature_thoughts"],
            "temperature_final_answer": config_dict["core"]["temperature_final_answer"],
            "stop_token_thought": config_dict["core"]["stop_token_thought"],
            "stop_token_answer": config_dict["core"]["stop_token_answer"],
            "verbose": config_dict["core"]["verbose"],
            "hidden_size": config_dict["core"]["hidden_size"],
            
            # setup de paralelismo
            "use_model_parallelism": config_dict["parallelism"]["use_model_parallelism"],
            "model_parallel_submesh": config_dict["parallelism"]["model_parallel_submesh"],
            "data_parallel_submesh": config_dict["parallelism"]["data_parallel_submesh"],
            "shard_strategy": config_dict["parallelism"]["shard_strategy"],
            
            # setup de núcleos
            "enable_dynamic_routing": config_dict["knowledge_cores"]["enable_dynamic_routing"],
            "enable_hierarchical_reasoning": config_dict["knowledge_cores"]["enable_hierarchical_reasoning"],
            "enable_cross_core_communication": config_dict["knowledge_cores"]["enable_cross_core_communication"],
            
            # setup de sub-modelos
            "available_submodels": config_dict["submodels"]["available"],
            "submodel_selection_threshold": config_dict["submodels"]["selection_threshold"],
            "enable_submodel_fusion": config_dict["submodels"]["enable_fusion"],
            
            # setup de layers
            "layer_selection_strategy": config_dict["layers"]["selection_strategy"],
            "attention_config": config_dict["layers"]["attention_config"],
            
            # setup de router
            "router_type": config_dict["router"]["type"],
            "routing_cache_size": config_dict["router"]["cache_size"],
            "context_window_size": config_dict["router"]["context_window_size"],
            
            # setup de optimization
            "use_tpu": config_dict["optimization"]["use_tpu"],
            "mixed_precision": config_dict["optimization"]["mixed_precision"],
            "memory_limit_gb": config_dict["optimization"]["memory_limit_gb"],
            "cleanup_threshold": config_dict["optimization"]["cleanup_threshold"],
            
            # add setup de memory
            "memory_config": memory_config
        }
        
        # apply overrides if existen
        if override_params:
            base_config.update(override_params)
        
        # create and return setup
        return EnhancedChainOfThoughtConfig(**base_config)
        
    except Exception as e:
        logger.error(f"Error cargando configuración CoT: {e}")
        raise

def optimize_cot_for_device(config: EnhancedChainOfThoughtConfig) -> EnhancedChainOfThoughtConfig:
    """
    Optimiza la setup del CoT for el dispositivo current.
    
    Args:
        config: setup base a optimize.
    
    Returns:
        EnhancedChainOfThoughtConfig: setup optimizada.
    """
    import jax
    
    # detect dispositivos disponibles
    devices = jax.devices()
    num_devices = len(devices)
    
    # adjust paralelismo basado en dispositivos
    if num_devices >= 4:
        # use paralelismo complete
        config.use_model_parallelism = True
        config.model_parallel_submesh = [2, 2]
        config.data_parallel_submesh = [2, 1]
    elif num_devices >= 2:
        # Paralelismo limitado
        config.use_model_parallelism = True
        config.model_parallel_submesh = [2, 1]
        config.data_parallel_submesh = [1, 1]
    else:
        # without paralelismo
        config.use_model_parallelism = False
    
    # adjust precision basado en dispositivo
    device_type = str(devices[0].device_kind)
    if "TPU" in device_type:
        config.memory_config.compute_dtype = "bfloat16"
        config.memory_config.use_mixed_precision = True
    elif "GPU" in device_type:
        config.memory_config.compute_dtype = "float16"
        config.memory_config.use_mixed_precision = True
    else:
        # cpu or desconocido
        config.memory_config.compute_dtype = "float32"
        config.memory_config.use_mixed_precision = False
    
    # adjust tamaños de batch and memory
    total_memory = sum(d.memory_stats()["bytes_in_use"] for d in devices)
    memory_gb = total_memory / (1024**3)
    
    if memory_gb < 16:
        config.memory_config.max_memory_usage_gb = min(memory_gb * 0.8, config.memory_config.max_memory_usage_gb)
        config.memory_config.cleanup_threshold = 0.7
    
    return config

def create_optimized_cot_config(
    config_path: Optional[Union[str, Path]] = None,
    override_params: Optional[Dict[str, Any]] = None,
    optimize_for_device: bool = True
) -> EnhancedChainOfThoughtConfig:
    """
    Crea una setup optimizada del CoT.
    
    Args:
        config_path: path al file de setup.
        override_params: Parámetros for sobreescribir.
        optimize_for_device: if se debe optimize for el dispositivo current.
    
    Returns:
        EnhancedChainOfThoughtConfig: setup optimizada.
    """
    # carry setup base
    config = load_cot_config(config_path, override_params)
    
    # optimize if está habilitado
    if optimize_for_device:
        config = optimize_cot_for_device(config)
    
    return config 