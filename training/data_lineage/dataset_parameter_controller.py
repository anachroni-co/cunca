#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎛️ Dataset Parameter Controller for CapibaraGPT-v2
===================================================

Advanced system for granular control of model parameters by dataset.
Allows enabling/disabling specific parameters that were influenced by 
particular datasets for compliance, debugging, and fine-tuning.

Features:
- Parameter masking by dataset
- Real-time parameter enable/disable
- Influence-based parameter mapping
- Compliance-driven parameter control
- Performance impact monitoring
"""

import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Set, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import time
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ParameterMask:
    """Mask for controlling which parameters are active."""
    dataset_id: str
    parameter_names: List[str]
    mask_values: Dict[str, jnp.ndarray]  # True = enabled, False = disabled
    influence_scores: Dict[str, float]
    created_at: float
    mask_type: str = "binary"  # "binary", "weighted", "gradient_based"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def apply_mask(self, parameters: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Apply the mask to model parameters."""
        masked_params = parameters.copy()
        
        for param_name in self.parameter_names:
            if param_name in parameters and param_name in self.mask_values:
                if self.mask_type == "binary":
                    # Binary masking: completely enable/disable
                    masked_params[param_name] = jnp.where(
                        self.mask_values[param_name],
                        parameters[param_name],
                        jnp.zeros_like(parameters[param_name])
                    )
                elif self.mask_type == "weighted":
                    # Weighted masking: scale by influence
                    influence = self.influence_scores.get(param_name, 1.0)
                    scale_factor = influence if self.mask_values[param_name].all() else 0.0
                    masked_params[param_name] = parameters[param_name] * scale_factor
                elif self.mask_type == "gradient_based":
                    # Gradient-based masking: preserve important gradients
                    masked_params[param_name] = jnp.where(
                        self.mask_values[param_name],
                        parameters[param_name],
                        parameters[param_name] * 0.1  # Reduce to 10% instead of zero
                    )
        
        return masked_params
    
    def get_active_parameter_count(self) -> int:
        """Get count of active parameters."""
        total_active = 0
        for param_name, mask in self.mask_values.items():
            total_active += int(jnp.sum(mask))
        return total_active
    
    def get_total_parameter_count(self) -> int:
        """Get total parameter count."""
        total_params = 0
        for param_name, mask in self.mask_values.items():
            total_params += mask.size
        return total_params

@dataclass
class DatasetControlPolicy:
    """Policy for controlling dataset-specific parameters."""
    policy_name: str
    enabled_datasets: Set[str]
    disabled_datasets: Set[str]
    mask_type: str = "binary"
    compliance_mode: bool = False
    auto_update: bool = True
    priority_order: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_dataset_enabled(self, dataset_id: str) -> bool:
        """Check if a dataset is enabled in this policy."""
        if dataset_id in self.disabled_datasets:
            return False
        if self.enabled_datasets and dataset_id not in self.enabled_datasets:
            return False
        return True
    
    def add_dataset(self, dataset_id: str, enabled: bool = True):
        """Add or update dataset in policy."""
        if enabled:
            self.enabled_datasets.add(dataset_id)
            self.disabled_datasets.discard(dataset_id)
        else:
            self.disabled_datasets.add(dataset_id)
            self.enabled_datasets.discard(dataset_id)

class DatasetParameterController:
    """
    Advanced controller for granular parameter management by dataset.
    
    Enables/disables model parameters based on their training data lineage,
    providing compliance and debugging capabilities.
    """
    
    def __init__(
        self,
        model_parameters: Dict[str, jnp.ndarray],
        lineage_file: Optional[Union[str, Path]] = None,
        control_dir: Union[str, Path] = "parameter_control"
    ):
        self.control_dir = Path(control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        
        # Model state
        self.original_parameters = model_parameters.copy()
        self.current_parameters = model_parameters.copy()
        
        # Control structures
        self.parameter_masks: Dict[str, ParameterMask] = {}
        self.dataset_policies: Dict[str, DatasetControlPolicy] = {}
        self.parameter_lineage: Dict[str, List[str]] = {}  # param -> datasets
        self.dataset_parameters: Dict[str, List[str]] = {}  # dataset -> params
        
        # Load lineage data if available
        if lineage_file:
            self._load_parameter_lineage(lineage_file)
        
        # Performance tracking
        self.performance_impact: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"🎛️ DatasetParameterController initialized")
        logger.info(f"📊 Managing {len(model_parameters)} parameter groups")
    
    def _load_parameter_lineage(self, lineage_file: Union[str, Path]):
        """Load parameter lineage from audit logs."""
        lineage_path = Path(lineage_file)
        
        if not lineage_path.exists():
            logger.warning(f"⚠️ Lineage file not found: {lineage_path}")
            return
        
        with open(lineage_path, 'r') as f:
            lineage_data = json.load(f)
        
        # Extract parameter-dataset mappings
        if "parameter_summary" in lineage_data:
            for param_name, param_info in lineage_data["parameter_summary"].items():
                self.parameter_lineage[param_name] = param_info["datasets_affecting"]
        
        # Build reverse mapping
        for param_name, datasets in self.parameter_lineage.items():
            for dataset_id in datasets:
                if dataset_id not in self.dataset_parameters:
                    self.dataset_parameters[dataset_id] = []
                if param_name not in self.dataset_parameters[dataset_id]:
                    self.dataset_parameters[dataset_id].append(param_name)
        
        logger.info(f"📚 Loaded lineage for {len(self.parameter_lineage)} parameters")
        logger.info(f"🗂️ Tracking {len(self.dataset_parameters)} datasets")
    
    def create_dataset_mask(
        self,
        dataset_id: str,
        enabled: bool = True,
        mask_type: str = "binary",
        influence_threshold: float = 0.1
    ) -> ParameterMask:
        """Create a parameter mask for a specific dataset."""
        
        if dataset_id not in self.dataset_parameters:
            logger.warning(f"⚠️ No parameters found for dataset {dataset_id}")
            return None
        
        parameter_names = self.dataset_parameters[dataset_id]
        mask_values = {}
        influence_scores = {}
        
        for param_name in parameter_names:
            if param_name in self.current_parameters:
                param_shape = self.current_parameters[param_name].shape
                
                if mask_type == "binary":
                    # Simple enable/disable
                    mask_values[param_name] = jnp.full(param_shape, enabled, dtype=bool)
                    influence_scores[param_name] = 1.0 if enabled else 0.0
                
                elif mask_type == "gradient_based":
                    # TODO: Implement gradient-based importance scoring
                    # For now, use random importance as placeholder
                    importance = np.random.random(param_shape)
                    mask_values[param_name] = importance > influence_threshold
                    influence_scores[param_name] = float(np.mean(importance))
                
                elif mask_type == "weighted":
                    # Weighted by estimated influence
                    # TODO: Implement actual influence calculation
                    influence = np.random.random()
                    mask_values[param_name] = jnp.full(param_shape, True, dtype=bool)
                    influence_scores[param_name] = influence
        
        mask = ParameterMask(
            dataset_id=dataset_id,
            parameter_names=parameter_names,
            mask_values=mask_values,
            influence_scores=influence_scores,
            created_at=time.time(),
            mask_type=mask_type,
            metadata={
                "total_parameters": sum(mask.size for mask in mask_values.values()),
                "enabled": enabled,
                "influence_threshold": influence_threshold
            }
        )
        
        self.parameter_masks[dataset_id] = mask
        logger.info(f"🎭 Created {mask_type} mask for dataset {dataset_id}")
        logger.info(f"📊 Mask covers {len(parameter_names)} parameter groups")
        
        return mask
    
    def enable_dataset_parameters(self, dataset_id: str, mask_type: str = "binary") -> bool:
        """Enable all parameters influenced by a specific dataset."""
        mask = self.create_dataset_mask(
            dataset_id=dataset_id,
            enabled=True,
            mask_type=mask_type
        )
        
        if mask:
            self._apply_mask(mask)
            logger.info(f"✅ Enabled parameters for dataset {dataset_id}")
            return True
        
        return False
    
    def disable_dataset_parameters(self, dataset_id: str, mask_type: str = "binary") -> bool:
        """Disable all parameters influenced by a specific dataset."""
        mask = self.create_dataset_mask(
            dataset_id=dataset_id,
            enabled=False,
            mask_type=mask_type
        )
        
        if mask:
            self._apply_mask(mask)
            logger.info(f"❌ Disabled parameters for dataset {dataset_id}")
            return True
        
        return False
    
    def _apply_mask(self, mask: ParameterMask):
        """Apply a parameter mask to current model parameters."""
        self.current_parameters = mask.apply_mask(self.current_parameters)
        
        # Track performance impact
        active_count = mask.get_active_parameter_count()
        total_count = mask.get_total_parameter_count()
        active_ratio = active_count / total_count if total_count > 0 else 0
        
        if mask.dataset_id not in self.performance_impact:
            self.performance_impact[mask.dataset_id] = {}
        
        self.performance_impact[mask.dataset_id].update({
            "active_parameters": active_count,
            "total_parameters": total_count,
            "active_ratio": active_ratio,
            "mask_type": mask.mask_type
        })
    
    def create_control_policy(
        self,
        policy_name: str,
        enabled_datasets: Optional[List[str]] = None,
        disabled_datasets: Optional[List[str]] = None,
        compliance_mode: bool = False
    ) -> DatasetControlPolicy:
        """Create a control policy for multiple datasets."""
        policy = DatasetControlPolicy(
            policy_name=policy_name,
            enabled_datasets=set(enabled_datasets or []),
            disabled_datasets=set(disabled_datasets or []),
            compliance_mode=compliance_mode,
            metadata={
                "created_at": time.time(),
                "total_datasets_tracked": len(self.dataset_parameters)
            }
        )
        
        self.dataset_policies[policy_name] = policy
        logger.info(f"📋 Created control policy: {policy_name}")
        
        return policy
    
    def apply_control_policy(self, policy_name: str) -> bool:
        """Apply a control policy to manage multiple datasets."""
        if policy_name not in self.dataset_policies:
            logger.error(f"❌ Policy not found: {policy_name}")
            return False
        
        policy = self.dataset_policies[policy_name]
        applied_count = 0
        
        # Apply policy to all tracked datasets
        for dataset_id in self.dataset_parameters.keys():
            if policy.is_dataset_enabled(dataset_id):
                if self.enable_dataset_parameters(dataset_id, mask_type="binary"):
                    applied_count += 1
            else:
                if self.disable_dataset_parameters(dataset_id, mask_type="binary"):
                    applied_count += 1
        
        logger.info(f"✅ Applied policy {policy_name} to {applied_count} datasets")
        return True
    
    def reset_parameters(self):
        """Reset all parameters to their original state."""
        self.current_parameters = self.original_parameters.copy()
        self.parameter_masks.clear()
        self.performance_impact.clear()
        logger.info("🔄 Reset all parameters to original state")
    
    def get_dataset_influence_report(self, dataset_id: str) -> Dict[str, Any]:
        """Generate influence report for a specific dataset."""
        if dataset_id not in self.dataset_parameters:
            return {"error": f"Dataset {dataset_id} not found"}
        
        parameter_names = self.dataset_parameters[dataset_id]
        total_params = 0
        active_params = 0
        
        for param_name in parameter_names:
            if param_name in self.current_parameters:
                param_count = self.current_parameters[param_name].size
                total_params += param_count
                
                # Check if parameter is currently active
                if dataset_id in self.parameter_masks:
                    mask = self.parameter_masks[dataset_id]
                    if param_name in mask.mask_values:
                        active_params += int(jnp.sum(mask.mask_values[param_name]))
                else:
                    active_params += param_count  # Assume active if no mask
        
        return {
            "dataset_id": dataset_id,
            "total_parameters": total_params,
            "active_parameters": active_params,
            "active_ratio": active_params / total_params if total_params > 0 else 0,
            "parameter_groups": len(parameter_names),
            "parameter_names": parameter_names,
            "has_mask": dataset_id in self.parameter_masks,
            "performance_impact": self.performance_impact.get(dataset_id, {})
        }
    
    def get_global_control_summary(self) -> Dict[str, Any]:
        """Get summary of global parameter control state."""
        total_datasets = len(self.dataset_parameters)
        active_datasets = len([d for d in self.dataset_parameters.keys() 
                             if d not in self.parameter_masks or 
                             self.parameter_masks[d].get_active_parameter_count() > 0])
        
        total_params = sum(param.size for param in self.original_parameters.values())
        active_params = sum(param.size for param in self.current_parameters.values() 
                          if not jnp.allclose(param, jnp.zeros_like(param)))
        
        return {
            "control_summary": {
                "total_datasets": total_datasets,
                "active_datasets": active_datasets,
                "disabled_datasets": total_datasets - active_datasets,
                "total_parameters": total_params,
                "active_parameters": active_params,
                "disabled_parameters": total_params - active_params,
                "active_ratio": active_params / total_params if total_params > 0 else 0
            },
            "policies": {
                name: {
                    "enabled_datasets": len(policy.enabled_datasets),
                    "disabled_datasets": len(policy.disabled_datasets),
                    "compliance_mode": policy.compliance_mode
                }
                for name, policy in self.dataset_policies.items()
            },
            "masks": {
                dataset_id: {
                    "mask_type": mask.mask_type,
                    "active_parameters": mask.get_active_parameter_count(),
                    "total_parameters": mask.get_total_parameter_count()
                }
                for dataset_id, mask in self.parameter_masks.items()
            }
        }
    
    def export_control_state(self, output_path: Union[str, Path]) -> Dict[str, Any]:
        """Export current control state for backup/restore."""
        output_path = Path(output_path)
        
        # Prepare export data (excluding JAX arrays)
        export_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_parameters": len(self.original_parameters),
                "total_datasets": len(self.dataset_parameters)
            },
            "parameter_lineage": self.parameter_lineage,
            "dataset_parameters": self.dataset_parameters,
            "policies": {
                name: {
                    "policy_name": policy.policy_name,
                    "enabled_datasets": list(policy.enabled_datasets),
                    "disabled_datasets": list(policy.disabled_datasets),
                    "mask_type": policy.mask_type,
                    "compliance_mode": policy.compliance_mode,
                    "metadata": policy.metadata
                }
                for name, policy in self.dataset_policies.items()
            },
            "performance_impact": self.performance_impact,
            "control_summary": self.get_global_control_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"💾 Control state exported to {output_path}")
        return export_data
    
    def load_control_state(self, input_path: Union[str, Path]) -> bool:
        """Load control state from backup."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error(f"❌ Control state file not found: {input_path}")
            return False
        
        try:
            with open(input_path, 'r') as f:
                state_data = json.load(f)
            
            self.parameter_lineage = state_data.get("parameter_lineage", {})
            self.dataset_parameters = state_data.get("dataset_parameters", {})
            self.performance_impact = state_data.get("performance_impact", {})
            
            # Rebuild policies
            self.dataset_policies.clear()
            for name, policy_data in state_data.get("policies", {}).items():
                policy = DatasetControlPolicy(
                    policy_name=policy_data["policy_name"],
                    enabled_datasets=set(policy_data["enabled_datasets"]),
                    disabled_datasets=set(policy_data["disabled_datasets"]),
                    mask_type=policy_data.get("mask_type", "binary"),
                    compliance_mode=policy_data.get("compliance_mode", False),
                    metadata=policy_data.get("metadata", {})
                )
                self.dataset_policies[name] = policy
            
            logger.info("✅ Control state loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load control state: {e}")
            return False

# Factory function
def create_dataset_parameter_controller(
    model_parameters: Dict[str, jnp.ndarray],
    lineage_file: Optional[str] = None
) -> DatasetParameterController:
    """Factory function to create dataset parameter controller."""
    return DatasetParameterController(model_parameters, lineage_file)