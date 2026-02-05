#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Blockchain + Smart Contracts Integration for CapibaraGPT-v2
===============================================================

Advanced hybrid system that combines:
- Blockchain audit logging (from data_lineage)
- Smart contracts automation (from utils)
- Training data traceability
- Automated compliance enforcement

This creates a powerful system where:
1. Blockchain tracks data lineage immutably
2. Smart contracts enforce compliance rules automatically
3. Training parameters are controlled by both systems
4. Violations trigger automatic remediation
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum

# Try to import from both systems
try:
    from .blockchain_audit_log import BlockchainAuditLog, ImmutableLogEntry
    from .dataset_parameter_controller import DatasetParameterController
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False

try:
    import sys
    sys.path.append('/workspace')
    from capibara.utils.smart_utils_contracts import (
        SmartContractsManager, SmartContract, ContractType, 
        ContractStatus, ViolationSeverity, ContractViolation
    )
    SMART_CONTRACTS_AVAILABLE = True
except ImportError:
    SMART_CONTRACTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ComplianceLevel(Enum):
    """Compliance levels for datasets."""
    OPEN_SOURCE = "open_source"
    RESEARCH_ONLY = "research_only" 
    COMMERCIAL = "commercial"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    CLASSIFIED = "classified"

@dataclass
class DatasetComplianceRule:
    """Compliance rule for a specific dataset."""
    dataset_id: str
    compliance_level: ComplianceLevel
    allowed_regions: List[str]
    allowed_uses: List[str]
    forbidden_uses: List[str]
    retention_policy_days: Optional[int] = None
    encryption_required: bool = False
    audit_frequency_hours: int = 24
    auto_disable_on_violation: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class TrainingDataComplianceContract(SmartContract):
    """
    Smart contract for training data compliance enforcement.
    
    Automatically monitors and enforces compliance rules for training datasets.
    """
    
    def __init__(
        self,
        contract_id: str,
        compliance_rules: List[DatasetComplianceRule],
        blockchain_audit: Optional[BlockchainAuditLog] = None,
        parameter_controller: Optional[DatasetParameterController] = None
    ):
        super().__init__(contract_id, ContractType.QUALITY)
        self.compliance_rules = {rule.dataset_id: rule for rule in compliance_rules}
        self.blockchain_audit = blockchain_audit
        self.parameter_controller = parameter_controller
        self.violation_history: List[ContractViolation] = []
        
        # Compliance tracking
        self.last_audit_check = {}
        self.dataset_status = {}
        
        logger.info(f" Training Data Compliance Contract initialized: {contract_id}")
        logger.info(f" Monitoring {len(compliance_rules)} compliance rules")
    
    def validate_conditions(self, context: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate training data compliance conditions."""
        violations = []
        
        # Check each dataset against its compliance rules
        for dataset_id, rule in self.compliance_rules.items():
            try:
                # Check audit frequency
                if self._is_audit_due(dataset_id, rule):
                    audit_result = self._audit_dataset_compliance(dataset_id, rule, context)
                    if not audit_result['compliant']:
                        violations.extend(audit_result['violations'])
                
                # Check parameter usage
                if self.parameter_controller:
                    param_report = self.parameter_controller.get_dataset_influence_report(dataset_id)
                    if self._check_parameter_compliance(dataset_id, rule, param_report):
                        violations.append(f"Parameter usage violation for {dataset_id}")
                
                # Check blockchain audit trail
                if self.blockchain_audit:
                    if self._check_audit_trail_compliance(dataset_id, rule):
                        violations.append(f"Audit trail compliance violation for {dataset_id}")
                        
            except Exception as e:
                violations.append(f"Error checking compliance for {dataset_id}: {str(e)}")
        
        is_compliant = len(violations) == 0
        return is_compliant, violations
    
    def execute_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute compliance enforcement actions."""
        actions_taken = []
        
        for dataset_id, rule in self.compliance_rules.items():
            if dataset_id in context.get('violated_datasets', []):
                
                # Auto-disable if required
                if rule.auto_disable_on_violation and self.parameter_controller:
                    success = self.parameter_controller.disable_dataset_parameters(dataset_id)
                    if success:
                        actions_taken.append(f"Auto-disabled parameters for {dataset_id}")
                        
                        # Log to blockchain
                        if self.blockchain_audit:
                            self.blockchain_audit.add_training_event(
                                dataset_id=dataset_id,
                                affected_parameters=[],
                                parameter_deltas={},
                                gradient_norms={},
                                data_hash=f"compliance_violation_{int(time.time())}",
                                data_size=0,
                                metadata={
                                    "event_type": "compliance_violation",
                                    "action": "auto_disable",
                                    "compliance_level": rule.compliance_level.value,
                                    "violation_timestamp": datetime.now().isoformat()
                                }
                            )
                
                # Create compliance policy
                if self.parameter_controller:
                    policy_name = f"emergency_compliance_{dataset_id}"
                    policy = self.parameter_controller.create_control_policy(
                        policy_name=policy_name,
                        disabled_datasets=[dataset_id],
                        compliance_mode=True
                    )
                    actions_taken.append(f"Created emergency compliance policy: {policy_name}")
        
        return {
            'success': len(actions_taken) > 0,
            'actions_taken': actions_taken,
            'timestamp': datetime.now().isoformat()
        }
    
    def _is_audit_due(self, dataset_id: str, rule: DatasetComplianceRule) -> bool:
        """Check if audit is due for a dataset."""
        last_audit = self.last_audit_check.get(dataset_id, 0)
        hours_since_last = (time.time() - last_audit) / 3600
        return hours_since_last >= rule.audit_frequency_hours
    
    def _audit_dataset_compliance(
        self, 
        dataset_id: str, 
        rule: DatasetComplianceRule, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Audit dataset compliance against rules."""
        violations = []
        
        # Check usage patterns
        current_use = context.get('dataset_usage', {}).get(dataset_id, 'unknown')
        if current_use in rule.forbidden_uses:
            violations.append(f"Forbidden use detected: {current_use}")
        
        if rule.allowed_uses and current_use not in rule.allowed_uses:
            violations.append(f"Use not in allowed list: {current_use}")
        
        # Check geographic restrictions
        current_region = context.get('processing_region', 'unknown')
        if rule.allowed_regions and current_region not in rule.allowed_regions:
            violations.append(f"Processing in disallowed region: {current_region}")
        
        # Check retention policy
        if rule.retention_policy_days:
            dataset_age_days = context.get('dataset_ages', {}).get(dataset_id, 0)
            if dataset_age_days > rule.retention_policy_days:
                violations.append(f"Dataset exceeds retention policy: {dataset_age_days} days")
        
        # Update audit timestamp
        self.last_audit_check[dataset_id] = time.time()
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'audit_timestamp': datetime.now().isoformat()
        }
    
    def _check_parameter_compliance(
        self, 
        dataset_id: str, 
        rule: DatasetComplianceRule,
        param_report: Dict[str, Any]
    ) -> bool:
        """Check if parameter usage complies with rules."""
        if param_report.get('error'):
            return False
        
        # High-security datasets should have minimal parameter influence
        if rule.compliance_level in [ComplianceLevel.CONFIDENTIAL, ComplianceLevel.CLASSIFIED]:
            if param_report.get('active_ratio', 0) > 0.1:  # Max 10% active
                return True  # Violation
        
        # Restricted datasets should be controllable
        if rule.compliance_level == ComplianceLevel.RESTRICTED:
            if not param_report.get('has_mask', False):
                return True  # Violation - should have control mask
        
        return False  # No violation
    
    def _check_audit_trail_compliance(
        self, 
        dataset_id: str, 
        rule: DatasetComplianceRule
    ) -> bool:
        """Check if audit trail complies with requirements."""
        if not self.blockchain_audit:
            return rule.compliance_level in [ComplianceLevel.CONFIDENTIAL, ComplianceLevel.CLASSIFIED]
        
        # Get dataset history from blockchain
        dataset_history = self.blockchain_audit.get_dataset_history(dataset_id)
        
        # High-security datasets require complete audit trail
        if rule.compliance_level in [ComplianceLevel.CONFIDENTIAL, ComplianceLevel.CLASSIFIED]:
            if len(dataset_history) == 0:
                return True  # Violation - no audit trail
        
        return False  # No violation

class BlockchainSmartContractsManager:
    """
    Hybrid manager that combines blockchain audit with smart contracts.
    
    Provides comprehensive training data governance with:
    - Immutable audit logging
    - Automated compliance enforcement
    - Real-time parameter control
    - Violation detection and response
    """
    
    def __init__(
        self,
        audit_log_dir: Union[str, Path] = "blockchain_audit",
        contracts_check_interval: int = 300,  # 5 minutes
        model_parameters: Optional[Dict[str, Any]] = None
    ):
        self.audit_log_dir = Path(audit_log_dir)
        self.audit_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.blockchain_audit = None
        self.parameter_controller = None
        self.smart_contracts_manager = None
        
        # Initialize available systems
        self._initialize_systems(model_parameters)
        
        # Compliance tracking
        self.compliance_rules: Dict[str, DatasetComplianceRule] = {}
        self.active_contracts: Dict[str, TrainingDataComplianceContract] = {}
        
        # Monitoring
        self.monitoring_active = False
        self.check_interval = contracts_check_interval
        
        logger.info(f" Blockchain + Smart Contracts Manager initialized")
        self._log_system_status()
    
    def _initialize_systems(self, model_parameters: Optional[Dict[str, Any]]):
        """Initialize available systems."""
        # Initialize blockchain audit
        if BLOCKCHAIN_AVAILABLE:
            try:
                from .blockchain_audit_log import create_blockchain_audit_log
                self.blockchain_audit = create_blockchain_audit_log(str(self.audit_log_dir))
                logger.info(" Blockchain audit system initialized")
            except Exception as e:
                logger.error(f" Failed to initialize blockchain audit: {e}")
        
        # Initialize parameter controller
        if model_parameters and BLOCKCHAIN_AVAILABLE:
            try:
                from .dataset_parameter_controller import create_dataset_parameter_controller
                # Export audit report for lineage
                if self.blockchain_audit:
                    audit_report_path = self.audit_log_dir / "lineage_report.json"
                    self.blockchain_audit.export_audit_report(audit_report_path)
                    self.parameter_controller = create_dataset_parameter_controller(
                        model_parameters, str(audit_report_path)
                    )
                    logger.info(" Parameter controller initialized")
            except Exception as e:
                logger.error(f" Failed to initialize parameter controller: {e}")
        
        # Initialize smart contracts manager
        if SMART_CONTRACTS_AVAILABLE:
            try:
                self.smart_contracts_manager = SmartContractsManager(
                    check_interval_seconds=self.check_interval
                )
                logger.info(" Smart contracts manager initialized")
            except Exception as e:
                logger.error(f" Failed to initialize smart contracts: {e}")
    
    def add_compliance_rule(self, rule: DatasetComplianceRule):
        """Add a compliance rule for a dataset."""
        self.compliance_rules[rule.dataset_id] = rule
        logger.info(f" Added compliance rule for {rule.dataset_id} (level: {rule.compliance_level.value})")
    
    def create_compliance_contract(
        self, 
        contract_id: str,
        dataset_rules: List[DatasetComplianceRule]
    ) -> bool:
        """Create a training data compliance contract."""
        if not self.smart_contracts_manager:
            logger.error(" Smart contracts manager not available")
            return False
        
        try:
            contract = TrainingDataComplianceContract(
                contract_id=contract_id,
                compliance_rules=dataset_rules,
                blockchain_audit=self.blockchain_audit,
                parameter_controller=self.parameter_controller
            )
            
            self.smart_contracts_manager.register_contract(contract)
            self.smart_contracts_manager.activate_contract(contract_id)
            self.active_contracts[contract_id] = contract
            
            logger.info(f" Created compliance contract: {contract_id}")
            return True
            
        except Exception as e:
            logger.error(f" Failed to create compliance contract: {e}")
            return False
    
    def log_training_event(
        self,
        dataset_id: str,
        affected_parameters: List[str],
        parameter_deltas: Dict[str, float],
        gradient_norms: Dict[str, float],
        data_hash: str,
        data_size: int,
        **kwargs
    ) -> Optional[str]:
        """Log training event with compliance checking."""
        if not self.blockchain_audit:
            logger.warning("️ Blockchain audit not available")
            return None
        
        # Add compliance metadata
        compliance_rule = self.compliance_rules.get(dataset_id)
        metadata = kwargs.get('metadata', {})
        if compliance_rule:
            metadata.update({
                'compliance_level': compliance_rule.compliance_level.value,
                'requires_audit': True,
                'encryption_required': compliance_rule.encryption_required
            })
        
        # Log to blockchain
        entry_id = self.blockchain_audit.add_training_event(
            dataset_id=dataset_id,
            affected_parameters=affected_parameters,
            parameter_deltas=parameter_deltas,
            gradient_norms=gradient_norms,
            data_hash=data_hash,
            data_size=data_size,
            metadata=metadata,
            **{k: v for k, v in kwargs.items() if k != 'metadata'}
        )
        
        # Trigger compliance check if needed
        if compliance_rule and compliance_rule.compliance_level in [
            ComplianceLevel.CONFIDENTIAL, ComplianceLevel.CLASSIFIED
        ]:
            asyncio.create_task(self._immediate_compliance_check(dataset_id))
        
        return entry_id
    
    async def _immediate_compliance_check(self, dataset_id: str):
        """Perform immediate compliance check for high-security datasets."""
        if not self.smart_contracts_manager:
            return
        
        context = {
            'triggered_by': 'training_event',
            'dataset_id': dataset_id,
            'timestamp': datetime.now().isoformat(),
            'processing_region': 'default',  # Should be detected dynamically
            'dataset_usage': {dataset_id: 'training'}
        }
        
        # Check all active compliance contracts
        for contract in self.active_contracts.values():
            try:
                is_valid, violations = contract.validate_conditions(context)
                if not is_valid:
                    logger.warning(f"️ Compliance violation detected: {violations}")
                    
                    # Execute remediation actions
                    context['violated_datasets'] = [dataset_id]
                    result = contract.execute_action(context)
                    
                    if result['success']:
                        logger.info(f" Remediation actions taken: {result['actions_taken']}")
                    
            except Exception as e:
                logger.error(f" Error in compliance check: {e}")
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get comprehensive compliance status."""
        status = {
            'system_status': {
                'blockchain_audit': self.blockchain_audit is not None,
                'parameter_controller': self.parameter_controller is not None,
                'smart_contracts': self.smart_contracts_manager is not None
            },
            'compliance_rules': len(self.compliance_rules),
            'active_contracts': len(self.active_contracts),
            'monitoring_active': self.monitoring_active
        }
        
        # Add blockchain stats
        if self.blockchain_audit:
            is_valid, issues = self.blockchain_audit.verify_blockchain_integrity()
            status['blockchain_integrity'] = {
                'valid': is_valid,
                'issues': issues,
                'total_blocks': len(self.blockchain_audit.blocks)
            }
        
        # Add parameter control stats
        if self.parameter_controller:
            control_summary = self.parameter_controller.get_global_control_summary()
            status['parameter_control'] = control_summary['control_summary']
        
        # Add smart contracts stats
        if self.smart_contracts_manager:
            status['smart_contracts_status'] = self.smart_contracts_manager.get_global_status()
        
        return status
    
    def _log_system_status(self):
        """Log current system status."""
        logger.info(" SYSTEM STATUS:")
        logger.info(f"    Blockchain Audit: {'' if self.blockchain_audit else ''}")
        logger.info(f"   ️ Parameter Controller: {'' if self.parameter_controller else ''}")
        logger.info(f"    Smart Contracts: {'' if self.smart_contracts_manager else ''}")

# Factory function
def create_hybrid_governance_system(
    model_parameters: Optional[Dict[str, Any]] = None,
    audit_dir: str = "hybrid_governance"
) -> BlockchainSmartContractsManager:
    """Factory function to create hybrid governance system."""
    return BlockchainSmartContractsManager(
        audit_log_dir=audit_dir,
        model_parameters=model_parameters
    )