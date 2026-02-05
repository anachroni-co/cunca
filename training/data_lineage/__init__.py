"""
Data Lineage & Traceability Module for CapibaraGPT-v2

Advanced system for tracking data influence on model parameters with:
- Blockchain-like immutable audit logs
- Parameter-to-dataset influence mapping
- Granular parameter control (enable/disable by dataset)
- Real-time lineage tracking during training
- Compliance-ready audit trails

Key Components:
- DataLineageTracker: Core tracking system
- ParameterInfluenceMapper: Maps datasets to specific parameters
- BlockchainAuditLog: Immutable audit trail
- DatasetParameterController: Enable/disable parameters by dataset
- ComplianceReporter: Generate audit reports
"""

from .data_lineage_tracker import (
    DataLineageTracker,
    DatasetInfluence,
    ParameterLineage,
    TrainingEvent
)

from .parameter_influence_mapper import (
    ParameterInfluenceMapper,
    InfluenceVector,
    DatasetParameterMapping
)

from .blockchain_audit_log import (
    BlockchainAuditLog,
    AuditBlock,
    DataProvenanceHash,
    ImmutableLogEntry
)

from .dataset_parameter_controller import (
    DatasetParameterController,
    ParameterMask,
    DatasetControlPolicy
)

from .inference_stage_parameter_controller import (
    InferenceStageParameterController,
    InferenceStageParameterMask,
    InferenceConfiguration,
    InferenceMode,
    MaskingStrategy,
    create_inference_stage_controller
)

from .blockchain_smart_contracts_integration import (
    BlockchainSmartContractsManager,
    TrainingDataComplianceContract,
    DatasetComplianceRule,
    ComplianceLevel,
    create_hybrid_governance_system
)

from .compliance_reporter import (
    ComplianceReporter,
    AuditReport,
    LineageReport,
    InfluenceReport
)

__all__ = [
    'DataLineageTracker',
    'ParameterInfluenceMapper',
    'BlockchainAuditLog',
    'DatasetParameterController',
    'InferenceStageParameterController',
    'BlockchainSmartContractsManager',
    'ComplianceReporter',
    'DatasetInfluence',
    'ParameterLineage',
    'TrainingEvent',
    'InfluenceVector',
    'DatasetParameterMapping',
    'AuditBlock',
    'DataProvenanceHash',
    'ImmutableLogEntry',
    'ParameterMask',
    'InferenceStageParameterMask',
    'InferenceConfiguration',
    'InferenceMode',
    'MaskingStrategy',
    'DatasetControlPolicy',
    'TrainingDataComplianceContract',
    'DatasetComplianceRule',
    'ComplianceLevel',
    'AuditReport',
    'LineageReport',
    'InfluenceReport',
    'create_inference_stage_controller',
    'create_hybrid_governance_system'
]
