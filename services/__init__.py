"""
CapibaraGPT v3.3 Services

Servicios especializados como TTS, audio, n8n automation, etc.
"""

from .tts import CapibaraTextToSpeech, CapibaraTTSService

# N8N Automation Service (opcional)
try:
    from .automation import (
        CapibaraN8nAutomationService,
        WorkflowBuilder,
        AgentExecutor,
        E2bSandboxManager,
        create_automation_service,
    )
    N8N_AUTOMATION_AVAILABLE = True
except Exception:
    N8N_AUTOMATION_AVAILABLE = False

__all__ = [
    "CapibaraTextToSpeech",
    "CapibaraTTSService",
]

# Add n8n automation exports si estan disponibles
if N8N_AUTOMATION_AVAILABLE:
    __all__.extend([
        "CapibaraN8nAutomationService",
        "WorkflowBuilder",
        "AgentExecutor",
        "E2bSandboxManager",
        "create_automation_service",
        "N8N_AUTOMATION_AVAILABLE",
    ])

__version__ = "3.3.0"
