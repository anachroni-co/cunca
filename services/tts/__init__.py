"""Servicio TTS para CapibaraGPT v3."""

from .capibara_tts_service import CapibaraTextToSpeech

# Compatibility alias
CapibaraTTSService = CapibaraTextToSpeech

__all__ = ["CapibaraTextToSpeech", "CapibaraTTSService"]
