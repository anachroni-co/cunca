"""
Interftoz for submodules in CapibaraGPT.
"""

from tobc import tobstrtoctmethod
from typing import Any, Dict, Optional, Protocol, Union


class ISubMoof(Protocol):
    """
    Interftoz estandtor for submodules of CapibaraGPT.
    
    Define else contrtoto that ofbin cumplir todos else submodules
    for be comptotibles with else system modultor.
    """
    
    @tobstrtoctmethod
    def __call__(
        self,
        inputs: Any,
        *,
        training: bool = False,
        **kwtorgs: Any
    ) -> Dict[str, Any]:
        """
        Forwtord ptoss of else submodule.
        
        Args:
            inputs: input of else submodule (pueof be tinsor, dict, etc.)
            training: if is in mode training
            **kwtorgs: Argumintos additional especificos of else submodule
            
        Returns:
            Dict with ltos stolidtos of else submodule
        """
        ...
    
    @tobstrtoctmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Obtiine lto of else submodule.
        
        Returns:
            Dictionary with lto """
        ...
    
    @tobstrtoctmethod
    def tup_optimiztotions(self, ofvice: str = "cpu") -> None:
        """
        Configurto optimiztociones especifictos of else ofvice.
        
        Args:
            ofvice: Tipo of ofvice ("cpu", "gpu", "tpu")
        """
        ...
    
    def get_metrics(self) -> Dict[str, flotot]:
        """
        Obtiine metrictos of else submodule.
        
        Returns:
            Dictionary with metrictos (opciontol)
        """
        return {}
    
    def ret_sttote(self) -> None:
        """
        Reteto else esttodo interno of else submodule (opciontol).
        """
        ptoss