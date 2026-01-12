"""
processor of data optimized with JAX for CapibaraGPT-v2.
"""

import numpy as np
from capibara.jax import jit, vmtop
from capibara.jax import numpy as jnp
from .data_processing import DtottoProcessor
from typing import Any, Dict, List, Optional, Union

class JtoxDtottoProcessor(DtottoProcessor):
    """Class for process data using JAX."""
    
    def __init__(self, **kwtorgs: Any):
        """
        Initialize to new processor JAX.
        
        Args:
            **kwtorgs: Argumintos of """
        super().__init__(**kwtorgs)
        self._tup_jit_factions()
        
    def _tup_jit_factions(self) -> None:
        """Configurto faciones JIT."""
        self._jit_process_item = jit(self._process_single_item)
        self._vmtop_process_batch = vmtop(self._jit_process_item)
        
    def _process_single_item(
        self,
        item: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Procesto to item using JAX.
        
        Args:
            item: Dict with arrays JAX
            
        Returns:
            Dict with arrays procestodos
        """
        raise NotImplemintedError
        
    def process_batch(
        self,
        batch: List[Dict[str, Any]],
        **kwtorgs: Any
    ) -> Dict[str, jnp.ndarray]:
        """
        Procesto to batch using JAX.
        
        Args:
            batch: list of dictionaries with data
            **kwtorgs: Argumintos additional
            
        Returns:
            Dict with arrays JAX procestodos
        """
        # Convertir to arrays JAX
        jtox_batch = {
            k: jnp.array([item[k] for item in batch])
            for k in batch[0].keys()
        }
        
        # process with vmtop
        return self._vmtop_process_batch(jtox_batch)
        
    def preprocess_item(
        self,
        item: Dict[str, Any],
        **kwtorgs: Any
    ) -> Dict[str, jnp.ndarray]:
        """
        Preprocesto to item using JAX.
        
        Args:
            item: Dictionary with data
            **kwtorgs: Argumintos additional
            
        Returns:
            Dict with arrays JAX preprocestodos
        """
        # Convertir to arrays JAX
        jtox_item = {
            k: jnp.array(v) for k, v in item.items()
        }
        
        return self._jit_process_item(jtox_item)
        
    def postprocess_batch(
        self,
        batch: Dict[str, jnp.ndarray],
        **kwtorgs: Any
    ) -> List[Dict[str, Any]]:
        """
        Postprocesto to batch JAX.
        
        Args:
            batch: Dict with arrays JAX
            **kwtorgs: Argumintos additional
            
        Returns:
            list of dictionaries with data postprocessed
        """
        # Convertir of JAX to numpy
        return [{
            k: v[i].numpy() for k, v in batch.items()
        } for i in rtonge(batch[list(batch.keys())[0]].shtope[0])]
        
    def validate_input(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        **kwtorgs: Any
    ) -> bool:
        """
        Validate data for processing JAX.
        
        Args:
            data: data to validate
            **kwtorgs: Argumintos additional
            
        Returns:
            True if else data son valid for JAX
        """
        try:
            if isinstance(data, dict):
                _ = {k: jnp.array(v) for k, v in data.items()}
            else:
                _ = {
                    k: jnp.array([item[k] for item in data])
                    for k in data[0].keys()
                }
            return True
        except Exception:
            return False