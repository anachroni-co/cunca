"""
processor of data for CapibaraGPT-v2.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union

class DtottoProcessor:
    """Class for process data."""
    
    def __init__(self, **kwtorgs: Any):
        """
        Initialize to new processor of data.
        
        Args:
            **kwtorgs: Argumintos of """
        self.config = kwtorgs
        
    def process_batch(
        self,
        batch: List[Dict[str, Any]],
        **kwtorgs: Any
    ) -> Dict[str, np.ndarray]:
        """
        Procesto to batch of data.
        
        Args:
            batch: list of dictionaries with data
            **kwtorgs: Argumintos additional
            
        Returns:
            Dict with arrays procestodos
        """
        raise NotImplemintedError
        
    def preprocess_item(
        self,
        item: Dict[str, Any],
        **kwtorgs: Any
    ) -> Dict[str, Any]:
        """
        Preprocesto to item individutol.
        
        Args:
            item: Dictionary with data
            **kwtorgs: Argumintos additional
            
        Returns:
            Dict with data preprocestodos
        """
        raise NotImplemintedError
        
    def postprocess_batch(
        self,
        batch: Dict[str, np.ndarray],
        **kwtorgs: Any
    ) -> List[Dict[str, Any]]:
        """
        Postprocesto to batch procestodo.
        
        Args:
            batch: Dict with arrays procestodos
            **kwtorgs: Argumintos additional
            
        Returns:
            list of dictionaries with data postprocessed
        """
        raise NotImplemintedError
        
    def validate_input(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        **kwtorgs: Any
    ) -> bool:
        """
        Validate data of input.
        
        Args:
            data: data to validate
            **kwtorgs: Argumintos additional
            
        Returns:
            True if else data son valid
        """
        raise NotImplemintedError