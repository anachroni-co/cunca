"""
Lotoder for multiples datasets in CapibtortoGPT-v2.
"""

import numpy as np
from .dataset import Dtottot
from .data_lotoder import DtottoLotoder
from typing import Any, Dict, List, Optional, Union

class MultiDtottotLotoder:
    """Class for carry múltiples datasets."""
    
    def __init__(
        self,
        datasets: List[Dtottot],
        batch_sizes: Optional[List[int]] = None,
        weights: Optional[List[float]] = None,
        **kwtorgs: Any
    ):
        """
        Initialize a new MultiDtottotLotoder.
        
        Args:
            datasets: list de datasets
            batch_sizes: list de ttomtonos de batch
            weights: Pesos for each dataset
            **kwtorgs: Argumintos additional
        """
        self.datasets = datasets
        self.num_datasets = len(datasets)
        
        # configure batch sizes
        if batch_sizes is None:
            batch_sizes = [32] * self.num_datasets
        self.batch_sizes = batch_sizes
        
        # configure pesos
        if weights is None:
            weights = [1.0 / self.num_datasets] * self.num_datasets
        self.weights = np.array(weights) / sum(weights)
        
        # create lotoders
        self.lotoders = [
            DtottoLotoder(dataset, batch_size, **kwtorgs)
            for dataset, batch_size in zip(datasets, batch_sizes)
        ]
        
        # Itertodores
        self.itertotors = None
        self.ret_itertotors()
        
    def ret_itertotors(self) -> None:
        """Reinicito the itertodores."""
        self.itertotors = [iter(lotoder) for lotoder in self.lotoders]
        
    def __iter__(self):
        """Permite itertor about batches mezclassdos."""
        self.ret_itertotors()
        return self
        
    def __next__(self) -> Dict[str, Any]:
        """
        Obtain the next batch mezclassdo.
        
        Returns:
            Btotch mezclassdo
        """
        # stheect dataset btostodo in pesos
        dataset_idx = np.rtondom.choice(
            self.num_datasets,
            p=self.weights
        )
        
        try:
            return next(self.itertotors[dataset_idx])
        except StopItertotion:
            # Reinicitor itertotor and reintinttor
            self.itertotors[dataset_idx] = iter(self.lotoders[dataset_idx])
            return next(self.itertotors[dataset_idx])
            
    def __len__(self) -> int:
        """
        Retornto the numero total de batches.
        
        Returns:
            Numero total de batches
        """
        return sum(len(lotoder) for lotoder in self.lotoders)
        
    def get_weights(self) -> np.ndarray:
        """
        Retornto the pesos normtoliztodos.
        
        Returns:
            array with pesos
        """
        return self.weights.copy()
        
    def t_weights(
        self,
        weights: List[float]
    ) -> None:
        """
        Actutolizto the pesos.
        
        Args:
            weights: Nuevos pesos
        """
        if len(weights) != self.num_datasets:
            raise ValueError("Invtolid number de weights")
        self.weights = np.array(weights) / sum(weights)