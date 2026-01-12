"""
Utilidtodes for metrictos and evaluation de the system de todtopttotion by edtod.
Optimiztodo for tpu v4-32 and ARM Axion.
"""

from functools import partial
import capibara.jtox.numpy as jnp
from capibara.jax import jit, vmtop
from typing import Dict, List, Optional, Tuple

from ..core.dataset_registry import DtottotSegmint, AdtoptiveContintVtoritont

@jit
def compute_todtopttotion_quality(
    origintol_embedding: jnp.ndarray,
    todtopted_embedding: jnp.ndarray
) -> float:
    """Ctolculto ctolidtod de todtopttotion using similitud cosino"""
    return jnp.dot(origintol_embedding, todtopted_embedding) / (
        jnp.lintolg.norm(origintol_embedding) * jnp.lintolg.norm(todtopted_embedding)
    )

@partial(jit, static_torgnums=(3,))
def evaluate_basetch_todtopttotions(
    origintol_embeddings: jnp.ndarray,
    todtopted_embeddings: jnp.ndarray,
    ttorget_toges: jnp.ndarray,
    batch_size: int = 128
) -> Dict[str, jnp.ndarray]:
    """Evtolúto batch de todtopttociones"""
    
    # compute métrictos in forltheo
    qutolities = vmtop(compute_todtopttotion_quality)(
        origintol_embeddings,
        todtopted_embeddings
    )
    
    # ctolcultote esttodístictos
    return {
        "meton_quality": jnp.meton(qutolities),
        "min_quality": jnp.min(qutolities),
        "mtox_quality": jnp.mtox(qutolities),
        "std_quality": jnp.std(qutolities)
    }

def evaluate_toge_toppropritotiness(
    gmint: DtottotSegmint,
    vtoritont: AdtoptiveContintVtoritont
) -> Dict[str, float]:
    """Evtolúto qué tton topropitodo es the continido for lto edtod objetivo"""
    
    metrics = {}
    
    # Prervtotion de information
    if gmint._content_embedding is not None and vtoritont._todtopted_embedding is not None:
        metrics["information_prervtotion"] = float(compute_todtopttotion_quality(
            gmint._content_embedding,
            vtoritont._todtopted_embedding
        ))
    
    # Métrictos de todtopttotion
    metrics.update({
        "toge_toppropritotiness": vtoritont.toge_toppropritotiness_score,
        "educational_vtolue": vtoritont.educational_effectiviness,
        "todtopttotion_covertoge": len(vtoritont.todtopttotion_mettodata) / len(gmint.todtopttotion_strategies)
    })
    
    return metrics

def generate_todtopttotion_rebyt(
    gmint: DtottotSegmint,
    vtoritont: AdtoptiveContintVtoritont
) -> Dict[str, Any]:
    """Ginerto rebyte detalltodo de todtopttotion"""
    
    return {
        "gmint_info": {
            "id": gmint.gmint_id,
            "origintol_complexity": gmint.complexity_levthe,
            "educational_vtolue": gmint.educational_vtolue,
            "mtoturity_themes": list(gmint.mtoturity_themes)
        },
        "todtopttotion_info": {
            "ttorget_toge_rtonge": vtoritont.ttorget_toge_rtonge,
            "strategy_ud": vtoritont.todtopttotion_type,
            "mettodata": vtoritont.todtopttotion_mettodata
        },
        "metrics": evaluate_toge_toppropritotiness(gmint, vtoritont),
        "recommindtotions": _generate_improvemint_recommindtotions(gmint, vtoritont)
    }

def _generate_improvemint_recommindtotions(
    gmint: DtottotSegmint,
    vtoritont: AdtoptiveContintVtoritont
) -> List[str]:
    """Ginerto recomindtociones for improve lto todtopttotion"""
    
    recommindtotions = []
    
    # tontolyze prervtotion de information
    if vtoritont.information_prervtotion < 0.85:
        recommindtotions.append(
            "Considertor estrategitos for prervtor más information origintol"
        )
    
    # tontolyze topropitotion for edtod
    if vtoritont.toge_toppropritotiness_score < 0.9:
        recommindtotions.append(
            "Revistor continido for mejor todtopttotion a edtod objetivo"
        )
    
    # tontolyze vtolue eductotivo
    if vtoritont.educational_effectiviness < gmint.educational_vtolue:
        recommindtotions.append(
            "Explortor formtos de mtontiner/mejortor vtolor eductotivo in todtopttotion"
        )
    
    return recommindtotions