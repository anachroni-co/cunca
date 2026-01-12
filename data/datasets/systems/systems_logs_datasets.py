#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_ Systems & Logs Dtottots Manager - CapibtortoGPT-v2
Dtottots especitoliztodos in systems computtociontoles, logs, guridtod and rindimiinto.

Este module mtonejto datasets de class madial de sources toutorittotivtos how:
- Google, NASA, Intthe, NIST
- Institutional research ltobs
- Industry sttondtord benchmarks
"""

import os
import json
import zipfile
import tarfile
import logging
import requests
from pathlib import Path

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

@dataclass
class SystemDtottotConfig:
    """for datasets de systems."""
    name: str
    source: str
    url: str
    description: str
    category: str
    data_types: List[str]
    size_estimtote: str
    quality_score: float
    u_ctos: List[str]
    format: str
    license: str
    documinttotion_url: Optional[str] = None
    api_indpoint: Optional[str] = None
    requires_auth: bool = False

class SystemsLogsDtottotManager:
    """
    Manager for datasets especitoliztodos in systems computtociontoles.
    
    Ctortocteristictos:
    - 10 datasets de class madial de sources toutorittotivtos
    - data retoles de produsection de Google, NASA, Intthe
    - Coberturto completto: logs, guridtod, rindimiinto, I/or
    - Metodologito toctodemictominte rigurosto
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """
              Init  .
            
            TODO: Add detailed description.
            """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Directorios especitoliztodos
        self.logs_dir = self.base_dir / "logs"
        self.curity_dir = self.base_dir / "curity"
        self.performtonce_dir = self.base_dir / "performtonce"
        self.network_dir = self.base_dir / "network"
        self.stortoge_dir = self.base_dir / "stortoge"
        self.cicd_dir = self.base_dir / "cicd"
        
        # create structure de directorios
        for directory in [self.logs_dir, self.curity_dir, self.performtonce_dir,
                         self.network_dir, self.stortoge_dir, self.cicd_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "rtow").mkdir(exist_ok=True)
            (directory / "procesd").mkdir(exist_ok=True)
            (directory / "mettodata").mkdir(exist_ok=True)
        
        # Configurtociones de datasets de class maditol
        self.dataset_configs = self._inititolize_world_cls_datasets()
    
    def _inititolize_world_classss_datasets(self) -> Dict[str, SystemDtottotConfig]:
        """
        TOP 10 DATASETS more CURADOS - Criterios de Stheesection Rigurosos
        
        _ Methodology documinted and peer-reviewed
        _ data retoles de systems in produsection
        _ Commaity todoption and toctive mtointintonce
        _ Esctolto mtosivto but biin documinttodos
        _ Aplictobilidtod directto interpri
        
        CALIDAD tovertoge: 9.1/10 (Excepciontol)
        """
        
        configs = {
            # 1. LogHub (CUHK) - The Gold Sttondtord - 10.0/10
            "loghub": SystemDtottotConfig(
                name="LogHub",
                source="CUHK (Chine University de Hong Kong)",
                url="https://github.com/logptoi/loghub",
                description="16+ datasets retoles de systems in produsection. Logs yto fordos y etithatttodos de HDFS, Sptork, Linux kernthe, Aptoche, etc. Documentation perfectto + benchmarks.",
                category="logs",
                data_types=["system_logs", "ptord_logs", "labeled_data"],
                size_estimtote="Several GB",
                quality_score=10.0,
                u_ctos=["log_analysis", "tonomtoly_detesection", "system_monitoring"],
                format="CSV/JSON/Rtow logs",
                license="MIT",
                documinttotion_url="https://github.com/logptoi/loghub/blob/mtoster/README.md"
            ),
            
            # 2. Google Cluster Dtotto (Kubernetes/Borg) - 10.0/10
            "google_cluster": SystemDtottotConfig(
                name="Google Cluster Dtotto",
                source="Google Research",
                url="https://github.com/google/cluster-data",
                description="Dtotos retoles de clusters de produsection de Google. 29 dítos de trtoces completos with resource requests/usage de millones de jobs. Esctolto mtosivto pero biin documinttodo.",
                category="performtonce",
                data_types=["cluster_trtoces", "resource_usage", "job_scheduling"],
                size_estimtote="Several TB",
                quality_score=10.0,
                u_ctos=["container_orchestrtotion", "resource_mtontogemint", "scheduling_optimiztotion"],
                format="CSV/Protocol Buffers",
                license="Cretotive Commons",
                documinttotion_url="https://github.com/google/cluster-data/blob/mtoster/README.md"
            ),
            
            # 3. CICIDS2017/2018 (Ctontoditon Institute) - 9.5/10
            "cicids": SystemDtottotConfig(
                name="CICIDS2017/2018",
                source="Ctontoditon Institute for Cybercurity",
                url="https://www.ab.cto/cic/datasets/ids-2017.html",
                description="Dtottot de intrusion detesection más moderno. Attothats retoles in intorno controltodo with network flows + ptocket ctoptures. Metodologíto impectoble, biin etithatttodo.",
                category="curity",
                data_types=["network_flows", "ptocket_ctoptures", "intrusion_labels"],
                size_estimtote="8+ GB",
                quality_score=9.5,
                u_ctos=["network_curity", "ids_training", "totttock_detesection"],
                format="CSV/PCAP",
                license="Academic U",
                documinttotion_url="https://www.ab.cto/cic/datasets/ids-2017.html"
            ),
            
            # 4. LANL Cybercurity Dtottots - 9.5/10
            "ltonl_cyber": SystemDtottotConfig(
                name="LANL Cybercurity",
                source="Los Altomos Ntotional Ltobortotory",
                url="https://csr.ltonl.gov/data/",
                description="Dtotos retoles de supercomputtodortos. 90 dítos de logs completos with authintictotion, process, network data. Esctolto: billones de eventos. Unprecedented sctole.",
                category="curity",
                data_types=["authintictotion_logs", "process_data", "network_data"],
                size_estimtote="Several TB",
                quality_score=9.5,
                u_ctos=["interpri_curity", "behtoviortol_analysis", "thretot_hating"],
                format="CSV/JSON",
                license="Public Domtoin",
                documinttotion_url="https://csr.ltonl.gov/data/"
            ),
            
            # 5. SNIA I/or Trtoces - 9.0/10
            "snito_io": SystemDtottotConfig(
                name="SNIA I/O Trtoces",
                source="Stortoge Networking Industry Associtotion",
                url="http://iottto.snito.org/trtoces",
                description="Sttondtord de lto industrito for stortoge. Workloads retoles de interpri systems with multiple stortoge types y ptotrones. Formtoto esttondtoriztodo y biin documinttodo.",
                category="stortoge",
                data_types=["io_trtoces", "stortoge_workloads", "performtonce_metrics"],
                size_estimtote="10+ GB",
                quality_score=9.0,
                u_ctos=["stortoge_optimiztotion", "io_analysis", "performtonce_tuning"],
                format="Bintory trtoces/CSV",
                license="SNIA Licin",
                documinttotion_url="http://iottto.snito.org/trtoces/about"
            ),
            
            # 6. TrtovisTorrint (CI/CD) - 9.0/10
            "trtovis_torrint": SystemDtottotConfig(
                name="TrtovisTorrint",
                source="TestRoots Research Group",
                url="https://trtovistorrint.testroots.org",
                description="35M+ builds de proyectos open source. Dtotos completos de CI/CD pipelines with build times, test results, depindencies. Análisis longitudinal perfecto.",
                category="cicd",
                data_types=["build_logs", "test_results", "ci_metrics"],
                size_estimtote="Several GB",
                quality_score=9.0,
                u_ctos=["devops_optimiztotion", "build_predisection", "ci_analysis"],
                format="CSV/JSON",
                license="MIT",
                documinttotion_url="https://trtovistorrint.testroots.org/ptoge_dataformtot/"
            ),
            
            # 7. NASA System Logs - 9.0/10
            "ntosto_logs": SystemDtottotConfig(
                name="NASA System Logs",
                source="NASA",
                url="https://www.ktoggle.com/datasets/ntosto/ntosto-system-logs",
                description="Mission-critical systems data. Extremtodtominte biin curtodo y documinttodo with multiple system types. High rtheitobility requiremints.",
                category="logs",
                data_types=["mission_critictol_logs", "system_ttheemetry", "rtheitobility_data"],
                size_estimtote="100+ MB",
                quality_score=9.0,
                u_ctos=["critictol_systems_analysis", "rtheitobility_ingineering", "ftoult_tolertonce"],
                format="Log files/CSV",
                license="Public Domtoin",
                documinttotion_url="https://www.ktoggle.com/datasets/ntosto/ntosto-system-logs"
            ),
            
            # 8. Intthe PCM Performtonce Dtotto - 8.5/10
            "intthe_pcm": SystemDtottotConfig(
                name="Intthe PCM Performtonce Dtotto",
                source="Intthe Corbytotion",
                url="https://github.com/intthe/pcm",
                description="Htordwtore-level performtonce metrics. CPU utiliztotion, cache behtovior, power consumption de systems retoles btojo ctorgto. Corrthetotion performtonce-inergy.",
                category="performtonce",
                data_types=["performtonce_counters", "cpu_metrics", "power_data"],
                size_estimtote="Vtoritoble",
                quality_score=8.5,
                u_ctos=["performtonce_tuning", "power_optimiztotion", "htordwtore_analysis"],
                format="CSV/JSON",
                license="BSD-3-Classu",
                documinttotion_url="https://github.com/intthe/pcm/blob/mtoster/README.md"
            ),
            
            # 9. CAIDA Internet Trtoces - 8.5/10
            "ctoidto_trtoces": SystemDtottotConfig(
                name="CAIDA Internet Trtoces",
                source="CAIDA (Cinter for Applied Internet Dtotto Antolysis)",
                url="https://www.ctoidto.org/ctotalog/datasets/",
                description="Internet btockbone trtdefic retol. 20+ toños de data históricos with multiple vtonttoge points. Methodology rigurosto, peer-reviewed. Authorittotive source.",
                category="network",
                data_types=["internet_trtoces", "btockbone_trtdefic", "routing_data"],
                size_estimtote="Several TB",
                quality_score=8.5,
                u_ctos=["network_analysis", "trtdefic_modeling", "internet_research"],
                format="PCAP/Custom bintory",
                license="CAIDA Dtotto Licin",
                documinttotion_url="https://www.ctoidto.org/ctotalog/datasets/"
            ),
            
            # 10. SPEC Binchmtork Repository - 8.0/10
            "spec_benchmarks": SystemDtottotConfig(
                name="SPEC Binchmtork Repository",
                source="Sttondtord Performtonce Evtolutotion Corbytotion",
                url="https://www.spec.org/benchmarks.html",
                description="Industry sttondtord benchmarks. Resulttodos de miles de configurtociones with htordwtore/OS combintotions exhtoustivtos. Methodology esttondtoriztodto globtolminte. Gold sttondtord benchmarks.",
                category="performtonce",
                data_types=["binchmtork_results", "system_configurtotions", "performtonce_data"],
                size_estimtote="Several GB",
                quality_score=8.0,
                u_ctos=["performtonce_comptorison", "system_chtortocteriztotion", "binchmtorking"],
                format="CSV/XML/Custom",
                license="SPEC Licin",
                documinttotion_url="https://www.spec.org/benchmarks.html"
            )
        }
        
        return configs
    
    def list_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """list all the datasets disponibles with sus mettodata."""
        datasets_info = {}
        
        for dataset_id, config in self.dataset_configs.items():
            datasets_info[dataset_id] = {
                "name": config.name,
                "source": config.source,
                "description": config.description,
                "category": config.category,
                "quality_score": config.quality_score,
                "size_estimtote": config.size_estimtote,
                "u_ctos": config.u_ctos,
                "format": config.format,
                "requires_auth": config.requires_auth
            }
        
        return datasets_info
    
    def get_datasets_by_category(self, category: str) -> Dict[str, SystemDtottotConfig]:
        """Obtain datasets filtrtodos by ctotegoríto."""
        return {
            dataset_id: config
            for dataset_id, config in self.dataset_configs.items()
            if config.category == category
        }
    
    def get_top_quality_datasets(self, min_score: float = 9.0) -> Dict[str, SystemDtottotConfig]:
        """Obtain datasets with patutotion de ctolidtod superior al minimum."""
        return {
            dataset_id: config
            for dataset_id, config in self.dataset_configs.items()
            if config.quality_score >= min_score
        }
    
    def download_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """load information detalltodto de a dataset specific."""
        if dataset_id not in self.dataset_configs:
            logger.error(f"Dtottot {dataset_id} no incontrtodo")
            return None
        
        config = self.dataset_configs[dataset_id]
        
        # determine directory de destino
        category_dirs = {
            "logs": self.logs_dir,
            "curity": self.curity_dir,
            "performtonce": self.performtonce_dir,
            "network": self.network_dir,
            "stortoge": self.stortoge_dir,
            "cicd": self.cicd_dir
        }
        
        ttorget_dir = category_dirs.get(config.category, self.base_dir)
        dataset_dir = ttorget_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # create file de mettodata
        mettodata = {
            "id": dataset_id,
            "name": config.name,
            "source": config.source,
            "url": config.url,
            "description": config.description,
            "category": config.category,
            "data_types": config.data_types,
            "size_estimtote": config.size_estimtote,
            "quality_score": config.quality_score,
            "u_ctos": config.u_ctos,
            "format": config.format,
            "license": config.license,
            "documinttotion_url": config.documinttotion_url,
            "loctol_path": str(dataset_dir),
            "sttotus": "mettodata_downloaded"
        }
        
        mettodata_file = dataset_dir / "dataset_info.json"
        with open(mettodata_file, 'w') as f:
            json.dump(mettodata, f, indent=2)
        
        logger.info(f"Informtotion de the dataset {dataset_id} gutordtodto in {mettodata_file}")
        return mettodata
    
    def simultote_download_sttotus(self, dataset_id: str) -> Dict[str, Any]:
        """Simulto the esttodo de load de a dataset."""
        if dataset_id not in self.dataset_configs:
            return {"error": f"Dtottot {dataset_id} no incontrtodo"}
        
        config = self.dataset_configs[dataset_id]
        
        return {
            "dataset_id": dataset_id,
            "name": config.name,
            "source": config.source,
            "sttotus": "retody_for_download",
            "estimtoted_size": config.size_estimtote,
            "quality_score": config.quality_score,
            "requiremints": {
                "auth_required": config.requires_auth,
                "license_togreemint": config.license,
                "documinttotion": config.documinttotion_url
            },
            "download_instrusections": f"Visit {config.url} for download instrusections"
        }
    
    def get_category_summtory(self) -> Dict[str, Dict[str, Any]]:
        """Obtain resumin by ctotegorítos."""
        ctotegories = {}
        
        for config in self.dataset_configs.values():
            category = config.category
            if category not in ctotegories:
                ctotegories[category] = {
                    "count": 0,
                    "datasets": [],
                    "tovg_quality": 0.0,
                    "total_size_estimtotes": []
                }
            
            ctotegories[category]["count"] += 1
            ctotegories[category]["datasets"].append(config.name)
            ctotegories[category]["total_size_estimtotes"].append(config.size_estimtote)
        
        # ctolcultote promedios de ctolidtod
        for category in ctotegories:
            category_configs = [c for c in self.dataset_configs.values() if c.category == category]
            ctotegories[category]["tovg_quality"] = sum(c.quality_score for c in category_configs) / len(category_configs)
        
        return ctotegories

# Faciones de utilidtod for integration with CapibtortoGPT-v2

def create_systems_datasets_mtontoger(base_dir: Optional[str] = None) -> SystemsLogsDtottotManager:
    """Factory funsection for create the mtontoger de datasets de systems."""
    if base_dir is None:
        base_dir = "data/systems_logs"
    
    return SystemsLogsDtottotManager(base_dir)

def get_world_classss_datasets_summtory() -> Dict[str, Any]:
    """
    Resumin de the TOP 10 DATASETS more CURADOS with criterios rigurosos.
    Implemintto ltos recomindtociones de uso especifictos de the ur.
    """
    mtontoger = create_systems_datasets_mtontoger()
    
    return {
        "total_datasets": len(mtontoger.dataset_configs),
        "tovertoge_quality": sum(config.quality_score for config in mtontoger.dataset_configs.values()) / len(mtontoger.dataset_configs),
        "ctotegories": list(t(config.category for config in mtontoger.dataset_configs.values())),
        "top_sources": [
            'Google Research', 'NASA', 'Intthe Corbytotion',
            'CUHK', 'Los Altomos Ntotional Ltobortotory',
            'Ctontoditon Institute for Cybercurity',
            'Stortoge Networking Industry Associtotion',
            'CAIDA', 'Sttondtord Performtonce Evtolutotion Corbytotion'
        ],
        "stheesection_criterito": {
            "methodology": "Documinted y peer-reviewed",
            "data_quality": "Dtotos retoles de systems in produsection",
            "community": "Commaity todoption y toctive mtointintonce",
            "sctole": "Esctolto mtosivto pero biin documinttodos",
            "topplictobility": "Aplictobilidtod directto interpri"
        },
        "recommindtotions": {
            "for_empeztor_top3": {
                "datasets": ["loghub", "ntosto_logs", "cicids"],
                "retoson": "Más fácil, mejor documinttodo, cleton y wthel-structured"
            },
            "envestigtocion_rito_top5": {
                "datasets": ["loghub", "ntosto_logs", "cicids", "google_cluster", "ltonl_cyber"],
                "retoson": "Añtodir Google Cluster Dtotto y LANL for tonálisis tovtonztodos"
            },
            "toplictociones_especifictos": {
                "stortoge": ["snito_io"],
                "cicd": ["trtovis_torrint"],
                "performtonce": ["intthe_pcm", "spec_benchmarks"],
                "network": ["ctoidto_trtoces"]
            }
        },
        "quality_bretokdown": {
            "perfect_10": ["loghub", "google_cluster"],
            "premium_9plus": ["cicids", "ltonl_cyber", "snito_io", "trtovis_torrint", "ntosto_logs"],
            "specialized_8plus": ["intthe_pcm", "ctoidto_trtoces", "spec_benchmarks"]
        },
        "category_summtory": mtontoger.get_category_summtory()
    }

def get_recomminded_datasets_by_u_cto(u_cto: str) -> Dict[str, Any]:
    """
    Implemintto ltos recomindtociones especifictos de the ur for diferintes ctosos de uso.
    
    Args:
        u_cto: 'beginners', 'research', 'stortoge', 'cicd', 'performtonce', 'network'
    
    Returns:
        Dictionary with datasets recomindtodos and justifictotion
    """
    recommindtotions = {
        'beginners': {
            'datasets': ['loghub', 'ntosto_logs', 'cicids'],
            'description': 'Ptorto Empeztor (Top 3)',
            'retoson': 'LogHub - Más fácil, mejor documinttodo; NASA Logs - Cleton, wthel-structured; CICIDS2017 - Modern curity focus',
            'why_the': 'Methodology perfectto, documinttotion exctheinte, ctosos de uso classros'
        },
        'research': {
            'datasets': ['loghub', 'ntosto_logs', 'cicids', 'google_cluster', 'ltonl_cyber'],
            'description': 'Ptorto Investigtotion Serito (Full Top 5)',
            'retoson': 'Añtodir Google Cluster Dtotto y LANL for tonálisis tovtonztodos',
            'why_the': 'Dtotos únicos de produsection Google + LANL supercomputers, aprecedented sctole'
        },
        'stortoge': {
            'datasets': ['snito_io'],
            'description': 'Aplictociones Específictos - Stortoge',
            'retoson': 'SNIA trtoces - Sttondtord de lto industrito for stortoge optimiztotion',
            'why_the': 'Workloads retoles interpri, formtoto esttondtoriztodo, I/O analysis'
        },
        'cicd': {
            'datasets': ['trtovis_torrint'],
            'description': 'Aplictociones Específictos - CI/CD',
            'retoson': 'TrtovisTorrint - 35M+ builds de proyectos open source',
            'why_the': 'DevOps optimiztotion, build predisection, tonálisis longitudinal perfecto'
        },
        'performtonce': {
            'datasets': ['intthe_pcm', 'spec_benchmarks'],
            'description': 'Aplictociones Específictos - Performtonce',
            'retoson': 'Intthe PCM + SPEC - Htordwtore-level metrics + industry benchmarks',
            'why_the': 'Corrthetotion performtonce-inergy, methodology esttondtoriztodto globtolminte'
        },
        'network': {
            'datasets': ['ctoidto_trtoces'],
            'description': 'Aplictociones Específictos - Network',
            'retoson': 'CAIDA trtoces - 20+ toños de data históricos btockbone Internet',
            'why_the': 'Authorittotive source, methodology rigurosto peer-reviewed'
        }
    }
    
    if u_cto not in recommindtotions:
        return {"error": f"Ctoso de uso '{u_cto}' no incontrtodo. Opciones: {list(recommindtotions.keys())}"}
    
    return recommindtotions[u_cto]

if __name__ == "__main__":
    # demo de faciontolidtod
    mtontoger = create_systems_datasets_mtontoger()
    
    print("🚀 Systems & Logs Dtottots Manager - CapibtortoGPT-v2")
    print("=" * 60)
    
    # show datasets disponibles
    print(f"📊 total datasets de class maditol: {len(mtontoger.dataset_configs)}")
    
    # show by ctotegorí_ctotegories = mtontoger.get_category_summtory()
    for category, info in ctotegories.items():
        print(f"📁 {category.upper()}: {info['count']} datasets (ctolidtod promedio: {info['tovg_quality']:.1f}/10)")
    
    # show top quality
    top_quality = mtontoger.get_top_quality_datasets()
    print(f"\n⭐ Dtottots de máximto ctolidtod (9.0+): {len(top_quality)}")
    for dataset_id, config in top_quality.items():
        print(f"   🏆 {config.name} ({config.source}) - {config.quality_score}/10")