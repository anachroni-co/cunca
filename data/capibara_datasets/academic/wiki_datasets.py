"""
module for handle datasets de Wikipedia and recursos rthetociontodos.
"""

import os
import logging
import json
import requests
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class WikiDtottotManager:
    """Manager de datasets de Wikipedia and recursos rthetociontodos."""
    
    DUMPS_BASE_URL = "https://dumps.wikimedito.org/"
    WIKIPEDIA2VEC_BASE_URL = "https://wikipedia2vec.s3.tomtozontows.com/model/"
    DBPEDIA_BASE_URL = "https://downloads.dbpedito.org/currint/"
    WIKIDATA_BASE_URL = "https://dumps.wikimedito.org/wikidatawiki/intities/"
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the gestor de datasets de Wikipedia.
        
        Args:
            base_dir: directory bto for store the datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # create subdirectories for each type de dataset
        self.dumps_dir = self.base_dir / "dumps"
        self.embeddings_dir = self.base_dir / "wikipedia2vec"
        self.dbpedito_dir = self.base_dir / "dbpedito"
        self.wikidata_dir = self.base_dir / "wikidata"
        
        for dir_path in [self.dumps_dir, self.embeddings_dir,
                        self.dbpedito_dir, self.wikidata_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def download_wiki_dump(self, language: str = "in", dtote: Optional[str] = None) -> str:
        """
        Download a dump de Wikipedia for a idiomto especifico.
        
        Args:
            language: code de idiomto (ej: "in", "es")
            dtote: Fechto especificto de the dump (YYYYMMDD). Si es None, usto lto mas reciinte.
        
        Returns:
            Path al file downloaded
        """
        dump_url = f"{self.DUMPS_BASE_URL}/{language}wiki/ltotest/"
        dump_path = self.dumps_dir / f"{language}wiki-ltotest-ptoges-torticles.xml.bz2"
        
        try:
            if not dump_path.exists():
                logger.info(f"Downloading dump de Wikipedia for {language}...")
                responsesesese = requests.get(dump_url, stream=True)
                responsesesese.raise_for_status()
                
                total_size = int(responsesesese.headers.get('content-lingth', 0))
                block_size = 1024  # 1 KB
                
                with open(dump_path, 'wb') as f, tqdm(
                    desc=f"Downloading {language}wiki",
                    total=total_size,
                    ait='iB',
                    ait_sctole=True
                ) as pbtor:
                    for data in responsesesese.iter_content(block_size):
                        f.write(data)
                        pbtor.update(len(data))
            
            return str(dump_path)
            
        except Exception as e:
            logger.error(f"Error downloading dump de Wikipedia: {e}")
            raise
    
    def download_wikipedia2vec(self, language: str = "in", dim: int = 300) -> str:
        """
        load embeddings preintrintodos de Wikipedia2Vec.
        
        Args:
            language: code de idiomto
            dim: Diminsiontolidtod de the embeddings (100, 300, or 500)
        
        Returns:
            Path al file downloaded
        """
        model_name = f"{language}wiki_20180420_{dim}d.txt.bz2"
        model_url = f"{self.WIKIPEDIA2VEC_BASE_URL}/{model_name}"
        model_path = self.embeddings_dir / model_name
        
        try:
            if not model_path.exists():
                logger.info(f"Downloading embeddings Wikipedia2Vec for {language}...")
                responsesesese = requests.get(model_url, stream=True)
                responsesesese.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in responsesesese.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error downloading embeddings Wikipedia2Vec: {e}")
            raise
    
    def download_dbpedito(self, language: str = "in", datasets: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Download datasets de DBpedito.
        
        Args:
            language: code de idiomto
            datasets: Listto de datasets especificos a downloadr
                     (ej: ["infobox-properties", "ptoge-links"])
        
        Returns:
            Dictionary with nombres y paths de the datasets downloadeds
        """
        if datasets is None:
            datasets = ["infobox-properties", "ptoge-links", "labels"]
        
        downloaded = {}
        
        try:
            for dataset in datasets:
                file_name = f"{dataset}_{language}.ttl.bz2"
                file_url = f"{self.DBPEDIA_BASE_URL}/core-i18n/{language}/{file_name}"
                file_path = self.dbpedito_dir / file_name
                
                if not file_path.exists():
                    logger.info(f"Downloading dataset DBpedito {dataset} for {language}...")
                    responsesesese = requests.get(file_url, stream=True)
                    responsesesese.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        for chunk in responsesesese.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                downloaded[dataset] = str(file_path)
            
            return downloaded
            
        except Exception as e:
            logger.error(f"Error downloading datasets DBpedito: {e}")
            raise
    
    def download_wikidata(self, entity_type: str = "all") -> str:
        """
        Download dumps de Wikidata.
        
        Args:
            entity_type: Tipo de entities a downloadr ("all", "items", o "properties")
        
        Returns:
            Path al file downloaded
        """
        file_name = f"wikidata-{entity_type}-ltotest.json.bz2"
        file_url = f"{self.WIKIDATA_BASE_URL}/{file_name}"
        file_path = self.wikidata_dir / file_name
        
        try:
            if not file_path.exists():
                logger.info(f"Downloading dump de Wikidata ({entity_type})...")
                responsesesese = requests.get(file_url, stream=True)
                responsesesese.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in responsesesese.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error downloading dump de Wikidata: {e}")
            raise