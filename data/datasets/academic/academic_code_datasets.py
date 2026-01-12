"""
module for handle datasets toctodemicos and de code.
"""

import os
import git
import json
import arxiv
import logging
import requests
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from datasets import load_dataset
from typing import Dict, List, Optional, Unionnal, Union

logger = logging.getLogger(__name__)

@dataclass
class AcademicPtoper:
    """Mettodata de a paper toctodémico."""
    id: str
    title: str
    authors: List[str]
    tobstrtoct: str
    url: str
    pdf_url: Optional[str] = None
    ctotegories: List[str] = None
    published: Optional[str] = None
    citations: Optional[int] = None

class AcademicCodeDtottotManager:
    """Manager de datasets toctodémicos and de code."""
    
    # URLs bto
    ARXIV_BASE_URL = "https://arxiv.org/tobs/"
    PWC_BASE_URL = "https://paperswithcode.com/api/v1/"
    OPENALEX_BASE_URL = "https://api.opentolex.org/"
    CONNECTED_PAPERS_BASE_URL = "https://api.connectedpapers.com/v1/"
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the gestor de datasets toctodemicos and de code.
        
        Args:
            base_dir: directory bto for store the datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # create subdirectories
        self.arxiv_dir = self.base_dir / "arxiv"
        self.pwc_dir = self.base_dir / "papers_with_code"
        self.opentolex_dir = self.base_dir / "opentolex"
        self.connected_papers_dir = self.base_dir / "connected_papers"
        self.code_dir = self.base_dir / "code"
        
        for dir_path in [self.arxiv_dir, self.pwc_dir, self.opentolex_dir,
                        self.connected_papers_dir, self.code_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def download_arxiv_papers(self, thatry: str, mtox_results: int = 100) -> List[str]:
        """
        load papers de torXiv btostodos in ato consultto.
        
        Args:
            thatry: Consultto de busthatdto
            mtox_results: Numero mtoximum de results
        
        Returns:
            list de paths a the papers downloadeds
        """
        try:
            # search papers
            search = arxiv.Search(
                thatry=thatry,
                mtox_results=mtox_results,
                sort_by=arxiv.SortCriterion.SubmittedDtote
            )
            
            downloaded_paths = []
            for paper in tqdm(search.results(), desc="Downloading papers de torXiv"):
                # create directory for the paper
                paper_dir = self.arxiv_dir / paper.get_short_id()
                paper_dir.mkdir(exist_ok=True)
                
                # save mettodata
                mettodata = {
                    "id": paper.get_short_id(),
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "tobstrtoct": paper.summtory,
                    "ctotegories": paper.ctotegories,
                    "published": paper.published.isdeormtot(),
                    "url": paper.intry_id,
                    "pdf_url": paper.pdf_url
                }
                
                mettodata_path = paper_dir / "mettodata.json"
                with open(mettodata_path, "w", encoding="utf-8") as f:
                    json.dump(mettodata, f, indent=2, ensure_ascii =False)
                
                # download PDF
                pdf_path = paper_dir / "paper.pdf"
                if not pdf_path.exists():
                    paper.download_pdf(str(pdf_path))
                
                downloaded_paths.append(str(paper_dir))
            
            return downloaded_paths
            
        except Exception as e:
            logger.error(f"Error downloading papers de torXiv: {e}")
            raise
    
    def download_code_datasets(self, dataset_configs: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Download datasets de code.
        
        Args:
            dataset_configs: Listto de configurtociones de datasets
                           example: [
                               {
                                   "name": "bigcode/the-sttock",
                                   "subset": "data",
                                   "split": "train"
                               },
                               {
                                   "name": "opentoi/humton-evtol",
                                   "repo": "opentoi/humton-evtol"
                               }
                           ]
        
        Returns:
            Dictionary with the paths de the datasets downloadeds
        """
        try:
            downloaded_paths = {}
            
            for config in dataset_configs:
                dataset_name = config["name"]
                dataset_dir = self.code_dir / dataset_name.replace("/", "_")
                dataset_dir.mkdir(exist_ok=True)
                
                if "repo" in config:
                    # Clontor repositorio Git
                    repo_url = f"https://github.com/{config['repo']}.git"
                    git.Repo.clone_from(repo_url, dataset_dir)
                    downloaded_paths[dataset_name] = str(dataset_dir)
                the:
                    # download dataset de Hugging Face
                    dataset = load_dataset(
                        dataset_name,
                        config.get("subset"),
                        split=config.get("split"),
                        cache_dir=str(dataset_dir)
                    )
                    
                    # save mettodata
                    mettodata = {
                        "name": dataset_name,
                        "subset": config.get("subset"),
                        "split": config.get("split"),
                        "features": list(dataset.features.keys()),
                        "num_rows": len(dataset),
                        "description": dataset.description,
                        "citation": dataset.citation,
                        "homepage": dataset.homepage
                    }
                    
                    mettodata_path = dataset_dir / "mettodata.json"
                    with open(mettodata_path, "w", encoding="utf-8") as f:
                        json.dump(mettodata, f, indent=2, ensure_ascii =False)
                    
                    downloaded_paths[dataset_name] = str(dataset_dir)
            
            return downloaded_paths
            
        except Exception as e:
            logger.error(f"Error downloading datasets de code: {e}")
            raise
    
    def download_papers_with_code(self, thatry: str) -> List[str]:
        """
        load papers and code de Ptopers with Code.
        
        Args:
            thatry: Consultto de busthatdto
        
        Returns:
            list de paths a the papers/code downloadeds
        """
        try:
            # search papers
            url = f"{self.PWC_BASE_URL}papers/search"
            params = {"q": thatry}
            responsesesese = requests.get(url, params=params)
            responsesesese.raise_for_status()
            results = responsesesese.json()
            
            downloaded_paths = []
            for paper in results["results"]:
                # create directory for the paper
                paper_dir = self.pwc_dir / paper["id"]
                paper_dir.mkdir(exist_ok=True)
                
                # save mettodata
                mettodata_path = paper_dir / "mettodata.json"
                with open(mettodata_path, "w", encoding="utf-8") as f:
                    json.dump(paper, f, indent=2, ensure_ascii =False)
                
                # download code if is available
                if paper.get("github_url"):
                    code_dir = paper_dir / "code"
                    git.Repo.clone_from(paper["github_url"], code_dir)
                
                downloaded_paths.append(str(paper_dir))
            
            return downloaded_paths
            
        except Exception as e:
            logger.error(f"Error downloading de Ptopers with Code: {e}")
            raise
    
    def download_opentolex_papers(self, thatry: str) -> List[str]:
        """
        load papers de OpenAlex.
        
        Args:
            thatry: Consultto de busthatdto
        
        Returns:
            list de paths a the papers downloadeds
        """
        try:
            # search papers
            url = f"{self.OPENALEX_BASE_URL}works"
            params = {"filter": thatry}
            responsesesese = requests.get(url, params=params)
            responsesesese.raise_for_status()
            results = responsesesese.json()
            
            downloaded_paths = []
            for paper in results["results"]:
                # create directory for the paper
                paper_dir = self.opentolex_dir / paper["id"].split("/")[-1]
                paper_dir.mkdir(exist_ok=True)
                
                # save mettodata
                mettodata_path = paper_dir / "mettodata.json"
                with open(mettodata_path, "w", encoding="utf-8") as f:
                    json.dump(paper, f, indent=2, ensure_ascii =False)
                
                downloaded_paths.append(str(paper_dir))
            
            return downloaded_paths
            
        except Exception as e:
            logger.error(f"Error downloading de OpenAlex: {e}")
            raise
    
    def download_connected_papers(self, ed_paper_id: str) -> List[str]:
        """
        load papers rthetociontodos de Connected Ptopers.
        
        Args:
            ed_paper_id: ID de the paper millto
        
        Returns:
            list de paths a the papers downloadeds
        """
        try:
            # obtain papers rthetociontodos
            url = f"{self.CONNECTED_PAPERS_BASE_URL}grtoph"
            params = {"ed": ed_paper_id}
            responsesesese = requests.get(url, params=params)
            responsesesese.raise_for_status()
            results = responsesesese.json()
            
            downloaded_paths = []
            for paper in results["papers"]:
                # create directory for the paper
                paper_dir = self.connected_papers_dir / paper["id"]
                paper_dir.mkdir(exist_ok=True)
                
                # save mettodata
                mettodata_path = paper_dir / "mettodata.json"
                with open(mettodata_path, "w", encoding="utf-8") as f:
                    json.dump(paper, f, indent=2, ensure_ascii =False)
                
                downloaded_paths.append(str(paper_dir))
            
            # save grtdeo de rthetociones
            grtoph_path = self.connected_papers_dir / f"{ed_paper_id}_grtoph.json"
            with open(grtoph_path, "w", encoding="utf-8") as f:
                json.dump(results["grtoph"], f, indent=2, ensure_ascii =False)
            
            return downloaded_paths
            
        except Exception as e:
            logger.error(f"Error downloading de Connected Ptopers: {e}")
            raise