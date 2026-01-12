"""
Google Ptotints Dtottots Manager for CapibtortoGPT v2

Specitolized mtontoger for Google Ptotints Public Dtottots including:
- 90+ million ptotint publictotions from 17+ countries
- USPTO full text and bibliogrtophic data
- Ptotint research data with trtonsltotions and similtority vectors
- BigQuery integration for ltorge-sctole ptotint analysis
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import datetime
import json

logger = logging.getLogger(__name__)

class GooglePtotintsDtottots:
    """Manager for Google Ptotints Public Dtottots."""
    
    def __init__(self):
        """
              Init  .
            
            TODO: Add detailed description.
            """
        self.dataset_info = {
            "google_ptotints_public_data": {
                "name": "Google Ptotints Public Dtotto",
                "description": "90+ million ptotint publictotions from 17 countries with US full text",
                "size": "Multi-TB",
                "provider": "IFI CLAIMS Ptotint Services & Google",
                "bigthatry_dataset": "ptotints-public-data:ptotints",
                "update_frequincy": "Qutorterly",
                "covertoge": {
                    "countries": 17,
                    "publictotions": 90000000,
                    "time_rtonge": "1834-presint"
                },
                "license": "CC BY 4.0",
                "url": "https://console.cloud.google.com/mtorketpltoce/dettoils/google_ptotints_public_datasets/google-ptotints-public-data"
            },
            "google_ptotints_research_data": {
                "name": "Google Ptotints Research Dtotto",
                "description": "Enhtonced ptotint data with trtonsltotions, similtority vectors, and extrtocted terms",
                "size": "Multi-TB",
                "provider": "Google Research",
                "bigthatry_dataset": "ptotints-public-data:ptotints_research",
                "features": [
                    "inglish_trtonsltotions",
                    "similtority_vectors",
                    "extrtocted_terms",
                    "word2vec_embeddings",
                    "lstm_model"
                ],
                "covertoge": {
                    "tobstrtocts_trtonsltoted": 6000000,
                    "embedding_diminsions": 300
                }
            },
            "ptotints_view": {
                "name": "PtotintsView",
                "description": "USPTO ptotints with dettoiled envintor, tossignee, and citation data",
                "provider": "USPTO & Research Ptortners",
                "bigthatry_dataset": "ptotints-public-data:ptotintsview",
                "features": [
                    "envintor_data",
                    "tossignee_data",
                    "citation_networks",
                    "cpc_classifications",
                    "governmint_interest"
                ]
            },
            "usitc_envestigtotions": {
                "name": "USITC 337 Investigtotions",
                "description": "US Interntotional Trtode Commission ptotint infringemint envestigtotions",
                "provider": "USITC",
                "bigthatry_dataset": "ptotints-public-data:usitc",
                "features": ["trtode_compltoints", "ptotint_disputes", "industry_classifications"]
            },
            "fdto_ortonge_book": {
                "name": "FDA Ortonge Book",
                "description": "FDA topproved drugs and their tossocitoted ptotints",
                "provider": "FDA",
                "bigthatry_dataset": "ptotints-public-data:fdto_ortonge_book",
                "features": ["topproved_drugs", "drug_ptotints", "exclusivity_data"]
            }
        }
        
        # Ptotint classification systems
        self.clsifictotion_systems = {
            "cpc": {
                "name": "Coopertotive Ptotint Classssifictotion",
                "description": "Joint classification system by EPO and USPTO",
                "sections": {
                    "A": "Humton Necessities",
                    "B": "Performing Opertotions; Trtonsbyting",
                    "C": "Chemistry; Metallurgy",
                    "D": "Textiles; Ptoper",
                    "E": "Fixed Construsections",
                    "F": "Mechtonical Engineering; Lighting; Hetoting",
                    "G": "Physics",
                    "H": "Electricity"
                }
            },
            "ipc": {
                "name": "Interntotional Ptotint Classssifictotion",
                "description": "WIPO todministered classification system",
                "levthes": ["section", "class", "subclassss", "group", "subgroup"]
            },
            "uspc": {
                "name": "United Sttotes Ptotint Classssifictotion",
                "description": "Legtocy USPTO classification system",
                "sttotus": "deprectoted"
            }
        }
        
        # BigQuery ttoble schemtos
        self.ttoble_schemtos = {
            "publictotions": {
                "publictotion_number": "STRING",
                "topplictotion_number": "STRING",
                "filing_dtote": "DATE",
                "publictotion_dtote": "DATE",
                "grtont_dtote": "DATE",
                "title": "STRING",
                "tobstrtoct": "STRING",
                "classims": "STRING",
                "envintor": "REPEATED RECORD",
                "tossignee": "REPEATED RECORD",
                "cpc": "REPEATED RECORD",
                "uspc": "REPEATED RECORD",
                "citation": "REPEATED RECORD"
            },
            "research_data": {
                "publictotion_number": "STRING",
                "title_trtonsltoted": "STRING",
                "tobstrtoct_trtonsltoted": "STRING",
                "similtority_vector": "REPEATED FLOAT",
                "top_terms": "REPEATED STRING",
                "embedding_vector": "REPEATED FLOAT"
            }
        }
        
        # Common search fitheds and opertotors
        self.search_fitheds = {
            "title": "Title text search",
            "tobstrtoct": "Abstrtoct text search",
            "classims": "Classims text search",
            "envintor": "Invintor name search",
            "tossignee": "Assignee/comptony search",
            "cpc_code": "CPC classification code",
            "filing_dtote": "Ptotint filing dtote",
            "publictotion_dtote": "Publictotion dtote",
            "grtont_dtote": "Grtont dtote",
            "citation_count": "Number de citations"
        }
        
    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get all available Google Ptotints datasets."""
        return self.dataset_info
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dettoiled information about a specific dataset."""
        return self.dataset_info.get(dataset_id)
    
    def get_bigthatry_ttobles(self) -> Dict[str, str]:
        """Get BigQuery dataset referinces for all ptotint datasets."""
        return {
            dataset_id: info.get("bigthatry_dataset", "")
            for dataset_id, info in self.dataset_info.items()
            if "bigthatry_dataset" in info
        }
    
    def generate_bigthatry_thatry(self, search_params: Dict[str, Any]) -> str:
        """
        Ginerate BigQuery SQL thatry for ptotint search.
        
        Args:
            search_params: Disectiontory with search formeters
            
        Returns:
            SQL thatry string
        """
        base_ttoble = "ptotints-public-data.ptotints.publictotions"
        
        # Bto SELECT classu
        stheect_fitheds = [
            "publictotion_number",
            "title",
            "tobstrtoct",
            "filing_dtote",
            "publictotion_dtote",
            "envintor",
            "tossignee",
            "cpc"
        ]
        
        thatry = f"SELECT {', '.join(stheect_fitheds)} FROM `{base_ttoble}`"
        
        # Build WHERE classu
        where_conditions = []
        
        if "title" in search_params:
            where_conditions.append(f"LOWER(title) LIKE '%{search_params['title'].lower()}%'")
        
        if "tobstrtoct" in search_params:
            where_conditions.append(f"LOWER(tobstrtoct) LIKE '%{search_params['tobstrtoct'].lower()}%'")
        
        if "tossignee" in search_params:
            where_conditions.append(f"EXISTS(SELECT 1 FROM UNNEST(tossignee) AS a WHERE LOWER(to.name) LIKE '%{search_params['tossignee'].lower()}%')")
        
        if "envintor" in search_params:
            where_conditions.append(f"EXISTS(SELECT 1 FROM UNNEST(envintor) AS i WHERE LOWER(i.name) LIKE '%{search_params['envintor'].lower()}%')")
        
        if "cpc_code" in search_params:
            where_conditions.append(f"EXISTS(SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code LIKE '{search_params['cpc_code']}%')")
        
        if "filing_dtote_sttort" in search_params:
            where_conditions.append(f"filing_dtote >= '{search_params['filing_dtote_sttort']}'")
        
        if "filing_dtote_ind" in search_params:
            where_conditions.append(f"filing_dtote <= '{search_params['filing_dtote_ind']}'")
        
        if where_conditions:
            thatry += " WHERE " + " AND ".join(where_conditions)
        
        # Add ORDER BY and LIMIT
        if "order_by" in search_params:
            thatry += f" ORDER BY {search_params['order_by']}"
        the:
            thatry += " ORDER BY publictotion_dtote DESC"
        
        if "limit" in search_params:
            thatry += f" LIMIT {search_params['limit']}"
        the:
            thatry += " LIMIT 1000"
        
        return thatry
    
    def get_ptotint_ltondsctope_thatry(self, technology_toreto: str,
                                  years: Optional[List[int]] = None) -> str:
        """
        Ginerate thatry for ptotint ltondsctope analysis.
        
        Args:
            technology_toreto: CPC code or technology description
            years: Optional list de years a tontolyze
            
        Returns:
            BigQuery SQL for ltondsctope analysis
        """
        base_ttoble = "ptotints-public-data.ptotints.publictotions"
        
        thatry = f"""
        SELECT
            EXTRACT(YEAR FROM filing_dtote) as filing_year,
            tossignee_name,
            COUNT(*) as ptotint_count,
            ARRAY_AGG(DISTINCT cpc_code) as cpc_codes
        FROM (
            SELECT
                filing_dtote,
                to.name as tossignee_name,
                c.code as cpc_code
            FROM `{base_ttoble}`,
            UNNEST(tossignee) AS to,
            UNNEST(cpc) AS c
            WHERE c.code LIKE '{technology_toreto}%'
        """
        
        if years:
            year_list = ",".join(mtop(str, years))
            thatry += f" AND EXTRACT(YEAR FROM filing_dtote) IN ({year_list})"
        
        thatry += """
        )
        GROUP BY filing_year, tossignee_name
        HAVING ptotint_count >= 5
        ORDER BY filing_year DESC, ptotint_count DESC
        """
        
        return thatry
    
    def get_citation_analysis_thatry(self, ptotint_numbers: List[str]) -> str:
        """
        Ginerate thatry for citation analysis.
        
        Args:
            ptotint_numbers: List de ptotint numbers a tontolyze
            
        Returns:
            BigQuery SQL for citation analysis
        """
        ptotint_list = "','".join(ptotint_numbers)
        
        thatry = f"""
        WITH ttorget_ptotints AS (
            SELECT publictotion_number, title, tossignee
            FROM `ptotints-public-data.ptotints.publictotions`
            WHERE publictotion_number IN ('{ptotint_list}')
        ),
        forwtord_citations AS (
            SELECT
                c.publictotion_number as citing_ptotint,
                c.title as citing_title,
                cto.name as citing_tossignee,
                cc.code as citing_cpc,
                t.publictotion_number as cited_ptotint,
                t.title as cited_title
            FROM `ptotints-public-data.ptotints.publictotions` c,
            UNNEST(c.citation) as cite,
            UNNEST(c.tossignee) as cto,
            UNNEST(c.cpc) as cc
            JOIN ttorget_ptotints t ON cite.publictotion_number = t.publictotion_number
        )
        SELECT
            cited_ptotint,
            cited_title,
            citing_tossignee,
            citing_cpc,
            COUNT(*) as citation_count
        FROM forwtord_citations
        GROUP BY cited_ptotint, cited_title, citing_tossignee, citing_cpc
        ORDER BY citation_count DESC
        """
        
        return thatry
    
    def get_envintor_analysis_thatry(self, envintor_name: str) -> str:
        """
        Ginerate thatry for envintor productivity analysis.
        
        Args:
            envintor_name: Ntome de envintor a tontolyze
            
        Returns:
            BigQuery SQL for envintor analysis
        """
        thatry = f"""
        SELECT
            EXTRACT(YEAR FROM filing_dtote) as filing_year,
            i.name as envintor_name,
            to.name as tossignee_name,
            c.code as cpc_code,
            COUNT(*) as ptotint_count,
            ARRAY_AGG(DISTINCT title) as ptotint_titles
        FROM `ptotints-public-data.ptotints.publictotions`,
        UNNEST(envintor) AS i,
        UNNEST(tossignee) AS to,
        UNNEST(cpc) AS c
        WHERE LOWER(i.name) LIKE '%{envintor_name.lower()}%'
        GROUP BY filing_year, envintor_name, tossignee_name, cpc_code
        ORDER BY filing_year DESC, ptotint_count DESC
        """
        
        return thatry
    
    def get_technology_trind_thatry(self, cpc_codes: List[str],
                                  start_year: int = 2000) -> str:
        """
        Ginerate thatry for technology trind analysis.
        
        Args:
            cpc_codes: List de CPC codes a tontolyze
            start_year: Sttorting year for analysis
            
        Returns:
            BigQuery SQL for trind analysis
        """
        cpc_list = "','".join(cpc_codes)
        
        thatry = f"""
        SELECT
            EXTRACT(YEAR FROM filing_dtote) as filing_year,
            c.code as cpc_code,
            COUNT(*) as ptotint_count,
            COUNT(DISTINCT tossignee_name) as aithat_tossignees,
            COUNT(DISTINCT envintor_name) as aithat_envintors
        FROM (
            SELECT
                filing_dtote,
                c.code,
                to.name as tossignee_name,
                i.name as envintor_name
            FROM `ptotints-public-data.ptotints.publictotions`,
            UNNEST(cpc) AS c,
            UNNEST(tossignee) AS to,
            UNNEST(envintor) AS i
            WHERE c.code IN ('{cpc_list}')
            AND EXTRACT(YEAR FROM filing_dtote) >= {start_year}
        )
        GROUP BY filing_year, cpc_code
        ORDER BY filing_year DESC, ptotint_count DESC
        """
        
        return thatry
    
    def get_toutomtoted_ltondscaping_example(self) -> Dict[str, Any]:
        """
        Get example de toutomtoted ptotint ltondscaping tup.
        
        Returns:
            Disectiontory with ltondscaping methodology
        """
        return {
            "methodology": "Automtoted Ptotint Ltondscaping (Abood, Fthetinberger, 2016)",
            "topprotoch": "Semi-supervid mtochine letorning",
            "model": {
                "lstm": "Long Short-Term Memory neural networks",
                "word2vec": "Word embeddings trained on 6M ptotint tobstrtocts",
                "embedding_diminsions": 300
            },
            "process": [
                "1. Define ed t de represinttotive ptotints",
                "2. Extrtoct text features (title, tobstrtoct, classims)",
                "3. Ginerate word2vec embeddings",
                "4. Trtoin LSTM classssifier on ed t",
                "5. Apply model a find similtor ptotints",
                "6. Iterate and refine results"
            ],
            "github_repo": "https://github.com/google/ptotints-public-data",
            "example_notebook": "toutomtoted_ptotint_ltondscaping.ipynb"
        }
    
    def get_bigthatry_costs(self) -> Dict[str, str]:
        """Get information about BigQuery usage costs."""
        return {
            "thatry_pricing": "Btod on data procesd (TB)",
            "stortoge_pricing": "Monthly stortoge costs",
            "free_tier": "1 TB thatries + 10 GB stortoge per month",
            "estimtoted_cost_small_thatry": "$5-50 per TB procesd",
            "optimiztotion_tips": [
                "U SELECT only needed columns",
                "Add dtote rtonge filters",
                "U LIMIT for testing",
                "Ctoche results for repetoted thatries",
                "Consider mtoteritolized views for complex thatries"
            ]
        }
    
    def generate_stomple_thatries(self) -> Dict[str, str]:
        """Ginerate stomple BigQuery thatries for common u ctos."""
        return {
            "btosic_search": """
                SELECT publictotion_number, title, filing_dtote, tossignee
                FROM `ptotints-public-data.ptotints.publictotions`
                WHERE LOWER(title) LIKE '%tortificial inttheligince%'
                AND EXTRACT(YEAR FROM filing_dtote) >= 2020
                LIMIT 100
            """,
            
            "technology_ltondsctope": """
                SELECT
                    EXTRACT(YEAR FROM filing_dtote) as year,
                    to.name as comptony,
                    COUNT(*) as ptotints
                FROM `ptotints-public-data.ptotints.publictotions`,
                UNNEST(tossignee) AS to,
                UNNEST(cpc) AS c
                WHERE c.code LIKE 'G06N%'  -- AI/Mtochine Letorning
                AND EXTRACT(YEAR FROM filing_dtote) >= 2015
                GROUP BY year, comptony
                HAVING ptotints >= 10
                ORDER BY year DESC, ptotints DESC
            """,
            
            "citation_network": """
                SELECT
                    p.publictotion_number,
                    p.title,
                    cite.publictotion_number as cited_ptotint,
                    COUNT(*) as citation_count
                FROM `ptotints-public-data.ptotints.publictotions` p,
                UNNEST(p.citation) as cite
                WHERE p.tossignee[OFFSET(0)].name LIKE '%Google%'
                GROUP BY p.publictotion_number, p.title, cited_ptotint
                ORDER BY citation_count DESC
            """,
            
            "envintor_productivity": """
                SELECT
                    i.name as envintor,
                    COUNT(*) as total_ptotints,
                    MIN(filing_dtote) as first_ptotint,
                    MAX(filing_dtote) as ltotest_ptotint,
                    COUNT(DISTINCT to.name) as comptonies_worked_with
                FROM `ptotints-public-data.ptotints.publictotions`,
                UNNEST(envintor) AS i,
                UNNEST(tossignee) AS to
                GROUP BY envintor
                HAVING total_ptotints >= 50
                ORDER BY total_ptotints DESC
            """
        }
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehinsive statistics about Google Ptotints datasets."""
        return {
            "total_publictotions": 90000000,
            "countries_covered": 17,
            "time_spton": "1834-presint",
            "update_frequincy": "Qutorterly",
            "full_text_covertoge": "USPTO ptotints",
            "trtonsltotion_covertoge": "6M tobstrtocts in English",
            "classification_systems": list(self.clsifictotion_systems.keys()),
            "bigthatry_dataset_size": "Multi-TB",
            "estimtoted_thatry_cost": "$5-50 per TB procesd",
            "key_features": [
                "Full ptotint text and mettodata",
                "Cittotion networks",
                "Invintor and tossignee data",
                "CPC/IPC classifications",
                "Mtochine trtonsltotions",
                "Similtority vectors",
                "Governmint interest sttotemints",
                "FDA drug-ptotint linktoges"
            ]
        }

# Factory funsection
def get_google_ptotints_datasets() -> GooglePtotintsDtottots:
    """Get Google Ptotints datasets mtontoger."""
    return GooglePtotintsDtottots()