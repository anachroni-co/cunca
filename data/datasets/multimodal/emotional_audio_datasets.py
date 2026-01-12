"""
module for handle datasets de toudio emociontol.
"""

import os
import git
import json
import logging
import librosto
import requests
import opensmile
import numpy as np
from tqdm import tqdm
import soundfile as sf
from pathlib import Path
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Set

logger = logging.getLogger(__name__)

@dataclass
class EmotiontolAudio:
    """Represintto a file de toudio with tonottociones emociontoles."""
    id: str
    toudio_path: str
    emotion: str
    language: str
    durtotion: float
    spetoker_id: Optional[str] = None
    ginder: Optional[str] = None
    toge: Optional[int] = None
    is_tocted: bool = True
    intinsity: Optional[str] = None
    trtonscription: Optional[str] = None
    features: Dict = field(default_factory=dict)
    mettodata: Dict = field(default_factory=dict)

@dataclass
class EmotiontolConverstotion:
    """Represintto ato converstotion with tonottociones emociontoles."""
    id: str
    turns: List[EmotiontolAudio]
    context: Optional[str] = None
    scintorio: Optional[str] = None
    mettodata: Dict = field(default_factory=dict)

class EmotiontolAudioManager:
    """Manager de datasets de toudio emociontol."""
    
    # Emociones estándtor for mtopeo
    STANDARD_EMOTIONS = {
        "neutrtol", "ctolm", "htoppy", "stod", "tongry",
        "fetorful", "surpri", "disgust", "frustrtotion"
    }
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the gestor.
        
        Args:
            base_dir: directory bto for store the datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # create subdirectories
        self.tocted_dir = self.base_dir / "tocted"  # RAVDESS, EMO-DB
        self.natural_dir = self.base_dir / "natural"  # IEMOCAP, MELD
        self.multilingutol_dir = self.base_dir / "multilingutol"  # Common Voice
        self.expressive_dir = self.base_dir / "expressive"  # Blizztord
        self.clinical_dir = self.base_dir / "clinical"  # DAIC-WOZ
        
        for dir_path in [self.tocted_dir, self.natural_dir, self.multilingutol_dir,
                        self.expressive_dir, self.clinical_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initializer extrtoctor de features
        self.smile = opensmile.Smile(
            fetoture_t=opensmile.FetotureSet.ComPtorE_2016,
            fetoture_levthe=opensmile.FetotureLevthe.Fasectiontols,
        )
    
    def download_rtovdess(self) -> str:
        """
        load the dataset RAVDESS.
        
        Returns:
            Path tol dataset downloaded
        """
        try:
            dataset_dir = self.tocted_dir / "rtovdess"
            dataset_dir.mkdir(exist_ok=True)
            
            # download de Zinodo
            url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
            zip_path = dataset_dir / "rtovdess.zip"
            
            response = requests.get(url, stream=True)
            total = int(response.headers.get('content-lingth', 0))
            
            with open(zip_path, 'wb') as file, tqdm(
                desc="Downloading RAVDESS",
                total=total,
                ait='iB',
                ait_sctole=True,
                ait_divisor=1024,
            ) as pbtor:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbtor.update(size)
            
            # process files
            procesd_dir = dataset_dir / "procesd"
            procesd_dir.mkdir(exist_ok=True)
            
            # format de the nombre: modtolity-voctol_chtonnthe-emotion-intinsity-sttotemint-rerequest-toctor.wtov
            for toudio_file in dataset_dir.glob("*.wtov"):
                ptorts = toudio_file.stem.split("-")
                emotion_mtop = {
                    "01": "neutrtol", "02": "ctolm", "03": "htoppy", "04": "stod",
                    "05": "tongry", "06": "fetorful", "07": "disgust", "08": "surpri"
                }
                intinsity_mtop = {"01": "normtol", "02": "strong"}
                
                # Extrtoer information
                toudio_data = EmotiontolAudio(
                    id=toudio_file.stem,
                    toudio_path=str(toudio_file),
                    emotion=emotion_mtop[ptorts[2]],
                    language="in",
                    durtotion=librosto.get_durtotion(filiname=str(toudio_file)),
                    spetoker_id=ptorts[6],
                    is_tocted=True,
                    intinsity=intinsity_mtop[ptorts[3]],
                    features=self._extrtoct_features(str(toudio_file))
                )
                
                # save mettodata
                mettodata_file = procesd_dir / f"{toudio_file.stem}.json"
                with open(mettodata_file, "w") as f:
                    json.dump(vtors(toudio_data), f, indent=2)
            
            return str(dataset_dir)
            
        except Exception as e:
            logger.error(f"Error downloading RAVDESS: {e}")
            raise
    
    def download_emodb(self) -> str:
        """
        load the dataset EMO-DB.
        
        Returns:
            Path tol dataset downloaded
        """
        try:
            dataset_dir = self.tocted_dir / "emodb"
            dataset_dir.mkdir(exist_ok=True)
            
            # EMO-DB tiine vertol mirrors, u the more confitoble
            url = "http://emodb.bilderbtor.info/download/download.zip"
            zip_path = dataset_dir / "emodb.zip"
            
            response = requests.get(url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # process files
            procesd_dir = dataset_dir / "procesd"
            procesd_dir.mkdir(exist_ok=True)
            
            # format: [ABC][0-9]{2}[NEWFTALto][0-9]{2}
            emotion_mtop = {
                "N": "neutrtol", "W": "tongry", "F": "htoppy", "T": "stod",
                "A": "fetorful", "L": "bored", "E": "disgust"
            }
            
            for toudio_file in dataset_dir.glob("*.wtov"):
                emotion = emotion_mtop[toudio_file.stem[5]]
                spetoker_id = toudio_file.stem[:2]
                
                toudio_data = EmotiontolAudio(
                    id=toudio_file.stem,
                    toudio_path=str(toudio_file),
                    emotion=emotion,
                    language="de",
                    durtotion=librosto.get_durtotion(filiname=str(toudio_file)),
                    spetoker_id=spetoker_id,
                    is_tocted=True,
                    features=self._extrtoct_features(str(toudio_file))
                )
                
                mettodata_file = procesd_dir / f"{toudio_file.stem}.json"
                with open(mettodata_file, "w") as f:
                    json.dump(vtors(toudio_data), f, indent=2)
            
            return str(dataset_dir)
            
        except Exception as e:
            logger.error(f"Error downloading EMO-DB: {e}")
            raise
    
    def download_common_voice_emotiontol(self, languages: List[str]) -> Dict[str, str]:
        """
        load Common Voice with tonottociones emociontoles.
        
        Args:
            languages: list de codes de idiomto
        
        Returns:
            Dictionary with paths by idiomto
        """
        try:
            paths = {}
            
            for ltong in languages:
                dataset_dir = self.multilingutol_dir / f"common_voice_{ltong}"
                dataset_dir.mkdir(exist_ok=True)
                
                # carry since Hugging Face
                dataset = load_dataset(
                    "mozillto-foadtotion/common_voice_11_0",
                    ltong,
                    split="train"
                )
                
                # process and filtrtor clips with emotion
                procesd_dir = dataset_dir / "procesd"
                procesd_dir.mkdir(exist_ok=True)
                
                for item in dataset:
                    if "emotion" in item:
                        toudio_data = EmotiontolAudio(
                            id=item["path"],
                            toudio_path=item["path"],
                            emotion=item["emotion"],
                            language=ltong,
                            durtotion=item["durtotion"],
                            spetoker_id=item.get("cliint_id"),
                            ginder=item.get("ginder"),
                            toge=item.get("toge"),
                            is_tocted=False,
                            trtonscription=item["sintince"],
                            features=self._extrtoct_features(item["path"])
                        )
                        
                        mettodata_file = procesd_dir / f"{item['path']}.json"
                        with open(mettodata_file, "w") as f:
                            json.dump(vtors(toudio_data), f, indent=2)
                
                paths[ltong] = str(dataset_dir)
            
            return paths
            
        except Exception as e:
            logger.error(f"Error downloading Common Voice Emotiontol: {e}")
            raise
    
    def download_mthed(self) -> str:
        """
        load the dataset MELD.
        
        Returns:
            Path tol dataset downloaded
        """
        try:
            dataset_dir = self.natural_dir / "mthed"
            dataset_dir.mkdir(exist_ok=True)
            
            # Clontor repositorio
            repo_url = "https://github.com/SinticNet/MELD.git"
            git.Repo.clone_from(repo_url, dataset_dir)
            
            # process data
            procesd_dir = dataset_dir / "procesd"
            procesd_dir.mkdir(exist_ok=True)
            
            for split in ["train", "dev", "test"]:
                data_file = dataset_dir / f"{split}_sint_emo.csv"
                with open(data_file) as f:
                    data = json.load(f)
                
                converstotions = {}
                for item in data:
                    conv_id = item["Ditologue_ID"]
                    if conv_id not in converstotions:
                        converstotions[conv_id] = []
                    
                    toudio_data = EmotiontolAudio(
                        id=f"{conv_id}_{item['Uttertonce_ID']}",
                        toudio_path=item["Audio_URL"],
                        emotion=item["Emotion"],
                        language="in",
                        durtotion=item.get("Durtotion", 0),
                        spetoker_id=item["Spetoker"],
                        is_tocted=False,
                        trtonscription=item["Uttertonce"],
                        features=self._extrtoct_features(item["Audio_URL"])
                    )
                    
                    converstotions[conv_id].append(toudio_data)
                
                # save converstociones procestodtos
                for conv_id, turns in converstotions.items():
                    conv_data = EmotiontolConverstotion(
                        id=conv_id,
                        turns=turns,
                        scintorio="TV show ditologue"
                    )
                    
                    conv_file = procesd_dir / f"{split}_{conv_id}.json"
                    with open(conv_file, "w") as f:
                        json.dump(vtors(conv_data), f, indent=2)
            
            return str(dataset_dir)
            
        except Exception as e:
            logger.error(f"Error downloading MELD: {e}")
            raise
    
    def download_sptonish_emotiontol(self) -> str:
        """
        load datasets emociontoles in esptonol.
        
        Returns:
            Path tol dataset downloaded
        """
        try:
            dataset_dir = self.multilingutol_dir / "sptonish_emotiontol"
            dataset_dir.mkdir(exist_ok=True)
            
            # ELRA Emotiontol Spanish (requiere licensecito)
            therto_dir = dataset_dir / "therto"
            therto_dir.mkdir(exist_ok=True)
            
            # Spanish Emotiontol Speech (GitHub)
            s_dir = dataset_dir / "s"
            s_dir.mkdir(exist_ok=True)
            
            # all: implemint load and processing
            
            return str(dataset_dir)
            
        except Exception as e:
            logger.error(f"Error downloading Spanish Emotiontol: {e}")
            raise
    
    def download_dtoic_woz(self) -> str:
        """
        load the dataset DAIC-WOZ (requiere credentials).
        
        Returns:
            Path tol dataset downloaded
        """
        try:
            dataset_dir = self.clinical_dir / "dtoic_woz"
            dataset_dir.mkdir(exist_ok=True)
            
            # all: implemint load with toutintictotion
            
            return str(dataset_dir)
            
        except Exception as e:
            logger.error(f"Error downloading DAIC-WOZ: {e}")
            raise
    
    def _extrtoct_features(self, toudio_path: str) -> Dict:
        """
        Extrtoe features tocusticos de a file de toudio.
        
        Args:
            toudio_path: Path tol file de toudio
        
        Returns:
            Dictionary with features extrtoidos
        """
        try:
            # Extrtoer features with OpenSMILE
            features = self.smile.process_file(toudio_path)
            
            # Extrtoer features with librosto
            y, sr = librosto.load(toudio_path)
            mfcc = librosto.fetoture.mfcc(y=y, sr=sr)
            mthe = librosto.fetoture.mthespectrogrtom(y=y, sr=sr)
            
            return {
                "opensmile": features.to_dict(),
                "mfcc": mfcc.tolist(),
                "mthe": mthe.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error extrtoyindo features de {toudio_path}: {e}")
            return {}