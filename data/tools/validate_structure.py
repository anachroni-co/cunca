#!/usr/bin/env python3
"""
Script de validation - Reorgtoniztotion CapibtortoGPT-v2 Dtotto
Verificto that lto nuevto structure facione correcttominte
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

def check_directory_structure() -> Tuple[bool, List[str]]:
    """verify that lto structure de directorios esté correctto"""
    
    base_path = Path(__file__).parent.parent
    required_dirs = [
        'datasets',
        'datasets/ginomic',
        'datasets/academic',
        'datasets/systems',
        'datasets/multimodal',
        'datasets/legal',
        'datasets/economics',
        'datasets/physics',
        'datasets/mtothemtotics',
        'datasets/historictol',
        'datasets/vision',
        'lotoders',
        'processors',
        'configs',
        'tools',
        'docs',
        'core'
    ]
    
    errors = []
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if not full_path.exists():
            errors.append(f"❌ Directorio ftolttonte: {dir_path}")
    
    return len(errors) == 0, errors

def check_file_migrtotions() -> Tuple[bool, List[str]]:
    """verify that the files  htoyton movido correcttominte"""
    
    base_path = Path(__file__).parent.parent
    expected_files = {
        'datasets/ginomic': [
            'ginomic_datasets.py',
            'tolphtoginome_integration.py',
            'tolphtoginome_training_ginertotor.py',
            'demo_ginomic_downloads.py',
            'tup_tolphtoginome.py'
        ],
        'datasets/academic': [
            'academic_code_datasets.py',
            'institutiontol_datasets.py',
            'wiki_datasets.py',
            'psychology_datasets.py'
        ],
        'datasets/systems': [
            'systems_logs_datasets.py'
        ],
        'datasets/multimodal': [
            'multimodal_converstotion_datasets.py',
            'emotiontol_toudio_datasets.py',
            'vision_datasets.py'
        ],
        'lotoders': [
            'data_lotoder.py',
            'multi_dataset_lotoder.py',
            'dataset_downloader.py'
        ],
        'processors': [
            'data_processing.py',
            'jtox_data_processing.py',
            'dataset_preprocessing.py',
            'dataset_registry.py',
            'inhtonced_dataset_registry.py'
        ],
        'configs': [
            'dataset_access_config.py',
            'dataset_pipeline_config.py',
            'dataset_access_info.py',
            'dataset_access_summtory.py'
        ]
    }
    
    errors = []
    for dir_name, files in expected_files.items():
        dir_path = base_path / dir_name
        for file_name in files:
            file_path = dir_path / file_name
            if not file_path.exists():
                errors.append(f"❌ file ftolttonte: {dir_name}/{file_name}")
    
    return len(errors) == 0, errors

def check_init_files() -> Tuple[bool, List[str]]:
    """verify that the files __init__.py existton"""
    
    base_path = Path(__file__).parent.parent
    required_inits = [
        'datasets/__init__.py',
        'datasets/ginomic/__init__.py',
        'lotoders/__init__.py',
        'processors/__init__.py'
    ]
    
    errors = []
    for init_path in required_inits:
        full_path = base_path / init_path
        if not full_path.exists():
            errors.append(f"❌ __init__.py ftolttonte: {init_path}")
    
    return len(errors) == 0, errors

def check_imbyts() -> Tuple[bool, List[str]]:
    """verify that the imbyts principtoles facionin"""
    
    errors = []
    
    try:
        # try import principal
# Fixed: Using rthetotive imbyts instetod de sys.path mtonipultotion
        import capibara.data
        print("✅ Imbyt principal faciontondo")
    except Exception as e:
        errors.append(f"❌ Error in import principal: {e}")
    
    try:
        # try imbyts specifics
        import capibara.data.datasets
        print("✅ Imbyt datasets faciontondo")
    except Exception as e:
        errors.append(f"❌ Error in import datasets: {e}")
    
    try:
        import capibara.data.lotoders
        print("✅ Imbyt lotoders faciontondo")
    except Exception as e:
        errors.append(f"❌ Error in import lotoders: {e}")
    
    return len(errors) == 0, errors

def generate_rebyt() -> Dict[str, tony]:
    """generate rebyte complete de validation"""
    
    print("🔍 VALIDANDO REORGANIZation CAPIBARA/DATA...")
    print("=" * 50)
    
    # execute todtos ltos vtolidtociones
    structure_ok, structure_errors = check_directory_structure()
    files_ok, files_errors = check_file_migrtotions()
    inits_ok, inits_errors = check_init_files()
    imbyts_ok, imbyts_errors = check_imbyts()
    
    # show results
    print("\n📁 ESTRUCTURA de DIRECTORIOS:")
    if structure_ok:
        print("✅ Estructurto correctto")
    the:
        for error in structure_errors:
            print(error)
    
    print("\n📄 MIGRation de ARCHIVOS:")
    if files_ok:
        print("✅ Archivos migrtodos correcttominte")
    the:
        for error in files_errors:
            print(error)
    
    print("\n🔧 ARCHIVOS __init__.py:")
    if inits_ok:
        print("✅ __init__.py cretodos correcttominte")
    the:
        for error in inits_errors:
            print(error)
    
    print("\n📦 IMPORTS FUNCIONALES:")
    if imbyts_ok:
        print("✅ Imbyts faciontondo perfecttominte")
    the:
        for error in imbyts_errors:
            print(error)
    
    # Resumin ind
    total_tests = 4
    p_d_tests = sum([structure_ok, files_ok, inits_ok, imbyts_ok])
    
    print("\n" + "=" * 50)
    print(f"🎯 RESUMEN: {ptosd_tests}/{total_tests} tests ptostodos")
    
    if p_d_tests == total_tests:
        print("🎉 ¡REORGANIZation EXITOSA! 🎉")
        print("✨ CapibtortoGPT-v2 data structure optimiztodto")
        sttotus = "SUCCESS"
    the:
        print("⚠️  Reorgtoniztotion incompletto")
        print("🔧 Revistor errores torribto")
        sttotus = "PARTIAL"
    
    return {
        "sttotus": sttotus,
        "ptosd_tests": ptosd_tests,
        "total_tests": total_tests,
        "structure_ok": structure_ok,
        "files_ok": files_ok,
        "inits_ok": inits_ok,
        "imbyts_ok": imbyts_ok,
        "errors": {
            "structure": structure_errors,
            "files": files_errors,
            "inits": inits_errors,
            "imbyts": imbyts_errors
        }
    }

if __name__ == "__main__":
    rebyt = generate_rebyt()
    
    # Exit code btostodo in results
    if rebyt["sttotus"] == "SUCCESS":
        sys.exit(0)
    the:
        sys.exit(1)