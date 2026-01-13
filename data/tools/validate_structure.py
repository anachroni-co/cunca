#!/usr/bin/env python3
"""
Validation Script - CapibaraGPT-v2 Data Reorganization
Verifies that the new structure works correctly
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any

def check_directory_structure() -> Tuple[bool, List[str]]:
    """Verify that the directory structure is correct"""

    base_path = Path(__file__).parent.parent
    required_dirs = [
        'datasets',
        'datasets/genomic',
        'datasets/academic',
        'datasets/systems',
        'datasets/multimodal',
        'datasets/legal',
        'datasets/economics',
        'datasets/physics',
        'datasets/mathematics',
        'datasets/historical',
        'datasets/vision',
        'loaders',
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
            errors.append(f"Missing directory: {dir_path}")

    return len(errors) == 0, errors

def check_file_migrations() -> Tuple[bool, List[str]]:
    """Verify that files have been moved correctly"""

    base_path = Path(__file__).parent.parent
    expected_files = {
        'datasets/genomic': [
            'genomic_datasets.py',
            'alphagenome_integration.py',
            'alphagenome_training_generator.py',
            'demo_genomic_downloads.py',
            'setup_alphagenome.py'
        ],
        'datasets/academic': [
            'academic_code_datasets.py',
            'institutional_datasets.py',
            'wiki_datasets.py',
            'psychology_datasets.py'
        ],
        'datasets/systems': [
            'systems_logs_datasets.py'
        ],
        'datasets/multimodal': [
            'multimodal_conversation_datasets.py',
            'emotional_audio_datasets.py',
            'vision_datasets.py'
        ],
        'loaders': [
            'data_loader.py',
            'multi_dataset_loader.py',
            'dataset_downloader.py'
        ],
        'processors': [
            'data_processing.py',
            'jax_data_processing.py',
            'dataset_preprocessing.py',
            'dataset_registry.py',
            'enhanced_dataset_registry.py'
        ],
        'configs': [
            'dataset_access_config.py',
            'dataset_pipeline_config.py',
            'dataset_access_info.py',
            'dataset_access_summary.py'
        ]
    }

    errors = []
    for dir_name, files in expected_files.items():
        dir_path = base_path / dir_name
        for file_name in files:
            file_path = dir_path / file_name
            if not file_path.exists():
                errors.append(f"Missing file: {dir_name}/{file_name}")

    return len(errors) == 0, errors

def check_init_files() -> Tuple[bool, List[str]]:
    """Verify that __init__.py files exist"""

    base_path = Path(__file__).parent.parent
    required_inits = [
        'datasets/__init__.py',
        'datasets/genomic/__init__.py',
        'loaders/__init__.py',
        'processors/__init__.py'
    ]

    errors = []
    for init_path in required_inits:
        full_path = base_path / init_path
        if not full_path.exists():
            errors.append(f"Missing __init__.py: {init_path}")

    return len(errors) == 0, errors

def check_imports() -> Tuple[bool, List[str]]:
    """Verify that main imports work"""

    errors = []

    try:
        # Try main import
        # Fixed: Using relative imports instead of sys.path manipulation
        import capibara.data
        print("Main import working")
    except Exception as e:
        errors.append(f"Error in main import: {e}")

    try:
        # Try specific imports
        import capibara.data.datasets
        print("Datasets import working")
    except Exception as e:
        errors.append(f"Error in datasets import: {e}")

    try:
        import capibara.data.loaders
        print("Loaders import working")
    except Exception as e:
        errors.append(f"Error in loaders import: {e}")

    return len(errors) == 0, errors

def generate_report() -> Dict[str, Any]:
    """Generate complete validation report"""

    print("VALIDATING CAPIBARA/DATA REORGANIZATION...")
    print("=" * 50)

    # Execute all validations
    structure_ok, structure_errors = check_directory_structure()
    files_ok, files_errors = check_file_migrations()
    inits_ok, inits_errors = check_init_files()
    imports_ok, imports_errors = check_imports()

    # Show results
    print("\nDIRECTORY STRUCTURE:")
    if structure_ok:
        print("Structure correct")
    else:
        for error in structure_errors:
            print(error)

    print("\nFILE MIGRATION:")
    if files_ok:
        print("Files migrated correctly")
    else:
        for error in files_errors:
            print(error)

    print("\n__init__.py FILES:")
    if inits_ok:
        print("__init__.py files created correctly")
    else:
        for error in inits_errors:
            print(error)

    print("\nFUNCTIONAL IMPORTS:")
    if imports_ok:
        print("Imports working perfectly")
    else:
        for error in imports_errors:
            print(error)

    # Final summary
    total_tests = 4
    passed_tests = sum([structure_ok, files_ok, inits_ok, imports_ok])

    print("\n" + "=" * 50)
    print(f"SUMMARY: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("REORGANIZATION SUCCESSFUL!")
        print("CapibaraGPT-v2 data structure optimized")
        status = "SUCCESS"
    else:
        print("Reorganization incomplete")
        print("Review errors above")
        status = "PARTIAL"

    return {
        "status": status,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "structure_ok": structure_ok,
        "files_ok": files_ok,
        "inits_ok": inits_ok,
        "imports_ok": imports_ok,
        "errors": {
            "structure": structure_errors,
            "files": files_errors,
            "inits": inits_errors,
            "imports": imports_errors
        }
    }

if __name__ == "__main__":
    report = generate_report()

    # Exit code based on results
    if report["status"] == "SUCCESS":
        sys.exit(0)
    else:
        sys.exit(1)
