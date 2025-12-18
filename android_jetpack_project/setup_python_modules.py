#!/usr/bin/env python3
import shutil
import os
from pathlib import Path

def setup_python_modules():
    repo_root = Path(__file__).parent.parent
    android_project = Path(__file__).parent
    python_dir = android_project / "app" / "src" / "main" / "python"
    
    vector_extraction_src = repo_root / "vector_extraction"
    classifiers_src = repo_root / "classifiers"
    
    vector_extraction_dst = python_dir / "vector_extraction"
    classifiers_dst = python_dir / "classifiers"
    
    if vector_extraction_src.exists():
        if vector_extraction_dst.exists():
            shutil.rmtree(vector_extraction_dst)
        shutil.copytree(vector_extraction_src, vector_extraction_dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        print(f"Copied vector_extraction to {vector_extraction_dst}")
    
    if classifiers_src.exists():
        if classifiers_dst.exists():
            shutil.rmtree(classifiers_dst)
        shutil.copytree(classifiers_src, classifiers_dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.joblib', '*.zip', 'grid_search_results'))
        print(f"Copied classifiers to {classifiers_dst}")

if __name__ == "__main__":
    setup_python_modules()

