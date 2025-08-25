#!/usr/bin/env python3
"""
Test script to validate the image analysis tutorial notebooks
画像解析チュートリアルノートブックを検証するテストスクリプト
"""

import os
import sys
import subprocess
import json

def test_notebook_syntax(notebook_path):
    """
    Test if a Jupyter notebook has valid JSON syntax
    Jupyter notebookが有効なJSON構文を持つかテスト
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
        
        required_keys = ['cells', 'metadata', 'nbformat', 'nbformat_minor']
        for key in required_keys:
            if key not in notebook_data:
                return False, f"Missing required key: {key}"
        
        if not isinstance(notebook_data['cells'], list):
            return False, "Cells should be a list"
        
        for i, cell in enumerate(notebook_data['cells']):
            if 'cell_type' not in cell:
                return False, f"Cell {i} missing cell_type"
            if 'source' not in cell:
                return False, f"Cell {i} missing source"
        
        return True, "Notebook syntax is valid"
    
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Error reading notebook: {e}"

def test_imports():
    """
    Test if required libraries can be imported
    必要なライブラリがインポートできるかテスト
    """
    required_imports = [
        'numpy',
        'matplotlib.pyplot',
        'sklearn.linear_model',
        'sklearn.model_selection',
        'pandas'
    ]
    
    optional_imports = [
        'cv2',
        'skimage',
        'scipy',
        'requests'
    ]
    
    results = {}
    
    for module in required_imports:
        try:
            __import__(module)
            results[module] = "✓ Available"
        except ImportError:
            results[module] = "✗ Missing (Required)"
    
    for module in optional_imports:
        try:
            __import__(module)
            results[module] = "✓ Available"
        except ImportError:
            results[module] = "⚠ Missing (Optional)"
    
    return results

def test_utils_modules():
    """
    Test if utils modules can be imported
    utilsモジュールがインポートできるかテスト
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from utils import Image1D, MatrixEstimator, GradientEstimator
        
        img_gen = Image1D(10)
        matrix_est = MatrixEstimator(10)
        grad_est = GradientEstimator(10)
        
        return True, "Utils modules imported successfully"
    except Exception as e:
        return False, f"Error importing utils: {e}"

def main():
    """
    Main test function
    メインテスト関数
    """
    print("Image Analysis Tutorial Test Suite")
    print("画像解析チュートリアルテストスイート")
    print("=" * 50)
    
    notebooks = [
        '01_image_filtering.ipynb',
        '02_machine_learning_basics.ipynb', 
        '03_mrf_analysis.ipynb'
    ]
    
    print("\n1. Testing Notebook Syntax / ノートブック構文テスト")
    print("-" * 30)
    
    all_notebooks_valid = True
    for notebook in notebooks:
        if os.path.exists(notebook):
            is_valid, message = test_notebook_syntax(notebook)
            status = "✓" if is_valid else "✗"
            print(f"{status} {notebook}: {message}")
            if not is_valid:
                all_notebooks_valid = False
        else:
            print(f"✗ {notebook}: File not found")
            all_notebooks_valid = False
    
    print("\n2. Testing Library Imports / ライブラリインポートテスト")
    print("-" * 30)
    
    import_results = test_imports()
    for module, status in import_results.items():
        print(f"{status}: {module}")
    
    print("\n3. Testing Utils Modules / Utilsモジュールテスト")
    print("-" * 30)
    
    utils_valid, utils_message = test_utils_modules()
    status = "✓" if utils_valid else "✗"
    print(f"{status} {utils_message}")
    
    print("\n4. Testing Project Structure / プロジェクト構造テスト")
    print("-" * 30)
    
    required_files = [
        'README.md',
        'requirements.txt',
        'utils/__init__.py',
        'utils/estimator.py',
        'utils/image.py'
    ]
    
    structure_valid = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}: Found")
        else:
            print(f"✗ {file_path}: Missing")
            structure_valid = False
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY / テスト要約")
    print("=" * 50)
    
    if all_notebooks_valid and utils_valid and structure_valid:
        print("✓ All tests passed! Tutorial is ready to use.")
        print("✓ すべてのテストが合格しました！チュートリアルは使用準備完了です。")
        return 0
    else:
        print("✗ Some tests failed. Please check the issues above.")
        print("✗ いくつかのテストが失敗しました。上記の問題を確認してください。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
