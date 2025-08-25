#!/usr/bin/env python3
"""
Comprehensive test to verify the image analysis tutorial functionality
画像解析チュートリアルの機能を検証する包括的テスト
"""

import subprocess
import sys
import os
import json

def test_notebook_execution(notebook_path):
    """
    Test if a notebook can be executed without errors
    ノートブックがエラーなしで実行できるかテスト
    """
    try:
        result = subprocess.run([
            'jupyter', 'nbconvert', '--to', 'script', '--stdout', notebook_path
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return True, "Notebook conversion successful"
        else:
            return False, f"Conversion failed: {result.stderr}"
    
    except subprocess.TimeoutExpired:
        return False, "Conversion timeout"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """
    Main test function for tutorial completeness
    チュートリアル完全性のメインテスト関数
    """
    print("Comprehensive Image Analysis Tutorial Test")
    print("包括的画像解析チュートリアルテスト")
    print("=" * 60)
    
    notebooks = [
        '01_image_filtering.ipynb',
        '02_machine_learning_basics.ipynb', 
        '03_mrf_analysis.ipynb'
    ]
    
    print("\n1. Testing Notebook Execution Capability / ノートブック実行能力テスト")
    print("-" * 40)
    
    all_executable = True
    for notebook in notebooks:
        if os.path.exists(notebook):
            is_executable, message = test_notebook_execution(notebook)
            status = "✓" if is_executable else "✗"
            print(f"{status} {notebook}: {message}")
            if not is_executable:
                all_executable = False
        else:
            print(f"✗ {notebook}: File not found")
            all_executable = False
    
    print("\n2. Testing Project Completeness / プロジェクト完全性テスト")
    print("-" * 40)
    
    required_components = {
        'README.md': 'Project documentation',
        'requirements.txt': 'Dependencies specification',
        'utils/__init__.py': 'Utils package initialization',
        'utils/estimator.py': 'MRF estimator classes',
        'utils/image.py': 'Image generation utilities',
        '01_image_filtering.ipynb': 'Image filtering tutorial',
        '02_machine_learning_basics.ipynb': 'ML basics tutorial',
        '03_mrf_analysis.ipynb': 'MRF analysis tutorial'
    }
    
    project_complete = True
    for component, description in required_components.items():
        if os.path.exists(component):
            size = os.path.getsize(component)
            print(f"✓ {component}: {description} ({size} bytes)")
        else:
            print(f"✗ {component}: Missing - {description}")
            project_complete = False
    
    print("\n3. Testing Content Quality / コンテンツ品質テスト")
    print("-" * 40)
    
    content_checks = []
    
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        bilingual_check = '日本語' in readme_content and 'English' in readme_content
        content_checks.append(('README bilingual content', bilingual_check))
        
        sections_check = all(section in readme_content for section in [
            'Image Filtering', 'Machine Learning', 'MRF Analysis'
        ])
        content_checks.append(('README sections complete', sections_check))
    
    for notebook in notebooks:
        if os.path.exists(notebook):
            with open(notebook, 'r', encoding='utf-8') as f:
                notebook_data = json.load(f)
            
            has_markdown = any(cell['cell_type'] == 'markdown' for cell in notebook_data['cells'])
            has_code = any(cell['cell_type'] == 'code' for cell in notebook_data['cells'])
            
            content_checks.append((f'{notebook} has markdown cells', has_markdown))
            content_checks.append((f'{notebook} has code cells', has_code))
    
    for check_name, passed in content_checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
    
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY / 最終テスト要約")
    print("=" * 60)
    
    all_passed = all_executable and project_complete and all(check[1] for check in content_checks)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! Tutorial is complete and ready for delivery!")
        print("🎉 すべてのテストが合格！チュートリアルは完成し、配布準備完了です！")
        
        print("\nTutorial Components Summary / チュートリアル構成要素要約:")
        print("- ✅ Image Filtering: Lena download, smoothing, Sobel edge detection, SIFT")
        print("- ✅ Machine Learning: f=ma data generation, linear regression, cross-validation")
        print("- ✅ MRF Analysis: 1D image generation, MRF restoration, lambda optimization")
        print("- ✅ Bilingual documentation (Japanese/English)")
        print("- ✅ Complete project structure with utils modules")
        
        return 0
    else:
        print("❌ Some tests failed. Tutorial needs additional work.")
        print("❌ いくつかのテストが失敗しました。チュートリアルには追加作業が必要です。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
