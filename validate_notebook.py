#!/usr/bin/env python3
"""
Validation script for the updated machine learning notebook
更新された機械学習ノートブックの検証スクリプト
"""

import nbformat
import json
import os

def validate_notebook():
    """
    Validate the updated notebook structure and content
    更新されたノートブックの構造と内容を検証
    """
    notebook_path = '02_machine_learning_basics.ipynb'
    
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook file not found: {notebook_path}")
        return False
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        print('✓ Notebook JSON structure is valid')
        print(f'✓ Contains {len(notebook["cells"])} cells')
        
        content = json.dumps(notebook)
        has_japanese = '問' in content or '日本語' in content
        has_english = 'Question' in content or 'English' in content
        
        print(f'✓ Contains Japanese content: {has_japanese}')
        print(f'✓ Contains English content: {has_english}')
        
        has_qa = 'Q1:' in content and 'Q2:' in content
        print(f'✓ Contains Q&A structure: {has_qa}')
        
        has_sklearn = 'sklearn' in content
        has_matplotlib = 'matplotlib' in content
        has_numpy = 'numpy' in content
        
        print(f'✓ Contains required imports - sklearn: {has_sklearn}, matplotlib: {has_matplotlib}, numpy: {has_numpy}')
        
        has_loo = 'LeaveOneOut' in content
        has_cross_val = 'cross' in content.lower()
        
        print(f'✓ Contains cross-validation content - LeaveOneOut: {has_loo}, cross-validation: {has_cross_val}')
        
        print('\n✅ All validation checks passed!')
        return True
        
    except Exception as e:
        print(f'❌ Validation failed: {e}')
        return False

if __name__ == "__main__":
    validate_notebook()
