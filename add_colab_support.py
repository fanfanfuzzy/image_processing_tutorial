#!/usr/bin/env python3
"""
Script to add Google Colab support to all tutorial notebooks
全チュートリアルノートブックにGoogle Colabサポートを追加するスクリプト
"""

import nbformat
import os
from pathlib import Path

def create_colab_setup_cell():
    """
    Create the Google Colab setup cell content
    Google Colabセットアップセルの内容を作成
    """
    colab_setup_code = '''# Google Colab Setup / Google Colab セットアップ
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    
    import sys
    from pathlib import Path
    
    proj_root = Path("/content/drive/MyDrive/image_processing_tutorial-main")
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    
    print("Google Colab environment detected and configured!")
    print("Google Colab環境が検出され、設定されました！")
else:
    print("Local Jupyter environment detected")
    print("ローカルJupyter環境が検出されました")'''
    
    return nbformat.v4.new_code_cell(colab_setup_code)

def add_colab_support_to_notebook(notebook_path):
    """
    Add Google Colab support to a specific notebook
    特定のノートブックにGoogle Colabサポートを追加
    
    Parameters:
    notebook_path: path to the notebook file / ノートブックファイルのパス
    """
    print(f"Processing notebook: {notebook_path}")
    print(f"ノートブック処理中: {notebook_path}")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        colab_cell = create_colab_setup_cell()
        
        if notebook.cells and notebook.cells[0].cell_type == 'code':
            if 'Google Colab Setup' in notebook.cells[0].source:
                print(f"  ✓ Colab setup already exists in {notebook_path}")
                print(f"  ✓ {notebook_path}にColabセットアップが既に存在します")
                return
        
        notebook.cells.insert(0, colab_cell)
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        
        print(f"  ✓ Successfully added Colab support to {notebook_path}")
        print(f"  ✓ {notebook_path}にColabサポートを正常に追加しました")
        
    except Exception as e:
        print(f"  ❌ Error processing {notebook_path}: {e}")
        print(f"  ❌ {notebook_path}の処理中にエラー: {e}")

def main():
    """
    Main function to add Colab support to all notebooks
    全ノートブックにColabサポートを追加するメイン関数
    """
    print("Adding Google Colab Support to Tutorial Notebooks")
    print("チュートリアルノートブックにGoogle Colabサポートを追加中")
    print("=" * 60)
    
    notebook_files = [
        "01_image_filtering.ipynb",
        "02_machine_learning_basics.ipynb", 
        "03_mrf_analysis.ipynb",
        "04_perceptron.ipynb",
        "05_multilayer_perceptron.ipynb"
    ]
    
    for notebook_file in notebook_files:
        if os.path.exists(notebook_file):
            add_colab_support_to_notebook(notebook_file)
        else:
            print(f"  ⚠️  Notebook not found: {notebook_file}")
            print(f"  ⚠️  ノートブックが見つかりません: {notebook_file}")
    
    print("\n" + "=" * 60)
    print("Google Colab support addition completed!")
    print("Google Colabサポートの追加が完了しました！")

if __name__ == "__main__":
    main()
