#!/usr/bin/env python3
"""
Script to create improved image download function for the image filtering notebook
画像フィルタリングノートブック用の改良された画像ダウンロード関数を作成するスクリプト
"""

import nbformat
import os

def create_improved_download_function():
    """Create the improved download function code"""
    improved_code = '''def download_lena_image():
    """
    Download Lena image from multiple reliable sources with fallback
    複数の信頼できるソースからレナ画像をダウンロード（フォールバック付き）
    """
    urls = [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for i, url in enumerate(urls):
        try:
            print(f"Trying source {i+1}/{len(urls)}: {url.split('/')[-1]}")
            print(f"ソース {i+1}/{len(urls)} を試行中: {url.split('/')[-1]}")
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            file_ext = 'jpg' if 'jpg' in url.lower() else 'png'
            filename = f'lena.{file_ext}'
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            img = cv2.imread(filename)
            if img is None:
                raise ValueError(f"Failed to load image from {filename}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            print(f"✓ Lena image downloaded successfully from source {i+1}!")
            print(f"✓ ソース {i+1} からレナ画像のダウンロードが完了しました！")
            print(f"Image shape: {img_rgb.shape}")
            print(f"Source: {url}")
            
            return img_rgb
            
        except Exception as e:
            print(f"❌ Source {i+1} failed: {e}")
            print(f"❌ ソース {i+1} が失敗: {e}")
            if i < len(urls) - 1:
                print("Trying next source... / 次のソースを試行中...")
            continue
    
    print("All download sources failed. Creating synthetic test image.")
    print("全ダウンロードソースが失敗しました。合成テスト画像を作成中。")
    
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    for i in range(0, 512, 64):
        img[i:i+32, :] = [128, 128, 128]  # Horizontal stripes
        img[:, i:i+32] = [64, 64, 64]    # Vertical stripes
    
    center = (256, 256)
    cv2.circle(img, center, 100, (255, 255, 255), -1)
    cv2.circle(img, center, 80, (0, 0, 0), -1)
    cv2.circle(img, center, 60, (200, 200, 200), -1)
    
    print("Synthetic test image created with structured patterns.")
    print("構造化パターンを持つ合成テスト画像を作成しました。")
    
    return img'''
    
    return improved_code

def update_notebook_with_improved_download():
    """Update the image filtering notebook with improved download function"""
    notebook_path = "01_image_filtering.ipynb"
    
    if not os.path.exists(notebook_path):
        print(f"Notebook {notebook_path} not found!")
        return False
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        improved_code = create_improved_download_function()
        
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code' and 'def download_lena_image():' in cell.source:
                lines = cell.source.split('\n')
                new_lines = []
                in_function = False
                indent_level = 0
                
                for line in lines:
                    if 'def download_lena_image():' in line:
                        in_function = True
                        indent_level = len(line) - len(line.lstrip())
                        new_lines.extend(improved_code.split('\n'))
                        continue
                    
                    if in_function:
                        current_indent = len(line) - len(line.lstrip()) if line.strip() else 0
                        if line.strip() and current_indent <= indent_level:
                            in_function = False
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                
                cell.source = '\n'.join(new_lines)
                break
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        
        print(f"✓ Successfully updated {notebook_path} with improved download function")
        return True
        
    except Exception as e:
        print(f"❌ Error updating notebook: {e}")
        return False

if __name__ == "__main__":
    print("Creating improved image download function...")
    print("改良された画像ダウンロード関数を作成中...")
    
    success = update_notebook_with_improved_download()
    
    if success:
        print("✓ Notebook updated successfully!")
        print("✓ ノートブックの更新が完了しました！")
    else:
        print("❌ Failed to update notebook")
        print("❌ ノートブックの更新に失敗しました")
