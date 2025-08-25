#!/usr/bin/env python3
"""
Script to create Jupyter notebook files with proper JSON structure
適切なJSON構造でJupyterノートブックファイルを作成するスクリプト
"""

import nbformat as nbf
import os

def create_image_filtering_notebook():
    """Create the image filtering notebook"""
    nb = nbf.v4.new_notebook()
    
    nb.cells.append(nbf.v4.new_markdown_cell("""# Image Filtering Tutorial / 画像フィルタリングチュートリアル

This notebook demonstrates fundamental image processing techniques including smoothing filters, edge detection, and feature detection.

このノートブックでは、平滑化フィルタ、エッジ検出、特徴検出を含む基本的な画像処理技術を実演します。

1. Lena Image Download / レナ画像のダウンロード
2. Smoothing Filters / 平滑化フィルタ
3. Edge Detection with Sobel Filters / ソーベルフィルタによるエッジ検出
4. Harris Corner Detection / ハリスコーナー検出
5. SIFT Feature Detection / SIFT特徴点検出"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Import required libraries / 必要なライブラリをインポート
import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests
from skimage import filters, feature, io
from scipy import ndimage
import os

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Lena Image Download / レナ画像のダウンロード

We'll download the famous Lena image, which is commonly used in image processing tutorials.

画像処理チュートリアルでよく使用される有名なレナ画像をダウンロードします。"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""def download_lena_image():
    \"\"\"
    Download Lena image from a reliable source
    信頼できるソースからレナ画像をダウンロード
    \"\"\"
    url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open('lena.png', 'wb') as f:
            f.write(response.content)
        
        img = cv2.imread('lena.png')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print("Lena image downloaded successfully! / レナ画像のダウンロードが完了しました！")
        print(f"Image shape: {img_rgb.shape}")
        
        return img_rgb
    
    except Exception as e:
        print(f"Error downloading image: {e}")
        print("Creating synthetic test image / 合成テスト画像を作成中")
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return img

lena_img = download_lena_image()
lena_gray = cv2.cvtColor(lena_img, cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(lena_img)
plt.title('Original Lena Image (Color)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(lena_gray, cmap='gray')
plt.title('Grayscale Lena Image')
plt.axis('off')

plt.tight_layout()
plt.show()"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Smoothing Filters / 平滑化フィルタ

Smoothing filters are used to reduce noise and blur images. We'll demonstrate Gaussian and median filters.

平滑化フィルタはノイズを減らし、画像をぼかすために使用されます。ガウシアンフィルタとメディアンフィルタを実演します。"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Apply different smoothing filters
gaussian_3 = cv2.GaussianBlur(lena_gray, (3, 3), 0)
gaussian_9 = cv2.GaussianBlur(lena_gray, (9, 9), 0)
gaussian_15 = cv2.GaussianBlur(lena_gray, (15, 15), 0)
median_5 = cv2.medianBlur(lena_gray, 5)
median_9 = cv2.medianBlur(lena_gray, 9)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(lena_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gaussian_3, cmap='gray')
plt.title('Gaussian Blur (3x3)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(gaussian_9, cmap='gray')
plt.title('Gaussian Blur (9x9)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(gaussian_15, cmap='gray')
plt.title('Gaussian Blur (15x15)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(median_5, cmap='gray')
plt.title('Median Filter (5x5)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(median_9, cmap='gray')
plt.title('Median Filter (9x9)')
plt.axis('off')

plt.tight_layout()
plt.show()"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Edge Detection with Sobel Filters / ソーベルフィルタによるエッジ検出

Sobel filters detect edges by computing gradients in horizontal and vertical directions.

ソーベルフィルタは水平および垂直方向の勾配を計算してエッジを検出します。"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Apply Sobel filters
sobel_x = cv2.Sobel(lena_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(lena_gray, cv2.CV_64F, 0, 1, ksize=3)

sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_direction = np.arctan2(sobel_y, sobel_x)

kernel_diag1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
kernel_diag2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

sobel_diag1 = cv2.filter2D(lena_gray.astype(np.float32), -1, kernel_diag1)
sobel_diag2 = cv2.filter2D(lena_gray.astype(np.float32), -1, kernel_diag2)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(lena_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(np.abs(sobel_x), cmap='gray')
plt.title('Sobel X (Vertical Edges)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(np.abs(sobel_y), cmap='gray')
plt.title('Sobel Y (Horizontal Edges)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title('Sobel Magnitude')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(np.abs(sobel_diag1), cmap='gray')
plt.title('Diagonal Sobel 1')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(np.abs(sobel_diag2), cmap='gray')
plt.title('Diagonal Sobel 2')
plt.axis('off')

plt.tight_layout()
plt.show()"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("""## 4. SIFT Feature Detection / SIFT特徴点検出

SIFT (Scale-Invariant Feature Transform) detects distinctive keypoints that are invariant to scale and rotation.

SIFT（スケール不変特徴変換）は、スケールと回転に不変な特徴的なキーポイントを検出します。"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Create SIFT detector
sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(lena_gray, None)

img_with_keypoints = cv2.drawKeypoints(lena_img, keypoints, None, 
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print(f"Number of SIFT keypoints detected: {len(keypoints)}")
print(f"Descriptor shape: {descriptors.shape if descriptors is not None else 'None'}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(lena_img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_with_keypoints)
plt.title(f'SIFT Keypoints ({len(keypoints)} detected)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(lena_gray, cmap='gray')
x_coords = [kp.pt[0] for kp in keypoints]
y_coords = [kp.pt[1] for kp in keypoints]
plt.scatter(x_coords, y_coords, c='red', s=10, alpha=0.7)
plt.title('SIFT Keypoint Locations')
plt.axis('off')

plt.tight_layout()
plt.show()"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("""## 5. Harris Corner Detection / ハリスコーナー検出

Harris corner detection finds corners by analyzing the local structure of the image.

ハリスコーナー検出は、画像の局所構造を分析してコーナーを見つけます。"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Harris corner detection
harris_corners = cv2.cornerHarris(lena_gray, 2, 3, 0.04)

harris_corners = cv2.dilate(harris_corners, None)

img_harris = lena_img.copy()
img_harris[harris_corners > 0.01 * harris_corners.max()] = [255, 0, 0]

coords = feature.corner_peaks(feature.corner_harris(lena_gray), min_distance=5)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(lena_img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_harris)
plt.title('Harris Corners (OpenCV)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(lena_gray, cmap='gray')
plt.plot(coords[:, 1], coords[:, 0], 'r+', markersize=8)
plt.title(f'Harris Corners (scikit-image): {len(coords)} detected')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"OpenCV Harris corners: {np.sum(harris_corners > 0.01 * harris_corners.max())} detected")
print(f"Scikit-image Harris corners: {len(coords)} detected")"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("""## Summary / まとめ

In this tutorial, we demonstrated various image filtering techniques:

このチュートリアルでは、様々な画像フィルタリング技術を実演しました：


1. **Image Download / 画像ダウンロード**: Successfully downloaded and processed the Lena image
   レナ画像のダウンロードと処理に成功

2. **Smoothing Filters / 平滑化フィルタ**: 
   - Gaussian blur for noise reduction / ノイズ減少のためのガウシアンぼかし
   - Median filter for salt-and-pepper noise / 塩胡椒ノイズのためのメディアンフィルタ

3. **Edge Detection / エッジ検出**: 
   - Sobel filters in X and Y directions / X・Y方向のソーベルフィルタ
   - Diagonal edge detection / 対角エッジ検出
   - Magnitude and direction computation / 大きさと方向の計算

4. **Feature Detection / 特徴検出**: 
   - SIFT keypoints for scale-invariant features / スケール不変特徴のためのSIFTキーポイント
   - Harris corner detection for structural features / 構造的特徴のためのハリスコーナー検出


These techniques form the foundation for:
- Object recognition and tracking / 物体認識と追跡
- Image registration and stitching / 画像レジストレーションと合成
- Computer vision preprocessing / コンピュータビジョン前処理
- Medical image analysis / 医用画像解析"""))
    
    return nb

def create_ml_basics_notebook():
    """Create the machine learning basics notebook"""
    nb = nbf.v4.new_notebook()
    
    nb.cells.append(nbf.v4.new_markdown_cell("""# Machine Learning Basics Tutorial / 機械学習基礎チュートリアル

This notebook demonstrates fundamental machine learning concepts using linear regression with artificial data generation and model selection.

このノートブックでは、人工データ生成とモデル選択を用いた線形回帰による基本的な機械学習概念を実演します。

1. Artificial Data Generation (f=ma) / 人工データ生成（f=ma）
2. Linear Regression with Scikit-learn / Scikit-learnによる線形回帰
3. Analytical Solution / 解析解
4. Model Selection using Cross-validation / 交差検証によるモデル選択
5. Comparison: f=ma vs f=ma+b / 比較：f=ma vs f=ma+b"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Import required libraries / 必要なライブラリをインポート
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

np.random.seed(42)

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Artificial Data Generation (f=ma) / 人工データ生成（f=ma）

We'll generate artificial data based on Newton's second law of motion: F = ma, where:
- F: Force (dependent variable)
- m: Mass (independent variable)  
- a: Acceleration (parameter)

ニュートンの運動第二法則に基づいて人工データを生成します：F = ma、ここで：
- F：力（従属変数）
- m：質量（独立変数）
- a：加速度（パラメータ）"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""def generate_physics_data(n_samples=50, acceleration=9.8, noise_level=0.1, add_bias=False, bias_value=2.0):
    \"\"\"
    Generate artificial data based on F = ma (with optional bias)
    F = ma（オプションのバイアス付き）に基づいて人工データを生成
    \"\"\"
    mass = np.linspace(0.5, 10.0, n_samples)
    
    if add_bias:
        force_true = acceleration * mass + bias_value
        print(f"True relationship: F = {acceleration:.2f} * m + {bias_value:.2f}")
    else:
        force_true = acceleration * mass
        print(f"True relationship: F = {acceleration:.2f} * m")
    
    noise = np.random.normal(0, noise_level * np.mean(force_true), n_samples)
    force_observed = force_true + noise
    
    return mass, force_observed, force_true

print("Generating data for model without bias (F = ma) / バイアスなしモデルのデータ生成（F = ma）")
mass1, force1, force_true1 = generate_physics_data(n_samples=30, acceleration=9.8, noise_level=0.15, add_bias=False)

print("\\nGenerating data for model with bias (F = ma + b) / バイアス付きモデルのデータ生成（F = ma + b）")
mass2, force2, force_true2 = generate_physics_data(n_samples=30, acceleration=9.8, noise_level=0.15, add_bias=True, bias_value=5.0)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.scatter(mass1, force1, alpha=0.7, label='Observed Data')
plt.plot(mass1, force_true1, 'r-', label='True Relationship', linewidth=2)
plt.xlabel('Mass (kg)')
plt.ylabel('Force (N)')
plt.title('Model 1: F = ma (No Bias)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(mass2, force2, alpha=0.7, label='Observed Data', color='orange')
plt.plot(mass2, force_true2, 'r-', label='True Relationship', linewidth=2)
plt.xlabel('Mass (kg)')
plt.ylabel('Force (N)')
plt.title('Model 2: F = ma + b (With Bias)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))
    
    return nb

def create_mrf_analysis_notebook():
    """Create the MRF analysis notebook"""
    nb = nbf.v4.new_notebook()
    
    nb.cells.append(nbf.v4.new_markdown_cell("""# MRF Analysis Tutorial / MRF解析チュートリアル

This notebook demonstrates Markov Random Field (MRF) analysis for 1D image restoration, including parameter optimization and comparison of estimation methods.

このノートブックでは、パラメータ最適化と推定手法の比較を含む、1次元画像修復のためのマルコフ確率場（MRF）解析を実演します。

1. 1D Image Data Generation / 1次元画像データ生成
2. MRF Model Theory / MRFモデル理論
3. Matrix-based Estimation / 行列ベース推定
4. Gradient-based Estimation / 勾配ベース推定
5. Lambda Parameter Analysis / λパラメータ解析
6. Comparative Analysis / 比較解析"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Import required libraries / 必要なライブラリをインポート
import numpy as np
import matplotlib.pyplot as plt
from utils import Image1D, MatrixEstimator, GradientEstimator
import time
from scipy.optimize import minimize_scalar

np.random.seed(42)

plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("""## 1. 1D Image Data Generation / 1次元画像データ生成

We'll generate 1D image data using a random walk model and add noise to simulate real-world conditions.

ランダムウォークモデルを使用して1次元画像データを生成し、実世界の条件をシミュレートするためにノイズを追加します。"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Parameters for image generation
N = 100  # Image length / 画像長
a = 1.0  # Variance parameter for true image / 真の画像の分散パラメータ
b = 0.5  # Variance parameter for noise / ノイズの分散パラメータ
seed = 42  # Random seed / ランダムシード

image_generator = Image1D(N)

u_true = image_generator.generate(a, seed=seed)

v_observed = image_generator.add_noise(u_true, b, seed=seed)

print("Image Generation Results / 画像生成結果")
print("=" * 50)
print(f"Image length: {N} pixels / 画像長: {N} ピクセル")
print(f"True image variance parameter (a): {a}")
print(f"Noise variance parameter (b): {b}")
print(f"Signal-to-noise ratio: {np.var(u_true)/np.var(v_observed - u_true):.2f}")
print(f"True image range: [{u_true.min():.2f}, {u_true.max():.2f}]")
print(f"Observed image range: [{v_observed.min():.2f}, {v_observed.max():.2f}]")

plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.plot(u_true, 'b-', linewidth=2, label='True Image')
plt.plot(v_observed, 'r--', alpha=0.7, label='Observed Image (with noise)')
plt.xlabel('Pixel Index')
plt.ylabel('Intensity')
plt.title('1D Image Data')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
noise = v_observed - u_true
plt.plot(noise, 'g-', alpha=0.7)
plt.xlabel('Pixel Index')
plt.ylabel('Noise Level')
plt.title('Additive Noise')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()"""))
    
    return nb

def main():
    """Main function to create all notebooks"""
    print("Creating Jupyter notebooks...")
    print("Jupyterノートブックを作成中...")
    
    notebooks = {
        '01_image_filtering.ipynb': create_image_filtering_notebook(),
        '02_machine_learning_basics.ipynb': create_ml_basics_notebook(),
        '03_mrf_analysis.ipynb': create_mrf_analysis_notebook()
    }
    
    for filename, notebook in notebooks.items():
        with open(filename, 'w', encoding='utf-8') as f:
            nbf.write(notebook, f)
        print(f"✓ Created {filename}")
    
    print("\nAll notebooks created successfully!")
    print("すべてのノートブックが正常に作成されました！")

if __name__ == "__main__":
    main()
