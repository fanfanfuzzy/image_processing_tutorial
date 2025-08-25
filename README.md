# Image Analysis Tutorial / 画像解析チュートリアル

This repository contains a comprehensive Python tutorial for image analysis covering three main areas: image filtering, machine learning basics, and MRF (Markov Random Field) analysis.

このリポジトリには、画像フィルタリング、機械学習の基礎、MRF（マルコフ確率場）解析の3つの主要分野をカバーする包括的なPython画像解析チュートリアルが含まれています。

## Contents / 内容

### 1. Image Filtering / 画像フィルタリング
**Notebook:** `01_image_filtering.ipynb`

- Lena image download and preprocessing / レナ画像のダウンロードと前処理
- Smoothing filters / 平滑化フィルタ
- Edge detection with Sobel filters (vertical, horizontal, diagonal) / ソーベルフィルタによるエッジ検出（縦、横、斜め）
- SIFT (Scale-Invariant Feature Transform) detection / SIFT特徴点検出
- Harris corner detection / ハリスコーナー検出

### 2. Machine Learning Basics / 機械学習の基礎
**Notebook:** `02_machine_learning_basics.ipynb`

- Artificial data generation (f=ma relationship) / 人工データ生成（f=ma関係）
- Linear regression with scikit-learn / scikit-learnによる線形回帰
- Analytical solution for linear regression / 線形回帰の解析解
- Cross-validation for model selection (f=ma vs f=ma+b) / 交差検証によるモデル選択（f=ma vs f=ma+b）

### 3. MRF Analysis / MRF解析
**Notebook:** `03_mrf_analysis.ipynb`

- 1D image data generation / 1次元画像データ生成
- MRF model for image restoration / MRFモデルによる画像修復
- Lambda parameter variation analysis / λパラメータ変動解析
- Matrix and gradient-based estimation methods / 行列および勾配ベース推定手法

## Requirements / 必要条件

```python
numpy>=1.20.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
scikit-image>=0.18.0
opencv-python>=4.5.0
scipy>=1.7.0
jupyter>=1.0.0
requests>=2.25.0
```

## Installation / インストール

```bash
# Clone the repository / リポジトリをクローン
git clone <repository-url>
cd image_analysis_tutorial

# Install dependencies / 依存関係をインストール
pip install -r requirements.txt

# Start Jupyter notebook / Jupyter notebookを起動
jupyter notebook
```

## Usage / 使用方法

1. Start with `01_image_filtering.ipynb` to learn basic image processing techniques
   `01_image_filtering.ipynb`から始めて基本的な画像処理技術を学習

2. Continue with `02_machine_learning_basics.ipynb` for regression analysis fundamentals
   `02_machine_learning_basics.ipynb`で回帰分析の基礎を継続学習

3. Finish with `03_mrf_analysis.ipynb` for advanced image restoration techniques
   `03_mrf_analysis.ipynb`で高度な画像修復技術を完了

## Key Features / 主な特徴

- **Bilingual documentation** (Japanese/English) / **二言語対応**（日本語/英語）
- **Hands-on examples** with real image data / 実際の画像データを使った**実践的な例**
- **Scientific visualization** with proper English labels / 適切な英語ラベルによる**科学的可視化**
- **Complete implementations** from basic to advanced techniques / 基礎から高度な技術まで**完全な実装**

## References / 参考文献

- OpenCV Tutorial: Feature Detection and Description
- Scikit-learn Documentation: Linear Models
- Computer Vision: Algorithms and Applications (Szeliski)
- Pattern Recognition and Machine Learning (Bishop)

## License / ライセンス

This tutorial is provided for educational purposes. Please refer to individual library licenses for usage restrictions.

このチュートリアルは教育目的で提供されています。使用制限については、個々のライブラリのライセンスを参照してください。
