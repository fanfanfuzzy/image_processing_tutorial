#!/usr/bin/env python3
"""
Script to create updated machine learning notebook following reference format
参考フォーマットに従って機械学習ノートブックを更新するスクリプト
"""

import nbformat as nbf
import json

def create_updated_ml_notebook():
    """
    Create updated machine learning notebook with Q&A format
    Q&A形式で更新された機械学習ノートブックを作成
    """
    
    nb = nbf.v4.new_notebook()
    
    title_cell = nbf.v4.new_markdown_cell("""# Machine Learning Basics: Linear Regression and Model Selection

This notebook demonstrates fundamental machine learning concepts using linear regression with f=ma relationship data. We will explore model selection using cross-validation.

このノートブックでは、f=ma関係のデータを使用した線形回帰により、機械学習の基本概念を実演します。交差検証を使用したモデル選択を探求します。

- Generate artificial data based on physical relationships / 物理的関係に基づく人工データの生成
- Implement linear regression with and without intercept / 切片ありなしの線形回帰の実装
- Understand cross-validation for model evaluation / モデル評価のための交差検証の理解
- Compare model performance using statistical metrics / 統計的指標を使用したモデル性能の比較""")
    
    import_cell = nbf.v4.new_code_cell("""# Import required libraries / 必要なライブラリをインポート
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model  # sklearn.linear_model.LinearRegression
from sklearn.model_selection import LeaveOneOut  # cross validation
from sklearn.metrics import mean_squared_error  # RMSE

np.random.seed(42)""")
    
    data_gen_cell = nbf.v4.new_markdown_cell("""## 1. Artificial Data Generation / 人工データ生成

We will generate artificial data following a linear relationship similar to Newton's second law (F = ma).
ニュートンの第二法則（F = ma）に類似した線形関係に従って人工データを生成します。""")
    
    data_gen_code = nbf.v4.new_code_cell("""# Artificial data generation

N = 21  # # of samples
a = 1
X = np.linspace(-10, 10, N)  # -10, 10 to N equally divided data.

noise_cov = 3
noise = np.random.normal(0, noise_cov, [N])
y = a * X + noise

print(f"Generated {N} data points with linear relationship y = {a} * X + noise")
print(f"線形関係 y = {a} * X + ノイズ で {N} 個のデータポイントを生成")""")
    
    q1_cell = nbf.v4.new_markdown_cell("""### Q1: Plot the artificial data / 問１：人工データのプロットしてください

Plot the generated data points to visualize the linear relationship with noise.
ノイズを含む線形関係を可視化するために生成されたデータポイントをプロットしてください。""")
    
    q1_code = nbf.v4.new_code_cell("""# Q1: Plot the artificial data.
plt.plot(X, y, 'r.')
plt.axis('equal')
plt.title('DATA')
plt.xlabel('X')
plt.ylabel('y')
plt.show()""")
    
    model_setup_cell = nbf.v4.new_markdown_cell("""## 2. Linear Regression Model Setup / 線形回帰モデルの設定

We will create two different linear models:
1. Model with intercept: y = ax + b
2. Model without intercept: y = ax

2つの異なる線形モデルを作成します：
1. 切片ありモデル：y = ax + b  
2. 切片なしモデル：y = ax""")
    
    q2_cell = nbf.v4.new_markdown_cell("""### Q2: Generate linear models with and without intercept / 問2: １次式のモデルのうち，切片があるモデルとないモデルをそれぞれ生成してください

Create two LinearRegression models using scikit-learn with different intercept settings.
異なる切片設定でscikit-learnを使用して2つのLinearRegressionモデルを作成してください。""")
    
    q2_code = nbf.v4.new_code_cell("""# Q2: Generate a linear model with and without intercept, respectively.
reg_ax_b = linear_model.LinearRegression(fit_intercept=True)  ## (option to generate a model with an intercept)
reg_ax = linear_model.LinearRegression(fit_intercept=False)   ## (option to generate a model with no intercept)

print("Created two models:")
print("2つのモデルを作成:")
print("- reg_ax_b: Linear regression with intercept (切片あり)")
print("- reg_ax: Linear regression without intercept (切片なし)")""")
    
    cv_section_cell = nbf.v4.new_markdown_cell("""## 3. Cross-Validation Setup / 交差検証の設定

We will use Leave-One-Out cross-validation to evaluate model performance.
Leave-One-Out交差検証を使用してモデル性能を評価します。""")
    
    cv_setup_code = nbf.v4.new_code_cell("""# Evaluate performance with K-partition cross validation

loo = LeaveOneOut()  # Generating training and test data set for leave one out cross validation
MSE_ax = {"mean": [], "std": []}
MSE_ax_b = {"mean": [], "std": []}

scores_ax = []
scores_ax_b = []

print("Leave-One-Out cross-validation setup complete")
print("Leave-One-Out交差検証の設定完了")""")
    
    q3_cell = nbf.v4.new_markdown_cell("""### Q3: Print train_index and test_index / 問3: printをもちいて，train_index, test_indexをプリントし、テストデータとトレーニングデータを確認してください

Examine the cross-validation splits to understand how Leave-One-Out works.
Leave-One-Outの動作を理解するために交差検証の分割を調べてください。""")
    
    q3_code = nbf.v4.new_code_cell("""# Q3: Print train_index and test_index using print and check the test and training data.

for train_index, test_index in loo.split(X):
    print("%s %s" % (train_index, test_index))""")
    
    q4_q5_cell = nbf.v4.new_markdown_cell("""### Q4 & Q5: Model Training and Prediction / 問４・５：モデル学習と予測

**Q4**: Train linear regression models using training data / 線形回帰のモデルをトレーニングデータを用いて学習してください

**Q5**: Make predictions on test data using trained models / 学習済みモデルを用いてテストデータについて予測してください""")
    
    q4_q5_code = nbf.v4.new_code_cell("""# main part: leave one out cross validation
for train_index, test_index in loo.split(X):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    reg_ax_b.fit(X_train[:, np.newaxis], y_train[:, np.newaxis])
    reg_ax.fit(X_train[:, np.newaxis], y_train[:, np.newaxis])
    
    pred_ax_b = reg_ax_b.predict(X_test[:, np.newaxis])
    pred_ax = reg_ax.predict(X_test[:, np.newaxis])
    
    scores_ax_b.append(mean_squared_error(y_test, pred_ax_b))
    scores_ax.append(mean_squared_error(y_test, pred_ax))

MSE_ax_b["mean"].append(np.mean(scores_ax_b))  # convert to positive
MSE_ax["mean"].append(np.mean(scores_ax))

print('Cross-validation error when y= ax+b as model: %.4f' % MSE_ax_b["mean"][0])
print('Cross-validation error when y= ax as model: %.4f' % MSE_ax["mean"][0])
print('y= ax+bをモデルとしたときの交差検証誤差： %.4f' % MSE_ax_b["mean"][0])
print('y= axをモデルとしたときの交差検証誤差： %.4f' % MSE_ax["mean"][0])""")
    
    results_cell = nbf.v4.new_markdown_cell("""## 4. Results Analysis / 結果分析

Compare the cross-validation errors between the two models to determine which performs better.
2つのモデル間の交差検証誤差を比較して、どちらが優れた性能を示すかを判定します。

- **Model 1 (y = ax)**: Linear regression without intercept / 切片なし線形回帰
- **Model 2 (y = ax + b)**: Linear regression with intercept / 切片あり線形回帰

The model with lower cross-validation error is considered better for this dataset.
交差検証誤差が低いモデルがこのデータセットに対してより良いと考えられます。""")
    
    conclusion_cell = nbf.v4.new_markdown_cell("""## 5. Conclusion / 結論

Based on the cross-validation results, we can make an informed decision about model selection:

交差検証結果に基づいて、モデル選択について情報に基づいた決定を行うことができます：

1. **Data Generation**: Created artificial data with known linear relationship / データ生成：既知の線形関係を持つ人工データを作成
2. **Model Comparison**: Compared models with and without intercept / モデル比較：切片ありなしのモデルを比較
3. **Cross-Validation**: Used Leave-One-Out CV for robust evaluation / 交差検証：堅牢な評価のためにLeave-One-Out CVを使用
4. **Model Selection**: Selected best model based on CV error / モデル選択：CV誤差に基づいて最適モデルを選択

This demonstrates the fundamental machine learning workflow of data generation, model training, evaluation, and selection.
これは、データ生成、モデル学習、評価、選択という機械学習の基本的なワークフローを実演しています。""")
    
    nb.cells = [
        title_cell,
        import_cell,
        data_gen_cell,
        data_gen_code,
        q1_cell,
        q1_code,
        model_setup_cell,
        q2_cell,
        q2_code,
        cv_section_cell,
        cv_setup_code,
        q3_cell,
        q3_code,
        q4_q5_cell,
        q4_q5_code,
        results_cell,
        conclusion_cell
    ]
    
    return nb

def main():
    """
    Main function to create and save the updated notebook
    更新されたノートブックを作成・保存するメイン関数
    """
    print("Creating updated machine learning notebook...")
    print("更新された機械学習ノートブックを作成中...")
    
    notebook = create_updated_ml_notebook()
    
    with open('02_machine_learning_basics_updated.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    
    print("✓ Created 02_machine_learning_basics_updated.ipynb")
    print("✓ 02_machine_learning_basics_updated.ipynb を作成しました")
    
    return True

if __name__ == "__main__":
    main()
