#!/usr/bin/env python3
"""
Script to create enhanced MRF notebook with integrated original image estimation
MRF原画像推定を統合した拡張MRFノートブック作成スクリプト
"""

import nbformat
import os

def create_enhanced_mrf_notebook():
    """
    Create enhanced MRF notebook with embedded estimation algorithms
    推定アルゴリズムを埋め込んだ拡張MRFノートブックを作成
    """
    print("Creating enhanced MRF notebook with original image estimation...")
    print("原画像推定機能付き拡張MRFノートブックを作成中...")
    
    with open("03_mrf_analysis.ipynb", 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    matrix_estimator_code = '''# Matrix-based Original Image Estimation Algorithm / 行列ベース原画像推定アルゴリズム
class MatrixEstimator:
    """
    Matrix-based MRF image estimation using direct matrix inversion
    直接行列逆変換を用いた行列ベースMRF画像推定
    
    The algorithm solves: (I + λ*L)*u = v
    where L is the Laplacian matrix for smoothness regularization
    アルゴリズムは (I + λ*L)*u = v を解きます
    ここでLは平滑化正則化のためのラプラシアン行列です
    """
    
    def __init__(self, N):
        self.N = N
        self.mat = self._prepare_matrix()
    
    def _prepare_matrix(self):
        """
        Prepare the Laplacian matrix for MRF regularization
        MRF正則化のためのラプラシアン行列を準備
        """
        mat = np.zeros((self.N, self.N))
        mat += np.eye(self.N) * 2  # Diagonal elements / 対角要素
        mat[:-1, 1:] -= np.eye(self.N - 1)  # Upper diagonal / 上対角
        mat[1:, :-1] -= np.eye(self.N - 1)  # Lower diagonal / 下対角
        mat[0, 0] -= 1  # Boundary condition / 境界条件
        mat[-1, -1] -= 1  # Boundary condition / 境界条件
        return mat
    
    def estimate(self, v, lambda_):
        """
        Estimate original image using matrix inversion
        行列逆変換を用いた原画像推定
        
        Parameters:
        v: observed noisy image / 観測されたノイズ画像
        lambda_: regularization parameter / 正則化パラメータ
        
        Returns:
        u_est: estimated original image / 推定された原画像
        """
        inv = np.linalg.inv(np.eye(self.N) + lambda_ * self.mat)
        u_est = np.dot(inv, v)
        return u_est

print("✓ Matrix Estimator class defined")
print("✓ 行列推定器クラスが定義されました")'''

    gradient_estimator_code = '''# Gradient-based Original Image Estimation Algorithm / 勾配ベース原画像推定アルゴリズム
class GradientEstimator:
    """
    Gradient descent-based MRF image estimation
    勾配降下法ベースMRF画像推定
    
    Iteratively minimizes the energy function:
    E(u) = ||u - v||² + λ * Σ(u[i] - u[i+1])²
    エネルギー関数を反復的に最小化:
    E(u) = ||u - v||² + λ * Σ(u[i] - u[i+1])²
    """
    
    def __init__(self):
        pass
    
    def estimate(self, v, lambda_, alpha=0.01, max_iter=1000, tol=1e-4, verbose=False):
        """
        Estimate original image using gradient descent
        勾配降下法を用いた原画像推定
        
        Parameters:
        v: observed noisy image / 観測されたノイズ画像
        lambda_: regularization parameter / 正則化パラメータ
        alpha: learning rate / 学習率
        max_iter: maximum iterations / 最大反復回数
        tol: convergence tolerance / 収束許容値
        verbose: print convergence info / 収束情報を表示
        
        Returns:
        u_est: estimated original image / 推定された原画像
        """
        N = len(v)
        est = np.random.randn(N)  # Random initialization / ランダム初期化
        
        convergence = False
        for iteration in range(max_iter):
            grad = est - v  # Data fidelity term / データ忠実度項
            smooth = est[:-1] - est[1:]  # Smoothness term / 平滑度項
            grad[:-1] += lambda_ * smooth
            grad[1:] -= lambda_ * smooth
            
            est -= alpha * grad
            
            if verbose and iteration % 100 == 0:
                energy = np.sum((est - v)**2) + lambda_ * np.sum(smooth**2)
                print(f"Iteration {iteration}: Energy = {energy:.6f}")
                print(f"反復 {iteration}: エネルギー = {energy:.6f}")
            
            if alpha * np.linalg.norm(grad) / N < tol:
                convergence = True
                break
        
        if not convergence:
            print("⚠️ Warning: Estimation did not converge")
            print("⚠️ 警告: 推定が収束しませんでした")
        else:
            print(f"✓ Converged after {iteration+1} iterations")
            print(f"✓ {iteration+1}回の反復後に収束しました")
        
        return est

print("✓ Gradient Estimator class defined")
print("✓ 勾配推定器クラスが定義されました")'''

    estimation_section_code = '''# Original Image Estimation / 原画像推定
print("Starting original image estimation / 原画像推定を開始します")
print("=" * 60)

matrix_est = MatrixEstimator(N)
gradient_est = GradientEstimator()

lambda_values = [0.1, 1.0, 10.0, 50.0]
estimation_results = {}

for lambda_val in lambda_values:
    print(f"\\nTesting λ = {lambda_val}")
    print(f"λ = {lambda_val} をテスト中")
    
    start_time = time.time()
    u_matrix = matrix_est.estimate(v_observed, lambda_val)
    matrix_time = time.time() - start_time
    
    start_time = time.time()
    u_gradient = gradient_est.estimate(v_observed, lambda_val, verbose=False)
    gradient_time = time.time() - start_time
    
    estimation_results[lambda_val] = {
        'matrix': u_matrix,
        'gradient': u_gradient,
        'matrix_time': matrix_time,
        'gradient_time': gradient_time
    }
    
    matrix_mse = np.mean((u_matrix - u_true)**2)
    gradient_mse = np.mean((u_gradient - u_true)**2)
    
    print(f"  Matrix method: MSE = {matrix_mse:.6f}, Time = {matrix_time:.4f}s")
    print(f"  Gradient method: MSE = {gradient_mse:.6f}, Time = {gradient_time:.4f}s")
    print(f"  行列法: MSE = {matrix_mse:.6f}, 時間 = {matrix_time:.4f}秒")
    print(f"  勾配法: MSE = {gradient_mse:.6f}, 時間 = {gradient_time:.4f}秒")

print("\\n✓ Original image estimation completed")
print("✓ 原画像推定が完了しました")'''

    visualization_code = '''# Visualization of Estimation Results / 推定結果の可視化
plt.figure(figsize=(20, 15))

for i, lambda_val in enumerate(lambda_values):
    results = estimation_results[lambda_val]
    
    plt.subplot(4, 4, i*4 + 1)
    plt.plot(u_true, 'b-', linewidth=2, label='True Image')
    plt.plot(results['matrix'], 'r--', linewidth=2, label='Matrix Estimation')
    plt.plot(v_observed, 'g:', alpha=0.5, label='Observed')
    plt.title(f'Matrix Method (λ={lambda_val})')
    plt.xlabel('Pixel Index')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, i*4 + 2)
    plt.plot(u_true, 'b-', linewidth=2, label='True Image')
    plt.plot(results['gradient'], 'orange', linestyle='--', linewidth=2, label='Gradient Estimation')
    plt.plot(v_observed, 'g:', alpha=0.5, label='Observed')
    plt.title(f'Gradient Method (λ={lambda_val})')
    plt.xlabel('Pixel Index')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, i*4 + 3)
    matrix_error = results['matrix'] - u_true
    gradient_error = results['gradient'] - u_true
    plt.plot(matrix_error, 'r-', label='Matrix Error')
    plt.plot(gradient_error, 'orange', label='Gradient Error')
    plt.title(f'Estimation Errors (λ={lambda_val})')
    plt.xlabel('Pixel Index')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.subplot(4, 4, i*4 + 4)
    methods = ['Matrix', 'Gradient']
    mse_values = [
        np.mean((results['matrix'] - u_true)**2),
        np.mean((results['gradient'] - u_true)**2)
    ]
    times = [results['matrix_time'], results['gradient_time']]
    
    x = np.arange(len(methods))
    plt.bar(x - 0.2, mse_values, 0.4, label='MSE', alpha=0.7)
    plt.bar(x + 0.2, np.array(times)*1000, 0.4, label='Time (ms)', alpha=0.7)
    plt.title(f'Performance (λ={lambda_val})')
    plt.xlabel('Method')
    plt.ylabel('Value')
    plt.xticks(x, methods)
    plt.legend()
    plt.yscale('log')

plt.tight_layout()
plt.show()

print("\\nSummary of Estimation Results / 推定結果の要約")
print("=" * 60)
print(f"{'Lambda':<8} {'Matrix MSE':<12} {'Gradient MSE':<14} {'Matrix Time':<12} {'Gradient Time':<14}")
print("-" * 70)

for lambda_val in lambda_values:
    results = estimation_results[lambda_val]
    matrix_mse = np.mean((results['matrix'] - u_true)**2)
    gradient_mse = np.mean((results['gradient'] - u_true)**2)
    print(f"{lambda_val:<8.1f} {matrix_mse:<12.6f} {gradient_mse:<14.6f} {results['matrix_time']:<12.4f} {results['gradient_time']:<14.4f}")

optimal_lambda_matrix = min(lambda_values, key=lambda l: np.mean((estimation_results[l]['matrix'] - u_true)**2))
optimal_lambda_gradient = min(lambda_values, key=lambda l: np.mean((estimation_results[l]['gradient'] - u_true)**2))

print(f"\\n✓ Optimal λ for Matrix method: {optimal_lambda_matrix}")
print(f"✓ Optimal λ for Gradient method: {optimal_lambda_gradient}")
print(f"✓ 行列法の最適λ: {optimal_lambda_matrix}")
print(f"✓ 勾配法の最適λ: {optimal_lambda_gradient}")'''

    new_cells = [
        nbformat.v4.new_markdown_cell("""

In this section, we implement and compare two different approaches for original image estimation using Markov Random Field (MRF) models:

このセクションでは、マルコフ確率場（MRF）モデルを用いた原画像推定の2つの異なるアプローチを実装し比較します：

1. **Matrix-based approach / 行列ベースアプローチ**: Direct matrix inversion method / 直接行列逆変換法
2. **Gradient-based approach / 勾配ベースアプローチ**: Iterative gradient descent method / 反復勾配降下法

Both methods aim to solve the optimization problem:
両手法とも以下の最適化問題を解くことを目的としています：

$$\\min_u \\|u - v\\|^2 + \\lambda \\sum_{i} (u_i - u_{i+1})^2$$

Where:
- $u$ is the estimated original image / 推定された原画像
- $v$ is the observed noisy image / 観測されたノイズ画像  
- $\\lambda$ is the regularization parameter / 正則化パラメータ
"""),
        nbformat.v4.new_code_cell(matrix_estimator_code),
        nbformat.v4.new_code_cell(gradient_estimator_code),
        nbformat.v4.new_code_cell(estimation_section_code),
        nbformat.v4.new_code_cell(visualization_code)
    ]
    
    notebook.cells.extend(new_cells)
    
    with open("03_mrf_analysis.ipynb", 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    
    print("✓ Enhanced MRF notebook created successfully!")
    print("✓ 拡張MRFノートブックが正常に作成されました！")
    
    return True

if __name__ == "__main__":
    success = create_enhanced_mrf_notebook()
    if success:
        print("\n🎉 MRF notebook enhancement completed!")
        print("🎉 MRFノートブック拡張が完了しました！")
    else:
        print("\n❌ Failed to enhance MRF notebook")
        print("❌ MRFノートブック拡張に失敗しました")
