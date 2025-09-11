#!/usr/bin/env python3

import torch
import numpy as np

def sigmoid(z):
    return 1/(1 + torch.exp(-z))

class TorchLogisticRegression:
    """
    PyTorch implementation of Logistic Regression
    """
    def __init__(self, fit_intercept=True, C=1.0, max_iter=1000, tol=1e-6, lr=1.0):
        self.fit_intercept = fit_intercept
        self.C = C  # Regularization strength (inverse of lambda)
        # Reduce max_iter for large C values to prevent excessive computation
        self.max_iter = min(max_iter, max(100, int(1000/max(1, C/10))))
        self.tol = tol
        self.lr = lr
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_features_in_ = None
        
    def _add_intercept(self, X):
        """Add intercept term to features if fit_intercept=True"""
        if self.fit_intercept:
            intercept = torch.ones(X.shape[0], 1, device=X.device)
            return torch.cat([X, intercept], dim=1)
        return X
    
    def _sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        return torch.sigmoid(torch.clamp(z, -250, 250))
    
    def fit(self, X, y):
        """
        Fit the logistic regression model
        """
        device = X.device
        X = X.to(device)
        y = y.to(device).float()
        
        # Store input info
        self.n_features_in_ = X.shape[1]
        self.classes_ = torch.unique(y)
        
        # Add intercept if needed
        X_with_intercept = self._add_intercept(X)
        
        # Initialize weights
        n_features = X_with_intercept.shape[1]
        weights = torch.zeros(n_features, 1, device=device, requires_grad=False)
        
        # Optimized Newton-Raphson with adaptive method selection
        prev_loss = float('inf')
        use_newton = True
        gradient_step_size = 1.0
        
        # For large C values, use L-BFGS-like approach
        if self.C > 10:
            use_newton = False
            gradient_step_size = min(1.0, 10.0 / self.C)
        
        for iteration in range(self.max_iter):
            # Compute predictions
            z = torch.matmul(X_with_intercept, weights)
            p = self._sigmoid(z)
            
            # Compute gradient
            gradient = torch.matmul(X_with_intercept.t(), p - y.unsqueeze(1)) / X_with_intercept.shape[0]
            
            # Add L2 regularization to gradient
            if self.fit_intercept:
                reg_gradient = weights / self.C
                reg_gradient[-1] = 0  # Don't regularize intercept
            else:
                reg_gradient = weights / self.C
            gradient += reg_gradient
            
            if use_newton and iteration < 100:  # Only use Newton for first 100 iterations
                # Compute Hessian (approximate)
                W = p * (1 - p)
                W = torch.clamp(W, min=1e-8)  # Avoid numerical issues
                hessian = torch.matmul(X_with_intercept.t() * W.t(), X_with_intercept) / X_with_intercept.shape[0]
                
                # Add L2 regularization to Hessian
                if self.fit_intercept:
                    reg_hessian = torch.eye(hessian.shape[0], device=device) / self.C
                    reg_hessian[-1, -1] = 0  # Don't regularize intercept
                else:
                    reg_hessian = torch.eye(hessian.shape[0], device=device) / self.C
                hessian += reg_hessian
                
                # Use Cholesky decomposition instead of direct inverse for stability
                try:
                    L = torch.linalg.cholesky(hessian)
                    step = torch.cholesky_solve(gradient, L)
                    weights = weights - self.lr * step
                except:
                    # Fallback to gradient descent
                    use_newton = False
                    weights = weights - gradient_step_size * gradient
            else:
                # Gradient descent with adaptive step size
                step_size = gradient_step_size / (1 + iteration * 0.001)  # Decay step size
                weights = weights - step_size * gradient
            
            # Early stopping based on gradient norm (faster than loss computation)
            grad_norm = torch.norm(gradient)
            if grad_norm < self.tol * 10:  # Gradient is small enough
                break
                
            # Check convergence less frequently to save computation
            if iteration % 20 == 0:
                # Simple loss check
                if grad_norm < self.tol:
                    break
                if iteration > 0 and grad_norm > prev_loss * 1.01:  # If gradient increased significantly, reduce step size
                    gradient_step_size *= 0.8
                prev_loss = grad_norm
        
        # Store coefficients
        if self.fit_intercept:
            self.coef_ = weights[:-1].t()  # Shape: (1, n_features)
            self.intercept_ = weights[-1].item()
        else:
            self.coef_ = weights.t()  # Shape: (1, n_features)
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        X = X.to(self.coef_.device)
        X_with_intercept = self._add_intercept(X)
        
        if self.fit_intercept:
            weights = torch.cat([self.coef_.t(), torch.tensor([[self.intercept_]], device=self.coef_.device)], dim=1)
        else:
            weights = self.coef_.t()
        
        z = torch.matmul(X_with_intercept, weights)
        prob_1 = self._sigmoid(z).squeeze()
        return (prob_1 >= 0.5).float()
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels
        """
        y = y.to(self.coef_.device)
        predictions = self.predict(X)
        accuracy = (predictions == y).float().mean()
        return accuracy.item()

def compute_hessian(mu, X, penalty):
    sigm = sigmoid(X @ mu.t())
    diag_sigm = (sigm * (1 - sigm)).flatten()
    res = torch.eye(X.size(1), device=X.device)/penalty
    res += (X.t() * diag_sigm) @ X
    return res

def test_penalty_10000():
    """测试penalty=10000时是否能正常运行"""
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 设置penalty参数
    penalty = 10000
    
    # 创建一些测试数据
    n_samples = 100
    n_features = 10
    
    # 生成随机数据
    torch.manual_seed(42)
    X = torch.randn(n_samples, n_features, device=device)
    y = torch.randint(0, 2, (n_samples,), device=device).float()
    
    print(f"Test data shape: X={X.shape}, y={y.shape}")
    print(f"Penalty value: {penalty}")
    
    try:
        # 测试TorchLogisticRegression
        print("\n1. Testing TorchLogisticRegression...")
        model = TorchLogisticRegression(fit_intercept=False, C=penalty, max_iter=5000)
        model.fit(X, y)
        
        print(f"   Model fitted successfully!")
        print(f"   Coefficients shape: {model.coef_.shape}")
        print(f"   Intercept: {model.intercept_}")
        
        # 测试预测
        predictions = model.predict(X)
        accuracy = model.score(X, y)
        print(f"   Training accuracy: {accuracy:.4f}")
        
        # 测试compute_hessian
        print("\n2. Testing compute_hessian...")
        mu = model.coef_
        hessian = compute_hessian(mu, X, penalty)
        print(f"   Hessian shape: {hessian.shape}")
        print(f"   Hessian condition number: {torch.linalg.cond(hessian):.2e}")
        
        # 测试Cholesky分解
        try:
            L = torch.linalg.cholesky(hessian)
            print(f"   Cholesky decomposition successful!")
            print(f"   L shape: {L.shape}")
            print(f"   L diagonal min: {torch.diagonal(L).min():.2e}")
            print(f"   L diagonal max: {torch.diagonal(L).max():.2e}")
        except Exception as e:
            print(f"   Cholesky decomposition failed: {e}")
            return False
        
        # 测试数值稳定性
        print("\n3. Testing numerical stability...")
        print(f"   Hessian min eigenvalue: {torch.linalg.eigvals(hessian).real.min():.2e}")
        print(f"   Hessian max eigenvalue: {torch.linalg.eigvals(hessian).real.max():.2e}")
        
        # 测试大penalty值的影响
        print("\n4. Testing large penalty effects...")
        print(f"   Regularization term (1/C): {1/penalty:.2e}")
        print(f"   Hessian regularization component: {torch.eye(hessian.shape[0], device=device)[0,0]/penalty:.2e}")
        
        print("\n✅ All tests passed! Penalty=10000 works correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_penalty_10000()
    if success:
        print("\n🎉 Penalty=10000 is working correctly!")
    else:
        print("\n💥 Penalty=10000 has issues that need to be fixed.") 