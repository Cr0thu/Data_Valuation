import numpy as np
import torch
import torch.nn as nn
import logging

class TorchLogisticRegression:
    """
    PyTorch实现的Logistic Regression，使用L-BFGS优化算法
    
    特点：
    - 与sklearn.LogisticRegression兼容的API
    - 使用L-BFGS优化算法
    - 支持L2正则化
    - 支持截距项
    - 数值稳定性优化
    """
    
    def __init__(self, fit_intercept=True, C=1.0, max_iter=5000, tol=1e-4, lr=1.0, device='cpu'):
        """
        初始化TorchLogisticRegression
        
        Args:
            fit_intercept (bool): 是否拟合截距项
            C (float): 正则化强度的倒数 (1/λ)
            max_iter (int): 最大迭代次数
            tol (float): 收敛容差
            lr (float): 学习率（用于梯度下降，L-BFGS中不使用）
            device (str): 计算设备 ('cpu' 或 'cuda')
        """
        self.fit_intercept = fit_intercept
        self.C = C
        self.max_iter = min(max_iter, max(100, int(1000/max(1, C/10))))
        self.tol = tol
        self.lr = lr
        self.coef_ = None
        self.intercept_ = None
        self.device = device
        
    def _add_intercept(self, X):
        """添加截距项到特征矩阵"""
        if self.fit_intercept:
            intercept = torch.ones(X.shape[0], 1, device=X.device)
            return torch.cat([X, intercept], dim=1)
        return X
    
    def fit(self, X, y):
        """
        训练模型
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            
        Returns:
            self: 训练后的模型实例
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        X = X.to(self.device)
        y = y.to(self.device).float()
        
        X_with_intercept = self._add_intercept(X)
        n_features = X_with_intercept.shape[1]
        weights = torch.zeros(n_features, 1, device=self.device, requires_grad=False)
        
        # 使用L-BFGS优化
        weights = self._optimize(X_with_intercept, y, weights)
        
        if self.fit_intercept:
            self.coef_ = weights[:-1].t()
            self.intercept_ = weights[-1].item()
        else:
            self.coef_ = weights.t()
            self.intercept_ = 0.0
        
        return self
    
    def _optimize(self, X, y, weights):
        """
        L-BFGS优化算法实现（sklearn风格）
        
        Args:
            X: 特征矩阵（包含截距项）
            y: 标签向量
            weights: 初始权重
            
        Returns:
            优化后的权重
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # L-BFGS参数
        m = 5  # 存储的向量对数量
        s_list = []  # 存储s向量
        y_list = []  # 存储y向量
        rho_list = []  # 存储rho值
        
        # 预计算一些常量
        H0_diag = 1.0  # 使用标量而不是矩阵
        prev_loss = float('inf')
        no_improvement_count = 0
        
        # 预分配内存
        if self.fit_intercept:
            reg_mask = torch.ones(n_features, 1, device=self.device)
            reg_mask[-1] = 0
        
        for iteration in range(self.max_iter):
            # 计算预测和梯度（向量化）
            z = torch.matmul(X, weights)
            p = torch.sigmoid(torch.clamp(z, -250, 250))
            diff = (p - y.unsqueeze(1)) / n_samples
            gradient = torch.matmul(X.t(), diff)
            
            # 添加L2正则化梯度（与sklearn完全一致）
            if self.fit_intercept:
                gradient += (weights * reg_mask) / self.C
            else:
                gradient += weights / self.C
            
            # 快速收敛检查
            grad_norm = torch.norm(gradient)
            if grad_norm < self.tol:
                break
            
            # 计算搜索方向（优化的两循环算法）
            q = gradient
            alphas = torch.zeros(len(s_list), device=self.device)
            
            # 第一个循环（向量化）
            for i in range(len(s_list) - 1, -1, -1):
                alphas[i] = rho_list[i] * torch.sum(s_list[i] * q)
                q = q - alphas[i] * y_list[i]
            
            # 缩放
            r = q * H0_diag
            
            # 第二个循环（向量化）
            for i in range(len(s_list)):
                beta = rho_list[i] * torch.sum(y_list[i] * r)
                r = r + s_list[i] * (alphas[i] - beta)
            
            search_dir = -r
            
            # 改进的线搜索（Wolfe条件）
            step_size = 1.0
            max_line_search = 20
            c1 = 1e-4  # Armijo条件参数
            c2 = 0.9   # 曲率条件参数
            
            for line_iter in range(max_line_search):
                new_weights = weights + step_size * search_dir
                new_loss = self._compute_loss(X, y.unsqueeze(1), new_weights)
                
                # Armijo条件
                if new_loss <= prev_loss + c1 * step_size * torch.sum(gradient * search_dir):
                    # 计算新梯度用于曲率条件
                    z_new = torch.matmul(X, new_weights)
                    p_new = torch.sigmoid(torch.clamp(z_new, -250, 250))
                    new_grad = torch.matmul(X.t(), (p_new - y.unsqueeze(1)) / n_samples)
                    if self.fit_intercept:
                        new_grad += (new_weights * reg_mask) / self.C
                    else:
                        new_grad += new_weights / self.C
                    
                    # 曲率条件
                    if torch.sum(new_grad * search_dir) >= c2 * torch.sum(gradient * search_dir):
                        break
                
                step_size *= 0.5
                if step_size < 1e-10:
                    step_size = 1e-10
                    break
            
            # 提前停止检查
            if abs(new_loss - prev_loss) < self.tol:
                no_improvement_count += 1
                if no_improvement_count >= 5:  # 增加容忍度
                    break
            else:
                no_improvement_count = 0
            
            # 防止损失爆炸
            if new_loss > 1000:
                logging.warning(f"Loss exploded to {new_loss:.6f}, stopping optimization")
                break
            
            # 更新权重和梯度
            old_weights = weights
            weights = new_weights
            prev_loss = new_loss
            
            # 更新L-BFGS存储（仅在必要时）
            if len(s_list) < m:  # 只在存储未满时更新
                s = weights - old_weights
                
                # 计算新梯度
                z_new = torch.matmul(X, weights)
                p_new = torch.sigmoid(torch.clamp(z_new, -250, 250))
                new_gradient = torch.matmul(X.t(), (p_new - y.unsqueeze(1)) / n_samples)
                
                if self.fit_intercept:
                    new_gradient += (weights * reg_mask) / self.C
                else:
                    new_gradient += weights / self.C
                
                y_grad = new_gradient - gradient
                
                # 检查数值稳定性
                s_dot_y = torch.sum(s * y_grad)
                if s_dot_y > 1e-10:
                    s_list.append(s)
                    y_list.append(y_grad)
                    rho_list.append(1.0 / s_dot_y)
                    
                    # 更新H0（限制范围防止数值不稳定）
                    H0_diag = torch.clamp(s_dot_y / torch.sum(y_grad * y_grad), 1e-6, 1e6)
        
        return weights
    
    def _compute_search_direction(self, gradient, s_list, y_list, rho_list, H0):
        """
        计算L-BFGS搜索方向
        
        Args:
            gradient: 当前梯度
            s_list: s向量列表
            y_list: y向量列表
            rho_list: rho值列表
            H0: 初始Hessian近似
            
        Returns:
            搜索方向
        """
        q = gradient.clone()
        alpha_list = []
        
        # 第一个循环
        for i in range(len(s_list) - 1, -1, -1):
            alpha = rho_list[i] * torch.dot(s_list[i].squeeze(), q.squeeze())
            alpha_list.append(alpha)
            q = q - alpha * y_list[i]
        
        # 应用初始Hessian近似
        r = torch.matmul(H0, q)
        
        # 第二个循环
        for i in range(len(s_list)):
            beta = rho_list[i] * torch.dot(y_list[i].squeeze(), r.squeeze())
            r = r + (alpha_list[-(i+1)] - beta) * s_list[i]
        
        return -r
    
    def _line_search(self, X, y, weights, search_dir, gradient, c1=1e-4, c2=0.9):
        """
        Wolfe条件线搜索
        
        Args:
            X: 特征矩阵
            y: 标签向量
            weights: 当前权重
            search_dir: 搜索方向
            gradient: 当前梯度
            c1: Armijo条件参数
            c2: 曲率条件参数
            
        Returns:
            最优步长
        """
        alpha = 1.0
        max_iter = 20
        
        for _ in range(max_iter):
            # 计算新权重
            new_weights = weights + alpha * search_dir
            
            # 计算新损失
            z_new = torch.matmul(X, new_weights)
            p_new = torch.sigmoid(torch.clamp(z_new, -250, 250))
            loss_new = self._compute_loss(X, y.unsqueeze(1), new_weights)
            
            # 计算新梯度
            new_gradient = torch.matmul(X.t(), p_new - y.unsqueeze(1)) / X.shape[0]
            if self.fit_intercept:
                new_reg_gradient = new_weights / self.C
                new_reg_gradient[-1] = 0
            else:
                new_reg_gradient = new_weights / self.C
            new_gradient += new_reg_gradient
            
            # 检查Wolfe条件
            grad_dot_dir = torch.dot(gradient.squeeze(), search_dir.squeeze())
            new_grad_dot_dir = torch.dot(new_gradient.squeeze(), search_dir.squeeze())
            
            # Armijo条件（充分下降）
            current_loss = self._compute_loss(X, y.unsqueeze(1), weights)
            if loss_new <= current_loss + c1 * alpha * grad_dot_dir:
                # 曲率条件
                if new_grad_dot_dir >= c2 * grad_dot_dir:
                    return alpha
            
            alpha *= 0.5
        
        return alpha
    
    def _compute_loss(self, X, y, weights):
        """
        计算损失函数（与sklearn一致的实现）
        
        Args:
            X: 特征矩阵
            y: 标签向量
            weights: 权重
            
        Returns:
            总损失（二元交叉熵 + L2正则化）
        """
        z = torch.matmul(X, weights)
        p = torch.sigmoid(torch.clamp(z, -250, 250))
        
        # 数值稳定性
        eps = 1e-15
        p = torch.clamp(p, eps, 1 - eps)
        
        # 计算二元交叉熵损失
        bce_loss = -torch.mean(y * torch.log(p) + (1 - y) * torch.log(1 - p))
        
        # 计算L2正则化项（与sklearn一致）
        if self.fit_intercept:
            l2_reg = 0.5 / self.C * torch.sum(weights[:-1] ** 2)  # 不正则化截距项
        else:
            l2_reg = 0.5 / self.C * torch.sum(weights ** 2)
        
        return bce_loss + l2_reg
    
    def predict_proba(self, X):
        """
        预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            概率矩阵 [P(0), P(1)]，与sklearn一致
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        X = X.to(self.device)
        X_with_intercept = self._add_intercept(X)
        
        if self.fit_intercept:
            weights = torch.cat([self.coef_.t(), torch.tensor([[self.intercept_]], device=self.device)], dim=1)
        else:
            weights = self.coef_.t()
        
        z = torch.matmul(X_with_intercept, weights)
        prob_1 = torch.sigmoid(z).squeeze()
        prob_0 = 1 - prob_1
        
        # 确保与sklearn一致的概率顺序：[P(0), P(1)]
        return torch.stack([prob_0, prob_1], dim=1)
    
    def predict(self, X):
        """
        预测类别
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的类别标签
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).float()  # 使用第二列（P(1)）来预测
    
    def score(self, X, y):
        """
        计算准确率
        
        Args:
            X: 特征矩阵
            y: 真实标签
            
        Returns:
            准确率
        """
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        y = y.to(self.device)
        predictions = self.predict(X)
        accuracy = (predictions == y).float().mean()
        return accuracy.item()


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = TorchLogisticRegression(fit_intercept=True, C=1.0, device='cpu')
    
    # 生成示例数据
    X = torch.randn(100, 10)
    y = (torch.randn(100) > 0).float()
    
    # 训练模型
    model.fit(X, y)
    
    # 预测
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    accuracy = model.score(X, y)
    
    print(f"Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}") 