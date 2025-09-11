import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import time
import warnings
import torchvision.models as models
import torchvision.transforms as transforms
import logging
import os
from datetime import datetime
warnings.filterwarnings("ignore")

# 设置logging
def setup_logging():
    """设置logging，同时输出到控制台和文件"""
    # 创建logs目录
    os.makedirs('logs', exist_ok=True)
    
    # 生成log文件名，包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/torch_vs_sklearn_comparison_{timestamp}.log"
    
    # 配置logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logging.info(f"Log file created: {log_filename}")
    return log_filename

# TorchLogisticRegression类
class TorchLogisticRegression:
    def __init__(self, fit_intercept=True, C=1.0, max_iter=100, tol=1e-8, lr=1.0, device='cpu', optimizer='lbfgs'):
        """
        初始化TorchLogisticRegression
        
        Args:
            fit_intercept: 是否包含截距项
            C: 正则化强度的倒数 (1/lambda)
            max_iter: 最大迭代次数
            tol: 收敛容差
            lr: 学习率（不再使用，保留兼容性）
            device: 计算设备
            optimizer: 优化器类型 ('lbfgs', 'newton')
        """
        self.fit_intercept = fit_intercept
        self.C = C
        # 使用极高精度
        if optimizer == 'lbfgs':
            self.max_iter = 1000  # 设置L-BFGS最大迭代次数为1000
            self.tol = min(tol, 1e-15)  # 极高精度收敛条件
        elif optimizer == 'newton':
            self.max_iter = max(max_iter, 100)  # 牛顿法也需要足够迭代次数
            self.tol = min(tol, 1e-20)  # 极高精度收敛条件
        else:
            self.max_iter = max_iter
            self.tol = tol
        
        self.lr = lr
        self.device = device
        self.optimizer_type = optimizer
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0
        self.loss_history = []
        self.gradient_norm_history = []
        
    def _add_intercept(self, X):
        if self.fit_intercept:
            intercept = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)  # 保持相同精度
            return torch.cat([X, intercept], dim=1)
        return X
    
    def fit(self, X, y, monitor_coef_diff=False, sklearn_model=None):
        """
        训练模型
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            monitor_coef_diff: 是否监控系数差异
            sklearn_model: sklearn模型对象，用于实时对比
            
        Returns:
            self: 训练后的模型实例
        """
        # 确保所有数据都是double精度
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64, device=self.device)
        
        X = X.to(self.device).double()  # 强制使用double精度
        y = y.to(self.device).double()  # 强制使用double精度
        
        X_with_intercept = self._add_intercept(X)
        n_features = X_with_intercept.shape[1]
        
        # 权重也使用double精度
        weights = torch.zeros(n_features, 1, device=self.device, dtype=torch.float64, requires_grad=False)
        logging.info(f"  Independent zero initialization, shape: {weights.shape}, dtype: {weights.dtype}")
        
        # 使用优化算法
        if self.optimizer_type == 'lbfgs':
            weights = self._optimize(X_with_intercept, y, weights, monitor_coef_diff, sklearn_model)
        elif self.optimizer_type == 'newton':
            weights = self._newton_optimize(X_with_intercept, y, weights, monitor_coef_diff, sklearn_model)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}. Choose from 'lbfgs', 'newton'")
        
        if self.fit_intercept:
            self.coef_ = weights[:-1].t()
            self.intercept_ = weights[-1].item()
        else:
            self.coef_ = weights.t()
            self.intercept_ = 0.0
        
        return self
    
    def _optimize(self, X, y, weights, monitor_coef_diff=False, sklearn_model=None):
        """
        使用torch自带的L-BFGS优化器（恢复版本）
        
        Args:
            X: 特征矩阵（包含截距项）
            y: 标签向量
            weights: 初始权重
            monitor_coef_diff: 是否监控系数差异
            sklearn_model: sklearn模型，用于对比
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # 将权重设置为可训练参数
        weights.requires_grad_(True)
        
        # 创建L-BFGS优化器
        optimizer = torch.optim.LBFGS(
            [weights],
            lr=0.01,
            max_iter=200,  # 每次优化步骤的最大迭代次数
            max_eval=None,  # 每次优化步骤的最大函数评估次数
            tolerance_grad=1e-20,  # 梯度容差
            tolerance_change=1e-20,  # 参数变化容差
            history_size=100,  # 存储的向量对数量
            line_search_fn="strong_wolfe"  # 使用强Wolfe线搜索
        )
        
        # 预计算一些常量
        prev_loss = float('inf')
        no_improvement_count = 0
        coef_diff_history = []
        
        # 系数差异监控
        if monitor_coef_diff and sklearn_model is not None:
            logging.info("Monitoring coefficient differences during training...")
        
        def closure():
            """L-BFGS优化器的闭包函数"""
            nonlocal prev_loss
            
            optimizer.zero_grad()
            
            # 计算损失
            loss = self._compute_loss(X, y.unsqueeze(1), weights)
            
            # 反向传播
            loss.backward()
            
            return loss
        
        # 主优化循环
        for iteration in range(self.max_iter):
            # 执行L-BFGS优化步骤
            try:
                loss = optimizer.step(closure)
                
                # 计算当前梯度范数
                if weights.grad is not None:
                    grad_norm = torch.norm(weights.grad).item()
                else:
                    grad_norm = 0.0
                
                # 记录每次迭代的损失和梯度范数
                logging.info(f"  Iteration {iteration}: loss = {loss.item():.10f}, grad_norm = {grad_norm:.10f}")
                
                # 检查收敛
                if abs(prev_loss - loss.item()) < self.tol:
                    no_improvement_count += 1
                    if no_improvement_count >= 5:
                        logging.info(f"  Converged by loss improvement at iteration {iteration}")
                        break
                else:
                    no_improvement_count = 0
                
                # 监控系数差异
                if monitor_coef_diff and sklearn_model is not None:
                    if iteration % 5 == 0 or iteration == 0:
                        current_coef = weights[:-1].detach().cpu().numpy().flatten() if self.fit_intercept else weights.detach().cpu().numpy().flatten()
                        
                        # 找到最接近当前迭代次数的sklearn系数
                        sklearn_iter, sklearn_coef = self._find_closest_sklearn_coef(iteration, sklearn_model.coef_history)
                        
                        coef_diff = np.mean(np.abs(current_coef - sklearn_coef))
                        coef_diff_history.append((iteration, coef_diff, sklearn_iter))
                        
                        logging.info(f"    coef_diff = {coef_diff:.10f} (vs sklearn iter {sklearn_iter})")
                
                prev_loss = loss.item()
                
                # 防止损失爆炸
                if loss.item() > 1000:
                    logging.warning(f"Loss exploded to {loss.item():.10f}, stopping optimization")
                    break
                    
            except Exception as e:
                logging.warning(f"L-BFGS optimization failed at iteration {iteration}: {e}")
                break
        
        # 记录最终迭代次数
        final_iteration = iteration + 1
        
        # 保存系数差异历史和最终迭代次数
        if monitor_coef_diff:
            self.coef_diff_history = coef_diff_history
            self.final_iteration = final_iteration
            logging.info(f"  Final iteration: {final_iteration}")
            logging.info(f"  Final loss: {prev_loss:.10f}")
        
        # 返回最终权重
        return weights.detach()
    
    def _newton_optimize(self, X, y, weights, monitor_coef_diff=False, sklearn_model=None):
        """
        简化的牛顿法优化算法实现（无中间记录）
        
        Args:
            X: 特征矩阵（包含截距项）
            y: 标签向量
            weights: 初始权重
            monitor_coef_diff: 是否监控系数差异
            sklearn_model: sklearn模型，用于对比
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # 预计算一些常量
        no_improvement_count = 0
        
        for iteration in range(self.max_iter):
            # 1. 计算预测值和损失
            z = torch.matmul(X, weights)
            p = torch.sigmoid(torch.clamp(z, -250, 250))
            loss = self._compute_loss(X, y.unsqueeze(1), weights)
            
            # 2. 计算梯度（与sum损失函数保持一致）
            diff = (p - y.unsqueeze(1))  # 移除 / n_samples，因为损失函数使用sum
            gradient = torch.matmul(X.t(), diff)
            
            # 添加L2正则化梯度
            if self.fit_intercept:
                gradient[:-1] += weights[:-1] / self.C
            else:
                gradient += weights / self.C
            
            # 计算梯度范数用于收敛检查
            grad_norm = torch.norm(gradient).item()
            
            # 3. 计算Hessian矩阵
            D = p * (1 - p)  # 对角元素
            D = D.squeeze()
            
            # 添加数值稳定性：防止D过小
            D = torch.clamp(D, min=1e-8, max=0.25)
            
            # 计算X^T * D * X
            XT_D = X.t() * D.unsqueeze(0)  # 广播乘法
            hessian = torch.matmul(XT_D, X)
            
            # 添加正则化项
            if self.fit_intercept:
                hessian[:-1, :-1] += torch.eye(n_features - 1, device=X.device) / self.C
            else:
                hessian += torch.eye(n_features, device=X.device) / self.C
            
            # 4. 计算牛顿步长
            try:
                # 使用Cholesky分解求解线性系统
                L = torch.linalg.cholesky(hessian)
                step = torch.cholesky_solve(-gradient, L)
            except:
                # 如果Cholesky分解失败，使用LU分解
                try:
                    step = torch.linalg.solve(hessian, -gradient)
                except:
                    # 如果都失败，使用伪逆
                    hessian_inv = torch.pinverse(hessian)
                    step = torch.matmul(hessian_inv, -gradient)
            
            # 5. 自适应步长线搜索
            step_norm = torch.norm(step).item()
            if step_norm > 1.0:  # 如果步长太大，进行缩放
                step = step / step_norm
            
            # 尝试不同的步长
            best_step_size = 1.0
            best_loss = loss.item()
            
            for step_size in [1.0, 0.5, 0.1, 0.01]:
                test_weights = weights + step_size * step
                test_z = torch.matmul(X, test_weights)
                test_p = torch.sigmoid(torch.clamp(test_z, -250, 250))
                test_loss = self._compute_loss(X, y.unsqueeze(1), test_weights)
                
                if test_loss.item() < best_loss:
                    best_step_size = step_size
                    best_loss = test_loss.item()
                else:
                    break  # 如果损失不再减少，停止尝试更小的步长
            
            # 6. 更新权重
            weights = weights + best_step_size * step
            
            # 7. 检查收敛（使用梯度范数）
            if grad_norm < self.tol * 100:  # 放宽收敛条件
                no_improvement_count += 1
                if no_improvement_count >= 3:
                    break
            else:
                no_improvement_count = 0
        
        # 只记录最终结果
        logging.info(f"Newton optimization completed, total iterations: {iteration + 1}")
        logging.info(f"Final gradient norm: {grad_norm:.10f}")
        logging.info(f"Final loss: {loss:.10f}")
        
        return weights
    
    def _find_closest_sklearn_coef(self, current_iter, sklearn_coef_history):
        """
        找到最接近当前迭代次数的sklearn系数
        
        Args:
            current_iter: 当前Torch模型的迭代次数
            sklearn_coef_history: sklearn模型的系数历史 [(iter, coef), ...]
            
        Returns:
            (closest_iter, closest_coef): 最接近的迭代次数和对应的系数
        """
        if not sklearn_coef_history:
            return 0, np.zeros(1)
        
        # 找到最接近的迭代次数
        closest_iter = 0
        min_diff = float('inf')
        
        for iter_num, coef in sklearn_coef_history:
            diff = abs(iter_num - current_iter)
            if diff < min_diff:
                min_diff = diff
                closest_iter = iter_num
                closest_coef = coef
        
        return closest_iter, closest_coef.flatten()
    
    def _compute_search_direction(self, gradient, s_list, y_list, rho_list, H0):
        """计算L-BFGS搜索方向"""
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
        """Wolfe条件线搜索"""
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
        """计算损失函数（使用nn.BCEWithLogitsLoss）"""
        z = torch.matmul(X, weights)
        
        # 使用BCEWithLogitsLoss，它内部会应用sigmoid并计算BCE损失
        # 数值上更稳定，不需要手动处理sigmoid的数值范围
        bce_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        bce_loss = bce_loss_fn(z.squeeze(), y.squeeze())
        
        # 计算L2正则化项（与sklearn一致）
        if self.fit_intercept:
            l2_reg = 0.5 / self.C * torch.sum(weights[:-1] ** 2)  # 不正则化截距项
        else:
            l2_reg = 0.5 / self.C * torch.sum(weights ** 2)
        
        return bce_loss + l2_reg
    
    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float64, device=self.device)  # 使用double精度
        
        X = X.to(self.device).double()  # 强制使用double精度
        X_with_intercept = self._add_intercept(X)
        
        if self.fit_intercept:
            weights = torch.cat([self.coef_.t(), torch.tensor([[self.intercept_]], device=self.device, dtype=torch.float64)], dim=1)
        else:
            weights = self.coef_.t()
        
        z = torch.matmul(X_with_intercept, weights)
        prob_1 = torch.sigmoid(torch.clamp(z, -500, 500)).squeeze()  # 增加sigmoid的数值范围
        prob_0 = 1 - prob_1
        
        # 确保与sklearn一致的概率顺序：[P(0), P(1)]
        return torch.stack([prob_0, prob_1], dim=1)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).float()  # 使用第二列（P(1)）来预测
    
    def score(self, X, y):
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        y = y.to(self.device)
        predictions = self.predict(X)
        accuracy = (predictions == y).float().mean()
        return accuracy.item()

    def _debug_loss_computation(self, X, y, weights):
        """调试损失计算，确保与sklearn一致"""
        n_samples = X.shape[0]
        
        # 计算预测值
        z = torch.matmul(X, weights)
        p = torch.sigmoid(torch.clamp(z, -500, 500))
        
        # 计算损失（完全按照sklearn的方式）
        bce_loss = -torch.mean(y * torch.log(p + 1e-15) + (1 - y) * torch.log(1 - p + 1e-15))
        
        # L2正则化项
        if self.fit_intercept:
            l2_reg = 0.5 / self.C * torch.sum(weights[:-1] ** 2)
        else:
            l2_reg = 0.5 / self.C * torch.sum(weights ** 2)
        
        total_loss = bce_loss + l2_reg
        
        # 输出调试信息
        logging.info(f"Debug loss computation:")
        logging.info(f"  BCE loss: {bce_loss.item():.10f}")
        logging.info(f"  L2 reg: {l2_reg.item():.10f}")
        logging.info(f"  Total loss: {total_loss.item():.10f}")
        logging.info(f"  C value: {self.C}")
        logging.info(f"  Weights norm: {torch.norm(weights).item():.10f}")
        logging.info(f"  Weights mean: {torch.mean(weights).item():.10f}")
        logging.info(f"  Weights std: {torch.std(weights).item():.10f}")
        
        return total_loss

def load_cifar10c_data():
    """加载CIFAR-10-C数据"""
    logging.info("Loading CIFAR-10-C data...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = torch.nn.Identity()
    resnet50.to(device).eval()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    try:
        labels = np.load('CIFAR-10-C/labels.npy')
        logging.info("Loaded CIFAR-10 labels successfully")
    except FileNotFoundError:
        logging.warning("CIFAR-10-C labels.npy not found. Using synthetic data.")
        return generate_synthetic_data(), device
    
    all_selected_indices = {}
    for label in range(10):
        all_selected_indices[label] = labels == label
    
    def data_preprocess_all_labels(images):
        embeddings = {}
        for label in range(10):
            selected_images = torch.stack([preprocess(image) for image in images[all_selected_indices[label]]]).to(device)
            with torch.no_grad():
                embedding = resnet50(selected_images)
                ones = torch.ones(embedding.size()[0], 1, device=device)
                embedding = torch.cat([embedding, ones], dim=1)
                perm = torch.randperm(len(embedding), device=device)
                embedding = embedding[perm]
            embeddings[label] = embedding
        return embeddings
    
    logging.info("Processing brightness data...")
    try:
        images = np.load('CIFAR-10-C/brightness.npy')
        all_images_a_embeddings = data_preprocess_all_labels(images)
    except FileNotFoundError:
        logging.warning("brightness.npy not found. Using synthetic data.")
        return generate_synthetic_data(), device
    
    logging.info("Processing contrast data...")
    try:
        images = np.load('CIFAR-10-C/contrast.npy')
        all_images_b_embeddings = data_preprocess_all_labels(images)
    except FileNotFoundError:
        logging.warning("contrast.npy not found. Using synthetic data.")
        return generate_synthetic_data(), device
    
    logging.info("Normalizing embeddings...")
    all_datasets = []
    for label in range(10):
        all_datasets.extend([
            all_images_a_embeddings[label],
            all_images_b_embeddings[label]
        ])
    
    combined_data = torch.cat(all_datasets, dim=0)
    mean = combined_data.mean(dim=0, keepdim=True)
    std = combined_data.std(dim=0, keepdim=True)
    
    for label in range(10):
        all_images_a_embeddings[label] = (all_images_a_embeddings[label] - mean) / (std + 1e-6)
        all_images_b_embeddings[label] = (all_images_b_embeddings[label] - mean) / (std + 1e-6)
    
    logging.info("All embeddings processed and normalized.")
    
    # 对所有embeddings进行标准化
    logging.info("Applying standardization to all embeddings...")
    
    # 收集所有embeddings用于计算全局统计量
    all_embeddings_list = []
    for label in range(10):
        all_embeddings_list.extend([
            all_images_a_embeddings[label],
            all_images_b_embeddings[label]
        ])
    
    # 合并所有数据计算全局均值和标准差
    combined_embeddings = torch.cat(all_embeddings_list, dim=0)
    global_mean = combined_embeddings.mean(dim=0, keepdim=True)
    global_std = combined_embeddings.std(dim=0, keepdim=True)
    
    # 对每个label的embeddings进行标准化
    for label in range(10):
        all_images_a_embeddings[label] = (all_images_a_embeddings[label] - global_mean) / (global_std + 1e-8)
        all_images_b_embeddings[label] = (all_images_b_embeddings[label] - global_mean) / (global_std + 1e-8)
    
    logging.info("Standardization completed:")
    logging.info(f"  Global mean: {global_mean.mean().item():.6f}")
    logging.info(f"  Global std: {global_std.mean().item():.6f}")
    logging.info(f"  Standardized embeddings shape: {all_images_a_embeddings[0].shape}")
    
    return all_images_a_embeddings, all_images_b_embeddings, device

def generate_synthetic_data(n_samples=2000, n_features=2049, random_state=42):
    """生成合成数据"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def generate_train_data_cifar10(train_size_a_0, train_size_a_1, train_size_b_0, train_size_b_1, 
                               label_0, label_1, all_images_a_embeddings, all_images_b_embeddings):
    """生成训练数据"""
    perm_indices_a_0 = torch.randperm(4000, device=all_images_a_embeddings[label_0].device)[:train_size_a_0]
    perm_indices_a_1 = torch.randperm(4000, device=all_images_a_embeddings[label_1].device)[:train_size_a_1]
    perm_indices_b_0 = torch.randperm(4000, device=all_images_b_embeddings[label_0].device)[:train_size_b_0]
    perm_indices_b_1 = torch.randperm(4000, device=all_images_b_embeddings[label_1].device)[:train_size_b_1]
    
    train_X_a_0 = all_images_a_embeddings[label_0][:4000][perm_indices_a_0]
    train_X_a_1 = all_images_a_embeddings[label_1][:4000][perm_indices_a_1]
    train_X_b_0 = all_images_b_embeddings[label_0][:4000][perm_indices_b_0]
    train_X_b_1 = all_images_b_embeddings[label_1][:4000][perm_indices_b_1]
    
    train_y_a_0 = torch.zeros(train_size_a_0, device=train_X_a_0.device)
    train_y_a_1 = torch.ones(train_size_a_1, device=train_X_a_1.device)
    train_y_b_0 = torch.zeros(train_size_b_0, device=train_X_b_0.device)
    train_y_b_1 = torch.ones(train_size_b_1, device=train_X_b_1.device)
    
    train_X = torch.cat([train_X_a_0, train_X_b_0, train_X_a_1, train_X_b_1], dim=0)
    train_y = torch.cat([train_y_a_0, train_y_b_0, train_y_a_1, train_y_b_1], dim=0)
    
    return train_X, train_y

def compare_models_cifar10(X_train, y_train, X_test, y_test, C_values=[0.1, 1.0, 10.0, 100.0], monitor_coef_diff=False):
    """比较三个模型的性能：sklearn L-BFGS, Torch L-BFGS, Torch Newton"""
    
    results = {
        'C': [],
        'sklearn_time': [],
        'torch_lbfgs_time': [],
        'torch_newton_time': [],
        'sklearn_iterations': [],
        'torch_lbfgs_iterations': [],
        'torch_newton_iterations': [],
        'sklearn_train_loss': [],
        'torch_lbfgs_train_loss': [],
        'torch_newton_train_loss': [],
        'sklearn_test_loss': [],
        'torch_lbfgs_test_loss': [],
        'torch_newton_test_loss': [],
        'sklearn_train_acc': [],
        'torch_lbfgs_train_acc': [],
        'torch_newton_train_acc': [],
        'sklearn_test_acc': [],
        'torch_lbfgs_test_acc': [],
        'torch_newton_test_acc': [],
        'coef_diff_history': []  # 添加系数差异历史
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    for C in C_values:
        logging.info(f"\n=== Testing with C={C} ===")
        logging.info(f"All models will start from independent zero initialization")
        
        # 先训练sklearn模型（如果启用监控）
        sklearn_model_temp = None
        sklearn_coef_history = []
        if monitor_coef_diff:
            logging.info("Training sklearn model with monitoring...")
            
            # 创建一个自定义的sklearn模型来监控训练过程
            class MonitoredLogisticRegression(LogisticRegression):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.coef_history = []
                    self.convergence_info = {}
                
                def fit(self, X, y):
                    # 记录初始状态
                    if hasattr(self, 'coef_'):
                        self.coef_history.append((0, self.coef_.copy()))
                    
                    # 调用父类的fit方法
                    result = super().fit(X, y)
                    
                    # 记录最终状态
                    self.coef_history.append((self.n_iter_[0], self.coef_.copy()))
                    
                    # 分析收敛情况
                    self._analyze_convergence()
                    
                    return result
                
                def _analyze_convergence(self):
                    """分析收敛情况"""
                    if len(self.coef_history) >= 2:
                        initial_coef = self.coef_history[0][1]
                        final_coef = self.coef_history[-1][1]
                        
                        # 计算系数变化
                        coef_change = np.mean(np.abs(final_coef - initial_coef))
                        coef_change_norm = np.linalg.norm(final_coef - initial_coef)
                        
                        # 收敛状态判断
                        if coef_change < 1e-8:
                            convergence_status = "CONVERGED"
                        elif coef_change < 1e-6:
                            convergence_status = "NEARLY_CONVERGED"
                        elif coef_change < 1e-4:
                            convergence_status = "STILL_LEARNING"
                        else:
                            convergence_status = "ACTIVE_LEARNING"
                        
                        self.convergence_info = {
                            'total_iterations': self.n_iter_[0],
                            'coefficient_change_l1': coef_change,
                            'coefficient_change_l2': coef_change_norm,
                            'convergence_status': convergence_status,
                            'final_coefficient_stats': {
                                'mean': np.mean(final_coef),
                                'std': np.std(final_coef),
                                'min': np.min(final_coef),
                                'max': np.max(final_coef)
                            }
                        }
            
            sklearn_model_temp = MonitoredLogisticRegression(
                C=C, 
                fit_intercept=False,
                max_iter=100000,  # 大幅增加最大迭代次数
                tol=1e-20,        # 极高精度收敛容差
                random_state=42,
                warm_start=False,  # 确保不使用warm start
            )
            
            if torch.is_tensor(X_train):
                X_train_np = X_train.cpu().numpy()
                y_train_np = y_train.cpu().numpy()
            else:
                X_train_np = X_train
                y_train_np = y_train
                
            sklearn_model_temp.fit(X_train_np, y_train_np)
            sklearn_coef_history = sklearn_model_temp.coef_history
            logging.info(f"sklearn model trained, final iterations: {sklearn_model_temp.n_iter_[0]}")
            logging.info(f"sklearn coefficient history points: {len(sklearn_coef_history)}")
            
            # 记录sklearn模型的收敛信息
            if hasattr(sklearn_model_temp, 'convergence_info') and sklearn_model_temp.convergence_info:
                conv_info = sklearn_model_temp.convergence_info
                logging.info(f"sklearn convergence analysis:")
                logging.info(f"  Total iterations: {conv_info['total_iterations']}")
                logging.info(f"  Coefficient change (L1): {conv_info['coefficient_change_l1']:.8f}")
                logging.info(f"  Coefficient change (L2): {conv_info['coefficient_change_l2']:.8f}")
                logging.info(f"  Convergence status: {conv_info['convergence_status']}")
                logging.info(f"  Final coefficient stats:")
                stats = conv_info['final_coefficient_stats']
                logging.info(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                logging.info(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
            
            logging.info(f"All models will start from independent zero initialization")
        
        # 准备测试数据（移到最前面）
        if torch.is_tensor(X_test):
            X_test_np = X_test.cpu().numpy()
            y_test_np = y_test.cpu().numpy()
        else:
            X_test_np = X_test
            y_test_np = y_test
        
        # 准备训练数据
        if torch.is_tensor(X_train):
            X_train_np = X_train.cpu().numpy()
            y_train_np = y_train.cpu().numpy()
        else:
            X_train_np = X_train
            y_train_np = y_train
        
        # 测试sklearn LogisticRegression（移到前面）
        logging.info(f"\n--- Training sklearn LogisticRegression ---")
        start_time = time.time()
        sklearn_model = LogisticRegression(
            C=C, 
            fit_intercept=False,
            max_iter=1000,  # 大幅增加最大迭代次数
            tol=1e-12,        # 极高精度收敛容差
            random_state=42,
            warm_start=False,  # 确保不使用warm start
            solver='sag',
        )
        
        sklearn_model.fit(X_train_np, y_train_np)
        sklearn_time = time.time() - start_time
        
        # 计算sklearn模型的梯度
        sklearn_gradient = compute_sklearn_gradient_analytical(
            X_train_np, y_train_np, sklearn_model.coef_, C, fit_intercept=False
        )
        sklearn_grad_norm = np.linalg.norm(sklearn_gradient)
        
        # 计算sklearn性能指标
        sklearn_train_probs = sklearn_model.predict_proba(X_train_np)
        sklearn_train_acc = sklearn_model.score(X_train_np, y_train_np)
        sklearn_test_probs = sklearn_model.predict_proba(X_test_np)
        sklearn_test_acc = sklearn_model.score(X_test_np, y_test_np)
        
        # sklearn的损失计算（使用手写的BCE函数）
        sklearn_train_loss = compute_binary_cross_entropy_loss(y_train_np, sklearn_train_probs)
        sklearn_test_loss = compute_binary_cross_entropy_loss(y_test_np, sklearn_test_probs)
        # 增加正则化项
        # sklearn: L2正则化项 = 0.5 / C * sum(coef^2)（不包含intercept）
        # sklearn_l2_reg = 0.5 / C * np.sum(sklearn_model.coef_ ** 2)
        # sklearn_train_loss += sklearn_l2_reg
        # sklearn_test_loss += sklearn_l2_reg
        logging.info(f"sklearn LogisticRegression:")
        logging.info(f"  Training time: {sklearn_time:.4f}s")
        logging.info(f"  Total iterations: {sklearn_model.n_iter_[0] if hasattr(sklearn_model, 'n_iter_') else 'Unknown'}")
        logging.info(f"  Final gradient norm: {sklearn_grad_norm:.10f}")
        logging.info(f"  Train loss: {sklearn_train_loss:.6f}")
        logging.info(f"  Train accuracy: {sklearn_train_acc:.6f}")
        logging.info(f"  Test loss: {sklearn_test_loss:.6f}")
        logging.info(f"  Test accuracy: {sklearn_test_acc:.6f}")
        logging.info(f"  Coefficients shape: {sklearn_model.coef_.shape}")
        logging.info(f"  Intercept: {sklearn_model.intercept_[0]:.6f}")
        
        # 测试两种Torch方法
        methods = [
            ('Torch L-BFGS', 'lbfgs'),
            ('Torch Newton', 'newton')
        ]
        
        torch_results = {}
        
        for method_name, optimizer_type in methods:
            logging.info(f"\n--- Training {method_name} ---")
            start_time = time.time()
            
            torch_model = TorchLogisticRegression(
                fit_intercept=False, 
                C=C, 
                device=device, 
                optimizer=optimizer_type,
                lr=1.0
            )
            
            # 简化：不再监控系数差异
            torch_model.fit(X_train, y_train, monitor_coef_diff=False, sklearn_model=None)
            torch_time = time.time() - start_time
            
            # 计算性能指标
            torch_train_probs = torch_model.predict_proba(X_train)
            torch_train_acc = torch_model.score(X_train, y_train)
            torch_test_probs = torch_model.predict_proba(X_test)
            torch_test_acc = torch_model.score(X_test, y_test)
            
            # PyTorch的损失计算（也不包含L2正则化）
            torch_train_loss = compute_binary_cross_entropy_loss(y_train_np, torch_train_probs.cpu().numpy())  # 只有BCE
            torch_test_loss = compute_binary_cross_entropy_loss(y_test_np, torch_test_probs.cpu().numpy())     # 只有BCE

            # # 添加正则化项
            # if torch_model.fit_intercept:
            #     l2_reg_train = 0.5 / torch_model.C * torch.sum(torch_model.coef_[:-1] ** 2)
            #     l2_reg_test = 0.5 / torch_model.C * torch.sum(torch_model.coef_[:-1] ** 2)
            # else:
            #     l2_reg_train = 0.5 / torch_model.C * torch.sum(torch_model.coef_ ** 2)
            #     l2_reg_test = 0.5 / torch_model.C * torch.sum(torch_model.coef_ ** 2)

            # torch_train_loss += l2_reg_train.item()
            # torch_test_loss += l2_reg_test.item()
            
            # 保存结果
            key_suffix = optimizer_type
            torch_results[f'{key_suffix}_time'] = torch_time
            torch_results[f'{key_suffix}_iterations'] = 'Unknown'  # 简化：不再记录迭代次数
            torch_results[f'{key_suffix}_train_loss'] = torch_train_loss
            torch_results[f'{key_suffix}_test_loss'] = torch_test_loss
            torch_results[f'{key_suffix}_train_acc'] = torch_train_acc
            torch_results[f'{key_suffix}_test_acc'] = torch_test_acc
            torch_results[f'{key_suffix}_model'] = torch_model  # 保存模型对象
            
            logging.info(f"{method_name}:")
            logging.info(f"  Training time: {torch_time:.4f}s")
            logging.info(f"  Train loss: {torch_train_loss:.6f}")
            logging.info(f"  Train accuracy: {torch_train_acc:.6f}")
            logging.info(f"  Test loss: {torch_test_loss:.6f}")
            logging.info(f"  Test accuracy: {torch_test_acc:.6f}")
            logging.info(f"  Coefficients shape: {torch_model.coef_.shape}")
            logging.info(f"  Intercept: {torch_model.intercept_:.6f}")
        
        # 记录所有结果
        results['C'].append(C)
        results['sklearn_time'].append(sklearn_time)
        results['sklearn_iterations'].append(sklearn_model.n_iter_[0] if hasattr(sklearn_model, 'n_iter_') else 'Unknown')
        results['sklearn_train_loss'].append(sklearn_train_loss)
        results['sklearn_test_loss'].append(sklearn_test_loss)
        results['sklearn_train_acc'].append(sklearn_train_acc)
        results['sklearn_test_acc'].append(sklearn_test_acc)
        
        # 记录Torch结果
        for key_suffix in ['lbfgs', 'newton']:
            results[f'torch_{key_suffix}_time'].append(torch_results[f'{key_suffix}_time'])
            results[f'torch_{key_suffix}_iterations'].append(torch_results[f'{key_suffix}_iterations'])
            results[f'torch_{key_suffix}_train_loss'].append(torch_results[f'{key_suffix}_train_loss'])
            results[f'torch_{key_suffix}_test_loss'].append(torch_results[f'{key_suffix}_test_loss'])
            results[f'torch_{key_suffix}_train_acc'].append(torch_results[f'{key_suffix}_train_acc'])
            results[f'torch_{key_suffix}_test_acc'].append(torch_results[f'{key_suffix}_test_acc'])
        
        # 简化：不再保存系数差异历史
        results['coef_diff_history'].append([])
        
        # 打印系数对比（前5个系数）
        logging.info(f"\nCoefficient comparison (first 5):")
        sklearn_coef = sklearn_model.coef_.flatten()
        for method_name, optimizer_type in methods:
            torch_coef = torch_model.coef_.cpu().numpy().flatten()
            coef_diff = np.max(np.abs(torch_coef - sklearn_coef))
            logging.info(f"  {method_name} vs sklearn: {coef_diff:.6f}")
        
        # 打印系数统计信息
        logging.info(f"\nCoefficient statistics:")
        for method_name, optimizer_type in methods:
            logging.info(f"  {method_name}:")
            # 重新创建并训练模型来获取系数
            temp_model = TorchLogisticRegression(
                C=C, 
                fit_intercept=False,
                max_iter=200,
                optimizer=optimizer_type,
                device=device
            )
            temp_model.fit(X_train, y_train)
            torch_coef = temp_model.coef_.cpu().numpy().flatten()
            logging.info(f"    Mean: {np.mean(torch_coef):.6f}, Std: {np.std(torch_coef):.6f}")
            logging.info(f"    Min: {np.min(torch_coef):.6f}, Max: {np.max(torch_coef):.6f}")
        
        sklearn_coef = sklearn_model.coef_.flatten()
        logging.info(f"  sklearn:")
        logging.info(f"    Mean: {np.mean(sklearn_coef):.6f}, Std: {np.std(sklearn_coef):.6f}")
        logging.info(f"    Min: {np.min(sklearn_coef):.6f}, Max: {np.max(sklearn_coef):.6f}")
        
        # 计算并比较参数距离
        logging.info(f"\n=== Parameter Distance Analysis ===")
        sklearn_coef = sklearn_model.coef_.flatten()
        
        # 存储参数距离结果
        param_distance_results = {
            'C': C,
            'sklearn_coef_norm': np.linalg.norm(sklearn_coef),
            'torch_lbfgs_coef_norm': 0,
            'torch_newton_coef_norm': 0,
            'torch_lbfgs_l2_distance': 0,
            'torch_newton_l2_distance': 0,
            'torch_lbfgs_l1_distance': 0,
            'torch_newton_l1_distance': 0,
            'torch_lbfgs_max_distance': 0,
            'torch_newton_max_distance': 0,
            'torch_lbfgs_cosine_similarity': 0,
            'torch_newton_cosine_similarity': 0
        }
        
        for method_name, optimizer_type in methods:
            # 获取torch模型的系数
            if optimizer_type == 'lbfgs':
                torch_coef = torch_results['lbfgs_model'].coef_.cpu().numpy().flatten()
            else:  # newton
                torch_coef = torch_results['newton_model'].coef_.cpu().numpy().flatten()
            
            # 计算各种距离度量
            l2_distance = np.linalg.norm(torch_coef - sklearn_coef)
            l1_distance = np.sum(np.abs(torch_coef - sklearn_coef))
            max_distance = np.max(np.abs(torch_coef - sklearn_coef))
            mean_distance = np.mean(np.abs(torch_coef - sklearn_coef))
            
            # 计算余弦相似度
            cosine_similarity = np.dot(torch_coef, sklearn_coef) / (np.linalg.norm(torch_coef) * np.linalg.norm(sklearn_coef))
            
            # 计算相对误差
            relative_l2_error = l2_distance / np.linalg.norm(sklearn_coef)
            relative_l1_error = l1_distance / np.sum(np.abs(sklearn_coef))
            
            # 存储结果
            if optimizer_type == 'lbfgs':
                param_distance_results['torch_lbfgs_coef_norm'] = np.linalg.norm(torch_coef)
                param_distance_results['torch_lbfgs_l2_distance'] = l2_distance
                param_distance_results['torch_lbfgs_l1_distance'] = l1_distance
                param_distance_results['torch_lbfgs_max_distance'] = max_distance
                param_distance_results['torch_lbfgs_cosine_similarity'] = cosine_similarity
            else:  # newton
                param_distance_results['torch_newton_coef_norm'] = np.linalg.norm(torch_coef)
                param_distance_results['torch_newton_l2_distance'] = l2_distance
                param_distance_results['torch_newton_l1_distance'] = l1_distance
                param_distance_results['torch_newton_max_distance'] = max_distance
                param_distance_results['torch_newton_cosine_similarity'] = cosine_similarity
            
            # 输出详细的参数距离信息
            logging.info(f"\n{method_name} vs sklearn parameter comparison:")
            logging.info(f"  Coefficient norms:")
            logging.info(f"    sklearn: {np.linalg.norm(sklearn_coef):.10f}")
            logging.info(f"    {method_name}: {np.linalg.norm(torch_coef):.10f}")
            logging.info(f"  Distance metrics:")
            logging.info(f"    L2 distance: {l2_distance:.10f}")
            logging.info(f"    L1 distance: {l1_distance:.10f}")
            logging.info(f"    Max distance: {max_distance:.10f}")
            logging.info(f"    Mean distance: {mean_distance:.10f}")
            logging.info(f"  Relative errors:")
            logging.info(f"    Relative L2 error: {relative_l2_error:.10f}")
            logging.info(f"    Relative L1 error: {relative_l1_error:.10f}")
            logging.info(f"  Similarity:")
            logging.info(f"    Cosine similarity: {cosine_similarity:.10f}")
            
            # 输出前10个系数的详细对比
            logging.info(f"  First 10 coefficients comparison:")
            for i in range(min(10, len(sklearn_coef))):
                diff = torch_coef[i] - sklearn_coef[i]
                rel_diff = diff / sklearn_coef[i] if sklearn_coef[i] != 0 else float('inf')
                logging.info(f"    Coef[{i}]: sklearn={sklearn_coef[i]:.8f}, {method_name}={torch_coef[i]:.8f}, diff={diff:.8f}, rel_diff={rel_diff:.8f}")
        
        # 将参数距离结果添加到results中
        if 'param_distance_results' not in results:
            results['param_distance_results'] = []
        results['param_distance_results'].append(param_distance_results)
    
    return results

def compute_binary_cross_entropy_loss(y_true, y_pred_proba):
    """
    手写二元交叉熵损失函数
    
    Args:
        y_true: 真实标签 (0或1)
        y_pred_proba: 预测概率 (形状为(n_samples, 2)或(n_samples,))
    
    Returns:
        loss: 平均二元交叉熵损失
    """
    import numpy as np
    
    # 确保y_true是numpy数组
    y_true = np.array(y_true)
    
    # 处理不同的y_pred_proba格式
    if y_pred_proba.ndim == 2:
        # 如果输入是(n_samples, 2)格式，取第二列作为正类概率
        y_pred = y_pred_proba[:, 1]
    else:
        # 如果输入是(n_samples,)格式，直接使用
        y_pred = y_pred_proba
    
    # 数值稳定性：限制概率范围
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # 计算二元交叉熵损失
    # BCE = -[y*log(p) + (1-y)*log(1-p)]
    bce_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # 返回平均损失
    return np.mean(bce_loss)

def compute_sklearn_gradient_analytical(X, y, coef, C, fit_intercept=False):
    """
    使用解析方法计算sklearn LogisticRegression的梯度
    
    Args:
        X: 特征矩阵
        y: 标签向量
        coef: 权重系数
        C: 正则化参数
        fit_intercept: 是否包含截距项
    
    Returns:
        gradient: 梯度向量
    """
    import numpy as np
    
    # 计算预测概率
    z = np.dot(X, coef.flatten())
    p = 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # sigmoid with clipping
    
    # 计算梯度
    n_samples = X.shape[0]
    diff = (p - y) / n_samples
    gradient = np.dot(X.T, diff)
    
    # 添加L2正则化梯度
    if fit_intercept:
        gradient[:-1] += coef.flatten()[:-1] / C  # 不正则化截距项
    else:
        gradient += coef.flatten() / C
    
    return gradient

def main():
    """主函数"""
    log_filename = setup_logging()
    
    logging.info("=" * 80)
    logging.info("TorchLogisticRegression vs sklearn LogisticRegression Comparison")
    logging.info("Using CIFAR-10-C data with same preprocessing as PMI_bias_cifar_copy.py")
    logging.info("=" * 80)
    
    try:
        all_images_a_embeddings, all_images_b_embeddings, device = load_cifar10c_data()
        logging.info("Successfully loaded CIFAR-10-C data!")
        
        train_size_a_0, train_size_a_1 = 100, 100
        train_size_b_0, train_size_b_1 = 100, 100
        test_size_a_0, test_size_a_1 = 200, 200
        test_size_b_0, test_size_b_1 = 200, 200
        
        np.random.seed(42)
        selected_labels = np.random.choice(10, 2, replace=False)
        label_0, label_1 = selected_labels[0], selected_labels[1]
        logging.info(f"Using labels: {label_0} and {label_1}")
        
        logging.info("Generating training data...")
        X_train, y_train = generate_train_data_cifar10(
            train_size_a_0, train_size_a_1, train_size_b_0, train_size_b_1,
            label_0, label_1, all_images_a_embeddings, all_images_b_embeddings
        )
        
        logging.info("Generating test data...")
        X_test = torch.cat([
            all_images_a_embeddings[label_0][4000:4000+test_size_a_0],
            all_images_b_embeddings[label_0][4000:4000+test_size_b_0],
            all_images_a_embeddings[label_1][4000:4000+test_size_a_1],
            all_images_b_embeddings[label_1][4000:4000+test_size_a_1],
        ])
        y_test = torch.cat([
            torch.zeros(test_size_a_0, device=device),
            torch.zeros(test_size_b_0, device=device),
            torch.ones(test_size_a_1, device=device),
            torch.ones(test_size_b_1, device=device)
        ])
        
        logging.info(f"Training set: {X_train.shape}")
        logging.info(f"Test set: {X_test.shape}")
        logging.info(f"Training labels distribution: {torch.bincount(y_train.long())}")
        logging.info(f"Test labels distribution: {torch.bincount(y_test.long())}")
        
        # 对X_train和X_test进行PCA降维
        logging.info("Applying PCA dimensionality reduction...")
        from sklearn.decomposition import PCA
        
        # 转换为numpy进行PCA处理
        if torch.is_tensor(X_train):
            X_train_np = X_train.cpu().numpy()
            X_test_np = X_test.cpu().numpy()
        else:
            X_train_np = X_train
            X_test_np = X_test
        
        # 设置PCA参数
        n_components = min(200, X_train_np.shape[1])  # 降维到500维或保持原维度
        pca = PCA(n_components=n_components, random_state=42)
        
        # 在训练集上拟合PCA
        X_train_pca = pca.fit_transform(X_train_np)
        X_test_pca = pca.transform(X_test_np)
        
        # 转换回torch tensor
        X_train = torch.tensor(X_train_pca, dtype=torch.float64, device=device)
        X_test = torch.tensor(X_test_pca, dtype=torch.float64, device=device)
        
        # # 对X_train和X_test进行标准化
        # logging.info("Applying standardization...")
        # from sklearn.preprocessing import StandardScaler
        
        # # 转换为numpy进行标准化处理
        # X_train_np = X_train.cpu().numpy()
        # X_test_np = X_test.cpu().numpy()
        
        # # 在训练集上拟合标准化器
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train_np)
        # X_test_scaled = scaler.transform(X_test_np)
        
        # # 转换回torch tensor
        # X_train = torch.tensor(X_train_scaled, dtype=torch.float64, device=device)
        # X_test = torch.tensor(X_test_scaled, dtype=torch.float64, device=device)
        
        logging.info(f"Standardization completed:")
        logging.info(f"  Training set mean: {X_train.mean().item():.6f}")
        logging.info(f"  Training set std: {X_train.std().item():.6f}")
        logging.info(f"  Test set mean: {X_test.mean().item():.6f}")
        logging.info(f"  Test set std: {X_test.std().item():.6f}")
        logging.info(f"  Training set after standardization: {X_train.shape}")
        logging.info(f"  Test set after standardization: {X_test.shape}")
        
    except Exception as e:
        logging.error(f"Error loading CIFAR-10-C data: {e}")
        logging.info("Falling back to synthetic data...")
        
        X, y = generate_synthetic_data(n_samples=2000, n_features=2049, random_state=42)
        split_idx = int(0.8 * len(y))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logging.info(f"Using synthetic data: {X.shape[0]} samples, {X.shape[1]} features")
    
    logging.info("")
    
    logging.info("Comparing models with different regularization parameters...")
    results = compare_models_cifar10(X_train, y_train, X_test, y_test, C_values=[0.1, 1.0, 10.0, 100.0], monitor_coef_diff=True)
    
    logging.info("\n" + "="*80)
    logging.info("SUMMARY")
    logging.info("="*80)
    
    # 计算各种方法的平均性能
    methods = ['sklearn', 'torch_lbfgs', 'torch_newton']
    
    for method in methods:
        if method == 'sklearn':
            method_name = "sklearn L-BFGS"
        elif method == 'torch_lbfgs':
            method_name = "Torch L-BFGS"
        else:
            method_name = "Torch Newton"
        
        # 训练时间
        if f'{method}_time' in results and results[f'{method}_time']:
            avg_time = np.mean([t for t in results[f'{method}_time'] if isinstance(t, (int, float))])
            logging.info(f"{method_name} average training time: {avg_time:.4f}s")
        
        # 迭代次数
        if f'{method}_iterations' in results and results[f'{method}_iterations']:
            valid_iter = [i for i in results[f'{method}_iterations'] if i != 'Unknown']
            if valid_iter:
                avg_iter = np.mean(valid_iter)
                logging.info(f"{method_name} average iterations: {avg_iter:.1f}")
        
        # 训练损失
        if f'{method}_train_loss' in results and results[f'{method}_train_loss']:
            avg_train_loss = np.mean(results[f'{method}_train_loss'])
            logging.info(f"{method_name} average train loss: {avg_train_loss:.6f}")
        
        # 测试损失
        if f'{method}_test_loss' in results and results[f'{method}_test_loss']:
            avg_test_loss = np.mean(results[f'{method}_test_loss'])
            logging.info(f"{method_name} average test loss: {avg_test_loss:.6f}")
        
        # 训练准确率
        if f'{method}_train_acc' in results and results[f'{method}_train_acc']:
            avg_train_acc = np.mean(results[f'{method}_train_acc'])
            logging.info(f"{method_name} average train accuracy: {avg_train_acc:.6f}")
        
        # 测试准确率
        if f'{method}_test_acc' in results and results[f'{method}_test_acc']:
            avg_test_acc = np.mean(results[f'{method}_test_acc'])
            logging.info(f"{method_name} average test accuracy: {avg_test_acc:.6f}")
        
        logging.info("")  # 空行分隔
    
    # 计算与sklearn的差异
    logging.info("Performance comparison vs sklearn:")
    for method in ['torch_lbfgs', 'torch_newton']:
        if method == 'torch_lbfgs':
            method_name = "Torch L-BFGS"
        else:
            method_name = "Torch Newton"
        
        # 时间比
        if f'{method}_time' in results and 'sklearn_time' in results:
            time_ratios = []
            for i in range(len(results['sklearn_time'])):
                if (isinstance(results[f'{method}_time'][i], (int, float)) and 
                    isinstance(results['sklearn_time'][i], (int, float)) and 
                    results['sklearn_time'][i] > 0):
                    time_ratios.append(results[f'{method}_time'][i] / results['sklearn_time'][i])
            
            if time_ratios:
                avg_time_ratio = np.mean(time_ratios)
                logging.info(f"{method_name} vs sklearn time ratio: {avg_time_ratio:.2f}x")
        
        # 准确率差异
        if f'{method}_test_acc' in results and 'sklearn_test_acc' in results:
            acc_diffs = [t - s for t, s in zip(results[f'{method}_test_acc'], results['sklearn_test_acc'])]
            avg_acc_diff = np.mean(acc_diffs)
            logging.info(f"{method_name} vs sklearn accuracy difference: {avg_acc_diff:.6f}")
        
        # 损失差异
        if f'{method}_test_loss' in results and 'sklearn_test_loss' in results:
            loss_diffs = [t - s for t, s in zip(results[f'{method}_test_loss'], results['sklearn_test_loss'])]
            avg_loss_diff = np.mean(loss_diffs)
            logging.info(f"{method_name} vs sklearn loss difference: {avg_loss_diff:.6f}")
        
        logging.info("")  # 空行分隔
    
    # 计算参数距离汇总
    logging.info("\n" + "="*80)
    logging.info("PARAMETER DISTANCE SUMMARY")
    logging.info("="*80)
    
    if 'param_distance_results' in results and results['param_distance_results']:
        for i, param_result in enumerate(results['param_distance_results']):
            C = param_result['C']
            logging.info(f"\nC = {C}:")
            logging.info(f"  sklearn coefficient norm: {param_result['sklearn_coef_norm']:.10f}")
            
            for method in ['lbfgs', 'newton']:
                method_name = f"Torch {method.upper()}"
                logging.info(f"  {method_name}:")
                logging.info(f"    Coefficient norm: {param_result[f'torch_{method}_coef_norm']:.10f}")
                logging.info(f"    L2 distance: {param_result[f'torch_{method}_l2_distance']:.10f}")
                logging.info(f"    L1 distance: {param_result[f'torch_{method}_l1_distance']:.10f}")
                logging.info(f"    Max distance: {param_result[f'torch_{method}_max_distance']:.10f}")
                logging.info(f"    Cosine similarity: {param_result[f'torch_{method}_cosine_similarity']:.10f}")

if __name__ == "__main__":
    main() 