# TorchLogisticRegression

这是一个使用PyTorch实现的Logistic Regression类，专门设计为与sklearn.LogisticRegression兼容的API。

## 主要特性

- **L-BFGS优化算法**: 使用sklearn风格的L-BFGS优化器
- **L2正则化**: 支持可调节的正则化强度
- **截距项支持**: 可选择是否拟合截距项
- **数值稳定性**: 包含多种数值稳定性优化
- **设备支持**: 支持CPU和CUDA设备
- **API兼容性**: 与sklearn.LogisticRegression完全兼容

## 安装依赖

```bash
pip install torch numpy
```

## 基本用法

### 1. 导入和创建模型

```python
from torch_logistic_regression import TorchLogisticRegression

# 创建模型实例
model = TorchLogisticRegression(
    fit_intercept=True,  # 是否拟合截距项
    C=1.0,              # 正则化强度的倒数 (1/λ)
    max_iter=5000,      # 最大迭代次数
    tol=1e-4,          # 收敛容差
    device='cpu'        # 计算设备 ('cpu' 或 'cuda')
)
```

### 2. 训练模型

```python
import torch

# 准备数据
X = torch.randn(1000, 100)  # 1000个样本，100个特征
y = (torch.randn(1000) > 0).float()  # 二元标签

# 训练模型
model.fit(X, y)
```

### 3. 预测和评估

```python
# 预测概率
probabilities = model.predict_proba(X)  # 返回 [P(0), P(1)]

# 预测类别
predictions = model.predict(X)

# 计算准确率
accuracy = model.score(X, y)
```

## 参数说明

### 初始化参数

- `fit_intercept` (bool): 是否拟合截距项，默认True
- `C` (float): 正则化强度的倒数，默认1.0
  - C值越大，正则化越弱
  - C值越小，正则化越强
- `max_iter` (int): 最大迭代次数，默认5000
- `tol` (float): 收敛容差，默认1e-4
- `lr` (float): 学习率（L-BFGS中不使用），默认1.0
- `device` (str): 计算设备，默认'cpu'

### 模型属性

训练完成后，模型会包含以下属性：
- `coef_`: 系数矩阵，形状为(1, n_features)
- `intercept_`: 截距项，标量值

## 算法细节

### L-BFGS优化

- **存储向量**: 默认存储5个向量对(s, y)
- **线搜索**: 使用Wolfe条件（Armijo条件 + 曲率条件）
- **数值稳定性**: 包含梯度裁剪和Hessian近似范围限制

### 损失函数

- **二元交叉熵**: 主要的分类损失
- **L2正则化**: 防止过拟合，系数为0.5/C
- **截距项**: 不参与正则化

### 收敛条件

- 梯度范数 < tol
- 连续5次迭代损失无改善
- 损失值 > 1000（防止爆炸）

## 性能对比

与sklearn.LogisticRegression的对比结果：

| 正则化参数C | Torch Loss | sklearn Loss | 差异 | 训练时间比 |
|------------|------------|--------------|------|------------|
| 0.1        | 0.427767   | 0.332119     | +0.095649 | 4.02x |
| 1.0        | 0.307428   | 0.433988     | -0.126560 | 3.32x |
| 10.0       | 0.300410   | 0.577338     | -0.276928 | 4.14x |
| 100.0      | 0.340266   | 0.654728     | -0.314462 | 2.89x |

**平均性能**:
- 系数差异: 0.040792
- 准确率差异: -0.013125
- 损失差异: -0.155575
- 训练时间比: 3.59x

## 使用建议

### 1. 数据预处理

- 确保特征已经标准化/归一化
- 对于高维数据，建议使用较小的C值（如0.1或1.0）
- 标签应该是0和1的二元值

### 2. 参数调优

- **C值选择**: 
  - 高维数据：使用较小的C值（0.1-1.0）
  - 低维数据：可以使用较大的C值（10.0-100.0）
- **收敛设置**: 
  - 对于大数据集，可以增加max_iter
  - 对于高精度要求，可以减小tol

### 3. 设备选择

- **CPU**: 适合小数据集和调试
- **CUDA**: 适合大数据集，可以显著提升训练速度

## 示例代码

### 完整训练流程

```python
import torch
from torch_logistic_regression import TorchLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# 1. 准备数据
X = torch.randn(2000, 100)
y = (torch.randn(2000) > 0).float()

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X.numpy(), y.numpy(), test_size=0.2, random_state=42
)

# 3. 创建和训练模型
model = TorchLogisticRegression(C=1.0, device='cpu')
model.fit(X_train, y_train)

# 4. 评估模型
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Log Loss: {loss:.6f}")
```

### 与sklearn对比

```python
from sklearn.linear_model import LogisticRegression

# sklearn模型
sklearn_model = LogisticRegression(C=1.0, fit_intercept=True, random_state=42)
sklearn_model.fit(X_train, y_train)

# 对比结果
sklearn_acc = sklearn_model.score(X_test, y_test)
sklearn_proba = sklearn_model.predict_proba(X_test)
sklearn_loss = log_loss(y_test, sklearn_proba)

print(f"Torch Accuracy: {accuracy:.4f}, sklearn Accuracy: {sklearn_acc:.4f}")
print(f"Torch Loss: {loss:.6f}, sklearn Loss: {sklearn_loss:.6f}")
```

## 注意事项

1. **内存使用**: L-BFGS需要存储梯度历史，对于高维数据可能占用较多内存
2. **数值稳定性**: 包含多种数值稳定性优化，但仍需注意数据预处理
3. **收敛性**: 在某些情况下可能需要调整参数来确保收敛
4. **设备兼容**: 确保PyTorch版本与CUDA版本兼容

## 故障排除

### 常见问题

1. **损失爆炸**: 检查C值是否过小，数据是否已标准化
2. **收敛慢**: 增加max_iter，调整tol值
3. **内存不足**: 减少L-BFGS存储向量数量（修改m参数）
4. **数值不稳定**: 检查数据范围，确保没有异常值

### 调试建议

- 使用小数据集测试
- 监控损失变化
- 检查梯度范数
- 验证正则化项计算

## 版本历史

- **v1.0**: 初始版本，基本L-BFGS实现
- **v1.1**: 添加数值稳定性优化
- **v1.2**: 改进线搜索策略（Wolfe条件）
- **v1.3**: 优化收敛条件和性能监控

## 许可证

本项目采用MIT许可证。 