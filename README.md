# TorchLogisticRegression vs sklearn LogisticRegression 对比测试

这个项目包含了两个测试文件，用于对比自定义的`TorchLogisticRegression`实现与sklearn包中的`LogisticRegression`的性能差异。

## 文件说明

### 1. `quick_test.py` - 快速测试版本
- 简化版的`TorchLogisticRegression`实现
- 快速对比两个模型的性能
- 测试不同正则化参数(C)下的表现
- 适合快速验证和调试

### 2. `test_torch_vs_sklearn.py` - 完整测试版本
- 完整的`TorchLogisticRegression`实现（包含原始代码中的所有优化）
- 详细的性能对比分析
- 生成可视化图表
- 测试不同数据大小下的性能
- 适合深入分析和研究

### 3. `requirements.txt` - 依赖包列表
- 列出了运行测试所需的所有Python包

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 快速测试
```bash
python quick_test.py
```

### 完整测试
```bash
python test_torch_vs_sklearn.py
```

## 测试内容

两个测试文件都会对比以下方面：

1. **训练时间** - 两个模型的训练速度对比
2. **测试损失** - 使用log_loss计算的测试集损失
3. **测试准确率** - 在测试集上的分类准确率
4. **模型系数** - 训练得到的权重系数差异
5. **截距项** - 截距项的差异
6. **不同正则化参数** - 测试C=0.1, 1.0, 10.0, 100.0时的表现

## 输出结果

- 控制台输出详细的对比结果
- 完整测试版本会生成`torch_vs_sklearn_comparison.png`图表文件
- 包含平均性能差异和速度提升倍数的总结

## 注意事项

1. 如果系统有CUDA GPU，`TorchLogisticRegression`会自动使用GPU加速
2. 两个实现都支持L2正则化（通过C参数控制）
3. 为了公平比较，两个模型都使用相同的随机种子和收敛条件
4. 测试数据是随机生成的合成数据，确保可重现性

## 预期结果

通常情况下：
- `TorchLogisticRegression`在GPU上训练速度更快
- 两个模型的准确率和损失应该非常接近
- 系数差异应该很小（< 1e-3）
- 截距差异应该很小（< 1e-3）

如果发现显著差异，可能的原因：
1. 收敛条件设置不同
2. 正则化实现细节不同
3. 数值精度差异
4. 优化算法差异 