# 梯度下降算法

## 定义

梯度下降 (Gradient Descent) 是一种用于最小化损失函数的一阶迭代优化算法。它是机器学习中最基础也是最重要的优化算法之一。

## 核心思想

梯度下降的核心思想是：沿着损失函数梯度的反方向（最陡下降方向）更新模型参数，逐步逼近损失函数的最小值点。

## 数学公式

参数更新公式：

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)$$

其中：
- $\theta_t$ 是第 t 步的参数向量
- $\eta$ 是学习率 (learning rate)
- $\nabla_\theta J(\theta_t)$ 是损失函数 $J$ 关于参数 $\theta$ 的梯度
- $\theta_{t+1}$ 是更新后的参数向量

## 梯度的计算

对于线性回归，损失函数通常使用均方误差 (MSE)：

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

其中 $h_\theta(x) = \theta^T x$ 是预测值。

对 $\theta_j$ 求偏导：

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

## 学习率的选择

学习率 $\eta$ 是梯度下降中最重要的超参数：
- **学习率过小**：收敛速度慢，需要更多迭代次数
- **学习率过大**：可能越过最小值点，导致发散
- **自适应学习率**：如 Adam、Adagrad 等优化器会自动调整学习率

## 三种梯度下降变体

### 1. 批量梯度下降 (Batch Gradient Descent)

使用全部训练数据计算梯度：

$$\theta_{t+1} = \theta_t - \eta \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta J(\theta; x^{(i)}, y^{(i)})$$

优点：收敛稳定，能找到全局最优
缺点：计算量大，内存消耗大

### 2. 随机梯度下降 (Stochastic Gradient Descent, SGD)

每次只使用一个训练样本计算梯度：

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta; x^{(i)}, y^{(i)})$$

优点：计算速度快，内存消耗小
缺点：收敛不稳定，有噪声

### 3. 小批量梯度下降 (Mini-batch Gradient Descent)

使用一小批训练样本计算梯度：

$$\theta_{t+1} = \theta_t - \eta \frac{1}{b} \sum_{i=1}^{b} \nabla_\theta J(\theta; x^{(i)}, y^{(i)})$$

其中 $b$ 是批量大小（通常取 32、64、128）。

优点：平衡了收敛速度和稳定性
缺点：需要调整批量大小超参数

## 梯度下降的收敛性

梯度下降收敛的条件：
1. 学习率足够小
2. 损失函数是凸函数（保证收敛到全局最优）
3. 梯度范数趋于零

对于非凸函数，梯度下降可能收敛到局部最优或鞍点。

## Python 实现示例

```python
import numpy as np

def gradient_descent(X, y, lr=0.01, n_iters=1000):
    """
    批量梯度下降实现
    
    参数:
        X: 特征矩阵 (m x n)
        y: 标签向量 (m x 1)
        lr: 学习率
        n_iters: 迭代次数
    
    返回:
        theta: 训练后的参数
        costs: 每次迭代的损失值
    """
    m, n = X.shape
    theta = np.zeros(n)
    costs = []
    
    for _ in range(n_iters):
        # 计算预测值
        predictions = X @ theta
        
        # 计算误差
        error = predictions - y
        
        # 计算梯度
        gradient = (1/m) * (X.T @ error)
        
        # 更新参数
        theta = theta - lr * gradient
        
        # 计算损失
        cost = (1/(2*m)) * np.sum(error**2)
        costs.append(cost)
    
    return theta, costs
```

## 实际应用中的技巧

1. **特征缩放**：对特征进行标准化或归一化，加速收敛
2. **动量法 (Momentum)**：积累梯度方向的动量，加速收敛并减少震荡
3. **学习率衰减**：随着迭代次数增加逐渐减小学习率
4. **早停 (Early Stopping)**：验证集损失不再下降时停止训练

## 相关算法

- **随机梯度下降 (SGD)**
- **动量法 (Momentum)**
- **Nesterov 加速梯度 (NAG)**
- **AdaGrad**
- **RMSprop**
- **Adam 优化器**

梯度下降是深度学习训练的基础，理解其原理对于掌握更高级的优化算法至关重要。
