# 反向传播算法

## 定义

反向传播 (Backpropagation) 是一种用于训练神经网络的监督学习算法。它通过链式法则计算损失函数关于每个参数的梯度，然后使用梯度下降更新参数。

## 算法流程

### 前向传播 (Forward Propagation)

1. 输入层接收输入数据
2. 数据逐层向前传播，计算每个神经元的输出
3. 计算最终损失值

### 反向传播 (Backward Propagation)

1. 从输出层开始，计算损失函数关于输出层的梯度
2. 使用链式法则，逐层向后计算梯度
3. 计算损失函数关于每个参数的梯度

## 链式法则

反向传播的核心是链式法则。对于复合函数 $f(g(x))$：

$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}$$

对于多层神经网络，链式法则可以扩展到多个中间变量：

$$\frac{\partial J}{\partial w} = \frac{\partial J}{\partial z_L} \cdot \frac{\partial z_L}{\partial a_{L-1}} \cdot \frac{\partial a_{L-1}}{\partial z_{L-1}} \cdot \ldots \cdot \frac{\partial z}{\partial w}$$

## 梯度计算

### 输出层梯度

对于输出层 $L$，损失函数关于加权输入 $z_L$ 的梯度：

$$\delta^L = \frac{\partial J}{\partial z^L} = \frac{\partial J}{\partial a^L} \odot \sigma'(z^L)$$

其中 $\odot$ 表示逐元素相乘，$\sigma'(z)$ 是激活函数的导数。

### 隐藏层梯度

对于隐藏层 $l$，使用链式法则：

$$\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$$

### 参数梯度

权重 $w^l$ 的梯度：

$$\frac{\partial J}{\partial w^l} = \delta^l (a^{l-1})^T$$

偏置 $b^l$ 的梯度：

$$\frac{\partial J}{\partial b^l} = \sum_i \delta_i^l$$

## 激活函数的导数

### Sigmoid 函数

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

导数：

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

### ReLU 函数

$$\text{ReLU}(z) = \max(0, z)$$

导数：

$$\text{ReLU}'(z) = \begin{cases} 
1 & z > 0 \\
0 & z \leq 0 
\end{cases}$$

### Tanh 函数

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

导数：

$$\tanh'(z) = 1 - \tanh(z)^2$$

## Python 实现示例

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.01
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for w, b in zip(self.weights, self.biases):
            z = self.activations[-1] @ w + b
            self.z_values.append(z)
            a = self.sigmoid(z)
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, lr=0.01):
        m = X.shape[0]
        delta = self.activations[-1] - y
        
        # 反向传播
        for i in range(len(self.weights)-1, -1, -1):
            dW = (1/m) * (self.activations[i].T @ delta)
            dB = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.sigmoid_derivative(self.z_values[i-1])
            
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * dB
    
    def train(self, X, y, epochs=1000, lr=0.01):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y, lr)
```

## 反向传播的注意事项

### 梯度消失问题

当网络很深时，梯度可能变得非常小，导致前面的层无法有效学习。

解决方案：
1. 使用 ReLU 激活函数
2. 使用残差连接 (Residual Connections)
3. 使用批归一化 (Batch Normalization)

### 梯度爆炸问题

梯度可能变得非常大，导致参数更新过大而发散。

解决方案：
1. 使用梯度裁剪 (Gradient Clipping)
2. 使用权重初始化策略（如 Xavier/Glorot 初始化）

### 数值稳定性

sigmoid 和 tanh 函数在极端值时梯度接近零，导致数值不稳定。

解决方案：
1. 使用 ReLU 或其变体
2. 使用更稳定的损失函数

## 反向传播的复杂度

- 时间复杂度：$O(m \cdot n^2)$，其中 $m$ 是样本数，$n$ 是网络规模
- 空间复杂度：$O(n)$，用于存储激活值和梯度

## 与其他算法的关系

反向传播是深度学习训练的核心算法，与以下算法紧密相关：
- **梯度下降**：用于更新参数
- **随机梯度下降**：随机采样进行反向传播
- **动量法**：加速收敛
- **Adam**：自适应学习率优化器

理解反向传播对于掌握深度学习至关重要。
