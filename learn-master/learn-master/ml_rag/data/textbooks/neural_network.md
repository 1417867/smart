# 神经网络基础

## 定义

神经网络 (Neural Network) 是一种受生物神经网络启发的机器学习模型，由大量神经元通过连接组成，能够学习复杂的非线性映射。

## 神经元模型

### 人工神经元

一个神经元接收多个输入，通过加权求和后经过激活函数输出：

$$a = \sigma(z) = \sigma(w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b)$$

其中：
- $x_i$ 是输入
- $w_i$ 是权重
- $b$ 是偏置
- $z = \sum_i w_i x_i + b$ 是加权输入
- $\sigma$ 是激活函数
- $a$ 是输出

### 激活函数

激活函数引入非线性，使神经网络能够学习复杂函数：

#### 1. Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

输出范围：(0, 1)
用途：二分类问题的输出层

#### 2. Tanh

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

输出范围：(-1, 1)
用途：隐藏层

#### 3. ReLU

$$\text{ReLU}(z) = \max(0, z)$$

输出范围：[0, ∞)
用途：隐藏层（目前最常用）

#### 4. Leaky ReLU

$$\text{LeakyReLU}(z) = \max(\alpha z, z)$$

解决 ReLU 的死神经元问题

## 神经网络结构

### 前馈神经网络

信息从输入层单向流向输出层，没有循环连接。

结构组成：
1. **输入层**：接收原始数据
2. **隐藏层**：进行特征提取和变换
3. **输出层**：产生最终预测

### 深度神经网络

包含多个隐藏层的神经网络：

$$L = \text{输入层} + \text{隐藏层} + \text{输出层}$$

深度神经网络能够学习层次化的特征表示：
- 第1层：学习简单特征（如边缘）
- 第2层：学习组合特征（如形状）
- 第3层：学习高级特征（如物体）

## 前向传播

前向传播是计算神经网络输出的过程：

对于第 $l$ 层：
$$z^l = a^{l-1} w^l + b^l$$
$$a^l = \sigma(z^l)$$

其中：
- $a^{l-1}$ 是第 $l-1$ 层的激活值
- $w^l$ 是第 $l$ 层的权重矩阵
- $b^l$ 是第 $l$ 层的偏置向量

## 损失函数

损失函数衡量预测值与真实值之间的差距：

### 均方误差 (MSE)

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

用途：回归问题

### 交叉熵损失 (Cross-Entropy)

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))]$$

用途：分类问题

### 多类交叉熵

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{k} y_j^{(i)} \log(h_\theta(x^{(i)}))_j$$

用途：多分类问题

## 参数更新

使用梯度下降更新参数：

$$w_{new} = w_{old} - \eta \frac{\partial J}{\partial w}$$
$$b_{new} = b_{old} - \eta \frac{\partial J}{\partial b}$$

## Python 实现示例

```python
import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def compute_loss(self, y):
        m = y.shape[0]
        log_probs = -np.log(self.a2[np.arange(m), y.argmax(axis=1)])
        return np.mean(log_probs)
```

## 训练过程

1. **前向传播**：计算预测值
2. **计算损失**：衡量预测误差
3. **反向传播**：计算梯度
4. **参数更新**：使用梯度下降

重复以上步骤直到损失收敛。

## 神经网络的能力

### 通用逼近定理

一个包含足够多隐藏层神经元的前馈神经网络可以逼近任何连续函数。

### 学习能力

神经网络能够从数据中自动学习特征表示，无需人工设计特征。

### 泛化能力

训练好的神经网络能够对未见过的数据进行预测。

## 常见架构

### 多层感知机 (MLP)
- 全连接层
- 用于分类和回归

### 卷积神经网络 (CNN)
- 卷积层 + 池化层
- 用于图像处理

### 循环神经网络 (RNN)
- 循环连接
- 用于序列数据

### Transformer
- 自注意力机制
- 用于 NLP

## 挑战与解决方案

### 过拟合
- 使用正则化（L1/L2）
- 使用 Dropout
- 使用数据增强

### 梯度消失/爆炸
- 使用 ReLU 激活函数
- 使用残差连接
- 使用批归一化

### 训练不稳定
- 使用合适的初始化
- 使用学习率调度
- 使用自适应优化器

神经网络是深度学习的基础，理解其原理对于掌握现代 AI 技术至关重要。
