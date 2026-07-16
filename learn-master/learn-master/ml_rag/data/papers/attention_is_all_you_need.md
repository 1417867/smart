# Attention Is All You Need

## 论文信息

- **标题**: Attention Is All You Need
- **作者**: Vaswani et al. (Google Brain)
- **发表时间**: 2017年
- **会议**: NeurIPS 2017
- **引用数**: 超过120,000次

## 核心贡献

本文提出了 Transformer 架构，完全基于注意力机制，摒弃了传统的循环和卷积结构。

## 背景

传统序列模型（如 RNN、LSTM）存在以下问题：
1. **计算效率低**：需要顺序处理，无法并行化
2. **长距离依赖**：难以捕捉远距离的依赖关系
3. **梯度消失**：深度网络训练困难

## 注意力机制

### 缩放点积注意力

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q$ (Query): 查询矩阵
- $K$ (Key): 键矩阵
- $V$ (Value): 值矩阵
- $d_k$: 键向量的维度
- $\sqrt{d_k}$: 缩放因子，防止内积过大

### 多头注意力

使用多个注意力头，学习不同类型的关系：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

## Transformer 架构

### 编码器 (Encoder)

由 N 个相同的层组成，每层包含：
1. **多头自注意力层**
2. **位置前馈网络**

每层都有残差连接和层归一化：

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

### 解码器 (Decoder)

由 N 个相同的层组成，每层包含：
1. **掩码多头自注意力层**（防止前瞻）
2. **多头注意力层**（编码器-解码器注意力）
3. **位置前馈网络**

### 位置编码

由于 Transformer 没有循环结构，需要显式编码位置信息：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

## 前馈网络

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

隐藏层维度通常是模型维度的 4 倍。

## 训练细节

### 优化器

使用 Adam 优化器，学习率随训练步数变化：

$$lrate = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup\_steps^{-1.5})$$

### 正则化

- **Dropout**: 在注意力权重和前馈网络中使用
- **标签平滑**: 防止模型过于自信

## 实验结果

在 WMT 2014 英德翻译任务上取得了当时最优的 BLEU 分数：

| 模型 | BLEU |
|------|------|
| Transformer (base) | 28.4 |
| Transformer (big) | 29.9 |
| Google's GNMT | 28.1 |

## Python 实现示例

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
    
    attn_weights = softmax(scores)
    output = np.matmul(attn_weights, V)
    
    return output, attn_weights

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

## 影响

本文开创了基于注意力的序列建模范式，直接影响了：

1. **BERT**: 基于 Transformer 编码器的预训练模型
2. **GPT**: 基于 Transformer 解码器的预训练模型
3. **T5**: 统一的文本到文本 Transformer
4. **BART**: 序列到序列生成模型

Transformer 已成为 NLP 的标准架构，并扩展到计算机视觉等领域。
