# PyTorch 基础教程

## 简介

PyTorch 是一个基于 Python 的深度学习框架，提供：
- 动态计算图
- 自动微分
- GPU 加速
- 丰富的工具库

## 张量操作

### 创建张量

```python
import torch

# 创建零张量
x = torch.zeros(3, 4)

# 创建随机张量
x = torch.randn(2, 3)

# 从 NumPy 数组创建
import numpy as np
arr = np.array([1, 2, 3])
x = torch.from_numpy(arr)

# 指定数据类型
x = torch.tensor([1, 2, 3], dtype=torch.float32)
```

### 张量运算

```python
# 基本运算
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

c = a + b          # 加法
c = a * b          # 乘法
c = a @ b          # 矩阵乘法

# 矩阵运算
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B          # 矩阵乘法 (3, 5)
```

### 索引和切片

```python
x = torch.randn(5, 4)

# 获取第1行
row = x[0]

# 获取前3行
rows = x[:3]

# 获取第2列
col = x[:, 1]

# 获取子矩阵
sub = x[1:4, 1:3]
```

## 自动微分

### 计算梯度

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward()       # 计算梯度

print(x.grad)      # 输出: tensor(7.)
```

### 多元函数梯度

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x[0] ** 2 + x[1] ** 2

y.backward()

print(x.grad)      # 输出: tensor([2., 4.])
```

### 禁用梯度计算

```python
with torch.no_grad():
    y = x ** 2     # 不会计算梯度
```

## 神经网络模块

### nn.Module 基类

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型
model = MyModel(10, 20, 2)
```

### 预定义层

```python
# 线性层
fc = nn.Linear(in_features, out_features)

# 卷积层
conv = nn.Conv2d(in_channels, out_channels, kernel_size)

# 池化层
pool = nn.MaxPool2d(kernel_size)

# 激活函数
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)
```

## 训练循环

### 完整训练示例

```python
import torch.optim as optim

# 超参数
learning_rate = 0.001
epochs = 100

# 数据
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# 模型、损失函数、优化器
model = MyModel(10, 20, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # 反向传播和优化
    optimizer.zero_grad()    # 清零梯度
    loss.backward()          # 计算梯度
    optimizer.step()         # 更新参数
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

## GPU 加速

### 检查 GPU 可用性

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 移动张量到 GPU

```python
x = x.to(device)
model = model.to(device)
```

## 数据加载

### Dataset 和 DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建 DataLoader
dataset = MyDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 遍历数据
for batch_data, batch_labels in dataloader:
    # 训练逻辑
    pass
```

## 保存和加载模型

### 保存模型

```python
torch.save(model.state_dict(), 'model.pth')
```

### 加载模型

```python
model = MyModel(10, 20, 2)
model.load_state_dict(torch.load('model.pth'))
model.eval()    # 设置为评估模式
```

## 常用技巧

### 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 学习率调度

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(epochs):
    # 训练逻辑
    scheduler.step()    # 更新学习率
```

### 混合精度训练

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(X)
    loss = criterion(outputs, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

PyTorch 是目前最流行的深度学习框架之一，掌握其基础对于进行深度学习研究和开发至关重要。
