# scikit-learn 快速入门

## 简介

scikit-learn 是 Python 中最流行的机器学习库，提供：
- 监督学习算法（分类、回归）
- 无监督学习算法（聚类、降维）
- 模型选择和评估工具
- 数据预处理工具

## 基本流程

### 1. 导入库

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### 2. 加载数据

```python
# 加载内置数据集
iris = datasets.load_iris()
X = iris.data    # 特征
y = iris.target  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3. 创建模型

```python
model = LogisticRegression()
```

### 4. 训练模型

```python
model.fit(X_train, y_train)
```

### 5. 预测

```python
y_pred = model.predict(X_test)
```

### 6. 评估

```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

## 常用算法

### 分类算法

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 逻辑回归
lr = LogisticRegression()

# 支持向量机
svm = SVC(kernel='rbf')

# 决策树
dt = DecisionTreeClassifier(max_depth=3)

# 随机森林
rf = RandomForestClassifier(n_estimators=100)
```

### 回归算法

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 线性回归
lr = LinearRegression()

# 决策树回归
dt = DecisionTreeRegressor()

# 随机森林回归
rf = RandomForestRegressor()
```

### 聚类算法

```python
from sklearn.cluster import KMeans, DBSCAN

# K-Means
kmeans = KMeans(n_clusters=3)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
```

### 降维算法

```python
from sklearn.decomposition import PCA

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```

## 数据预处理

### 标准化和归一化

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 标准化 (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 归一化 (0-1)
minmax = MinMaxScaler()
X_normalized = minmax.fit_transform(X)
```

### 特征编码

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 独热编码
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(X).toarray()
```

### 缺失值处理

```python
from sklearn.impute import SimpleImputer

# 均值填充
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

## 模型评估

### 分类指标

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 准确率
accuracy = accuracy_score(y_true, y_pred)

# 精确率、召回率、F1
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 分类报告
report = classification_report(y_true, y_pred)
```

### 回归指标

```python
from sklearn.metrics import mean_squared_error, r2_score

# MSE
mse = mean_squared_error(y_true, y_pred)

# R²
r2 = r2_score(y_true, y_pred)
```

## 模型选择

### 交叉验证

```python
from sklearn.model_selection import cross_val_score

# 5折交叉验证
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Scores: {scores}")
print(f"Mean Score: {scores.mean():.2f}")
```

### 网格搜索

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, None]
}

# 网格搜索
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

# 最佳参数
print(grid.best_params_)
```

## 流水线

```python
from sklearn.pipeline import Pipeline

# 创建流水线
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', LogisticRegression())
])

# 训练
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)
```

## 保存模型

```python
import joblib

# 保存模型
joblib.dump(model, 'model.joblib')

# 加载模型
model = joblib.load('model.joblib')
```

## 示例：完整分类流程

```python
# 加载数据
digits = datasets.load_digits()
X, y = digits.data, digits.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建流水线
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# 训练
pipeline.fit(X_train, y_train)

# 评估
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
```

scikit-learn 是机器学习入门的最佳选择，其简洁的 API 设计和丰富的文档使其成为学习和应用机器学习的首选工具。
