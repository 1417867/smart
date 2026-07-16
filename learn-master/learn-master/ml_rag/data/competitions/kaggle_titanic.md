# Kaggle Titanic 竞赛方案

## 竞赛信息

- **竞赛名称**: Titanic - Machine Learning from Disaster
- **竞赛类型**: 分类问题
- **目标**: 预测乘客是否幸存

## 数据概况

| 特征 | 描述 | 类型 |
|------|------|------|
| PassengerId | 乘客ID | 数值 |
| Survived | 是否幸存 (0=否, 1=是) | 目标变量 |
| Pclass | 船舱等级 | 类别 |
| Name | 姓名 | 文本 |
| Sex | 性别 | 类别 |
| Age | 年龄 | 数值 |
| SibSp | 兄弟姐妹/配偶数量 | 数值 |
| Parch | 父母/子女数量 | 数值 |
| Ticket | 票号 | 文本 |
| Fare | 票价 | 数值 |
| Cabin | 舱位号 | 文本 |
| Embarked | 登船港口 | 类别 |

## 数据探索

### 缺失值分析

```python
import pandas as pd

df = pd.read_csv('train.csv')

# 查看缺失值比例
print(df.isnull().sum() / len(df) * 100)

# Age: 19.87% 缺失
# Cabin: 77.10% 缺失
# Embarked: 0.22% 缺失
```

### 幸存者比例

```python
survival_rate = df['Survived'].mean()
print(f"幸存者比例: {survival_rate:.2%}")
```

### 特征与生存率关系

```python
# 性别与生存率
print(df.groupby('Sex')['Survived'].mean())

# 船舱等级与生存率
print(df.groupby('Pclass')['Survived'].mean())

# 登船港口与生存率
print(df.groupby('Embarked')['Survived'].mean())
```

## 特征工程

### 年龄缺失值处理

```python
# 使用中位数填充
df['Age'] = df['Age'].fillna(df['Age'].median())

# 或者使用分组中位数
df['Age'] = df.groupby('Sex')['Age'].transform(
    lambda x: x.fillna(x.median())
)
```

### 家庭规模特征

```python
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 家庭规模分组
df['FamilyGroup'] = pd.cut(df['FamilySize'], 
                          bins=[0, 1, 4, 10], 
                          labels=['Alone', 'Small', 'Large'])
```

### 头衔特征

```python
import re

def extract_title(name):
    title = re.search(' ([A-Za-z]+)\.', name)
    if title:
        return title.group(1)
    return ''

df['Title'] = df['Name'].apply(extract_title)

# 合并稀有头衔
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 
               'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
```

### 票价特征

```python
# 票价分组
df['FareBand'] = pd.qcut(df['Fare'], 4, labels=[1, 2, 3, 4])
```

### 编码类别特征

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'].fillna('S'))
df['Title'] = le.fit_transform(df['Title'])
```

## 模型选择

### 常用模型

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier
)

models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}
```

### 模型评估

```python
from sklearn.model_selection import cross_val_score

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

## 模型调优

### 随机森林参数调优

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
```

## 集成学习

### Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier()),
    ('svm', SVC())
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stacking.fit(X_train, y_train)
```

## 提交准备

```python
# 加载测试数据
test_df = pd.read_csv('test.csv')

# 应用相同的预处理
# ...

# 预测
predictions = model.predict(test_df)

# 创建提交文件
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})

submission.to_csv('submission.csv', index=False)
```

## 关键发现

1. **性别是最重要的特征**：女性生存率远高于男性
2. **船舱等级重要**：1等舱乘客生存率最高
3. **年龄重要**：儿童和老人生存率较高
4. **家庭规模重要**：中等规模家庭生存率最高
5. **头衔包含信息**：贵族头衔生存率更高

## 进阶技巧

1. **特征交互**：Pclass * Sex, Age * Pclass
2. **交叉验证策略**：分层 K 折
3. **模型融合**：Stacking, Blending, Voting
4. **特征选择**：递归特征消除
5. **异常值处理**：IQR 方法

Titanic 竞赛是机器学习入门的经典项目，通过这个项目可以学习到完整的机器学习流程。
