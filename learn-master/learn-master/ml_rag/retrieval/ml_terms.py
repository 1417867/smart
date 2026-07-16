ML_TERMS = {
    "supervised learning": ["监督学习", "supervised"],
    "unsupervised learning": ["无监督学习", "unsupervised"],
    "reinforcement learning": ["强化学习", "reinforcement", "RL"],
    "overfitting": ["过拟合", "过拟合问题"],
    "underfitting": ["欠拟合", "欠拟合问题"],
    "bias-variance tradeoff": ["偏差方差权衡", "偏差-方差权衡", "bias variance"],
    "gradient descent": ["梯度下降", "梯度下降法"],
    "stochastic gradient descent": ["随机梯度下降", "SGD", "stochastic gradient"],
    "backpropagation": ["反向传播", "反向传播算法", "BP算法", "backprop"],
    "regularization": ["正则化", "正则化方法"],
    "normalization": ["归一化", "标准化"],
    "standardization": ["标准化", "Z-score标准化"],
    "cross-validation": ["交叉验证", "cross validation"],
    "hyperparameter tuning": ["超参数调优", "超参数优化"],
    "linear regression": ["线性回归", "linear reg"],
    "logistic regression": ["逻辑回归", "logistic reg", "分类回归"],
    "support vector machine": ["支持向量机", "SVM", "支持向量机"],
    "decision tree": ["决策树", "decision trees"],
    "random forest": ["随机森林", "随机森林算法"],
    "XGBoost": ["XGBoost", "极端梯度提升"],
    "k-means": ["k均值", "k均值聚类", "kmeans"],
    "PCA": ["主成分分析", "principal component"],
    "t-SNE": ["t分布邻域嵌入", "t-SNE可视化"],
    "CNN": ["卷积神经网络", "convolutional neural", "卷积网络"],
    "RNN": ["循环神经网络", "recurrent neural", "循环网络"],
    "LSTM": ["长短期记忆网络", "long short-term", "LSTM网络"],
    "GRU": ["门控循环单元", "gated recurrent"],
    "Transformer": ["Transformer模型", "transformer架构"],
    "GAN": ["生成对抗网络", "generative adversarial"],
    "VAE": ["变分自编码器", "variational autoencoder"],
    "BERT": ["BERT模型", "bidirectional encoder"],
    "GPT": ["GPT模型", "generative pre-trained"],
    "ResNet": ["残差网络", "residual network"],
    "U-Net": ["U-Net模型", "U型网络"],
    "YOLO": ["YOLO模型", "you only look once"],
    "batch normalization": ["批归一化", "batch norm"],
    "layer normalization": ["层归一化", "layer norm"],
    "dropout": ["dropout", "随机失活"],
    "learning rate decay": ["学习率衰减", "学习率下降"],
    "early stopping": ["早停", "提前停止"],
    "data augmentation": ["数据增强", "数据扩充"],
    "transfer learning": ["迁移学习", "transfer learning"],
    "fine-tuning": ["微调", "fine tuning"],
    "knowledge distillation": ["知识蒸馏", "知识提取"],
    "activation function": ["激活函数", "非线性激活"],
    "loss function": ["损失函数", "代价函数"],
    "objective function": ["目标函数"],
    "chain rule": ["链式法则", "链式求导"],
    "Lagrangian": ["拉格朗日函数", "拉格朗日乘子"],
    "convex optimization": ["凸优化", "凸优化问题"],
    "Adam optimizer": ["Adam优化器", "Adam算法"],
    "learning rate schedule": ["学习率调度", "学习率策略"],
    "precision": ["精确率", "查准率"],
    "recall": ["召回率", "查全率"],
    "F1 score": ["F1分数", "F1指标"],
    "AUC-ROC": ["AUC曲线", "ROC曲线", "AUC"],
    "confusion matrix": ["混淆矩阵"],
    "MSE": ["均方误差", "mean squared"],
    "MAE": ["平均绝对误差", "mean absolute"],
    "RMSE": ["均方根误差", "root mean"],
    "R-squared": ["R平方", "决定系数"],
    "perplexity": ["困惑度", "perplexity"],
    "BLEU": ["BLEU分数", "机器翻译评估"],
    "attention mechanism": ["注意力机制", "attention"],
    "self-attention": ["自注意力", "self attention"],
    "multi-head attention": ["多头注意力", "multi-head"],
    "encoder decoder": ["编码器解码器", "seq2seq"],
    "embedding": ["嵌入", "词嵌入", "向量表示"],
    "pre-trained model": ["预训练模型", "pretrained"],
    "fine-tune": ["微调"],
    "feature engineering": ["特征工程", "特征提取"],
    "feature selection": ["特征选择"],
    "dimensionality reduction": ["降维", "维度约减"],
    "ensemble learning": ["集成学习", "ensemble"],
    "bagging": ["装袋法", "bootstrap aggregating"],
    "boosting": ["提升法", "boosting算法"],
    "stacking": ["堆叠法", "stacking集成"],
    "Markov decision process": ["马尔可夫决策过程", "MDP"],
    "Q-learning": ["Q学习", "Q-learning算法"],
    "policy gradient": ["策略梯度", "policy gradient"],
    "value function": ["值函数", "value function"],
    "policy function": ["策略函数", "policy"],
    "actor critic": ["演员评论家", "actor-critic"],
    "natural language processing": ["自然语言处理", "NLP"],
    "computer vision": ["计算机视觉", "CV"],
    "object detection": ["目标检测", "物体检测"],
    "image classification": ["图像分类"],
    "semantic segmentation": ["语义分割"],
    "sentiment analysis": ["情感分析", "情感分类"],
    "machine translation": ["机器翻译"],
    "named entity recognition": ["命名实体识别", "NER"],
    "question answering": ["问答系统", "QA"],
    "text generation": ["文本生成"],
    "recommender system": ["推荐系统", "推荐算法"],
    "collaborative filtering": ["协同过滤", "CF"],
    "matrix factorization": ["矩阵分解"],
    "deep learning": ["深度学习", "DL"],
    "neural network": ["神经网络", "NN"],
    "feedforward network": ["前馈网络", "feedforward"],
    "recurrent network": ["循环网络"],
    "convolutional network": ["卷积网络"],
    "attention network": ["注意力网络"],
    "graph neural network": ["图神经网络", "GNN", "图网络"],
    "graph convolutional network": ["图卷积网络", "GCN"],
    "long short term memory": ["长短期记忆", "LSTM"],
    "gated recurrent unit": ["门控循环单元", "GRU"],
    "autoencoder": ["自编码器", "auto encoder"],
    "sparse autoencoder": ["稀疏自编码器"],
    "denoising autoencoder": ["降噪自编码器", "DAE"],
    "restricted boltzmann machine": ["受限玻尔兹曼机", "RBM"],
    "deep belief network": ["深度信念网络", "DBN"],
    "generative model": ["生成模型"],
    "discriminative model": ["判别模型"],
    "latent variable model": ["隐变量模型"],
    "probabilistic graphical model": ["概率图模型", "PGM"],
    "Bayesian network": ["贝叶斯网络"],
    "Markov random field": ["马尔可夫随机场", "MRF"],
    "hidden Markov model": ["隐马尔可夫模型", "HMM"],
    "conditional random field": ["条件随机场", "CRF"],
    "expectation maximization": ["期望最大化", "EM算法"],
    "variational inference": ["变分推断", "变分推理"],
    "Monte Carlo methods": ["蒙特卡洛方法", "Monte Carlo"],
    "Markov chain Monte Carlo": ["马尔可夫链蒙特卡洛", "MCMC"],
    "Gibbs sampling": ["吉布斯采样"],
    "importance sampling": ["重要性采样"],
    "reinforcement learning": ["强化学习", "RL"],
    "Markov decision process": ["马尔可夫决策过程", "MDP"],
    "Bellman equation": ["贝尔曼方程"],
    "value iteration": ["值迭代"],
    "policy iteration": ["策略迭代"],
    "Q-learning": ["Q学习"],
    "SARSA": ["SARSA算法"],
    "Deep Q-Network": ["深度Q网络", "DQN"],
    "Double DQN": ["双重DQN"],
    "Dueling DQN": ["决斗DQN"],
    "Prioritized Experience Replay": ["优先经验回放", "PER"],
    "policy gradient": ["策略梯度"],
    "REINFORCE": ["REINFORCE算法"],
    "actor-critic": ["演员评论家"],
    "Proximal Policy Optimization": ["近端策略优化", "PPO"],
    "Trust Region Policy Optimization": ["信任区域策略优化", "TRPO"],
    "Deep Deterministic Policy Gradient": ["深度确定性策略梯度", "DDPG"],
    "Twin Delayed DDPG": ["TD3算法"],
    "Soft Actor-Critic": ["软演员评论家", "SAC"],
    "curriculum learning": ["课程学习"],
    "imitation learning": ["模仿学习"],
    "inverse reinforcement learning": ["逆强化学习", "IRL"],
}

DIFFICULTY_TERMS = {
    "beginner": [
        "线性回归", "逻辑回归", "决策树", "随机森林", "K均值", 
        "梯度下降", "监督学习", "无监督学习", "交叉验证", 
        "特征工程", "过拟合", "欠拟合", "精确率", "召回率", "F1"
    ],
    "intermediate": [
        "神经网络", "卷积神经网络", "循环神经网络", "LSTM", "GRU",
        "反向传播", "批归一化", "Dropout", "迁移学习", "微调",
        "支持向量机", "PCA", "t-SNE", "XGBoost", "集成学习",
        "强化学习", "Q学习", "马尔可夫决策过程"
    ],
    "advanced": [
        "Transformer", "BERT", "GPT", "注意力机制", "自注意力",
        "生成对抗网络", "变分自编码器", "图神经网络", "GCN",
        "策略梯度", "PPO", "DDPG", "SAC", "Actor-Critic",
        "概率图模型", "变分推断", "MCMC", "蒙特卡洛方法",
        "元学习", "终身学习", "少样本学习", "零样本学习",
        "对比学习", "自监督学习", "预训练模型", "大语言模型"
    ]
}

DOMAIN_TERMS = {
    "dl": ["神经网络", "深度学习", "CNN", "RNN", "LSTM", "Transformer", 
           "BERT", "GPT", "GAN", "VAE", "ResNet", "反向传播"],
    "nlp": ["自然语言处理", "BERT", "GPT", "Transformer", "情感分析", 
            "命名实体识别", "机器翻译", "文本生成", "问答系统",
            "词嵌入", "注意力机制"],
    "cv": ["计算机视觉", "CNN", "ResNet", "YOLO", "U-Net", "图像分类", 
           "目标检测", "语义分割", "图像生成", "GAN"],
    "rl": ["强化学习", "Q学习", "DQN", "PPO", "策略梯度", "马尔可夫决策过程",
           "Actor-Critic", "SAC", "DDPG"],
    "optimization": ["梯度下降", "SGD", "Adam", "优化器", "学习率", 
                     "凸优化", "拉格朗日", "正则化"],
    "ml": ["监督学习", "无监督学习", "线性回归", "逻辑回归", "决策树", 
           "随机森林", "SVM", "XGBoost", "聚类", "特征工程"]
}

def find_ml_terms(text: str) -> list:
    found = []
    for term, aliases in ML_TERMS.items():
        all_forms = [term] + aliases
        for form in all_forms:
            if form.lower() in text.lower():
                found.append(term)
                break
    return list(set(found))

def detect_difficulty(text: str) -> str:
    advanced_count = sum(1 for term in DIFFICULTY_TERMS["advanced"] if term.lower() in text.lower())
    intermediate_count = sum(1 for term in DIFFICULTY_TERMS["intermediate"] if term.lower() in text.lower())
    
    if advanced_count >= 2:
        return "advanced"
    elif intermediate_count >= 2 or advanced_count >= 1:
        return "intermediate"
    else:
        return "beginner"

def detect_domains(text: str) -> list:
    domains = []
    for domain, terms in DOMAIN_TERMS.items():
        if any(term.lower() in text.lower() for term in terms):
            domains.append(domain)
    return domains if domains else ["ml"]
