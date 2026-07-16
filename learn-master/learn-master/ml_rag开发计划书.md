# 机器学习专属知识库 + RAG 智能体 — 完整开发计划书

> **项目定位**: 在 SmartLearn 通用 RAG 基础设施之上，构建机器学习领域的专属知识库与 RAG 智能体
> **目标用户**: 机器学习学习者、数据科学竞赛参赛者、AI 方向研究生
> **核心价值**: 让 AI 基于精选 ML 知识库进行精准、可溯源的专业问答

---

## 目录

1. [项目总览](#一项目总览)
2. [技术栈](#二技术栈)
3. [Phase 1: 基础设施准备](#三phase-1-基础设施准备-day-12)
4. [Phase 2: ML 数据采集与知识库构建](#四phase-2-ml-数据采集与知识库构建-day-37)
5. [Phase 3: ML 专属检索增强](#五phase-3-ml-专属检索增强-day-810)
6. [Phase 4: ML RAG 智能体开发](#六phase-4-ml-rag-智能体开发-day-1115)
7. [Phase 5: 前端交互与可视化](#七phase-5-前端交互与可视化-day-1618)
8. [Phase 6: 测试、部署与迭代](#八phase-6-测试部署与迭代-day-1921)
9. [交付清单汇总](#九交付清单汇总)
10. [里程碑与甘特图](#十里程碑与甘特图)

---

## 一、项目总览

### 1.1 项目背景

SmartLearn 已具备完整的通用 RAG 基础设施：

| 已有能力 | 状态 | 技术实现 |
|---------|------|---------|
| 知识库 CRUD | ✅ | `KnowledgeServiceImpl` + Prisma |
| 文档解析 | ✅ | PDF/DOCX/TXT/MD/代码文件 |
| 智能分块 | ✅ | `ChunkerService` (512 tokens, 段落感知) |
| 向量嵌入 | ✅ | OpenAI text-embedding-3-small (1536维) |
| pgvector 存储 | ✅ | `<=>` 余弦距离 + `::vector` cast |
| 语义检索 | ✅ | `RAGServiceImpl.search()` |
| 混合检索 | ✅ | BM25 + RRF 融合 + Reranker |
| KB Seed | ✅ | 预注入 System Prompt |
| RAG Tool | ✅ | Agent 按需调用 |

**但缺少**：
- ❌ 机器学习领域专用语料库
- ❌ ML 术语感知的分块策略
- ❌ ML 公式/代码/论文的特殊处理
- ❌ 多 KB 跨库联合检索
- ❌ 检索结果的可解释性引用
- ❌ ML 专属的 RAG 智能体 (评估回答质量 + 溯源)

### 1.2 开发目标

```
┌─────────────────────────────────────────────────────────────────┐
│                ML RAG 智能体 — 目标架构                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   用户提问                                 │   │
│  │  "什么是反向传播？推导一下梯度计算过程"                    │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            ML RAG Agent (新增)                             │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │ 1. 查询理解与重写 (Query Rewriting)                 │  │   │
│  │  │    "反向传播" → ["反向传播算法","BP算法",           │  │   │
│  │  │     "Backpropagation","梯度下降推导"]               │  │   │
│  │  ├────────────────────────────────────────────────────┤  │   │
│  │  │ 2. 多策略检索                                       │  │   │
│  │  │    向量检索 → 数学公式匹配 → 代码片段匹配            │  │   │
│  │  │    → 论文引用匹配 → 综合排序                        │  │   │
│  │  ├────────────────────────────────────────────────────┤  │   │
│  │  │ 3. 检索后处理 (Post-Retrieval)                     │  │   │
│  │  │    去重 → 重排序 → 公式渲染 → 溯源标注              │  │   │
│  │  ├────────────────────────────────────────────────────┤  │   │
│  │  │ 4. 回答生成 (Grounded Generation)                  │  │   │
│  │  │    System Prompt + ML KB Context + 引用要求         │  │   │
│  │  │    → 结构化回答 (含公式 + 代码 + 引用)              │  │   │
│  │  ├────────────────────────────────────────────────────┤  │   │
│  │  │ 5. 回答质量评估 (Self-Reflection)                  │  │   │
│  │  │    事实一致性检查 → 引用准确性验证 → 完整性评分     │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              ML 知识库集群 (新增 4 个 KB)                  │   │
│  │  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌─────────┐ │   │
│  │  │ ML 教材    │ │ 经典论文   │ │ 代码库   │ │ 竞赛    │ │   │
│  │  │ 花书/     │ │ ResNet/   │ │ sklearn/ │ │ Kaggle  │ │   │
│  │  │ PRML/李航  │ │ Transformer│ │ PyTorch  │ │ 方案    │ │   │
│  │  │ 西瓜书     │ │ /BERT/GPT  │ │ 官方文档 │ │ 合集    │ │   │
│  │  └────────────┘ └────────────┘ └──────────┘ └─────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 总工作量估算

| 阶段 | 内容 | 工作量 |
|------|------|--------|
| Phase 1 | 基础设施准备 | 2 天 |
| Phase 2 | ML 数据采集与 KB 构建 | 5 天 |
| Phase 3 | ML 专属检索增强 | 3 天 |
| Phase 4 | ML RAG 智能体开发 | 5 天 |
| Phase 5 | 前端交互与可视化 | 3 天 |
| Phase 6 | 测试、部署与迭代 | 3 天 |
| **合计** | | **21 天** |

---

## 二、技术栈

```
┌─────────────────────────────────────────────────────────────────┐
│                ML RAG 智能体 — 技术栈全景                          │
│                                                                  │
│  层级              技术选型                    说明               │
│  ─────────────────────────────────────────────────────────────  │
│  前端框架          Next.js 15 + React 19      复用现有           │
│  UI 组件           Radix UI + Tailwind CSS    复用现有           │
│  状态管理          Zustand (knowledge-store)  扩展               │
│                                                                  │
│  API 层            Next.js API Routes         新增端点           │
│  参数验证          Zod 3.24                   复用现有           │
│                                                                  │
│  智能体编排        LangGraph JS 0.3           新增 ML-RAG Graph  │
│  LLM 调用          Vercel AI SDK v5           复用现有           │
│  LLM 提供商        12+ 提供商                 复用现有           │
│                                                                  │
│  文档解析          pdf-parse + mammoth        复用现有           │
│  ML 分块器         ChunkerService (ML优化)    新增               │
│  公式提取          KaTeX AST parser           新增               │
│  代码识别          Shiki language detection   复用现有           │
│                                                                  │
│  向量嵌入          text-embedding-3-small     复用现有           │
│  向量存储          pgvector + PostgreSQL      复用现有           │
│  混合检索          BM25 + RRF + Reranker      复用+增强          │
│  Embedding 代理    SiliconFlow (低成本)        新增               │
│                                                                  │
│  ML 语料           OpenML / Papers With Code  新增 API 集成      │
│  Web 搜索          Tavily + DuckDuckGo        复用现有           │
│                                                                  │
│  测试              Vitest + Playwright         复用现有           │
│  部署              Docker Compose              复用现有           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、Phase 1: 基础设施准备 (Day 1-2)

### 3.1 任务总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: 基础设施准备 (2 天)                                     │
│                                                                  │
│  任务                                 工时    交付文件            │
│  ─────────────────────────────────────────────────────────────  │
│  1.1 创建 ML KB 专属目录结构          2h    目录树               │
│  1.2 配置 ML 专用环境变量             1h    .env 更新            │
│  1.3 新增 Prisma ML 元数据模型         3h    schema 迁移文件      │
│  1.4 搭建 ML 文档采集管线骨架          4h    collector/ 模块     │
│  1.5 Embedding 成本优化配置            2h    embedding config    │
│  1.6 编写 Phase 1 单元测试             4h    *.test.ts           │
│  ─────────────────────────────────────────────────────────────  │
│  合计                                 16h                        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 任务详情

#### 1.1 创建 ML KB 专属目录结构

```
lib/
├── ml-rag/                          # 新建 ML RAG 专属模块
│   ├── index.ts                     # 统一导出
│   ├── config.ts                    # ML RAG 配置常量
│   ├── collector/                   # 文档采集管线
│   │   ├── index.ts
│   │   ├── arxiv-collector.ts       # arXiv 论文采集
│   │   ├── doc-collector.ts         # 官方文档采集
│   │   └── textbook-processor.ts    # 教材处理
│   ├── chunker/                     # ML 专属分块
│   │   ├── index.ts
│   │   ├── ml-chunker.ts            # ML 感知分块器
│   │   ├── formula-extractor.ts     # 公式提取器
│   │   └── code-block-detector.ts   # 代码块检测器
│   ├── retrieval/                   # 检索增强
│   │   ├── index.ts
│   │   ├── query-rewriter.ts        # 查询重写
│   │   ├── multi-kb-search.ts       # 多 KB 联合检索
│   │   └── post-processor.ts        # 检索后处理
│   ├── agent/                       # RAG 智能体
│   │   ├── index.ts
│   │   ├── ml-rag-graph.ts          # LangGraph 状态图
│   │   ├── prompt-templates.ts      # Prompt 模板
│   │   └── quality-evaluator.ts     # 回答质量评估
│   └── __tests__/                   # 测试
│       ├── ml-chunker.test.ts
│       ├── query-rewriter.test.ts
│       ├── multi-kb-search.test.ts
│       └── ml-rag-graph.test.ts
│
data/
├── ml-kb/                           # ML 语料存储
│   ├── textbooks/                   # 教材原始文件
│   ├── papers/                      # 论文原始文件
│   ├── docs/                        # 官方文档快照
│   └── competitions/                # 竞赛方案
```

#### 1.2 配置 ML 专用环境变量

```bash
# .env 新增

# -- ML RAG 配置 --
ML_RAG_ENABLED=true
ML_KB_DIR=data/ml-kb

# ML 专属 Embedding (可选低成本方案)
ML_EMBEDDING_PROVIDER=siliconflow
ML_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
ML_EMBEDDING_API_KEY=${SILICONFLOW_API_KEY}

# ML 语料 API Keys
ARXIV_API_ENABLED=false            # arXiv API 免费, 按需启用
PAPERS_WITH_CODE_API_KEY=          # 可选

# ML RAG 参数
ML_CHUNK_SIZE=1024                 # ML 文档分块大小
ML_CHUNK_OVERLAP=200               # 重叠量
ML_RETRIEVAL_TOP_K=10              # 检索结果数
ML_RERANK_ENABLED=true             # 重排序开关
ML_QUERY_REWRITE_ENABLED=true      # 查询重写开关
```

#### 1.3 新增 Prisma ML 元数据模型

```prisma
// prisma/schema.prisma 新增

/// ML 知识库 — 在通用 KB 基础上附加 ML 领域元数据
model MlKnowledgeBase {
  id            String   @id @default(cuid())
  kbId          String   @unique @map("kb_id")  // 指向 DtKnowledgeBase
  domain        String   // ml | dl | nlp | cv | rl | optimization
  language      String   @default("zh")          // zh | en | mixed
  paperCount    Int      @default(0)
  codeSnippets  Int      @default(0)
  formulaCount  Int      @default(0)
  metadata      Json?
  createdAt     DateTime @default(now()) @map("created_at")

  @@index([domain])
  @@map("ml_knowledge_bases")
}

/// ML 文档增强 — 扩展通用 DtDocument 的 ML 专属元数据
model MlDocumentMeta {
  id            String   @id @default(cuid())
  documentId    String   @unique @map("document_id") // 指向 DtDocument
  sourceType    String   // textbook | paper | official_doc | competition
  arxivId       String?  @map("arxiv_id")
  authors       String[]
  publishYear   Int?
  mlKeywords    String[]  // 从文档中提取的 ML 术语
  hasFormulas   Boolean  @default(false) @map("has_formulas")
  hasCode       Boolean  @default(false) @map("has_code")
  difficulty    String?  // beginner | intermediate | advanced
  prerequisites String[] // 前置知识点
  bibtex        String?  // BibTeX 引用
  createdAt     DateTime @default(now()) @map("created_at")

  @@index([sourceType])
  @@index([mlKeywords])
  @@map("ml_document_meta")
}

/// RAG 查询日志 — 用户查询与反馈记录 (用于质量迭代)
model MlRagQueryLog {
  id            String   @id @default(cuid())
  userId        String   @map("user_id")
  query         String                     // 原始查询
  rewrittenQueries String[]                // 重写后的查询
  kbIds         String[] @map("kb_ids")    // 检索的 KB
  retrievalTimeMs Int    @map("retrieval_time_ms")
  topSources    Json                       // 检索到的 top source IDs
  generatedAnswer String?                  // 生成的回答
  userFeedback  String?  @map("user_feedback")  // helpful | not_helpful | partially
  userComment   String?  @map("user_comment")
  createdAt     DateTime @default(now()) @map("created_at")

  @@index([userId])
  @@index([createdAt])
  @@map("ml_rag_query_logs")
}
```

**数据库迁移命令**：
```bash
npx prisma migrate dev --name add_ml_rag_models
```

#### 1.4 搭建 ML 文档采集管线骨架

```typescript
// lib/ml-rag/collector/index.ts
export interface CollectResult {
  sourceType: 'textbook' | 'paper' | 'official_doc' | 'competition';
  title: string;
  content: string;
  metadata: {
    authors?: string[];
    arxivId?: string;
    publishYear?: number;
    mlKeywords: string[];
    hasFormulas: boolean;
    hasCode: boolean;
    difficulty?: 'beginner' | 'intermediate' | 'advanced';
    bibtex?: string;
  };
}

export interface CollectorConfig {
  targetDir: string;       // data/ml-kb/
  maxFileSizeMB: number;   // 默认 50
  supportedFormats: string[]; // ['.pdf', '.md', '.txt', '.ipynb']
}

// arxiv-collector.ts — arXiv API 采集
// 搜索关键词: "machine learning", "deep learning", "neural networks"
// 按引用数排序, 取 top 50
// 下载 PDF + 元数据

// doc-collector.ts — 官方文档采集
// scikit-learn, PyTorch, TensorFlow 文档
// wget 镜像 + 按章节拆分

// textbook-processor.ts — 教材处理
// 花书(Deep Learning), PRML, 西瓜书, 李航统计学习方法
// 手动整理目录结构 + 分章节存储
```

#### 1.5 Embedding 成本优化配置

```
┌─────────────────────────────────────────────────────────────────┐
│              Embedding 成本对比与优化策略                           │
│                                                                  │
│  方案           模型                      每100万token   适用场景  │
│  ────────────────────────────────────────────────────────────── │
│  A (默认)    text-embedding-3-small    $0.02         通用        │
│  B (推荐)    BAAI/bge-large-zh-v1.5   ¥0.72/百万token 中文教材  │
│              (via SiliconFlow)                               │
│  C (免费)    BAAI/bge-small-zh-v1.5   免费 (本地)     开发测试    │
│              (via Ollama 本地部署)                            │
│                                                                  │
│  混合策略:                                                       │
│  · 中文内容 → BGE 中文模型 (方案B, 更好的中文语义理解)            │
│  · 英文内容 → text-embedding-3-small (方案A)                     │
│  · 公式块  → 转 LaTeX 文本后嵌入                                 │
│  · 代码块  → AST 结构文本后嵌入                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.6 Phase 1 交付文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `lib/ml-rag/index.ts` | 新建 | 模块统一导出 |
| `lib/ml-rag/config.ts` | 新建 | ML RAG 全局配置 |
| `lib/ml-rag/collector/index.ts` | 新建 | 采集器接口定义 |
| `prisma/schema.prisma` | 修改 | 新增 3 个 ML 模型 |
| `prisma/migrations/*/migration.sql` | 自动生成 | 迁移 SQL |
| `.env` 更新文档 | 修改 | ML 环境变量 |
| `lib/ml-rag/__tests__/config.test.ts` | 新建 | 配置测试 |

---

## 四、Phase 2: ML 数据采集与知识库构建 (Day 3-7)

### 4.1 任务总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: ML 数据采集与知识库构建 (5 天)                          │
│                                                                  │
│  任务                                 工时    交付文件            │
│  ─────────────────────────────────────────────────────────────  │
│  2.1 arXiv 经典论文采集器              6h    arxiv-collector.ts  │
│  2.2 官方文档采集器 (sklearn/PyTorch)   6h    doc-collector.ts   │
│  2.3 ML 教材处理器                     6h    textbook-processor  │
│  2.4 竞赛方案采集器                     4h    competition-collect│
│  2.5 ML 感知分块器                      8h    ml-chunker.ts      │
│  2.6 公式/代码提取器                    6h    formula/code ext   │
│  2.7 批量索引管道                      4h    batch-indexer.ts   │
│  ─────────────────────────────────────────────────────────────  │
│  合计                                 40h                        │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 任务详情

#### 2.1 arXiv 经典论文采集器

```typescript
// lib/ml-rag/collector/arxiv-collector.ts

/**
 * arXiv ML 经典论文采集
 *
 * 采集策略:
 * 1. 搜索 ML 各子领域 (DL/NLP/CV/RL/GNN/优化)
 * 2. 按引用数/影响力排序
 * 3. 筛选出 top 50 篇经典论文
 * 4. 下载 PDF + 提取元数据
 *
 * 目标论文列表 (示例):
 * ┌────────────────────────────────────────────────────────┐
 * │ 领域     论文                         年份   引用数      │
 * │ ────────────────────────────────────────────────────── │
 * │ DL     AlexNet (Krizhevsky)         2012   150K+      │
 * │ DL     ResNet (He et al.)           2015   130K+      │
 * │ DL     BatchNorm (Ioffe & Szegedy)  2015   80K+       │
 * │ NLP    Attention (Vaswani et al.)   2017   120K+      │
 * │ NLP    BERT (Devlin et al.)         2018   90K+       │
 * │ NLP    GPT (Radford et al.)         2018   30K+       │
 * │ CV     GAN (Goodfellow et al.)      2014   70K+       │
 * │ CV     U-Net (Ronneberger)          2015   60K+       │
 * │ RL     DQN (Mnih et al.)            2015   35K+       │
 * │ 优化    Adam (Kingma & Ba)           2014   120K+      │
 * │ GNN    GCN (Kipf & Welling)         2016   25K+       │
 * │ ...更多  (top 50)                                          │
 * └────────────────────────────────────────────────────────┘
 *
 * API: http://export.arxiv.org/api/query?search_query=...
 * 速率限制: 1 req / 3s (礼貌访问)
 */

export interface ArxivPaper {
  arxivId: string;
  title: string;
  authors: string[];
  abstract: string;
  publishedYear: number;
  pdfUrl: string;
  categories: string[];      // cs.LG, cs.CV, cs.CL, etc.
  mlKeywords: string[];      // 提取的关键词
  citationCount?: number;
}

export class ArxivCollector {
  private readonly BASE_URL = 'http://export.arxiv.org/api/query';
  private readonly DELAY_MS = 3000; // Rate limiting

  async searchByDomain(
    domain: string,
    maxResults: number = 50
  ): Promise<ArxivPaper[]>;

  async downloadPaper(paper: ArxivPaper): Promise<string>;
  // 返回本地文件路径

  async collectAll(
    domains: string[],
    outputDir: string
  ): Promise<CollectResult[]>;
}
```

#### 2.2 官方文档采集器

```typescript
// lib/ml-rag/collector/doc-collector.ts

/**
 * ML 框架官方文档采集
 *
 * 采集源:
 * ┌────────────────────────────────────────────────────────┐
 * │ 框架        文档地址                   采集方式          │
 * │ ────────────────────────────────────────────────────── │
 * │ scikit-learn https://scikit-learn.org/  wget 镜像       │
 * │ PyTorch     https://pytorch.org/docs/   wget 镜像       │
 * │ NumPy       https://numpy.org/doc/      wget 镜像       │
 * │ Pandas      https://pandas.pydata.org/  wget 镜像       │
 * │ XGBoost     https://xgboost.readthedocs.io/ wget 镜像  │
 * │ LightGBM    https://lightgbm.readthedocs.io/ wget 镜像  │
 * └────────────────────────────────────────────────────────┘
 *
 * 处理流程:
 * 1. wget 下载 HTML 文档
 * 2. html-to-text 提取纯文本
 * 3. 按 API 章节拆分为独立文档
 * 4. 提取函数签名 + 参数说明 + 代码示例
 */

export interface DocSection {
  apiName: string;            // e.g., "sklearn.ensemble.RandomForestClassifier"
  signature: string;          // 函数签名
  description: string;        // 描述
  parameters: string;         // 参数说明
  codeExamples: string[];     // 代码示例
  seeAlso: string[];          // 相关 API
  category: string;           // ensemble / linear_model / clustering / ...
}

export class DocCollector {
  async mirrorDocs(
    framework: string,
    docUrl: string,
    outputDir: string
  ): Promise<string>;

  async extractApiSections(
    htmlDir: string
  ): Promise<DocSection[]>;

  async generateMLDocuments(
    sections: DocSection[]
  ): Promise<CollectResult[]>;
}
```

#### 2.3 ML 教材处理器

```typescript
// lib/ml-rag/collector/textbook-processor.ts

/**
 * ML 经典教材结构化处理
 *
 * 待处理教材 (版权允许的公开版本):
 * ┌────────────────────────────────────────────────────────┐
 * │ 教材                      作者          语言  章节数     │
 * │ ────────────────────────────────────────────────────── │
 * │ 深度学习 (花书)           Goodfellow   EN    20 章     │
 * │ PRML (模式识别与ML)       Bishop       EN    14 章     │
 * │ 统计学习方法              李航         ZH    11 章     │
 * │ 机器学习 (西瓜书)         周志华       ZH    16 章     │
 * │ ESL (统计学习基础)        Hastie       EN    18 章     │
 * │ 动手学深度学习            李沐         ZH/EN 22 章     │
 * └────────────────────────────────────────────────────────┘
 *
 * 处理流程:
 * 1. 按章节拆分 → chapter-level documents
 * 2. 提取每章的核心概念 → mlKeywords
 * 3. 标注难度 → difficulty
 * 4. 标注前置知识 → prerequisites
 * 5. 提取公式列表 → hasFormulas + formulaCount
 */

export interface TextbookChapter {
  bookTitle: string;
  chapterNumber: number;
  chapterTitle: string;
  content: string;
  concepts: string[];         // 核心概念
  formulas: string[];         // LaTeX 公式列表
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  prerequisites: string[];    // 前置章节/知识点
}

export class TextbookProcessor {
  async processBook(
    bookPath: string,
    bookMeta: { title: string; language: string }
  ): Promise<TextbookChapter[]>;

  async extractFormulas(
    chapter: TextbookChapter
  ): Promise<string[]>;  // LaTeX strings

  async enrichMetadata(
    chapter: TextbookChapter
  ): Promise<CollectResult>;
}
```

#### 2.4 竞赛方案采集器

```
┌─────────────────────────────────────────────────────────────────┐
│              Kaggle 竞赛方案采集                                   │
│                                                                  │
│  采集策略: 优先采集 Top-3 方案, 按竞赛类型分类                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 竞赛类型      代表竞赛              Top方案关键词           │   │
│  │ ──────────────────────────────────────────────────────── │   │
│  │ 表格数据      Titanic, House Prices  EDA, Feature Eng    │   │
│  │ 图像分类      ImageNet, CIFAR       ResNet, Augmentation │   │
│  │ NLP           Sentiment, Translation BERT, LSTM, GPT     │   │
│  │ 时间序列      Stock, Energy         LSTM, XGBoost        │   │
│  │ 推荐系统      Netflix, RecSys       CF, Matrix Fact.     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  数据来源:                                                        │
│  · Kaggle Discussion (公开 notebook)                            │
│  · GitHub Awesome-Kaggle 合集                                   │
│  · Papers With Code 竞赛 Leaderboard                             │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.5 ML 感知分块器

这是区别于通用分块器的核心模块：

```typescript
// lib/ml-rag/chunker/ml-chunker.ts

/**
 * ML 感知分块器
 *
 * 在通用 ChunkerService 基础上增加 ML 领域感知:
 * 1. 公式块保护 — 不切断 LaTeX 公式 ($$ ... $$ 和 $ ... $)
 * 2. 代码块保护 — 不切断 ```python ... ``` 代码块
 * 3. 定义完整性 — 确保"定义/定理/引理"的完整
 * 4. 推导连续性 — 数学推导步骤保持连续
 * 5. 引用保留 — 保留 [Author, Year] 格式引用
 * 6. 术语密度 — 计算每个 chunk 的 ML 术语密度作为元数据
 */

export interface MlChunkConfig {
  chunkSize: number;         // 1024 (ML文档建议更大)
  chunkOverlap: number;      // 200
  protectFormulas: boolean;  // true
  protectCodeBlocks: boolean;// true
  protectDefinitions: boolean;// true
  termDensityThreshold: number; // 0.05 过滤低术语密度 chunk
}

export interface MlChunk extends TextChunk {
  formulaCount: number;
  codeSnippetCount: number;
  mlTermDensity: number;     // ML术语数 / 总词数
  definitionBlocks: string[];
  sourceType: 'textbook' | 'paper' | 'doc' | 'competition';
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
}

// ML 术语词典 (部分)
const ML_TERMS = new Set([
  // 基础概念
  'supervised learning', 'unsupervised learning', 'reinforcement learning',
  'overfitting', 'underfitting', 'bias-variance tradeoff',
  'gradient descent', 'stochastic gradient descent', 'backpropagation',
  'regularization', 'normalization', 'standardization',
  'cross-validation', 'hyperparameter tuning',
  // 模型
  'linear regression', 'logistic regression', 'SVM', 'decision tree',
  'random forest', 'XGBoost', 'k-means', 'PCA', 't-SNE',
  'CNN', 'RNN', 'LSTM', 'GRU', 'Transformer', 'GAN', 'VAE',
  'BERT', 'GPT', 'ResNet', 'U-Net', 'YOLO',
  // 训练技巧
  'batch normalization', 'layer normalization', 'dropout',
  'learning rate decay', 'early stopping', 'data augmentation',
  'transfer learning', 'fine-tuning', 'knowledge distillation',
  // 数学
  'activation function', 'loss function', 'objective function',
  'backpropagation', 'chain rule', 'Lagrangian', 'convex optimization',
  'stochastic gradient', 'Adam optimizer', 'learning rate schedule',
  // 评估
  'precision', 'recall', 'F1 score', 'AUC-ROC', 'confusion matrix',
  'MSE', 'MAE', 'RMSE', 'R-squared', 'perplexity', 'BLEU',
  // ... 更多 (200+ 术语)
]);
```

#### 2.6 公式与代码提取器

```typescript
// lib/ml-rag/chunker/formula-extractor.ts

/**
 * LaTeX 公式提取器
 *
 * 功能:
 * 1. 识别 $$ ... $$ 块级公式 (单独存储, 使用特殊 embedding 策略)
 * 2. 识别 $ ... $ 行内公式
 * 3. 公式 → 语义描述转换 (用 LLM 生成公式的自然语言描述)
 * 4. 公式索引: formula → chunk 反向索引
 *
 * 公式的 Embedding 策略:
 * · 不是直接 embed LaTeX 代码
 * · 而是 embed 公式的"语义描述 + LaTeX 源码"拼接
 * · 例: "反向传播链式法则公式: ∂L/∂w = ∂L/∂y * ∂y/∂w"
 */

// lib/ml-rag/chunker/code-block-detector.ts

/**
 * 代码块检测器
 *
 * 功能:
 * 1. 识别 ```python ... ``` 等围栏代码块
 * 2. 识别 import 语句
 * 3. 识别 API 调用 (sklearn.*, torch.*, tf.*)
 * 4. 代码 → AST 结构文本 (提取函数名、类名、调用链)
 * 5. 代码索引: API name → chunk 反向索引
 */
```

#### 2.7 批量索引管道

```typescript
// lib/ml-rag/batch-indexer.ts

/**
 * ML 语料批量索引管道
 *
 * 流程:
 * ┌──────────────────────────────────────────────────────────┐
 * │ 1. Collect → 采集所有文档到 data/ml-kb/                   │
 * │ 2. Parse   → ParsingService 解析为纯文本                  │
 * │ 3. Extract → 提取公式/代码/ML术语元数据                    │
 * │ 4. Chunk   → MlChunker 分块                               │
 * │ 5. Embed   → EmbeddingService 批量嵌入 (10/批)            │
 * │ 6. Store   → pgvector + Prisma                           │
 * │ 7. Verify  → 抽样验证检索质量                              │
 * └──────────────────────────────────────────────────────────┘
 *
 * 使用: npx tsx lib/ml-rag/batch-indexer.ts
 * 预计总时间: ~30 分钟 (取决于语料大小和 API 速率)
 */

export async function runBatchIndexing(
  options: {
    targetKbs: ('textbooks' | 'papers' | 'docs' | 'competitions')[];
    embeddingProvider: 'openai' | 'siliconflow' | 'ollama';
    maxConcurrency: number;  // 3 (防止 API 限流)
  }
): Promise<{
  totalDocs: number;
  totalChunks: number;
  totalTokens: number;
  totalCost: number;  // 估算成本
  durationMs: number;
  errors: Array<{ doc: string; error: string }>;
}>;
```

### 4.3 数据采集后的知识库规模预估

```
┌─────────────────────────────────────────────────────────────────┐
│              4 个 ML 知识库规模预估                               │
│                                                                  │
│  KB名称        文档数   总字符数    分块数   向量维度   存储大小   │
│  ─────────────────────────────────────────────────────────────  │
│  ML 教材        20      2.5M       2,800    1536      ~17MB     │
│  经典论文       50      1.8M       2,100    1536      ~13MB     │
│  官方文档       200+    5.0M       5,500    1536      ~34MB     │
│  竞赛方案       30      0.8M       900      1536      ~6MB      │
│  ─────────────────────────────────────────────────────────────  │
│  合计           300+    10.1M      11,300            ~70MB     │
│                                                                  │
│  Embedding 成本估算 (text-embedding-3-small):                    │
│  · 11,300 chunks × ~500 tokens/chunk = ~5.65M tokens            │
│  · $0.02 / 1M tokens × 5.65 = ~$0.11                            │
│  → 嵌入全部语料仅需约 $0.12!                                     │
│                                                                  │
│  Embedding 成本估算 (SiliconFlow BGE 中文模型):                   │
│  · ¥0.72 / 百万 token × ~5.65 = ~¥4.07                          │
│  → 中文教材使用 BGE, 英文使用 OpenAI, 总成本 < $1                │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Phase 2 交付文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `lib/ml-rag/collector/arxiv-collector.ts` | 新建 | arXiv 论文采集 |
| `lib/ml-rag/collector/doc-collector.ts` | 新建 | 官方文档采集 |
| `lib/ml-rag/collector/textbook-processor.ts` | 新建 | 教材处理 |
| `lib/ml-rag/collector/competition-collector.ts` | 新建 | 竞赛方案采集 |
| `lib/ml-rag/chunker/ml-chunker.ts` | 新建 | ML 感知分块器 |
| `lib/ml-rag/chunker/formula-extractor.ts` | 新建 | 公式提取器 |
| `lib/ml-rag/chunker/code-block-detector.ts` | 新建 | 代码检测器 |
| `lib/ml-rag/batch-indexer.ts` | 新建 | 批量索引管道 |
| `lib/ml-rag/config.ts` | 修改 | 更新 ML chunk 默认值 |
| `data/ml-kb/` | 新增目录 | ML 语料存储 |

---

## 五、Phase 3: ML 专属检索增强 (Day 8-10)

### 5.1 任务总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: ML 专属检索增强 (3 天)                                  │
│                                                                  │
│  任务                                 工时    交付文件            │
│  ─────────────────────────────────────────────────────────────  │
│  3.1 ML 查询重写器                     5h    query-rewriter.ts   │
│  3.2 多 KB 联合检索                    5h    multi-kb-search.ts  │
│  3.3 公式感知检索                      4h    formula-search.ts   │
│  3.4 代码感知检索                      4h    code-search.ts      │
│  3.5 检索后处理管道                    4h    post-processor.ts   │
│  3.6 检索质量评估脚本                  2h    eval-retrieval.ts  │
│  ─────────────────────────────────────────────────────────────  │
│  合计                                 24h                        │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 任务详情

#### 3.1 ML 查询重写器

```typescript
// lib/ml-rag/retrieval/query-rewriter.ts

/**
 * ML 查询重写器
 *
 * 功能: 将用户自然语言查询改写为多个更适合检索的查询变体
 *
 * 策略:
 * ┌──────────────────────────────────────────────────────────┐
 * │ 1. 术语标准化                                             │
 * │    "反向传播" → "反向传播 (Backpropagation)"              │
 * │    "梯度下降" → "梯度下降 (Gradient Descent)"             │
 * │    "过拟合"   → "过拟合 (Overfitting)"                   │
 * │                                                          │
 * │ 2. 英文变体生成                                           │
 * │    "反向传播" → "backpropagation algorithm"              │
 * │                                                          │
 * │ 3. 上下文补充                                             │
 * │    "Adam 是什么"                                          │
 * │    → "Adam 优化器 (Adaptive Moment Estimation)"          │
 * │    → "Adam optimization algorithm Kingma Ba 2014"        │
 * │                                                          │
 * │ 4. 子问题分解                                             │
 * │    "比较 SGD 和 Adam"                                     │
 * │    → ["SGD 优化器原理", "Adam 优化器原理",                │
 * │        "SGD vs Adam 对比", "优化器选择指南"]              │
 * │                                                          │
 * │ 5. 查询难度评估                                           │
 * │    "什么是线性回归" → difficulty: beginner                │
 * │    "推导 VAE 的 ELBO" → difficulty: advanced              │
 * └──────────────────────────────────────────────────────────┘
 *
 * 实现: LLM 调用 (轻量模型, gpt-4o-mini) + ML术语词典规则
 * 延迟目标: < 500ms
 */

export interface QueryRewriteResult {
  originalQuery: string;
  standardizedQuery: string;   // 术语标准化后
  variants: string[];          // 检索变体 (中/英/补充)
  subQuestions: string[];      // 子问题分解
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  targetDomains: string[];     // 匹配的 ML 子领域
}
```

#### 3.2 多 KB 联合检索

```typescript
// lib/ml-rag/retrieval/multi-kb-search.ts

/**
 * 多 KB 联合检索器
 *
 * 策略:
 * ┌──────────────────────────────────────────────────────────┐
 * │ 1. 并行检索所有 KB                                        │
 * │    · ML教材 KB (权重 1.0 ~ 1.5, 根据查询难度)           │
 * │    · 经典论文 KB (权重 1.0)                              │
 * │    · 官方文档 KB (权重 0.8, 偏代码查询提升到 1.5)       │
 * │    · 竞赛方案 KB (权重 0.5)                              │
 * │                                                          │
 * │ 2. 结果合并与去重                                         │
 * │    · 语义去重: 内容相似度 > 0.9 → 合并                   │
 * │    · 源去重: 同一文档的多个 chunk → 保留 top-2           │
 * │                                                          │
 * │ 3. 多样性增强 (MMR)                                       │
 * │    · Maximal Marginal Relevance                          │
 * │    · λ = 0.7 (相关性 vs 多样性的权衡)                    │
 * │                                                          │
 * │ 4. 最终排序                                               │
 * │    · 综合分数 = 向量相似度 × 0.4 + BM25 × 0.2            │
 * │                 + Reranker × 0.3 + 源权重 × 0.1          │
 * └──────────────────────────────────────────────────────────┘
 */

export interface MultiKBSearchOptions {
  kbIds: string[];
  kbWeights?: Record<string, number>;
  topK: number;                 // 最终返回结果数 (默认 10)
  dedup: boolean;               // 去重 (默认 true)
  diversityLambda: number;      // MMR λ (默认 0.7)
  includeFormulas: boolean;     // 包含匹配的公式 (默认 true)
  includeCode: boolean;         // 包含匹配的代码 (默认 true)
}

export interface MultiKBSearchResult {
  results: RAGSource[];
  dedupRemoved: number;
  diversityScore: number;
  perKbStats: Record<string, { found: number; contributed: number }>;
}
```

#### 3.3 公式感知检索

```typescript
// lib/ml-rag/retrieval/formula-search.ts

/**
 * 公式感知检索
 *
 * 问题: 用户问"交叉熵损失函数的公式是什么？"
 * 通用 RAG 可能找到提到"交叉熵"的 chunk, 但不一定包含公式
 *
 * 解决方案:
 * 1. 建立公式 → chunk_id 的反向索引
 * 2. 用户查询时检测是否包含公式需求关键词:
 *    "公式" / "推导" / "证明" / "损失函数" / "激活函数"
 * 3. 如果检测到 → 优先返回包含公式的 chunk
 *    → 公式 LaTeX 渲染为可读数学式
 */

export interface FormulaSearchResult {
  formulaLaTeX: string;
  formulaDescription: string;  // LLM 生成的语义描述
  chunkId: string;
  documentTitle: string;
  context: string;             // 公式所在段落的上下文
}
```

#### 3.4 检索后处理管道

```
┌─────────────────────────────────────────────────────────────────┐
│              检索后处理管道 (Post-Retrieval Pipeline)              │
│                                                                  │
│  Raw Results (top 30)                                            │
│      │                                                           │
│      ▼                                                           │
│  ┌────────────────────┐                                         │
│  │ 1. 分数过滤         │ score < 0.3 → discard                   │
│  └────────┬───────────┘                                         │
│           │                                                       │
│           ▼                                                       │
│  ┌────────────────────┐                                         │
│  │ 2. 语义去重         │ cosine(content_i, content_j) > 0.9     │
│  │                     │ → 保留 score 高的                       │
│  └────────┬───────────┘                                         │
│           │                                                       │
│           ▼                                                       │
│  ┌────────────────────┐                                         │
│  │ 3. MMR 多样性      │ 平衡相关性与多样性                        │
│  └────────┬───────────┘                                         │
│           │                                                       │
│           ▼                                                       │
│  ┌────────────────────┐                                         │
│  │ 4. 公式/代码补充    │ 从反向索引补全缺失的公式/代码             │
│  └────────┬───────────┘                                         │
│           │                                                       │
│           ▼                                                       │
│  ┌────────────────────┐                                         │
│  │ 5. 溯源信息补全     │ 添加作者/年份/来源/BibTeX                 │
│  └────────┬───────────┘                                         │
│           │                                                       │
│           ▼                                                       │
│  ┌────────────────────┐                                         │
│  │ 6. Context 组装     │ [Source: X, Chunk Y]\n content          │
│  └────────────────────┘   maxContextLength: 12000 (ML场景更大)   │
│                                                                  │
│  Final Results (top 10)                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Phase 3 交付文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `lib/ml-rag/retrieval/query-rewriter.ts` | 新建 | ML 查询重写 |
| `lib/ml-rag/retrieval/multi-kb-search.ts` | 新建 | 多 KB 联合检索 |
| `lib/ml-rag/retrieval/formula-search.ts` | 新建 | 公式感知检索 |
| `lib/ml-rag/retrieval/code-search.ts` | 新建 | 代码感知检索 |
| `lib/ml-rag/retrieval/post-processor.ts` | 新建 | 检索后处理管道 |
| `lib/ml-rag/retrieval/eval-retrieval.ts` | 新建 | 检索质量评估脚本 |
| `lib/ml-rag/__tests__/query-rewriter.test.ts` | 新建 | 测试 |
| `lib/ml-rag/__tests__/multi-kb-search.test.ts` | 新建 | 测试 |

---

## 六、Phase 4: ML RAG 智能体开发 (Day 11-15)

### 6.1 任务总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: ML RAG 智能体开发 (5 天)                                │
│                                                                  │
│  任务                                 工时    交付文件            │
│  ─────────────────────────────────────────────────────────────  │
│  4.1 LangGraph 状态图设计              6h    ml-rag-graph.ts     │
│  4.2 Prompt 工程 (System + Few-shot)   5h    prompt-templates.ts │
│  4.3 RAG Agent 节点实现                8h    agent/*.ts          │
│  4.4 引用溯源系统                       4h    citation-engine.ts │
│  4.5 回答质量自评估                     6h    quality-evaluator  │
│  4.6 SmartLearn Capability 注册         3h    集成到 bootstrap   │
│  4.7 API 端点实现                       4h    api/v1/ml-rag/    │
│  4.8 端到端集成测试                     4h    e2e test          │
│  ─────────────────────────────────────────────────────────────  │
│  合计                                 40h                        │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 任务详情

#### 4.1 LangGraph 状态图设计

```typescript
// lib/ml-rag/agent/ml-rag-graph.ts

/**
 * ML RAG 智能体 — LangGraph StateGraph
 *
 * 拓扑 (5 节点 + 条件路由):
 *
 *          ┌────────┐
 *          │  START │
 *          └────┬───┘
 *               │
 *               ▼
 *   ┌──────────────────────┐
 *   │ ① query_analyzer      │  分析查询意图 + 难度 + 领域
 *   │   (Query Analyzer)   │  → difficulty, domains
 *   └──────────┬───────────┘
 *              │
 *              ▼
 *   ┌──────────────────────┐
 *   │ ② query_rewriter      │  生成检索变体 + 子问题分解
 *   │   (Query Rewriter)   │  → rewritten queries
 *   └──────────┬───────────┘
 *              │
 *              ▼
 *   ┌──────────────────────┐
 *   │ ③ retriever           │  多KB联合检索 + 公式/代码补充
 *   │   (Multi-KB Retriever)│  → search results
 *   └──────────┬───────────┘
 *              │
 *              ▼
 *   ┌──────────────────────┐
 *   │④ grounding_generator  │  基于检索结果生成回答
 *   │   (GroundedGenerator)│  → answer + citations
 *   └──────────┬───────────┘
 *              │
 *              ▼
 *   ┌──────────────────────┐
 *   │⑤ quality_reflector    │  自我评估 + 不足 → 补检
 *   │   (QualityReflector) │  → final answer + quality score
 *   └──────────┬───────────┘
 *              │
 *     ┌────────┼────────┐
 *     │ passed │        │ needs_more
 *     ▼        │        ▼
 *   ┌──────┐  │  ┌──────────────┐
 *   │ END  │  │  │ → retriever   │  (回到 ③，最多 2 次)
 *   └──────┘  │  └──────────────┘
 *             │
 *   质量评分 ≥ 7/10  → END
 *   质量评分 < 7/10  → 回到 retriever (补充检索)
 *   重试 > 2 次     → END (带质量警告)
 */

const MLRagState = Annotation.Root({
  // 输入
  query: Annotation<string>(),
  userId: Annotation<string>(),
  selectedKbIds: Annotation<string[]>(),

  // 查询分析结果
  difficulty: Annotation<'beginner' | 'intermediate' | 'advanced' | null>(),
  domains: Annotation<string[]>(),
  rewrittenQueries: Annotation<string[]>(),

  // 检索结果
  searchResults: Annotation<MultiKBSearchResult | null>(),
  retrievalIterations: Annotation<number>(),

  // 生成结果
  answer: Annotation<string>(),
  citations: Annotation<Citation[]>(),
  formulasIncluded: Annotation<boolean>(),
  codeIncluded: Annotation<boolean>(),

  // 质量评估
  qualityScore: Annotation<number>(),       // 0-10
  qualityIssues: Annotation<string[]>(),
  needsMoreRetrieval: Annotation<boolean>(),
});

type MLRagStateType = typeof MLRagState.State;

export function compileMLRagGraph() {
  return new StateGraph(MLRagState)
    .addNode('query_analyzer', queryAnalyzerNode)
    .addNode('query_rewriter', queryRewriterNode)
    .addNode('retriever', retrieverNode)
    .addNode('grounding_generator', groundingGeneratorNode)
    .addNode('quality_reflector', qualityReflectorNode)
    .addEdge(START, 'query_analyzer')
    .addEdge('query_analyzer', 'query_rewriter')
    .addEdge('query_rewriter', 'retriever')
    .addEdge('retriever', 'grounding_generator')
    .addEdge('grounding_generator', 'quality_reflector')
    .addConditionalEdges('quality_reflector', afterReflection, {
      END: END,
      retriever: 'retriever',
    })
    .compile();
}
```

#### 4.2 Prompt 工程

```typescript
// lib/ml-rag/agent/prompt-templates.ts

/**
 * ML RAG 智能体 — Prompt 模板体系
 */

// 1. Query Analyzer Prompt
export const QUERY_ANALYZER_PROMPT = `你是一个机器学习查询分析专家。
分析用户的查询，返回 JSON:
{
  "difficulty": "beginner|intermediate|advanced",
  "domains": ["dl","nlp","cv","rl","optimization"...],
  "hasFormulaRequest": boolean,
  "hasCodeRequest": boolean,
  "expectedAnswerType": "definition|derivation|comparison|implementation|application"
}`;

// 2. Grounded Generator Prompt (核心)
export const GROUNDED_GENERATOR_SYSTEM = `你是一个机器学习专家导师。
请基于提供的知识库内容回答用户问题。

【重要规则】
1. **严格基于来源**: 你的回答必须基于下方 [知识库上下文] 中提供的信息。
   如果上下文不足以回答，请明确说明，不要编造。
2. **引用格式**: 每个关键陈述后标注来源，格式 [Source: 文档名, Chunk N]
3. **公式渲染**: 使用 LaTeX 语法 $$...$$ 或 $...$，确保公式完整
4. **代码示例**: 如有相关代码，提供可运行的 Python 代码
5. **难度适配**: 根据用户的 level，调节解释的深度
6. **知识链接**: 如果提到相关概念，建议用户进一步探索

【回答结构】
1. 核心回答 (带引用)
2. 公式推导 (如适用)
3. 代码示例 (如适用)
4. 相关概念 (建议进一步阅读)
5. 参考文献`;

// 3. Quality Reflector Prompt
export const QUALITY_REFLECTOR_PROMPT = `评估以下 ML 问答的质量。

【评估维度】(每项 0-2 分, 总分 10)
1. 事实准确性: 回答是否与知识库内容一致?
2. 引用完整性: 关键陈述是否都有引用?
3. 公式正确性: 公式是否完整且正确?
4. 代码可用性: 代码是否可运行?
5. 教学适配性: 解释难度是否适合用户?

返回 JSON:
{
  "scores": { "accuracy": 0-2, "citations": 0-2, "formulas": 0-2, "code": 0-2, "teaching": 0-2 },
  "totalScore": 0-10,
  "issues": ["具体问题1", "具体问题2"],
  "suggestions": ["改进建议1"],
  "needsMoreRetrieval": boolean
}`;

// 4. Few-shot 示例
export const FEW_SHOT_EXAMPLES = [
  {
    query: "什么是梯度下降？",
    context: "[Source: 深度学习花书, Chunk 5] ...梯度下降是一种迭代优化算法...",
    answer: `梯度下降 (Gradient Descent) 是一种用于最小化损失函数的一阶迭代优化算法 [Source: 深度学习花书, Chunk 5]。

**核心思想**: 沿着梯度的反方向（最陡下降方向）更新参数。

$$\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta J(\\theta_t)$$

其中 $\\eta$ 是学习率，$\\nabla_\\theta J(\\theta_t)$ 是损失函数关于参数的梯度 [Source: 深度学习花书, Chunk 6]。

\`\`\`python
def gradient_descent(grad_fn, init_params, lr=0.01, n_iters=100):
    params = init_params
    for _ in range(n_iters):
        grad = grad_fn(params)
        params = params - lr * grad
    return params
\`\`\`

**相关概念**: 随机梯度下降 (SGD)、动量法、Adam 优化器`,
  },
  // ... 更多示例
];
```

#### 4.3 RAG Agent 5 个节点实现

```
┌─────────────────────────────────────────────────────────────────┐
│           ML RAG Agent 5 节点功能详表                              │
│                                                                  │
│  节点                    输入→输出                  LLM调用       │
│  ─────────────────────────────────────────────────────────────  │
│  ① query_analyzer       query → difficulty,         gpt-4o-mini │
│                          domains, type                          │
│                                                                  │
│  ② query_rewriter       query → 5 variants           gpt-4o-mini │
│                          + sub-questions                         │
│                                                                  │
│  ③ retriever            variants + kb_ids →          无 LLM     │
│                          MultiKBSearchResult                     │
│                          (向量+BM25+RRF+Reranker)                │
│                                                                  │
│  ④ grounding_generator  query + context →           主模型      │
│                          answer + citations          (可配置)    │
│                                                                  │
│  ⑤ quality_reflector    query + answer +             gpt-4o-mini│
│                          context → qualityScore                  │
│                          + issues + needsMoreRetrieval           │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.4 引用溯源系统

```typescript
// lib/ml-rag/agent/citation-engine.ts

/**
 * ML 引用溯源引擎
 *
 * 功能:
 * 1. 解析 LLM 输出中的 [Source: X, Chunk N] 标记
 * 2. 验证引用是否真实存在 (防止 LLM 编造引用)
 * 3. 生成 BibTeX 格式引用 (论文)
 * 4. 生成可点击的链接
 *
 * 引用格式:
 * ┌──────────────────────────────────────────────────────────┐
 * │ 📚 参考文献                                              │
 * │                                                          │
 * │ [1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). │
 * │     Deep Learning. MIT Press. 第 5 章, Chunk 15.        │
 * │                                                          │
 * │ [2] Kingma, D. P., & Ba, J. (2014). Adam: A Method for  │
 * │     Stochastic Optimization. arXiv:1412.6980. §2.1.     │
 * │                                                          │
 * │ [3] scikit-learn 官方文档. sklearn.linear_model.         │
 * │     LogisticRegression. API Reference. Chunk 42.         │
 * └──────────────────────────────────────────────────────────┘
 */

export interface Citation {
  index: number;
  sourceType: 'textbook' | 'paper' | 'official_doc' | 'competition';
  documentTitle: string;
  chunkIndex: number;
  chunkId: string;
  authors?: string[];
  year?: number;
  bibtex?: string;
  verified: boolean;  // 是否验证该引用确实存在于 KB 中
}
```

#### 4.5 SmartLearn Capability 注册

```typescript
// lib/ml-rag/agent/index.ts

/**
 * 注册为 SmartLearn 的 GraphCapability
 * 用户可以在 chat 页面选择 "ML 问答" 能力
 */

import { GraphCapability, createCapabilityManifest } from '@/lib/deeptutor/core/capability-protocol';

export class MLRagCapability extends GraphCapability {
  readonly manifest = createCapabilityManifest({
    name: 'ml_rag',
    description: '机器学习知识库问答 — 基于精选 ML 教材/论文/文档的精准回答',
    stages: ['query_analyzer', 'query_rewriter', 'retriever', 'grounding_generator', 'quality_reflector'],
    toolsUsed: ['rag', 'web_search'],
    cliAliases: ['ml', 'ml-qa'],
    requestSchema: {
      type: 'object',
      properties: {
        query: { type: 'string' },
        kbIds: { type: 'array', items: { type: 'string' } },
      },
      required: ['query'],
    },
  });

  compileGraph() {
    return compileMLRagGraph();
  }
}
```

**注册到 bootstrap**:

```typescript
// lib/deeptutor/bootstrap.ts 新增:

// Phase 6 — ML RAG Capability
import { MLRagCapability } from '@/lib/ml-rag/agent';

// 在 bootstrap() 中:
const mlRagCapability = new MLRagCapability();
capabilityRegistry.register(mlRagCapability);
log.info('Phase 6: registered ml_rag capability');
```

#### 4.6 API 端点

```typescript
// app/api/v1/ml-rag/route.ts (新建)

/**
 * POST /api/v1/ml-rag — ML RAG 问答
 *
 * Request:
 * {
 *   "query": "什么是反向传播？推导一下公式",
 *   "kbIds": ["kb_ml_textbooks", "kb_ml_papers"],  // 可选, 不传则搜索全部 ML KB
 *   "difficulty": "auto",  // auto | beginner | intermediate | advanced
 *   "includeCode": true,
 *   "includeFormulas": true,
 *   "sessionId": "...",    // 可选, 关联对话
 * }
 *
 * Response: SSE Stream
 * events: stage_start → query_analyzed → rewritten → retrieved →
 *         generating → quality_check → done
 */

// app/api/v1/ml-rag/search/route.ts (新建)

/**
 * POST /api/v1/ml-rag/search — 仅检索 (不生成回答)
 *
 * 用于用户想先浏览检索结果再提问的场景
 */

// app/api/v1/ml-rag/stats/route.ts (新建)

/**
 * GET /api/v1/ml-rag/stats — ML RAG 使用统计
 *
 * Response:
 * {
 *   "totalQueries": 1523,
 *   "avgQualityScore": 7.8,
 *   "topQueries": [...],
 *   "perDomainStats": { "dl": 500, "nlp": 300, ... },
 *   "kbCoverage": { ... }
 * }
 */
```

### 6.3 Phase 4 交付文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `lib/ml-rag/agent/ml-rag-graph.ts` | 新建 | LangGraph 状态图 |
| `lib/ml-rag/agent/prompt-templates.ts` | 新建 | Prompt 模板体系 |
| `lib/ml-rag/agent/query-analyzer.ts` | 新建 | 查询分析节点 |
| `lib/ml-rag/agent/grounding-generator.ts` | 新建 | 回答生成节点 |
| `lib/ml-rag/agent/quality-reflector.ts` | 新建 | 质量评估节点 |
| `lib/ml-rag/agent/citation-engine.ts` | 新建 | 引用溯源引擎 |
| `lib/ml-rag/agent/index.ts` | 新建 | Capability 导出 |
| `app/api/v1/ml-rag/route.ts` | 新建 | 问答 API |
| `app/api/v1/ml-rag/search/route.ts` | 新建 | 检索 API |
| `app/api/v1/ml-rag/stats/route.ts` | 新建 | 统计 API |
| `lib/deeptutor/bootstrap.ts` | 修改 | 注册 ml_rag capability |
| `lib/ml-rag/__tests__/ml-rag-graph.test.ts` | 新建 | 图编译测试 |

---

## 七、Phase 5: 前端交互与可视化 (Day 16-18)

### 7.1 任务总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 5: 前端交互与可视化 (3 天)                                  │
│                                                                  │
│  任务                                 工时    交付文件            │
│  ─────────────────────────────────────────────────────────────  │
│  5.1 ML 问答面板 UI                    6h    ml-rag-panel.tsx    │
│  5.2 检索结果可视化                     4h    search-results     │
│  5.3 知识图谱展示                       4h    knowledge-graph    │
│  5.4 引用卡片组件                       4h    citation-card.tsx  │
│  5.5 Chat 集成 (能力选择)               3h    修改 chat page     │
│  5.6 使用统计仪表盘                     3h    ml-stats.tsx       │
│  ─────────────────────────────────────────────────────────────  │
│  合计                                 24h                        │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 前端组件设计

```
┌─────────────────────────────────────────────────────────────────┐
│              ML RAG 前端组件树                                    │
│                                                                  │
│  app/ml-qa/page.tsx                   # ML 问答专用页面          │
│  ├── MlRagPanel                       # 主面板                   │
│  │   ├── QueryInput                   # 查询输入框               │
│  │   │   ├── DifficultySelector       # 难度选择器              │
│  │   │   └── KbSelector               # KB 多选                 │
│  │   ├── SearchResultsPanel           # 检索结果面板 (右侧)     │
│  │   │   ├── SearchResultCard         # 单个检索结果卡片        │
│  │   │   │   ├── SourceBadge          # 来源标签                │
│  │   │   │   ├── FormulaBlock         # 公式渲染                │
│  │   │   │   └── CodeBlock            # 代码渲染                │
│  │   │   └── ResultDiversityChart     # 结果多样性分布          │
│  │   ├── AnswerPanel                  # 回答面板 (中间)         │
│  │   │   ├── AnswerContent            # 流式 Markdown 渲染      │
│  │   │   │   ├── InlineCitation       # 行内引用标记            │
│  │   │   │   ├── RenderedFormula      # KaTeX 渲染              │
│  │   │   │   └── HighlightedCode      # Shiki 代码高亮          │
│  │   │   ├── CitationList             # 引用列表                │
│  │   │   │   └── CitationCard         # 引用卡片                │
│  │   │   └── QualityBadge             # 质量评分徽章            │
│  │   └── KnowledgeGraphView           # 知识图谱 (底部)         │
│  │       └── ConceptNode              # 概念节点 (ECharts)      │
│  │                                                                  │
│  components/ml-rag/                    # 可复用组件               │
│  ├── ml-rag-panel.tsx                                               │
│  ├── search-result-card.tsx                                         │
│  ├── citation-card.tsx                                              │
│  ├── formula-block.tsx                                              │
│  ├── quality-badge.tsx                                              │
│  └── knowledge-graph-view.tsx                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Chat 集成 — 新增 "ML问答" 能力选项

```typescript
// app/chat/page.tsx 修改

// 新增 capability 选项:
const capabilities = [
  { id: 'chat', label: '对话', icon: MessageSquare, capabilityName: 'chat' },
  { id: 'ml_rag', label: 'ML问答', icon: BrainCircuit, capabilityName: 'ml_rag' }, // ← 新增
  { id: 'solve', label: '解题', icon: Lightbulb, capabilityName: 'deep_solve' },
  { id: 'quiz', label: '测验', icon: HelpCircle, capabilityName: 'mastery_path' },
  { id: 'research', label: '研究', icon: SearchIcon, capabilityName: 'deep_research' },
  { id: 'visualize', label: '可视化', icon: BarChart3, capabilityName: 'visualize' },
];

// 当 activeCapability === 'ml_rag' 时:
// - 显示额外的 KB 过滤器 (ML教材 / 经典论文 / 官方文档 / 竞赛方案)
// - 显示公式渲染开关
// - 显示难度选择器
```

### 7.4 Phase 5 交付文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `components/ml-rag/ml-rag-panel.tsx` | 新建 | 主面板 |
| `components/ml-rag/search-result-card.tsx` | 新建 | 检索结果卡片 |
| `components/ml-rag/citation-card.tsx` | 新建 | 引用卡片 |
| `components/ml-rag/formula-block.tsx` | 新建 | 公式渲染块 |
| `components/ml-rag/quality-badge.tsx` | 新建 | 质量评分徽章 |
| `components/ml-rag/knowledge-graph-view.tsx` | 新建 | 知识图谱视图 |
| `app/ml-qa/page.tsx` | 新建 | ML 问答页面 |
| `app/chat/page.tsx` | 修改 | 新增 ml_rag 能力选项 |
| `lib/store/ml-rag-store.ts` | 新建 | ML RAG 状态管理 |

---

## 八、Phase 6: 测试、部署与迭代 (Day 19-21)

### 8.1 任务总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 6: 测试、部署与迭代 (3 天)                                  │
│                                                                  │
│  任务                                 工时    交付文件            │
│  ─────────────────────────────────────────────────────────────  │
│  6.1 检索质量基准测试                  5h    retrieval-benchmark │
│  6.2 回答质量人工评估                   4h    quality-eval.md    │
│  6.3 端到端测试用例编写                 4h    e2e/ml-rag.spec.ts │
│  6.4 性能压测与优化                     4h    perf-report.md     │
│  6.5 用户反馈系统                       3h    feedback-api       │
│  6.6 部署文档更新                       2h    更新部署说明书      │
│  6.7 最终版本发布                       2h    git tag v1.1.0    │
│  ─────────────────────────────────────────────────────────────  │
│  合计                                 24h                        │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 检索质量基准测试

```typescript
// lib/ml-rag/__tests__/retrieval-benchmark.ts

/**
 * 检索质量基准测试
 *
 * 使用 50 个预定义 ML 问题 + 期望的正确答案来源
 * 评估指标:
 *   - Recall@K: top-K 结果中包含正确答案的比例
 *   - MRR (Mean Reciprocal Rank): 第一个正确答案的平均排名倒数
 *   - NDCG@K: 归一化折损累计增益
 *
 * 测试查询示例:
 * ┌──────────────────────────────────────────────────────────┐
 * │ #  查询                      期望来源              领域   │
 * │ ──────────────────────────────────────────────────────── │
 * │ 1  反向传播的数学推导         花书 Ch6.5            DL    │
 * │ 2  Adam 优化器的参数          Adam 论文 §2.1       优化  │
 * │ 3  BERT 的预训练任务          BERT 论文 §3.1       NLP   │
 * │ 4  随机森林的 Gini 不纯度    sklearn 文档          ML    │
 * │ 5  ResNet 残差连接原理       ResNet 论文 §3.2      CV    │
 * │ ...  ...                     ...                   ...   │
 * │ 50 Transformer 多头注意力    Attention 论文 §3.2   NLP   │
 * └──────────────────────────────────────────────────────────┘
 *
 * 目标:
 *   Recall@10 ≥ 0.85
 *   MRR ≥ 0.70
 *   NDCG@10 ≥ 0.80
 */
```

### 8.3 端到端测试

```typescript
// e2e/ml-rag.spec.ts (新建)

test.describe('ML RAG E2E', () => {
  test('ML QA page loads with KB selector', async ({ page }) => {
    await page.goto('/ml-qa');
    await expect(page.locator('[data-testid="kb-selector"]')).toBeVisible();
  });

  test('query returns cited answer', async ({ page }) => {
    await page.goto('/ml-qa');
    await page.fill('[data-testid="query-input"]', '什么是梯度下降？');
    await page.click('[data-testid="submit-btn"]');

    // 等待回答生成完成
    await expect(page.locator('[data-testid="answer-content"]')).toBeVisible({ timeout: 30000 });

    // 验证包含引用
    const answerText = await page.locator('[data-testid="answer-content"]').textContent();
    expect(answerText).toContain('[Source:');

    // 验证质量评分
    await expect(page.locator('[data-testid="quality-badge"]')).toBeVisible();
  });

  test('formula is rendered correctly', async ({ page }) => {
    await page.goto('/ml-qa');
    await page.fill('[data-testid="query-input"]', '交叉熵损失函数的公式');
    await page.click('[data-testid="submit-btn"]');

    // 等待 KaTeX 渲染
    await expect(page.locator('.katex')).toBeVisible({ timeout: 30000 });
  });

  test('citation verification works', async ({ page }) => {
    // 验证引用不会指向不存在的 chunk
    // ...
  });
});
```

### 8.4 用户反馈闭环

```
┌─────────────────────────────────────────────────────────────────┐
│              用户反馈闭环                                          │
│                                                                  │
│  用户收到回答                                                     │
│      │                                                           │
│      ├── 👍 有帮助 → MlRagQueryLog.userFeedback = "helpful"      │
│      │              → 增加该检索结果的权重                         │
│      │                                                           │
│      ├── 👎 无帮助 → MlRagQueryLog.userFeedback = "not_helpful"  │
│      │              → 记录具体原因                                │
│      │              → 触发检索策略复查                            │
│      │                                                           │
│      └── 🔄 部分有用 → "partially"                              │
│                      → 用户标注正确/错误的具体部分                 │
│                                                                  │
│  每周自动分析反馈 → 生成改进报告 → 更新检索策略                     │
│                                                                  │
│  反馈 API: POST /api/v1/ml-rag/feedback                         │
│  {                                                               │
│    "queryLogId": "...",                                          │
│    "feedback": "helpful" | "not_helpful" | "partially",          │
│    "comment": "...",  // 可选                                    │
│    "wrongParts": ["..."]  // 用户标注的错误内容                   │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 8.5 Phase 6 交付文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `lib/ml-rag/__tests__/retrieval-benchmark.ts` | 新建 | 检索基准测试 |
| `docs/ml-rag-quality-eval.md` | 新建 | 质量评估报告 |
| `e2e/ml-rag.spec.ts` | 新建 | E2E 测试 |
| `docs/ml-rag-perf-report.md` | 新建 | 性能测试报告 |
| `app/api/v1/ml-rag/feedback/route.ts` | 新建 | 用户反馈 API |
| `部署说明书.md` | 更新 | 添加 ML RAG 部署说明 |

---

## 九、交付清单汇总

### 9.1 代码文件清单

```
┌─────────────────────────────────────────────────────────────────┐
│              代码交付清单 (42 个文件)                              │
│                                                                  │
│  新建文件 (38 个):                                                │
│  ─────────────────────────────────────────────────────────────  │
│  lib/ml-rag/                                 (14 个)            │
│  ├── index.ts, config.ts, batch-indexer.ts                       │
│  ├── collector/ (4 个)                                           │
│  ├── chunker/   (3 个)                                           │
│  ├── retrieval/ (5 个) + eval-retrieval.ts                      │
│  └── agent/     (6 个)                                       │
│                                                                  │
│  app/api/v1/ml-rag/                          (4 个)            │
│  ├── route.ts, search/route.ts                                   │
│  ├── stats/route.ts, feedback/route.ts                           │
│                                                                  │
│  app/ml-qa/                                  (1 个)             │
│  └── page.tsx                                                    │
│                                                                  │
│  components/ml-rag/                           (6 个)             │
│  lib/store/ml-rag-store.ts                   (1 个)             │
│  lib/ml-rag/__tests__/                       (5 个)             │
│  e2e/ml-rag.spec.ts                          (1 个)             │
│                                                                  │
│  修改文件 (4 个):                                                 │
│  ─────────────────────────────────────────────────────────────  │
│  prisma/schema.prisma             新增 3 个 ML 模型              │
│  lib/deeptutor/bootstrap.ts       注册 ml_rag capability         │
│  app/chat/page.tsx                新增 ML问答 能力选项           │
│  .env.example                     新增 ML 环境变量               │
│                                                                  │
│  数据文件:                                                        │
│  ─────────────────────────────────────────────────────────────  │
│  data/ml-kb/textbooks/            教材原始文件                    │
│  data/ml-kb/papers/               论文 PDF                       │
│  data/ml-kb/docs/                 官方文档快照                    │
│  data/ml-kb/competitions/         竞赛方案                        │
│                                                                  │
│  数据库变更:                                                      │
│  ─────────────────────────────────────────────────────────────  │
│  Prisma Migration: add_ml_rag_models                             │
│  新增 3 张表: MlKnowledgeBase, MlDocumentMeta, MlRagQueryLog    │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 文档交付清单

| 文件 | 说明 |
|------|------|
| `ml_rag开发计划书.md` | 本文档 |
| `docs/ml-rag-quality-eval.md` | 质量评估报告 (Phase 6) |
| `docs/ml-rag-perf-report.md` | 性能测试报告 (Phase 6) |
| `部署说明书.md` (更新) | 添加 ML RAG 部署章节 |
| API 文档 | 自动生成 (Swagger/OpenAPI 可选) |

---

## 十、里程碑与甘特图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     ML RAG 开发甘特图 (21 天)                               │
│                                                                          │
│  Week 1        Week 2        Week 3                                      │
│  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │13 │14 │15 │16 │17 │18 │19 │20 │21 │
│  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│                                                                          │
│  Phase 1: 基础设施                                                        │
│  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Day 1-2                 │
│                                                                          │
│  Phase 2: 数据采集与 KB 构建                                               │
│  ░░░░░░░░████████████████████████████████░░░░░░░  Day 3-7                 │
│           │ Milestone 1: 4 个 KB 构建完成 │                               │
│                                                                          │
│  Phase 3: ML 专属检索增强                                                  │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██████████████░░  Day 8-10                │
│                                │ Milestone 2: 检索质量达标│                │
│                                                                          │
│  Phase 4: RAG 智能体开发                                                   │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████  Day 11-15       │
│                                         │ Milestone 3: Agent 可对话 │      │
│                                                                          │
│  Phase 5: 前端交互                                                        │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████  Day 16-18│
│                                                     │ Milestone 4: UI 可用│
│                                                                          │
│  Phase 6: 测试部署迭代                                                     │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████  │
│                                                     Day 19-21            │
│                                                     │ Milestone 5: v1.1.0│
│                                                                          │
│  关键里程碑:                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ M1 (Day 7):  4 个 ML 知识库构建完成, 11,300+ chunks 索引完毕       │ │
│  │ M2 (Day 10): 检索质量达标 (Recall@10 ≥ 0.85, MRR ≥ 0.70)          │ │
│  │ M3 (Day 15): RAG Agent 能基于 KB 生成带引用的专业回答               │ │
│  │ M4 (Day 18): 前端交互完整可用, 用户体验流畅                         │ │
│  │ M5 (Day 21): 全部测试通过, 部署上线, 版本发布 v1.1.0               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### 资源需求汇总

| 资源 | 需求 | 说明 |
|------|------|------|
| 开发人力 | 1 名全栈工程师 | 21 天 (可并行缩减) |
| OpenAI API Key | 是 | embedding + LLM 调用 |
| SiliconFlow API Key | 推荐 | 低成本中文 embedding |
| PostgreSQL + pgvector | 是 | 复用现有 |
| 磁盘空间 | ~500 MB | 语料 PDF + 向量数据库 |
| 网络带宽 | 公网访问 | arXiv / 文档镜像下载 |

---

> **开发计划书版本**: v1.0
> **目标交付日期**: 21 个工作日
> **技术基础**: SmartLearn v2.0 通用 RAG 基础设施
