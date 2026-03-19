# Bangumi Recommender

基于 [bangumi15M](https://github.com/lyqcom/Bangumi15M) 数据集构建的番组计划（bgm.tv）个性化推荐系统，采用多阶段混合架构：协同过滤（Multi-VAE）+ 内容嵌入（LLM Embedding）+ 混合融合排名。

![demo](docs/demo.png)

## 特性

- **Multi-VAE 协同过滤**：在 1520 万条交互记录上训练，NDCG@20 = **0.3237**，Recall@20 = **0.3590**
- **LLM 语义嵌入**：使用 Qwen3-Embedding-8B（4096 维）对 43 万条条目生成语义向量
- **FAISS 近邻检索**：PCA 降至 512 维，5 种内容类型独立索引，毫秒级响应
- **MMR 多样性排名**：防止推荐结果同质化
- **冷启动处理**：收藏数 < 10 时自动提升内容权重
- **实时用户数据**：通过 Bangumi API v0 实时拉取用户收藏，SQLite 缓存避免重复请求
- **全类型支持**：动画、书籍、音乐、游戏、三次元

---

## 技术架构

```
用户输入 (bgm.tv 用户名)
        │
        ▼
┌─────────────────┐
│  Bangumi API    │  实时拉取用户收藏（带 SQLite 缓存）
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌──────────────┐
│Multi- │  │ 内容嵌入     │
│VAE CF │  │ (FAISS 检索) │
│       │  │ PCA 512-dim  │
└───┬───┘  └──────┬───────┘
    │              │
    └──────┬───────┘
           ▼
    ┌─────────────┐
    │ 混合融合    │  α×CF + (1-α)×内容
    │ MMR 多样性  │
    │ 热度微调    │
    └─────┬───────┘
           ▼
    推荐结果 + 中文理由
```

### 技术栈

| 层 | 技术 |
|---|---|
| 协同过滤 | PyTorch Multi-VAE（β-VAE 退火，Dropout=0.5） |
| 语义嵌入 | Nebius API / Qwen3-Embedding-8B (4096-dim) |
| 向量检索 | FAISS (IndexFlatIP / IndexIVFFlat + PCA) |
| 后端 | FastAPI + uvicorn，异步 SQLite 缓存 |
| 前端 | Vue 3 + Vite + TailwindCSS |
| 部署 | Docker Compose |

---

## 数据集

| 数据集 | 来源 | 描述 |
|---|---|---|
| bangumi15M | [lyqcom/Bangumi15M](https://github.com/lyqcom/Bangumi15M) | 1520 万条匿名交互，43.4 万条目元数据 |
| Bangumi Archive | [bangumi/archive](https://github.com/bangumi/archive) | 每周全量 Wiki 导出（含简介、标签） |

---

## 目录结构

```
.
├── scripts/
│   ├── 01_prepare_data.py        # 数据清洗，构建稀疏交互矩阵
│   ├── 02_train_cf_model.py      # Multi-VAE 训练（支持 CUDA）
│   ├── 03_generate_embeddings.py # 批量生成 LLM 嵌入（流式写盘，断点续跑）
│   ├── 04_build_faiss_index.py   # PCA 降维 + 构建 FAISS 索引
│   └── 05_download_archive.py    # 解析 Bangumi Archive ZIP
├── backend/
│   ├── main.py                   # FastAPI 入口，lifespan 管理单例
│   ├── config.py                 # 全局配置常量
│   ├── deps.py                   # 服务单例初始化
│   ├── routers/
│   │   └── recommend.py          # /api/recommend, /api/user/{name}/profile, /api/health
│   ├── services/
│   │   ├── bgm_api.py            # Bangumi API 客户端（限速 + 缓存）
│   │   ├── cf_recommender.py     # CF 推理（Multi-VAE 编码器→解码器）
│   │   ├── content_recommender.py # 内容推荐（用户画像 + FAISS 检索）
│   │   ├── hybrid_ranker.py      # 混合融合 + MMR + 推荐理由生成
│   │   └── user_encoder.py       # 用户收藏解析与统计
│   └── models/
│       └── schemas.py            # Pydantic 响应模型
├── frontend/
│   ├── src/App.vue               # Vue 3 主界面（搜索、用户画像、推荐卡片）
│   └── vite.config.js            # Vite 配置（API 代理 → localhost:8000）
├── Dockerfile.backend
├── frontend/Dockerfile
├── docker-compose.yml
└── requirements.txt
```

`data/` 目录（不入 git，需自行生成）：

```
data/
├── processed/
│   ├── interaction_matrix.npz    # 稀疏用户-条目交互矩阵
│   ├── subjects_meta.parquet     # 条目元数据
│   ├── archive_subjects.parquet  # Archive 简介数据
│   └── {user,item}_id_map.json   # ID 映射表
├── embeddings/
│   ├── subject_embeddings.npy    # 43.4 万条目嵌入（6.8 GB）
│   ├── subject_ids.json          # 嵌入行序 → subject_id
│   ├── pca_matrix.npy            # PCA 投影矩阵（512×4096）
│   ├── pca_mean.npy              # PCA 中心向量
│   └── faiss_index_type{N}.bin   # 各类型 FAISS 索引
└── models/
    ├── multivae_best.pt          # 最优 Multi-VAE 权重
    └── multivae_config.json      # 模型超参
```

---

## 快速开始

### 前置要求

- Python 3.10+
- Node.js 18+
- 16 GB+ RAM（嵌入文件 6.8 GB，内存映射加载）
- CUDA GPU（推荐，仅训练阶段需要；CPU 也可但很慢）
- [Nebius API Key](https://studio.nebius.ai/)（生成嵌入用，或替换为其他 OpenAI 兼容 API）

### 1. 克隆与安装

```bash
git clone https://github.com/<your-username>/Bangumi-Recommender.git
cd Bangumi-Recommender
pip install -r requirements.txt
```

### 2. 准备数据集

**bangumi15M 数据集**（放到 `bangumi15M_data/raw_data/`）：

从 [bangumi15M releases](https://github.com/lyqcom/Bangumi15M/releases) 下载，解压后应有：
- `AnonymousUserCollection.csv`
- `Subjects.csv`

**Bangumi Archive**（可选，提供简介数据）：

从 [bangumi/archive releases](https://github.com/bangumi/archive/releases) 下载最新 `dump-*.zip`，放到项目根目录。

### 3. 运行数据处理流水线

```bash
# Step 1: 数据清洗，构建稀疏交互矩阵
python scripts/01_prepare_data.py

# Step 2: 解析 Archive 简介（如有下载）
# 将 dump-*.zip 解压到 data/archive/subject.jsonlines，然后：
python scripts/05_download_archive.py

# Step 3: 训练 Multi-VAE（建议 GPU）
# GPU: RTX 3070 约 15 分钟；CPU 约 20 小时
python scripts/02_train_cf_model.py

# Step 4: 生成 LLM 嵌入（需要 Nebius API Key）
# 编辑 scripts/03_generate_embeddings.py，填入 API_KEY
python scripts/03_generate_embeddings.py
# 支持断点续跑：中断后重新运行自动从上次位置继续

# Step 5: 构建 FAISS 索引（自动 PCA 降维）
python scripts/04_build_faiss_index.py
```

### 4. 启动服务

**方式 A：直接运行**

```bash
# 后端
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 前端（另一个终端）
cd frontend
npm install
npm run dev       # 开发模式（带 HMR）
# 或 npm run build && npm run preview  # 生产构建
```

访问 `http://localhost:5173`（开发）或 `http://localhost:4173`（预览）。

**方式 B：Docker Compose**

```bash
docker compose up --build
```

访问 `http://localhost:3000`。

### 5. API 使用

```bash
# 获取推荐（动画，Top 20）
GET /api/recommend?username=sai&subject_type=2&top_n=20

# 获取用户画像统计
GET /api/user/sai/profile

# 健康检查
GET /api/health
```

**请求参数**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `username` | string | 必填 | bgm.tv 用户名 |
| `subject_type` | int | `2` | 1=书籍, 2=动画, 3=音乐, 4=游戏, 6=三次元 |
| `top_n` / `limit` | int | `20` | 返回数量 |
| `nsfw` | bool | `false` | 是否包含 NSFW 内容 |

**响应示例**

```json
{
  "username": "sai",
  "subject_type": 2,
  "recommendations": [
    {
      "subject_id": 5118,
      "name_cn": "阿基拉",
      "name": "AKIRA",
      "score": 0.985,
      "reason": "与你口味相似的用户也喜欢这部作品；在 大友克洋、剧场版、科幻 等标签上与你的收藏相似",
      "bangumi_score": 7.9,
      "tags": ["大友克洋", "剧场版", "科幻", "1988"],
      "image_url": "https://api.bgm.tv/v0/subjects/5118/image?type=medium",
      "subject_type": 2
    }
  ]
}
```

---

## 模型细节

### Multi-VAE 协同过滤

| 项目 | 值 |
|---|---|
| 训练数据 | 167,790 用户 × 22,027 动画，1520 万条交互 |
| 架构 | 22027 → 600 → **200**（μ, logσ²）→ 200 → 600 → 22027 |
| 训练配置 | Adam lr=1e-3, batch=512, 50 epochs, Dropout=0.5 |
| β-VAE 退火 | β: 0→0.2，前 80% epoch 线性增长 |
| 交互权重 | 收藏=1.0, 在看=0.8, 想看=0.6, 搁置=0.3, 抛弃=0.1 |
| 最优指标 | NDCG@20 = **0.3237**, Recall@20 = **0.3590** |

### 内容嵌入

| 项目 | 值 |
|---|---|
| 嵌入模型 | Qwen/Qwen3-Embedding-8B（via Nebius API） |
| 原始维度 | 4096 |
| FAISS 维度 | 512（PCA，100K 样本拟合） |
| 条目数量 | 434,269 |
| 文本构建 | 类型、平台、时间、标题、标签、评分、简介 |

### 混合排名

```
final_score = α × CF_score + (1 - α) × Content_score + ε × log(收藏数)

α = 0.7  (普通用户)
α = 0.4  (冷启动：动画收藏 < 10 部)
ε = 0.05 (热度微调权重)
```

MMR 多样性：λ = 0.3，在相关性与结果多样性之间平衡。

---

## 配置

编辑 `backend/config.py` 调整推荐行为：

```python
CF_WEIGHT_DEFAULT = 0.7       # CF 权重 α
CF_WEIGHT_COLD_START = 0.4    # 冷启动 α
COLD_START_THRESHOLD = 10     # 冷启动阈值（动画收藏数）
DIVERSITY_LAMBDA = 0.3        # MMR 多样性参数
POPULARITY_WEIGHT = 0.05      # 热度微调权重
CACHE_TTL_USER_COLLECTION = 3600  # 用户收藏缓存 TTL（秒）
```

---

## 致谢

- [bangumi15M](https://github.com/lyqcom/Bangumi15M) — 训练数据集
- [Bangumi Archive](https://github.com/bangumi/archive) — 条目 Wiki 数据
- [bgm.tv API](https://bgm.tv/wiki/api) — 实时用户数据
- [Nebius AI Studio](https://studio.nebius.ai/) — Qwen3 嵌入 API

---

## License

MIT
