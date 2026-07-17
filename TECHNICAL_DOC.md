# 交通时空知识图谱 — 技术详解文档

> **Traffic Spatio-Temporal Knowledge Graph**  
> DCRNN 交通预测 × Neo4j 知识图谱 × DeepSeek LLM 智能查询

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [DCRNN 模型详解](#3-dcrnn-模型详解)
4. [知识图谱构建](#4-知识图谱构建)
5. [LLM 智能查询系统](#5-llm-智能查询系统)
6. [Web 应用](#6-web-应用)
7. [实验与结果](#7-实验与结果)
8. [代码结构](#8-代码结构)
9. [部署方案](#9-部署方案)
10. [未来方向](#10-未来方向)

---

## 1. 项目概述

### 1.1 背景

交通预测是智能交通系统（ITS）的核心问题。给定道路上传感器采集的历史速度数据，预测未来 15-60 分钟的交通状态。传统方法仅考虑道路网络的物理拓扑，忽略了传感器的语义信息和空间知识。

本项目的核心创新在于：**将交通传感器建模为知识图谱实体，引入地理语义和 LLM 智能分析，增强传统深度学习模型的预测与解释能力。**

### 1.2 数据

| 数据集 | 说明 |
|--------|------|
| **METR-LA** | 洛杉矶高速公路 207 个环形探测器，2012 年 3-6 月，34,272 个时间步（5 分钟间隔），速度范围 0-70 mph |
| **路网邻接矩阵** | 基于传感器空间距离阈值构建的有向图（高斯核加权） |
| **传感器坐标** | 207 个传感器的 GPS 经纬度（原始 ID 来自 Caltrans PeMS 系统） |

### 1.3 技术栈

| 层次 | 技术 |
|------|------|
| **深度学习** | PyTorch 2.5, DCRNN (ICLR 2018), NVIDIA RTX 4070 (8GB) |
| **图数据库** | Neo4j AuraDB (云端), Cypher 查询语言 |
| **LLM** | DeepSeek-chat API, Text-to-Cypher 智能查询 |
| **Web** | Flask 3.x, 纯 HTML/CSS/JS 聊天界面 |
| **GIS** | osmnx (OpenStreetMap), Overpass API |
| **部署** | GitHub + bore 隧道公网访问 |

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户界面层                               │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ chat.html   │  │ query_kg.py │  │ 终端交互查询        │  │
│  │ (Web聊天)   │  │ (命令行CLI)  │  │                    │  │
│  └──────┬──────┘  └──────┬───────┘  └─────────┬──────────┘  │
│         │                │                    │              │
├─────────┼────────────────┼────────────────────┼──────────────┤
│         │          智能分析层                    │              │
│  ┌──────┴────────────────┴────────────────────┴──────────┐   │
│  │                server.py (Flask API)                   │   │
│  │  POST /api/query  →  LLM Cypher生成  →  执行  →  回答   │   │
│  └────────────────────────┬──────────────────────────────┘   │
│                           │                                   │
├───────────────────────────┼───────────────────────────────────┤
│                      LLM 推理层                                │
│  ┌────────────────────────┴──────────────────────────────┐   │
│  │              DeepSeek-chat (API)                       │   │
│  │  中文提问 → Schema注入 → Cypher生成 → 结果格式化        │   │
│  └────────────────────────┬──────────────────────────────┘   │
│                           │                                   │
├───────────────────────────┼───────────────────────────────────┤
│                      数据存储层                                │
│  ┌────────────────────────┴──────────────────────────────┐   │
│  │              Neo4j AuraDB (云端图数据库)                │   │
│  │  466个节点, 2151条边, 7种关系类型                        │   │
│  └───────────────────────────────────────────────────────┘   │
│                           │                                   │
├───────────────────────────┼───────────────────────────────────┤
│                     数据处理层                                 │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐   │
│  │DCRNN模型 │  │osmnx富化 │  │趋势分析    │  │POI匹配   │   │
│  │(预测)    │  │(道路/POI)│  │(聚类/统计) │  │          │   │
│  └──────────┘  └──────────┘  └───────────┘  └──────────┘   │
│                           │                                   │
├───────────────────────────┼───────────────────────────────────┤
│                     原始数据层                                 │
│  ┌────────────────────────┴──────────────────────────────┐   │
│  │    METR-LA HDF5 (34,272×207) + 传感器坐标 + 路网图    │   │
│  └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. DCRNN 模型详解

### 3.1 扩散卷积 (Diffusion Convolution)

扩散卷积是 DCRNN 的核心创新。其将交通流建模为图上**有向扩散过程**：

```
扩散过程:  X_t = (1-α)X_{t-1} + α A X_{t-1}
```

其中 A 是转移矩阵（出度随机游走：A = D_out^(-1) * W，入度随机游走：A = D_in^(-1) * W^T）。

**扩散卷积层**使用**切比雪夫多项式**截断展开，避免昂贵的特征分解。第 k 阶扩散：

```
T_0(A) = I
T_1(A) = A
T_k(A) = 2A·T_{k-1}(A) - T_{k-2}(A)
```

### 3.2 DCGRU 单元

DCGRU 将标准 GRU 的门控机制中的**矩阵乘法替换为扩散卷积**，使得门控能够感知图上的空间依赖。

```
标准 GRU:           DCGRU:
r = σ(W_r[x, h])    r = σ(Θ_r ★_G [x, h])    # 重置门，扩散卷积替代矩阵乘
u = σ(W_u[x, h])    u = σ(Θ_u ★_G [x, h])    # 更新门
c = tanh(W_c[x, r⊙h])  c = tanh(Θ_c ★_G [x, r⊙h])  # 候选状态
h = u⊙h + (1-u)⊙c   h = u⊙h + (1-u)⊙c        # 新状态
```

其中 ★_G 表示在多张图上的**批量扩散卷积**，支持同时使用多张支撑图（多张图的卷积核堆叠后一次矩阵乘法完成）。

**核心实现**（`dcrnn_cell.py:60-78`）：
```python
def _gconv(self, inputs, state, weights, bias, output_size):
    x = torch.cat([inputs, state], dim=2)           # (B, N, C)
    # 展开为 M 张扩散图上的批量矩阵乘
    x_diffused = torch.bmm(self._diff_kernel, x0)   # (M, N, C*B)
    # 线性变换: 图卷积核参数化
    x = x @ weights + bias
    return x.view(B, N * output_size)
```

### 3.3 Seq2Seq 架构

**编码器**：2 层 DCGRU，每层 64 隐藏单元，处理历史 12 个时间步（60 分钟）的传感器速度数据。

**解码器**：2 层 DCGRU，最后一层带投影降维。自回归预测未来 12 步（60 分钟）。训练时使用**计划采样**（Scheduled Sampling）：

```
P(使用真实值) = 1 - τ/(τ + exp(step/τ)),  τ = 2000
```

随着训练推进，模型逐步从依赖真实值过渡到依赖自身预测，缓解训练-推理分布偏移。

### 3.4 扩散支撑图

| 支撑图 | 来源 | 物理意义 |
|--------|------|----------|
| 图 1 | 路网邻接矩阵 → 出度随机游走 D_out^(-1) W | 交通流出扩散 |
| 图 2 | 路网邻接矩阵 → 入度随机游走 D_in^(-1) W^T | 交通流入扩散 |
| 图 3 (KG增强) | 知识图谱语义邻接 A_ij = exp(-d_ij²/2σ²) | 地理邻近性先验 |

### 3.5 模型参数

| 参数 | 值 |
|------|-----|
| 输入维度 | 2（速度 + 时间特征） |
| 输出维度 | 1（预测速度） |
| 编码器层数 | 2 |
| 隐藏单元 | 64 |
| 节点数 | 207 |
| 最大扩散阶数 | 2 |
| 滤波器类型 | dual_random_walk |
| **总参数量** | **372,352** |

---

## 4. 知识图谱构建

### 4.1 Neo4j 数据模型

```
          ┌──────────────────┐
          │   Sensor (207)   │
          │  sid, lat, lon   │
          │  road_type       │  ← osmnx OSM 道路类型
          │  cluster         │  ← K-Means 聚类标签
          │  avg_speed       │  ← 历史统计
          │  congestion_ratio│
          │  morning_avg     │
          │  evening_avg     │
          │  peak_drop       │
          │  weekday_avg     │
          │  weekend_avg     │
          │  worst_hour      │
          │  mae_15/30/60min │  ← DCRNN 预测误差
          │  description     │  ← LLM 生成描述
          └──┬───────┬──────┘
             │       │
    ROAD_DISTANCE  NEAR
      {km: 0.5}   {distance_m: 300}
             │       │
     ┌───────┘       └──────────┐
     ▼                          ▼
┌──────────┐              ┌──────────┐
│  Sensor  │              │ POI(259) │
│(邻居传感器)│              │ name,type│
└──────────┘              └──────────┘

附加关系:
  SIMILAR_TO {cluster}  — 同一交通模式聚类内的传感器
```

### 4.2 构建流程

```
STEP 1: build_kg.py
  ┌─────────────────────────────────────┐
  │ 读取传感器坐标 CSV (207 条)          │
  │ → 计算地理距离矩阵 (207×207)          │
  │ → 每传感器连接最近 5 个邻居            │
  │ → 写入 Neo4j: CREATE Sensor nodes +  │
  │              ROAD_DISTANCE edges     │
  │ → 高斯核语义邻接矩阵 → kg_adj.pkl    │
  └─────────────────────────────────────┘

STEP 2: enrich_kg.py (osmnx)
  ┌─────────────────────────────────────┐
  │ graph_from_point(15km) 下载LA路网   │
  │ → 每传感器匹配最近道路边              │
  │ → 标注 road_type (motorway/primary..)│
  │ → 写入 Neo4j: SET s.road_type       │
  └─────────────────────────────────────┘

STEP 3: fill_pois.py (osmnx)
  ┌─────────────────────────────────────┐
  │ 12 个子区域 × 4km POI 下载          │
  │ → 8,144 个去重 POI                  │
  │ → 每传感器匹配 500m 内 POI (最多5个)  │
  │ → 写入 Neo4j: MERGE POI, NEAR edge  │
  └─────────────────────────────────────┘

STEP 4: write_trends.py
  ┌─────────────────────────────────────┐
  │ 读取 METR-LA 原始 HDF5 (34,272×207) │
  │ → 按小时/早晚高峰/工作日分组统计       │
  │ → 计算 avg_speed, congestion_ratio, │
  │        peak_drop, worst_hour 等      │
  │ → 写入 Neo4j: SET 趋势属性          │
  └─────────────────────────────────────┘

STEP 5: cluster_sensors.py
  ┌─────────────────────────────────────┐
  │ 读取 Neo4j 趋势特征 → 标准化 → K-Means│
  │ → 3 类: 全天通畅(79) / 高峰敏感(117)│
  │          / 高频拥堵(11)              │
  │ → 类内 top-3 最近邻 → SIMILAR_TO 边 │
  └─────────────────────────────────────┘

STEP 6: llm_kg.py
  ┌─────────────────────────────────────┐
  │ DeepSeek 批量生成传感器中文描述       │
  │ → 207 条描述写入 Neo4j              │
  │ → 自动 Schema 提取 + Text-to-Cypher │
  └─────────────────────────────────────┘
```

### 4.3 最终知识图谱规模

| 统计项 | 数值 |
|--------|------|
| 节点总数 | **466** |
| Sensor 节点 | 207 |
| POI 节点 | 259 |
| 边总数 | **2,151** |
| ROAD_DISTANCE | 1,035 |
| NEAR (传感器→POI) | 495 |
| SIMILAR_TO (聚类) | 621 |
| Sensor 属性数 | **24 个/传感器** |

---

## 5. LLM 智能查询系统

### 5.1 Text-to-Cypher 流水线

```
用户输入中文问题
      │
      ▼
┌──────────────┐
│  Schema 注入  │  Neo4j 数据模型 + Few-shot 示例
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ DeepSeek-chat│  生成 Cypher 查询语句
│ (0.2 temp)   │  model: deepseek-chat
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Neo4j 执行   │  执行 Cypher，返回结果
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ DeepSeek-chat│  结果 → 自然语言回答
│ (0.2 temp)   │
└──────┬───────┘
       │
       ▼
      中文回答
```

### 5.2 Schema 注入设计

LLM 的 prompt 包含完整的 Neo4j 数据模型描述，确保生成正确的 Cypher：

```
Neo4j 交通传感器知识图谱 (LA高速公路, 207个传感器, 2012年3-6月):

节点属性 (Sensor):
  基础: sid, lat, lon, road_type, description, cluster
  预测: mae_15min, mae_30min, mae_60min
  趋势: avg_speed, congestion_ratio, morning_avg, evening_avg,
        offpeak_avg, peak_drop, weekday_avg, weekend_avg,
        worst_hour, worst_speed

POI (259个): {name, type}

关系:
  ROAD_DISTANCE {km}  — 传感器间道路距离
  NEAR {distance_m}   — 传感器→周边设施
  SIMILAR_TO {cluster} — 同聚类传感器

Cypher 示例:
  Q: congestion_ratio 最高的3个传感器？
  → MATCH (s:Sensor) RETURN s.sid, s.congestion_ratio
     ORDER BY s.congestion_ratio DESC LIMIT 3
```

### 5.3 LLM 传感器描述生成

每 5 个传感器为一组（节省 token），DeepSeek 根据道路类型、预测误差、周边 POI 生成 30 字以内的中文描述：

> "位于I-10高速，晚高峰拥堵严重，60分钟预测误差达8.2，周边有加油站和快餐店"

### 5.4 可查询维度

| 维度 | 示例问题 |
|------|----------|
| **道路类型** | "motorway 上有多少传感器？" "哪种道路预测误差最小？" |
| **预测误差** | "mae_60min > 8 的传感器有哪些？" "预测最准的前 5 个？" |
| **交通趋势** | "早晚高峰降幅最大的传感器？" "周末比工作日快多少？" |
| **聚类分析** | "高频拥堵型传感器有多少？" "全天通畅型分布？" |
| **POI** | "周边餐厅最多的传感器？" "靠近学校的传感器拥堵如何？" |
| **相似度** | "和 773869 模式最相似的传感器？" |
| **交叉查询** | "高速上预测误差最大 + 拥堵最高的传感器？" |

---

## 6. Web 应用

### 6.1 架构

```
chat.html (纯前端)           server.py (Flask 后端)
┌──────────────┐     POST     ┌─────────────────────┐
│ 聊天界面       │ ──────────→ │ /api/query           │
│ 消息气泡       │ ←────────── │ {question: "..."}    │
│ Cypher 展示    │   JSON     │ → LLM → Cypher → 结果 │
│ 加载动画       │            └─────────────────────┘
└──────────────┘
```

### 6.2 前端设计

- **深色主题**：专业科技感配色（`#1a1a2e` 底色，`#e94560` 强调色）
- **聊天气泡**：用户红色右对齐，系统深蓝左对齐
- **Cypher 展示**：每条回答可展开查看生成的 Cypher
- **加载动画**：三点跳动动画
- **响应式**：纯 HTML/CSS/JS，无框架依赖

### 6.3 启动方式

```bash
# 本地
python server.py
# 访问 http://localhost:5000

# 公网 (bore 隧道)
bore local 5000 --to bore.pub
# 获得 http://bore.pub:PORT 公网地址
```

---

## 7. 实验与结果

### 7.1 基线 DCRNN 训练结果

| 指标 | 15 min | 30 min | 45 min | 60 min |
|------|--------|--------|--------|--------|
| **MAE (mph)** | 2.42 | 2.97 | 3.39 | 5.61 |
| **MAPE (%)** | 5.61 | 6.78 | 7.66 | 11.88 |
| **RMSE (mph)** | 6.21 | 7.90 | 9.08 | 14.19 |

- 最佳验证 MAE: **3.93** (epoch 32)
- Epoch 40 测试集 60min MAE: 5.61

### 7.2 与论文对比

| 来源 | 15min MAE | 30min MAE | 60min MAE |
|------|-----------|-----------|-----------|
| 本文复现 | 2.42 | 2.97 | 5.61 |
| DCRNN 原论文 | 2.77 | 3.15 | 3.60 |

15min 和 30min 指标与论文基本持平，60min 偏高，可能原因：
1. batch_size 受限（32 vs 论文 64）
2. 训练 epoch 数较少（45 vs 论文 100+）
3. 未使用论文作者的完整调参策略

### 7.3 交通模式聚类分析

| 聚类 | 传感器数 | 平均速度 | 拥堵率 | 高峰降幅 | 特征 |
|------|----------|----------|--------|----------|------|
| **全天通畅型** | 79 | 63 mph | 2.1% | ~5 mph | 偏远高速，几乎不堵 |
| **高峰敏感型** | 117 | 57 mph | 14.3% | ~12 mph | 市区高速，有明显高峰 |
| **高频拥堵型** | 11 | 40 mph | 55.8% | ~20 mph | 核心瓶颈路段，常年拥堵 |

### 7.4 交通趋势关键发现

- 平均速度：58.5 mph
- 平均拥堵比例：11.8%
- 高峰速度降幅：12.2 mph（晚高峰比早高峰严重 4 mph）
- 周末 vs 工作日：周末快 4.5 mph
- 最堵时段集中在下午 5 点（最堵传感器速度降至 ~30 mph）

---

## 8. 代码结构

```
dcrnn_pytorch/
├── 核心模型
│   ├── dcrnn_cell.py          # DCGRU 单元 (103 行)
│   ├── dcrnn_model.py         # Seq2Seq 编码器-解码器 (108 行)
│   ├── utils.py               # 数据加载/评估/图矩阵 (148 行)
│   ├── train.py               # 基线训练脚本 (203 行)
│   └── train_kg.py            # KG 增强训练 (163 行)
│
├── 知识图谱
│   ├── build_kg.py            # Neo4j KG 构建 (66 行)
│   ├── enrich_kg.py           # osmnx 道路类型富化 (125 行)
│   ├── fill_pois.py           # POI 批量下载与匹配 (100 行)
│   ├── write_trends.py        # 时序趋势统计 (133 行)
│   ├── cluster_sensors.py     # K-Means 聚类 (94 行)
│   ├── write_predictions.py   # 预测指标回写 (95 行)
│   ├── export_kg_image.py     # KG 可视化导出
│   └── check_neo4j.py         # 数据库状态检查
│
├── LLM 系统
│   ├── llm_kg.py              # 传感器描述 + 批量查询 (193 行)
│   └── query_kg.py            # 命令行交互式查询 (168 行)
│
├── Web 应用
│   ├── server.py              # Flask 后端 (65 行)
│   ├── chat.html              # 聊天前端 (~100 行)
│   ├── config.py              # 环境变量管理
│   └── requirements.txt       # Python 依赖
│
├── 配置文件
│   ├── config/dcrnn_la.yaml   # 模型超参数
│   ├── .env.example           # 环境变量模板
│   └── .gitignore             # Git 忽略规则
│
├── 数据 (部分 gitignored)
│   ├── data/sensor_graph/     # 传感器坐标 + 邻接矩阵
│   └── data/METR-LA/          # 预处理训练序列 (gitignored)
│
├── 输出
│   └── outputs/               # 7 张 PPT 就绪图表
│
└── 文档
    ├── README.md              # 项目说明
    └── TECHNICAL_DOC.md       # 本文档 (技术详解)
```

---

## 9. 部署方案

### 9.1 安全性措施

| 措施 | 实现 |
|------|------|
| **密钥分离** | 所有密钥存储在 `.env` 文件中，通过 `config.py` 读取 |
| **Git 保护** | `.gitignore` 排除 `.env`、`cache/`、训练数据 |
| **模板文件** | `.env.example` 提供空模板，可安全提交 |
| **环境变量** | 支持系统环境变量（部署平台直接注入） |
| **11 个脚本** | 全部使用 `from config import` 替代硬编码 |

### 9.2 部署选项

| 方案 | 优势 | 劣势 |
|------|------|------|
| **bore 隧道** (当前) | 无需注册，一行命令 | 端口随机，电脑需保持开机 |
| **Render** | 免费，自动 HTTPS，休眠唤醒 | 需绑卡验证 |
| **Railway** | $5/月免费额度，不休眠 | 需绑卡 |
| **PythonAnywhere** | 完全免费，无需绑卡 | 外网访问可能受限 |

### 9.3 bore 隧道使用

```bash
# 启动本地服务
python server.py &

# 创建公网隧道
bore local 5000 --to bore.pub
# 输出: listening at bore.pub:29314
# 公网地址: http://bore.pub:29314
```

---

## 10. 未来方向

### 10.1 KG 增强模型训练

`train_kg.py` 已编写待运行。将知识图谱语义邻接矩阵作为第 3 个扩散支撑图，验证地理先验是否能提升长时预测精度（尤其是当前薄弱的 60min 指标）。

### 10.2 图神经网络对比

可尝试将 KG 中的 POI 关系编码为异构图，使用 RGCN/HAN 等异构图神经网络替代 DCRNN 的空间编码器，引入更丰富的语义信息。

### 10.3 LLM 增强方向

- **图谱推理**：让 LLM 基于传感器相似关系链式推理
- **自动报告生成**：输入时间段 → LLM 自动生成交通分析报告
- **多跳问答**：支持 "拥堵区域附近的传感器预测误差如何？" 等复杂查询

### 10.4 交互可视化

集成 Leaflet 地图，在地图上展示传感器位置、实时速度热力图、聚类分布，支持点击查看详情和趋势图。

---

## 附录 A：Cypher 查询参考

```cypher
-- 基础统计
MATCH (s:Sensor) RETURN count(s), avg(s.avg_speed), avg(s.congestion_ratio)

-- 拥堵 Top 5
MATCH (s:Sensor) RETURN s.sid, s.congestion_ratio
ORDER BY s.congestion_ratio DESC LIMIT 5

-- 预测误差分析
MATCH (s:Sensor) WHERE s.road_type='motorway'
RETURN avg(s.mae_15min), avg(s.mae_30min), avg(s.mae_60min)

-- POI 关联
MATCH (s:Sensor)-[:NEAR]->(p:POI) WHERE p.type='restaurant'
RETURN s.sid, count(p) ORDER BY count(p) DESC LIMIT 5

-- 聚类分析
MATCH (s:Sensor) WHERE s.cluster='高频拥堵型'
RETURN s.sid, s.peak_drop ORDER BY s.peak_drop DESC

-- 相似传感器
MATCH (s:Sensor {sid: 773869})-[:SIMILAR_TO]->(t:Sensor)
RETURN t.sid, t.cluster, t.avg_speed
```

## 附录 B：环境配置

```bash
# Python 环境
conda create -n dcrnn python=3.11
conda activate dcrnn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy pyyaml h5py
pip install neo4j openai osmnx scikit-learn flask

# 配置密钥
cp .env.example .env
# 编辑 .env 填入真实密钥

# 数据准备
python generate_data.py          # h5 → 训练序列
python build_kg.py               # 构建 Neo4j 知识图谱
python enrich_kg.py              # osmnx 道路富化
python write_trends.py           # 趋势指标计算
python cluster_sensors.py        # 模式聚类

# 训练
python train.py                  # 基线 DCRNN

# 启动服务
python server.py                 # Web 服务 → localhost:5000
python query_kg.py               # 命令行交互查询
```

---

> **维护者**: zZENwe  
> **GitHub**: https://github.com/zZENwe/traffic-kg  
> **数据集**: METR-LA (Li et al., ICLR 2018)  
> **模型**: DCRNN (Li et al., ICLR 2018)  
> **LLM**: DeepSeek-chat
