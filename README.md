# DCRNN 交通流量预测 + 知识图谱增强

基于 **DCRNN**（Diffusion Convolutional Recurrent Neural Network, ICLR 2018）在 METR-LA 数据集上的交通流量预测复现，并引入**传感器空间知识图谱**进行知识增强。

## 项目结构

```
dcrnn_pytorch/
├── README.md
├── config/
│   └── dcrnn_la.yaml          # 模型与训练超参数
│
├── dcrnn_cell.py              # DCGRU单元：扩散图卷积替代矩阵乘法
├── dcrnn_model.py             # Seq2Seq编码器-解码器（计划采样）
├── utils.py                   # 数据加载、标准化、评估指标、图矩阵构建
├── train.py                   # 基线训练入口（纯路网邻接矩阵）
├── train_kg.py                # KG增强训练入口（路网 + 语义邻接矩阵）
├── run.py                     # 统一入口（--task baseline / kg）
│
├── generate_data.py           # 数据预处理：h5 → 训练/验证/测试序列
├── build_kg.py                # 构建Neo4j传感器知识图谱 + 语义邻接矩阵
├── check_neo4j.py             # 检查Neo4j数据库状态
├── export_kg_image.py         # 导出知识图谱为PNG图片
├── test_model.py              # 模型正确性测试（随机数据验证前向+反向传播）
│
├── data/
│   ├── metr-la.h5             # METR-LA原始数据（34,272时间片 × 207传感器）
│   ├── METR-LA/               # 预处理序列数据（train.npz, val.npz, test.npz）
│   └── sensor_graph/
│       ├── graph_sensor_locations.csv  # 207个传感器经纬度
│       ├── graph_sensor_ids.txt        # 传感器ID列表
│       ├── adj_mx.pkl                  # 路网邻接矩阵（原始，基于空间距离阈值）
│       └── kg_adj.pkl                  # 语义邻接矩阵（高斯核归一化）
│
├── outputs/
│   ├── kg_full.png            # 传感器知识图谱全景（207节点，1035边）
│   └── kg_zoom.png            # 洛杉矶市区放大（~187节点，带距离标注）
│
└── logs/
    ├── best_model.pt          # 最佳模型检查点（epoch 32, val_mae=3.93）
    └── training.log           # 完整训练日志（45 epochs）
```

## 环境

- Python 3.11（Conda 环境 `dcrnn`）
- PyTorch 2.5.1+cu121
- NVIDIA RTX 4070 Laptop GPU（8GB VRAM）
- Neo4j AuraDB（云数据库）

## 核心概念

### DCRNN 原理

| 组件 | 说明 |
|------|------|
| **扩散卷积** | 将交通视为图上的扩散过程，用随机游走矩阵的切比雪夫多项式近似图卷积 |
| **DCGRU** | 将 GRU 中的矩阵乘法替换为扩散卷积，使门控机制感知空间结构 |
| **Seq2Seq** | 编码器编码历史12步 → 解码器预测未来12步（1小时） |
| **计划采样** | 训练时逐步用模型预测替代真实值，缓解训练/推理不一致 |

### KG增强策略

- **基线**：DCRNN 使用路网邻接矩阵的 2 个扩散支持（出度+入度随机游走）
- **增强**：引入知识图谱语义邻接矩阵作为第 3 个扩散支持（高斯核平滑的地理邻近性矩阵）
- **目的**：将传感器空间邻近性作为图卷积的额外先验知识

### 知识图谱

- 207 个传感器节点
- 1035 条 ROAD_DISTANCE 边（每个传感器连接最近的5个邻居）
- 语义邻接矩阵：$A_{ij} = \exp(-d_{ij}^2 / (2\sigma^2))$（高斯核，$\sigma$ 为非零距离的标准差）
- 存储于 Neo4j AuraDB：`neo4j+s://05be6316.databases.neo4j.io`

## 快速开始

```bash
# 激活环境
conda activate dcrnn

# 数据预处理（如METR-LA目录下没有数据）
python generate_data.py

# 检查Neo4j知识图谱
python check_neo4j.py

# 导出知识图谱图片
python export_kg_image.py

# 训练基线模型
python run.py --task baseline

# 训练KG增强模型
python run.py --task kg

# 验证模型正确性
python test_model.py
```

## 训练结果

### 基线 DCRNN（纯路网）

| 指标 | 15分钟 | 30分钟 | 60分钟 |
|------|--------|--------|--------|
| **MAE** | 2.42 | 2.97 | 5.61 |
| **MAPE** | 5.61% | 6.78% | 11.88% |
| **RMSE** | 6.21 | 7.90 | 14.19 |

- 最佳验证 MAE：**3.93**（epoch 32）
- 总参数量：372,352
- 训练设备：CUDA（RTX 4070 Laptop）
- 每个 epoch 约 70 秒

### 论文参考值（DCRNN, ICLR 2018）

| 指标 | 15分钟 | 30分钟 | 60分钟 |
|------|--------|--------|--------|
| **MAE** | 2.77 | 3.15 | 3.60 |

> 注：本复现 15min/30min 结果与论文接近，60min 偏高，原因可能为：(1) batch_size 受限（32 vs 论文64）；(2) 未使用论文作者的完整调参策略；(3) 训练步数较少。**对于本科/硕士毕业汇报已足够。**

### KG增强版

尚未训练。如训练，使用 `python run.py --task kg`，模型将使用 3 个扩散支持（2 路网 + 1 语义）。

## 检查点恢复

```bash
# 从检查点继续训练
python run.py --task baseline --resume logs/best_model.pt
```

## PPT素材图说明

`outputs/` 目录包含 7 张可直接用于 PPT 的图片：

| 图片 | 内容 | 建议放置位置 |
|------|------|-------------|
| `kg_zoom.png` | LA市区传感器知识图谱可视化，节点=传感器，边=空间邻近关系，标注了距离(km) | **知识图谱构建页** |
| `kg_full.png` | 207个传感器全局知识图谱概览，展示完整空间覆盖 | 知识图谱页（背景/概览） |
| `model_config.png` | DCRNN模型超参数配置表，包含输入输出维度、网络结构、参数量等 | **方法介绍页**（放右侧） |
| `training_curves.png` | 左：训练/验证MAE曲线，标注最佳epoch；右：不同epoch下各时间尺度测试MAE | **实验过程页** |
| `predictions.png` | 2个传感器在15/30/60分钟预测 vs 真实速度对比曲线 | **预测效果展示页** |
| `metrics_by_horizon.png` | 12个预测步（5min~60min）的MAE和MAPE柱状图 | **结果分析页** |
| `comparison_and_kg.png` | 左：本复现 vs 论文ICLR 2018的MAE对比；右：KG增强DCRNN架构示意图 | **结果对比 + 方法创新页** |

> 建议PPT顺序：背景 → 方法(config) → 知识图谱(kg_zoom) → KG增强(comparison右半) → 训练(curves) → 预测效果(predictions) → 结果分析(metrics) → 对比(comparison左半)
