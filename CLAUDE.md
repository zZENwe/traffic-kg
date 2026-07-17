# CLAUDE.md — AI 协作指南

## 项目概述

**交通时空知识图谱**：DCRNN 交通预测 + Neo4j 知识图谱 + DeepSeek LLM 智能查询。使用 METR-LA 数据集（洛杉矶 207 个高速传感器，2012 年 3-6 月）。

## 环境

```bash
conda activate dcrnn
# Python 3.11, PyTorch 2.5.1+cu121, NVIDIA RTX 4070 (8GB)
```

## 核心命令

```bash
python server.py          # 启动 Web 聊天服务 → http://localhost:5000
python query_kg.py        # 启动命令行交互查询
python train.py           # 训练基线 DCRNN (~50min)
python build_kg.py        # 构建 Neo4j 知识图谱
python enrich_kg.py       # osmnx 道路类型富化
python write_trends.py    # 时序趋势统计并写入 Neo4j
python cluster_sensors.py # K-Means 传感器聚类
python write_predictions.py # 模型预测指标回写 Neo4j
python check_neo4j.py     # 检查 Neo4j 数据库状态
```

## 架构速览

```
用户提问 → Flask API → DeepSeek Text-to-Cypher → Neo4j 执行 → DeepSeek 格式化回答
```

- **前端**: `chat.html` (纯 HTML/CSS/JS, 无框架)
- **后端**: `server.py` (Flask, 65 行)
- **LLM**: `llm_kg.py`, `query_kg.py` (DeepSeek-chat API)
- **模型**: `dcrnn_cell.py` (DCGRU 单元), `dcrnn_model.py` (Seq2Seq), `utils.py` (图矩阵/数据加载)

## Neo4j 数据库

- 云端 AuraDB: `neo4j+s://05be6316.databases.neo4j.io`
- 466 个节点 (207 Sensor + 259 POI), 2,151 条边
- 密钥在 `.env` 文件中, 通过 `config.py` 加载
- `.env` 已加入 `.gitignore`, 使用 `.env.example` 作为模板

## 传感器属性 (Neo4j Sensor 节点)

基础: sid, lat, lon, road_type, description, cluster
预测: mae_15min, mae_30min, mae_60min
趋势: avg_speed, congestion_ratio, morning_avg, evening_avg, offpeak_avg, peak_drop, weekday_avg, weekend_avg, worst_hour, worst_speed
关系: ROAD_DISTANCE {km}, NEAR {distance_m}, SIMILAR_TO {cluster}

## 数据流

METR-LA HDF5 → generate_data.py → train/val/test.npz → DCRNN (PyTorch) → 预测
传感器 CSV → build_kg.py → Neo4j ← write_trends.py ← HDF5 时序统计
                                            ← enrich_kg.py ← osmnx OSM
                                            ← cluster_sensors.py ← K-Means
                                            ← llm_kg.py ← DeepSeek API

## 重要说明

- `.env` 包含真实密钥，不要提交到 Git
- 训练数据和模型检查点在 `data/METR-LA/` 和 `logs/` (已 gitignore)
- DeepSeek API 是 OpenAI 兼容的，base_url 是 `https://api.deepseek.com`
- osmnx 在大陆可能慢，已启用缓存 (`cache/` 目录已 gitignore)
- 详细技术文档见 `TECHNICAL_DOC.md`

## 给你的 AI 的提示

把这个仓库链接发给 Claude/Cursor/Windsurf 等 AI 工具，然后说：

> "帮我看看这个交通知识图谱项目。先读 README.md 和 TECHNICAL_DOC.md 了解架构，然后根据 .env.example 创建 .env 文件，填好密钥后就可以运行了。先跑 python check_neo4j.py 确认数据库连接，再跑 python server.py 启动服务。"

AI 会自动读取 CLAUDE.md 获取上下文，然后按你的需求操作。
