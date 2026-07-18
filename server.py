"""Web chat UI for Traffic KG querying."""
import json
import re
import os
import numpy as np
import h5py
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from neo4j import GraphDatabase
from openai import OpenAI
from config import NEO4J_URI, NEO4J_AUTH, DEEPSEEK_KEY

app = Flask(__name__)

URI = NEO4J_URI
AUTH = NEO4J_AUTH
DEEPSEEK_KEY = DEEPSEEK_KEY

client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
driver = GraphDatabase.driver(URI, auth=AUTH)

# ---- Load time-series data for heatmap visualization ----
TIMESERIES_CACHE = None

def load_timeseries():
    global TIMESERIES_CACHE
    if TIMESERIES_CACHE:
        return TIMESERIES_CACHE

    base = os.path.dirname(os.path.abspath(__file__))
    h5_path = os.path.join(base, 'data', 'metr-la.h5')
    csv_path = os.path.join(base, 'data', 'sensor_graph', 'graph_sensor_locations.csv')

    if not os.path.exists(h5_path):
        return None

    # Load speed data and downsample to every 30min (6 steps of 5min)
    with h5py.File(h5_path, 'r') as f:
        vals = f['data/block0_values'][:]  # (34272, 207)
        times_raw = f['data/axis1'][:]
    times_ns = times_raw.astype(np.int64)
    if times_ns[0] > 1e15:
        times_s = times_ns / 1e9
    else:
        times_s = times_ns

    # Downsample: every 6 steps = ~30min, pick a representative day (e.g. day 30)
    step = 12  # ~1 hour intervals for smoother animation
    vals_ds = vals[::step, :]  # (N_steps, 207)
    times_ds = times_s[::step]

    # Convert to list of timestamps
    from datetime import datetime, timezone, timedelta
    # PST = UTC-8 (LA in March is UTC-7 with DST, but timestamps are UTC)
    timestamps = [datetime.fromtimestamp(t, tz=timezone.utc).strftime('%m/%d %H:%M') for t in times_ds[:288]]  # ~24h worth at 1h intervals

    # Load sensor coordinates
    sensors_df = pd.read_csv(csv_path, index_col=0)
    sensors = [{"sid": int(sensors_df.iloc[i]['sensor_id']),
                "lat": float(sensors_df.iloc[i]['latitude']),
                "lon": float(sensors_df.iloc[i]['longitude'])}
               for i in range(min(len(sensors_df), vals.shape[1]))]

    # Build edges from top-3 nearest neighbors
    coords = sensors_df[['latitude', 'longitude']].values
    geo_dist = np.sqrt(
        (coords[:, None, 0] - coords[None, :, 0]) ** 2 +
        (coords[:, None, 1] - coords[None, :, 1]) ** 2
    )
    edges = []
    n = len(sensors_df)
    for i in range(n):
        nearest = np.argsort(geo_dist[i])[1:4]  # top 3 neighbors
        for j in nearest:
            edges.append({"from": int(sensors_df.iloc[i]['sensor_id']),
                          "to": int(sensors_df.iloc[int(j)]['sensor_id']),
                          "km": round(float(geo_dist[i, j]), 3)})

    # Speed data: limit to first 288 steps (~1 day at 1h) for manageable transfer
    speeds = np.round(vals_ds[:288, :], 1).tolist()

    TIMESERIES_CACHE = {
        "sensors": sensors,
        "edges": edges,
        "speeds": speeds,
        "timestamps": timestamps,
        "speedMin": float(np.nanmin(vals_ds[:288])),
        "speedMax": float(np.nanmax(vals_ds[:288])),
    }
    return TIMESERIES_CACHE

SCHEMA = """你是Neo4j Cypher生成器。有207个Sensor节点和259个POI节点。

=== 关键规则(最重要!) ===
拥堵/交通/速度/几点/高峰期 → 用 congestion_ratio, peak_drop, worst_hour, morning_avg, evening_avg 等
预测/误差/准确度 → 用 mae_15min, mae_30min, mae_60min
天气/雨 → 用 rain_speed_drop, rain_congestion_increase
道路/分类 → 用 road_type, cluster
周边/设施 → 用 POI节点+NEAR关系
模式/相似 → 用 SIMILAR_TO关系

=== Sensor属性速查 ===
sid, lat, lon, road_type, cluster, description
mae_15min, mae_30min, mae_60min (预测误差,问预测时用)
avg_speed(均速40-65mph), congestion_ratio(拥堵占比0-0.55), peak_drop(高峰降幅mph)
morning_avg, evening_avg, offpeak_avg, weekday_avg, weekend_avg
worst_hour(最堵小时0-23), worst_speed(最堵时速)
rain_speed_drop, rain_congestion_increase, heavy_rain_drop, rain_hours

=== POI: name, type ===
关系: ROAD_DISTANCE{km}, NEAR{distance_m}, SIMILAR_TO{cluster}

=== 示例 ===
最堵传感器 → MATCH (s:Sensor) RETURN s.sid, s.congestion_ratio ORDER BY s.congestion_ratio DESC LIMIT 5
几点最堵 → MATCH (s:Sensor) RETURN s.worst_hour as h, count(s) as n ORDER BY n DESC LIMIT 1
早高峰最慢 → MATCH (s:Sensor) RETURN s.sid, s.morning_avg ORDER BY s.morning_avg ASC LIMIT 5
预测最不准 → MATCH (s:Sensor) RETURN s.sid, s.mae_60min ORDER BY s.mae_60min DESC LIMIT 5
雨天降速大 → MATCH (s:Sensor) RETURN s.sid, s.rain_speed_drop ORDER BY s.rain_speed_drop DESC LIMIT 5
高速公路多少 → MATCH (s:Sensor) WHERE s.road_type='motorway' RETURN count(s)
POI最多 → MATCH (s)-[:NEAR]->(p) RETURN s.sid, count(p) ORDER BY count(p) DESC LIMIT 5
模式相似 → MATCH (s {sid:773869})-[:SIMILAR_TO]->(t) RETURN t.sid, t.cluster"""

def call_llm(messages):
    r = client.chat.completions.create(
        model="deepseek-chat", messages=messages,
        temperature=0.2, max_tokens=1024,
    )
    return r.choices[0].message.content

@app.route('/')
def index():
    return send_from_directory('.', 'chat.html')

@app.route('/heatmap')
def heatmap_page():
    return send_from_directory('.', 'heatmap.html')

@app.route('/api/query', methods=['POST'])
def query():
    question = request.json.get('question', '')
    reasoning = request.json.get('reasoning', False)
    if not question:
        return jsonify({'error': 'empty question'})

    # === Reasoning Mode: multi-query deep analysis (manual toggle) ===
    if reasoning:
        with driver.session() as session:
            # Extract sensor ID if mentioned
            sids = re.findall(r'\b(\d{5,6})\b', question)
            target_sid = int(sids[0]) if sids else None

            # Run 4 comprehensive queries
            data = {}

            # 1. Target sensor(s) detail
            if target_sid:
                r = session.run("""
                    MATCH (s:Sensor {sid: $sid})
                    OPTIONAL MATCH (s)-[:NEAR]->(p:POI)
                    RETURN s.sid as sid, s.road_type as road, s.cluster as cluster,
                           s.congestion_ratio as cong, s.peak_drop as pd,
                           s.morning_avg as ma, s.evening_avg as ea, s.offpeak_avg as oa,
                           s.weekday_avg as wd, s.weekend_avg as we,
                           s.mae_15min as m15, s.mae_30min as m30, s.mae_60min as m60,
                           s.rain_speed_drop as rd, s.rain_congestion_increase as rci,
                           s.worst_hour as wh, s.worst_speed as ws,
                           s.avg_speed as spd, s.description as desc,
                           collect(p.name)[0..5] as pois
                    ORDER BY s.congestion_ratio DESC LIMIT 5
                """, sid=target_sid)
            else:
                r = session.run("""
                    MATCH (s:Sensor)
                    OPTIONAL MATCH (s)-[:NEAR]->(p:POI)
                    RETURN s.sid as sid, s.road_type as road, s.cluster as cluster,
                           s.congestion_ratio as cong, s.peak_drop as pd,
                           s.morning_avg as ma, s.evening_avg as ea, s.offpeak_avg as oa,
                           s.weekday_avg as wd, s.weekend_avg as we,
                           s.mae_15min as m15, s.mae_30min as m30, s.mae_60min as m60,
                           s.rain_speed_drop as rd, s.rain_congestion_increase as rci,
                           s.worst_hour as wh, s.worst_speed as ws,
                           s.avg_speed as spd, s.description as desc,
                           collect(p.name)[0..5] as pois
                    ORDER BY s.congestion_ratio DESC LIMIT 5
                """)
            data['target'] = r.data()

            # 2. Similar sensors for comparison
            if target_sid:
                r = session.run("""
                    MATCH (s:Sensor {sid: $sid})-[:SIMILAR_TO]->(t:Sensor)
                    RETURN t.sid as sid, t.cluster as cluster, t.congestion_ratio as cong,
                           t.peak_drop as pd, t.rain_speed_drop as rd
                    LIMIT 5
                """, sid=target_sid)
                data['similar'] = r.data()

            # 3. Road type & cluster averages
            r = session.run("""
                MATCH (s:Sensor)
                RETURN s.road_type as road, s.cluster as cluster,
                       count(s) as n, avg(s.congestion_ratio) as cong,
                       avg(s.peak_drop) as pd, avg(s.rain_speed_drop) as rd,
                       avg(s.mae_60min) as mae
                ORDER BY n DESC
            """)
            data['averages'] = r.data()

            # 4. Global extremes for context
            r = session.run("""
                MATCH (s:Sensor)
                RETURN max(s.congestion_ratio) as max_cong,
                       min(s.avg_speed) as min_speed,
                       max(s.peak_drop) as max_pd,
                       max(s.rain_speed_drop) as max_rd
            """)
            data['extremes'] = r.data()

            cypher = 'REASONING_MODE'  # mark for response

            # Build rich reasoning prompt
            reason_prompt = f"""你是交通分析师AI。基于以下多维数据进行深度推理分析。要求:
1. 先指出关键数据,再分析因果链条
2. 横向对比同类传感器,指出异常值
3. 给出1-2条可操作建议
4. 结尾给出2-3个相关追问
格式: [分析]...[建议]...---追问...

背景: LA高速207个传感器,均值speed=58.5mph, cong=12%, peak_drop=12mph, rain_drop=1.7mph

用户问题: {question}

目标传感器数据: {json.dumps(data.get('target',[]), ensure_ascii=False, default=str)}
相似传感器: {json.dumps(data.get('similar',[]), ensure_ascii=False, default=str)}
分组统计: {json.dumps(data.get('averages',[]), ensure_ascii=False, default=str)}
全局极值: {json.dumps(data.get('extremes',[]), ensure_ascii=False, default=str)}

请分析:"""
            answer = call_llm([{"role": "user", "content": reason_prompt}])
            parts = answer.split('---', 1)
            main = parts[0].strip()
            sug = parts[1].strip() if len(parts) > 1 else ''
            return jsonify({'answer': main, 'cypher': '多查询综合分析', 'suggestions': sug,
                            'records': [data]})

    # === Phase 1: Keyword-based Cypher templates (100% reliable for common queries) ===
    cypher = None
    q = question

    # Congestion / Traffic patterns
    if any(w in q for w in ['拥堵', '堵车', '最堵', '堵不堵', '挤不挤']):
        if any(w in q for w in ['几点', '时间段', '什么时候', '哪个小时']):
            cypher = "MATCH (s:Sensor) RETURN s.worst_hour as h, count(s) as n, avg(s.worst_speed) as s ORDER BY h"
        elif any(w in q for w in ['雨天', '下雨', '降雨']):
            cypher = "MATCH (s:Sensor) RETURN s.sid, s.rain_congestion_increase as ci ORDER BY ci DESC LIMIT 5"
        elif any(w in q for w in ['高峰', '早晚']):
            cypher = "MATCH (s:Sensor) RETURN s.sid, s.peak_drop, s.morning_avg, s.evening_avg ORDER BY s.peak_drop DESC LIMIT 5"
        elif any(w in q for w in ['高速', 'motorway', '道路', '类型']):
            cypher = "MATCH (s:Sensor) RETURN s.road_type, avg(s.congestion_ratio) as c ORDER BY c DESC"
        else:
            cypher = "MATCH (s:Sensor) RETURN s.sid, s.congestion_ratio, s.cluster, s.road_type ORDER BY s.congestion_ratio DESC LIMIT 10"

    # Speed patterns
    elif any(w in q for w in ['速度', '多快', '时速', '快慢']):
        if any(w in q for w in ['早', '早上', '早高峰', '上午']):
            cypher = "MATCH (s:Sensor) RETURN s.sid, s.morning_avg ORDER BY s.morning_avg ASC LIMIT 5"
        elif any(w in q for w in ['晚', '晚上', '晚高峰', '傍晚']):
            cypher = "MATCH (s:Sensor) RETURN s.sid, s.evening_avg ORDER BY s.evening_avg ASC LIMIT 5"
        elif any(w in q for w in ['周末']):
            cypher = "MATCH (s:Sensor) RETURN s.sid, s.weekend_avg ORDER BY s.weekend_avg DESC LIMIT 10"
        elif any(w in q for w in ['工作']):
            cypher = "MATCH (s:Sensor) RETURN s.sid, s.weekday_avg ORDER BY s.weekday_avg DESC LIMIT 10"
        elif any(w in q for w in ['雨天', '下雨']):
            cypher = "MATCH (s:Sensor) RETURN s.sid, s.rain_speed_drop ORDER BY s.rain_speed_drop DESC LIMIT 5"
        else:
            cypher = "MATCH (s:Sensor) RETURN s.sid, s.avg_speed ORDER BY s.avg_speed ASC LIMIT 10"

    # Weather
    elif any(w in q for w in ['天气', '下雨', '雨天', '降雨']):
        if any(w in q for w in ['拥堵', '堵', '塞']):
            cypher = "MATCH (s:Sensor) RETURN s.sid, s.rain_congestion_increase ORDER BY s.rain_congestion_increase DESC LIMIT 5"
        elif any(w in q for w in ['降速', '慢', '速度', '影响']):
            cypher = "MATCH (s:Sensor) RETURN s.sid, s.rain_speed_drop, s.heavy_rain_drop, s.rain_congestion_increase ORDER BY s.rain_speed_drop DESC LIMIT 5"
        else:
            cypher = "MATCH (s:Sensor) RETURN s.sid, s.rain_speed_drop, s.rain_congestion_increase ORDER BY s.rain_speed_drop DESC LIMIT 5"

    # Prediction / Model accuracy
    elif any(w in q for w in ['预测', '误差', 'mae', '准确', 'DCRNN', '模型']):
        if any(w in q for w in ['道路', '类型', '高速', 'motorway']):
            cypher = "MATCH (s:Sensor) RETURN s.road_type, avg(s.mae_15min), avg(s.mae_30min), avg(s.mae_60min) ORDER BY avg(s.mae_60min) DESC"
        else:
            cypher = "MATCH (s:Sensor) RETURN s.sid, s.mae_15min, s.mae_30min, s.mae_60min, s.road_type ORDER BY s.mae_60min DESC LIMIT 10"

    # Cluster / similarity
    elif any(w in q for w in ['聚类', '分类', '类型', '模式']):
        if any(w in q for w in ['个数', '多少', '数量']):
            cypher = "MATCH (s:Sensor) RETURN s.cluster, count(s) as n ORDER BY n DESC"
        else:
            cypher = "MATCH (s:Sensor) RETURN s.cluster, avg(s.avg_speed), avg(s.congestion_ratio), count(s) ORDER BY avg(s.congestion_ratio) DESC"
    elif any(w in q for w in ['相似', '像', '相近']):
        cypher = "MATCH (s)-[r:SIMILAR_TO]->(t) RETURN s.sid, collect(t.sid)[0..3] as similar ORDER BY s.sid LIMIT 5"

    # POI
    elif any(w in q for w in ['poi', '周边', '设施', '附近', '餐厅', '学校', '加油站', '商店']):
        cypher = "MATCH (s)-[:NEAR]->(p) RETURN s.sid, s.road_type, collect(p.name)[0..3] as pois, count(p) as n ORDER BY n DESC LIMIT 10"

    # Road type
    elif any(w in q for w in ['高速', '道路', 'motorway', '主干道', '匝道']):
        cypher = "MATCH (s:Sensor) RETURN s.road_type, count(s) as n, avg(s.avg_speed) as spd, avg(s.congestion_ratio) as cong ORDER BY n DESC"

    # Stats overview
    elif any(w in q for w in ['概览', '统计', '多少传感器', '总共有']):
        cypher = "MATCH (s:Sensor) RETURN count(s) as total, avg(s.avg_speed) as spd, avg(s.congestion_ratio) as cong, avg(s.mae_60min) as mae"

    # === Phase 2: Fallback to LLM if no template matched ===
    if cypher is None:
        cypher_prompt = f"""{SCHEMA}

问题: {question}
只输出Cypher:"""
        cypher = call_llm([{"role": "user", "content": cypher_prompt}]).strip()
    if cypher.startswith("```"):
        cypher = cypher.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if cypher.lower().startswith("cypher"):
        cypher = cypher[6:].strip()

    if cypher == "UNSUPPORTED":
        return jsonify({'error': '无法理解此问题', 'cypher': ''})

    # Execute
    try:
        with driver.session() as session:
            records = session.run(cypher).data()
    except Exception as e:
        return jsonify({'error': f'查询失败: {e}', 'cypher': cypher})

    if not records:
        return jsonify({'answer': '没有找到匹配数据。', 'cypher': cypher})

    # Format answer with insight + suggestions
    context = """你是交通分析师AI。根据数据回答问题，要求：
1. 答案用1-3句话，指出关键数字和趋势
2. 如果数据中有异常或有趣的模式，主动指出
3. 接着给出2-3条相关追问建议。格式严格如下：

[回答内容]

---
💡 你还可以问：
• 追问1
• 追问2

参考: 全传感器均值avg_speed=58.5mph, congestion_ratio=12%, peak_drop=12.2mph, mae_60min=5.9, rain_drop=1.7mph
不要使用emoji表情"""

    fmt_prompt = f"""{context}

问题: {question}
查询结果: {json.dumps(records[:15], ensure_ascii=False, default=str)}
{'...结果已截断' if len(records) > 15 else ''}

分析并回答:"""
    answer = call_llm([{"role": "user", "content": fmt_prompt}])

    # Separate answer from suggestions
    parts = answer.split('---', 1)
    main_answer = parts[0].strip()
    suggestions = parts[1].strip() if len(parts) > 1 else ''

    return jsonify({
        'answer': main_answer,
        'cypher': cypher,
        'suggestions': suggestions,
        'records': records[:20],
    })

@app.route('/api/extract_text', methods=['POST'])
def extract_text():
    """Extract entities & relationships from unstructured text via LLM and store in Neo4j."""
    text = (request.json.get('text', '') or '').strip()
    if not text:
        return jsonify({'error': 'empty text'})

    # Get existing schema for context
    with driver.session() as session:
        labels = [r[0] for r in session.run("CALL db.labels()").values()]
        rels = [r[0] for r in session.run("CALL db.relationshipTypes()").values()]

    # LLM prompt to extract structured knowledge
    extract_prompt = f"""你是一个知识图谱抽取器。从以下文本中提取实体和关系。

=== 已有的图Schema ===
节点标签: {labels}
关系类型: {rels}

=== 规则 ===
1. 识别文本中的实体(人物、地点、组织、事件、概念等)
2. 为每个实体分配type(用已有的标签，或新建)
3. 提取实体间的关系
4. 为实体添加描述和关键属性
5. 返回严格JSON: {{"entities":[{{"name":"...","type":"Type","props":{{"key":"value"}}}}],"relations":[{{"from":"实体name","type":"REL_TYPE","to":"实体name","props":{{}}}}]}}

=== 文本 ===
{text[:3000]}

只返回JSON:"""

    try:
        resp = call_llm([{"role": "user", "content": extract_prompt}])
        resp = resp.strip()
        if resp.startswith("```"):
            resp = resp.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(resp)
    except Exception as e:
        return jsonify({'error': f'LLM解析失败: {e}', 'raw': resp[:200]})

    entities = data.get('entities', [])
    relations = data.get('relations', [])
    if not entities:
        return jsonify({'error': '未能从文本中提取到实体'})

    # Store in Neo4j
    node_count, rel_count = 0, 0
    with driver.session() as session:
        for ent in entities:
            ent_type = ent.get('type', 'Entity')
            # Sanitize label (Neo4j labels can't have spaces)
            safe_type = ent_type.replace(' ', '_').replace('-', '_')
            props = ent.get('props', {}) or {}
            props['name'] = ent['name']
            props['description'] = ent.get('description', '')
            props['source'] = 'llm_extract'
            session.run(f"""
                MERGE (n:{safe_type} {{name: $name}})
                SET n += $props
            """, name=ent['name'], props=props)
            node_count += 1

        for rel in relations:
            from_name = rel.get('from', '')
            to_name = rel.get('to', '')
            rel_type = rel.get('type', 'RELATED')
            safe_rel = rel_type.replace(' ', '_').upper()
            r_props = rel.get('props', {}) or {}
            try:
                session.run(f"""
                    MATCH (a {{name: $from_name}}), (b {{name: $to_name}})
                    MERGE (a)-[r:{safe_rel}]->(b)
                    SET r += $props
                """, from_name=from_name, to_name=to_name, props=r_props)
                rel_count += 1
            except Exception:
                pass

    return jsonify({
        'message': f'成功抽取 {node_count} 个实体、{rel_count} 条关系',
        'entities': node_count,
        'relations': rel_count,
        'preview': entities[:5],
    })

@app.route('/api/timeseries')
def timeseries():
    data = load_timeseries()
    if data is None:
        return jsonify({'error': 'METR-LA HDF5 data not found'})
    return jsonify(data)

if __name__ == '__main__':
    print("启动服务: http://localhost:7070")
    app.run(host='0.0.0.0', port=7070, debug=False)
