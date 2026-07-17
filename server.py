"""Web chat UI for Traffic KG querying."""
import json
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

@app.route('/api/query', methods=['POST'])
def query():
    question = request.json.get('question', '')
    if not question:
        return jsonify({'error': 'empty question'})

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
    })

if __name__ == '__main__':
    print("启动服务: http://localhost:6060")
    app.run(host='0.0.0.0', port=6060, debug=False)
