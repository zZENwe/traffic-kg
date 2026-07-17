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

SCHEMA = """Neo4j交通传感器知识图谱(LA高速公路, 207个传感器, 2012年3-6月):

节点属性(Sensor):
  基础: sid, lat, lon, road_type, description, cluster(聚类标签)
  预测: mae_15min, mae_30min, mae_60min (越低越准)
  趋势: avg_speed, congestion_ratio, morning_avg, evening_avg,
        offpeak_avg, peak_drop, weekday_avg, weekend_avg,
        worst_hour, worst_speed
  天气: rain_speed_drop(雨天降速mph), rain_congestion_increase(雨天拥堵增幅%),
        heavy_rain_drop(大雨降速), rain_hours(降雨时长)
节点(POI, 259个): {name, type}
关系:
  ROAD_DISTANCE {km}, NEAR {distance_m}, SIMILAR_TO {cluster}
road_type: motorway=186, motorway_link=11, tertiary=7, primary=3
平均速度58.5mph, 拥堵比例12%, 高峰降幅12mph, 雨天平均降速1.7mph, 拥堵+3.9%

Cypher示例(严格参考):
Q: 预测误差最大的5个传感器？-> MATCH (s:Sensor) RETURN s.sid, s.mae_60min ORDER BY s.mae_60min DESC LIMIT 5
Q: 高速上有多少传感器？-> MATCH (s:Sensor) WHERE s.road_type='motorway' RETURN count(s)
Q: 拥堵最严重的前3个？-> MATCH (s:Sensor) RETURN s.sid, s.congestion_ratio ORDER BY s.congestion_ratio DESC LIMIT 3
Q: 早晚高峰降幅最大的传感器？-> MATCH (s:Sensor) RETURN s.sid, s.peak_drop ORDER BY s.peak_drop DESC LIMIT 5
Q: 哪个传感器POI最多？-> MATCH (s:Sensor)-[:NEAR]->(p:POI) RETURN s.sid, count(p) as cnt ORDER BY cnt DESC LIMIT 5
Q: 全天通畅型传感器有多少？-> MATCH (s:Sensor) WHERE s.cluster='全天通畅型' RETURN count(s)
Q: 传感器773869和谁模式最相似？-> MATCH (s:Sensor {sid:773869})-[:SIMILAR_TO]->(t) RETURN t.sid, t.cluster
Q: 下雨天降速最严重的5个传感器？-> MATCH (s:Sensor) RETURN s.sid, s.rain_speed_drop ORDER BY s.rain_speed_drop DESC LIMIT 5
Q: 雨天拥堵增加最多的传感器？-> MATCH (s:Sensor) RETURN s.sid, s.rain_congestion_increase ORDER BY s.rain_congestion_increase DESC LIMIT 5"""

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

    # Text-to-Cypher
    cypher_prompt = f"""{SCHEMA}

将中文问题转为Cypher查询(Neo4j 5.x)。只输出Cypher，不要解释或markdown。不能转换则输出UNSUPPORTED。

问题: {question}
Cypher:"""
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

    # Format answer
    fmt_prompt = f"""将查询结果用中文简洁回答(1-3句话)。

问题: {question}
数据: {json.dumps(records[:15], ensure_ascii=False, default=str)}
{'...截断' if len(records) > 15 else ''}
回答:"""
    answer = call_llm([{"role": "user", "content": fmt_prompt}])

    return jsonify({'answer': answer, 'cypher': cypher})

if __name__ == '__main__':
    print("启动服务: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
