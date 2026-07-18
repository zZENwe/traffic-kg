"""LLM-enhanced KG: sensor descriptions + text-to-Cypher querying with DeepSeek."""
import json
from neo4j import GraphDatabase
from openai import OpenAI
from config import NEO4J_URI, NEO4J_AUTH, DEEPSEEK_KEY


client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def call_llm(messages):
    """Call DeepSeek chat API."""
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return r.choices[0].message.content


def generate_sensor_descriptions():
    """Generate Chinese descriptions for all sensors via DeepSeek and write to Neo4j."""
    print("=" * 60)
    print("Task 1: Generating sensor descriptions via LLM")
    print("=" * 60)

    with driver.session() as session:
        sensors = session.run("""
            MATCH (s:Sensor)
            OPTIONAL MATCH (s)-[:NEAR]->(p:POI)
            RETURN s.sid as sid, s.road_type as road, s.district as district,
                   s.mae_15min as m15, s.mae_30min as m30, s.mae_60min as m60,
                   s.lat as lat, s.lon as lon,
                   collect(DISTINCT {name: p.name, type: p.type, dist: p.distance_m})[0..3] as pois
            ORDER BY s.sid
        """).data()
        print(f"  Fetched {len(sensors)} sensors from Neo4j")

    # Batch-describe sensors (5 per LLM call to save tokens)
    print("  Generating descriptions...")
    descriptions = {}
    for batch_start in range(0, len(sensors), 5):
        batch = sensors[batch_start:batch_start + 5]
        sensor_texts = []
        for s in batch:
            pois_str = ", ".join([f"{p['name']}({p['type']}, {p['dist']}m)"
                                  for p in s.get('pois', []) if p['name']]) or "无"
            sensor_texts.append(
                f"ID={s['sid']}: 道路={s['road']}, "
                f"预测误差(15/30/60分钟)={s['m15']:.1f}/{s['m30']:.1f}/{s['m60']:.1f}, "
                f"周边={pois_str}"
            )

        prompt = f"""你是一个交通分析师。为以下洛杉矶METR-LA交通传感器各写一句话描述（中文，不超过30字），
包含：道路类型、预测难度（误差越小越容易预测）、周边环境。

传感器：
{chr(10).join(sensor_texts)}

返回JSON数组：[{{"sid": 传感器ID, "desc": "一句话描述"}}, ...]"""

        try:
            resp = call_llm([{"role": "user", "content": prompt}])
            resp = resp.strip()
            if resp.startswith("```"):
                resp = resp.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(resp)
            for d in data:
                descriptions[d['sid']] = d['desc']
            print(f"    Batch {batch_start//5+1}/{(len(sensors)+4)//5}: {len(data)} sensors")
        except Exception as e:
            print(f"    Batch {batch_start//5+1} failed: {e}")
            continue

    print(f"  Generated {len(descriptions)} descriptions")

    # Batch write descriptions to Neo4j
    if descriptions:
        batch = [{"sid": sid, "desc": desc} for sid, desc in descriptions.items()]
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS data
                MATCH (s:Sensor {sid: data.sid})
                SET s.description = data.desc
            """, batch=batch)
            print(f"  Written {len(descriptions)} descriptions to Neo4j")


def build_schema():
    """Build schema prompt from live Neo4j introspection."""
    with driver.session() as session:
        labels = [r[0] for r in session.run("CALL db.labels()").values()]
        rels = [r[0] for r in session.run("CALL db.relationshipTypes()").values()]

        schema_parts = []
        for label in labels:
            sample = session.run(f"MATCH (n:{label}) RETURN n LIMIT 1").single()
            if sample:
                keys = list(sample['n'].keys())
                schema_parts.append(f"  {label}: {{{', '.join(keys)}}}")
        schema_str = "\n".join(schema_parts)

    return f"""Neo4j图数据库Schema:

节点标签及属性:
{schema_str}

关系类型:
  {', '.join(rels)}
  - (:Sensor)-[:ROAD_DISTANCE {{km: float}}]->(:Sensor)  传感器间道路距离
  - (:Sensor)-[:NEAR {{distance_m: float}}]->(:POI)      传感器周边设施

关键说明:
- 207个传感器在洛杉矶，road_type有motorway/motorway_link/primary/tertiary
- mae_15min/mae_30min/mae_60min是预测误差（越低越容易预测）
- POI的type有amenity/shop/leisure/tourism等
"""


def text_to_cypher(question, schema):
    """Convert Chinese natural language to Cypher query."""
    prompt = f"""{schema}

用户用中文提问关于交通传感器知识图谱的问题。请生成一条Cypher查询来回答。

用户问题: {question}

只返回Cypher查询语句，不要解释。如果不能生成有效查询，返回"UNSUPPORTED"。
Cypher:"""

    cypher = call_llm([{"role": "user", "content": prompt}])
    cypher = cypher.strip()
    if cypher.startswith("```"):
        cypher = cypher.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if cypher.lower().startswith("cypher"):
        cypher = cypher[6:].strip()
    return cypher


def answer_question(question, schema):
    """Full pipeline: question -> Cypher -> execute -> natural language answer."""
    print(f"\n  Q: {question}")

    cypher = text_to_cypher(question, schema)
    print(f"  Cypher: {cypher}")

    if cypher == "UNSUPPORTED":
        return "无法理解这个问题，请换一种问法。"

    try:
        with driver.session() as session:
            result = session.run(cypher)
            records = result.data()

        if not records:
            return "未找到匹配结果。"

        format_prompt = f"""将以下Cypher查询结果用中文简洁回答。

问题: {question}
结果: {json.dumps(records[:10], ensure_ascii=False, default=str)}
{'...(结果已截断)' if len(records) > 10 else ''}

用1-2句话回答，不要太长。"""
        answer = call_llm([{"role": "user", "content": format_prompt}])
        return answer

    except Exception as e:
        return f"查询执行失败: {e}"


def run_text_to_cypher_demo():
    """Run demo Text-to-Cypher queries."""
    print("\n" + "=" * 60)
    print("Task 2: Text-to-Cypher Natural Language Querying")
    print("=" * 60)

    schema = build_schema()

    demo_questions = [
        "哪种道路类型的传感器预测误差最大？",
        "高速公路(motorway)上有多少个传感器？它们的平均预测误差是多少？",
        "哪些传感器周边500米内POI最多？列出前5个",
        "预测最困难的5个传感器（60分钟误差最大）在哪里？它们是什么道路类型？",
    ]

    for q in demo_questions:
        ans = answer_question(q, schema)
        print(f"  A: {ans}")


if __name__ == "__main__":
    generate_sensor_descriptions()
    run_text_to_cypher_demo()
    driver.close()
    print("\nDone!")
