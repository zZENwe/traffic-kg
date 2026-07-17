"""Interactive Chinese NL querying for the Traffic KG via DeepSeek + Neo4j."""
import json
import sys
from neo4j import GraphDatabase
from openai import OpenAI
from config import NEO4J_URI, NEO4J_AUTH, DEEPSEEK_KEY

URI = NEO4J_URI
AUTH = NEO4J_AUTH
DEEPSEEK_KEY = DEEPSEEK_KEY

client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
driver = GraphDatabase.driver(URI, auth=AUTH)

SCHEMA_CACHE = None

def get_schema():
    global SCHEMA_CACHE
    if SCHEMA_CACHE:
        return SCHEMA_CACHE

    with driver.session() as session:
        labels = [r[0] for r in session.run("CALL db.labels()").values()]
        rels = [r[0] for r in session.run("CALL db.relationshipTypes()").values()]

        props = {}
        for label in labels:
            sample = session.run(f"MATCH (n:{label}) RETURN n LIMIT 1").single()
            if sample:
                props[label] = list(sample['n'].keys())

        counts = {}
        for label in labels:
            c = session.run(f"MATCH (n:{label}) RETURN count(n)").single().value()
            counts[label] = c

    SCHEMA_CACHE = {
        'labels': labels, 'rels': rels, 'props': props, 'counts': counts,
    }
    return SCHEMA_CACHE


def build_schema_prompt():
    s = get_schema()
    parts = []
    for label in s['labels']:
        parts.append(f"  {label}({s['counts'][label]}个): {{{', '.join(s['props'][label])}}}")
    return "\n".join([
        "Neo4j交通传感器知识图谱 -- LA高速公路传感器网络(207个):",
        *parts,
        f"关系: {', '.join(s['rels'])}",
        "ROAD_DISTANCE: 传感器间道路距离(km), NEAR: 传感器周边设施(距离m)",
        "road_type: motorway(高速)=186, motorway_link(匝道)=11, tertiary(三级路)=7, primary(主干道)=3",
        "mae_15min/30min/60min: 预测误差(越低越好), description: LLM生成的中文描述",
    ])


def call_llm(messages):
    r = client.chat.completions.create(
        model="deepseek-chat", messages=messages,
        temperature=0.2, max_tokens=1024,
    )
    return r.choices[0].message.content


def ask(question):
    schema = build_schema_prompt()

    # Step 1: question -> Cypher
    cypher_prompt = f"""{schema}

将用户中文问题转为Cypher查询(Neo4j 5.x)。只输出Cypher，不要解释。
若无法转换输出 UNSUPPORTED。

用户: {question}
Cypher:"""

    cypher = call_llm([{"role": "user", "content": cypher_prompt}]).strip()
    if cypher.startswith("```"):
        cypher = cypher.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if cypher.lower().startswith("cypher"):
        cypher = cypher[6:].strip()

    print(f"  -> {cypher}")

    if cypher == "UNSUPPORTED":
        return "抱歉，这个问题我无法转换为数据库查询。请换一种问法。"

    # Step 2: Execute Cypher
    try:
        with driver.session() as session:
            result = session.run(cypher)
            records = result.data()
    except Exception as e:
        return f"查询执行出错: {e}"

    if not records:
        return "没有找到匹配的数据。"

    # Step 3: Format results
    fmt_prompt = f"""将查询结果用中文简洁回答(1-3句话)。

问题: {question}
数据: {json.dumps(records[:15], ensure_ascii=False, default=str)}
{'...(已截断)' if len(records) > 15 else ''}

回答:"""

    answer = call_llm([{"role": "user", "content": fmt_prompt}])
    return answer


def show_stats():
    s = get_schema()
    print(f"节点: {sum(s['counts'].values())}  ")
    for label, cnt in s['counts'].items():
        print(f"  {label}: {cnt}个")
    print(f"关系: {', '.join(s['rels'])}")
    with driver.session() as session:
        total = session.run("MATCH ()-[r]->() RETURN count(r)").single().value()
        print(f"总边数: {total}")


def show_demo():
    print("试试这些问题:")
    print("  1. 预测误差最大的5个传感器在什么位置？")
    print("  2. 高速公路和普通道路的传感器数量各是多少？")
    print("  3. 周边餐厅最多的传感器是哪个？")
    print("  4. 60分钟预测误差小于4的传感器有哪几个？")
    print("  5. 帮我看看传感器773869周围有哪些POI")


def main():
    print("=" * 50)
    print("  交通时空知识图谱 - 自然语言查询")
    print("  输入中文问题，LLM自动查图回答")
    print("  命令: /stats /demo /schema /exit")
    print("=" * 50)

    while True:
        try:
            q = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not q:
            continue
        if q == "/exit":
            print("再见!")
            break
        if q == "/stats":
            show_stats()
            continue
        if q == "/demo":
            show_demo()
            continue
        if q == "/schema":
            print(build_schema_prompt())
            continue

        ans = ask(q)
        print(f"\n{ans}")


if __name__ == "__main__":
    main()
    driver.close()
