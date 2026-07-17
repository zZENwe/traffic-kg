"""Cluster sensors by traffic patterns and create SIMILAR_TO edges in Neo4j."""
import numpy as np
from sklearn.cluster import KMeans
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_AUTH

URI = NEO4J_URI
AUTH = NEO4J_AUTH

print("Reading sensor trend data from Neo4j...")
driver = GraphDatabase.driver(URI, auth=AUTH)

with driver.session() as session:
    result = session.run("""
        MATCH (s:Sensor)
        RETURN s.sid as sid, s.avg_speed as avg, s.congestion_ratio as cr,
               s.morning_avg as ma, s.evening_avg as ea, s.peak_drop as pd,
               s.weekday_avg as wd, s.weekend_avg as we, s.road_type as rt
        ORDER BY s.sid
    """).data()

print(f"Read {len(result)} sensors")

# Build feature matrix
sids = [r['sid'] for r in result]
features = []
for r in result:
    features.append([
        r['avg'] or 0, r['cr'] or 0, r['ma'] or 0,
        r['ea'] or 0, r['pd'] or 0, r['wd'] or 0, r['we'] or 0,
    ])
X = np.array(features)
# Normalize
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# K-means clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
labels = kmeans.fit_predict(X)

# Name clusters based on characteristics
cluster_info = {}
for c in range(n_clusters):
    mask = labels == c
    cluster_info[c] = {
        'count': mask.sum(),
        'avg_speed': float(result[np.where(mask)[0][0]]['avg']) if mask.any() else 0,
        'avg_cr': float(np.mean([float(r['cr'] or 0) for i, r in enumerate(result) if labels[i] == c])),
        'avg_pd': float(np.mean([float(r['pd'] or 0) for i, r in enumerate(result) if labels[i] == c])),
    }

# Auto-name clusters
sorted_clusters = sorted(cluster_info.items(), key=lambda x: -x[1]['avg_speed'])
names = {}
for rank, (c_id, info) in enumerate(sorted_clusters):
    if info['avg_cr'] > 0.3:
        name = "高频拥堵型"
    elif info['avg_pd'] > 15:
        name = "高峰敏感型"
    elif info['avg_speed'] > 60:
        name = "全天通畅型"
    else:
        name = "中等波动型"
    names[c_id] = name

print("\nClusters:")
for c_id, info in sorted_clusters:
    print(f"  {names[c_id]}: {info['count']} sensors, "
          f"avg_speed={info['avg_speed']:.0f}mph, "
          f"cong={info['avg_cr']:.1%}, peak_drop={info['avg_pd']:.0f}mph")

# Write labels and SIMILAR_TO edges to Neo4j
print("\nWriting clusters to Neo4j...")
with driver.session() as session:
    # Remove old cluster edges
    session.run("MATCH ()-[r:SIMILAR_TO]->() DELETE r")

    # Set cluster labels
    for i, sid in enumerate(sids):
        c_id = int(labels[i])
        session.run("""
            MATCH (s:Sensor {sid: $sid})
            SET s.cluster = $cluster
        """, sid=sid, cluster=names[c_id])

    # Create SIMILAR_TO edges within each cluster (connect to top-3 nearest)
    for c in range(n_clusters):
        c_indices = [i for i in range(len(sids)) if labels[i] == c]
        c_features = X[c_indices]
        c_sids = [sids[i] for i in c_indices]

        if len(c_indices) < 2:
            continue

        for i, idx in enumerate(c_indices):
            # Find 3 nearest neighbors in same cluster
            dists = np.linalg.norm(c_features - c_features[i], axis=1)
            nearest = np.argsort(dists)[1:4]  # skip self
            for j in nearest:
                session.run("""
                    MATCH (a:Sensor {sid: $s1}), (b:Sensor {sid: $s2})
                    MERGE (a)-[:SIMILAR_TO {cluster: $cl}]->(b)
                """, s1=sids[idx], s2=c_sids[j], cl=names[c])

    # Verify
    edges = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r)").single().value()
    verification = session.run("""
        MATCH (s:Sensor) RETURN s.cluster as cl, count(s) as cnt ORDER BY cnt DESC
    """).data()
    print(f"  Created {edges} SIMILAR_TO edges")
    for v in verification:
        print(f"    {v['cl']}: {v['cnt']} sensors")

driver.close()
print("Done!")
