"""Build sensor KG in Neo4j and save semantic adjacency matrix."""
import pandas as pd
import numpy as np
import pickle
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_AUTH

URI = NEO4J_URI
AUTH = NEO4J_AUTH

sensors = pd.read_csv("data/sensor_graph/graph_sensor_locations.csv", index_col=0)
n = len(sensors)
print(f"Sensors: {n}")

# Compute geographic distance matrix
coords = sensors[['latitude', 'longitude']].values
geo_dist = np.sqrt(
    (coords[:, None, 0] - coords[None, :, 0]) ** 2 +
    (coords[:, None, 1] - coords[None, :, 1]) ** 2
)

# Connect and build graph
driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()
print("Neo4j connected!")

with driver.session() as session:
    # Clear old data
    session.run("MATCH (n) DETACH DELETE n")

    # Create 207 sensor nodes
    for idx, row in sensors.iterrows():
        session.run("""
            CREATE (s:Sensor {sid: $sid, lat: $lat, lon: $lon})
        """, sid=int(row['sensor_id']),
            lat=float(row['latitude']), lon=float(row['longitude']))

    # Create top-5 nearest for each (excluding self)
    edge_count = 0
    for i in range(n):
        nearest = np.argsort(geo_dist[i])[1:6]  # skip self (idx 0)
        for j in nearest:
            d = round(float(geo_dist[i, j]), 3)
            edge_count += 1
            session.run("""
                MATCH (a:Sensor {sid: $si}), (b:Sensor {sid: $sj})
                CREATE (a)-[:ROAD_DISTANCE {km: $d}]->(b)
            """, si=int(sensors.iloc[i]['sensor_id']),
                sj=int(sensors.iloc[int(j)]['sensor_id']), d=d)

    print(f"Created {n} nodes, {edge_count} edges")

driver.close()

# Save semantic adjacency matrix
sigma = np.std(geo_dist[geo_dist > 0])
semantic_adj = np.exp(-geo_dist ** 2 / (2 * sigma ** 2))
np.fill_diagonal(semantic_adj, 0)
d = semantic_adj.sum(axis=1)
d[d == 0] = 1
semantic_adj = (semantic_adj / d[:, None]).astype(np.float32)

with open("data/sensor_graph/kg_adj.pkl", "wb") as f:
    pickle.dump(semantic_adj, f)
print(f"KG adjacency saved: {semantic_adj.shape}")
print("Done!")
