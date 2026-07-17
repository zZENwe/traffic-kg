"""Enrich Neo4j sensor KG with OSM road types and POIs. Batch approach."""
import osmnx as ox
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from shapely.geometry import Point
import time
from config import NEO4J_URI, NEO4J_AUTH

ox.settings.use_cache = True
ox.settings.log_console = False

URI = NEO4J_URI
AUTH = NEO4J_AUTH

sensors = pd.read_csv("data/sensor_graph/graph_sensor_locations.csv", index_col=0)
n = len(sensors)
lats = sensors['latitude'].values
lons = sensors['longitude'].values
center = (lats.mean(), lons.mean())
sensor_points = [Point(lon, lat) for lat, lon in zip(lats, lons)]
print(f"{n} sensors, center={center}")

# === 1. Download full drive network (one query) ===
print("\n[1/3] Downloading LA drive network (15km radius)...")
t0 = time.time()
G = ox.graph_from_point(center, dist=15000, network_type='drive')
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
edges = edges[edges['highway'].notna()].copy()
edges['highway_type'] = edges['highway'].apply(lambda x: x[0] if isinstance(x, list) else x)
print(f"  {len(edges)} edges in {time.time()-t0:.0f}s, types: {sorted(edges['highway_type'].unique())}")

# === 2. Match sensors to nearest road (local computation) ===
print("\n[2/3] Matching sensors to roads...")
road_types = []
for i, pt in enumerate(sensor_points):
    dists = edges.geometry.distance(pt)
    nearest = edges.iloc[dists.argmin()]
    road_types.append(str(nearest['highway_type']))

sensors['road_type'] = road_types
print(f"  Distribution: {pd.Series(road_types).value_counts().to_dict()}")

# === 3. Download POIs by sub-regions ===
print("\n[3/3] Downloading POIs by sub-regions...")
# Divide area into 3x3 grid
sub_centers = []
for lat_c in np.linspace(lats.min(), lats.max(), 3):
    for lon_c in np.linspace(lons.min(), lons.max(), 3):
        sub_centers.append((lat_c, lon_c))

all_pois = []
for j, sc in enumerate(sub_centers):
    radius = 6000  # 6km per cell covers the ~20x35km area with 3x3 grid
    try:
        pois = ox.features_from_point(sc, dist=radius, tags={
            'amenity': True, 'shop': True, 'leisure': True,
            'tourism': True, 'landuse': ['commercial', 'industrial', 'residential', 'retail'],
            'highway': ['motorway_junction'],
        })
        if pois is not None and len(pois) > 0:
            valid = pois[pois.geometry.notna()].copy()
            all_pois.append(valid)
            print(f"  Cell {j+1}/9 {sc}: {len(valid)} POIs")
        else:
            print(f"  Cell {j+1}/9 {sc}: 0 POIs")
    except Exception as e:
        print(f"  Cell {j+1}/9 {sc}: FAILED ({e})")
        continue

poi_count = sum(len(p) for p in all_pois)
print(f"  Total: {poi_count} POIs from {len(all_pois)} cells")

# === 4. Match sensors to nearest POIs ===
print("\n--- Matching POIs to sensors ---")
sensor_pois = []
for i, pt in enumerate(sensor_points):
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{n}...")
    buffer = pt.buffer(0.0045)  # ~500m
    nearby_list = []
    for pois_df in all_pois:
        nearby = pois_df[pois_df.geometry.within(buffer)]
        for _, poi in nearby.iterrows():
            tags = poi.to_dict()
            name = tags.get('name', '')
            if not isinstance(name, str) or not name or name == 'nan':
                continue
            poi_type = str(tags.get('amenity') or tags.get('shop') or
                          tags.get('leisure') or tags.get('tourism') or
                          tags.get('landuse') or tags.get('highway') or 'place')
            d = pt.distance(poi.geometry) * 111000
            nearby_list.append((name.strip(), poi_type, round(d, 0)))
    nearby_list.sort(key=lambda x: x[2])
    sensor_pois.append(nearby_list[:5])

avg_pois = np.mean([len(p) for p in sensor_pois])
print(f"  Avg POIs per sensor: {avg_pois:.1f}")

# === 5. Write to Neo4j ===
print("\n--- Writing to Neo4j ---")
driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()

with driver.session() as session:
    # Remove old POI data if any
    session.run("MATCH (n:POI) DETACH DELETE n")

    for idx, row in sensors.iterrows():
        sid = int(row['sensor_id'])
        session.run("MATCH (s:Sensor {sid: $sid}) SET s.road_type = $rt",
                    sid=sid, rt=row['road_type'])
    print(f"  Set road_type for {n} sensors")

    poi_nodes, rel_edges = 0, 0
    seen_poi = set()
    for i, row in sensors.iterrows():
        sid = int(row['sensor_id'])
        for name, poi_type, dist in sensor_pois[i]:
            key = (name, poi_type)
            session.run("""
                MATCH (s:Sensor {sid: $sid})
                MERGE (p:POI {name: $name, type: $type})
                MERGE (s)-[:NEAR {distance_m: $dist}]->(p)
            """, sid=sid, name=name, type=poi_type, dist=dist)
            rel_edges += 1
            if key not in seen_poi:
                seen_poi.add(key)
                poi_nodes += 1

    print(f"  {poi_nodes} unique POI nodes, {rel_edges} NEAR edges")

    # Final stats
    nodes_c = session.run("MATCH (n) RETURN count(n)").single().value()
    edges_c = session.run("MATCH ()-[r]->() RETURN count(r)").single().value()
    labels = [r[0] for r in session.run("CALL db.labels()").values()]
    rels = [r[0] for r in session.run("CALL db.relationshipTypes()").values()]
    print(f"\n  Neo4j: {nodes_c} nodes, {edges_c} edges, labels={labels}, relations={rels}")

driver.close()
print("\nDone!")
