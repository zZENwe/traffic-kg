"""Fill missing POIs for sensors. Uses sub-region batch queries."""
import osmnx as ox
import pandas as pd
import numpy as np
import time
from neo4j import GraphDatabase
from shapely.geometry import Point
from config import NEO4J_URI, NEO4J_AUTH

ox.settings.use_cache = True
ox.settings.log_console = False

URI = NEO4J_URI
AUTH = NEO4J_AUTH

sensors = pd.read_csv("data/sensor_graph/graph_sensor_locations.csv", index_col=0)
lats, lons = sensors['latitude'].values, sensors['longitude'].values

# Read missing SIDs
with open('_missing_sids.txt', 'r') as f:
    missing_sids = set(int(x) for x in f.read().split(','))
print(f"Missing sensors: {len(missing_sids)}")

# Find which sensors need POIs
missing_indices = [i for i in range(len(sensors))
                   if int(sensors.iloc[i]['sensor_id']) in missing_sids]
missing_lats = lats[missing_indices]
missing_lons = lons[missing_indices]
print(f"Will query {len(missing_indices)} sensors")

# Generate 12 sub-region centers covering the sensor area
# Use a 4x3 grid to cover the full bounding box
grid_lats = np.linspace(lats.min(), lats.max(), 4)
grid_lons = np.linspace(lons.min(), lons.max(), 3)
sub_centers = []
for glat in grid_lats:
    for glon in grid_lons:
        sub_centers.append((glat, glon))

print(f"\nDownloading POIs from {len(sub_centers)} sub-regions (4km radius)...")
all_pois = []
for j, sc in enumerate(sub_centers):
    t0 = time.time()
    try:
        pois = ox.features_from_point(sc, dist=4000, tags={
            'amenity': True, 'shop': True, 'leisure': True, 'tourism': True,
            'highway': ['motorway_junction'],
        })
        if pois is not None and len(pois) > 0:
            valid = pois[pois.geometry.notna()].copy()
            all_pois.append(valid)
            print(f"  {j+1}/{len(sub_centers)} ({sc[0]:.2f},{sc[1]:.2f}): {len(valid)} POIs in {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"  {j+1}/{len(sub_centers)}: FAILED ({e})")

print(f"\nTotal: {sum(len(p) for p in all_pois)} POIs")

# Deduplicate
seen = set()
unique_pois = []
for pois_df in all_pois:
    for _, poi in pois_df.iterrows():
        tags = poi.to_dict()
        name = tags.get('name', '')
        if not isinstance(name, str) or not name or name == 'nan':
            continue
        poi_type = str(tags.get('amenity') or tags.get('shop') or
                      tags.get('leisure') or tags.get('tourism') or
                      tags.get('highway') or 'place')
        key = (name.strip(), poi_type)
        if key not in seen:
            seen.add(key)
            unique_pois.append((name.strip(), poi_type, poi.geometry))
print(f"  {len(unique_pois)} unique POIs")

# Match only to sensors that need POIs
print(f"\nMatching to {len(missing_indices)} missing sensors...")
pts = [Point(lon, lat) for lat, lon in zip(missing_lats, missing_lons)]
sensor_pois = {}
for i, pt in enumerate(pts):
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(missing_indices)}...")
    nearby = []
    for name, ptype, geom in unique_pois:
        d = pt.distance(geom) * 111000
        if d < 500:
            nearby.append((name, ptype, round(d, 0)))
    nearby.sort(key=lambda x: x[2])
    sid = int(sensors.iloc[missing_indices[i]]['sensor_id'])
    sensor_pois[sid] = nearby[:5]

avg = np.mean([len(v) for v in sensor_pois.values()])
print(f"  Avg POIs/sensor: {avg:.1f}")

# Write to Neo4j
print("\nWriting to Neo4j...")
driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()

with driver.session() as session:
    new_pois, new_rels = 0, 0
    for sid, pois in sensor_pois.items():
        for name, ptype, dist in pois:
            new_rels += 1
            session.run("""
                MATCH (s:Sensor {sid: $sid})
                MERGE (p:POI {name: $name, type: $type})
                MERGE (s)-[:NEAR {distance_m: $dist}]->(p)
            """, sid=sid, name=name, type=ptype, dist=dist)
            new_pois += 1

    # Count final state
    nodes = session.run("MATCH (n) RETURN count(n)").single().value()
    edges = session.run("MATCH ()-[r]->() RETURN count(r)").single().value()
    poi_total = session.run("MATCH (n:POI) RETURN count(n)").single().value()
    covered = session.run("MATCH (s:Sensor)-[:NEAR]->(:POI) RETURN count(DISTINCT s)").single().value()
    print(f"  Added {new_pois} relationships")
    print(f"  Final: {nodes} nodes, {edges} edges, {poi_total} POIs, {covered}/207 sensors covered")

driver.close()
print("Done!")
