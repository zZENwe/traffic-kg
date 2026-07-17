"""Export Neo4j sensor graph as image."""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_AUTH

URI = NEO4J_URI
AUTH = NEO4J_AUTH

driver = GraphDatabase.driver(URI, auth=AUTH)

with driver.session() as session:
    # Fetch all nodes
    nodes_r = session.run("MATCH (s:Sensor) RETURN s.sid AS sid, s.lat AS lat, s.lon AS lon")
    nodes = [(r['sid'], r['lat'], r['lon']) for r in nodes_r]

    # Fetch all edges
    edges_r = session.run("MATCH (a:Sensor)-[r:ROAD_DISTANCE]->(b:Sensor) RETURN a.sid AS a, b.sid AS b, r.km AS km")
    edges = [(r['a'], r['b'], r['km']) for r in edges_r]

driver.close()

# Build graph
G = nx.DiGraph()
for sid, lat, lon in nodes:
    G.add_node(sid, pos=(lon, lat))
for a, b, km in edges:
    G.add_edge(a, b, weight=km)

pos = {sid: (lon, lat) for sid, lat, lon in nodes}

# Full view
fig, ax = plt.subplots(figsize=(20, 14))
nx.draw(G, pos, ax=ax, node_size=20, node_color='#2196F3', arrows=False,
        width=0.1, alpha=0.5, edge_color='#999999')
ax.set_title(f'METR-LA Sensor Graph: {len(nodes)} Sensors, {len(edges)} Road Distances', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/kg_full.png', dpi=150, bbox_inches='tight')
print(f"Full graph saved: {len(nodes)} nodes, {len(edges)} edges")

# Zoom-in view (downtown LA area)
zoom_nodes = [sid for sid, lat, lon in nodes if 33.9 < lat < 34.2 and -118.5 < lon < -118.1]
if zoom_nodes:
    H = G.subgraph(zoom_nodes)
    zpos = {n: pos[n] for n in H.nodes()}
    fig2, ax2 = plt.subplots(figsize=(16, 12))
    nx.draw(H, zpos, ax=ax2, node_size=50, node_color='#FF5722', arrows=False,
            width=0.3, alpha=0.7, edge_color='#666666')
    edge_labels = {(u, v): f'{d["weight"]:.1f}' for u, v, d in H.edges(data=True)}
    # Only label a few edges
    sample_edges = {k: v for i, (k, v) in enumerate(edge_labels.items()) if i % 20 == 0}
    nx.draw_networkx_edge_labels(H, zpos, edge_labels=sample_edges, font_size=5, alpha=0.5, ax=ax2)
    ax2.set_title(f'Downtown LA Zoom: {len(zoom_nodes)} Sensors', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/kg_zoom.png', dpi=150, bbox_inches='tight')
    print(f"Zoom graph saved: {len(zoom_nodes)} nodes")

plt.close('all')
print("Done!")
