from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_AUTH
d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
d.verify_connectivity()
with d.session() as s:
    r = s.run('CALL db.labels()')
    print('Labels:', [x['label'] for x in r])
    r = s.run('CALL db.relationshipTypes()')
    print('Relations:', [x['relationshipType'] for x in r])
    r = s.run('MATCH (n) RETURN count(n) as c')
    print('Nodes:', r.single()['c'])
    r = s.run('MATCH ()-[r]->() RETURN count(r) as c')
    print('Edges:', r.single()['c'])
    r = s.run('MATCH (s:Sensor) RETURN s LIMIT 2')
    for x in r:
        print('Sensor:', dict(x['s']))
d.close()
