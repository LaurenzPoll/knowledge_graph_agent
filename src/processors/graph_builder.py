import networkx as nx
from typing import List, Dict, Tuple

def build_graph(
    triples: List[Dict[str, str]],
) -> Tuple[nx.MultiDiGraph, List[Dict]]:
    """
    Build a NetworkX knowledge graph and a Cytoscape-compatible elements list,
    computing separate spring layouts per connected component and offsetting
    each cluster so they are visually distinct
    """
    # Extract subject and objects and create an union
    subjects = {triple["subject"] for triple in triples}
    objects  = {triple["object"]  for triple in triples}
    all_entities = subjects.union(objects)

    # assign every node a default type | Later this can be used to assign different types to make the graph more readable
    entity_types = {ent: "Unknown" for ent in all_entities}

    # map each predicate to itself
    relation_types = {triple["predicate"]: triple["predicate"] for triple in triples}

    # building the graph
    G = nx.MultiDiGraph()
    for ent, t in entity_types.items():
        G.add_node(ent, label=ent, type=t)
    for triple in triples:
        subject, object, predicate = triple["subject"], triple["object"], triple["predicate"]
        relation_type = relation_types.get(predicate, predicate)
        G.add_edge(subject, object, label=relation_type, relation_type=relation_type)


    # build cytoscape elements
    elements: List[Dict] = []
    for node, data in G.nodes(data=True):
        elements.append({
            "data": {"id": node, "label": data["label"], "type": data["type"]},
        })
    for u, v, data in G.edges(data=True):
        elements.append({
            "data": {
                "source": u,
                "target": v,
                "label": data["label"],
                "relation_type": data["relation_type"]
            }
        })

    return G, elements
