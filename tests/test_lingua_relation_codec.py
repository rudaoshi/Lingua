from collections import Counter

from lingua.learn.wordgraph.modeler.lingua_relation_codec import (
    decode_lingua_relation_instances,
    encode_lingua_relation_instances,
    relation_instance_signature,
    relation_instances,
)
from lingua.utils.normalization import normalize_lingua_edges_to_predicate_nodes


def canonical_graph(graph):
    nodes = tuple(
        (repr(node["spans"]), node["type"], node.get("confidence"))
        for node in graph["graph"]["nodes"]
    )
    edges = Counter(
        (
            int(edge["start"]),
            int(edge["end"]),
            edge["label"],
            edge.get("confidence"),
        )
        for edge in graph["graph"]["edges"]
    )
    return nodes, edges


def test_relation_codec_roundtrips_reified_direct_modification():
    graph = {
        "meta": {},
        "words": ["red", "chair"],
        "graph": {
            "nodes": [
                {"spans": [[0, 0]], "type": "ModificationalConstant"},
                {"spans": [[1, 1]], "type": "NominalConstant"},
            ],
            "edges": [
                {"start": 0, "end": 1, "label": "as:modification"},
            ],
        },
    }

    normalized, _ = normalize_lingua_edges_to_predicate_nodes(graph)
    encoded, stats = encode_lingua_relation_instances(graph)
    decoded = decode_lingua_relation_instances(encoded)

    assert stats["reified_relation_edges:as:modification"] == 1
    assert relation_instances(encoded) == [
        {
            "id": 2,
            "kind": "relation",
            "label": "Modification",
            "type": "ModificationalFunctor",
            "roles": [
                {"label": "variable", "target": 1},
                {"label": "body", "target": 0},
            ],
        }
    ]
    assert canonical_graph(decoded) == canonical_graph(normalized)


def test_relation_codec_keeps_same_label_aux_nodes_as_distinct_instances():
    graph = {
        "meta": {},
        "words": ["very", "red", "chair"],
        "graph": {
            "nodes": [
                {"spans": [[0, 0]], "type": "ModificationalConstant"},
                {"spans": [[1, 1]], "type": "ModificationalConstant"},
                {"spans": [[2, 2]], "type": "NominalConstant"},
                {"spans": "Modification", "type": "ModificationalFunctor"},
                {"spans": "Modification", "type": "ModificationalFunctor"},
            ],
            "edges": [
                {"start": 3, "end": 1, "label": "variable"},
                {"start": 3, "end": 0, "label": "body"},
                {"start": 4, "end": 2, "label": "variable"},
                {"start": 4, "end": 3, "label": "body"},
            ],
        },
    }

    encoded, stats = encode_lingua_relation_instances(graph)
    decoded = decode_lingua_relation_instances(encoded)
    signatures = [relation_instance_signature(item) for item in relation_instances(encoded)]

    assert stats == {}
    assert len(signatures) == 2
    assert len(set(signatures)) == 2
    assert signatures == [
        ("Modification", "ModificationalFunctor", (("body", 0), ("variable", 1))),
        ("Modification", "ModificationalFunctor", (("body", 3), ("variable", 2))),
    ]
    assert canonical_graph(decoded) == canonical_graph(graph)


def test_relation_codec_preserves_plain_edges_from_lexical_relation_nodes():
    graph = {
        "meta": {},
        "words": ["near", "door", "chair"],
        "graph": {
            "nodes": [
                {"spans": [[0, 0]], "type": "ModificationalFunctor"},
                {"spans": [[1, 1]], "type": "NominalConstant"},
                {"spans": [[2, 2]], "type": "NominalConstant"},
            ],
            "edges": [
                {"start": 0, "end": 2, "label": "variable"},
                {"start": 0, "end": 1, "label": "body"},
            ],
        },
    }

    encoded, stats = encode_lingua_relation_instances(graph)
    decoded = decode_lingua_relation_instances(encoded)

    assert stats == {}
    assert relation_instances(encoded) == []
    assert encoded["plain_edges"] == graph["graph"]["edges"]
    assert canonical_graph(decoded) == canonical_graph(graph)
