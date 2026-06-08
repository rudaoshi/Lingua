"""Relation-instance codec for Lingua graphs with explicit aux nodes.

This module is deliberately not a generative decoder.  It turns a Lingua graph
into a deterministic collection of lexical/root nodes plus relation instances,
where an aux node's identity is its relation label and role endpoints.  The
encoded form is suitable as a target view for biaffine relation-instance
scoring, and it can be materialized back into a Lingua JSON graph.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from lingua.utils.normalization import normalize_lingua_edges_to_predicate_nodes


GraphData = dict[str, Any]
EncodedGraph = dict[str, Any]


def is_relation_item(item: dict[str, Any]) -> bool:
    return item.get("kind") == "relation"


def is_aux_node(node: dict[str, Any]) -> bool:
    return isinstance(node.get("spans"), str)


def encode_lingua_relation_instances(
    graph_data: GraphData,
    *,
    reify_direct_edges: bool = True,
) -> tuple[EncodedGraph, dict[str, int]]:
    """Encode Lingua JSON as lexical nodes plus relation instances.

    If ``reify_direct_edges`` is true, direct non-core edges such as
    ``as:modification`` are first converted to explicit aux relation nodes.
    Existing aux nodes are then encoded as relation instances with their
    outgoing role edges.
    """

    if reify_direct_edges:
        working, stats = normalize_lingua_edges_to_predicate_nodes(graph_data)
    else:
        working = copy.deepcopy(graph_data)
        stats = {}

    graph = working["graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]

    outgoing: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        outgoing[int(edge["start"])].append(edge)

    items: list[dict[str, Any]] = []
    plain_edges: list[dict[str, Any]] = []

    for node_id, node in enumerate(nodes):
        if is_aux_node(node):
            relation = {
                "id": node_id,
                "kind": "relation",
                "label": node["spans"],
                "type": node["type"],
                "roles": [
                    _encode_role_edge(edge)
                    for edge in outgoing.get(node_id, [])
                ],
            }
            if "confidence" in node:
                relation["confidence"] = node["confidence"]
            items.append(relation)
        else:
            lexical = {
                "id": node_id,
                "kind": "lexical",
                "spans": copy.deepcopy(node["spans"]),
                "type": node["type"],
            }
            if "confidence" in node:
                lexical["confidence"] = node["confidence"]
            items.append(lexical)

    for edge in edges:
        if is_aux_node(nodes[int(edge["start"])]):
            continue
        plain_edges.append(copy.deepcopy(edge))

    encoded = {
        "meta": copy.deepcopy(working.get("meta", {})),
        "words": copy.deepcopy(working.get("words", [])),
        "items": items,
        "plain_edges": plain_edges,
    }
    if stats:
        encoded["normalization_stats"] = dict(stats)

    return encoded, dict(stats)


def decode_lingua_relation_instances(encoded: EncodedGraph) -> GraphData:
    """Materialize an encoded relation-instance graph back to Lingua JSON."""

    items = sorted(encoded["items"], key=lambda item: int(item["id"]))
    expected_ids = list(range(len(items)))
    actual_ids = [int(item["id"]) for item in items]
    if actual_ids != expected_ids:
        raise ValueError(f"encoded item ids must be contiguous from 0: {actual_ids[:20]}")

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for item in items:
        if is_relation_item(item):
            node = {
                "spans": item["label"],
                "type": item["type"],
            }
        else:
            node = {
                "spans": copy.deepcopy(item["spans"]),
                "type": item["type"],
            }
        if "confidence" in item:
            node["confidence"] = item["confidence"]
        nodes.append(node)

    for edge in encoded.get("plain_edges", []):
        edges.append(copy.deepcopy(edge))

    for item in items:
        if not is_relation_item(item):
            continue
        source = int(item["id"])
        for role in item.get("roles", []):
            edge = {
                "start": source,
                "end": int(role["target"]),
                "label": role["label"],
            }
            if "confidence" in role:
                edge["confidence"] = role["confidence"]
            edges.append(edge)

    return {
        "meta": copy.deepcopy(encoded.get("meta", {})),
        "words": copy.deepcopy(encoded.get("words", [])),
        "graph": {
            "nodes": nodes,
            "edges": edges,
        },
    }


def relation_instances(encoded: EncodedGraph) -> list[dict[str, Any]]:
    """Return relation items in node order."""

    return [
        item
        for item in sorted(encoded["items"], key=lambda item: int(item["id"]))
        if is_relation_item(item)
    ]


def relation_instance_signature(instance: dict[str, Any]) -> tuple[Any, ...]:
    """Permutation-stable identity for a relation instance.

    This is the unit a biaffine relation-instance classifier should learn:
    relation label plus role endpoints, not the arbitrary JSON node id.
    """

    roles = tuple(
        sorted(
            (role["label"], int(role["target"]))
            for role in instance.get("roles", [])
        )
    )
    return instance["label"], instance["type"], roles


def _encode_role_edge(edge: dict[str, Any]) -> dict[str, Any]:
    role = {
        "label": edge["label"],
        "target": int(edge["end"]),
    }
    if "confidence" in edge:
        role["confidence"] = edge["confidence"]
    return role
