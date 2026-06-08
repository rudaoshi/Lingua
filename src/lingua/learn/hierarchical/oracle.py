from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NodeRef:
    node_id: int
    text: str
    node_type: str
    spans: Any
    token_indices: tuple[int, ...]
    is_aux: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "text": self.text,
            "type": self.node_type,
            "spans": self.spans,
            "token_indices": list(self.token_indices),
            "is_aux": self.is_aux,
        }


@dataclass(frozen=True)
class HierarchicalOracleChild:
    edge_index: int
    label: str
    child: NodeRef
    child_incoming_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_index": self.edge_index,
            "label": self.label,
            "child": self.child.to_dict(),
            "child_incoming_count": self.child_incoming_count,
            "child_is_shared": self.child_incoming_count > 1,
        }


@dataclass(frozen=True)
class HierarchicalOracleUnit:
    node: NodeRef
    depth: int
    incoming_count: int
    is_root: bool
    descendant_count: int
    children: tuple[HierarchicalOracleChild, ...]

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def has_shared_child(self) -> bool:
        return any(child.child_incoming_count > 1 for child in self.children)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node.to_dict(),
            "depth": self.depth,
            "incoming_count": self.incoming_count,
            "is_root": self.is_root,
            "decision": "leaf" if self.is_leaf else "expand",
            "descendant_count": self.descendant_count,
            "has_shared_child": self.has_shared_child,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True)
class HierarchicalOracleExample:
    sentence: str | None
    words: tuple[str, ...]
    root_ids: tuple[int, ...]
    units: tuple[HierarchicalOracleUnit, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence": self.sentence,
            "words": list(self.words),
            "root_ids": list(self.root_ids),
            "node_count": len(self.units),
            "edge_count": sum(len(unit.children) for unit in self.units),
            "units": [unit.to_dict() for unit in self.units],
        }


def extract_hierarchical_oracle(sample: dict[str, Any]) -> HierarchicalOracleExample:
    """Extract one-layer recursive expansion targets from a gold Lingua graph.

    The oracle intentionally preserves the graph as a DAG: child nodes are
    referenced by id and marked as shared when they have multiple incoming edges.
    That keeps the representation useful for Lingua's compact `DESC(v)` style
    instead of forcing a tree-shaped constituency target.
    """

    lingua, sentence = _coerce_lingua_sample(sample)
    words = tuple(lingua.get("words", []))
    graph = lingua["graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]

    incoming_count, outgoing = _index_edges(nodes, edges)
    roots = tuple(index for index, count in enumerate(incoming_count) if count == 0)
    topological_order, depths = _topological_order(nodes, edges, incoming_count)
    node_refs = tuple(_node_ref(index, node, words) for index, node in enumerate(nodes))
    descendant_counts = _descendant_counts(len(nodes), outgoing)

    units = []
    for node_index in topological_order:
        children = tuple(
            HierarchicalOracleChild(
                edge_index=edge_index,
                label=edge["label"],
                child=node_refs[int(edge["end"])],
                child_incoming_count=incoming_count[int(edge["end"])],
            )
            for edge_index, edge in outgoing[node_index]
        )
        units.append(
            HierarchicalOracleUnit(
                node=node_refs[node_index],
                depth=depths[node_index],
                incoming_count=incoming_count[node_index],
                is_root=node_index in roots,
                descendant_count=descendant_counts[node_index],
                children=children,
            )
        )

    return HierarchicalOracleExample(
        sentence=sentence,
        words=words,
        root_ids=roots,
        units=tuple(units),
    )


def _coerce_lingua_sample(sample: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    if "lingua" in sample:
        return sample["lingua"], sample.get("sentence")
    if "graph" in sample and "words" in sample:
        return sample, sample.get("sentence")
    raise ValueError("Expected a dataset sample with `lingua`, or a raw Lingua object")


def _index_edges(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[list[int], dict[int, list[tuple[int, dict[str, Any]]]]]:
    incoming_count = [0] * len(nodes)
    outgoing: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for edge_index, edge in enumerate(edges):
        start = int(edge["start"])
        end = int(edge["end"])
        if start < 0 or start >= len(nodes) or end < 0 or end >= len(nodes):
            raise ValueError(f"Edge {edge_index} points outside node range: {edge}")
        incoming_count[end] += 1
        outgoing[start].append((edge_index, edge))

    return incoming_count, outgoing


def _topological_order(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    incoming_count: list[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    remaining_incoming = list(incoming_count)
    children_by_parent: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        children_by_parent[int(edge["start"])].append(int(edge["end"]))

    ready = deque(index for index, count in enumerate(remaining_incoming) if count == 0)
    order = []
    depths = [0] * len(nodes)

    while ready:
        node_index = ready.popleft()
        order.append(node_index)
        for child_index in children_by_parent[node_index]:
            depths[child_index] = max(depths[child_index], depths[node_index] + 1)
            remaining_incoming[child_index] -= 1
            if remaining_incoming[child_index] == 0:
                ready.append(child_index)

    if len(order) != len(nodes):
        raise ValueError("Lingua graph must be acyclic for hierarchical oracle extraction")

    return tuple(order), tuple(depths)


def _descendant_counts(
    node_count: int,
    outgoing: dict[int, list[tuple[int, dict[str, Any]]]],
) -> tuple[int, ...]:
    counts = []
    for start in range(node_count):
        seen = set()
        stack = [int(edge["end"]) for _, edge in outgoing[start]]
        while stack:
            node_index = stack.pop()
            if node_index in seen:
                continue
            seen.add(node_index)
            stack.extend(int(edge["end"]) for _, edge in outgoing[node_index])
        counts.append(len(seen))
    return tuple(counts)


def _node_ref(index: int, node: dict[str, Any], words: tuple[str, ...]) -> NodeRef:
    spans = node["spans"]
    token_indices = _token_indices(spans)
    if token_indices:
        text = " ".join(words[token_index] for token_index in token_indices if token_index < len(words))
    elif isinstance(spans, str):
        text = spans
    else:
        text = ""

    return NodeRef(
        node_id=index,
        text=text,
        node_type=node["type"],
        spans=spans,
        token_indices=tuple(token_indices),
        is_aux=isinstance(spans, str),
    )


def _token_indices(spans: Any) -> tuple[int, ...]:
    if isinstance(spans, str):
        return ()
    token_indices = []
    for span in spans:
        if len(span) != 2:
            raise ValueError(f"Invalid span: {span}")
        start, end = int(span[0]), int(span[1])
        if end < start:
            raise ValueError(f"Invalid reversed span: {span}")
        token_indices.extend(range(start, end + 1))
    return tuple(token_indices)
