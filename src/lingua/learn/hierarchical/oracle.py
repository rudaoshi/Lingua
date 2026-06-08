from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from lingua.utils.normalization import normalize_lingua_edges_to_predicate_nodes


@dataclass(frozen=True)
class TokenLabel:
    token_id: int
    token: str
    labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "token": self.token,
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class HierarchicalOracleChild:
    role: str
    edge_label: str
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    unit_kind: str
    incoming_count: int
    is_external: bool
    is_discontinuous: bool

    @property
    def is_shared(self) -> bool:
        return self.incoming_count > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "edge_label": self.edge_label,
            "token_ids": list(self.token_ids),
            "tokens": list(self.tokens),
            "text": _tokens_text(self.tokens),
            "unit_kind": self.unit_kind,
            "incoming_count": self.incoming_count,
            "is_shared": self.is_shared,
            "is_external": self.is_external,
            "is_discontinuous": self.is_discontinuous,
        }


@dataclass(frozen=True)
class HierarchicalOracleUnit:
    input_token_ids: tuple[int, ...]
    input_tokens: tuple[str, ...]
    unit_kind: str
    depth: int
    incoming_count: int
    is_root: bool
    self_token_ids: tuple[int, ...]
    children: tuple[HierarchicalOracleChild, ...]
    token_labels: tuple[TokenLabel, ...]

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def has_shared_child(self) -> bool:
        return any(child.is_shared for child in self.children)

    @property
    def has_external_child(self) -> bool:
        return any(child.is_external for child in self.children)

    @property
    def has_overlapping_roles(self) -> bool:
        return any(len(token_label.labels) > 1 for token_label in self.token_labels)

    def to_dict(self) -> dict[str, Any]:
        shared_links = [child.to_dict() for child in self.children if child.is_shared]
        external_links = [child.to_dict() for child in self.children if child.is_external]
        return {
            "input_token_ids": list(self.input_token_ids),
            "input_tokens": list(self.input_tokens),
            "input_text": _tokens_text(self.input_tokens),
            "unit_kind": self.unit_kind,
            "depth": self.depth,
            "incoming_count": self.incoming_count,
            "is_root": self.is_root,
            "decision": "leaf" if self.is_leaf else "expand",
            "self_token_ids": list(self.self_token_ids),
            "is_discontinuous": _is_discontinuous(self.input_token_ids),
            "has_shared_child": self.has_shared_child,
            "has_external_child": self.has_external_child,
            "has_overlapping_roles": self.has_overlapping_roles,
            "token_labels": [token_label.to_dict() for token_label in self.token_labels],
            "children": [child.to_dict() for child in self.children],
            "shared_links": shared_links,
            "external_links": external_links,
        }


@dataclass(frozen=True)
class HierarchicalOracleExample:
    sentence: str | None
    words: tuple[str, ...]
    root_token_sets: tuple[tuple[int, ...], ...]
    units: tuple[HierarchicalOracleUnit, ...]
    normalization_stats: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence": self.sentence,
            "words": list(self.words),
            "root_token_sets": [list(token_ids) for token_ids in self.root_token_sets],
            "unit_count": len(self.units),
            "edge_count": sum(len(unit.children) for unit in self.units),
            "normalization_stats": dict(self.normalization_stats),
            "units": [unit.to_dict() for unit in self.units],
        }


def extract_hierarchical_oracle(sample: dict[str, Any]) -> HierarchicalOracleExample:
    """Extract recursive token-set labeling targets from a gold Lingua graph.

    Each unit is represented by the token set covered by `DESC(v)`. The target for
    one layer is a unit-kind label plus BIO role labels over that token set. Gold
    graph node indices are used only internally to recover the layer structure.
    """

    lingua, sentence = _coerce_lingua_sample(sample)
    lingua, normalization_stats = normalize_lingua_edges_to_predicate_nodes(lingua)
    words = tuple(lingua.get("words", []))
    graph = lingua["graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]

    incoming_count, outgoing = _index_edges(nodes, edges)
    roots = tuple(index for index, count in enumerate(incoming_count) if count == 0)
    topological_order, depths = _topological_order(nodes, edges, incoming_count)
    self_token_ids = tuple(_token_indices(node["spans"]) for node in nodes)
    desc_token_ids = _descendant_token_ids(len(nodes), outgoing, self_token_ids, topological_order)

    units = []
    for node_index in topological_order:
        input_token_ids = desc_token_ids[node_index]
        input_tokens = _tokens_for_ids(words, input_token_ids)
        children = tuple(
            _child_from_edge(
                edge=edge,
                words=words,
                nodes=nodes,
                incoming_count=incoming_count,
                input_token_ids=input_token_ids,
                desc_token_ids=desc_token_ids,
            )
            for _, edge in outgoing[node_index]
        )
        token_labels = _token_labels(
            words=words,
            input_token_ids=input_token_ids,
            self_token_ids=self_token_ids[node_index],
            children=children,
        )
        units.append(
            HierarchicalOracleUnit(
                input_token_ids=input_token_ids,
                input_tokens=input_tokens,
                unit_kind=_unit_kind(nodes[node_index]),
                depth=depths[node_index],
                incoming_count=incoming_count[node_index],
                is_root=node_index in roots,
                self_token_ids=self_token_ids[node_index],
                children=children,
                token_labels=token_labels,
            )
        )

    return HierarchicalOracleExample(
        sentence=sentence,
        words=words,
        root_token_sets=tuple(desc_token_ids[index] for index in roots),
        units=tuple(units),
        normalization_stats={str(key): int(value) for key, value in normalization_stats.items()},
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


def _descendant_token_ids(
    node_count: int,
    outgoing: dict[int, list[tuple[int, dict[str, Any]]]],
    self_token_ids: tuple[tuple[int, ...], ...],
    topological_order: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    desc_sets = [set(tokens) for tokens in self_token_ids]
    for node_index in reversed(topological_order):
        for _, edge in outgoing[node_index]:
            desc_sets[node_index].update(desc_sets[int(edge["end"])])
    return tuple(tuple(sorted(desc_sets[index])) for index in range(node_count))


def _child_from_edge(
    edge: dict[str, Any],
    words: tuple[str, ...],
    nodes: list[dict[str, Any]],
    incoming_count: list[int],
    input_token_ids: tuple[int, ...],
    desc_token_ids: tuple[tuple[int, ...], ...],
) -> HierarchicalOracleChild:
    child_index = int(edge["end"])
    token_ids = desc_token_ids[child_index]
    input_token_set = set(input_token_ids)
    return HierarchicalOracleChild(
        role=_role_from_edge_label(edge["label"]),
        edge_label=edge["label"],
        token_ids=token_ids,
        tokens=_tokens_for_ids(words, token_ids),
        unit_kind=_unit_kind(nodes[child_index]),
        incoming_count=incoming_count[child_index],
        is_external=any(token_id not in input_token_set for token_id in token_ids),
        is_discontinuous=_is_discontinuous(token_ids),
    )


def _token_labels(
    words: tuple[str, ...],
    input_token_ids: tuple[int, ...],
    self_token_ids: tuple[int, ...],
    children: tuple[HierarchicalOracleChild, ...],
) -> tuple[TokenLabel, ...]:
    labels_by_token: dict[int, list[str]] = defaultdict(list)
    input_token_set = set(input_token_ids)

    _add_bio_labels(labels_by_token, self_token_ids, input_token_set, "HEAD")
    for child in children:
        _add_bio_labels(labels_by_token, child.token_ids, input_token_set, child.role)

    return tuple(
        TokenLabel(
            token_id=token_id,
            token=words[token_id] if token_id < len(words) else "",
            labels=tuple(sorted(set(labels_by_token.get(token_id, ["O"])))),
        )
        for token_id in input_token_ids
    )


def _add_bio_labels(
    labels_by_token: dict[int, list[str]],
    token_ids: tuple[int, ...],
    input_token_set: set[int],
    role: str,
) -> None:
    previous_token_id = None
    for token_id in sorted(token_id for token_id in token_ids if token_id in input_token_set):
        prefix = "I" if previous_token_id is not None and token_id == previous_token_id + 1 else "B"
        labels_by_token[token_id].append(f"{prefix}-{role}")
        previous_token_id = token_id


def _unit_kind(node: dict[str, Any]) -> str:
    spans = node["spans"]
    if isinstance(spans, str):
        return spans
    return node["type"]


def _role_from_edge_label(edge_label: str) -> str:
    if edge_label.startswith("as:"):
        return "AS_" + _role_from_edge_label(edge_label[3:])

    explicit_roles = {
        "pred.arg.1": "ARG1",
        "pred.arg.2": "ARG2",
        "pred.arg.3": "ARG3",
        "pred.arg.4": "ARG4",
        "func.arg": "FUNC_ARG",
    }
    if edge_label in explicit_roles:
        return explicit_roles[edge_label]

    return (
        edge_label.replace(".", "_")
        .replace(":", "_")
        .replace("-", "_")
        .upper()
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


def _tokens_for_ids(words: tuple[str, ...], token_ids: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(words[token_id] if token_id < len(words) else "" for token_id in token_ids)


def _tokens_text(tokens: tuple[str, ...]) -> str:
    return " ".join(token.strip() for token in tokens if token.strip())


def _is_discontinuous(token_ids: tuple[int, ...]) -> bool:
    if len(token_ids) <= 1:
        return False
    sorted_ids = sorted(token_ids)
    return any(right != left + 1 for left, right in zip(sorted_ids, sorted_ids[1:]))
