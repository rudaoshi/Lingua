"""Canonical graph normalizations used by evaluation scripts."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any


GraphData = dict[str, Any]

CORE_EDGE_LABELS = {"func.arg", "punctuation", "repeat", "variable", "body"}
RELATION_EDGE_TO_AUX_NODE = {
    "appositive": ("Apposition", "AppositionalPredicator"),
    "attribute": ("Attribute", "AttributePredicator"),
    "discourse": ("Discourse", "ModificationalFunctor"),
    "index": ("Index", "OtherConstant"),
    "modification": ("Modification", "ModificationalFunctor"),
    "nonsense": ("Nonsense", "OtherConstant"),
    "punctuation": ("Punctuation", "PunctuationalConstant"),
    "ref": ("Ref", "ReferentialPredicator"),
    "repeat": ("Repeat", "OtherConstant"),
    "vocative": ("Vocative", "ModificationalFunctor"),
}
EXPRESSION_LIKE_AUX_LABELS = {"Expression", "List", "Parataxis"}
AS_RELATION_PRIORITIES = {
    "appositive": 41,
    "discourse": 42,
    "ref": 43,
    "repeat": 44,
    "vocative": 45,
    "nonsense": 46,
    "parataxis": 47,
    "index": 48,
    "attribute": 49,
    "modification": 50,
    "punctuation": 51,
}


def is_core_predicate_edge(label: str) -> bool:
    return label in CORE_EDGE_LABELS or label.startswith("pred.arg.")


def normalize_lingua_edges_to_predicate_nodes(
    graph_data: GraphData,
    *,
    rewire_as_relation_roots: bool = True,
) -> tuple[GraphData, dict[str, int]]:
    """Reify Lingua non-core edges as explicit auxiliary structures.

    The normalized graph keeps ``pred.arg.*``, ``func.arg``, ``variable``,
    ``body``, ``punctuation``, and ``repeat`` edge labels.  Only ``as:*``
    relation edges, such as ``as:modification`` or ``as:punctuation``, are
    replaced by an auxiliary relation node whose arguments are connected with
    the role labels for that relation type.

    For ``as:X`` edges, the arguments follow the semantic direction of ``X``:
    ``A --as:X--> B`` becomes ``X(role.1=B, role.2=A)``.
    """

    normalized = copy.deepcopy(graph_data)
    graph = normalized["graph"]
    original_nodes = graph["nodes"]
    original_edges = graph["edges"]
    original_node_count = len(original_nodes)

    stats: Counter[str] = Counter()
    relation_nodes: dict[int, int] = {}
    incoming_as_edges: defaultdict[int, list[int]] = defaultdict(list)
    outgoing_as_edges: defaultdict[int, list[int]] = defaultdict(list)
    func_arg_parents_by_child: defaultdict[int, list[int]] = defaultdict(list)

    for edge_index, edge in enumerate(original_edges):
        label = str(edge["label"])
        if label == "func.arg":
            func_arg_parents_by_child[int(edge["end"])].append(int(edge["start"]))
        if is_core_predicate_edge(label) or not label.startswith("as:"):
            continue

        base_label = label[3:]
        node_spans, node_type = relation_edge_to_aux_node(base_label)
        relation_node: dict[str, Any] = {"spans": node_spans, "type": node_type}
        if "confidence" in edge:
            relation_node["confidence"] = edge["confidence"]
        relation_nodes[edge_index] = original_node_count + len(relation_nodes)
        outgoing_as_edges[int(edge["start"])].append(edge_index)
        incoming_as_edges[int(edge["end"])].append(edge_index)
        graph["nodes"].append(relation_node)
        stats["reified_relation_edges"] += 1
        stats[f"reified_relation_edges:{label}"] += 1

    grouped_as_body_roots, grouped_child_replacements, grouped_parent_root_replacements = (
        _group_ambiguous_as_relation_bodies(
            original_edges,
            relation_nodes,
            incoming_as_edges,
            outgoing_as_edges,
            func_arg_parents_by_child,
            stats,
        )
    )

    root_replacements: dict[int, int] = {}
    if rewire_as_relation_roots:
        for source, edge_indices in outgoing_as_edges.items():
            if len(edge_indices) == 1:
                replacement = _top_as_relation_root(
                    source,
                    original_edges,
                    relation_nodes,
                    outgoing_as_edges,
                )
                if replacement is None:
                    stats["skipped_cyclic_as_relation_root_rewires"] += 1
                    continue
                root_replacements[source] = replacement
                stats["rewired_as_relation_roots"] += 1
                if replacement != relation_nodes[edge_indices[0]]:
                    stats["folded_as_relation_root_rewires"] += 1
            else:
                stats["skipped_ambiguous_as_relation_root_rewires"] += 1
        for parent, replacement in grouped_parent_root_replacements.items():
            existing = root_replacements.get(parent)
            if existing is not None and existing != replacement:
                stats["skipped_conflicting_group_parent_root_rewires"] += 1
                continue
            root_replacements[parent] = replacement
            stats["rewired_group_parent_as_relation_roots"] += 1

    new_edges: list[dict[str, Any]] = []
    for edge_index, edge in enumerate(original_edges):
        label = str(edge["label"])
        if is_core_predicate_edge(label) or not label.startswith("as:"):
            copied_edge = copy.deepcopy(edge)
            start = int(copied_edge["start"])
            end = int(copied_edge["end"])
            copied_edge["end"] = grouped_child_replacements.get(
                (start, end),
                root_replacements.get(end, end),
            )
            new_edges.append(copied_edge)
            continue

        relation_index = relation_nodes[edge_index]
        start = int(edge["start"])
        end = int(edge["end"])
        arg1 = _projected_as_relation_target(
            end,
            edge_index,
            original_edges,
            relation_nodes,
            outgoing_as_edges,
        )
        arg2 = _lower_as_relation_root(
            start,
            edge_index,
            original_edges,
            relation_nodes,
            incoming_as_edges,
            grouped_as_body_roots,
            stats,
        )
        role1, role2 = relation_role_labels(graph["nodes"][relation_index])
        new_edges.append(_predicate_edge(relation_index, arg1, role1, edge))
        new_edges.append(_predicate_edge(relation_index, arg2, role2, edge))

    graph["edges"] = _dedupe_edges(new_edges)
    return normalized, dict(stats)


def _group_ambiguous_as_relation_bodies(
    edges: list[dict[str, Any]],
    relation_nodes: dict[int, int],
    incoming_as_edges: dict[int, list[int]],
    outgoing_as_edges: dict[int, list[int]],
    func_arg_parents_by_child: dict[int, list[int]],
    stats: Counter[str],
) -> tuple[dict[int, int], dict[tuple[int, int], int], dict[int, int]]:
    """Fold coordinated direct modifiers under their shared ``func.arg`` root."""

    grouped_as_body_roots: dict[int, int] = {}
    grouped_child_replacements: dict[tuple[int, int], int] = {}
    grouped_parent_root_replacements: dict[int, int] = {}

    for middle_node, lower_edge_indices in incoming_as_edges.items():
        if len(lower_edge_indices) <= 1:
            continue

        upper_edge_indices = outgoing_as_edges.get(middle_node, [])
        if len(upper_edge_indices) != 1:
            stats["skipped_ambiguous_as_relation_body_group_without_single_upper"] += 1
            continue

        upper_edge_index = upper_edge_indices[0]
        if str(edges[upper_edge_index]["label"]) != "as:modification":
            stats["skipped_ambiguous_as_relation_body_group_non_modification"] += 1
            continue
        if any(str(edges[edge_index]["label"]) != "as:modification" for edge_index in lower_edge_indices):
            stats["skipped_ambiguous_as_relation_body_group_mixed_labels"] += 1
            continue

        lower_sources = [int(edges[edge_index]["start"]) for edge_index in lower_edge_indices]
        common_parents = _common_func_arg_parents(lower_sources, func_arg_parents_by_child)
        if len(common_parents) != 1:
            stats["skipped_ambiguous_as_relation_body_group_without_shared_func_arg_parent"] += 1
            continue

        parent = common_parents[0]
        parent_children = {
            int(edge["end"])
            for edge in edges
            if str(edge["label"]) == "func.arg" and int(edge["start"]) == parent
        }
        if parent_children != set(lower_sources):
            stats["skipped_ambiguous_as_relation_body_group_with_extra_func_arg_children"] += 1
            continue

        top_relation = _top_as_relation_root(
            int(middle_node),
            edges,
            relation_nodes,
            outgoing_as_edges,
        )
        if top_relation is None:
            stats["skipped_ambiguous_as_relation_body_group_without_top_relation"] += 1
            continue

        grouped_as_body_roots[upper_edge_index] = parent
        grouped_parent_root_replacements[parent] = top_relation
        for lower_edge_index in lower_edge_indices:
            lower_source = int(edges[lower_edge_index]["start"])
            grouped_child_replacements[(parent, lower_source)] = relation_nodes[lower_edge_index]

        stats["folded_ambiguous_as_relation_bodies_with_func_arg_parent"] += 1
        stats["rewired_grouped_as_relation_children"] += len(lower_edge_indices)

    return grouped_as_body_roots, grouped_child_replacements, grouped_parent_root_replacements


def _common_func_arg_parents(
    child_indices: list[int],
    func_arg_parents_by_child: dict[int, list[int]],
) -> list[int]:
    if not child_indices:
        return []

    common = set(func_arg_parents_by_child.get(child_indices[0], []))
    for child_index in child_indices[1:]:
        common &= set(func_arg_parents_by_child.get(child_index, []))
    return sorted(common)


def _top_as_relation_root(
    node_index: int,
    edges: list[dict[str, Any]],
    relation_nodes: dict[int, int],
    outgoing_as_edges: dict[int, list[int]],
    seen: set[int] | None = None,
) -> int | None:
    """Return the top relation node for an ``as:*`` chain starting at node."""

    if seen is None:
        seen = set()
    if node_index in seen:
        return None
    seen.add(node_index)

    edge_indices = outgoing_as_edges.get(node_index, [])
    if len(edge_indices) != 1:
        return None

    edge_index = edge_indices[0]
    target = int(edges[edge_index]["end"])
    upper_relation = None
    target_edge_indices = outgoing_as_edges.get(target, [])
    if (
        len(target_edge_indices) == 1
        and _as_relation_priority(edges[target_edge_indices[0]])
        >= _as_relation_priority(edges[edge_index])
    ):
        upper_relation = _top_as_relation_root(
            target,
            edges,
            relation_nodes,
            outgoing_as_edges,
            seen,
        )
    return upper_relation if upper_relation is not None else relation_nodes[edge_index]


def _projected_as_relation_target(
    node_index: int,
    current_edge_index: int,
    edges: list[dict[str, Any]],
    relation_nodes: dict[int, int],
    outgoing_as_edges: dict[int, list[int]],
) -> int:
    """Return a lower-level projection for an ``as:*`` relation target."""

    edge_indices = outgoing_as_edges.get(node_index, [])
    if len(edge_indices) != 1:
        return node_index
    lower_edge_index = edge_indices[0]
    if _as_relation_priority(edges[lower_edge_index]) >= _as_relation_priority(
        edges[current_edge_index]
    ):
        return node_index
    projected = _top_as_relation_root(
        node_index,
        edges,
        relation_nodes,
        outgoing_as_edges,
    )
    return projected if projected is not None else node_index


def _lower_as_relation_root(
    node_index: int,
    current_edge_index: int,
    edges: list[dict[str, Any]],
    relation_nodes: dict[int, int],
    incoming_as_edges: dict[int, list[int]],
    grouped_as_body_roots: dict[int, int],
    stats: Counter[str],
) -> int:
    """Return the lower folded relation root for a node used as relation body."""

    if current_edge_index in grouped_as_body_roots:
        return grouped_as_body_roots[current_edge_index]

    edge_indices = [
        edge_index
        for edge_index in incoming_as_edges.get(node_index, [])
        if edge_index != current_edge_index
        and _as_relation_priority(edges[edge_index])
        <= _as_relation_priority(edges[current_edge_index])
    ]
    if len(edge_indices) == 1:
        stats["folded_as_relation_bodies"] += 1
        return relation_nodes[edge_indices[0]]
    if len(edge_indices) > 1:
        stats["skipped_ambiguous_as_relation_body_folds"] += 1
    return node_index


def _as_relation_priority(edge: dict[str, Any]) -> int:
    label = str(edge["label"])
    base_label = label[3:] if label.startswith("as:") else label
    return AS_RELATION_PRIORITIES.get(base_label, 50)


def standardize_lingua_edges_to_predicate_nodes(
    graph_data: GraphData,
    *,
    rewire_as_relation_roots: bool = True,
) -> tuple[GraphData, dict[str, int]]:
    """Alias for ``normalize_lingua_edges_to_predicate_nodes``."""

    return normalize_lingua_edges_to_predicate_nodes(
        graph_data,
        rewire_as_relation_roots=rewire_as_relation_roots,
    )


def simplify_lingua_predicate_nodes_to_edges(graph_data: GraphData) -> tuple[GraphData, dict[str, int]]:
    """Collapse canonical relation aux nodes back to direct ``as:*`` edges.

    This is the canonical dual of ``standardize_lingua_edges_to_predicate_nodes``:
    a relation node shaped like ``Modification(variable=H, body=B)`` becomes
    ``B --as:modification--> H``; similarly ``Punctuation(pred.arg.1=T,
    pred.arg.2=P)`` becomes ``P --as:punctuation--> T``.

    The transform is intentionally structural rather than provenance-based.
    It collapses aux nodes whose label/type and role shape match a reified
    relation. It is therefore a canonical projection, not a lossless undo of
    whether the aux node existed in the original annotation.
    """

    simplified = copy.deepcopy(graph_data)
    nodes = simplified["graph"]["nodes"]
    edges = simplified["graph"]["edges"]
    stats: Counter[str] = Counter()

    outgoing: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        outgoing[int(edge["start"])].append(edge)

    relation_infos: dict[int, dict[str, Any]] = {}
    for node_index, node in enumerate(nodes):
        relation_label = aux_node_to_relation_edge(node)
        if relation_label is None:
            continue
        role1, role2 = relation_role_labels(node)
        outgoing_edges = outgoing.get(node_index, [])
        role1_edges = [edge for edge in outgoing_edges if edge.get("label") == role1]
        role2_edges = [edge for edge in outgoing_edges if edge.get("label") == role2]
        role_edges = role1_edges + role2_edges
        if role1 == role2:
            stats["skipped_relation_nodes_with_symmetric_roles"] += 1
            continue
        if len(role1_edges) != 1 or len(role2_edges) != 1:
            stats["skipped_relation_nodes_with_non_binary_roles"] += 1
            continue
        if len(role_edges) != len(outgoing_edges):
            stats["skipped_relation_nodes_with_extra_outgoing"] += 1
            continue

        relation_infos[node_index] = {
            "label": relation_label,
            "role1": role1,
            "role2": role2,
            "target": int(role1_edges[0]["end"]),
            "source": int(role2_edges[0]["end"]),
            "confidence": relation_confidence(node, role1_edges[0], role2_edges[0]),
        }

    if not relation_infos:
        return simplified, {}

    def root_projection(index: int, seen: set[int] | None = None) -> int:
        if seen is None:
            seen = set()
        if index in seen:
            return index
        seen.add(index)
        info = relation_infos.get(index)
        if info is None:
            return index
        return root_projection(int(info["source"]), seen)

    def target_projection(index: int) -> int:
        return root_projection(index)

    def source_projection(index: int, seen: set[int] | None = None) -> int:
        if seen is None:
            seen = set()
        if index in seen:
            return index
        seen.add(index)
        info = relation_infos.get(index)
        if info is not None:
            return target_projection(int(info["target"]))

        grouped_source = grouped_relation_source(index)
        if grouped_source is not None:
            return grouped_source
        return index

    def grouped_relation_source(index: int) -> int | None:
        child_relation_targets: list[int] = []
        for edge in outgoing.get(index, []):
            if edge.get("label") != "func.arg":
                continue
            child = int(edge["end"])
            child_info = relation_infos.get(child)
            if child_info is None or child_info.get("label") != "as:modification":
                continue
            child_relation_targets.append(target_projection(int(child_info["target"])))
        if not child_relation_targets:
            return None
        unique_targets = set(child_relation_targets)
        if len(unique_targets) != 1:
            return None
        return child_relation_targets[0]

    def path_exists(adjacency: dict[int, list[int]], start: int, target: int) -> bool:
        stack = [start]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency.get(node, []))
        return False

    def base_adjacency_without_collapsed_relations() -> dict[int, list[int]]:
        adjacency: dict[int, list[int]] = defaultdict(list)
        for edge in edges:
            start = int(edge["start"])
            end = int(edge["end"])
            if start in relation_infos:
                continue
            if end in relation_infos:
                end = root_projection(end)
            if start != end:
                adjacency[start].append(end)
        return adjacency

    while True:
        adjacency = base_adjacency_without_collapsed_relations()
        unsafe_relation_nodes: list[int] = []
        for relation_node, info in sorted(relation_infos.items()):
            start = source_projection(int(info["source"]))
            end = target_projection(int(info["target"]))
            if start == end or path_exists(adjacency, end, start):
                unsafe_relation_nodes.append(relation_node)
                continue
            adjacency[start].append(end)
        if not unsafe_relation_nodes:
            break
        for relation_node in unsafe_relation_nodes:
            info = relation_infos.pop(relation_node)
            stats["skipped_relation_nodes_that_would_create_cycle"] += 1
            stats[f"skipped_relation_nodes_that_would_create_cycle:{info['label']}"] += 1

    if not relation_infos:
        return simplified, dict(stats)

    relation_nodes = set(relation_infos)

    def remap(index: int) -> int | None:
        if index in relation_nodes:
            return None
        removed_before = sum(1 for relation_node in relation_nodes if relation_node < index)
        return index - removed_before

    new_nodes = [
        copy.deepcopy(node)
        for index, node in enumerate(nodes)
        if index not in relation_nodes
    ]

    new_edges: list[dict[str, Any]] = []
    for edge in edges:
        start = int(edge["start"])
        end = int(edge["end"])
        if start in relation_nodes:
            continue
        if end in relation_nodes:
            end = root_projection(end)

        new_start = remap(start)
        new_end = remap(end)
        if new_start is None or new_end is None or new_start == new_end:
            continue

        new_edge = copy.deepcopy(edge)
        new_edge["start"] = new_start
        new_edge["end"] = new_end
        new_edges.append(new_edge)

    for _relation_node, info in sorted(relation_infos.items()):
        start = source_projection(int(info["source"]))
        end = target_projection(int(info["target"]))
        new_start = remap(start)
        new_end = remap(end)
        if new_start is None or new_end is None or new_start == new_end:
            stats["skipped_collapsed_relation_edges_with_invalid_endpoint"] += 1
            continue
        edge = {
            "start": new_start,
            "end": new_end,
            "label": info["label"],
        }
        if info.get("confidence") is not None:
            edge["confidence"] = info["confidence"]
        new_edges.append(edge)
        stats["collapsed_relation_nodes"] += 1
        stats[f"collapsed_relation_nodes:{info['label']}"] += 1

    simplified["graph"]["nodes"] = new_nodes
    simplified["graph"]["edges"] = _dedupe_edges(new_edges)
    return simplified, dict(stats)


def relation_edge_to_aux_node(edge_label: str) -> tuple[str, str]:
    if edge_label in RELATION_EDGE_TO_AUX_NODE:
        return RELATION_EDGE_TO_AUX_NODE[edge_label]
    return edge_label[:1].upper() + edge_label[1:], "OtherConstant"


def aux_node_to_relation_edge(node: dict[str, Any]) -> str | None:
    spans = node.get("spans")
    node_type = node.get("type")
    if not isinstance(spans, str):
        return None

    for edge_label, aux_node in RELATION_EDGE_TO_AUX_NODE.items():
        if aux_node == (spans, node_type):
            return f"as:{edge_label}"

    if (
        node_type == "OtherConstant"
        and spans
        and spans not in EXPRESSION_LIKE_AUX_LABELS
    ):
        return f"as:{spans[:1].lower()}{spans[1:]}"
    return None


def relation_confidence(
    node: dict[str, Any],
    role1_edge: dict[str, Any],
    role2_edge: dict[str, Any],
) -> Any:
    if "confidence" in node:
        return node["confidence"]
    role1_confidence = role1_edge.get("confidence")
    role2_confidence = role2_edge.get("confidence")
    if role1_confidence == role2_confidence:
        return role1_confidence
    return None


def relation_role_labels(node: dict[str, Any]) -> tuple[str, str]:
    node_type = str(node.get("type", ""))
    node_label = str(node.get("spans", ""))
    if node_type == "ModificationalFunctor":
        return "variable", "body"
    if node_label in {"Expression", "List", "Parataxis"}:
        return "func.arg", "func.arg"
    return "pred.arg.1", "pred.arg.2"


def _predicate_edge(start: int, end: int, label: str, source_edge: dict[str, Any]) -> dict[str, Any]:
    edge = {
        "start": start,
        "end": end,
        "label": label,
    }
    if "confidence" in source_edge:
        edge["confidence"] = source_edge["confidence"]
    return edge


def is_modification_aux_node(node: dict[str, Any]) -> bool:
    return node.get("spans") == "Modification" and node.get("type") == "ModificationalFunctor"


def normalize_simple_leaf_modifications(graph_data: GraphData) -> tuple[GraphData, dict[str, int]]:
    """Flatten simple leaf-body ``Modification`` nodes to ``as:modification``.

    This is an evaluation canonicalization for the standard equivalence:

    ``Modification(variable=H, body=B)`` == ``B --as:modification--> H``

    The equivalence is intentionally narrow.  It applies only when the
    ``Modification`` node has exactly one ``variable`` edge and one ``body``
    edge, and the body node has no outgoing edges in the current graph.
    """

    normalized = copy.deepcopy(graph_data)
    stats: Counter[str] = Counter()

    while True:
        candidate = _find_simple_leaf_modification(normalized)
        if candidate is None:
            break
        mod_idx, variable_idx, body_idx = candidate
        normalized = _collapse_modification(normalized, mod_idx, variable_idx, body_idx)
        stats["flattened_simple_leaf_modifications"] += 1

    return normalized, dict(stats)


def _find_simple_leaf_modification(graph_data: GraphData) -> tuple[int, int, int] | None:
    nodes = graph_data["graph"]["nodes"]
    edges = graph_data["graph"]["edges"]
    outgoing: dict[int, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        outgoing[int(edge["start"])].append(edge)
        incoming[int(edge["end"])].append(edge)

    for mod_idx, node in enumerate(nodes):
        if not is_modification_aux_node(node):
            continue

        mod_outgoing = outgoing.get(mod_idx, [])
        if len(mod_outgoing) != 2:
            continue

        variable_edges = [edge for edge in mod_outgoing if edge.get("label") == "variable"]
        body_edges = [edge for edge in mod_outgoing if edge.get("label") == "body"]
        if len(variable_edges) != 1 or len(body_edges) != 1:
            continue

        variable_idx = int(variable_edges[0]["end"])
        body_idx = int(body_edges[0]["end"])
        if variable_idx == mod_idx or body_idx == mod_idx or variable_idx == body_idx:
            continue

        if outgoing.get(body_idx):
            continue

        # Rewiring an incoming edge from the variable itself would make a
        # self-loop.  That is not the simple direct-modification equivalence.
        if any(int(edge["start"]) == variable_idx for edge in incoming.get(mod_idx, [])):
            continue

        return mod_idx, variable_idx, body_idx

    return None


def _collapse_modification(graph_data: GraphData, mod_idx: int, variable_idx: int, body_idx: int) -> GraphData:
    nodes = graph_data["graph"]["nodes"]
    edges = graph_data["graph"]["edges"]

    def remap(index: int) -> int | None:
        if index == mod_idx:
            return None
        return index - 1 if index > mod_idx else index

    new_graph_data = copy.deepcopy(graph_data)
    new_graph_data["graph"]["nodes"] = [
        copy.deepcopy(node)
        for index, node in enumerate(nodes)
        if index != mod_idx
    ]

    new_edges: list[dict[str, Any]] = []
    for edge in edges:
        start = int(edge["start"])
        end = int(edge["end"])

        if start == mod_idx:
            continue
        if end == mod_idx:
            # The collapsed subgraph is B --as:modification--> H, so B is the
            # local root that recovers the whole modifier phrase.
            end = body_idx

        new_start = remap(start)
        new_end = remap(end)
        if new_start is None or new_end is None or new_start == new_end:
            continue

        new_edge = copy.deepcopy(edge)
        new_edge["start"] = new_start
        new_edge["end"] = new_end
        new_edges.append(new_edge)

    new_edges.append(
        {
            "start": remap(body_idx),
            "end": remap(variable_idx),
            "label": "as:modification",
        }
    )
    new_graph_data["graph"]["edges"] = _dedupe_edges(new_edges)
    return new_graph_data


def _dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, str]] = set()
    deduped: list[dict[str, Any]] = []
    for edge in edges:
        key = (int(edge["start"]), int(edge["end"]), str(edge["label"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped
