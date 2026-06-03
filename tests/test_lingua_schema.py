import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from lingua.concept.standard import (
    LinguaGraphAuxNodeLabels,
    LinguaGraphEdges,
    LinguaGraphNodePoses,
)
from lingua.utils.segement import segment
from lingua.utils.topology_sorter import LinguaGraphTopologySorter


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.jsonl"
DATA_PATHS = [
    TRAIN_PATH,
    ROOT / "data" / "dev.jsonl",
    ROOT / "data" / "test.jsonl",
]
RELATIVE_REF_ARG2_WRAPPED_MARKERS = {
    "that",
    "which",
    "who",
    "whom",
    "whose",
}
RELATIVE_REF_ARG2_DIRECT_ADVERBS = {
    "when",
    "where",
}

AUX_NODE_TYPES = {
    "Apposition": "AppositionalPredicator",
    "Attribute": "AttributePredicator",
    "Copula": "FactualPredicator",
    "Discourse": "ModificationalFunctor",
    "Expression": "ExpressionFunctor",
    "List": "ListFunctor",
    "Modification": "ModificationalFunctor",
    "Parataxis": "ConjunctionalFunctor",
    "Ref": "ReferentialPredicator",
    "Vocative": "ModificationalFunctor",
    "Whether": "GeneralFunctor",
}


def iter_train_examples():
    with TRAIN_PATH.open() as f:
        for line_number, line in enumerate(f, start=1):
            yield line_number, json.loads(line)


def iter_lingua_examples():
    for path in DATA_PATHS:
        with path.open() as f:
            for line_number, line in enumerate(f, start=1):
                yield path.name, line_number, json.loads(line)


def is_aux_spans(spans):
    return isinstance(spans, str)


def node_token_indexes(node):
    indexes = []
    for start, end in node["spans"]:
        indexes.extend(range(start, end + 1))
    return indexes


def node_text(node, words):
    spans = node["spans"]
    if is_aux_spans(spans):
        return spans
    pieces = []
    for start, end in spans:
        pieces.extend(words[start : end + 1])
    return "".join(pieces).strip()


def node_span_bounds(node):
    spans = node["spans"]
    if is_aux_spans(spans):
        return None
    indexes = node_token_indexes(node)
    return min(indexes), max(indexes)


def assert_acyclic(adjacency):
    color = [0] * len(adjacency)

    def visit(node):
        color[node] = 1
        for child in adjacency[node]:
            if color[child] == 1:
                return False
            if color[child] == 0 and not visit(child):
                return False
        color[node] = 2
        return True

    for node in range(len(adjacency)):
        if color[node] == 0:
            assert visit(node)


def descendant_token_indexes(start, nodes, adjacency):
    seen = set()
    stack = [start]
    indexes = []
    while stack:
        node_index = stack.pop()
        if node_index in seen:
            continue
        seen.add(node_index)
        if not is_aux_spans(nodes[node_index]["spans"]):
            indexes.extend(node_token_indexes(nodes[node_index]))
        stack.extend(adjacency[node_index])
    return tuple(sorted(set(indexes)))


def is_reachable(start, target, adjacency):
    seen = set()
    stack = [start]
    while stack:
        node_index = stack.pop()
        if node_index == target:
            return True
        if node_index in seen:
            continue
        seen.add(node_index)
        stack.extend(adjacency[node_index])
    return False


def test_nonsense_edges_are_schema_labels():
    assert "nonsense" in LinguaGraphEdges
    assert "as:nonsense" in LinguaGraphEdges


def test_topology_sorter_priorities_cover_schema_edges():
    sorter = LinguaGraphTopologySorter()

    assert set(LinguaGraphEdges) <= set(sorter.edge_priorities)


def test_train_schema_labels_are_valid():
    label_counts = Counter()
    aux_counts = Counter()

    for line_number, example in iter_train_examples():
        graph = example["lingua"]["graph"]
        nodes = graph["nodes"]

        for node_index, node in enumerate(nodes):
            assert node["type"] in LinguaGraphNodePoses, (line_number, node_index, node)
            spans = node["spans"]
            if is_aux_spans(spans):
                assert spans in LinguaGraphAuxNodeLabels, (line_number, node_index, node)
                aux_counts[spans] += 1

        for edge_index, edge in enumerate(graph["edges"]):
            assert edge["label"] in LinguaGraphEdges, (line_number, edge_index, edge)
            assert 0 <= edge["start"] < len(nodes), (line_number, edge_index, edge)
            assert 0 <= edge["end"] < len(nodes), (line_number, edge_index, edge)
            label_counts[edge["label"]] += 1

    assert label_counts["nonsense"] == 0
    assert label_counts.get("as:nonsense", 0) == 0
    assert aux_counts["Ref"] == 714
    assert aux_counts["Modification"] == 5965


def test_aux_symbol_labels_match_required_node_types():
    for filename, line_number, example in iter_lingua_examples():
        for node_index, node in enumerate(example["lingua"]["graph"]["nodes"]):
            spans = node["spans"]
            if not is_aux_spans(spans) or spans not in AUX_NODE_TYPES:
                continue
            assert node["type"] == AUX_NODE_TYPES[spans], (
                filename,
                line_number,
                node_index,
                node,
                example["sentence"],
            )


def test_func_arg_symbol_wrappers_are_not_unary():
    for filename, line_number, example in iter_lingua_examples():
        graph = example["lingua"]["graph"]
        func_arg_counts = Counter()
        for edge in graph["edges"]:
            if edge["label"] == "func.arg":
                func_arg_counts[edge["start"]] += 1

        for node_index, node in enumerate(graph["nodes"]):
            spans = node["spans"]
            if not is_aux_spans(spans) or spans not in {"Expression", "List", "Parataxis"}:
                continue
            assert func_arg_counts[node_index] >= 2, (
                filename,
                line_number,
                node_index,
                node,
                example["sentence"],
            )


def test_compound_hyphens_are_not_punctuation_modification_bodies():
    blocked_right_types = {
        "FactualPredicator",
        "LogicalPredicator",
        "ModificationalFunctor",
        "GeneralFunctor",
        "ConjunctionalFunctor",
    }

    for filename, line_number, example in iter_lingua_examples():
        graph = example["lingua"]["graph"]
        nodes = graph["nodes"]
        words = example["lingua"]["words"]
        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        for graph_edge in graph["edges"]:
            outgoing[graph_edge["start"]].append(graph_edge)
            incoming[graph_edge["end"]].append(graph_edge)

        for edge in graph["edges"]:
            source = nodes[edge["start"]]
            target = nodes[edge["end"]]
            if (
                edge["label"] != "body"
                or source["spans"] != "Modification"
                or target["type"] not in {"PunctuationalConstant", "SymbolConstant"}
                or node_text(target, words) != "-"
            ):
                continue

            variable_edges = [
                child_edge
                for child_edge in outgoing[edge["start"]]
                if child_edge["label"] == "variable"
            ]
            close_edges = [
                child_edge
                for child_edge in outgoing[edge["end"]]
                if child_edge["label"] == "as:punctuation"
            ]
            if len(variable_edges) != 1 or len(close_edges) != 1:
                continue

            right = nodes[variable_edges[0]["end"]]
            left = nodes[close_edges[0]["end"]]
            left_bounds = node_span_bounds(left)
            hyphen_bounds = node_span_bounds(target)
            right_bounds = node_span_bounds(right)
            if not (left_bounds and hyphen_bounds and right_bounds):
                continue
            is_ordered_compound = left_bounds[1] < hyphen_bounds[0] < right_bounds[0]
            has_punctuation_parent = any(
                nodes[parent_edge["start"]]["type"] == "PunctuationalConstant"
                and parent_edge["label"] == "as:punctuation"
                for parent_edge in incoming[edge["start"]]
            )
            assert not (
                is_ordered_compound
                and right["type"] not in blocked_right_types
                and left["type"] not in {"PunctuationalConstant", "SymbolConstant"}
                and not has_punctuation_parent
            ), (
                filename,
                line_number,
                edge,
                close_edges[0],
                variable_edges[0],
                example["sentence"],
            )


def test_redundant_modification_artifacts_are_absent():
    for filename, line_number, example in iter_lingua_examples():
        graph = example["lingua"]["graph"]
        nodes = graph["nodes"]
        outgoing = defaultdict(list)
        adjacency = [[] for _ in nodes]
        for edge in graph["edges"]:
            outgoing[edge["start"]].append(edge)
            adjacency[edge["start"]].append(edge["end"])

        for node_index, node in enumerate(nodes):
            if node["spans"] != "Modification":
                continue

            variable_edges = [
                edge for edge in outgoing[node_index] if edge["label"] == "variable"
            ]
            body_edges = [
                edge for edge in outgoing[node_index] if edge["label"] == "body"
            ]
            if len(variable_edges) != 1 or len(body_edges) != 1:
                continue

            variable_index = variable_edges[0]["end"]
            body_index = body_edges[0]["end"]
            variable_desc = descendant_token_indexes(variable_index, nodes, adjacency)
            body_desc = descendant_token_indexes(body_index, nodes, adjacency)
            assert variable_desc != body_desc, (
                filename,
                line_number,
                node_index,
                example["sentence"],
            )

            if nodes[body_index]["type"] != "ModificationalFunctor":
                assert not is_reachable(variable_index, body_index, adjacency), (
                    filename,
                    line_number,
                    node_index,
                    variable_index,
                    body_index,
                    example["sentence"],
                )
                continue
            for edge in outgoing[body_index]:
                if edge["label"] == "variable":
                    assert (
                        descendant_token_indexes(edge["end"], nodes, adjacency)
                        != variable_desc
                    ), (
                        filename,
                        line_number,
                        node_index,
                        body_index,
                        example["sentence"],
                    )
            assert not is_reachable(variable_index, body_index, adjacency), (
                filename,
                line_number,
                node_index,
                variable_index,
                body_index,
                example["sentence"],
            )


def test_list_items_are_recoverable_fields():
    for filename, line_number, example in iter_lingua_examples():
        graph = example["lingua"]["graph"]
        nodes = graph["nodes"]
        outgoing = defaultdict(list)
        for edge in graph["edges"]:
            outgoing[edge["start"]].append(edge)

        for node_index, node in enumerate(nodes):
            if node["type"] != "ListFunctor":
                continue
            for edge in outgoing[node_index]:
                if edge["label"] != "func.arg":
                    continue
                item_index = edge["end"]
                item = nodes[item_index]
                assert item["type"] != "OtherConstant", (
                    filename,
                    line_number,
                    node_index,
                    item_index,
                    example["sentence"],
                )
                if item["type"] == "PunctuationalConstant":
                    assert any(
                        child_edge["label"] == "as:punctuation"
                        for child_edge in outgoing[item_index]
                    ), (
                        filename,
                        line_number,
                        node_index,
                        item_index,
                        example["sentence"],
                    )


def test_relative_ref_arg2_uses_expected_shape():
    for filename, line_number, example in iter_lingua_examples():
        graph = example["lingua"]["graph"]
        nodes = graph["nodes"]
        words = example["lingua"]["words"]
        outgoing = defaultdict(list)
        for graph_edge in graph["edges"]:
            outgoing[graph_edge["start"]].append(graph_edge)

        for edge in graph["edges"]:
            source = nodes[edge["start"]]
            if source["spans"] != "Ref" or edge["label"] != "pred.arg.2":
                continue

            target = nodes[edge["end"]]
            target_text = node_text(target, words).lower()
            if target_text in RELATIVE_REF_ARG2_DIRECT_ADVERBS:
                func_arg_edges = [
                    child_edge
                    for child_edge in outgoing[edge["end"]]
                    if child_edge["label"] == "func.arg"
                ]
                assert target["type"] == "GeneralFunctor", (
                    filename,
                    line_number,
                    edge,
                    target,
                    example["sentence"],
                )
                assert func_arg_edges, (
                    filename,
                    line_number,
                    edge,
                    target,
                    example["sentence"],
                )
                seen = set()
                stack = [child_edge["end"] for child_edge in func_arg_edges]
                while stack:
                    current = stack.pop()
                    if current in seen:
                        continue
                    seen.add(current)
                    for child_edge in outgoing[current]:
                        assert child_edge["end"] != edge["end"], (
                            filename,
                            line_number,
                            edge,
                            child_edge,
                            example["sentence"],
                        )
                        stack.append(child_edge["end"])
                continue

            assert target_text not in RELATIVE_REF_ARG2_WRAPPED_MARKERS, (
                filename,
                line_number,
                edge,
                target,
                example["sentence"],
            )

            if target["spans"] != "Modification":
                continue

            variable_edges = [
                child_edge
                for child_edge in outgoing[edge["end"]]
                if child_edge["label"] == "variable"
            ]
            for variable_edge in variable_edges:
                variable = nodes[variable_edge["end"]]
                assert variable["type"] != "LogicalPredicator", (
                    filename,
                    line_number,
                    edge,
                    variable_edge,
                    variable,
                    example["sentence"],
                )


def test_apposition_punctuation_arg1_is_host_close_wrapper():
    for filename, line_number, example in iter_lingua_examples():
        graph = example["lingua"]["graph"]
        nodes = graph["nodes"]
        outgoing = defaultdict(list)
        for edge in graph["edges"]:
            outgoing[edge["start"]].append(edge)

        for apposition_index, node in enumerate(nodes):
            if node["spans"] != "Apposition":
                continue

            arg1_edges = [
                edge
                for edge in outgoing[apposition_index]
                if edge["label"] == "pred.arg.1"
            ]
            assert len(arg1_edges) == 1, (
                filename,
                line_number,
                apposition_index,
                example["sentence"],
            )

            arg1_index = arg1_edges[0]["end"]
            arg1 = nodes[arg1_index]
            if arg1["type"] not in {"PunctuationalConstant", "SymbolConstant"}:
                continue

            close_edges = [
                edge
                for edge in outgoing[arg1_index]
                if edge["label"] == "as:punctuation"
            ]
            assert close_edges, (
                filename,
                line_number,
                apposition_index,
                arg1_index,
                arg1,
                example["sentence"],
            )
            assert all(edge["end"] != apposition_index for edge in close_edges), (
                filename,
                line_number,
                apposition_index,
                arg1_index,
                close_edges,
                example["sentence"],
            )


def test_numeric_value_label_dash_is_not_apposition():
    numeric_value = re.compile(r"\d+(?:\.\d+)?")

    for filename, line_number, example in iter_lingua_examples():
        graph = example["lingua"]["graph"]
        nodes = graph["nodes"]
        words = example["lingua"]["words"]
        outgoing = defaultdict(list)
        for edge in graph["edges"]:
            outgoing[edge["start"]].append(edge)

        for apposition_index, node in enumerate(nodes):
            if node["spans"] != "Apposition":
                continue

            arg1_edges = [
                edge
                for edge in outgoing[apposition_index]
                if edge["label"] == "pred.arg.1"
            ]
            if len(arg1_edges) != 1:
                continue

            arg1_index = arg1_edges[0]["end"]
            arg1 = nodes[arg1_index]
            if (
                arg1["type"] != "PunctuationalConstant"
                or node_text(arg1, words).strip() != "-"
            ):
                continue

            close_edges = [
                edge
                for edge in outgoing[arg1_index]
                if edge["label"] == "as:punctuation"
            ]
            if len(close_edges) != 1:
                continue

            value = nodes[close_edges[0]["end"]]
            assert not numeric_value.fullmatch(node_text(value, words).strip()), (
                filename,
                line_number,
                apposition_index,
                arg1_index,
                value,
                example["sentence"],
            )


def test_numeric_value_label_fragments_use_dash_expression():
    numeric_value = re.compile(r"\d+(?:\.\d+)?")

    for filename, line_number, example in iter_lingua_examples():
        words = [word.strip() for word in example["lingua"]["words"]]
        if (
            len(words) < 3
            or not numeric_value.fullmatch(words[0])
            or words[1] != "-"
            or words[2] not in {"Number", "Percentage"}
        ):
            continue

        graph = example["lingua"]["graph"]
        nodes = graph["nodes"]
        outgoing = defaultdict(list)
        token_owners = defaultdict(list)
        for node_index, node in enumerate(nodes):
            if is_aux_spans(node["spans"]):
                continue
            for token_index in node_token_indexes(node):
                token_owners[token_index].append(node_index)
        for edge in graph["edges"]:
            outgoing[edge["start"]].append(edge)

        assert len(token_owners[1]) == 1, (
            filename,
            line_number,
            token_owners[1],
            example["sentence"],
        )
        dash_index = token_owners[1][0]
        dash = nodes[dash_index]
        assert dash["type"] == "ExpressionFunctor", (
            filename,
            line_number,
            dash_index,
            dash,
            example["sentence"],
        )
        assert node_text(dash, example["lingua"]["words"]).strip() == "-", (
            filename,
            line_number,
            dash_index,
            dash,
            example["sentence"],
        )
        assert sum(edge["label"] == "func.arg" for edge in outgoing[dash_index]) >= 2, (
            filename,
            line_number,
            dash_index,
            outgoing[dash_index],
            example["sentence"],
        )


def test_all_lingua_words_match_segment():
    for filename, line_number, example in iter_lingua_examples():
        assert example["lingua"]["words"] == segment(example["sentence"]), (
            filename,
            line_number,
            example["sentence"],
        )


def test_train_tokens_are_partitioned_by_lexical_nodes():
    for line_number, example in iter_train_examples():
        words = example["lingua"]["words"]
        coverage = [0] * len(words)

        for node_index, node in enumerate(example["lingua"]["graph"]["nodes"]):
            spans = node["spans"]
            if is_aux_spans(spans):
                continue

            assert isinstance(spans, list), (line_number, node_index, node)
            for span in spans:
                assert isinstance(span, list) and len(span) == 2, (
                    line_number,
                    node_index,
                    span,
                )
                start, end = span
                assert 0 <= start <= end < len(words), (line_number, node_index, span)
                for token_index in range(start, end + 1):
                    coverage[token_index] += 1

        assert coverage == [1] * len(words), (line_number, example["sentence"])


def test_train_graphs_are_single_rooted_dags():
    root_type_counts = Counter()

    for line_number, example in iter_train_examples():
        graph = example["lingua"]["graph"]
        nodes = graph["nodes"]
        indegree = [0] * len(nodes)
        adjacency = [[] for _ in nodes]

        for edge in graph["edges"]:
            start = edge["start"]
            end = edge["end"]
            adjacency[start].append(end)
            indegree[end] += 1

        roots = [node_index for node_index, degree in enumerate(indegree) if degree == 0]
        assert len(roots) == 1, (line_number, roots, example["sentence"])
        root_type_counts[nodes[roots[0]]["type"]] += 1

        reachable = set()
        stack = [roots[0]]
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            stack.extend(adjacency[node])

        assert len(reachable) == len(nodes), (line_number, example["sentence"])
        assert_acyclic(adjacency)

    assert root_type_counts["PunctuationalConstant"] == 9459


def test_eval_graphs_are_single_rooted_dags():
    for filename, line_number, example in iter_lingua_examples():
        if filename == "train.jsonl":
            continue

        graph = example["lingua"]["graph"]
        nodes = graph["nodes"]
        indegree = [0] * len(nodes)
        adjacency = [[] for _ in nodes]

        for edge in graph["edges"]:
            start = edge["start"]
            end = edge["end"]
            adjacency[start].append(end)
            indegree[end] += 1

        roots = [node_index for node_index, degree in enumerate(indegree) if degree == 0]
        assert len(roots) == 1, (filename, line_number, roots, example["sentence"])

        reachable = set()
        stack = [roots[0]]
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            stack.extend(adjacency[node])

        assert len(reachable) == len(nodes), (filename, line_number, example["sentence"])
        assert_acyclic(adjacency)


def test_train_aux_structures_have_required_edges():
    aux_edge_counts = Counter()

    for line_number, example in iter_train_examples():
        graph = example["lingua"]["graph"]
        outgoing = defaultdict(list)
        for edge in graph["edges"]:
            outgoing[edge["start"]].append(edge["label"])

        for node_index, node in enumerate(graph["nodes"]):
            spans = node["spans"]
            if spans == "Modification":
                aux_edge_counts["Modification"] += 1
                labels = set(outgoing[node_index])
                assert {"variable", "body"} <= labels, (
                    line_number,
                    node_index,
                    labels,
                    example["sentence"],
                )
            elif spans == "Ref":
                aux_edge_counts["Ref"] += 1
                labels = set(outgoing[node_index])
                assert {"pred.arg.1", "pred.arg.2"} <= labels, (
                    line_number,
                    node_index,
                    labels,
                    example["sentence"],
                )

    assert aux_edge_counts["Ref"] == 714
    assert aux_edge_counts["Modification"] == 5965


def test_numeric_colon_formats_use_expression_functors():
    numeric_piece = re.compile(r"\d+(?:\.\d+)?")

    for filename, line_number, example in iter_lingua_examples():
        words = example["lingua"]["words"]
        nodes = example["lingua"]["graph"]["nodes"]
        token_owner = {}

        for node_index, node in enumerate(nodes):
            if is_aux_spans(node["spans"]):
                continue
            for token_index in node_token_indexes(node):
                token_owner[token_index] = node_index

        for token_index, word in enumerate(words):
            if word.strip() != ":":
                continue
            if token_index == 0 or token_index == len(words) - 1:
                continue
            if not numeric_piece.fullmatch(words[token_index - 1].strip()):
                continue
            if not numeric_piece.fullmatch(words[token_index + 1].strip()):
                continue

            node_index = token_owner[token_index]
            node = nodes[node_index]
            assert node["type"] == "ExpressionFunctor", (
                filename,
                line_number,
                token_index,
                node_index,
                node,
                example["sentence"],
            )
            assert node_token_indexes(node) == [token_index], (
                filename,
                line_number,
                token_index,
                node_index,
                node,
                example["sentence"],
            )


def test_punctuation_close_wrappers_are_not_plain_punctuation_children():
    for filename, line_number, example in iter_lingua_examples():
        nodes = example["lingua"]["graph"]["nodes"]
        outgoing = defaultdict(list)
        incoming_plain_punctuation = defaultdict(list)
        outgoing_close_punctuation = defaultdict(list)

        for edge in example["lingua"]["graph"]["edges"]:
            outgoing[edge["start"]].append(edge)
            if edge["label"] == "punctuation":
                incoming_plain_punctuation[edge["end"]].append(edge)
            elif edge["label"] == "as:punctuation":
                outgoing_close_punctuation[edge["start"]].append(edge)

        for node_index, node in enumerate(nodes):
            if is_aux_spans(node["spans"]) or node["type"] != "PunctuationalConstant":
                continue
            assert not (
                incoming_plain_punctuation.get(node_index)
                and outgoing_close_punctuation.get(node_index)
            ), (
                filename,
                line_number,
                node_index,
                incoming_plain_punctuation[node_index],
                outgoing_close_punctuation[node_index],
                example["sentence"],
            )

        for node_index, plain_edges in incoming_plain_punctuation.items():
            node = nodes[node_index]
            assert not is_aux_spans(node["spans"]) and node["type"] == "PunctuationalConstant", (
                filename,
                line_number,
                node_index,
                node,
                plain_edges,
                example["sentence"],
            )
            assert not outgoing.get(node_index), (
                filename,
                line_number,
                node_index,
                node,
                plain_edges,
                outgoing[node_index],
                example["sentence"],
            )
