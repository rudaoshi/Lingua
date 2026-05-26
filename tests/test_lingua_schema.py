import json
from collections import Counter, defaultdict
from pathlib import Path

from lingua.concept.standard import (
    LinguaGraphAuxNodeLabels,
    LinguaGraphEdges,
    LinguaGraphNodePoses,
)
from lingua.utils.topology_sorter import LinguaGraphTopologySorter


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.jsonl"


def iter_train_examples():
    with TRAIN_PATH.open() as f:
        for line_number, line in enumerate(f, start=1):
            yield line_number, json.loads(line)


def is_aux_spans(spans):
    return isinstance(spans, str)


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

    assert label_counts["nonsense"] == 2
    assert label_counts.get("as:nonsense", 0) == 0
    assert aux_counts["Ref"] == 725
    assert aux_counts["Modification"] == 6303


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

    assert root_type_counts["PunctuationalConstant"] == 9425


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

    assert aux_edge_counts["Ref"] == 725
    assert aux_edge_counts["Modification"] == 6303
