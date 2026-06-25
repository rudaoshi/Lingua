import json
import copy
from pathlib import Path

import networkx as nx

from lingua.learn.wordgraph.modeler.gp2word import (
    lingua2wordlingua,
    standardized_lingua2wordlingua,
)
from lingua.learn.wordgraph.modeler.word2gp import (
    ensure_single_rooted_dag,
    strip_hierarchy_edge_components,
    wordlingua2lingua,
)
from lingua.structure.gpgraph import GPGraph, GPGPhraseNode
from lingua.utils.metric import graph_match_metric
from lingua.utils.normalization import normalize_lingua_edges_to_predicate_nodes


ROOT = Path(__file__).resolve().parents[1]


def make_graph():
    graph = GPGraph()
    graph.words = ["a", "b", "c"]
    nodes = [
        graph.add_node(GPGPhraseNode(spans=[(idx, idx)], pos="NominalConstant"))
        for idx in range(3)
    ]
    return graph, nodes


def hierarchy_labels(graph):
    return [
        edge.label
        for _, edge, _ in graph.edges()
        if "@lower" in edge.label or "@upper" in edge.label
    ]


def root_texts(graph):
    return [graph.node_text(node) for node in graph.nodes() if graph.g.in_degree(node.ID) == 0]


def test_strip_hierarchy_edge_components_removes_only_upper_lower():
    graph, (a_node, b_node, c_node) = make_graph()
    graph.add_relation(a_node, b_node, "@lower", confidence=0.2)
    graph.add_relation(b_node, c_node, "@upper+pred.arg.1", confidence=0.7)
    graph.add_relation(c_node, a_node, "modification+@lower", confidence=0.3)

    _, updated = strip_hierarchy_edge_components(graph)

    assert updated
    assert graph.get_edge(a_node, b_node) is None
    assert graph.get_edge(b_node, c_node).label == "pred.arg.1"
    assert graph.get_edge(b_node, c_node).confidence == 0.7
    assert graph.get_edge(c_node, a_node).label == "modification"
    assert graph.get_edge(c_node, a_node).confidence == 0.3


def test_strip_hierarchy_edge_components_keeps_aux_relation_markers():
    graph, (a_node, b_node, _) = make_graph()
    graph.add_relation(a_node, b_node, "@modification_of")

    _, updated = strip_hierarchy_edge_components(graph)

    assert not updated
    assert graph.get_edge(a_node, b_node).label == "@modification_of"


def test_wordlingua2lingua_does_not_expose_hierarchy_hint_edges():
    graph, (a_node, b_node, c_node) = make_graph()
    graph.add_relation(a_node, b_node, "@lower")
    graph.add_relation(b_node, c_node, "@upper+pred.arg.2")

    converted, updated, _ = wordlingua2lingua(graph)

    assert updated
    assert hierarchy_labels(converted) == []
    assert len(root_texts(converted)) == 1
    assert converted.get_edge(a_node, b_node) is None
    assert converted.get_edge(b_node, c_node).label == "pred.arg.2"
    assert "nonsense" in [edge.label for _, edge, _ in converted.edges()]


def test_ensure_single_rooted_dag_connects_extra_roots_with_fallback_edge():
    graph, (a_node, b_node, c_node) = make_graph()
    graph.add_relation(b_node, c_node, "pred.arg.1")

    _, updated = ensure_single_rooted_dag(graph)

    assert updated
    assert len(root_texts(graph)) == 1
    assert "nonsense" in [edge.label for _, edge, _ in graph.edges()]
    assert graph.meta["word2gp_repairs"][0]["type"] == "multiple_roots"


def test_ensure_single_rooted_dag_does_not_prefer_punctuation_root():
    graph = GPGraph()
    graph.words = ["runs", "."]
    predicate = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="FactualPredicator"))
    punctuation = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="PunctuationalConstant"))

    _, updated = ensure_single_rooted_dag(graph)

    assert updated
    assert root_texts(graph) == ["runs"]
    assert graph.get_edge(predicate, punctuation).label == "nonsense"
    assert graph.meta["word2gp_repairs"][0]["primary_root"] == "runs"


def test_ensure_single_rooted_dag_does_not_retarget_punctuation_shorthand_root():
    graph = GPGraph()
    graph.words = ["runs", "quickly", "."]
    predicate = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="FactualPredicator"))
    adverb = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="ModificationalConstant"))
    punctuation = graph.add_node(GPGPhraseNode(spans=[(2, 2)], pos="PunctuationalConstant"))
    graph.add_relation(predicate, adverb, "pred.arg.2")
    graph.add_relation(punctuation, adverb, "as:punctuation")

    _, updated = ensure_single_rooted_dag(graph)

    assert updated
    assert root_texts(graph) == ["runs"]
    assert graph.get_edge(predicate, punctuation).label == "nonsense"
    assert graph.get_edge(punctuation, adverb).label == "as:punctuation"
    assert graph.meta["word2gp_repairs"][0]["fallback_label"] == "nonsense"


def test_ensure_single_rooted_dag_avoids_as_shorthand_parent_root():
    graph, (a_node, b_node, c_node) = make_graph()
    graph.add_relation(a_node, c_node, "as:modification")

    _, updated = ensure_single_rooted_dag(graph)

    assert updated
    assert root_texts(graph) == ["b"]
    assert graph.get_edge(b_node, a_node).label == "nonsense"
    assert graph.get_edge(a_node, c_node).label == "as:modification"
    assert graph.meta["word2gp_repairs"][0]["primary_root"] == "b"


def test_ensure_single_rooted_dag_uses_safe_descendant_when_roots_have_as_shorthand():
    graph = GPGraph()
    graph.words = ["a", "b", "c", "d"]
    a_node = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="NominalConstant"))
    b_node = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="NominalConstant"))
    c_node = graph.add_node(GPGPhraseNode(spans=[(2, 2)], pos="NominalConstant"))
    d_node = graph.add_node(GPGPhraseNode(spans=[(3, 3)], pos="NominalConstant"))
    graph.add_relation(a_node, c_node, "as:modification")
    graph.add_relation(b_node, d_node, "as:modification")

    _, updated = ensure_single_rooted_dag(graph)

    assert updated
    assert root_texts(graph) == ["a"]
    assert graph.get_edge(c_node, b_node).label == "nonsense"
    assert graph.get_edge(a_node, c_node).label == "as:modification"
    assert graph.get_edge(b_node, d_node).label == "as:modification"
    assert graph.meta["word2gp_repairs"][0]["fallback_parent"] == "c"


def test_ensure_single_rooted_dag_avoids_fallback_parent_reachable_from_extra_root():
    graph = GPGraph()
    graph.words = ["head", "inner", "."]
    head = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="FactualPredicator"))
    inner = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="ModificationalConstant"))
    punctuation = graph.add_node(GPGPhraseNode(spans=[(2, 2)], pos="PunctuationalConstant"))
    graph.add_relation(head, inner, "pred.arg.2")
    graph.add_relation(punctuation, inner, "as:punctuation")

    _, updated = ensure_single_rooted_dag(graph)

    assert updated
    assert root_texts(graph) == ["head"]
    assert graph.get_edge(head, punctuation).label == "nonsense"
    assert graph.meta["word2gp_repairs"][0]["fallback_parent"] == "head"


def load_sample(split, index):
    line = (ROOT / "data" / f"{split}.jsonl").read_text().splitlines()[index]
    return json.loads(line)


def load_train_graph(index):
    return GPGraph.parse(load_sample("train", index)["lingua"])


def load_standardized_graph(split, index):
    sample = load_sample(split, index)
    normalized, _ = normalize_lingua_edges_to_predicate_nodes(sample["lingua"])
    return sample["lingua"], GPGraph.parse(normalized)


def assert_standardized_roundtrip_exact(split, index):
    source_graph, normalized_graph = load_standardized_graph(split, index)
    word_graph, _, _ = standardized_lingua2wordlingua(source_graph)
    assert nx.is_directed_acyclic_graph(word_graph.g)

    restored, _, _ = wordlingua2lingua(copy.deepcopy(word_graph))
    metric = graph_match_metric(restored, normalized_graph, with_spans=True)

    assert metric["exact_same"], metric


def assert_roundtrip_exact(graph):
    word_graph, _, _ = lingua2wordlingua(graph)
    assert nx.is_directed_acyclic_graph(word_graph.g)

    restored, _, _ = wordlingua2lingua(word_graph)
    metric = graph_match_metric(restored, graph, with_spans=True)

    assert metric["exact_same"], metric


def test_roundtrip_handles_nested_whether_without_lifting_to_outer_aux():
    for train_index in [7507, 7908, 9480]:
        assert_roundtrip_exact(load_train_graph(train_index))


def test_roundtrip_does_not_emit_hierarchy_self_loops():
    for train_index in [1210, 1668, 3924, 4857, 4910]:
        assert_roundtrip_exact(load_train_graph(train_index))


def test_standardized_lingua2wordlingua_uses_original_aux_flattening():
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

    word_graph, updated, _ = standardized_lingua2wordlingua(graph)

    assert updated
    assert word_graph.data()["graph"]["nodes"] == [
        {"spans": ((0, 0),), "type": "ModificationalConstant"},
        {"spans": ((1, 1),), "type": "NominalConstant"},
    ]
    assert word_graph.data()["graph"]["edges"] == [
        {"start": 0, "end": 1, "label": "@modification_of"},
    ]


def test_standardized_lingua2wordlingua_flattens_punctuation_wrapper():
    graph = {
        "meta": {},
        "words": ["red", "."],
        "graph": {
            "nodes": [
                {"spans": [[0, 0]], "type": "ModificationalConstant"},
                {"spans": [[1, 1]], "type": "PunctuationalConstant"},
            ],
            "edges": [
                {"start": 1, "end": 0, "label": "as:punctuation"},
            ],
        },
    }

    normalized, _ = normalize_lingua_edges_to_predicate_nodes(graph)
    word_graph, _, _ = standardized_lingua2wordlingua(graph)

    assert word_graph.data()["graph"]["edges"] == [
        {"start": 1, "end": 0, "label": "@punctuation_of"},
    ]
    restored, _, _ = wordlingua2lingua(copy.deepcopy(word_graph))

    metric = graph_match_metric(restored, GPGraph.parse(normalized), with_spans=True)
    assert metric["exact_same"], metric


def test_wordlingua2lingua_recovers_repeated_punctuation_wrappers():
    graph = GPGraph()
    graph.words = ["Heading", ":", "."]
    heading = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="AttributePredicator"))
    colon = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="PunctuationalConstant"))
    period = graph.add_node(GPGPhraseNode(spans=[(2, 2)], pos="PunctuationalConstant"))
    graph.add_relation(period, colon, "@punctuation_of")
    graph.add_relation(period, heading, "@punctuation_of")

    restored, _, _ = wordlingua2lingua(copy.deepcopy(graph))
    expected = GPGraph.parse(
        {
            "meta": {},
            "words": ["Heading", ":", "."],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "AttributePredicator"},
                    {"spans": [[1, 1]], "type": "PunctuationalConstant"},
                    {"spans": [[2, 2]], "type": "PunctuationalConstant"},
                    {"spans": "Punctuation", "type": "PunctuationalConstant"},
                    {"spans": "Punctuation", "type": "PunctuationalConstant"},
                ],
                "edges": [
                    {"start": 3, "end": 1, "label": "pred.arg.1"},
                    {"start": 3, "end": 2, "label": "pred.arg.2"},
                    {"start": 4, "end": 0, "label": "pred.arg.1"},
                    {"start": 4, "end": 3, "label": "pred.arg.2"},
                ],
            },
        }
    )

    metric = graph_match_metric(restored, expected, with_spans=True)
    assert metric["exact_same"], metric


def test_wordlingua2lingua_lifts_child_of_whether_to_restored_aux_node():
    graph = GPGraph()
    graph.words = ["So", "any", "ideas"]
    so = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="LogicalPredicator"))
    any_node = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="DeterminerConstant"))
    ideas = graph.add_node(GPGPhraseNode(spans=[(2, 2)], pos="NominalConstant"))
    any_node.child_of_whether = True
    graph.add_relation(so, any_node, "pred.arg.2")
    graph.add_relation(any_node, ideas, "@modification_of")

    restored, _, _ = wordlingua2lingua(copy.deepcopy(graph))
    expected = GPGraph.parse(
        {
            "meta": {},
            "words": ["So", "any", "ideas"],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "LogicalPredicator"},
                    {"spans": [[1, 1]], "type": "DeterminerConstant"},
                    {"spans": [[2, 2]], "type": "NominalConstant"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                    {"spans": "Whether", "type": "GeneralFunctor"},
                ],
                "edges": [
                    {"start": 0, "end": 4, "label": "pred.arg.2"},
                    {"start": 3, "end": 2, "label": "variable"},
                    {"start": 3, "end": 1, "label": "body"},
                    {"start": 4, "end": 3, "label": "func.arg"},
                ],
            },
        }
    )

    metric = graph_match_metric(restored, expected, with_spans=True)
    assert metric["exact_same"], metric


def test_wordlingua2lingua_recovers_modification_before_punctuation_wrapper():
    graph = GPGraph()
    graph.words = ["had", ",", "too"]
    had = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="FactualPredicator"))
    comma = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="PunctuationalConstant"))
    too = graph.add_node(GPGPhraseNode(spans=[(2, 2)], pos="ModificationalConstant"))
    graph.add_relation(too, comma, "@modification_of")
    graph.add_relation(too, had, "@punctuation_of")

    restored, _, _ = wordlingua2lingua(copy.deepcopy(graph))
    expected = GPGraph.parse(
        {
            "meta": {},
            "words": ["had", ",", "too"],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "FactualPredicator"},
                    {"spans": [[1, 1]], "type": "PunctuationalConstant"},
                    {"spans": [[2, 2]], "type": "ModificationalConstant"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                    {"spans": "Punctuation", "type": "PunctuationalConstant"},
                ],
                "edges": [
                    {"start": 3, "end": 1, "label": "variable"},
                    {"start": 3, "end": 2, "label": "body"},
                    {"start": 4, "end": 0, "label": "pred.arg.1"},
                    {"start": 4, "end": 3, "label": "pred.arg.2"},
                ],
            },
        }
    )

    metric = graph_match_metric(restored, expected, with_spans=True)
    assert metric["exact_same"], metric


def test_wordlingua2lingua_orders_mixed_aux_with_lower_hierarchy():
    graph = GPGraph()
    graph.words = ["a", ",", "cream", "tea"]
    a_node = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="DeterminerConstant"))
    comma = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="PunctuationalConstant"))
    cream = graph.add_node(GPGPhraseNode(spans=[(2, 2)], pos="NominalConstant"))
    tea = graph.add_node(GPGPhraseNode(spans=[(3, 3)], pos="NominalConstant"))
    graph.add_relation(a_node, comma, "@modification_of")
    graph.add_relation(a_node, cream, "@punctuation_of")
    graph.add_relation(a_node, tea, "@modification_of")
    graph.add_relation(tea, comma, "@lower")
    graph.add_relation(tea, cream, "@lower")

    restored, _, _ = wordlingua2lingua(copy.deepcopy(graph))
    expected = GPGraph.parse(
        {
            "meta": {},
            "words": ["a", ",", "cream", "tea"],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "DeterminerConstant"},
                    {"spans": [[1, 1]], "type": "PunctuationalConstant"},
                    {"spans": [[2, 2]], "type": "NominalConstant"},
                    {"spans": [[3, 3]], "type": "NominalConstant"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                    {"spans": "Punctuation", "type": "PunctuationalConstant"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                ],
                "edges": [
                    {"start": 4, "end": 1, "label": "variable"},
                    {"start": 4, "end": 0, "label": "body"},
                    {"start": 5, "end": 2, "label": "pred.arg.1"},
                    {"start": 5, "end": 4, "label": "pred.arg.2"},
                    {"start": 6, "end": 3, "label": "variable"},
                    {"start": 6, "end": 5, "label": "body"},
                ],
            },
        }
    )

    metric = graph_match_metric(restored, expected, with_spans=True)
    assert metric["exact_same"], metric


def test_wordlingua2lingua_splits_duplicate_parent_edges_across_aux_chain():
    graph = GPGraph()
    graph.words = ["any", "questions", "or", "comments"]
    any_node = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="ModificationalConstant"))
    questions = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="NominalConstant"))
    or_node = graph.add_node(GPGPhraseNode(spans=[(2, 2)], pos="ConjunctionalFunctor"))
    comments = graph.add_node(GPGPhraseNode(spans=[(3, 3)], pos="NominalConstant"))
    graph.add_relation(any_node, questions, "@modification_of")
    graph.add_relation(any_node, comments, "@modification_of")
    graph.add_relation(comments, questions, "@lower")
    graph.add_relation(or_node, any_node, "func.arg+func.arg")

    restored, _, _ = wordlingua2lingua(copy.deepcopy(graph))
    expected = GPGraph.parse(
        {
            "meta": {},
            "words": ["any", "questions", "or", "comments"],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "ModificationalConstant"},
                    {"spans": [[1, 1]], "type": "NominalConstant"},
                    {"spans": [[2, 2]], "type": "ConjunctionalFunctor"},
                    {"spans": [[3, 3]], "type": "NominalConstant"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                ],
                "edges": [
                    {"start": 4, "end": 1, "label": "variable"},
                    {"start": 4, "end": 0, "label": "body"},
                    {"start": 5, "end": 3, "label": "variable"},
                    {"start": 5, "end": 4, "label": "body"},
                    {"start": 2, "end": 4, "label": "func.arg"},
                    {"start": 2, "end": 5, "label": "func.arg"},
                ],
            },
        }
    )

    metric = graph_match_metric(restored, expected, with_spans=True)
    assert metric["exact_same"], metric


def test_standardized_roundtrip_recovers_duplicate_modification_layers():
    for train_index in [1474, 2225, 2634, 3995, 5877, 7606, 9259, 9334]:
        assert_standardized_roundtrip_exact("train", train_index)


def test_standardized_roundtrip_keeps_whether_inside_predicator_wrappers():
    for train_index in [7507, 7908, 9480]:
        assert_standardized_roundtrip_exact("train", train_index)
    assert_standardized_roundtrip_exact("test", 1)
