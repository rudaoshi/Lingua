from lingua.learn.wordgraph.modeler.word2gp import (
    ensure_single_rooted_dag,
    strip_hierarchy_edge_components,
    wordlingua2lingua,
)
from lingua.structure.gpgraph import GPGraph, GPGPhraseNode


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
