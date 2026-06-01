from lingua.structure.gpgraph import GPGraph, GPGPhraseNode
from lingua.utils.metric import graph_match_metric


def make_graph(first_node_type):
    graph = GPGraph()
    graph.words = ["x", "y"]
    x_node = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos=first_node_type))
    y_node = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="NominalConstant"))
    graph.add_relation(x_node, y_node, "pred.arg.1")
    return graph


def test_graph_match_metric_is_type_aware_by_default():
    pred = make_graph("FactualPredicator")
    truth = make_graph("NominalConstant")

    metric = graph_match_metric(pred, truth)

    assert metric["matched_node_num"] == 1
    assert metric["matched_edge_num"] == 0
    assert not metric["exact_same"]
    assert metric["unmatched_pred_nodes"] == [
        {"text": "x", "type": "FactualPredicator"}
    ]
    assert metric["unmatched_true_nodes"] == [
        {"text": "x", "type": "NominalConstant"}
    ]


def test_graph_match_metric_can_ignore_type_for_legacy_text_matching():
    pred = make_graph("FactualPredicator")
    truth = make_graph("NominalConstant")

    metric = graph_match_metric(pred, truth, with_type=False)

    assert metric["matched_node_num"] == 2
    assert metric["matched_edge_num"] == 1
    assert metric["exact_same"]
