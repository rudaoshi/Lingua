import numpy as np

from lingua.learn.wordgraph.decoding.graph_decoder import GreedyPatternDecoder
from lingua.structure.gpgraph import GPGraph, GPGEdge, GPGPhraseNode


def root_texts(graph):
    return [
        graph.node_text(node)
        for node in graph.nodes()
        if graph.g.in_degree(node.ID) == 0
    ]


def node_at(graph, word_idx):
    for node in graph.nodes():
        if tuple(getattr(node, "spans", ())) == ((word_idx, word_idx),):
            return node
    raise AssertionError(f"missing node at word index {word_idx}")


def test_decoder_enforces_single_root_for_weakly_connected_multihead_graph():
    labels = ["next_word", "@expression", "@lower"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    id2node_type = {
        0: "NominalConstant",
        1: "FactualPredicator",
        2: "NominalConstant",
    }
    decoder = GreedyPatternDecoder(
        stats=None,
        id2label=id2label,
        label2id=label2id,
        id2node_type=id2node_type,
        arc_threshold=0.5,
    )

    words = ["Ahmed", "to", "Alhaznawi"]
    node_type_preds = np.array([0, 1, 2])
    child_of_whether = np.zeros(len(words), dtype=np.int64)
    arc_logits = np.full((len(words), len(words)), -8.0, dtype=np.float32)
    rel_logits = np.full((len(words), len(words), len(labels)), -8.0, dtype=np.float32)
    is_root_logits = np.zeros((len(words), 2), dtype=np.float32)
    is_root_logits[1, 1] = 8.0

    def prefer(parent, child, label, *, arc_logit=8.0, rel_logit=8.0):
        arc_logits[parent, child] = arc_logit
        rel_logits[parent, child, label2id[label]] = rel_logit

    # These high-confidence edges create one weak component with two roots:
    # to -> Alhaznawi <- Ahmed.
    prefer(1, 2, "next_word")
    prefer(0, 2, "@expression")
    # This low-arc candidate should not be selected in Phase 2, but it is the
    # best model-scored way to absorb the remaining extra root at the end.
    prefer(1, 0, "@expression", arc_logit=-4.0)

    graph, info = decoder.decode(
        words=words,
        arc_logits=arc_logits,
        rel_logits=rel_logits,
        node_type_preds=node_type_preds,
        child_of_whether_preds=child_of_whether,
        is_root_logits=is_root_logits,
    )

    assert root_texts(graph) == ["to"]
    assert graph.get_edge(node_at(graph, 1), node_at(graph, 0)) is not None
    assert info["single_root_repairs"][0]["parent"] == 1
    assert "validation_errors" not in info


def test_decoder_allows_single_lower_label_from_training_vocab():
    labels = ["@expression", "@lower"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    id2node_type = {
        0: "NominalConstant",
        1: "NominalConstant",
        2: "NominalConstant",
    }
    decoder = GreedyPatternDecoder(
        stats=None,
        id2label=id2label,
        label2id=label2id,
        id2node_type=id2node_type,
        arc_threshold=0.5,
    )

    words = ["A", "B", "C"]
    node_type_preds = np.array([0, 1, 2])
    child_of_whether = np.zeros(len(words), dtype=np.int64)
    arc_logits = np.full((len(words), len(words)), -8.0, dtype=np.float32)
    rel_logits = np.full((len(words), len(words), len(labels)), -8.0, dtype=np.float32)
    is_root_logits = np.zeros((len(words), 2), dtype=np.float32)
    is_root_logits[0, 1] = 8.0

    def prefer(parent, child, label, *, arc_logit=8.0, rel_logit=8.0):
        arc_logits[parent, child] = arc_logit
        rel_logits[parent, child, label2id[label]] = rel_logit

    prefer(0, 1, "@expression", arc_logit=7.0, rel_logit=7.0)
    prefer(1, 2, "@lower", arc_logit=8.0, rel_logit=8.0)

    graph, info = decoder.decode(
        words=words,
        arc_logits=arc_logits,
        rel_logits=rel_logits,
        node_type_preds=node_type_preds,
        child_of_whether_preds=child_of_whether,
        is_root_logits=is_root_logits,
    )

    lower_edge = graph.get_edge(node_at(graph, 1), node_at(graph, 2))
    assert lower_edge is not None
    assert lower_edge.label == "@lower"
    assert "validation_errors" not in info


def test_word_lingua_validation_requires_full_edge_label_in_training_vocab():
    labels = ["@lower"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    decoder = GreedyPatternDecoder(
        stats=None,
        id2label=id2label,
        label2id=label2id,
        id2node_type={0: "NominalConstant"},
        arc_threshold=0.5,
    )

    graph = GPGraph()
    graph.words = ["A", "B"]
    a_node = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="NominalConstant"))
    b_node = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="NominalConstant"))
    graph.add_edge(a_node, b_node, GPGEdge(label="@lower+@lower"))

    errors = decoder._validate_word_lingua_graph(
        graph,
        node_types=["NominalConstant", "NominalConstant"],
        decoding_info={},
    )

    assert any("not in training vocabulary" in error for error in errors)
