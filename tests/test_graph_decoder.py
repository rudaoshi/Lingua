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


def test_sdp_decoder_leaves_weakly_connected_multihead_graph_as_graph():
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
        enforce_single_root=False,
        decoder_mode="sdp_graph",
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

    assert sorted(root_texts(graph)) == ["Ahmed", "to"]
    assert graph.get_edge(node_at(graph, 1), node_at(graph, 0)) is None
    assert info["decoder_mode"] == "sdp_graph"
    assert info["decoder_options"]["enforce_single_root"] is False
    assert "single_root_repair_skipped" in info
    assert "validation_errors" not in info


def test_default_decoder_enforces_single_root_by_adding_scored_attachment():
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

    prefer(1, 2, "next_word")
    prefer(0, 2, "@expression")
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
    assert info["decoder_options"]["enforce_single_root"] is True
    assert info["phase4_multihead_fixes"] == []
    assert "validation_errors" not in info


def test_sdp_decoder_does_not_rewire_pred_arg1_to_common_head():
    labels = ["pred.arg.1", "pred.arg.4"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    id2node_type = {
        0: "NominalConstant",
        1: "FactualPredicator",
        2: "FactualPredicator",
    }
    decoder = GreedyPatternDecoder(
        stats=None,
        id2label=id2label,
        label2id=label2id,
        id2node_type=id2node_type,
        arc_threshold=0.5,
        decoder_mode="sdp_graph",
    )

    words = ["Seoul", "had", "to"]
    node_type_preds = np.array([0, 1, 2])
    child_of_whether = np.zeros(len(words), dtype=np.int64)
    arc_logits = np.full((len(words), len(words)), -8.0, dtype=np.float32)
    rel_logits = np.full((len(words), len(words), len(labels)), -8.0, dtype=np.float32)
    is_root_logits = np.zeros((len(words), 2), dtype=np.float32)
    is_root_logits[2, 1] = 8.0

    arc_logits[1, 0] = 8.0
    rel_logits[1, 0, label2id["pred.arg.1"]] = 8.0
    arc_logits[2, 0] = 8.0
    rel_logits[2, 0, label2id["pred.arg.1"]] = 8.0
    # Root repair can attach the extra root, but must not add a second
    # pred.arg.1 from the same predicate.
    arc_logits[2, 1] = 0.0
    rel_logits[2, 1, label2id["pred.arg.1"]] = 20.0
    rel_logits[2, 1, label2id["pred.arg.4"]] = 19.0

    graph, info = decoder.decode(
        words=words,
        arc_logits=arc_logits,
        rel_logits=rel_logits,
        node_type_preds=node_type_preds,
        child_of_whether_preds=child_of_whether,
        is_root_logits=is_root_logits,
    )

    assert graph.get_edge(node_at(graph, 1), node_at(graph, 0)) is not None
    assert graph.get_edge(node_at(graph, 2), node_at(graph, 0)) is not None
    assert graph.get_edge(node_at(graph, 1), node_at(graph, 2)) is None
    repair_edge = graph.get_edge(node_at(graph, 2), node_at(graph, 1))
    assert repair_edge is not None
    assert repair_edge.label == "pred.arg.4"
    assert info["phase4_multihead_fixes"] == []
    assert sorted(root_texts(graph)) == ["to"]
    assert "validation_errors" not in info


def test_single_root_repair_does_not_add_duplicate_pred_arg1():
    labels = ["pred.arg.1", "@attribute_of"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    id2node_type = {
        0: "FactualPredicator",
        1: "NominalConstant",
        2: "NominalConstant",
    }
    decoder = GreedyPatternDecoder(
        stats=None,
        id2label=id2label,
        label2id=label2id,
        id2node_type=id2node_type,
        arc_threshold=0.5,
        decoder_mode="sdp_graph",
    )

    words = ["likes", "Alice", "Bob"]
    node_type_preds = np.array([0, 1, 2])
    child_of_whether = np.zeros(len(words), dtype=np.int64)
    arc_logits = np.full((len(words), len(words)), -8.0, dtype=np.float32)
    rel_logits = np.full((len(words), len(words), len(labels)), -8.0, dtype=np.float32)
    is_root_logits = np.zeros((len(words), 2), dtype=np.float32)
    is_root_logits[0, 1] = 8.0

    # Phase 2 selects the first argument.
    arc_logits[0, 1] = 8.0
    rel_logits[0, 1, label2id["pred.arg.1"]] = 8.0

    # Root repair must absorb Bob, but the highest label would duplicate
    # pred.arg.1 on the same predicate. It should choose the legal label.
    arc_logits[0, 2] = 0.0
    rel_logits[0, 2, label2id["pred.arg.1"]] = 20.0
    rel_logits[0, 2, label2id["@attribute_of"]] = 19.0

    graph, info = decoder.decode(
        words=words,
        arc_logits=arc_logits,
        rel_logits=rel_logits,
        node_type_preds=node_type_preds,
        child_of_whether_preds=child_of_whether,
        is_root_logits=is_root_logits,
    )

    pred_arg1_edges = [
        (parent, child)
        for parent, edge, child in graph.edges()
        if graph.node_text(parent) == "likes" and str(edge.label) == "pred.arg.1"
    ]
    assert len(pred_arg1_edges) == 1
    repair_edge = graph.get_edge(node_at(graph, 0), node_at(graph, 2))
    assert repair_edge is not None
    assert repair_edge.label == "@attribute_of"
    assert info["single_root_repairs"][0]["label"] == "@attribute_of"
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


def test_phase5_skips_low_confidence_lower_closure():
    labels = ["@list", "@lower"]
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

    words = ["Schedule", "Standings", "References"]
    node_type_preds = np.array([0, 1, 2])
    child_of_whether = np.zeros(len(words), dtype=np.int64)
    arc_logits = np.full((len(words), len(words)), -12.0, dtype=np.float32)
    rel_logits = np.full((len(words), len(words), len(labels)), -12.0, dtype=np.float32)
    is_root_logits = np.zeros((len(words), 2), dtype=np.float32)
    is_root_logits[0, 1] = 8.0

    def prefer(parent, child, label, *, arc_logit=8.0, rel_logit=8.0):
        arc_logits[parent, child] = arc_logit
        rel_logits[parent, child, label2id[label]] = rel_logit

    prefer(0, 1, "@list")
    prefer(1, 2, "@list")
    # Phase 5 would like to add 0 -> 2 @lower for the @list chain, but
    # this edge is not model-supported.
    rel_logits[0, 2, label2id["@lower"]] = 8.0

    graph, info = decoder.decode(
        words=words,
        arc_logits=arc_logits,
        rel_logits=rel_logits,
        node_type_preds=node_type_preds,
        child_of_whether_preds=child_of_whether,
        is_root_logits=is_root_logits,
    )

    assert graph.get_edge(node_at(graph, 0), node_at(graph, 2)) is None
    assert info["phase5_upper_lower_edges"] == []
    assert any(
        item["source"] == "phase5"
        and item["label"] == "@lower"
        and item["reason"] == "arc_prob_below_threshold"
        for item in info["skipped_hierarchy_edges"]
    )


def test_phase3_does_not_use_near_zero_lower_for_connectivity():
    labels = ["as:modification", "@modification_of", "@expression", "@lower", "pred.arg.1"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    id2node_type = {
        0: "NominalConstant",
        1: "NominalConstant",
        2: "NominalConstant",
        3: "NominalConstant",
        4: "NominalConstant",
        5: "FactualPredicator",
    }
    decoder = GreedyPatternDecoder(
        stats=None,
        id2label=id2label,
        label2id=label2id,
        id2node_type=id2node_type,
        arc_threshold=0.5,
        debug=True,
    )

    words = ["UN", "Secretary", "General", "Kofi", "Annan", "is"]
    node_type_preds = np.array([0, 1, 2, 3, 4, 5])
    child_of_whether = np.zeros(len(words), dtype=np.int64)
    arc_logits = np.full((len(words), len(words)), -20.0, dtype=np.float32)
    rel_logits = np.full((len(words), len(words), len(labels)), -20.0, dtype=np.float32)
    is_root_logits = np.zeros((len(words), 2), dtype=np.float32)
    is_root_logits[5, 1] = 8.0

    def prefer(parent, child, label, *, arc_logit=8.0, rel_logit=8.0):
        arc_logits[parent, child] = arc_logit
        rel_logits[parent, child, label2id[label]] = rel_logit

    prefer(5, 0, "pred.arg.1")
    prefer(0, 1, "as:modification")
    prefer(1, 2, "as:modification")
    prefer(3, 4, "@expression")

    # This mirrors the title/person-name failure mode: the arc is strong and a
    # primary @ label is the model's favorite, but validator rejects it; @lower
    # is only a near-zero fallback and must not be used for connectivity.
    arc_logits[0, 3] = -4.0
    rel_logits[0, 3, label2id["@modification_of"]] = 20.0
    rel_logits[0, 3, label2id["@lower"]] = 0.0

    graph, info = decoder.decode(
        words=words,
        arc_logits=arc_logits,
        rel_logits=rel_logits,
        node_type_preds=node_type_preds,
        child_of_whether_preds=child_of_whether,
        is_root_logits=is_root_logits,
    )

    edge = graph.get_edge(node_at(graph, 0), node_at(graph, 3))
    assert edge is None or edge.label != "@lower"
    assert not any(
        component["selected_edge"]["label"] == "@lower"
        for component in info["phase3_components"]
        if component.get("selected_edge")
    )


def test_phase3_does_not_use_low_arc_lower_for_connectivity():
    labels = ["pred.arg.1", "@lower"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    id2node_type = {
        0: "FactualPredicator",
        1: "NominalConstant",
        2: "NominalConstant",
    }
    decoder = GreedyPatternDecoder(
        stats=None,
        id2label=id2label,
        label2id=label2id,
        id2node_type=id2node_type,
        arc_threshold=0.5,
        debug=True,
    )

    words = ["likes", "Alice", "Bob"]
    node_type_preds = np.array([0, 1, 2])
    child_of_whether = np.zeros(len(words), dtype=np.int64)
    arc_logits = np.full((len(words), len(words)), -20.0, dtype=np.float32)
    rel_logits = np.full((len(words), len(words), len(labels)), -20.0, dtype=np.float32)
    is_root_logits = np.zeros((len(words), 2), dtype=np.float32)
    is_root_logits[0, 1] = 8.0

    arc_logits[0, 1] = 8.0
    rel_logits[0, 1, label2id["pred.arg.1"]] = 8.0

    # The model likes the @lower label for this pair, but it does not believe
    # the arc exists. Connectivity repair should not use it as a hierarchy hint.
    arc_logits[0, 2] = -8.0
    rel_logits[0, 2, label2id["@lower"]] = 20.0
    rel_logits[0, 2, label2id["pred.arg.1"]] = 15.0

    graph, info = decoder.decode(
        words=words,
        arc_logits=arc_logits,
        rel_logits=rel_logits,
        node_type_preds=node_type_preds,
        child_of_whether_preds=child_of_whether,
        is_root_logits=is_root_logits,
    )

    edge = graph.get_edge(node_at(graph, 0), node_at(graph, 2))
    assert edge is None or edge.label != "@lower"
    assert not any(
        component["selected_edge"]["label"] == "@lower"
        for component in info["phase3_components"]
        if component.get("selected_edge")
    )


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


def test_word_lingua_validation_rejects_punctuation_pred_arg_parent():
    labels = ["pred.arg.1"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    decoder = GreedyPatternDecoder(
        stats=None,
        id2label=id2label,
        label2id=label2id,
        id2node_type={0: "PunctuationalConstant", 1: "NominalConstant"},
        arc_threshold=0.5,
    )

    graph = GPGraph()
    graph.words = [",", "word"]
    comma = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="PunctuationalConstant"))
    word = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="NominalConstant"))
    graph.add_edge(comma, word, GPGEdge(label="pred.arg.1"))

    errors = decoder._validate_word_lingua_graph(
        graph,
        node_types=["PunctuationalConstant", "NominalConstant"],
        decoding_info={},
    )

    assert any("PunctuationalConstant logical predicate parent" in error for error in errors)


def test_word_lingua_validation_rejects_pure_punctuation_predicator_node():
    labels = ["pred.arg.1"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    decoder = GreedyPatternDecoder(
        stats=None,
        id2label=id2label,
        label2id=label2id,
        id2node_type={0: "LogicalPredicator", 1: "NominalConstant"},
        arc_threshold=0.5,
    )

    graph = GPGraph()
    graph.words = [",", "word"]
    comma = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="LogicalPredicator"))
    word = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="NominalConstant"))
    graph.add_edge(comma, word, GPGEdge(label="pred.arg.1"))

    errors = decoder._validate_word_lingua_graph(
        graph,
        node_types=["LogicalPredicator", "NominalConstant"],
        decoding_info={},
    )

    assert any("pure punctuation/symbol predicator node" in error for error in errors)


def test_decoder_coerces_pure_punctuation_predicator_node_type():
    labels = ["next_word"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    id2node_type = {
        0: "LogicalPredicator",
        1: "NominalConstant",
        2: "PunctuationalConstant",
    }
    decoder = GreedyPatternDecoder(
        stats=None,
        id2label=id2label,
        label2id=label2id,
        id2node_type=id2node_type,
        arc_threshold=0.99,
        enforce_single_root=False,
        decoder_mode="sdp_graph",
    )

    words = [",", "word"]
    node_type_preds = np.array([0, 1])
    child_of_whether = np.zeros(len(words), dtype=np.int64)
    arc_logits = np.full((len(words), len(words)), -8.0, dtype=np.float32)
    rel_logits = np.full((len(words), len(words), len(labels)), -8.0, dtype=np.float32)
    is_root_logits = np.zeros((len(words), 2), dtype=np.float32)

    graph, info = decoder.decode(
        words=words,
        arc_logits=arc_logits,
        rel_logits=rel_logits,
        node_type_preds=node_type_preds,
        child_of_whether_preds=child_of_whether,
        is_root_logits=is_root_logits,
    )

    comma = node_at(graph, 0)
    assert comma.pos == "PunctuationalConstant"
    assert info["node_type_repairs"] == [
        {
            "token_index": 0,
            "token": ",",
            "old_type": "LogicalPredicator",
            "new_type": "PunctuationalConstant",
            "reason": "pure_punctuation_or_symbol_cannot_be_predicator",
        }
    ]


def test_word_lingua_validation_allows_duplicate_label_component_when_in_vocab():
    labels = ["func.arg+func.arg"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    decoder = GreedyPatternDecoder(
        stats=None,
        id2label=id2label,
        label2id=label2id,
        id2node_type={0: "ConjunctionalFunctor", 1: "NominalConstant"},
        arc_threshold=0.5,
    )

    graph = GPGraph()
    graph.words = ["or", "loss"]
    or_node = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="ConjunctionalFunctor"))
    loss = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="NominalConstant"))
    graph.add_edge(or_node, loss, GPGEdge(label="func.arg+func.arg"))

    errors = decoder._validate_word_lingua_graph(
        graph,
        node_types=["ConjunctionalFunctor", "NominalConstant"],
        decoding_info={},
    )

    assert errors == []


def test_ilp_rejects_nonhierarchy_edge_between_sibling_aux_children():
    labels = ["@expression", "@modification_of", "as:modification", "@lower"]
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
        decoder_mode="ilp",
        ilp_time_limit=5,
    )

    words = ["root", "expr", "modifier"]
    node_type_preds = np.array([0, 1, 2])
    child_of_whether = np.zeros(len(words), dtype=np.int64)
    arc_logits = np.full((len(words), len(words)), -12.0, dtype=np.float32)
    rel_logits = np.full((len(words), len(words), len(labels)), -12.0, dtype=np.float32)
    is_root_logits = np.zeros((len(words), 2), dtype=np.float32)
    is_root_logits[0, 1] = 8.0

    def prefer(parent, child, label, *, arc_logit=10.0, rel_logit=12.0):
        arc_logits[parent, child] = arc_logit
        rel_logits[parent, child, label2id[label]] = rel_logit

    prefer(0, 1, "@expression", arc_logit=12.0)
    prefer(0, 2, "@modification_of", arc_logit=12.0)
    # Without the aux-child subgraph constraint, this attractive ordinary edge
    # would sit between two sibling aux children and later crash word2gp.
    prefer(1, 2, "as:modification", arc_logit=10.0)

    graph, info = decoder.decode(
        words=words,
        arc_logits=arc_logits,
        rel_logits=rel_logits,
        node_type_preds=node_type_preds,
        child_of_whether_preds=child_of_whether,
        is_root_logits=is_root_logits,
    )

    assert graph.get_edge(node_at(graph, 0), node_at(graph, 1)).label == "@expression"
    assert graph.get_edge(node_at(graph, 0), node_at(graph, 2)).label == "@modification_of"
    assert graph.get_edge(node_at(graph, 1), node_at(graph, 2)) is None
    assert info["ilp"]["aux_child_subgraph_constraints"] > 0
    assert "validation_errors" not in info


def test_ilp_as_shorthand_edge_excludes_other_outgoing_edges():
    labels = ["as:modification", "@expression"]
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
        decoder_mode="ilp",
        ilp_time_limit=5,
    )

    words = ["head", "short", "tail"]
    node_type_preds = np.array([0, 1, 2])
    child_of_whether = np.zeros(len(words), dtype=np.int64)
    arc_logits = np.full((len(words), len(words)), -12.0, dtype=np.float32)
    rel_logits = np.full((len(words), len(words), len(labels)), -12.0, dtype=np.float32)
    is_root_logits = np.full((len(words), 2), -8.0, dtype=np.float32)
    is_root_logits[0, 1] = 8.0

    def prefer(parent, child, label, *, arc_logit=10.0, rel_logit=12.0):
        arc_logits[parent, child] = arc_logit
        rel_logits[parent, child, label2id[label]] = rel_logit

    prefer(0, 1, "as:modification", arc_logit=12.0)
    # This edge is attractive, but selecting it with 0 -> 1 as:modification
    # would give node 0 an invalid shorthand plus extra outgoing structure.
    prefer(0, 2, "@expression", arc_logit=11.0)
    prefer(1, 2, "@expression", arc_logit=9.0)

    graph, info = decoder.decode(
        words=words,
        arc_logits=arc_logits,
        rel_logits=rel_logits,
        node_type_preds=node_type_preds,
        child_of_whether_preds=child_of_whether,
        is_root_logits=is_root_logits,
    )

    assert graph.get_edge(node_at(graph, 0), node_at(graph, 1)).label == "as:modification"
    assert graph.get_edge(node_at(graph, 0), node_at(graph, 2)) is None
    assert graph.get_edge(node_at(graph, 1), node_at(graph, 2)).label == "@expression"
    assert info["ilp"]["as_shorthand_exclusion_constraints"] > 0
    assert "validation_errors" not in info


def test_ilp_rejects_punctuation_pred_arg_parent_even_when_high_scoring():
    labels = ["pred.arg.1", "@expression"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    id2node_type = {
        0: "PunctuationalConstant",
        1: "NominalConstant",
        2: "FactualPredicator",
    }
    decoder = GreedyPatternDecoder(
        stats=None,
        id2label=id2label,
        label2id=label2id,
        id2node_type=id2node_type,
        arc_threshold=0.5,
        decoder_mode="ilp",
        ilp_time_limit=5,
    )

    words = [",", "Alice", "likes"]
    node_type_preds = np.array([0, 1, 2])
    child_of_whether = np.zeros(len(words), dtype=np.int64)
    arc_logits = np.full((len(words), len(words)), -12.0, dtype=np.float32)
    rel_logits = np.full((len(words), len(words), len(labels)), -12.0, dtype=np.float32)
    is_root_logits = np.full((len(words), 2), -8.0, dtype=np.float32)
    is_root_logits[2, 1] = 8.0

    def prefer(parent, child, label, *, arc_logit=10.0, rel_logit=12.0):
        arc_logits[parent, child] = arc_logit
        rel_logits[parent, child, label2id[label]] = rel_logit

    prefer(2, 0, "@expression", arc_logit=12.0)
    prefer(2, 1, "pred.arg.1", arc_logit=12.0)
    # This edge is even more attractive, but punctuation tokens cannot be
    # logical predicate parents.
    prefer(0, 1, "pred.arg.1", arc_logit=14.0, rel_logit=14.0)

    graph, info = decoder.decode(
        words=words,
        arc_logits=arc_logits,
        rel_logits=rel_logits,
        node_type_preds=node_type_preds,
        child_of_whether_preds=child_of_whether,
        is_root_logits=is_root_logits,
    )

    assert graph.get_edge(node_at(graph, 2), node_at(graph, 0)).label == "@expression"
    assert graph.get_edge(node_at(graph, 2), node_at(graph, 1)).label == "pred.arg.1"
    assert graph.get_edge(node_at(graph, 0), node_at(graph, 1)) is None
    assert info["ilp"]["static_reject_counts"]["PUNCTUATIONAL_PRED_ARG_PARENT"] >= 1
    assert "validation_errors" not in info
