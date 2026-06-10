from lingua.utils.normalization import (
    normalize_lingua_edges_to_predicate_nodes,
    normalize_simple_leaf_modifications,
)


def assert_only_standardized_edges(graph):
    labels = [edge["label"] for edge in graph["graph"]["edges"]]
    assert all(
        label in {"func.arg", "punctuation", "repeat", "variable", "body"}
        or label.startswith("pred.arg.")
        for label in labels
    )


def root_indices(graph):
    node_count = len(graph["graph"]["nodes"])
    indegree = [0] * node_count
    for edge in graph["graph"]["edges"]:
        indegree[int(edge["end"])] += 1
    return [index for index, degree in enumerate(indegree) if degree == 0]


def test_normalize_simple_leaf_modification_to_as_modification():
    graph = {
        "meta": {},
        "words": ["red", "chair"],
        "graph": {
            "nodes": [
                {"spans": "Modification", "type": "ModificationalFunctor"},
                {"spans": [[1, 1]], "type": "NominalConstant"},
                {"spans": [[0, 0]], "type": "ModificationalConstant"},
            ],
            "edges": [
                {"start": 0, "end": 1, "label": "variable"},
                {"start": 0, "end": 2, "label": "body"},
            ],
        },
    }

    normalized, stats = normalize_simple_leaf_modifications(graph)

    assert stats == {"flattened_simple_leaf_modifications": 1}
    assert normalized["graph"]["nodes"] == [
        {"spans": [[1, 1]], "type": "NominalConstant"},
        {"spans": [[0, 0]], "type": "ModificationalConstant"},
    ]
    assert normalized["graph"]["edges"] == [
        {"start": 1, "end": 0, "label": "as:modification"}
    ]


def test_normalize_simple_leaf_modification_rewires_incoming_parent():
    graph = {
        "meta": {},
        "words": ["saw", "red", "chair"],
        "graph": {
            "nodes": [
                {"spans": [[0, 0]], "type": "FactualPredicator"},
                {"spans": "Modification", "type": "ModificationalFunctor"},
                {"spans": [[2, 2]], "type": "NominalConstant"},
                {"spans": [[1, 1]], "type": "ModificationalConstant"},
            ],
            "edges": [
                {"start": 0, "end": 1, "label": "pred.arg.2"},
                {"start": 1, "end": 2, "label": "variable"},
                {"start": 1, "end": 3, "label": "body"},
            ],
        },
    }

    normalized, stats = normalize_simple_leaf_modifications(graph)

    assert stats == {"flattened_simple_leaf_modifications": 1}
    assert normalized["graph"]["edges"] == [
        {"start": 0, "end": 2, "label": "pred.arg.2"},
        {"start": 2, "end": 1, "label": "as:modification"},
    ]


def test_normalize_simple_leaf_modification_keeps_complex_body():
    graph = {
        "meta": {},
        "words": ["chair", "near", "door"],
        "graph": {
            "nodes": [
                {"spans": "Modification", "type": "ModificationalFunctor"},
                {"spans": [[0, 0]], "type": "NominalConstant"},
                {"spans": [[1, 1]], "type": "ModificationalFunctor"},
                {"spans": [[2, 2]], "type": "NominalConstant"},
            ],
            "edges": [
                {"start": 0, "end": 1, "label": "variable"},
                {"start": 0, "end": 2, "label": "body"},
                {"start": 2, "end": 3, "label": "body"},
                {"start": 2, "end": 1, "label": "variable"},
            ],
        },
    }

    normalized, stats = normalize_simple_leaf_modifications(graph)

    assert stats == {}
    assert normalized == graph


def test_normalize_lingua_edges_to_predicate_nodes_reifies_as_modification():
    graph = {
        "meta": {},
        "words": ["saw", "red", "chair"],
        "graph": {
            "nodes": [
                {"spans": [[0, 0]], "type": "FactualPredicator"},
                {"spans": [[2, 2]], "type": "NominalConstant"},
                {"spans": [[1, 1]], "type": "ModificationalConstant"},
            ],
            "edges": [
                {"start": 0, "end": 2, "label": "pred.arg.2"},
                {"start": 2, "end": 1, "label": "as:modification"},
            ],
        },
    }

    normalized, stats = normalize_lingua_edges_to_predicate_nodes(graph)

    assert stats["reified_relation_edges"] == 1
    assert stats["reified_relation_edges:as:modification"] == 1
    assert stats["rewired_as_relation_roots"] == 1
    assert normalized["graph"]["nodes"] == [
        {"spans": [[0, 0]], "type": "FactualPredicator"},
        {"spans": [[2, 2]], "type": "NominalConstant"},
        {"spans": [[1, 1]], "type": "ModificationalConstant"},
        {"spans": "Modification", "type": "ModificationalFunctor"},
    ]
    assert normalized["graph"]["edges"] == [
        {"start": 0, "end": 3, "label": "pred.arg.2"},
        {"start": 3, "end": 1, "label": "variable"},
        {"start": 3, "end": 2, "label": "body"},
    ]
    assert_only_standardized_edges(normalized)


def test_normalize_lingua_edges_to_predicate_nodes_keeps_variable_body():
    graph = {
        "meta": {},
        "words": ["near", "door", "chair"],
        "graph": {
            "nodes": [
                {"spans": [[0, 0]], "type": "ModificationalFunctor"},
                {"spans": [[2, 2]], "type": "NominalConstant"},
                {"spans": [[1, 1]], "type": "NominalConstant"},
            ],
            "edges": [
                {"start": 0, "end": 1, "label": "variable"},
                {"start": 0, "end": 2, "label": "body"},
            ],
        },
    }

    normalized, stats = normalize_lingua_edges_to_predicate_nodes(graph)

    assert stats == {}
    assert normalized["graph"]["nodes"] == graph["graph"]["nodes"]
    assert normalized["graph"]["edges"] == [
        {"start": 0, "end": 1, "label": "variable"},
        {"start": 0, "end": 2, "label": "body"},
    ]
    assert_only_standardized_edges(normalized)


def test_normalize_lingua_edges_to_predicate_nodes_reifies_punctuation_and_preserves_func_arg():
    graph = {
        "meta": {},
        "words": ["A", ",", "B"],
        "graph": {
            "nodes": [
                {"spans": [[1, 1]], "type": "PunctuationalConstant"},
                {"spans": [[0, 0]], "type": "NominalConstant"},
                {"spans": [[2, 2]], "type": "NominalConstant"},
                {"spans": "Expression", "type": "ExpressionFunctor"},
            ],
            "edges": [
                {"start": 0, "end": 1, "label": "as:punctuation", "confidence": 0.8},
                {"start": 3, "end": 1, "label": "func.arg"},
                {"start": 3, "end": 2, "label": "func.arg"},
            ],
        },
    }

    normalized, stats = normalize_lingua_edges_to_predicate_nodes(graph)

    assert stats["reified_relation_edges:as:punctuation"] == 1
    assert normalized["graph"]["nodes"][-1] == {
        "spans": "Punctuation",
        "type": "PunctuationalConstant",
        "confidence": 0.8,
    }
    assert normalized["graph"]["edges"] == [
        {"start": 4, "end": 1, "label": "pred.arg.1", "confidence": 0.8},
        {"start": 4, "end": 0, "label": "pred.arg.2", "confidence": 0.8},
        {"start": 3, "end": 1, "label": "func.arg"},
        {"start": 3, "end": 2, "label": "func.arg"},
    ]
    assert_only_standardized_edges(normalized)


def test_normalize_lingua_edges_to_predicate_nodes_keeps_plain_punctuation_and_repeat():
    graph = {
        "meta": {},
        "words": ["go", "go", "--"],
        "graph": {
            "nodes": [
                {"spans": [[0, 0]], "type": "FactualPredicator"},
                {"spans": [[1, 1]], "type": "FactualPredicator"},
                {"spans": [[2, 2]], "type": "PunctuationalConstant"},
            ],
            "edges": [
                {"start": 0, "end": 1, "label": "repeat"},
                {"start": 0, "end": 2, "label": "punctuation"},
            ],
        },
    }

    normalized, stats = normalize_lingua_edges_to_predicate_nodes(graph)

    assert stats == {}
    assert normalized == graph
    assert_only_standardized_edges(normalized)


def test_normalize_lingua_edges_to_predicate_nodes_assigns_contiguous_aux_indices():
    graph = {
        "meta": {},
        "words": ["red", "chair", "."],
        "graph": {
            "nodes": [
                {"spans": [[0, 0]], "type": "ModificationalConstant"},
                {"spans": [[1, 1]], "type": "NominalConstant"},
                {"spans": [[2, 2]], "type": "PunctuationalConstant"},
            ],
            "edges": [
                {"start": 0, "end": 1, "label": "as:modification"},
                {"start": 2, "end": 0, "label": "as:punctuation"},
            ],
        },
    }

    normalized, stats = normalize_lingua_edges_to_predicate_nodes(graph)

    assert stats["reified_relation_edges"] == 2
    assert len(normalized["graph"]["nodes"]) == 5
    assert normalized["graph"]["nodes"][3]["spans"] == "Modification"
    assert normalized["graph"]["nodes"][4]["spans"] == "Punctuation"
    assert all(edge["start"] < 5 and edge["end"] < 5 for edge in normalized["graph"]["edges"])
    assert_only_standardized_edges(normalized)


def test_normalize_lingua_edges_to_predicate_nodes_folds_as_modification_chain_to_single_root():
    graph = {
        "meta": {},
        "words": ["saw", "very", "red", "chair"],
        "graph": {
            "nodes": [
                {"spans": [[0, 0]], "type": "FactualPredicator"},
                {"spans": [[1, 1]], "type": "ModificationalConstant"},
                {"spans": [[2, 2]], "type": "ModificationalConstant"},
                {"spans": [[3, 3]], "type": "NominalConstant"},
            ],
            "edges": [
                {"start": 0, "end": 1, "label": "pred.arg.2"},
                {"start": 1, "end": 2, "label": "as:modification"},
                {"start": 2, "end": 3, "label": "as:modification"},
            ],
        },
    }

    normalized, stats = normalize_lingua_edges_to_predicate_nodes(graph)

    assert stats["reified_relation_edges"] == 2
    assert stats["folded_as_relation_bodies"] == 1
    assert stats["folded_as_relation_root_rewires"] == 1
    assert normalized["graph"]["nodes"][4:] == [
        {"spans": "Modification", "type": "ModificationalFunctor"},
        {"spans": "Modification", "type": "ModificationalFunctor"},
    ]
    assert normalized["graph"]["edges"] == [
        {"start": 0, "end": 5, "label": "pred.arg.2"},
        {"start": 4, "end": 2, "label": "variable"},
        {"start": 4, "end": 1, "label": "body"},
        {"start": 5, "end": 3, "label": "variable"},
        {"start": 5, "end": 4, "label": "body"},
    ]
    assert root_indices(normalized) == [0]
    assert_only_standardized_edges(normalized)


def test_normalize_lingua_edges_to_predicate_nodes_wraps_modification_with_punctuation():
    graph = {
        "meta": {},
        "words": ["red", "chair", "."],
        "graph": {
            "nodes": [
                {"spans": [[0, 0]], "type": "ModificationalConstant"},
                {"spans": [[1, 1]], "type": "NominalConstant"},
                {"spans": [[2, 2]], "type": "PunctuationalConstant"},
            ],
            "edges": [
                {"start": 0, "end": 1, "label": "as:modification"},
                {"start": 2, "end": 0, "label": "as:punctuation"},
            ],
        },
    }

    normalized, stats = normalize_lingua_edges_to_predicate_nodes(graph)

    assert stats["reified_relation_edges"] == 2
    assert stats.get("folded_as_relation_bodies", 0) == 0
    assert normalized["graph"]["nodes"][3:] == [
        {"spans": "Modification", "type": "ModificationalFunctor"},
        {"spans": "Punctuation", "type": "PunctuationalConstant"},
    ]
    assert normalized["graph"]["edges"] == [
        {"start": 3, "end": 1, "label": "variable"},
        {"start": 3, "end": 0, "label": "body"},
        {"start": 4, "end": 3, "label": "pred.arg.1"},
        {"start": 4, "end": 2, "label": "pred.arg.2"},
    ]
    assert root_indices(normalized) == [4]
    assert_only_standardized_edges(normalized)


def test_normalize_lingua_edges_to_predicate_nodes_folds_coordinated_modifier_group():
    graph = {
        "meta": {},
        "words": ["found", "and", "pre", "post", "menopausal", "women"],
        "graph": {
            "nodes": [
                {"spans": [[0, 0]], "type": "FactualPredicator"},
                {"spans": [[1, 1]], "type": "ConjunctionalFunctor"},
                {"spans": [[2, 2]], "type": "ModificationalConstant"},
                {"spans": [[3, 3]], "type": "ModificationalConstant"},
                {"spans": [[4, 4]], "type": "ModificationalConstant"},
                {"spans": [[5, 5]], "type": "NominalConstant"},
            ],
            "edges": [
                {"start": 0, "end": 1, "label": "pred.arg.2"},
                {"start": 1, "end": 2, "label": "func.arg"},
                {"start": 1, "end": 3, "label": "func.arg"},
                {"start": 2, "end": 4, "label": "as:modification"},
                {"start": 3, "end": 4, "label": "as:modification"},
                {"start": 4, "end": 5, "label": "as:modification"},
            ],
        },
    }

    normalized, stats = normalize_lingua_edges_to_predicate_nodes(graph)

    assert stats["folded_ambiguous_as_relation_bodies_with_func_arg_parent"] == 1
    assert stats["rewired_grouped_as_relation_children"] == 2
    assert stats["rewired_group_parent_as_relation_roots"] == 1
    assert normalized["graph"]["nodes"][6:] == [
        {"spans": "Modification", "type": "ModificationalFunctor"},
        {"spans": "Modification", "type": "ModificationalFunctor"},
        {"spans": "Modification", "type": "ModificationalFunctor"},
    ]
    assert normalized["graph"]["edges"] == [
        {"start": 0, "end": 8, "label": "pred.arg.2"},
        {"start": 1, "end": 6, "label": "func.arg"},
        {"start": 1, "end": 7, "label": "func.arg"},
        {"start": 6, "end": 4, "label": "variable"},
        {"start": 6, "end": 2, "label": "body"},
        {"start": 7, "end": 4, "label": "variable"},
        {"start": 7, "end": 3, "label": "body"},
        {"start": 8, "end": 5, "label": "variable"},
        {"start": 8, "end": 1, "label": "body"},
    ]
    assert root_indices(normalized) == [0]
    assert_only_standardized_edges(normalized)
