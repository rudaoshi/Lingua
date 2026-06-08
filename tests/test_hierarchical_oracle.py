import pytest

from lingua.learn.hierarchical import NON_AUX_KIND, extract_hierarchical_oracle


def labels_by_token(unit):
    return {
        token_label["token_id"]: token_label["labels"]
        for token_label in unit["token_labels"]
    }


def test_extracts_token_set_expansion_targets():
    sample = {
        "sentence": "I saw red chair",
        "lingua": {
            "words": ["I", "saw", "red", "chair"],
            "graph": {
                "nodes": [
                    {"spans": [[1, 1]], "type": "FactualPredicator"},
                    {"spans": [[0, 0]], "type": "NominalConstant"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                    {"spans": [[3, 3]], "type": "NominalConstant"},
                    {"spans": [[2, 2]], "type": "ModificationalConstant"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "pred.arg.1"},
                    {"start": 0, "end": 2, "label": "pred.arg.2"},
                    {"start": 2, "end": 3, "label": "variable"},
                    {"start": 2, "end": 4, "label": "body"},
                ],
            },
        },
    }

    oracle = extract_hierarchical_oracle(sample).to_dict()

    assert oracle["sentence"] == "I saw red chair"
    assert oracle["root_token_sets"] == [[0, 1, 2, 3]]
    assert oracle["unit_count"] == 5
    assert oracle["edge_count"] == 4

    root_unit = oracle["units"][0]
    assert root_unit["input_token_ids"] == [0, 1, 2, 3]
    assert root_unit["unit_pos"] == "FactualPredicator"
    assert root_unit["aux_kind"] == NON_AUX_KIND
    assert root_unit["decision"] == "expand"
    assert root_unit["self_token_ids"] == [1]
    assert labels_by_token(root_unit) == {
        0: ["B-ARG1"],
        1: ["B-HEAD"],
        2: ["B-ARG2"],
        3: ["I-ARG2"],
    }
    assert [
        (child["role"], child["token_ids"], child["unit_pos"], child["aux_kind"])
        for child in root_unit["children"]
    ] == [
        ("ARG1", [0], "NominalConstant", NON_AUX_KIND),
        ("ARG2", [2, 3], "ModificationalFunctor", "Modification"),
    ]

    modification_unit = next(
        unit for unit in oracle["units"] if unit["aux_kind"] == "Modification"
    )
    assert modification_unit["unit_pos"] == "ModificationalFunctor"
    assert modification_unit["is_aux"]
    assert modification_unit["input_token_ids"] == [2, 3]
    assert modification_unit["decision"] == "expand"
    assert labels_by_token(modification_unit) == {
        2: ["B-BODY"],
        3: ["B-VARIABLE"],
    }
    assert [
        (child["role"], child["token_ids"], child["unit_pos"], child["aux_kind"])
        for child in modification_unit["children"]
    ] == [
        ("VARIABLE", [3], "NominalConstant", NON_AUX_KIND),
        ("BODY", [2], "ModificationalConstant", NON_AUX_KIND),
    ]

    leaf_units = {
        tuple(unit["input_token_ids"]): unit
        for unit in oracle["units"]
        if unit["decision"] == "leaf"
    }
    assert set(leaf_units) == {(0,), (2,), (3,)}
    assert labels_by_token(leaf_units[(2,)]) == {2: ["B-HEAD"]}


def test_normalizes_as_modification_before_extracting_targets():
    sample = {
        "sentence": "saw red chair",
        "lingua": {
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
        },
    }

    oracle = extract_hierarchical_oracle(sample).to_dict()

    assert oracle["normalization_stats"]["reified_relation_edges:as:modification"] == 1
    assert oracle["unit_count"] == 4
    assert oracle["root_token_sets"] == [[0, 1, 2]]

    root_unit = oracle["units"][0]
    assert [
        (child["role"], child["token_ids"], child["unit_pos"], child["aux_kind"])
        for child in root_unit["children"]
    ] == [("ARG2", [1, 2], "ModificationalFunctor", "Modification")]
    assert labels_by_token(root_unit) == {
        0: ["B-HEAD"],
        1: ["B-ARG2"],
        2: ["I-ARG2"],
    }

    modification_unit = next(
        unit for unit in oracle["units"] if unit["aux_kind"] == "Modification"
    )
    assert modification_unit["unit_pos"] == "ModificationalFunctor"
    assert modification_unit["is_aux"]
    assert labels_by_token(modification_unit) == {
        1: ["B-BODY"],
        2: ["B-VARIABLE"],
    }
    assert [
        (child["role"], child["token_ids"], child["unit_pos"], child["aux_kind"])
        for child in modification_unit["children"]
    ] == [
        ("VARIABLE", [2], "NominalConstant", NON_AUX_KIND),
        ("BODY", [1], "ModificationalConstant", NON_AUX_KIND),
    ]


def test_preserves_dag_sharing_as_shared_token_set_links():
    sample = {
        "lingua": {
            "words": ["cats", "chase", "eat", "mice"],
            "graph": {
                "nodes": [
                    {"spans": [[1, 1]], "type": "FactualPredicator"},
                    {"spans": [[2, 2]], "type": "FactualPredicator"},
                    {"spans": [[0, 0]], "type": "NominalConstant"},
                    {"spans": [[3, 3]], "type": "NominalConstant"},
                ],
                "edges": [
                    {"start": 0, "end": 2, "label": "pred.arg.1"},
                    {"start": 0, "end": 3, "label": "pred.arg.2"},
                    {"start": 1, "end": 2, "label": "pred.arg.1"},
                    {"start": 1, "end": 3, "label": "pred.arg.2"},
                ],
            },
        }
    }

    oracle = extract_hierarchical_oracle(sample).to_dict()

    assert oracle["root_token_sets"] == [[0, 1, 3], [0, 2, 3]]
    expand_units = [unit for unit in oracle["units"] if unit["decision"] == "expand"]
    assert len(expand_units) == 2
    assert all(unit["has_shared_child"] for unit in expand_units)
    assert all(len(unit["shared_links"]) == 2 for unit in expand_units)
    assert [unit["input_token_ids"] for unit in expand_units] == [
        [0, 1, 3],
        [0, 2, 3],
    ]
    assert all(unit["is_discontinuous"] for unit in expand_units)


def test_marks_overlapping_roles_when_a_token_gets_multiple_layer_labels():
    sample = {
        "lingua": {
            "words": ["alpha", "beta"],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "FactualPredicator"},
                    {"spans": [[0, 0]], "type": "NominalConstant"},
                    {"spans": [[1, 1]], "type": "NominalConstant"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "pred.arg.1"},
                    {"start": 0, "end": 2, "label": "pred.arg.2"},
                ],
            },
        }
    }

    oracle = extract_hierarchical_oracle(sample).to_dict()
    root_unit = oracle["units"][0]

    assert root_unit["has_overlapping_roles"]
    assert labels_by_token(root_unit)[0] == ["B-ARG1", "B-HEAD"]


def test_rejects_cyclic_graphs():
    sample = {
        "lingua": {
            "words": ["a", "b"],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "NominalConstant"},
                    {"spans": [[1, 1]], "type": "NominalConstant"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "ref"},
                    {"start": 1, "end": 0, "label": "ref"},
                ],
            },
        }
    }

    with pytest.raises(ValueError, match="acyclic"):
        extract_hierarchical_oracle(sample)
