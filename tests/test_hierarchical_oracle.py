import pytest

from lingua.learn.hierarchical import extract_hierarchical_oracle


def test_extracts_one_layer_expansion_targets():
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
    assert oracle["root_ids"] == [0]
    assert oracle["node_count"] == 5
    assert oracle["edge_count"] == 4

    root_unit = oracle["units"][0]
    assert root_unit["node"]["text"] == "saw"
    assert root_unit["decision"] == "expand"
    assert root_unit["descendant_count"] == 4
    assert [child["label"] for child in root_unit["children"]] == ["pred.arg.1", "pred.arg.2"]

    modification_unit = next(
        unit for unit in oracle["units"] if unit["node"]["text"] == "Modification"
    )
    assert modification_unit["decision"] == "expand"
    assert [child["child"]["text"] for child in modification_unit["children"]] == [
        "chair",
        "red",
    ]

    leaf_units = {unit["node"]["text"] for unit in oracle["units"] if unit["decision"] == "leaf"}
    assert leaf_units == {"I", "chair", "red"}


def test_preserves_dag_sharing_instead_of_tree_expansion():
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

    assert oracle["root_ids"] == [0, 1]
    expand_units = [unit for unit in oracle["units"] if unit["decision"] == "expand"]
    assert len(expand_units) == 2
    assert all(unit["has_shared_child"] for unit in expand_units)
    assert all(
        child["child_is_shared"]
        for unit in expand_units
        for child in unit["children"]
    )


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
