import json
from collections import defaultdict
from pathlib import Path

import pytest

from lingua.learn.hierarchical import (
    CONTENT_VIEW,
    DEFAULT_VIEW,
    NON_AUX_KIND,
    extract_hierarchical_oracle,
)
from lingua.learn.hierarchical.dataset import build_hierarchical_examples_from_oracle


ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = [
    ROOT / "data" / "train.jsonl",
    ROOT / "data" / "dev.jsonl",
    ROOT / "data" / "test.jsonl",
]


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
    assert root_unit["view_kind"] == DEFAULT_VIEW
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
    assert modification_unit["view_kind"] == DEFAULT_VIEW
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
    assert modification_unit["view_kind"] == DEFAULT_VIEW
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


def test_encodes_sibling_contained_descendants_as_shared_args():
    sample = {
        "lingua": {
            "words": ["volume", "submitted"],
            "graph": {
                "nodes": [
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                    {"spans": [[0, 0]], "type": "NominalConstant"},
                    {"spans": [[1, 1]], "type": "FactualPredicator"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "variable"},
                    {"start": 0, "end": 2, "label": "body"},
                    {"start": 2, "end": 1, "label": "pred.arg.1"},
                ],
            },
        }
    }

    oracle = extract_hierarchical_oracle(sample).to_dict()
    modification_unit = oracle["units"][0]
    body_child = next(
        child for child in modification_unit["children"] if child["role"] == "BODY_REF"
    )

    assert modification_unit["input_token_ids"] == [0, 1]
    assert modification_unit["view_kind"] == DEFAULT_VIEW
    assert not modification_unit["has_overlapping_roles"]
    assert modification_unit["has_shared_arg"]
    assert labels_by_token(modification_unit) == {
        0: ["B-VARIABLE"],
        1: ["O"],
    }
    assert body_child["token_ids"] == [0, 1]
    assert body_child["label_token_ids"] == []
    assert body_child["shared_arg_links"] == [
        {
            "owner_role": "BODY_REF",
            "shared_role": "VARIABLE",
            "token_ids": [0],
            "tokens": ["volume"],
            "text": "volume",
            "shared_unit_pos": "NominalConstant",
            "shared_aux_kind": NON_AUX_KIND,
        }
    ]
    assert modification_unit["shared_arg_links"] == body_child["shared_arg_links"]


def test_complex_modification_wrapper_uses_default_view_body_ref():
    sample = {
        "lingua": {
            "words": ["nothing", "to", "do", "with", "Israel"],
            "graph": {
                "nodes": [
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                    {"spans": [[0, 0]], "type": "NominalConstant"},
                    {"spans": [[3, 3]], "type": "ModificationalFunctor"},
                    {"spans": [[1, 2]], "type": "FactualPredicator"},
                    {"spans": [[4, 4]], "type": "NominalConstant"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "variable"},
                    {"start": 0, "end": 2, "label": "body"},
                    {"start": 2, "end": 3, "label": "variable"},
                    {"start": 2, "end": 4, "label": "body"},
                    {"start": 3, "end": 1, "label": "pred.arg.1"},
                ],
            },
        }
    }

    oracle = extract_hierarchical_oracle(sample).to_dict()
    attach_unit = next(unit for unit in oracle["units"] if unit["aux_kind"] == "Modification")
    content_unit = next(
        unit
        for unit in oracle["units"]
        if unit["unit_pos"] == "ModificationalFunctor" and unit["aux_kind"] == NON_AUX_KIND
    )

    assert attach_unit["view_kind"] == DEFAULT_VIEW
    assert attach_unit["input_token_ids"] == [0, 1, 2, 3, 4]
    assert labels_by_token(attach_unit) == {
        0: ["B-VARIABLE"],
        1: ["O"],
        2: ["O"],
        3: ["O"],
        4: ["O"],
    }
    assert [
        (child["role"], child["token_ids"], child["label_token_ids"])
        for child in attach_unit["children"]
    ] == [
        ("VARIABLE", [0], [0]),
        ("BODY_REF", [0, 1, 2, 3, 4], []),
    ]

    assert content_unit["view_kind"] == CONTENT_VIEW
    assert labels_by_token(content_unit) == {
        0: ["B-VARIABLE"],
        1: ["I-VARIABLE"],
        2: ["I-VARIABLE"],
        3: ["B-HEAD"],
        4: ["B-BODY"],
    }


def test_same_span_whether_wrapper_uses_default_view_func_arg_ref():
    sample = {
        "lingua": {
            "words": ["Are", "you", "ready"],
            "graph": {
                "nodes": [
                    {"spans": "Whether", "type": "GeneralFunctor"},
                    {"spans": [[0, 0]], "type": "FactualPredicator"},
                    {"spans": [[1, 1]], "type": "NominalConstant"},
                    {"spans": [[2, 2]], "type": "ModificationalConstant"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "func.arg"},
                    {"start": 1, "end": 2, "label": "pred.arg.1"},
                    {"start": 1, "end": 3, "label": "pred.arg.2"},
                ],
            },
        }
    }

    oracle = extract_hierarchical_oracle(sample).to_dict()
    whether_unit = oracle["units"][0]
    content_unit = oracle["units"][1]

    assert whether_unit["input_token_ids"] == [0, 1, 2]
    assert content_unit["input_token_ids"] == [0, 1, 2]
    assert whether_unit["view_kind"] == DEFAULT_VIEW
    assert content_unit["view_kind"] == CONTENT_VIEW
    assert whether_unit["unit_pos"] == "GeneralFunctor"
    assert whether_unit["aux_kind"] == "Whether"
    assert labels_by_token(whether_unit) == {
        0: ["O"],
        1: ["O"],
        2: ["O"],
    }
    assert [
        (child["role"], child["token_ids"], child["label_token_ids"])
        for child in whether_unit["children"]
    ] == [
        ("FUNC_ARG_REF", [0, 1, 2], []),
    ]

    assert content_unit["unit_pos"] == "FactualPredicator"
    assert content_unit["aux_kind"] == NON_AUX_KIND
    assert labels_by_token(content_unit) == {
        0: ["B-HEAD"],
        1: ["B-ARG1"],
        2: ["B-ARG2"],
    }


def test_question_punctuation_stays_under_whether_default_view():
    sample = {
        "lingua": {
            "words": ["Are", "you", "ready", "?"],
            "graph": {
                "nodes": [
                    {"spans": "Whether", "type": "GeneralFunctor"},
                    {"spans": [[0, 0]], "type": "FactualPredicator"},
                    {"spans": [[1, 1]], "type": "NominalConstant"},
                    {"spans": [[2, 2]], "type": "ModificationalConstant"},
                    {"spans": [[3, 3]], "type": "PunctuationalConstant"},
                ],
                "edges": [
                    {"start": 0, "end": 4, "label": "func.arg"},
                    {"start": 4, "end": 1, "label": "as:punctuation"},
                    {"start": 1, "end": 2, "label": "pred.arg.1"},
                    {"start": 1, "end": 3, "label": "pred.arg.2"},
                ],
            },
        }
    }

    oracle = extract_hierarchical_oracle(sample).to_dict()
    whether_unit = next(unit for unit in oracle["units"] if unit["aux_kind"] == "Whether")
    punctuation_unit = next(unit for unit in oracle["units"] if unit["aux_kind"] == "Punctuation")
    proposition_unit = next(
        unit for unit in oracle["units"] if unit["unit_pos"] == "FactualPredicator"
    )
    question_unit = next(
        unit
        for unit in oracle["units"]
        if unit["unit_pos"] == "PunctuationalConstant" and unit["aux_kind"] == NON_AUX_KIND
    )

    assert whether_unit["input_token_ids"] == [0, 1, 2, 3]
    assert whether_unit["view_kind"] == DEFAULT_VIEW
    assert whether_unit["unit_pos"] == "GeneralFunctor"
    assert whether_unit["aux_kind"] == "Whether"
    assert labels_by_token(whether_unit) == {
        0: ["O"],
        1: ["O"],
        2: ["O"],
        3: ["O"],
    }
    assert [
        (child["role"], child["token_ids"], child["label_token_ids"])
        for child in whether_unit["children"]
    ] == [
        ("FUNC_ARG_REF", [0, 1, 2, 3], []),
    ]

    assert punctuation_unit["view_kind"] == CONTENT_VIEW
    assert punctuation_unit["unit_pos"] == "PunctuationalConstant"
    assert labels_by_token(punctuation_unit) == {
        0: ["B-ARG1"],
        1: ["I-ARG1"],
        2: ["I-ARG1"],
        3: ["B-ARG2"],
    }
    assert [
        (child["role"], child["token_ids"], child["label_token_ids"])
        for child in punctuation_unit["children"]
    ] == [
        ("ARG1", [0, 1, 2], [0, 1, 2]),
        ("ARG2", [3], [3]),
    ]

    assert proposition_unit["view_kind"] == DEFAULT_VIEW
    assert proposition_unit["unit_pos"] == "FactualPredicator"
    assert question_unit["input_token_ids"] == [3]


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


def test_corpus_has_no_query_context_decode_conflicts():
    failures = []

    for path in DATA_PATHS:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                oracle = extract_hierarchical_oracle(sample).to_dict()
                examples = build_hierarchical_examples_from_oracle(
                    str(sample.get("id") or f"{path.name}:{line_number}"),
                    oracle,
                )
                grouped = defaultdict(set)
                for example in examples:
                    key = (
                        tuple(example.input_token_ids),
                        example.view_kind,
                        example.query_role_by_token,
                        example.shared_role_by_token,
                    )
                    grouped[key].add(
                        (
                            example.view_kind,
                            example.unit_pos,
                            example.aux_kind,
                            example.decision,
                            example.role_labels_by_token,
                        )
                    )

                for key, labels in grouped.items():
                    if len(labels) <= 1:
                        continue
                    token_ids = key[0]
                    view_kind = key[1]
                    failures.append(
                        (
                            path.name,
                            line_number,
                            list(token_ids),
                            view_kind,
                            sorted(labels),
                            sample.get("sentence", ""),
                        )
                    )

    assert not failures[:20]
