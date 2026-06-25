import importlib.util
from pathlib import Path


VALIDATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "lingua-parse-validator"
    / "scripts"
    / "validate_graph.py"
)
spec = importlib.util.spec_from_file_location("validate_graph_script", VALIDATOR_PATH)
validate_graph_script = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validate_graph_script)


def test_relative_where_ref_arg2_rejects_extra_variable_child():
    sample = {
        "sentence": "the area where you add plaster",
        "lingua": {
            "words": ["the", " area", " where", " you", " add", " plaster"],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "DeterminerConstant"},
                    {"spans": [[1, 1]], "type": "NominalConstant"},
                    {"spans": [[2, 2]], "type": "GeneralFunctor"},
                    {"spans": [[3, 3]], "type": "NominalConstant"},
                    {"spans": [[4, 4]], "type": "FactualPredicator"},
                    {"spans": [[5, 5]], "type": "NominalConstant"},
                    {"spans": "Ref", "type": "ReferentialPredicator"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "as:modification"},
                    {"start": 2, "end": 4, "label": "func.arg"},
                    {"start": 2, "end": 0, "label": "variable"},
                    {"start": 4, "end": 3, "label": "pred.arg.1"},
                    {"start": 4, "end": 5, "label": "pred.arg.2"},
                    {"start": 6, "end": 0, "label": "pred.arg.1"},
                    {"start": 6, "end": 2, "label": "pred.arg.2"},
                ],
            },
        },
    }

    errors = validate_graph_script.validate_graph(1, sample)

    assert any(error["code"] == "RELATIVE_ADVERB_EXTRA_CHILD_ERROR" for error in errors)


def test_relative_when_ref_arg2_warns_on_modification_body_wrapper():
    sample = {
        "sentence": "the time when they arrived",
        "lingua": {
            "words": ["the", " time", " when", " they", " arrived"],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "DeterminerConstant"},
                    {"spans": [[1, 1]], "type": "NominalConstant"},
                    {"spans": [[2, 2]], "type": "GeneralFunctor"},
                    {"spans": [[3, 3]], "type": "NominalConstant"},
                    {"spans": [[4, 4]], "type": "FactualPredicator"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                    {"spans": "Ref", "type": "ReferentialPredicator"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "as:modification"},
                    {"start": 2, "end": 4, "label": "func.arg"},
                    {"start": 4, "end": 3, "label": "pred.arg.1"},
                    {"start": 5, "end": 0, "label": "variable"},
                    {"start": 5, "end": 2, "label": "body"},
                    {"start": 6, "end": 0, "label": "pred.arg.1"},
                    {"start": 6, "end": 5, "label": "pred.arg.2"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(
        finding["code"] == "RELATIVE_ADVERB_REF_WRAPPER_WARNING"
        and finding["severity"] == "warning"
        for finding in findings
    )


def test_temporal_when_inside_argument_relative_is_not_wrapper_warning():
    sample = {
        "sentence": "the person who had a manual when he arrived",
        "lingua": {
            "words": ["the", " person", " who", " had", " a", " manual", " when", " he", " arrived"],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "DeterminerConstant"},
                    {"spans": [[1, 1]], "type": "NominalConstant"},
                    {"spans": [[2, 2]], "type": "GeneralFunctor"},
                    {"spans": [[3, 3]], "type": "FactualPredicator"},
                    {"spans": [[4, 4]], "type": "DeterminerConstant"},
                    {"spans": [[5, 5]], "type": "NominalConstant"},
                    {"spans": [[6, 6]], "type": "LogicalPredicator"},
                    {"spans": [[7, 7]], "type": "NominalConstant"},
                    {"spans": [[8, 8]], "type": "FactualPredicator"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                    {"spans": "Ref", "type": "ReferentialPredicator"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "as:modification"},
                    {"start": 3, "end": 2, "label": "pred.arg.1"},
                    {"start": 3, "end": 4, "label": "pred.arg.2"},
                    {"start": 4, "end": 5, "label": "as:modification"},
                    {"start": 6, "end": 3, "label": "pred.arg.1"},
                    {"start": 6, "end": 8, "label": "pred.arg.2"},
                    {"start": 8, "end": 7, "label": "pred.arg.1"},
                    {"start": 9, "end": 2, "label": "variable"},
                    {"start": 9, "end": 6, "label": "body"},
                    {"start": 10, "end": 0, "label": "pred.arg.1"},
                    {"start": 10, "end": 9, "label": "pred.arg.2"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert not any(
        finding["code"] == "RELATIVE_ADVERB_REF_WRAPPER_WARNING"
        for finding in findings
    )


def test_parataxis_rejects_child_ancestor_siblings():
    sample = {
        "sentence": "A happened, and B followed.",
        "lingua": {
            "words": ["A", " happened", ",", " and", " B", " followed", "."],
            "graph": {
                "nodes": [
                    {"spans": [[0, 1]], "type": "FactualPredicator"},
                    {"spans": [[3, 5]], "type": "FactualPredicator"},
                    {"spans": [[4, 4]], "type": "NominalConstant"},
                    {"spans": [[6, 6]], "type": "PunctuationalConstant"},
                    {"spans": "Parataxis", "type": "ConjunctionalFunctor"},
                ],
                "edges": [
                    {"start": 1, "end": 2, "label": "pred.arg.1"},
                    {"start": 3, "end": 4, "label": "as:punctuation"},
                    {"start": 4, "end": 0, "label": "func.arg"},
                    {"start": 4, "end": 1, "label": "func.arg"},
                    {"start": 1, "end": 0, "label": "pred.arg.2"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(
        finding["code"] == "PARATAXIS_CHILD_ANCESTOR_ERROR"
        for finding in findings
    )


def test_expression_rejects_child_ancestor_siblings():
    sample = {
        "sentence": "(or after 9pm, no corking fee!)",
        "lingua": {
            "words": ["(", "or", " after", " 9pm", ",", " no", " corking", " fee", "!", ")"],
            "graph": {
                "nodes": [
                    {"spans": [[1, 1]], "type": "ConjunctionalFunctor"},
                    {"spans": [[2, 3]], "type": "ModificationalFunctor"},
                    {"spans": [[8, 8]], "type": "PunctuationalConstant"},
                    {"spans": [[5, 7]], "type": "NominalConstant"},
                    {"spans": [[0, 0], [9, 9]], "type": "PunctuationalConstant"},
                    {"spans": "Expression", "type": "ExpressionFunctor"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "func.arg"},
                    {"start": 0, "end": 2, "label": "func.arg"},
                    {"start": 2, "end": 3, "label": "as:punctuation"},
                    {"start": 4, "end": 5, "label": "as:punctuation"},
                    {"start": 5, "end": 0, "label": "func.arg"},
                    {"start": 5, "end": 2, "label": "func.arg"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(
        finding["code"] == "AUX_FUNCTOR_CHILD_ANCESTOR_ERROR"
        for finding in findings
    )


def test_what_cannot_be_ref_antecedent():
    sample = {
        "sentence": "What I want is simple.",
        "lingua": {
            "words": ["What", " I", " want", " is", " simple", "."],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "GeneralFunctor"},
                    {"spans": [[1, 1]], "type": "NominalConstant"},
                    {"spans": [[2, 2]], "type": "FactualPredicator"},
                    {"spans": [[3, 3]], "type": "FactualPredicator"},
                    {"spans": [[4, 4]], "type": "ModificationalConstant"},
                    {"spans": [[5, 5]], "type": "PunctuationalConstant"},
                    {"spans": "Ref", "type": "ReferentialPredicator"},
                ],
                "edges": [
                    {"start": 2, "end": 1, "label": "pred.arg.1"},
                    {"start": 2, "end": 0, "label": "pred.arg.2"},
                    {"start": 3, "end": 6, "label": "pred.arg.1"},
                    {"start": 3, "end": 4, "label": "pred.arg.2"},
                    {"start": 5, "end": 3, "label": "as:punctuation"},
                    {"start": 6, "end": 0, "label": "pred.arg.1"},
                    {"start": 6, "end": 2, "label": "pred.arg.2"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(finding["code"] == "WHAT_REF_ANTECEDENT_ERROR" for finding in findings)


def test_ref_arg2_rejects_apposition_aux():
    sample = {
        "sentence": "the meeting, which has two parts -- first and second",
        "lingua": {
            "words": [
                "the",
                " meeting",
                ",",
                " which",
                " has",
                " two",
                " parts",
                " --",
                " first",
                " and",
                " second",
            ],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "DeterminerConstant"},
                    {"spans": [[1, 1]], "type": "NominalConstant"},
                    {"spans": [[2, 2]], "type": "PunctuationalConstant"},
                    {"spans": [[3, 3]], "type": "GeneralFunctor"},
                    {"spans": [[4, 4]], "type": "FactualPredicator"},
                    {"spans": [[5, 6]], "type": "NominalConstant"},
                    {"spans": [[7, 7]], "type": "PunctuationalConstant"},
                    {"spans": [[8, 10]], "type": "ConjunctionalFunctor"},
                    {"spans": "Apposition", "type": "AppositionalPredicator"},
                    {"spans": "Ref", "type": "ReferentialPredicator"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "as:modification"},
                    {"start": 2, "end": 0, "label": "as:punctuation"},
                    {"start": 4, "end": 3, "label": "pred.arg.1"},
                    {"start": 4, "end": 5, "label": "pred.arg.2"},
                    {"start": 6, "end": 5, "label": "as:punctuation"},
                    {"start": 7, "end": 6, "label": "func.arg"},
                    {"start": 7, "end": 5, "label": "func.arg"},
                    {"start": 8, "end": 6, "label": "pred.arg.1"},
                    {"start": 8, "end": 7, "label": "pred.arg.2"},
                    {"start": 9, "end": 2, "label": "pred.arg.1"},
                    {"start": 9, "end": 8, "label": "pred.arg.2"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(
        finding["code"] == "REF_ARG2_NON_CLAUSAL_AUX_ERROR"
        for finding in findings
    )


def test_however_discourse_marker_rejects_modification_aux():
    sample = {
        "sentence": "A, however, left.",
        "lingua": {
            "words": ["A", ",", " however", ",", " left", "."],
            "graph": {
                "nodes": [
                    {"spans": [[4, 4]], "type": "FactualPredicator"},
                    {"spans": [[0, 0]], "type": "NominalConstant"},
                    {"spans": [[1, 1]], "type": "PunctuationalConstant"},
                    {"spans": [[2, 2]], "type": "ModificationalConstant"},
                    {"spans": [[3, 3]], "type": "PunctuationalConstant"},
                    {"spans": [[5, 5]], "type": "PunctuationalConstant"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                ],
                "edges": [
                    {"start": 0, "end": 2, "label": "pred.arg.1"},
                    {"start": 2, "end": 1, "label": "as:punctuation"},
                    {"start": 4, "end": 3, "label": "as:punctuation"},
                    {"start": 5, "end": 6, "label": "as:punctuation"},
                    {"start": 6, "end": 0, "label": "variable"},
                    {"start": 6, "end": 4, "label": "body"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(
        finding["code"] == "DISCOURSE_MARKER_MODIFICATION_ERROR"
        for finding in findings
    )


def test_ref_arg2_modification_variable_warns_when_it_contains_antecedent():
    sample = {
        "sentence": "the gulf, spreading over land,",
        "lingua": {
            "words": ["the", " gulf", ",", " spreading", " over", " land", ","],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "DeterminerConstant"},
                    {"spans": [[1, 1]], "type": "NominalConstant"},
                    {"spans": [[2, 2]], "type": "PunctuationalConstant"},
                    {"spans": [[3, 3]], "type": "FactualPredicator"},
                    {"spans": [[4, 4]], "type": "ModificationalFunctor"},
                    {"spans": [[5, 5]], "type": "NominalConstant"},
                    {"spans": [[6, 6]], "type": "PunctuationalConstant"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                    {"spans": "Ref", "type": "ReferentialPredicator"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "as:modification"},
                    {"start": 2, "end": 0, "label": "as:punctuation"},
                    {"start": 3, "end": 2, "label": "pred.arg.1"},
                    {"start": 4, "end": 3, "label": "variable"},
                    {"start": 4, "end": 5, "label": "body"},
                    {"start": 6, "end": 8, "label": "as:punctuation"},
                    {"start": 7, "end": 2, "label": "variable"},
                    {"start": 7, "end": 4, "label": "body"},
                    {"start": 8, "end": 0, "label": "pred.arg.1"},
                    {"start": 8, "end": 7, "label": "pred.arg.2"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(
        finding["code"] == "REF_ARG2_VARIABLE_CONTAINS_ANTECEDENT_WARNING"
        and finding["severity"] == "warning"
        for finding in findings
    )


def test_consecutive_period_punctuation_chain_warns():
    sample = {
        "sentence": "A. .",
        "lingua": {
            "words": ["A", ".", " ."],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "NominalConstant"},
                    {"spans": [[1, 1]], "type": "PunctuationalConstant"},
                    {"spans": [[2, 2]], "type": "PunctuationalConstant"},
                ],
                "edges": [
                    {"start": 1, "end": 0, "label": "as:punctuation"},
                    {"start": 2, "end": 1, "label": "as:punctuation"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(
        finding["code"] == "CONSECUTIVE_PERIOD_PUNCTUATION_CHAIN_WARNING"
        and finding["severity"] == "warning"
        for finding in findings
    )


def test_modification_body_ref_rejects_duplicate_variable_antecedent():
    sample = {
        "sentence": "the environment that changed",
        "lingua": {
            "words": ["the", " environment", " that", " changed"],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "DeterminerConstant"},
                    {"spans": [[1, 1]], "type": "NominalConstant"},
                    {"spans": [[2, 2]], "type": "GeneralFunctor"},
                    {"spans": [[3, 3]], "type": "FactualPredicator"},
                    {"spans": "Ref", "type": "ReferentialPredicator"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "as:modification"},
                    {"start": 3, "end": 2, "label": "pred.arg.1"},
                    {"start": 4, "end": 1, "label": "pred.arg.1"},
                    {"start": 4, "end": 3, "label": "pred.arg.2"},
                    {"start": 5, "end": 1, "label": "variable"},
                    {"start": 5, "end": 4, "label": "body"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(
        finding["code"] == "MODIFICATION_BODY_REF_DUPLICATE_VARIABLE_ERROR"
        for finding in findings
    )


def test_modification_body_ref_rejects_descendant_antecedent():
    sample = {
        "sentence": "It came from Liverpool, that mattered.",
        "lingua": {
            "words": ["It", " came", " from", " Liverpool", ",", " that", " mattered", "."],
            "graph": {
                "nodes": [
                    {"spans": [[1, 1]], "type": "FactualPredicator"},
                    {"spans": [[0, 0]], "type": "NominalConstant"},
                    {"spans": [[2, 2]], "type": "ModificationalFunctor"},
                    {"spans": [[3, 3]], "type": "NominalConstant"},
                    {"spans": [[4, 4]], "type": "PunctuationalConstant"},
                    {"spans": [[5, 5]], "type": "GeneralFunctor"},
                    {"spans": [[6, 6]], "type": "FactualPredicator"},
                    {"spans": "Ref", "type": "ReferentialPredicator"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "pred.arg.1"},
                    {"start": 2, "end": 0, "label": "variable"},
                    {"start": 2, "end": 4, "label": "body"},
                    {"start": 4, "end": 3, "label": "as:punctuation"},
                    {"start": 6, "end": 5, "label": "pred.arg.1"},
                    {"start": 7, "end": 4, "label": "pred.arg.1"},
                    {"start": 7, "end": 6, "label": "pred.arg.2"},
                    {"start": 8, "end": 2, "label": "variable"},
                    {"start": 8, "end": 7, "label": "body"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(
        finding["code"] == "MODIFICATION_BODY_REF_DESCENDANT_ANTECEDENT_ERROR"
        for finding in findings
    )


def test_modification_body_discourse_warns():
    sample = {
        "sentence": "A, however.",
        "lingua": {
            "words": ["A", ",", " however", "."],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "NominalConstant"},
                    {"spans": [[2, 2]], "type": "ModificationalConstant"},
                    {"spans": "Discourse", "type": "ModificationalFunctor"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                ],
                "edges": [
                    {"start": 2, "end": 0, "label": "variable"},
                    {"start": 2, "end": 1, "label": "body"},
                    {"start": 3, "end": 0, "label": "variable"},
                    {"start": 3, "end": 2, "label": "body"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(
        finding["code"] == "MODIFICATION_BODY_DISCOURSE_WARNING"
        and finding["severity"] == "warning"
        for finding in findings
    )


def test_nested_modification_scope_warns_on_related_variables():
    sample = {
        "sentence": "There were options available (aside).",
        "lingua": {
            "words": ["There", " were", " options", " available", " (", " aside", ")"],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "NominalConstant"},
                    {"spans": [[1, 1]], "type": "FactualPredicator"},
                    {"spans": [[2, 2]], "type": "NominalConstant"},
                    {"spans": [[3, 3]], "type": "ModificationalConstant"},
                    {"spans": [[4, 4], [6, 6]], "type": "PunctuationalConstant"},
                    {"spans": [[5, 5]], "type": "OtherConstant"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                ],
                "edges": [
                    {"start": 1, "end": 0, "label": "pred.arg.1"},
                    {"start": 1, "end": 3, "label": "pred.arg.2"},
                    {"start": 3, "end": 2, "label": "as:modification"},
                    {"start": 4, "end": 5, "label": "as:punctuation"},
                    {"start": 6, "end": 3, "label": "variable"},
                    {"start": 6, "end": 4, "label": "body"},
                    {"start": 7, "end": 1, "label": "variable"},
                    {"start": 7, "end": 6, "label": "body"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(
        finding["code"] == "NESTED_MODIFICATION_SCOPE_WARNING"
        and finding["severity"] == "warning"
        for finding in findings
    )


def test_ref_arg2_punctuation_closed_modification_warns():
    sample = {
        "sentence": "my lock (which broke)",
        "lingua": {
            "words": ["my", " lock", " (", " which", " broke", ")"],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "DeterminerConstant"},
                    {"spans": [[1, 1]], "type": "NominalConstant"},
                    {"spans": [[2, 2], [5, 5]], "type": "PunctuationalConstant"},
                    {"spans": [[3, 3]], "type": "GeneralFunctor"},
                    {"spans": [[4, 4]], "type": "FactualPredicator"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                    {"spans": "Ref", "type": "ReferentialPredicator"},
                ],
                "edges": [
                    {"start": 0, "end": 1, "label": "as:modification"},
                    {"start": 4, "end": 3, "label": "pred.arg.1"},
                    {"start": 5, "end": 3, "label": "variable"},
                    {"start": 5, "end": 4, "label": "body"},
                    {"start": 2, "end": 5, "label": "as:punctuation"},
                    {"start": 6, "end": 1, "label": "pred.arg.1"},
                    {"start": 6, "end": 2, "label": "pred.arg.2"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(
        finding["code"] == "REF_ARG2_PUNCTUATION_CLOSED_MODIFICATION_WARNING"
        and finding["severity"] == "warning"
        for finding in findings
    )


def test_attribute_punctuation_modification_warns():
    sample = {
        "sentence": "A - (B)",
        "lingua": {
            "words": ["A", " -", " (", " B", ")"],
            "graph": {
                "nodes": [
                    {"spans": [[0, 0]], "type": "NominalConstant"},
                    {"spans": [[1, 1]], "type": "PunctuationalConstant"},
                    {"spans": [[2, 2], [4, 4]], "type": "PunctuationalConstant"},
                    {"spans": [[3, 3]], "type": "NominalConstant"},
                    {"spans": "Modification", "type": "ModificationalFunctor"},
                    {"spans": "Attribute", "type": "AttributePredicator"},
                ],
                "edges": [
                    {"start": 1, "end": 0, "label": "as:punctuation"},
                    {"start": 2, "end": 3, "label": "as:punctuation"},
                    {"start": 4, "end": 0, "label": "variable"},
                    {"start": 4, "end": 2, "label": "body"},
                    {"start": 5, "end": 1, "label": "pred.arg.1"},
                    {"start": 5, "end": 4, "label": "pred.arg.2"},
                ],
            },
        },
    }

    findings = validate_graph_script.validate_graph(1, sample)

    assert any(
        finding["code"] == "ATTRIBUTE_PUNCTUATION_MODIFICATION_WARNING"
        and finding["severity"] == "warning"
        for finding in findings
    )
