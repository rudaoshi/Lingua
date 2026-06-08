#!/usr/bin/env python3
"""Manual sentence-level repairs from the POS/type audit.

Each case below was inspected as a whole graph before being added.  The script
intentionally avoids a lexical batch rule: it only edits the exact split/line
and asserts the expected sentence/node text before touching the graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repair_be_copula_rules import (  # noqa: E402
    add_edge,
    compact_graph,
    graph,
    node_text,
    read_jsonl,
    remove_nodes,
    remove_edges,
    set_node_token_indexes,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

MOD_CONST = "ModificationalConstant"
FACTUAL = "FactualPredicator"


def node_type(sample: dict[str, Any], node: int) -> str:
    return graph(sample)["nodes"][node].get("type", "")


def set_node_type(sample: dict[str, Any], node: int, node_type_name: str) -> None:
    graph(sample)["nodes"][node]["type"] = node_type_name


def assert_text(sample: dict[str, Any], node: int, expected: str) -> None:
    actual = node_text(sample, node)
    if actual != expected:
        raise AssertionError(f"node {node}: expected {expected!r}, got {actual!r}")


def assert_sentence(sample: dict[str, Any], prefix: str) -> None:
    sentence = sample["sentence"]
    if not sentence.startswith(prefix):
        raise AssertionError(f"expected sentence prefix {prefix!r}, got {sentence!r}")


def add_edge_once(sample: dict[str, Any], start: int, end: int, label: str) -> bool:
    for edge in graph(sample)["edges"]:
        if edge["start"] == start and edge["end"] == end and edge["label"] == label:
            return False
    add_edge(sample, start, end, label)
    return True


def remove_exact(sample: dict[str, Any], start: int, end: int, label: str) -> bool:
    before = len(graph(sample)["edges"])
    remove_edges(sample, lambda edge: edge["start"] == start and edge["end"] == end and edge["label"] == label)
    return len(graph(sample)["edges"]) != before


def replace_exact(sample: dict[str, Any], start: int, old_end: int, label: str, new_end: int) -> bool:
    changed = False
    for edge in graph(sample)["edges"]:
        if edge["start"] == start and edge["end"] == old_end and edge["label"] == label:
            edge["end"] = new_end
            changed = True
    if not changed:
        add_edge_once(sample, start, new_end, label)
    return True


def dedupe_edges(sample: dict[str, Any]) -> None:
    seen: set[tuple[int, int, str]] = set()
    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        key = (edge["start"], edge["end"], edge["label"])
        if key in seen:
            continue
        seen.add(key)
        new_edges.append(edge)
    graph(sample)["edges"] = new_edges


def demote_state(sample: dict[str, Any], node: int, text: str) -> list[str]:
    assert_text(sample, node, text)
    changes: list[str] = []
    if node_type(sample, node) != MOD_CONST:
        set_node_type(sample, node, MOD_CONST)
        changes.append(f"{text}:type=ModificationalConstant")
    before = len(graph(sample)["edges"])
    remove_edges(sample, lambda edge: edge["start"] == node and edge["label"] == "pred.arg.1")
    if len(graph(sample)["edges"]) != before:
        changes.append(f"{text}:drop_pred.arg.1")
    return changes


def add_node(sample: dict[str, Any], spans: list[list[int]], node_type_name: str) -> int:
    graph(sample)["nodes"].append({"spans": spans, "type": node_type_name})
    return len(graph(sample)["nodes"]) - 1


def case_train_459(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "The M300 series are the lightest")
    return demote_state(sample, 14, "heavier")


def case_train_590(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Dr. Aster is very kind")
    return demote_state(sample, 3, "positive")


def case_train_893(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "It can ward off and cure")
    return demote_state(sample, 3, "youthful")


def case_train_1040(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "The place is chill and comfortable.")
    if node_text(sample, 5) == "is":
        return demote_state(sample, 1, "comfortable")
    assert_text(sample, 5, "is chill")
    assert_text(sample, 1, "comfortable")
    changes: list[str] = []
    set_node_token_indexes(sample, 5, [2])
    chill = add_node(sample, [[3, 3]], MOD_CONST)
    remove_exact(sample, 0, 5, "func.arg")
    add_edge_once(sample, 5, 0, "pred.arg.2")
    add_edge_once(sample, 0, chill, "func.arg")
    replace_exact(sample, 4, 0, "as:punctuation", 5)
    changes.extend(demote_state(sample, 1, "comfortable"))
    changes.append("is_chill:split_be_and_state_coordination")
    return changes


def case_train_1059(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "(It's indoors and outdoors!)")
    if node_text(sample, 5) == "' s":
        return demote_state(sample, 1, "outdoors")
    assert_text(sample, 5, "' s indoors")
    assert_text(sample, 1, "outdoors")
    changes: list[str] = []
    set_node_token_indexes(sample, 5, [2, 3])
    indoors = add_node(sample, [[4, 4]], MOD_CONST)
    remove_exact(sample, 0, 5, "func.arg")
    add_edge_once(sample, 5, 0, "pred.arg.2")
    add_edge_once(sample, 0, indoors, "func.arg")
    replace_exact(sample, 3, 0, "as:punctuation", 5)
    changes.extend(demote_state(sample, 1, "outdoors"))
    changes.append("is_indoors:split_be_and_state_coordination")
    return changes


def case_train_1297(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "So be very careful")
    changes = demote_state(sample, 2, "safe")
    changes.extend(demote_state(sample, 12, "toxic"))
    return changes


def case_train_1769(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "The staff do occasionally get game info correct")
    changes = demote_state(sample, 0, "correct")
    replace_exact(sample, 19, 0, "pred.arg.2", 10)
    add_edge_once(sample, 19, 0, "pred.arg.4")
    changes.append("get:arg2=game_info,arg4=correct")
    return changes


def case_train_2379(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "The doctor begs them")
    return demote_state(sample, 4, "pure")


def case_train_2491(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Our letter of credit person")
    return demote_state(sample, 5, "transferable")


def case_train_2749(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "They are very well made and realistic.")
    if node_text(sample, 4) == "are":
        return demote_state(sample, 1, "realistic")
    assert_text(sample, 4, "are made")
    assert_text(sample, 1, "realistic")
    changes: list[str] = []
    set_node_token_indexes(sample, 4, [1])
    made = add_node(sample, [[4, 4]], MOD_CONST)
    replace_exact(sample, 5, 4, "as:modification", made)
    add_edge_once(sample, 4, 0, "pred.arg.2")
    replace_exact(sample, 3, 0, "as:punctuation", 4)
    changes.extend(demote_state(sample, 1, "realistic"))
    changes.append("are_made:split_be_and_well_made_state_coordination")
    return changes


def case_train_2928(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "We thought you might like to know")
    return demote_state(sample, 4, "interesting")


def case_train_3171(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Because of an international medical convention")
    return demote_state(sample, 1, "ongoing")


def case_train_3740(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "To make the war unpalatable")
    changes = demote_state(sample, 0, "unpalatable")
    replace_exact(sample, 8, 1, "pred.arg.2", 3)
    add_edge_once(sample, 8, 1, "pred.arg.4")
    changes.append("make:arg2=war,arg4=unpalatable_to_public")
    return changes


def case_train_3963(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "I love this cage")
    changes = demote_state(sample, 11, "spacious")
    replace_exact(sample, 0, 10, "func.arg", 12)
    replace_exact(sample, 12, 16, "pred.arg.2", 10)
    remove_exact(sample, 10, 12, "func.arg")
    add_edge_once(sample, 10, 14, "func.arg")
    add_edge_once(sample, 10, 3, "func.arg")
    add_edge_once(sample, 10, 15, "func.arg")
    remove_edges(sample, lambda edge: edge["start"] == 16 or edge["end"] == 16)
    if len(graph(sample)["nodes"]) == 17:
        graph(sample)["nodes"].pop()
    changes.append("is:make_comma_and_the_state_coordination")
    return changes


def case_train_3978(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, '" Further, I find it even more disturbing')
    changes = demote_state(sample, 33, "disturbing")
    replace_exact(sample, 0, 42, "pred.arg.2", 35)
    add_edge_once(sample, 0, 42, "pred.arg.4")
    changes.append("find:arg2=it_that_ref,arg4=disturbing")
    return changes


def case_train_4076(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "HR, intended to be")
    return demote_state(sample, 3, "anonymous")


def case_train_4604(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Making the country safer")
    return demote_state(sample, 1, "safer")


def case_train_1139(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "He died aged 95")
    assert_text(sample, 2, "aged")
    replace_exact(sample, 2, 0, "pred.arg.1", 1)
    return ["aged:arg1=He_not_died"]


def case_train_4631(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "This is one of the worst places")
    return demote_state(sample, 5, "short")


def case_train_4856(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "The presence of the Sea Tigers")
    changes = demote_state(sample, 17, "dangerous")
    replace_exact(sample, 0, 19, "pred.arg.2", 10)
    add_edge_once(sample, 0, 19, "pred.arg.4")
    changes.append("makes:arg2=situation,arg4=extremely_dangerous")
    return changes


def case_train_4919(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "While we should be proud")
    changes = demote_state(sample, 5, "ready")
    replace_exact(sample, 3, 6, "pred.arg.2", 15)
    add_edge_once(sample, 3, 6, "pred.arg.4")
    changes.append("getting:arg2=testimony,arg4=ready_by_deadline")
    return changes


def case_train_4983(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "So, no it is not as good")
    if node_text(sample, 18) == "s":
        return demote_state(sample, 19, "peaceful")
    assert_text(sample, 18, "s quieter")
    changes = demote_state(sample, 19, "peaceful")
    set_node_token_indexes(sample, 18, [13])
    quieter = add_node(sample, [[14, 14]], MOD_CONST)
    replace_exact(sample, 17, 3, "func.arg", 18)
    remove_exact(sample, 3, 18, "func.arg")
    add_edge_once(sample, 18, 3, "pred.arg.2")
    add_edge_once(sample, 3, quieter, "func.arg")
    changes.append("s_quieter:split_be_and_state_coordination")
    return changes


def case_train_5071(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "I paid 2 k cash")
    return demote_state(sample, 10, "blown")


def case_train_5148(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Although the announcement was probably made")
    return demote_state(sample, 23, "hopeful")


def case_train_5450(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Service was a touch slow")
    return demote_state(sample, 0, "friendly")


def case_train_5782(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Skin will be darker")
    return demote_state(sample, 1, "lighter")


def case_train_5822(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "I have attempted to revise")
    return demote_state(sample, 0, "clearer")


def case_train_5845(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Workers who do these introductions")
    if node_text(sample, 24) == "are":
        return demote_state(sample, 8, "willing")
    assert_text(sample, 24, "are able")
    changes = demote_state(sample, 8, "willing")
    set_node_token_indexes(sample, 24, [19])
    able = add_node(sample, [[20, 20]], MOD_CONST)
    replace_exact(sample, 9, 7, "pred.arg.1", 24)
    remove_exact(sample, 7, 24, "func.arg")
    remove_exact(sample, 24, 25, "pred.arg.2")
    add_edge_once(sample, 24, 7, "pred.arg.2")
    add_edge_once(sample, 7, able, "func.arg")
    add_edge_once(sample, able, 25, "pred.arg.2")
    changes.append("are_able:split_be_and_willing_coordination")
    return changes


def case_train_6140(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "The cashier was also short")
    changes = demote_state(sample, 0, "unapologetic")
    if not any(node.get("spans") == "Parataxis" for node in graph(sample)["nodes"]):
        inner = add_node(sample, "Parataxis", "ConjunctionalFunctor")  # type: ignore[arg-type]
        remove_exact(sample, 11, 0, "func.arg")
        replace_exact(sample, 12, 15, "pred.arg.2", inner)
        add_edge_once(sample, inner, 15, "func.arg")
        add_edge_once(sample, inner, 0, "func.arg")
        changes.append("was:inner_state_coordination_short_unapologetic")
    return changes


def case_train_6521(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "If you need a cater")
    changes = demote_state(sample, 5, "easy")
    replace_exact(sample, 17, 6, "body", 15)
    remove_exact(sample, 6, 15, "func.arg")
    remove_exact(sample, 15, 18, "pred.arg.2")
    add_edge_once(sample, 15, 6, "pred.arg.2")
    add_edge_once(sample, 6, 18, "func.arg")
    changes.append("is_affordable:make_is_top_for_easy_coordination")
    return changes


def case_train_6557(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "I hope you have found")
    return demote_state(sample, 1, "helpful")


def case_train_6632(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Again, decades of this sort")
    changes = demote_state(sample, 1, "untouched")
    replace_exact(sample, 12, 13, "pred.arg.2", 7)
    add_edge_once(sample, 12, 1, "pred.arg.4")
    is_object_wrapper = (
        len(graph(sample)["nodes"]) > 13
        and graph(sample)["nodes"][13].get("spans") == "Modification"
        and any(edge["start"] == 13 and edge["end"] == 7 and edge["label"] == "variable" for edge in graph(sample)["edges"])
        and any(edge["start"] == 13 and edge["end"] == 1 and edge["label"] == "body" for edge in graph(sample)["edges"])
    )
    if is_object_wrapper:
        remove_nodes(sample, {13})
        changes.append("leave:remove_object_complement_wrapper")
    changes.append("leave:arg2=person,arg4=untouched")
    return changes


def case_train_6730(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Please leave as a word document")
    changes = demote_state(sample, 5, "necessary")
    replace_exact(sample, 6, 5, "pred.arg.2", 4)
    add_edge_once(sample, 6, 5, "pred.arg.4")
    changes.append("think:arg2=that,arg4=necessary")
    return changes


def case_train_6947(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "The landlord is not nice")
    return demote_state(sample, 1, "helpful")


def case_train_7473(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "She is caring, punctual")
    if node_text(sample, 7) == "is":
        changes = demote_state(sample, 0, "punctual")
        changes.extend(demote_state(sample, 8, "enthusiastic"))
        return changes
    assert_text(sample, 7, "is caring")
    changes = demote_state(sample, 0, "punctual")
    changes.extend(demote_state(sample, 8, "enthusiastic"))
    set_node_token_indexes(sample, 7, [1])
    caring = add_node(sample, [[2, 2]], MOD_CONST)
    replace_exact(sample, 5, 6, "as:punctuation", 7)
    remove_exact(sample, 6, 7, "func.arg")
    add_edge_once(sample, 7, 6, "pred.arg.2")
    add_edge_once(sample, 6, caring, "func.arg")
    changes.append("is_caring:split_be_and_state_coordination")
    return changes


def case_train_8332(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "She gave me amazing pictures")
    changes = demote_state(sample, 15, "memorable")
    replace_exact(sample, 2, 16, "pred.arg.2", 12)
    add_edge_once(sample, 2, 16, "pred.arg.4")
    changes.append("made:arg2=wedding_day,arg4=so_memorable")
    return changes


def case_train_8520(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "The urge to protect")
    changes = demote_state(sample, 18, "grateful")
    replace_exact(sample, 0, 1, "pred.arg.2", 6)
    add_edge_once(sample, 0, 1, "pred.arg.4")
    changes.append("makes:arg2=me,arg4=more_grateful_for_blessings")
    return changes


def case_train_8651(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "She was way easy")
    changes = demote_state(sample, 22, "easy")
    remove_exact(sample, 3, 26, "pred.arg.3")
    add_edge_once(sample, 3, 26, "pred.arg.4")
    changes.append("made:arg4=so_easy")
    return changes


def case_train_8818(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "You were extremely polite")
    changes = demote_state(sample, 1, "professional")
    remove_exact(sample, 0, 6, "func.arg")
    remove_exact(sample, 4, 5, "pred.arg.2")
    add_edge_once(sample, 0, 5, "func.arg")
    add_edge_once(sample, 4, 0, "pred.arg.2")
    replace_exact(sample, 3, 0, "as:punctuation", 6)
    changes.append("were:make_state_coordination_arg2")
    return changes


def case_train_8942(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Never had a problem")
    changes = demote_state(sample, 13, "helpful")
    replace_exact(sample, 1, 15, "pred.arg.2", 3)
    add_edge_once(sample, 1, 15, "pred.arg.4")
    changes.append("found:arg2=them,arg4=very_helpful")
    return changes


def case_train_9148(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Quality of work is sufficient")
    return demote_state(sample, 5, "outstanding")


def case_train_9555(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "I finally called Matt")
    changes = demote_state(sample, 2, "easy")
    changes.extend(demote_state(sample, 3, "reasonable"))
    return changes


def case_train_9623(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "I was then moved")
    return demote_state(sample, 4, "rough")


def case_train_9993(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Please pass my appreciation")
    return demote_state(sample, 6, "enjoyable")


def case_train_10256(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Alteration of the Mississippi River")
    changes = demote_state(sample, 25, "vulnerable")
    replace_exact(sample, 24, 0, "pred.arg.2", 3)
    add_edge_once(sample, 24, 0, "pred.arg.4")
    changes.append("left:arg2=area,arg4=abnormally_vulnerable_to_forces")
    return changes


def case_train_10433(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "We sat in the front dining area")
    changes = demote_state(sample, 4, "pleasant")
    replace_exact(sample, 0, 2, "func.arg", 14)
    remove_exact(sample, 2, 14, "func.arg")
    remove_exact(sample, 12, 13, "pred.arg.2")
    add_edge_once(sample, 12, 2, "pred.arg.2")
    add_edge_once(sample, 2, 13, "func.arg")
    changes.append("was:make_cozy_pleasant_coordination_arg2")
    return changes


def case_train_10627(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Get that right")
    changes = demote_state(sample, 1, "right")
    replace_exact(sample, 0, 1, "pred.arg.2", 4)
    add_edge_once(sample, 0, 1, "pred.arg.4")
    changes.append("Get:arg2=that,arg4=right")
    return changes


def case_train_10629(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "They're incredibly social")
    changes = demote_state(sample, 3, "trainable")
    replace_exact(sample, 1, 2, "as:punctuation", 5)
    replace_exact(sample, 5, 9, "pred.arg.2", 2)
    remove_exact(sample, 2, 5, "func.arg")
    add_edge_once(sample, 2, 6, "func.arg")
    add_edge_once(sample, 2, 8, "func.arg")
    if len(graph(sample)["nodes"]) > 9 and graph(sample)["nodes"][9].get("spans") == "Parataxis":
        remove_nodes(sample, {9})
        changes.append("re:remove_aux_state_parataxis")
    changes.append("re:use_token_state_coordination")
    return changes


def case_train_10957(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "I have lived in the UTC")
    return demote_state(sample, 5, "here")


def case_train_11031(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "But I found the location")
    changes = demote_state(sample, 4, "wonderful")
    changes.extend(demote_state(sample, 11, "kind"))
    replace_exact(sample, 2, 4, "pred.arg.2", 7)
    add_edge_once(sample, 2, 4, "pred.arg.4")
    replace_exact(sample, 3, 12, "pred.arg.2", 9)
    add_edge_once(sample, 3, 12, "pred.arg.4")
    changes.append("found:arg2_objects,arg4_states")
    return changes


def case_train_11043(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "No relative of anyone influential")
    return demote_state(sample, 1, "influential")


def case_dev_106(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "I had another conversation")
    return demote_state(sample, 17, "retired")


def case_dev_414(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "I can not say enough")
    assert_text(sample, 0, "enough")
    changes: list[str] = []
    if node_type(sample, 0) != "NominalConstant":
        set_node_type(sample, 0, "NominalConstant")
        changes.append("enough:type=NominalConstant")
    return changes


def case_test_394(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "This proprietary Gel Suspension")
    return demote_state(sample, 1, "possible")


def case_test_516(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Approximately 3 times higher levels")
    changes = demote_state(sample, 5, "higher")
    remove_exact(sample, 5, 8, "pred.arg.2")
    add_edge_once(sample, 6, 5, "as:modification")
    replace_exact(sample, 22, 5, "body", 8)
    changes.append("higher:attach_degree_times_modifier")
    return changes


def case_test_592(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "The entire original cast")
    changes = demote_state(sample, 2, "original")
    remove_exact(sample, 2, 4, "pred.arg.2")
    add_edge_once(sample, 3, 2, "as:modification")
    replace_exact(sample, 24, 2, "body", 4)
    changes.append("original:attach_entire_modifier")
    return changes


def case_test_856(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "Very affordable")
    changes = demote_state(sample, 5, "cheap")
    replace_exact(sample, 17, 5, "pred.arg.2", 9)
    add_edge_once(sample, 17, 5, "pred.arg.4")
    changes.append("call:arg2=it,arg4=cheap")
    return changes


def case_test_933(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "I found them extremely professional")
    return demote_state(sample, 7, "professional")


def case_test_1000(sample: dict[str, Any]) -> list[str]:
    assert_sentence(sample, "It's fine - just make sure")
    return demote_state(sample, 1, "warm")


REPAIRS: dict[tuple[str, int], Callable[[dict[str, Any]], list[str]]] = {
    ("train", 459): case_train_459,
    ("train", 590): case_train_590,
    ("train", 893): case_train_893,
    ("train", 1040): case_train_1040,
    ("train", 1059): case_train_1059,
    ("train", 1297): case_train_1297,
    ("train", 1769): case_train_1769,
    ("train", 2379): case_train_2379,
    ("train", 2491): case_train_2491,
    ("train", 2749): case_train_2749,
    ("train", 2928): case_train_2928,
    ("train", 3171): case_train_3171,
    ("train", 3740): case_train_3740,
    ("train", 3963): case_train_3963,
    ("train", 3978): case_train_3978,
    ("train", 4076): case_train_4076,
    ("train", 4604): case_train_4604,
    ("train", 1139): case_train_1139,
    ("train", 4631): case_train_4631,
    ("train", 4856): case_train_4856,
    ("train", 4919): case_train_4919,
    ("train", 4983): case_train_4983,
    ("train", 5071): case_train_5071,
    ("train", 5148): case_train_5148,
    ("train", 5450): case_train_5450,
    ("train", 5782): case_train_5782,
    ("train", 5822): case_train_5822,
    ("train", 5845): case_train_5845,
    ("train", 6140): case_train_6140,
    ("train", 6521): case_train_6521,
    ("train", 6557): case_train_6557,
    ("train", 6632): case_train_6632,
    ("train", 6730): case_train_6730,
    ("train", 6947): case_train_6947,
    ("train", 7473): case_train_7473,
    ("train", 8332): case_train_8332,
    ("train", 8520): case_train_8520,
    ("train", 8651): case_train_8651,
    ("train", 8818): case_train_8818,
    ("train", 8942): case_train_8942,
    ("train", 9148): case_train_9148,
    ("train", 9555): case_train_9555,
    ("train", 9623): case_train_9623,
    ("train", 9993): case_train_9993,
    ("train", 10256): case_train_10256,
    ("train", 10433): case_train_10433,
    ("train", 10627): case_train_10627,
    ("train", 10629): case_train_10629,
    ("train", 10957): case_train_10957,
    ("train", 11031): case_train_11031,
    ("train", 11043): case_train_11043,
    ("dev", 106): case_dev_106,
    ("dev", 414): case_dev_414,
    ("test", 394): case_test_394,
    ("test", 516): case_test_516,
    ("test", 592): case_test_592,
    ("test", 856): case_test_856,
    ("test", 933): case_test_933,
    ("test", 1000): case_test_1000,
}


REVIEWED_KEEP: dict[tuple[str, int], str] = {
    ("train", 52): "spaCy POS false positive: 'forward' is an imperative/control verb in 'Go ahead and forward'",
    ("train", 55): "spaCy POS false positive: 'Skip' is an imperative verb; typo 'te' is separate",
    ("train", 147): "reduced age clause: 'He died, aged 87' keeps aged as a recoverable predicate with arg1 He",
    ("train", 423): "spaCy POS false positive: 'initial' is the verb in 'to initial signature pages'",
    ("train", 1004): "different-subject copular ellipsis: 'advancement may be slow and satisfaction rare'; keep rare as elliptical predicate",
    ("train", 1062): "headline/broadcast fragment: 'Allan Chernoff live from ...' uses live as an elliptical predicate",
    ("train", 1363): "spaCy POS false positive: 'registered other Serbs' is a verbal predicate",
    ("train", 1415): "different-subject copular ellipsis: 'deliveries were correct and the service courteous'; keep courteous as elliptical predicate",
    ("train", 2150): "spaCy POS false positive: 'streamlined operations' is a verbal predicate",
    ("train", 3241): "spaCy POS false positive: 'like to be handled' is a verbal predicate coordinated with 'are social'",
    ("train", 3375): "fragment/ellipsis has no be token: 'Attitude of staff very bad'; keep bad as the recoverable predicate",
    ("train", 3614): "spaCy POS false positive: both 'induced' nodes are verbal predicates",
    ("train", 3726): "spaCy POS false positive: 'select' is an imperative verb",
    ("train", 5113): "elliptical comparison fragment: 'it better then not going' keeps better as the recoverable predicate",
    ("train", 5215): "typo for understood: 'undrstood that ...' is a verbal predicate",
    ("train", 5332): "relational adjective: 'worth her' keeps worth as a predicate because it selects arg2",
    ("train", 5932): "parenthetical reduced clause: 'many of them still unresolved' keeps unresolved as an elliptical predicate",
    ("train", 6366): "spaCy POS false positive: 'frequent RI' is a verbal predicate",
    ("train", 6494): "spaCy POS false positive: 'complete' and 'forward' are imperative predicates",
    ("train", 6728): "spaCy POS false positive: 'present operas' is the verb in an infinitival complement",
    ("train", 6761): "missing-copula fragment: 'you interested in ...' keeps interested as an elliptical predicate",
    ("train", 7074): "typo for read: 'I red somewhere' is a verbal predicate",
    ("train", 8213): "different-subject copular ellipsis: 'prices reasonable, staff friendly' keeps both as predicates",
    ("train", 8247): "spaCy POS false positive: 'forward a copy' is an imperative predicate",
    ("train", 8440): "question fragment: 'baby budgie alone at night?' keeps alone as the recoverable predicate",
    ("train", 8568): "spaCy POS false positive: 'markets close' is a verbal predicate",
    ("train", 9213): "spaCy POS false positive: 'leaders ski' is a verbal predicate",
    ("train", 9352): "spaCy POS false positive: 'select EnronOnLine' is an imperative predicate",
    ("train", 9376): "spaCy POS false positive: 'Forgot to cc' is a verbal predicate",
    ("train", 9577): "reduced participial adjunct: 'Concerned for her state of mind' keeps concerned as a predicate with arg1 I",
    ("train", 9728): "spaCy POS false positive: 'select PROGRAMS' is an imperative predicate",
    ("train", 9769): "spaCy POS false positive: 'affirm freedom' is a verbal predicate",
    ("train", 10080): "spaCy POS false positive: 'locations that I frequent' is a verbal predicate",
    ("train", 10303): "spaCy POS false positive: 'SUCKS' is a verbal predicate despite spaCy ADV",
    ("train", 10779): "spaCy POS false positive: 'suppress production' is a verbal predicate",
    ("dev", 536): "hyphenated relational modifier: headed keeps its current relational graph rather than a simple state demotion",
    ("dev", 876): "typo for step: 'you steep into their school' is a verbal predicate",
    ("test", 634): "typo for bear: 'bare to sit' is a verbal predicate/control complement",
    ("test", 676): "reduced relative: 'patrons willing to stand' keeps willing as a recoverable predicate with complement",
    ("test", 896): "spaCy POS false positive on copula token: 'is' remains the finite predicate",
    ("test", 924): "spaCy POS false positive: 'left school' is a verbal predicate",
}


def run(apply: bool, data_dir: Path, report_dir: Path) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    case_dir = report_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    changed_records: list[dict[str, Any]] = []
    reviewed_records: list[dict[str, Any]] = []

    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        new_rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(rows, 1):
            key = (split, line_number)
            if key in REVIEWED_KEEP:
                reviewed_records.append(
                    {
                        "split": split,
                        "line": line_number,
                        "sentence": row["sentence"],
                        "decision": "keep",
                        "reason": REVIEWED_KEEP[key],
                        "graph": compact_graph(row),
                    }
                )
                new_rows.append(row)
                continue

            repair = REPAIRS.get(key)
            if repair is None:
                new_rows.append(row)
                continue

            before = deepcopy(row)
            after = deepcopy(row)
            changes = repair(after)
            dedupe_edges(after)
            if before == after:
                new_rows.append(row)
                continue
            record = {
                "split": split,
                "line": line_number,
                "sentence": row["sentence"],
                "changes": changes,
                "before": compact_graph(before),
                "after": compact_graph(after),
            }
            changed_records.append(record)
            case_path = case_dir / f"{split}_{line_number:05d}.txt"
            case_path.write_text(
                f"{split}:{line_number}\n{row['sentence']}\n\n"
                f"CHANGES: {', '.join(changes)}\n\n"
                f"BEFORE\n{record['before']}\n\nAFTER\n{record['after']}\n",
                encoding="utf-8",
            )
            new_rows.append(after)

        if apply:
            write_jsonl(path, new_rows)

    (report_dir / "changes.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in changed_records)
        + ("\n" if changed_records else ""),
        encoding="utf-8",
    )
    (report_dir / "reviewed_keep.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in reviewed_records)
        + ("\n" if reviewed_records else ""),
        encoding="utf-8",
    )
    summary = [
        "# POS Review Manual Cases",
        "",
        f"- apply: {apply}",
        f"- changed samples: {len(changed_records)}",
        f"- reviewed keep samples: {len(reviewed_records)}",
        "",
        "## Changed",
    ]
    for record in changed_records:
        summary.append(f"- {record['split']}:{record['line']} {', '.join(record['changes'])}")
    summary.extend(["", "## Reviewed Keep"])
    for record in reviewed_records:
        summary.append(f"- {record['split']}:{record['line']} {record['reason']}")
    (report_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--report-dir", type=Path, default=REPORTS / "pos_review_manual_cases")
    args = parser.parse_args()
    return run(args.apply, args.data_dir, args.report_dir)


if __name__ == "__main__":
    raise SystemExit(main())
