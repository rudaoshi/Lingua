#!/usr/bin/env python3
"""Audit `to V` infinitives as selected complements vs purpose adjuncts.

This script is read-only. It uses the current standard distinction:

- selected non-finite complements may be `pred.arg.4`;
- purpose `to V` adjuncts should be anchored by `to/ModificationalFunctor`.

The semantic labels are heuristic and intended for review routing, not for
automatic repair.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

COMPLEMENT_GOVERNOR_LEMMAS = {
    "agree",
    "allow",
    "appear",
    "appoint",
    "ask",
    "attempt",
    "be",
    "choose",
    "claim",
    "come",
    "continue",
    "decide",
    "deserve",
    "enable",
    "expect",
    "fail",
    "force",
    "forget",
    "get",
    "go",
    "have",
    "help",
    "hesitate",
    "hope",
    "intend",
    "invite",
    "learn",
    "like",
    "love",
    "manage",
    "mean",
    "need",
    "offer",
    "order",
    "permit",
    "plan",
    "prefer",
    "prepare",
    "promise",
    "refuse",
    "require",
    "seem",
    "start",
    "strive",
    "suppose",
    "tend",
    "threaten",
    "try",
    "use",
    "want",
    "wish",
}

OBJECT_CONTROL_LEMMAS = {
    "allow",
    "appoint",
    "ask",
    "cause",
    "convince",
    "enable",
    "encourage",
    "expect",
    "force",
    "get",
    "invite",
    "order",
    "permit",
    "persuade",
    "require",
    "tell",
    "urge",
    "want",
}

PURPOSE_LIKE_GOVERNOR_LEMMAS = {
    "add",
    "build",
    "bring",
    "call",
    "come",
    "create",
    "design",
    "develop",
    "do",
    "establish",
    "gather",
    "give",
    "go",
    "hold",
    "make",
    "move",
    "open",
    "provide",
    "send",
    "set",
    "take",
    "undertake",
    "use",
    "visit",
    "work",
}

MANUAL_FLAGGED_REVIEW: dict[tuple[str, str, str], tuple[str, str, str]] = {
    ("train", "25", "24"): (
        "current_wrong",
        "adjectival_selected_complement",
        "`hard-pressed to beat` is a complement of the state, not a purpose modifier of `be`.",
    ),
    ("train", "164", "8"): (
        "current_ok",
        "event_purpose_modifier",
        "`turned loose to get dinner` is purpose-like and the anchored `to` modifier is acceptable.",
    ),
    ("train", "621", "15"): (
        "current_wrong",
        "purpose_arg4",
        "`take time out to do` has a complete matrix event; `to do` should be anchored purpose/modification.",
    ),
    ("train", "661", "41"): (
        "current_ok",
        "event_purpose_modifier",
        "Sentence-initial `To save money` modifies the ordering event.",
    ),
    ("train", "1126", "15"): (
        "current_wrong",
        "purpose_arg4",
        "`used her knowledge to give` is instrument/purpose, not selected arg4.",
    ),
    ("train", "1316", "10"): (
        "current_wrong",
        "purpose_arg4",
        "`using the function to value` is instrument/purpose, not selected arg4.",
    ),
    ("train", "1622", "6"): (
        "current_wrong",
        "purpose_arg4",
        "`Contact ... to make an appointment` is purpose of contacting.",
    ),
    ("train", "1779", "8"): (
        "current_wrong",
        "purpose_arg4",
        "`Take time to discover` is purpose/activity of taking time, not a time-cost complement.",
    ),
    ("train", "2412", "26"): (
        "current_wrong",
        "purpose_arg4",
        "`sent ... to Chicago to review` uses `to review` as mission/purpose.",
    ),
    ("train", "2426", "15"): (
        "current_wrong",
        "purpose_arg4",
        "`using ... to denote` is instrument/purpose.",
    ),
    ("train", "260", "12"): (
        "current_ok",
        "object_control_complement",
        "`recommend ... to give` is a selected object-control complement.",
    ),
    ("train", "2648", "9"): (
        "current_ok",
        "event_purpose_modifier",
        "`started this page to help` is purpose of the page-starting event.",
    ),
    ("train", "2816", "13"): (
        "current_wrong",
        "purpose_arg4",
        "`using the cheapest surgeon to perform` is purpose/function of the chosen surgeon.",
    ),
    ("train", "3479", "14"): (
        "current_ok",
        "event_purpose_modifier",
        "`getting a second preorder to have` is purpose of getting the preorder.",
    ),
    ("train", "3645", "7"): (
        "current_ok",
        "time_cost_complement",
        "`took 10 minutes to restore` is a time-cost construction; arg4 is acceptable.",
    ),
    ("train", "4189", "24"): (
        "current_wrong",
        "adjectival_selected_complement",
        "`good to have ...` is a complement of the state `good`, not event purpose.",
    ),
    ("train", "5028", "12"): (
        "current_wrong",
        "nominal_selected_modifier_scope",
        "`time to do it` belongs with `time`; it should not modify bare `be` as purpose.",
    ),
    ("train", "509", "13"): (
        "current_wrong",
        "nominal_purpose_scope",
        "`campaign to promote` modifies the campaign, not `conceptualize` as arg4.",
    ),
    ("train", "5529", "13"): (
        "current_wrong",
        "adjectival_selected_complement",
        "`wise to play up` is an adjectival complement, not purpose.",
    ),
    ("train", "5700", "10"): (
        "current_ok",
        "time_cost_complement",
        "`takes two short minutes to cover` is a time-cost construction.",
    ),
    ("train", "6035", "14"): (
        "current_wrong",
        "purpose_arg4",
        "`use anyone else to handle` is service purpose/function, not selected arg4.",
    ),
    ("train", "611", "19"): (
        "policy_review",
        "object_control_vs_purpose",
        "`elect X to fix` sits on the object-control/purpose boundary; standard needs a policy choice.",
    ),
    ("train", "6460", "11"): (
        "current_wrong",
        "purpose_arg4",
        "`takes his time to make` describes the activity/purpose of taking time.",
    ),
    ("train", "7152", "16"): (
        "current_wrong",
        "nominal_selected_modifier_scope",
        "`time to drink tea` belongs with `time`, not with `have` as event purpose.",
    ),
    ("train", "7628", "20"): (
        "current_wrong",
        "adjectival_selected_complement",
        "`sufficient to promote` is a complement of `sufficient`.",
    ),
    ("train", "7847", "20"): (
        "current_ok",
        "event_purpose_modifier",
        "`there ... to meet her` is purpose of being there.",
    ),
    ("train", "7993", "19"): (
        "current_wrong",
        "purpose_arg4",
        "`sent ... to stage attacks` is mission/purpose of sending.",
    ),
    ("train", "8330", "15"): (
        "current_ok",
        "event_purpose_modifier",
        "`what they need to live` treats `to live` as purpose/need-scope with an object present.",
    ),
    ("train", "8578", "23"): (
        "current_ok",
        "time_cost_complement",
        "`would take a week to clear` is a time-cost construction.",
    ),
    ("train", "9574", "24"): (
        "current_ok",
        "event_purpose_modifier",
        "`front to force` is purpose/function of the diplomatic front.",
    ),
    ("train", "9590", "16"): (
        "current_wrong",
        "adjectival_selected_complement",
        "`sufficient to drive` is a complement of `sufficient`.",
    ),
    ("train", "9774", "24"): (
        "current_wrong",
        "purpose_arg4",
        "`use the talks to try` is instrument/purpose.",
    ),
    ("train", "9893", "23"): (
        "current_wrong",
        "purpose_arg4",
        "`words ... used to describe` is use-for-purpose.",
    ),
    ("train", "9915", "18"): (
        "current_wrong",
        "purpose_arg4",
        "`used gel shift assays to determine` is instrument/purpose.",
    ),
    ("train", "10171", "15"): (
        "current_wrong",
        "purpose_arg4",
        "`trick that I used to tame` is instrument/purpose.",
    ),
    ("train", "10176", "20"): (
        "current_ok",
        "condition_or_purpose_modifier",
        "Sentence-initial `To be a great coach` is condition/purpose-like modifier.",
    ),
    ("train", "10350", "12"): (
        "current_wrong",
        "purpose_arg4",
        "`use ambulances to reach` is instrument/purpose.",
    ),
    ("train", "10541", "13"): (
        "current_ok",
        "time_cost_complement",
        "`took days ... to fulfill` is a time-cost construction.",
    ),
    ("train", "10675", "12"): (
        "current_wrong",
        "purpose_arg4",
        "`sent my wife and daughter over to check out` is purpose of sending.",
    ),
    ("dev", "155", "14"): (
        "current_ok",
        "event_purpose_modifier",
        "`get ahold ... to give us` is purpose/result of getting ahold.",
    ),
    ("test", "254", "23"): (
        "current_ok",
        "discourse_modifier",
        "`To take the simplest example` is a discourse/example modifier.",
    ),
    ("test", "392", "23"): (
        "current_ok",
        "time_cost_complement",
        "`took forever to copy` is a time-cost construction.",
    ),
    ("train", "1056", "17"): (
        "current_ok",
        "ecm_complement",
        "`find X to be` is an ECM-style selected complement.",
    ),
    ("train", "1651", "7"): (
        "policy_review",
        "object_control_vs_purpose",
        "`hire X to unlock` can be read as selected task complement or purpose.",
    ),
    ("train", "1788", "21"): (
        "policy_review",
        "malformed_or_missing_object",
        "`pressuring to get` lacks a clear object/control target in the sample.",
    ),
    ("train", "2179", "10"): (
        "current_ok",
        "adjectival_selected_complement",
        "`seems prudent ... to understand` is a selected adjectival complement.",
    ),
    ("train", "2379", "13"): (
        "current_ok",
        "object_control_complement",
        "`begs them to stop` is selected object-control.",
    ),
    ("train", "2837", "17"): (
        "policy_review",
        "passive_attempt_scope",
        "`attempts have been made to introduce` is likely licensed by `attempts`; repair policy depends on nominal-complement scope.",
    ),
    ("train", "2851", "34"): (
        "current_wrong",
        "purpose_arg4",
        "`re-designated ... to be more descriptive` is purpose/rationale of designation.",
    ),
    ("train", "3128", "14"): (
        "current_wrong",
        "purpose_arg4",
        "`scheduled a call to wrap it up` is purpose of scheduling the call.",
    ),
    ("train", "3260", "28"): (
        "current_ok",
        "result_complement",
        "`turning the dining room to face` is a result/orientation complement.",
    ),
    ("train", "3545", "6"): (
        "current_ok",
        "selected_complement",
        "`waiting to see` is a selected complement.",
    ),
    ("train", "3605", "12"): (
        "current_ok",
        "causative_complement",
        "`led me to believe` is selected causative/object-control.",
    ),
    ("train", "3736", "20"): (
        "current_wrong",
        "degree_complement_scope",
        "`enough points to satisfy` should scope through `enough`/degree, not `earn` arg4.",
    ),
    ("train", "4903", "25"): (
        "current_wrong",
        "purpose_arg4",
        "`utilized ... to activate` is use-for-purpose.",
    ),
    ("train", "4924", "7"): (
        "current_ok",
        "selected_complement",
        "`remember to make sure` is a selected complement.",
    ),
    ("train", "4993", "7"): (
        "current_ok",
        "object_control_complement",
        "`pressure me to buy` is selected object-control.",
    ),
    ("train", "5245", "27"): (
        "current_wrong",
        "nominal_purpose_scope",
        "`coordination to get bodies around` belongs with `coordination`, not `lost` arg4.",
    ),
    ("train", "5763", "12"): (
        "current_ok",
        "object_control_complement",
        "`lobbied the association to encourage` is selected object-control.",
    ),
    ("train", "5796", "22"): (
        "current_ok",
        "ecm_complement",
        "`consider this doctor to be` is ECM-style selected complement.",
    ),
    ("train", "5996", "19"): (
        "current_ok",
        "result_complement",
        "`build this up to be` is a result complement.",
    ),
    ("train", "6579", "18"): (
        "current_ok",
        "object_control_complement",
        "`called on FERC to do` is selected object-control.",
    ),
    ("train", "6864", "23"): (
        "current_ok",
        "result_complement",
        "`expanded ... to include` is a result/complement construction.",
    ),
    ("train", "7128", "14"): (
        "current_ok",
        "object_control_complement",
        "`remind you to do` is selected object-control.",
    ),
    ("train", "7211", "9"): (
        "current_ok",
        "selected_complement",
        "`remember to bring` is a selected complement.",
    ),
    ("train", "7346", "17"): (
        "current_ok",
        "result_complement",
        "`expanded his focus to include` is a result/complement construction.",
    ),
    ("train", "7442", "25"): (
        "current_ok",
        "object_control_complement",
        "`called on government to develop` is selected object-control.",
    ),
    ("train", "7952", "11"): (
        "current_ok",
        "adjectival_selected_complement",
        "`harder to read` is an adjectival complement.",
    ),
    ("train", "8702", "8"): (
        "policy_review",
        "idiom_purpose_boundary",
        "`go out of one's way to V` is idiomatic and straddles selected complement vs purpose.",
    ),
    ("train", "8789", "7"): (
        "current_wrong",
        "purpose_arg4_or_missing_predicate",
        "`shall you tomorrow to discuss` appears to miss `call`; current arg4 is not reliable.",
    ),
    ("train", "9184", "22"): (
        "current_ok",
        "object_control_complement",
        "`chose ... to start` is selected object-control.",
    ),
    ("train", "9376", "3"): (
        "current_ok",
        "selected_complement",
        "`Forgot to cc` is a selected complement.",
    ),
    ("train", "9638", "22"): (
        "current_ok",
        "ecm_complement",
        "`find a travel agent to be` is ECM-style selected complement.",
    ),
    ("train", "10035", "6"): (
        "current_wrong",
        "purpose_arg4",
        "`see them to stop them` is seeing for the purpose of stopping.",
    ),
    ("train", "10120", "15"): (
        "current_ok",
        "object_control_complement",
        "`proposed Michael Martin to be` is selected object-control/role complement.",
    ),
    ("train", "10387", "12"): (
        "current_ok",
        "causative_complement",
        "`led many to question` is selected causative/object-control.",
    ),
    ("train", "10855", "13"): (
        "current_ok",
        "ecm_complement",
        "`believe national reconciliation to be` is ECM-style selected complement.",
    ),
    ("dev", "122", "18"): (
        "current_ok",
        "causative_complement",
        "`prompt the Federal Reserve to ease` is selected causative/object-control.",
    ),
    ("test", "274", "28"): (
        "current_wrong",
        "nominal_purpose_scope",
        "`counterparty list to trade` modifies the list/trading purpose, not `approve`.",
    ),
    ("test", "341", "18"): (
        "current_ok",
        "causative_complement",
        "`prompted us to postulate` is selected causative/object-control.",
    ),
    ("test", "466", "12"): (
        "current_wrong",
        "nominal_purpose_scope",
        "`male to fertilize the eggs` is nominal modifier/function, not `has` arg4.",
    ),
    ("test", "582", "7"): (
        "current_wrong",
        "purpose_arg4",
        "`Going back to have ... done` is purpose of going back.",
    ),
    ("test", "724", "15"): (
        "current_ok",
        "adjectival_selected_complement",
        "`eager to answer` is an adjectival complement.",
    ),
}

BE_FORMS = {"'m", "'re", "'s", "am", "are", "be", "been", "being", "is", "m", "re", "s", "was", "were"}
AUX_FORMS = {
    "'d",
    "'ll",
    "'ve",
    "ca",
    "can",
    "could",
    "d",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "ll",
    "may",
    "might",
    "must",
    "need",
    "needed",
    "shall",
    "should",
    "ve",
    "will",
    "wo",
    "would",
}
IGNORED_FOR_HEAD = BE_FORMS | AUX_FORMS | {"'", "n", "n't", "not", "nt", "t", "to"}
BASE_VERBS_ENDING_ING = {
    "bring",
    "cling",
    "fling",
    "ring",
    "sing",
    "spring",
    "sting",
    "string",
    "swing",
    "wing",
}

IRREGULAR = {
    "began": "begin",
    "begun": "begin",
    "built": "build",
    "came": "come",
    "chosen": "choose",
    "did": "do",
    "done": "do",
    "found": "find",
    "gave": "give",
    "given": "give",
    "gone": "go",
    "got": "get",
    "had": "have",
    "made": "make",
    "meant": "mean",
    "said": "say",
    "sent": "send",
    "set": "set",
    "shown": "show",
    "told": "tell",
    "took": "take",
    "tried": "try",
    "undertaken": "undertake",
    "used": "use",
    "went": "go",
    "wo": "will",
}


def norm(token: str) -> str:
    return token.strip().lower().replace("’", "'")


def lemma_like(token: str) -> str:
    token = norm(token)
    if token in IRREGULAR:
        return IRREGULAR[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        stem = token[:-3]
        cands = [stem]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            cands.append(stem[:-1])
        cands.append(stem + "e")
        for cand in cands:
            if cand in COMPLEMENT_GOVERNOR_LEMMAS | PURPOSE_LIKE_GOVERNOR_LEMMAS | OBJECT_CONTROL_LEMMAS:
                return cand
        return cands[0]
    if len(token) > 4 and token.endswith("ed"):
        stem = token[:-2]
        cands = [stem]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            cands.append(stem[:-1])
        cands.append(stem + "e")
        for cand in cands:
            if cand in COMPLEMENT_GOVERNOR_LEMMAS | PURPOSE_LIKE_GOVERNOR_LEMMAS | OBJECT_CONTROL_LEMMAS:
                return cand
        return cands[0]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def spans(node: dict[str, Any]) -> list[tuple[int, int]]:
    raw = node.get("spans")
    if not isinstance(raw, list):
        return []
    out: list[tuple[int, int]] = []
    for span in raw:
        if isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span):
            out.append((span[0], span[1]))
    return out


def node_tokens(sample: dict[str, Any], node_index: int) -> list[tuple[int, str, str]]:
    words = sample["lingua"]["words"]
    node = graph(sample)["nodes"][node_index]
    toks: list[tuple[int, str, str]] = []
    for start, end in spans(node):
        for index in range(start, end + 1):
            toks.append((index, words[index].strip(), norm(words[index])))
    return sorted(toks)


def node_text(sample: dict[str, Any], node_index: int) -> str:
    node = graph(sample)["nodes"][node_index]
    if isinstance(node.get("spans"), str):
        return node["spans"]
    return " ".join(surface for _index, surface, _norm in node_tokens(sample, node_index)).strip()


def node_type(sample: dict[str, Any], node_index: int) -> str:
    return str(graph(sample)["nodes"][node_index].get("type"))


def edge_in(sample: dict[str, Any], node_index: int, label: str | None = None) -> list[dict[str, Any]]:
    return [
        edge
        for edge in graph(sample)["edges"]
        if edge["end"] == node_index and (label is None or edge["label"] == label)
    ]


def edge_out(sample: dict[str, Any], node_index: int, label: str | None = None) -> list[dict[str, Any]]:
    return [
        edge
        for edge in graph(sample)["edges"]
        if edge["start"] == node_index and (label is None or edge["label"] == label)
    ]


def descendant_indexes(sample: dict[str, Any], node_index: int) -> list[int]:
    seen: set[int] = set()
    stack = [node_index]
    indexes: list[int] = []
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        indexes.extend(index for index, _surface, _norm in node_tokens(sample, current))
        stack.extend(edge["end"] for edge in edge_out(sample, current))
    return indexes


def lexical_head_lemma(sample: dict[str, Any], node_index: int) -> str:
    norms = [norm for _index, _surface, norm in node_tokens(sample, node_index)]
    content = [lemma_like(norm) for norm in norms if norm not in IGNORED_FOR_HEAD]
    if content:
        return content[-1]
    if any(norm in BE_FORMS for norm in norms):
        return "be"
    if norms:
        return lemma_like(norms[-1])
    return ""


def has_to_start(sample: dict[str, Any], node_index: int) -> bool:
    toks = node_tokens(sample, node_index)
    return bool(toks) and toks[0][2] == "to"


def contains_to(sample: dict[str, Any], node_index: int) -> bool:
    return any(norm == "to" for _index, _surface, norm in node_tokens(sample, node_index))


def contains_infinitival_to(sample: dict[str, Any], node_index: int) -> bool:
    toks = node_tokens(sample, node_index)
    for pos, (_index, _surface, item) in enumerate(toks):
        if item != "to":
            continue
        if pos + 1 < len(toks) and looks_bare_infinitive_token(toks[pos + 1][2]):
            return True
    return False


def looks_bare_infinitive_token(item: str) -> bool:
    return not (item.endswith("ing") and item not in BASE_VERBS_ENDING_ING)


def body_looks_bare_infinitive(sample: dict[str, Any], body_node: int) -> bool:
    toks = node_tokens(sample, body_node)
    if not toks:
        return False
    return looks_bare_infinitive_token(toks[0][2])


def outgoing_arg_texts(sample: dict[str, Any], node_index: int) -> dict[str, str]:
    return {edge["label"]: node_text(sample, edge["end"]) for edge in edge_out(sample, node_index)}


def subject_overlap(sample: dict[str, Any], parent: int, child: int) -> bool:
    parent_arg1 = edge_out(sample, parent, "pred.arg.1")
    child_arg1 = edge_out(sample, child, "pred.arg.1")
    if not parent_arg1 or not child_arg1:
        return False
    p_sets = [set(descendant_indexes(sample, edge["end"])) for edge in parent_arg1]
    c_sets = [set(descendant_indexes(sample, edge["end"])) for edge in child_arg1]
    return any(p and c and bool(p & c) for p in p_sets for c in c_sets)


def has_parent_object_before_child(sample: dict[str, Any], parent: int, child: int) -> bool:
    child_indexes = descendant_indexes(sample, child)
    if not child_indexes:
        return False
    child_min = min(child_indexes)
    for edge in edge_out(sample, parent, "pred.arg.2"):
        obj_indexes = descendant_indexes(sample, edge["end"])
        if obj_indexes and min(obj_indexes) < child_min:
            return True
    return False


def semantic_guess_for_arg4(sample: dict[str, Any], parent: int, child: int) -> tuple[str, str]:
    lemma = lexical_head_lemma(sample, parent)
    parent_text = node_text(sample, parent).lower()
    args = outgoing_arg_texts(sample, parent)
    has_object = "pred.arg.2" in args
    if "have to" in parent_text or parent_text.endswith("have") or lemma == "have":
        return "complement", "semi_modal_or_selected_have_to"
    if lemma in OBJECT_CONTROL_LEMMAS and has_object:
        return "purpose", "object_control_as_purpose_policy"
    if lemma in COMPLEMENT_GOVERNOR_LEMMAS and not (lemma in PURPOSE_LIKE_GOVERNOR_LEMMAS and has_object):
        return "complement", "selected_complement_governor"
    if lemma in PURPOSE_LIKE_GOVERNOR_LEMMAS and has_object:
        return "purpose", "purpose_like_governor_with_complete_object"
    if subject_overlap(sample, parent, child) and not has_object:
        return "complement", "subject_control_or_raising_like"
    if has_object and not has_parent_object_before_child(sample, parent, child):
        return "review", "object_scope_unclear"
    if has_object:
        return "purpose", "matrix_has_complete_object_before_to_clause"
    return "review", "semantic_selection_unclear"


def semantic_guess_for_to_functor(sample: dict[str, Any], to_node: int) -> tuple[str, str]:
    variable = edge_out(sample, to_node, "variable")
    body = edge_out(sample, to_node, "body")
    if len(variable) != 1 or len(body) != 1:
        return "not_to_verb", "to_functor_without_single_variable_body"
    variable_node = variable[0]["end"]
    body_node = body[0]["end"]
    if node_type(sample, body_node) != "FactualPredicator":
        return "not_to_verb", "body_not_factual_predicator"
    if not body_looks_bare_infinitive(sample, body_node):
        return "not_to_verb", "to_plus_gerund_or_nominal"
    if node_type(sample, variable_node) != "FactualPredicator":
        return "other_modifier", "to_verb_modifies_non_predicate_head"
    lemma = lexical_head_lemma(sample, variable_node)
    if lemma in OBJECT_CONTROL_LEMMAS and "pred.arg.2" in outgoing_arg_texts(sample, variable_node):
        return "purpose", "object_control_as_purpose_policy"
    if lemma in COMPLEMENT_GOVERNOR_LEMMAS and not lemma in PURPOSE_LIKE_GOVERNOR_LEMMAS:
        return "complement", "to_functor_attached_to_complement_selecting_governor"
    return "purpose", "anchored_to_modifier_with_event_body"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_audit_cases(data_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ("train", "dev", "test"):
        for line_no, sample in enumerate(read_jsonl(data_dir / f"{split}.jsonl"), 1):
            for node_index, node in enumerate(graph(sample)["nodes"]):
                typ = node.get("type")
                if typ == "FactualPredicator" and contains_infinitival_to(sample, node_index):
                    for incoming in edge_in(sample, node_index):
                        if incoming["label"] != "pred.arg.4":
                            continue
                        parent = incoming["start"]
                        guess, reason = semantic_guess_for_arg4(sample, parent, node_index)
                        current = "arg4_to_factual"
                        judgment = "likely_correct" if guess == "complement" else "likely_wrong" if guess == "purpose" else "needs_review"
                        rows.append(
                            make_row(
                                sample,
                                split,
                                line_no,
                                current,
                                guess,
                                judgment,
                                reason,
                                to_node=node_index,
                                parent_node=parent,
                                edge_label="pred.arg.4",
                            )
                        )

                if typ == "ModificationalFunctor" and contains_to(sample, node_index):
                    guess, reason = semantic_guess_for_to_functor(sample, node_index)
                    if guess == "not_to_verb":
                        continue
                    variable = edge_out(sample, node_index, "variable")
                    body = edge_out(sample, node_index, "body")
                    parent_node = variable[0]["end"] if variable else -1
                    body_node = body[0]["end"] if body else node_index
                    current = "to_modificational_functor"
                    judgment = (
                        "likely_correct"
                        if guess == "purpose"
                        else "out_of_scope"
                        if guess == "other_modifier"
                        else "likely_wrong"
                        if guess == "complement"
                        else "needs_review"
                    )
                    rows.append(
                        make_row(
                            sample,
                            split,
                            line_no,
                            current,
                            guess,
                            judgment,
                            reason,
                            to_node=node_index,
                            parent_node=parent_node,
                            body_node=body_node,
                            edge_label="variable/body",
                        )
                    )
    return rows


def make_row(
    sample: dict[str, Any],
    split: str,
    line_no: int,
    current_shape: str,
    semantic_guess: str,
    judgment: str,
    reason: str,
    to_node: int,
    parent_node: int,
    edge_label: str,
    body_node: int | None = None,
) -> dict[str, str]:
    if body_node is None:
        body_node = to_node
    parent_text = node_text(sample, parent_node) if parent_node >= 0 else ""
    return {
        "split": split,
        "line_no": str(line_no),
        "to_node": str(to_node),
        "parent_node": str(parent_node),
        "body_node": str(body_node),
        "current_shape": current_shape,
        "semantic_guess": semantic_guess,
        "judgment": judgment,
        "reason": reason,
        "edge_label": edge_label,
        "parent_lemma": lexical_head_lemma(sample, parent_node) if parent_node >= 0 else "",
        "parent_text": parent_text,
        "to_text": node_text(sample, to_node),
        "body_text": node_text(sample, body_node),
        "parent_args": json.dumps(outgoing_arg_texts(sample, parent_node), ensure_ascii=False, separators=(",", ":"))
        if parent_node >= 0
        else "",
        "sentence": sample.get("sentence", ""),
    }


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_manual_flagged_review(rows: list[dict[str, str]], report_dir: Path, fields: list[str]) -> None:
    flagged = [row for row in rows if row["judgment"] in {"likely_wrong", "needs_review"}]
    manual_rows: list[dict[str, str]] = []
    for row in flagged:
        key = (row["split"], row["line_no"], row["to_node"])
        decision, category, note = MANUAL_FLAGGED_REVIEW.get(key, ("unreviewed", "", ""))
        manual_rows.append(
            {
                **row,
                "manual_decision": decision,
                "manual_category": category,
                "manual_note": note,
            }
        )
    manual_fields = ["manual_decision", "manual_category", "manual_note", *fields]
    write_tsv(report_dir / "manual_flagged_review.tsv", manual_rows, manual_fields)

    summary_rows: list[dict[str, str]] = []
    for key, count in Counter(row["manual_decision"] for row in manual_rows).most_common():
        summary_rows.append({"section": "manual_decision", "key": key, "count": str(count)})
    for key, count in Counter(row["manual_category"] for row in manual_rows).most_common():
        summary_rows.append({"section": "manual_category", "key": key, "count": str(count)})
    for (decision, category), count in Counter(
        (row["manual_decision"], row["manual_category"]) for row in manual_rows
    ).most_common():
        summary_rows.append({"section": "manual_decision_category", "key": f"{decision}|{category}", "count": str(count)})
    write_tsv(report_dir / "manual_summary.tsv", summary_rows, ["section", "key", "count"])


def write_reports(rows: list[dict[str, str]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "line_no",
        "to_node",
        "parent_node",
        "body_node",
        "current_shape",
        "semantic_guess",
        "judgment",
        "reason",
        "edge_label",
        "parent_lemma",
        "parent_text",
        "to_text",
        "body_text",
        "parent_args",
        "sentence",
    ]
    write_tsv(report_dir / "cases.tsv", rows, fields)
    for judgment in ("likely_wrong", "needs_review", "likely_correct", "out_of_scope"):
        write_tsv(report_dir / f"{judgment}.tsv", [row for row in rows if row["judgment"] == judgment], fields)
    for guess in ("complement", "purpose", "review", "other_modifier"):
        write_tsv(report_dir / f"{guess}.tsv", [row for row in rows if row["semantic_guess"] == guess], fields)
    write_manual_flagged_review(rows, report_dir, fields)

    summary_rows: list[dict[str, str]] = []
    for key, count in Counter(row["current_shape"] for row in rows).most_common():
        summary_rows.append({"section": "current_shape", "key": key, "count": str(count)})
    for key, count in Counter(row["semantic_guess"] for row in rows).most_common():
        summary_rows.append({"section": "semantic_guess", "key": key, "count": str(count)})
    for key, count in Counter(row["judgment"] for row in rows).most_common():
        summary_rows.append({"section": "judgment", "key": key, "count": str(count)})
    for (shape, guess, judgment), count in Counter(
        (row["current_shape"], row["semantic_guess"], row["judgment"]) for row in rows
    ).most_common():
        summary_rows.append({"section": "shape_guess_judgment", "key": f"{shape}|{guess}|{judgment}", "count": str(count)})
    for (judgment, reason), count in Counter((row["judgment"], row["reason"]) for row in rows).most_common():
        summary_rows.append({"section": "judgment_reason", "key": f"{judgment}|{reason}", "count": str(count)})
    write_tsv(report_dir / "summary.tsv", summary_rows, ["section", "key", "count"])

    standard = [
        "# Standard Check",
        "",
        "- `pred.arg.4` is for selected complements such as `attempt to leave`.",
        "- Purpose `to V` adjuncts, especially paraphrasable as `in order to`, should not be `pred.arg.4`.",
        "- Anchored purpose adjuncts should use `to/ModificationalFunctor` with `variable` pointing to the modified event and `body` to the purpose event.",
        "- The semantic labels in this audit are heuristic routing labels; `needs_review.tsv` and `likely_wrong.tsv` need human review before repair.",
        "",
    ]
    (report_dir / "standard_check.md").write_text("\n".join(standard), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--report-dir", type=Path, default=REPORTS / "to_infinitive_complement_purpose_audit")
    args = parser.parse_args()
    rows = iter_audit_cases(args.data_dir)
    write_reports(rows, args.report_dir)
    counts = Counter(row["judgment"] for row in rows)
    print(json.dumps({"cases": len(rows), "judgment_counts": dict(counts), "report_dir": str(args.report_dir)}, indent=2))


if __name__ == "__main__":
    main()
