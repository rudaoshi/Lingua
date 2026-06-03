"""Factorized auxiliary targets for word-lingua edge labels."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence


NO_FACTOR = "__none__"

FACTOR_NAMES = (
    "component_1",
    "component_2",
    "semantic",
    "role",
    "hierarchy",
    "surface",
    "compound",
)

ROLE_COMPONENTS = {
    "pred.arg.1",
    "pred.arg.2",
    "pred.arg.3",
    "pred.arg.4",
    "variable",
    "body",
    "func.arg",
}

HIERARCHY_COMPONENTS = {
    "@lower",
    "@upper",
    "@modification_of",
    "@ref_of",
    "@appositive_of",
    "@attribute_of",
    "@list",
    "@expression",
    "@discourse_of",
    "@parataxis",
    "@vocative_of",
    "@copula",
}

SURFACE_COMPONENTS = {
    "as:modification",
    "as:punctuation",
    "as:discourse",
    "as:vocative",
    "as:ref",
    "punctuation",
    "next_word",
    "repeat",
}

SEMANTIC_PRIORITY = (
    ("ref", {"@ref_of", "as:ref"}),
    ("apposition", {"@appositive_of"}),
    ("attribute", {"@attribute_of"}),
    ("expression", {"@expression"}),
    ("list", {"@list"}),
    ("modification", {"@modification_of", "as:modification", "variable", "body"}),
    ("predication", {"pred.arg.1", "pred.arg.2", "pred.arg.3", "pred.arg.4"}),
    ("discourse", {"@discourse_of", "as:discourse"}),
    ("parataxis", {"@parataxis"}),
    ("vocative", {"@vocative_of", "as:vocative"}),
    ("copula", {"@copula"}),
    ("punctuation", {"as:punctuation", "punctuation"}),
    ("linear", {"next_word"}),
    ("repeat", {"repeat"}),
    ("lowering", {"@lower"}),
    ("uppering", {"@upper"}),
    ("functor_arg", {"func.arg"}),
)


@dataclass(frozen=True)
class EdgeFactorMaps:
    """Serializable factor vocabularies and per-label factor targets."""

    factor_vocabs: Dict[str, List[str]]
    label_factors: Dict[str, Dict[str, str]]

    def id_targets_by_label(self, edge_types: Sequence[str]) -> Dict[str, List[int]]:
        """Return one factor-id target list per factor, aligned to edge_types."""
        vocab_ids = {
            name: {value: idx for idx, value in enumerate(vocab)}
            for name, vocab in self.factor_vocabs.items()
        }
        targets: Dict[str, List[int]] = {}
        for name in FACTOR_NAMES:
            values: List[int] = []
            for label in edge_types:
                factor_value = self.label_factors[label][name]
                values.append(vocab_ids[name][factor_value])
            targets[name] = values
        return targets


def split_edge_label(label: str) -> List[str]:
    """Split a compressed word-lingua label into atomic path components."""
    return [part for part in label.split("+") if part]


def _first_component_in(components: Sequence[str], allowed: set[str]) -> str:
    for component in components:
        if component in allowed:
            return component
    return NO_FACTOR


def _semantic_factor(components: Sequence[str]) -> str:
    component_set = set(components)
    for value, triggers in SEMANTIC_PRIORITY:
        if component_set & triggers:
            return value
    return "other"


def decompose_edge_label(label: str) -> Dict[str, str]:
    """Decompose one legal word-lingua edge label into auxiliary factors.

    The factors are not independently decoded. They are auxiliary supervision for
    the same legal edge label inventory, so rare compressed labels share training
    signal with their component labels.
    """
    components = split_edge_label(label)
    if not components:
        raise ValueError("edge label must not be empty")

    return {
        "component_1": components[0],
        "component_2": components[1] if len(components) > 1 else NO_FACTOR,
        "semantic": _semantic_factor(components),
        "role": _first_component_in(components, ROLE_COMPONENTS),
        "hierarchy": _first_component_in(components, HIERARCHY_COMPONENTS),
        "surface": _first_component_in(components, SURFACE_COMPONENTS),
        "compound": "compound" if len(components) > 1 else "simple",
    }


def build_edge_factor_maps(edge_types: Sequence[str]) -> EdgeFactorMaps:
    """Build factor vocabularies from an ordered edge label vocabulary."""
    edge_types = list(edge_types)
    if not edge_types:
        raise ValueError("edge_types must not be empty")

    label_factors = {label: decompose_edge_label(label) for label in edge_types}

    vocabs: Dict[str, OrderedDict[str, None]] = {
        name: OrderedDict([(NO_FACTOR, None)]) for name in FACTOR_NAMES
    }
    for label in edge_types:
        factors = label_factors[label]
        for name in FACTOR_NAMES:
            vocabs[name][factors[name]] = None

    # Keep NO_FACTOR only where it can actually occur; this avoids pointless
    # output classes such as component_1=__none__.
    factor_vocabs = {}
    for name, values in vocabs.items():
        value_list = list(values.keys())
        used_values = {label_factors[label][name] for label in edge_types}
        if NO_FACTOR not in used_values:
            value_list = [value for value in value_list if value != NO_FACTOR]
        factor_vocabs[name] = value_list

    return EdgeFactorMaps(factor_vocabs=factor_vocabs, label_factors=label_factors)


def normalize_edge_factor_vocabs(
    factor_vocabs: Mapping[str, Sequence[str]] | None,
) -> Dict[str, List[str]]:
    """Validate and normalize serialized factor vocabularies."""
    if not factor_vocabs:
        return {}

    normalized: Dict[str, List[str]] = {}
    for name in FACTOR_NAMES:
        if name not in factor_vocabs:
            raise ValueError(f"missing edge factor vocabulary: {name}")
        values = list(factor_vocabs[name])
        if not values:
            raise ValueError(f"edge factor vocabulary is empty: {name}")
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ValueError(f"duplicate values in edge factor {name}: {duplicates}")
        normalized[name] = values
    return normalized
