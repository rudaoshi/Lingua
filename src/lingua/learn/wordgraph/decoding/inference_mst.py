#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MST-based decoding utilities for Word-Lingua.

This module mirrors `lingua.learn.wordgraph.decoding.inference` (forward/encoding helpers),
but adds a *tree* decoder based on Chu–Liu–Edmonds (maximum spanning arborescence).

Key points / conventions:
- Graph nodes correspond 1:1 to input `words` (NO synthetic/virtual ROOT node exists in GPGraph).
- `MSTDecoder` internally uses a virtual root only during decoding.
- Returned `heads` uses dependency convention: exactly one token has head == -1 (the root).

Intended usage:
- Use `forward_words(...)` from `decoding.inference` to run the model.
- Call `decode_logits_mst(...)` to get (heads, label_ids) + readable edges.
- Optionally call `mst_to_gpgraph(...)` to convert the tree into a `GPGraph`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterable, Sequence, Set, Union
from collections import Counter

import numpy as np
import torch

from lingua.structure.gpgraph import GPGraph, GPGPhraseNode, GPGEdge

from lingua.learn.wordgraph.decoding.inference import (
    WordLinguaModelOutputs,
    forward_words,
)
from lingua.learn.wordgraph.decoding.mst_decoder import MSTDecoder
from lingua.learn.wordgraph.model.biaffine import ID2NODE_TYPE


@dataclass(frozen=True)
class MSTDecodeResult:
    heads: List[int]  # len = num_words; exactly one == -1
    label_ids: List[int]  # len = num_words; label_ids[root] is a dummy value (0)
    labels: Optional[List[str]]  # len = num_words if id2label provided
    # Tree edges implied by (heads, label_ids) for convenience/readability.
    edges: List[Tuple[int, int, int]]  # (head, dep, label_id) with head >= 0
    # Optional relaxation output: extra incoming parent edges per node.
    extra_edges: List[Tuple[int, int, int]] = field(default_factory=list)
    # Convenience: edges + extra_edges (deduped, head>=0).
    all_edges: List[Tuple[int, int, int]] = field(default_factory=list)


def _invert_id2label(id2label: Dict[int, str]) -> Dict[str, int]:
    """
    Build label_str -> label_id map (first occurrence wins).
    """
    out: Dict[str, int] = {}
    for k, v in id2label.items():
        if v not in out:
            out[v] = int(k)
    return out


def _normalize_label_groups(
    *,
    label_groups: Sequence[Sequence[Union[int, str]]],
    label2id: Optional[Dict[str, int]] = None,
) -> List[Counter[int]]:
    """
    Normalize a user-provided list of label-groups into a list[Counter[label_id]].

    Each inner list is interpreted as:
      "these labels are allowed to co-exist as *multiple incoming parent edges* to the same node".

    NOTE: Duplicates are meaningful (multiplicity). For example:
      ["pred.arg.1", "pred.arg.1", "variable"]
    means up to 2 incoming parents with label pred.arg.1 and 1 with label variable (including the tree edge).
    """
    norm: List[Counter[int]] = []
    for g in label_groups:
        c: Counter[int] = Counter()
        for x in g:
            if isinstance(x, (int, np.integer)):
                c[int(x)] += 1
            elif isinstance(x, str):
                if label2id is None:
                    raise ValueError("label_groups contains strings but label2id is None")
                if x not in label2id:
                    raise ValueError(f"Unknown label '{x}' in label_groups")
                c[int(label2id[x])] += 1
            else:
                raise TypeError(f"Unsupported label type in label_groups: {type(x)}")
        if c:
            norm.append(c)
    return norm


def _would_create_cycle(children_by_head: List[Set[int]], head: int, dep: int) -> bool:
    """
    Return True iff adding edge head->dep would create a directed cycle in the current graph.

    Condition: there is already a path dep -> ... -> head.
    """
    if head == dep:
        return True
    # DFS from dep to see if we can reach head.
    stack = [dep]
    seen: Set[int] = set()
    while stack:
        cur = stack.pop()
        if cur == head:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in children_by_head[cur]:
            if nxt not in seen:
                stack.append(nxt)
    return False


def relax_mst_tree_to_graph(
    *,
    arc_scores: np.ndarray,  # [N, N]
    label_scores: np.ndarray,  # [N, N, L]
    heads: Sequence[int],
    label_ids: Sequence[int],
    allowed_parent_label_groups: Sequence[Sequence[Union[int, str]]],
    id2label: Optional[Dict[int, str]] = None,
    max_extra_parents_per_node: int = 2,
    min_extra_edge_score: Optional[float] = None,
    use_joint_score: bool = True,
    allow_root_incoming: bool = False,
) -> List[Tuple[int, int, int]]:
    """
    Relax the strict single-parent tree constraint by adding extra incoming edges.

    Rule:
    - For each dependent node d, the multiset of labels on its incoming parent edges must fit into
      at least one group in `allowed_parent_label_groups` (duplicates = allowed multiplicity).
    - We require the original tree-edge label to appear in the chosen group; if not, we add no
      extra parents for that node.
    - We also enforce acyclicity: any extra edge that would create a directed cycle is skipped.

    Scoring:
    - For each candidate edge h->d, we pick the best label: argmax label_scores[h,d,:]
    - The candidate score is arc_scores[h,d] + label_scores[h,d,label] if use_joint_score else arc_scores[h,d].
    """
    n = int(len(heads))
    if arc_scores.shape[0] != n or arc_scores.shape[1] != n:
        raise ValueError(f"arc_scores must be [N,N] with N={n}, got {arc_scores.shape}")
    if label_scores.shape[0] != n or label_scores.shape[1] != n:
        raise ValueError(f"label_scores must be [N,N,L] with N={n}, got {label_scores.shape}")

    label2id = _invert_id2label(id2label) if id2label else None
    groups = _normalize_label_groups(label_groups=allowed_parent_label_groups, label2id=label2id)
    if not groups:
        return []

    # Start with the tree edges (acyclic).
    children_by_head: List[Set[int]] = [set() for _ in range(n)]
    for d, h in enumerate(heads):
        if int(h) >= 0:
            children_by_head[int(h)].add(int(d))

    extra_edges: List[Tuple[int, int, int]] = []
    for d in range(n):
        # Usually we keep the chosen root as in-degree 0 in the relaxed graph too.
        if (not allow_root_incoming) and int(heads[d]) < 0:
            continue

        tree_label = int(label_ids[d])
        eligible_groups = [g for g in groups if int(g.get(tree_label, 0)) > 0]
        if not eligible_groups:
            continue

        # Precompute best label + score for each candidate head.
        cand: List[Tuple[float, int, int]] = []  # (score, head, label_id)
        for h in range(n):
            if h == d:
                continue
            if int(heads[d]) >= 0 and h == int(heads[d]):
                continue  # already in tree
            best_label = int(np.argmax(label_scores[h, d, :]))
            if use_joint_score:
                score = float(arc_scores[h, d]) + float(label_scores[h, d, best_label])
            else:
                score = float(arc_scores[h, d])
            if min_extra_edge_score is not None and score < float(min_extra_edge_score):
                continue
            cand.append((score, h, best_label))

        if not cand:
            continue

        cand.sort(key=lambda x: x[0], reverse=True)

        # Choose the best group for this node (max sum of selected extra edge scores),
        # while respecting multiplicity AND avoiding cycles.
        best_group: Optional[Counter[int]] = None
        best_pick: List[Tuple[int, int]] = []  # (head, label_id)
        best_sum = float("-inf")

        for g in eligible_groups:
            remaining = Counter(g)
            remaining[tree_label] -= 1
            if remaining[tree_label] <= 0:
                remaining.pop(tree_label, None)

            # Simulate picks for this group using a temp adjacency to enforce acyclicity.
            tmp_children = [set(s) for s in children_by_head]
            picked: List[Tuple[int, int]] = []
            ssum = 0.0

            for score, h, lab in cand:
                if len(picked) >= int(max_extra_parents_per_node):
                    break
                if int(remaining.get(int(lab), 0)) <= 0:
                    continue
                # Cycle check
                if _would_create_cycle(tmp_children, int(h), int(d)):
                    continue
                # Accept
                picked.append((int(h), int(lab)))
                ssum += float(score)
                remaining[int(lab)] -= 1
                if remaining[int(lab)] <= 0:
                    remaining.pop(int(lab), None)
                tmp_children[int(h)].add(int(d))

            if picked and ssum > best_sum:
                best_sum = ssum
                best_group = g
                best_pick = picked

        if not best_pick or best_group is None:
            continue

        # Apply chosen picks to the global graph (should already be cycle-safe).
        for h, lab in best_pick:
            if _would_create_cycle(children_by_head, int(h), int(d)):
                continue
            extra_edges.append((int(h), int(d), int(lab)))
            children_by_head[int(h)].add(int(d))

    return extra_edges


def _root_scores_from_is_root_logits(is_root_logits_1: torch.Tensor) -> np.ndarray:
    """
    Convert is_root logits [N, 2] into per-node root preference scores [N,].
    """
    r = is_root_logits_1[:, :2].float().cpu().numpy()
    # score = logit(root=1) - logit(root=0)
    return (r[:, 1] - r[:, 0]).astype(np.float32)


def _root_scores_from_arc_logits(arc_logits_1: torch.Tensor) -> np.ndarray:
    """
    Derive root preference from arc logits [N, N] when the model has no is_root head.

    Heuristic: choose a node that has *weak incoming support* as root.
    We approximate this by negative best incoming score (ignoring self loops):
        root_score[i] = -max_h arc_logits[h, i]
    """
    scores = arc_logits_1.float().cpu().numpy()
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"arc_logits_1 must be square [N,N], got shape={scores.shape}")
    n = scores.shape[0]
    s = scores.copy()
    np.fill_diagonal(s, float("-inf"))
    best_in = np.max(s, axis=0) if n > 0 else np.array([], dtype=np.float32)
    return (-best_in).astype(np.float32)


def decode_logits_mst(
    *,
    arc_logits_1: torch.Tensor,
    rel_logits_1: torch.Tensor,
    id2label: Dict[int, str],
    is_root_logits_1: Optional[torch.Tensor] = None,
    length: Optional[int] = None,
    relax_parent_label_groups: Optional[Sequence[Sequence[Union[int, str]]]] = None,
    relax_max_extra_parents_per_node: int = 2,
    relax_min_extra_edge_score: Optional[float] = None,
    relax_use_joint_score: bool = True,
    relax_allow_root_incoming: bool = False,
) -> MSTDecodeResult:
    """
    Decode a single example into a single-root dependency tree using MST.

    Args:
        arc_logits_1: [N, N] head->dep scores (logits are fine).
        rel_logits_1: [N, N, L] label scores (logits are fine).
        id2label: mapping from label id -> label string (used only for readable output).
        is_root_logits_1: optional [N, 2] is_root head logits.
        length: optional number of nodes to decode (<= N).
    """
    if length is None:
        length = int(arc_logits_1.shape[0])
    length = int(length)

    arc_scores = arc_logits_1[:length, :length].float().cpu().numpy()
    label_scores = rel_logits_1[:length, :length, :].float().cpu().numpy()

    if is_root_logits_1 is not None:
        root_scores = _root_scores_from_is_root_logits(is_root_logits_1[:length, :])
    else:
        root_scores = _root_scores_from_arc_logits(arc_logits_1[:length, :length])

    decoder = MSTDecoder(id2label=id2label)
    heads, label_ids = decoder.decode(
        arc_scores=arc_scores,
        label_scores=label_scores,
        root_scores=root_scores,
        length=length,
    )

    labels = [id2label.get(i, f"LABEL_{i}") for i in label_ids] if id2label else None
    edges: List[Tuple[int, int, int]] = [(h, d, int(label_ids[d])) for d, h in enumerate(heads) if h >= 0]

    extra_edges: List[Tuple[int, int, int]] = []
    if relax_parent_label_groups:
        extra_edges = relax_mst_tree_to_graph(
            arc_scores=arc_scores,
            label_scores=label_scores,
            heads=heads,
            label_ids=label_ids,
            allowed_parent_label_groups=relax_parent_label_groups,
            id2label=id2label,
            max_extra_parents_per_node=int(relax_max_extra_parents_per_node),
            min_extra_edge_score=relax_min_extra_edge_score,
            use_joint_score=bool(relax_use_joint_score),
            allow_root_incoming=bool(relax_allow_root_incoming),
        )

    # Dedup: tree edges win; then extra edges.
    seen = set()
    all_edges: List[Tuple[int, int, int]] = []
    for e in edges + extra_edges:
        if e in seen:
            continue
        seen.add(e)
        all_edges.append(e)

    return MSTDecodeResult(
        heads=heads,
        label_ids=label_ids,
        labels=labels,
        edges=edges,
        extra_edges=extra_edges,
        all_edges=all_edges,
    )


def mst_to_gpgraph(
    *,
    words: List[str],
    mst: MSTDecodeResult,
    node_type_logits_1: Optional[torch.Tensor] = None,  # [N, T]
    child_of_whether_logits_1: Optional[torch.Tensor] = None,  # [N, 2]
    id2node_type: Optional[Dict[int, str]] = None,
    id2label: Optional[Dict[int, str]] = None,
    use_relaxed_edges: bool = False,
) -> GPGraph:
    """
    Convert an MST dependency tree into a `GPGraph`.

    Notes:
    - GPGraph has NO virtual root node. The tree root is implied by in-degree 0.
    - Each token becomes a `GPGPhraseNode` with spans=[(i, i)] and an optional `pos`.
    """
    n = len(words)
    if len(mst.heads) != n:
        raise ValueError(f"len(words)={n} but len(heads)={len(mst.heads)}")

    if id2node_type is None:
        id2node_type = ID2NODE_TYPE
    if id2label is None:
        id2label = {}

    # Node POS
    if node_type_logits_1 is not None and node_type_logits_1.shape[0] >= n:
        node_type_pred = node_type_logits_1[:n].argmax(dim=-1).tolist()
        node_pos = [id2node_type.get(int(t), "NominalConstant") for t in node_type_pred]
    else:
        node_pos = ["NominalConstant"] * n

    # child_of_whether
    if child_of_whether_logits_1 is not None and child_of_whether_logits_1.shape[0] >= n:
        cow_pred = child_of_whether_logits_1[:n].argmax(dim=-1).tolist()
        cow_bool = [bool(int(x) == 1) for x in cow_pred]
    else:
        cow_bool = [False] * n

    graph = GPGraph()
    graph.words = words

    word_to_node: Dict[int, GPGPhraseNode] = {}
    for i, w in enumerate(words):
        node = GPGPhraseNode(id=str(i), spans=[(i, i)], pos=node_pos[i])
        if cow_bool[i]:
            node.child_of_whether = True
        graph.add_node(node)
        word_to_node[i] = node

    # Add edges (head -> dep)
    edges_to_add: Iterable[Tuple[int, int, int]]
    if use_relaxed_edges and mst.all_edges:
        edges_to_add = mst.all_edges
    else:
        edges_to_add = mst.edges

    for head_idx, dep_idx, label_id in edges_to_add:
        if int(head_idx) < 0:
            continue
        if int(head_idx) >= n or int(dep_idx) >= n:
            continue
        label_id = int(label_id)
        label_str = id2label.get(label_id, str(label_id))
        graph.add_edge(word_to_node[int(head_idx)], word_to_node[int(dep_idx)], GPGEdge(label=label_str))

    return graph


@torch.no_grad()
def predict_mst_gpgraph(
    *,
    model: Any,
    tokenizer: Any,
    words: List[str],
    device: str,
    max_length: int,
    id2label: Dict[int, str],
) -> Tuple[GPGraph, MSTDecodeResult, WordLinguaModelOutputs]:
    """
    End-to-end helper: encode words, run the model, MST-decode, and convert to GPGraph.
    """
    fw = forward_words(model=model, tokenizer=tokenizer, words=words, device=device, max_length=max_length)
    outs: WordLinguaModelOutputs = fw["outs"]

    arc_logits_1 = outs.arc_logits[0]
    rel_logits_1 = outs.rel_logits[0]
    is_root_logits_1 = outs.is_root_logits[0] if outs.is_root_logits is not None else None

    mst = decode_logits_mst(
        arc_logits_1=arc_logits_1,
        rel_logits_1=rel_logits_1,
        id2label=id2label,
        is_root_logits_1=is_root_logits_1,
        length=len(words),
    )

    graph = mst_to_gpgraph(
        words=words,
        mst=mst,
        node_type_logits_1=outs.node_type_logits[0] if outs.node_type_logits is not None else None,
        child_of_whether_logits_1=outs.child_of_whether_logits[0] if outs.child_of_whether_logits is not None else None,
        id2label=id2label,
    )

    return graph, mst, outs


if __name__ == "__main__":
    # Minimal smoke example (no model) to exercise MST decoding utilities.
    id2label_ex = {0: "nsubj", 1: "det", 2: "case", 3: "obj", 4: "root"}
    words_ex = ["The", "cat", "sat"]
    n = len(words_ex)

    arc = torch.zeros((n, n), dtype=torch.float32)
    arc[2, 1] = 8.0
    arc[1, 0] = 6.0
    # Provide is_root logits so we don't rely on arc-based root heuristics.
    is_root = torch.tensor([[-5.0, -5.0], [-5.0, -5.0], [-5.0, 10.0]], dtype=torch.float32)

    rel = torch.zeros((n, n, len(id2label_ex)), dtype=torch.float32)
    rel[2, 1, 0] = 10.0  # sat -> cat : nsubj
    rel[1, 0, 1] = 10.0  # cat -> The : det

    mst_ex = decode_logits_mst(
        arc_logits_1=arc,
        rel_logits_1=rel,
        id2label=id2label_ex,
        is_root_logits_1=is_root,
        length=n,
    )
    g_ex = mst_to_gpgraph(words=words_ex, mst=mst_ex, id2label=id2label_ex)
    print("heads:", mst_ex.heads)
    print("labels:", mst_ex.labels)
    print("num_edges:", len(list(g_ex.edges())))


