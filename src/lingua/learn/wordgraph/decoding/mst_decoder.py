#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maximum Spanning Tree (MST) decoder for dependency parsing.

This module implements the Chu-Liu-Edmonds algorithm (also known as Edmonds' algorithm)
to decode biaffine neural network outputs (arc scores, label scores, and root scores)
into a valid dependency tree.

The algorithm:
1. Finds the maximum spanning tree in a directed graph
2. Ensures exactly one root
3. Ensures no cycles
4. Each word has exactly one head (except root)

References:
- Chu and Liu (1965): "On the Shortest Arborescence of a Directed Graph"
- Edmonds (1967): "Optimum Branchings"
- McDonald et al. (2005): "Non-projective Dependency Parsing using Spanning Tree Algorithms"
"""

from typing import Dict, List, Tuple, Optional, Set, TYPE_CHECKING
import numpy as np
from collections import defaultdict
import logging

if TYPE_CHECKING:
    from lingua.learn.wordgraph.decoding.statistics import WordLinguaStatistics

logger = logging.getLogger(__name__)


class MSTDecoder:
    """
    Maximum Spanning Tree decoder for dependency parsing.
    
    Decodes arc scores, label scores, and root scores from a biaffine neural network
    into a valid dependency tree using the Chu-Liu-Edmonds algorithm.
    """
    
    def __init__(
        self, 
        id2label: Optional[Dict[int, str]] = None,
        statistics: Optional["WordLinguaStatistics"] = None,
        constraint_weight: float = 1.0
    ):
        """
        Initialize the MST decoder.
        
        Args:
            id2label: Optional mapping from label IDs to label strings.
                     If not provided, labels will be represented as integers.
            statistics: Optional WordLinguaStatistics object for applying
                       statistical constraints during decoding.
            constraint_weight: Weight for statistical constraints (default: 1.0).
                             Higher values give more weight to constraints.
        """
        self.id2label = id2label or {}
        self.label2id = {v: k for k, v in self.id2label.items()} if id2label else {}
        self.statistics = statistics
        self.constraint_weight = constraint_weight
    
    def decode(
        self,
        arc_scores: np.ndarray,
        label_scores: Optional[np.ndarray] = None,
        root_scores: Optional[np.ndarray] = None,
        length: Optional[int] = None,
        node_pos: Optional[List[str]] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Decode scores into a dependency tree.
        
        Args:
            arc_scores: Arc scores of shape [seq_len, seq_len] where arc_scores[h, d]
                       is the score of head h -> dependent d.
            label_scores: Label scores of shape [seq_len, seq_len, num_labels].
            root_scores: Root scores of shape [seq_len] indicating probability of each
                        word being the root. If None, uses arc_scores[0, :] as root scores.
            length: Actual length of the sentence (excluding padding). If None, uses full length.
            node_pos: Optional list of POS tags for each node, used for constraint-based decoding.
        
        Returns:
            heads: List of head indices for each word. heads[i] is the head of word i.
                  The root word has head -1.
            labels: List of label indices for each word. labels[i] is the label of the
                   edge from heads[i] to i.
        """
        if length is None:
            length = arc_scores.shape[0]
        
        # Handle empty sentence
        if length == 0:
            return [], []
        
        # Ensure scores are numpy arrays
        arc_scores = np.array(arc_scores)
        if label_scores is not None:
            label_scores = np.array(label_scores)
        if root_scores is not None:
            root_scores = np.array(root_scores)
        
        # Extract valid portion
        arc_scores = arc_scores[:length, :length]
        
        # Apply constraints to compute joint arc scores if statistics and label_scores available
        if self.statistics is not None and label_scores is not None and node_pos is not None:
            label_scores_valid = label_scores[:length, :length, :]
            # Apply constraints to label scores
            constrained_label_scores = self._apply_label_constraints(
                label_scores_valid, node_pos[:length]
            )
            # Compute joint arc scores for MST
            joint_arc_scores = self._compute_joint_arc_scores(
                arc_scores, constrained_label_scores
            )
            # Decode tree structure using joint scores
            heads = self._decode_mst(joint_arc_scores, root_scores, length)
            # Decode labels using constrained scores
            labels = self._decode_labels(heads, constrained_label_scores, node_pos[:length])
        else:
            # Standard decoding without constraints
            heads = self._decode_mst(arc_scores, root_scores, length)
            # Decode labels
            if label_scores is not None:
                label_scores = label_scores[:length, :length, :]
                labels = self._decode_labels(heads, label_scores)
            else:
                labels = [0] * length
        
        return heads, labels
    
    def _apply_label_constraints(
        self,
        label_scores: np.ndarray,
        node_pos: List[str],
        parent_labels: Optional[List[Optional[int]]] = None
    ) -> np.ndarray:
        """
        Apply statistical constraints to label scores.
        
        Adds log-probabilities from multiple constraint types:
        - p(child_label | node_pos): based on head node's POS
        - p(child_label | parent_label, node_pos): if parent label info is available
        - p(parent_label | child_label): penalizes unlikely parent-child relationships
        
        Args:
            label_scores: Label scores of shape [length, length, num_labels].
            node_pos: POS tags for each node.
            parent_labels: Optional list of parent edge labels for each node.
                          If provided, enables p(child|parent,pos) constraint.
        
        Returns:
            Constrained label scores of shape [length, length, num_labels].
        """
        if self.statistics is None or not self.id2label:
            return label_scores
        
        length = label_scores.shape[0]
        num_labels = label_scores.shape[2]
        constrained_scores = label_scores.copy()
        
        for head in range(length):
            head_pos = node_pos[head] if head < len(node_pos) else "UNK"
            
            for dep in range(length):
                if head == dep:
                    continue
                    
                for label_id in range(num_labels):
                    label_str = self.id2label.get(label_id, str(label_id))
                    total_log_prob = 0.0
                    
                    # Constraint 1: p(child_label | node_pos)
                    prob_pos = self.statistics.get_prob_child_given_pos(head_pos, label_str)
                    total_log_prob += np.log(prob_pos + 1e-10)
                    
                    # Constraint 2: p(child_label | parent_label, node_pos)
                    # If head node has a known parent edge label, use it
                    if parent_labels is not None and head < len(parent_labels):
                        parent_label_id = parent_labels[head]
                        if parent_label_id is not None:
                            parent_label_str = self.id2label.get(parent_label_id, str(parent_label_id))
                            prob_parent = self.statistics.get_prob_child_given_parent(
                                parent_label_str, head_pos, label_str
                            )
                            total_log_prob += np.log(prob_parent + 1e-10)
                    
                    # Constraint 3: p(parent_label | child_label)
                    # Check if this child label typically appears under certain parent labels
                    # This helps ensure consistency in the tree structure
                    if parent_labels is not None and head < len(parent_labels):
                        parent_label_id = parent_labels[head]
                        if parent_label_id is not None:
                            parent_label_str = self.id2label.get(parent_label_id, str(parent_label_id))
                            prob_reverse = self.statistics.get_prob_parent_given_child(
                                label_str, parent_label_str
                            )
                            total_log_prob += np.log(prob_reverse + 1e-10)
                    
                    constrained_scores[head, dep, label_id] += self.constraint_weight * total_log_prob
        
        return constrained_scores
    
    def _compute_joint_arc_scores(
        self,
        arc_scores: np.ndarray,
        label_scores: np.ndarray
    ) -> np.ndarray:
        """
        Compute joint arc scores by combining arc scores with best label scores.
        
        Joint score = arc_score + max(label_scores over labels)
        
        Args:
            arc_scores: Arc scores of shape [length, length].
            label_scores: (Constrained) label scores of shape [length, length, num_labels].
        
        Returns:
            Joint arc scores of shape [length, length].
        """
        # For each arc, add the best label score
        best_label_scores = np.max(label_scores, axis=2)
        return arc_scores + best_label_scores
    
    def _decode_mst(
        self,
        arc_scores: np.ndarray,
        root_scores: Optional[np.ndarray],
        length: int
    ) -> List[int]:
        """
        Decode maximum spanning tree using Chu-Liu-Edmonds algorithm.
        
        Args:
            arc_scores: Arc scores of shape [length, length].
            root_scores: Optional root scores of shape [length].
            length: Length of the sentence.
        
        Returns:
            heads: List of head indices for each word.
        """
        if length == 0:
            return []
        
        if length == 1:
            return [-1]  # Single word is root

        # Decide root preference scores for each token (length,)
        #
        # In this codebase/tests we treat:
        # - `arc_scores[h, d]` as head->dependent scores among real tokens
        # - If `root_scores` is not provided, we fall back to `arc_scores[0, :]`
        #   as a reasonable default root-preference signal.
        if root_scores is not None:
            root_scores_arr = np.array(root_scores[:length], dtype=float)
        else:
            root_scores_arr = np.array(arc_scores[0, :length], dtype=float)
        
        # Create score matrix with root scores
        # We use index 0 as a virtual root node
        scores = np.full((length + 1, length + 1), float('-inf'))
        scores[0, 0] = 0  # Root to itself has score 0
        
        # Fill in arc scores (shift indices by 1 to account for virtual root at 0)
        scores[1:, 1:] = arc_scores
        
        # Fill in root scores
        scores[0, 1:] = root_scores_arr
        
        # Run Chu-Liu-Edmonds algorithm
        heads_with_root = self._chu_liu_edmonds(scores)
        
        # Remove virtual root (shift indices back)
        heads = [h - 1 for h in heads_with_root[1:]]

        # Enforce exactly one root (-1) in the final output (dependency parsing convention)
        heads = self._enforce_single_root(
            heads=heads,
            arc_scores=arc_scores,
            root_scores=root_scores_arr,
        )

        return heads

    def _enforce_single_root(
        self,
        *,
        heads: List[int],
        arc_scores: np.ndarray,
        root_scores: np.ndarray,
    ) -> List[int]:
        """
        Ensure exactly one token has head == -1 (the root).

        Chu–Liu–Edmonds with a virtual root produces a maximum arborescence which may
        have multiple direct children of the virtual root. For dependency parsing we
        enforce a single syntactic root by selecting the best root candidate and
        reattaching other root candidates to their best non-cycling heads.
        """
        length = len(heads)
        if length <= 1:
            return heads

        roots = [i for i, h in enumerate(heads) if h < 0]
        if len(roots) == 1:
            return heads

        # If no root exists (should be rare), pick the best root candidate.
        if len(roots) == 0:
            best_root = int(np.argmax(root_scores[:length]))
            old_head = heads[best_root]
            heads[best_root] = -1
            # If we just introduced a second root (in case old_head was -1), fix below.
            roots = [best_root]
        else:
            # Pick a single root among existing root candidates.
            best_root = max(roots, key=lambda i: float(root_scores[i]))

        # Reattach all other roots.
        for r in roots:
            if r == best_root:
                continue

            # Candidate heads sorted by descending arc score.
            cand_scores = arc_scores[:length, r].copy()
            cand_scores[r] = float("-inf")  # cannot attach to itself
            for cand in np.argsort(-cand_scores):
                cand = int(cand)
                if not np.isfinite(cand_scores[cand]):
                    continue
                trial = list(heads)
                trial[r] = cand
                # Keep best_root as the only root.
                trial[best_root] = -1
                # If we still have any other root, temporarily attach it to best_root.
                for i in range(length):
                    if i != best_root and trial[i] < 0:
                        trial[i] = best_root
                if self._find_cycle([-1] + [h + 1 for h in trial]) is None:
                    heads[r] = cand
                    break
            else:
                # Fallback: attach to the chosen root.
                heads[r] = best_root

        # Final safety: ensure exactly one root remains.
        for i in range(length):
            if i != best_root and heads[i] < 0:
                heads[i] = best_root
        heads[best_root] = -1
        return heads
    
    def _chu_liu_edmonds(self, scores: np.ndarray) -> List[int]:
        """
        Chu-Liu-Edmonds algorithm for finding maximum spanning tree.
        
        Args:
            scores: Score matrix of shape [n, n] where scores[i, j] is the
                   score of edge i -> j. Node 0 is the root.
        
        Returns:
            heads: List of head indices for each node.
        """
        n = scores.shape[0]
        
        # Find the best incoming edge for each node
        heads = self._greedy_best_heads(scores)
        
        # Check for cycles
        cycle = self._find_cycle(heads)
        
        if cycle is None:
            # No cycle found, we have a valid tree
            return heads
        
        # Contract the cycle and recurse
        return self._contract_cycle(scores, heads, cycle)
    
    def _greedy_best_heads(self, scores: np.ndarray) -> List[int]:
        """
        Greedily select the best incoming edge for each node.
        
        Args:
            scores: Score matrix of shape [n, n].
        
        Returns:
            heads: List of head indices for each node.
        """
        n = scores.shape[0]
        heads = []
        
        for i in range(n):
            if i == 0:
                # Root has no head
                heads.append(-1)
            else:
                # Find best head for node i
                best_head = np.argmax(scores[:, i])
                heads.append(int(best_head))
        
        return heads
    
    def _find_cycle(self, heads: List[int]) -> Optional[Set[int]]:
        """
        Find a cycle in the current heads assignment using DFS.
        
        Args:
            heads: Current head assignment.
        
        Returns:
            A set of node indices forming a cycle, or None if no cycle exists.
        """
        n = len(heads)
        visited = [False] * n
        rec_stack = [False] * n
        
        def dfs(node: int, path: List[int]) -> Optional[Set[int]]:
            visited[node] = True
            rec_stack[node] = True
            path.append(node)
            
            head = heads[node]
            if head >= 0:  # Has a head
                if rec_stack[head]:
                    # Found a cycle
                    cycle_start = path.index(head)
                    cycle = set(path[cycle_start:])
                    return cycle
                elif not visited[head]:
                    result = dfs(head, path)
                    if result is not None:
                        return result
            
            path.pop()
            rec_stack[node] = False
            return None
        
        # Start DFS from each unvisited node (except root)
        for start in range(1, n):
            if not visited[start]:
                result = dfs(start, [])
                if result is not None:
                    return result
        
        return None
    
    def _contract_cycle(
        self,
        scores: np.ndarray,
        heads: List[int],
        cycle: Set[int]
    ) -> List[int]:
        """
        Contract a cycle and recursively find MST.
        
        This implements the standard Chu-Liu-Edmonds cycle contraction:
        1. Contract cycle into a single super-node
        2. Adjust edge weights for edges entering/leaving the cycle
        3. Recursively find MST on contracted graph
        4. Expand back to original graph
        
        Args:
            scores: Original score matrix.
            heads: Current head assignment with a cycle.
            cycle: Set of nodes forming the cycle.
        
        Returns:
            heads: Final head assignment after resolving the cycle.
        """
        n = scores.shape[0]
        cycle_nodes = sorted(cycle)
        cycle_rep = min(cycle)  # Representative node for the cycle
        
        # Track which node in cycle each edge enters (for expansion later)
        best_entry_node = {}
        best_entry_score = {}
        
        # Create mapping between old and new node indices
        old_to_new = {}
        new_to_old = []
        
        # Map all cycle nodes to a single new node
        for old_idx in range(n):
            if old_idx in cycle:
                old_to_new[old_idx] = len(new_to_old) if cycle_rep not in old_to_new else old_to_new[cycle_rep]
                if cycle_rep not in old_to_new or old_idx == cycle_rep:
                    new_to_old.append(cycle_rep)
                    if old_idx == cycle_rep:
                        old_to_new[cycle_rep] = len(new_to_old) - 1
            else:
                old_to_new[old_idx] = len(new_to_old)
                new_to_old.append(old_idx)
        
        # Remove duplicate if added
        new_to_old = list(dict.fromkeys(new_to_old))
        old_to_new = {old: i for i, old in enumerate(new_to_old) if old not in cycle}
        cycle_new_idx = 0
        for i, old in enumerate(new_to_old):
            if old == cycle_rep:
                cycle_new_idx = i
                break
        for node in cycle:
            old_to_new[node] = cycle_new_idx
        
        n_new = len(new_to_old)
        new_scores = np.full((n_new, n_new), float('-inf'))
        
        # Compute scores for contracted graph
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                new_i = old_to_new[i]
                new_j = old_to_new[j]
                
                if new_i == new_j:
                    # Both in cycle, skip
                    continue
                
                if j in cycle and i not in cycle:
                    # Edge from outside into cycle
                    # Adjust score by subtracting the edge we're replacing in the cycle
                    for node in cycle:
                        in_cycle_score = scores[heads[node], node]
                        adjusted_score = scores[i, node] - in_cycle_score
                        
                        if adjusted_score > new_scores[new_i, new_j]:
                            new_scores[new_i, new_j] = adjusted_score
                            best_entry_node[(new_i, new_j)] = node
                            best_entry_score[(new_i, new_j)] = scores[i, node]
                
                elif i in cycle and j not in cycle:
                    # Edge from cycle to outside
                    # Take maximum score from any node in cycle
                    if scores[i, j] > new_scores[new_i, new_j]:
                        new_scores[new_i, new_j] = scores[i, j]
                
                else:
                    # Edge not involving cycle
                    if scores[i, j] > new_scores[new_i, new_j]:
                        new_scores[new_i, new_j] = scores[i, j]
        
        # Recursively solve contracted graph
        new_heads = self._chu_liu_edmonds(new_scores)
        
        # Expand solution back to original graph
        final_heads = [-1] * n
        
        # First, restore the cycle edges (all except one)
        for node in cycle:
            final_heads[node] = heads[node]
        
        # Find which edge enters the cycle in the solution
        entering_node = None
        for new_i, new_head in enumerate(new_heads):
            if new_i == cycle_new_idx and new_head >= 0:
                # The cycle node has a head in the new graph
                # Find which external node this maps to
                external_node = new_to_old[new_head]
                # Find which cycle node receives this edge
                key = (new_head, cycle_new_idx)
                if key in best_entry_node:
                    entering_node = best_entry_node[key]
                    final_heads[entering_node] = external_node
                break
        
        # For nodes outside cycle, map their heads back
        for new_i, new_head in enumerate(new_heads):
            old_i = new_to_old[new_i]
            
            if old_i not in cycle:
                if new_head < 0:
                    final_heads[old_i] = -1
                elif new_head == cycle_new_idx:
                    # Head is the cycle, find best node in cycle to attach to
                    best_node = -1
                    best_score = float('-inf')
                    for node in cycle:
                        if scores[node, old_i] > best_score:
                            best_score = scores[node, old_i]
                            best_node = node
                    final_heads[old_i] = best_node
                else:
                    final_heads[old_i] = new_to_old[new_head]
        
        return final_heads
    
    def _decode_labels(
        self,
        heads: List[int],
        label_scores: np.ndarray,
        node_pos: Optional[List[str]] = None
    ) -> List[int]:
        """
        Decode edge labels given the tree structure.
        
        When statistics are available and node_pos is provided, applies sibling
        constraints: p(child_label | other_sibling_label).
        
        Args:
            heads: Head assignment.
            label_scores: Label scores of shape [length, length, num_labels].
            node_pos: Optional POS tags for each node.
        
        Returns:
            labels: List of label indices for each edge.
        """
        length = len(heads)
        labels = [0] * length  # Initialize with default label
        
        # Group children by head for sibling constraint
        head_to_children: Dict[int, List[int]] = defaultdict(list)
        for i in range(length):
            head = heads[i]
            if head >= 0:
                head_to_children[head].append(i)
        
        # First pass: assign labels without sibling constraints
        for i in range(length):
            head = heads[i]
            if head < 0:
                labels[i] = 0  # Root node
            else:
                best_label = np.argmax(label_scores[head, i, :])
                labels[i] = int(best_label)
        
        # Second pass: apply sibling constraints if statistics available
        if self.statistics is not None and self.id2label:
            for head, children in head_to_children.items():
                if len(children) > 1:
                    # Apply sibling constraints iteratively
                    for iteration in range(2):  # 2 iterations for refinement
                        for child in children:
                            other_labels = [labels[c] for c in children if c != child]
                            # Re-score with sibling constraints
                            scores = label_scores[head, child, :].copy()
                            for label_id in range(scores.shape[0]):
                                label_str = self.id2label.get(label_id, str(label_id))
                                sibling_score = 0.0
                                for other_id in other_labels:
                                    other_str = self.id2label.get(other_id, str(other_id))
                                    prob = self.statistics.get_prob_child_given_sibling(
                                        other_str, label_str
                                    )
                                    sibling_score += np.log(prob + 1e-10)
                                scores[label_id] += self.constraint_weight * sibling_score
                            labels[child] = int(np.argmax(scores))
        
        return labels
    
    def decode_to_dict(
        self,
        arc_scores: np.ndarray,
        label_scores: Optional[np.ndarray] = None,
        root_scores: Optional[np.ndarray] = None,
        length: Optional[int] = None,
        words: Optional[List[str]] = None,
        node_pos: Optional[List[str]] = None
    ) -> Dict[str, List]:
        """
        Decode and return results as a dictionary with readable labels.
        
        Args:
            arc_scores: Arc scores.
            label_scores: Label scores.
            root_scores: Root scores.
            length: Sentence length.
            words: Optional list of words for reference.
            node_pos: Optional POS tags for constraint-based decoding.
        
        Returns:
            Dictionary containing:
                - heads: List of head indices
                - labels: List of label strings (if id2label is provided)
                - label_ids: List of label indices
                - words: List of words (if provided)
                - edges: List of (head_idx, dependent_idx, label) tuples
        """
        heads, label_ids = self.decode(arc_scores, label_scores, root_scores, length, node_pos)
        
        result = {
            "heads": heads,
            "label_ids": label_ids,
        }
        
        if self.id2label:
            result["labels"] = [self.id2label.get(lid, f"LABEL_{lid}") for lid in label_ids]
        
        if words is not None:
            result["words"] = words
        
        # Create edge list
        edges = []
        for i, (head, label_id) in enumerate(zip(heads, label_ids)):
            if head >= 0:
                label_str = self.id2label.get(label_id, f"LABEL_{label_id}") if self.id2label else str(label_id)
                edges.append((head, i, label_str))
        result["edges"] = edges
        
        return result


def decode_batch(
    arc_scores: np.ndarray,
    label_scores: Optional[np.ndarray] = None,
    root_scores: Optional[np.ndarray] = None,
    lengths: Optional[List[int]] = None,
    id2label: Optional[Dict[int, str]] = None,
    statistics: Optional["WordLinguaStatistics"] = None,
    constraint_weight: float = 1.0,
    node_pos_list: Optional[List[List[str]]] = None
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Decode a batch of sentences.
    
    Args:
        arc_scores: Arc scores of shape [batch_size, seq_len, seq_len].
        label_scores: Label scores of shape [batch_size, seq_len, seq_len, num_labels].
        root_scores: Root scores of shape [batch_size, seq_len].
        lengths: List of actual lengths for each sentence.
        id2label: Optional label mapping.
        statistics: Optional WordLinguaStatistics for constraint-based decoding.
        constraint_weight: Weight for statistical constraints.
        node_pos_list: Optional list of POS tag lists for each sentence.
    
    Returns:
        batch_heads: List of head lists for each sentence.
        batch_labels: List of label lists for each sentence.
    """
    decoder = MSTDecoder(
        id2label=id2label,
        statistics=statistics,
        constraint_weight=constraint_weight
    )
    batch_size = arc_scores.shape[0]
    
    if lengths is None:
        lengths = [arc_scores.shape[1]] * batch_size
    
    batch_heads = []
    batch_labels = []
    
    for i in range(batch_size):
        arc_s = arc_scores[i]
        label_s = label_scores[i] if label_scores is not None else None
        root_s = root_scores[i] if root_scores is not None else None
        node_pos = node_pos_list[i] if node_pos_list is not None else None
        
        heads, labels = decoder.decode(arc_s, label_s, root_s, lengths[i], node_pos)
        batch_heads.append(heads)
        batch_labels.append(labels)
    
    return batch_heads, batch_labels


if __name__ == "__main__":
    # Example usage and testing
    print("=== MST Decoder Example ===\n")
    
    # Create a simple example
    # Sentence: "The cat sat on the mat"
    # Words indexed as: 0=The, 1=cat, 2=sat, 3=on, 4=the, 5=mat
    words = ["The", "cat", "sat", "on", "the", "mat"]
    length = len(words)
    
    # Create synthetic arc scores (higher score = better head)
    # Expected tree: ROOT -> sat(2), sat -> cat(1), sat -> on(3), cat -> The(0), on -> mat(5), mat -> the(4)
    np.random.seed(42)
    arc_scores = np.random.randn(length, length) * 0.1
    
    # Set up expected structure with high scores
    arc_scores[0, 2] = 10.0  # ROOT -> sat
    arc_scores[2, 1] = 8.0   # sat -> cat
    arc_scores[2, 3] = 7.0   # sat -> on
    arc_scores[1, 0] = 6.0   # cat -> The
    arc_scores[3, 5] = 5.0   # on -> mat
    arc_scores[5, 4] = 4.0   # mat -> the
    
    # Create label scores (3 labels: nsubj, det, case)
    label_scores = np.random.randn(length, length, 3) * 0.1
    label_scores[2, 1, 0] = 5.0  # sat -> cat : nsubj
    label_scores[1, 0, 1] = 5.0  # cat -> The : det
    label_scores[2, 3, 2] = 5.0  # sat -> on : case
    
    # Create root scores
    root_scores = np.random.randn(length) * 0.1
    root_scores[2] = 10.0  # "sat" should be root
    
    # Create decoder
    id2label = {0: "nsubj", 1: "det", 2: "case"}
    decoder = MSTDecoder(id2label=id2label)
    
    # Decode
    result = decoder.decode_to_dict(
        arc_scores=arc_scores,
        label_scores=label_scores,
        root_scores=root_scores,
        length=length,
        words=words
    )
    
    print("Input sentence:", " ".join(words))
    print("\nDecoded dependency tree:")
    print(f"{'Index':<8} {'Word':<10} {'Head':<8} {'Label':<10}")
    print("-" * 40)
    for i, (word, head, label) in enumerate(zip(result["words"], result["heads"], result["labels"])):
        head_word = "ROOT" if head < 0 else result["words"][head]
        print(f"{i:<8} {word:<10} {head:<8} {label:<10}")
    
    print("\nEdges (head -> dependent : label):")
    for head, dep, label in result["edges"]:
        head_word = "ROOT" if head < 0 else result["words"][head]
        dep_word = result["words"][dep]
        print(f"  {head_word}({head}) -> {dep_word}({dep}) : {label}")
    
    print("\n=== Batch Decoding Example ===\n")
    
    # Batch example with 2 sentences
    batch_arc_scores = np.random.randn(2, 4, 4)
    batch_label_scores = np.random.randn(2, 4, 4, 3)
    batch_root_scores = np.random.randn(2, 4)
    lengths = [3, 4]  # First sentence has 3 words, second has 4
    
    batch_heads, batch_labels = decode_batch(
        arc_scores=batch_arc_scores,
        label_scores=batch_label_scores,
        root_scores=batch_root_scores,
        lengths=lengths,
        id2label=id2label
    )
    
    print(f"Batch size: {len(batch_heads)}")
    for i, (heads, labels) in enumerate(zip(batch_heads, batch_labels)):
        print(f"\nSentence {i+1} (length={lengths[i]}):")
        print(f"  Heads: {heads}")
        print(f"  Labels: {labels}")

