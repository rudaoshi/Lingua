from enum import IntEnum
from typing import List, Dict, Tuple, Optional

class EdgeValidationCode(IntEnum):
    OK = 0
    WOULD_CREATE_CYCLE = 1
    NEXT_WORD_NOT_ADJACENT = 2
    NEXT_WORD_CHILD_HAS_NON_NEXTWORD_INCOMING = 3
    NEXT_WORD_CHILD_HAS_NON_NEXTWORD_OUTGOING = 4
    NEXT_WORD_CHILD_HAS_AT_OUTGOING = 5
    CHILD_IS_NEXT_WORD_TARGET = 6
    PARENT_IS_NEXT_WORD_TARGET = 7
    UPPER_LOWER_FROM_ROOT = 8
    UPPER_LOWER_PARENT_HAS_NON_AT_INCOMING = 9
    ORDER_SAFETY_CHILD_HAS_UPPER_LOWER_OUTGOING = 10
    AT_CHAIN_PARENT_INCONSISTENT = 11
    AT_CHAIN_NOT_ALLOWED = 12
    AT_CHAIN_CHILD_INCONSISTENT = 13
    AT_SIBLING_PREFIX_CONFLICT = 14
    PATTERN_CONSTRAINT = 15
    ASPUNCT_CAN_ONLY_SILIBING_WITH_NEXTWORD = 16
    DUPLICATED_EDGE_LABELS = 17
    PUNCTUATION_ONLY_CONNECTS_LEAF_NODE = 18
    UPPER_LOWER_CANNOT_CONNECTS_ROOT = 19
    NEXT_WORD_CONSTRAINT_VIOLATED = 20
    PUNCTUATION_CONSTRAINT_VIOLATED = 21
    FACTUAL_PREDICATOR_CONSTRAINT_VIOLATED = 22
    MODIFICATION_TRIANGLE_ALL_MOD_EDGES = 23


class EdgeValidator:
    """
    Stateful edge validator used by constrained decoding phases.

    This intentionally supports slightly different rule-sets for:
    - Phase 2 (candidate edge selection)
    - Phase 3 (connecting unreachable nodes)
    - Phase 4 forced mode (last resort; subset of constraints)

    It returns an EdgeValidationCode to make debugging and reuse easy.
    """

    def __init__(
        self,
        *,
        root_idx: int,
        node_types: List[str],
        node_incoming_edges: Dict[int, List[str]],
        node_outgoing_edges: Dict[int, List[str]],
        children_of: Dict[int, set],
        parents_of: Dict[int, set],
        edge_labels_by_pair: Dict[Tuple[int, int], str],
        next_word_targets: set,
        adjacent_only_types: set,
        allowed_chain_at: set,
        would_create_cycle,
        pattern_check,
    ):
        self.root_idx = int(root_idx)
        self.node_types = node_types
        self.node_incoming_edges = node_incoming_edges
        self.node_outgoing_edges = node_outgoing_edges
        self.children_of = children_of
        self.parents_of = parents_of
        self.edge_labels_by_pair = edge_labels_by_pair
        self.next_word_targets = next_word_targets
        self.adjacent_only_types = adjacent_only_types
        self.allowed_chain_at = allowed_chain_at
        self.would_create_cycle = would_create_cycle  # callable(parent_idx, child_idx)->bool
        self.pattern_check = pattern_check  # callable(parent_idx, label)->bool

    @staticmethod
    def _at_component(label: str, *, allow_prefix: bool = False) -> Optional[str]:
        """
        If the edge LABEL starts with "@", return its "@..." component for constraints.
        Example: "@modification+reverse_pred.arg.1" -> "@modification"
        If allow_prefix=True, allow any label containing "@" and return prefix up to "+".
        """
        s = str(label)
        if allow_prefix:
            if "@" in s:
                return s.split("+", 1)[0]
            return None
        if s.startswith("@"):
            return s.split("+", 1)[0]
        return None

    @staticmethod
    def _is_modification_edge(label: str) -> bool:
        """
        "Modification edge" definition (per user request): label starts with "@modification".
        """
        s = str(label)
        return s.startswith("@modification")

    def _has_any_edge_between(self, a: int, b: int) -> bool:
        """Undirected existence of an edge between nodes a and b (any label, either direction)."""
        aa = int(a)
        bb = int(b)
        return (aa, bb) in self.edge_labels_by_pair or (bb, aa) in self.edge_labels_by_pair

    def _has_mod_edge_between(self, a: int, b: int) -> bool:
        """Undirected existence of an @modification* edge between nodes a and b (either direction)."""
        aa = int(a)
        bb = int(b)
        lbl1 = self.edge_labels_by_pair.get((aa, bb))
        if lbl1 is not None and self._is_modification_edge(lbl1):
            return True
        lbl2 = self.edge_labels_by_pair.get((bb, aa))
        return lbl2 is not None and self._is_modification_edge(lbl2)

    def _neighbors_undirected(self, a: int) -> set:
        """Undirected neighbors of node a based on current directed adjacency."""
        aa = int(a)
        return set(self.children_of.get(aa, set())) | set(self.parents_of.get(aa, set()))

    def _incoming_at_components_for(self, node_idx: int, *, allow_prefix: bool = False) -> set:
        comps = set()
        for e in self.node_incoming_edges.get(node_idx, []):
            atc = self._at_component(e, allow_prefix=allow_prefix)
            if atc is not None:
                comps.add(atc)
        return comps

    def _outgoing_at_components_for(self, node_idx: int, *, allow_prefix: bool = False) -> set:
        comps = set()
        for e in self.node_outgoing_edges.get(node_idx, []):
            atc = self._at_component(e, allow_prefix=allow_prefix)
            if atc is not None:
                comps.add(atc)
        return comps

    def _has_any_non_at_incoming(self, node_idx: int) -> bool:
        # True iff there exists an incoming edge label into node_idx that does NOT contain "@"
        return any("@" not in str(e) for e in self.node_incoming_edges.get(node_idx, []))

    def _has_any_upper_lower_outgoing(self, node_idx: int) -> bool:
        """Check if node has any @upper or @lower outgoing edges."""
        return any(
            str(e) == "@upper" or str(e) == "@lower" or 
            str(e).startswith("upper_") or str(e).startswith("lower_")  # backward compat
            for e in self.node_outgoing_edges.get(node_idx, [])
        )

    def validate(
        self,
        parent_idx: int,
        child_idx: int,
        label: str,
        *,
        mode: str,
        check_cycle: bool = True,
        check_pattern: bool = True,
    ) -> EdgeValidationCode:
        """
        Validate a prospective edge (parent_idx -> child_idx with label).
        `mode` controls small behavioral differences between decoding phases.
        """
        p_idx = int(parent_idx)
        c_idx = int(child_idx)
        lbl = str(label)

        lbl_edges = set(lbl.split("+"))

        # Constraint 1: no cycles (DAG constraint)
        if check_cycle and self.would_create_cycle is not None:

            if self.would_create_cycle(p_idx, c_idx):
                    return EdgeValidationCode.WOULD_CREATE_CYCLE

            if any(edge.startswith("reverse_") for edge in lbl_edges):
                if self.would_create_cycle(c_idx, p_idx):
                    return EdgeValidationCode.WOULD_CREATE_CYCLE

        # -------------------------
        # @modification transitive constraint
        # -------------------------
        # New rule:
        # Let A,B,C be three nodes. If there is an edge between A and B (any label, either direction),
        # and there is an edge starting with "@modification" between B and C (either direction),
        # then there should be NO edge starting with "@modification" between A and C (either direction).
        #
        # We enforce this incrementally for ANY newly proposed edge (not only modification edges),
        # by checking only patterns that could be affected by adding (p_idx, c_idx).
        will_add_mod = self._is_modification_edge(lbl)

        # 1) Adding an edge between U and V must not connect two nodes that both already have
        #    @modification edges to the same third node W.
        u = p_idx
        v = c_idx
        cand_w = self._neighbors_undirected(u) | self._neighbors_undirected(v)
        for w in cand_w:
            w = int(w)
            if w == u or w == v:
                continue
            if self._has_mod_edge_between(u, w) and self._has_mod_edge_between(v, w):
                return EdgeValidationCode.MODIFICATION_TRIANGLE_ALL_MOD_EDGES

        # 2) If the new edge is itself an @modification edge between U and V, then for every neighbor A of U:
        #    (A-U is an edge) and (A-V is @modification) is forbidden. Symmetrically for neighbors of V.
        if will_add_mod:
            for a in self._neighbors_undirected(u):
                a = int(a)
                if a == v:
                    continue
                if self._has_mod_edge_between(a, v):
                    return EdgeValidationCode.MODIFICATION_TRIANGLE_ALL_MOD_EDGES
            for a in self._neighbors_undirected(v):
                a = int(a)
                if a == u:
                    continue
                if self._has_mod_edge_between(a, u):
                    return EdgeValidationCode.MODIFICATION_TRIANGLE_ALL_MOD_EDGES

        
        # -------------------------
        # FactualPredicator constraint
        # -------------------------

        if self.node_types[p_idx] == "FactualPredicator":
            if  "as:modification" in lbl_edges:
                return EdgeValidationCode.FACTUAL_PREDICATOR_CONSTRAINT_VIOLATED

        # if self.node_types[p_idx] == "PunctuationalConstant":
        #     if not "as:punctuation" in lbl_edges:
        #         return EdgeValidationCode.PUNCTUATION_ONLY_CONNECTS_LEAF_NODE
            
        if "punctuation" in lbl_edges:
            if self.node_outgoing_edges.get(c_idx, []):
                return EdgeValidationCode.PUNCTUATION_ONLY_CONNECTS_LEAF_NODE

        if any(edge == "punctuation" for edge in self.node_incoming_edges.get(c_idx, [])):
            return EdgeValidationCode.PUNCTUATION_CONSTRAINT_VIOLATED

        if any(edge.startswith("upper_@") or edge.startswith("lower_@") for edge in lbl_edges):
            if not self.node_incoming_edges.get(p_idx, []):
                return EdgeValidationCode.UPPER_LOWER_CANNOT_CONNECTS_ROOT


        if "next_word" in lbl_edges:
            if self.node_incoming_edges.get(c_idx, []):
                return EdgeValidationCode.NEXT_WORD_CONSTRAINT_VIOLATED

        if any("next_word" in edge for edge in self.node_incoming_edges.get(c_idx, [])):
            return EdgeValidationCode.NEXT_WORD_CONSTRAINT_VIOLATED


        # as:punctuation can only sibling with next_word
        # if all(edge != "next_word" and "@" not in edge for edge in lbl_edges):
        #     if any(e == "as:punctuation" for e in self.node_incoming_edges.get(p_idx, [])):
        #         return EdgeValidationCode.ASPUNCT_CAN_ONLY_SILIBING_WITH_NEXTWORD
        # if any(edge == "as:punctuation" for edge in lbl_edges):
        #     if any(e != "next_word" for e in self.node_outgoing_edges.get(p_idx, [])):
        #         return EdgeValidationCode.ASPUNCT_CAN_ONLY_SILIBING_WITH_NEXTWORD

        existing_silibing_edges = set()
        for e in self.node_outgoing_edges.get(p_idx, []):
            existing_silibing_edges.update(e.split("+"))

        for e in self.node_incoming_edges.get(p_idx, []):
            edges = e.split("+")
            for edge in edges:
                if edge.startswith("reverse_"):
                    existing_silibing_edges.add(edge[len("reverse_"):])
            
        overlapping_edges = lbl_edges & existing_silibing_edges
        if any(edge in {"pred.arg.1", "pred.arg.2", "pred.arg.3", "pred.arg.4", 
                "variable", "body"} for edge in overlapping_edges):
            return EdgeValidationCode.DUPLICATED_EDGE_LABELS

        if any(edge.startswith("reverse_") for edge in lbl_edges):
            reverse_lbl_edges = set(edge[len("reverse_"):] for edge in lbl_edges if edge.startswith("reverse_"))
            existing_silibing_edges = set()
            for e in self.node_outgoing_edges.get(c_idx, []):
                existing_silibing_edges.update(e.split("+"))

            for e in self.node_incoming_edges.get(c_idx, []):
                edges = e.split("+")
                for edge in edges:
                    if edge.startswith("reverse_"):
                        existing_silibing_edges.add(edge[len("reverse_"):])
                
            overlapping_edges = reverse_lbl_edges & existing_silibing_edges
            if any(edge in {"pred.arg.1", "pred.arg.2", "pred.arg.3", "pred.arg.4", 
                    "variable", "body"} for edge in overlapping_edges):
                return EdgeValidationCode.DUPLICATED_EDGE_LABELS
        
        # -------------------------
        # next_word constraints
        # -------------------------
        # NOTE: Phase 4 forced mode intentionally does NOT enforce the strict next_word constraints.
        if mode != "phase4_forced":
            if lbl == "next_word":
                p_type = self.node_types[p_idx]
                c_type = self.node_types[c_idx]
                dist = abs(p_idx - c_idx)

                # adjacency constraint for *Constant types
                if (p_type in self.adjacent_only_types or c_type in self.adjacent_only_types) and dist != 1:
                    return EdgeValidationCode.NEXT_WORD_NOT_ADJACENT

                # target node constraints
                non_nw_incoming = [e for e in self.node_incoming_edges[c_idx] if e != "next_word"]
                if non_nw_incoming:
                    return EdgeValidationCode.NEXT_WORD_CHILD_HAS_NON_NEXTWORD_INCOMING

                non_nw_outgoing = [e for e in self.node_outgoing_edges[c_idx] if e != "next_word"]
                if non_nw_outgoing:
                    return EdgeValidationCode.NEXT_WORD_CHILD_HAS_NON_NEXTWORD_OUTGOING

                # Phase 3 is stricter: don't make a node a next_word target if it already emits "@..."
                if mode == "phase3":
                    if any("@" in str(out_lbl) for out_lbl in self.node_outgoing_edges.get(c_idx, [])):
                        return EdgeValidationCode.NEXT_WORD_CHILD_HAS_AT_OUTGOING
            else:
                # Non-next_word edge cannot target a next_word target
                if c_idx in self.next_word_targets:
                    return EdgeValidationCode.CHILD_IS_NEXT_WORD_TARGET

            # next_word targets can only have next_word outgoing
            if p_idx in self.next_word_targets and lbl != "next_word":
                return EdgeValidationCode.PARENT_IS_NEXT_WORD_TARGET
        else:
            # Forced mode keeps only: nodes with incoming next_word cannot emit edges containing "@"
            if p_idx in self.next_word_targets and ("@" in lbl):
                return EdgeValidationCode.PARENT_IS_NEXT_WORD_TARGET

        # -------------------------
        # @upper/@lower constraints
        # -------------------------
        # Note: @upper and @lower are added in Phase 5, not during regular decoding.
        # But we still validate them if they appear.
        if lbl in ("@upper", "@lower") or lbl.startswith("upper_") or lbl.startswith("lower_"):
            # Root cannot emit @upper/@lower edges
            if p_idx == self.root_idx:
                return EdgeValidationCode.UPPER_LOWER_FROM_ROOT
            if self._has_any_non_at_incoming(p_idx):
                return EdgeValidationCode.UPPER_LOWER_PARENT_HAS_NON_AT_INCOMING

        # opposite direction for order safety
        if ("@" not in lbl) and self._has_any_upper_lower_outgoing(c_idx):
            return EdgeValidationCode.ORDER_SAFETY_CHILD_HAS_UPPER_LOWER_OUTGOING

        # -------------------------
        # @-chain consistency
        # -------------------------
        if mode != "phase4_forced":
            # Parent-side: if parent has incoming @x, outgoing @y must satisfy y==x and be allowed.
            atc_new = self._at_component(lbl)
            if atc_new is not None:
                parent_in_atcs = self._incoming_at_components_for(p_idx, allow_prefix=True)
                if parent_in_atcs:
                    if len(parent_in_atcs) != 1 or atc_new not in parent_in_atcs:
                        return EdgeValidationCode.AT_CHAIN_PARENT_INCONSISTENT
                    if atc_new not in self.allowed_chain_at:
                        return EdgeValidationCode.AT_CHAIN_NOT_ALLOWED

            # Child-side: if child already emits @y, adding prefix_@x edge must match/allow.
            atc_new_allow_prefix = self._at_component(lbl, allow_prefix=True)
            if atc_new_allow_prefix is not None:
                child_out_atcs = self._outgoing_at_components_for(c_idx)
                if child_out_atcs:
                    # order-sensitive logic in the existing code uses a single outgoing @ component
                    child_out_atc = child_out_atcs.pop()
                    if child_out_atc not in atc_new_allow_prefix:
                        return EdgeValidationCode.AT_CHAIN_CHILD_INCONSISTENT
                    if child_out_atc not in self.allowed_chain_at:
                        return EdgeValidationCode.AT_CHAIN_NOT_ALLOWED

                # sibling-prefix exclusion exists in Phase 2/3 only (not in forced Phase 4)
                if mode in ("phase2", "phase3"):
                    sibling_atcs = self._outgoing_at_components_for(p_idx, allow_prefix=True)
                    atc_prefix = atc_new_allow_prefix.split("@", 1)[0]
                    if any(atc_prefix == s.split("@", 1)[0] for s in sibling_atcs):
                        return EdgeValidationCode.AT_SIBLING_PREFIX_CONFLICT
        else:
            # Forced mode mirrors the existing Phase 4 forced-label logic, which uses stricter/older
            # @-component extraction and effectively forbids targeting a node that already emits any "@..." edge.
            atc_new = self._at_component(lbl)
            if atc_new is not None:
                parent_in_atcs = self._incoming_at_components_for(p_idx, allow_prefix=False)
                if parent_in_atcs:
                    if len(parent_in_atcs) != 1 or atc_new not in parent_in_atcs:
                        return EdgeValidationCode.AT_CHAIN_PARENT_INCONSISTENT
                    if atc_new not in self.allowed_chain_at:
                        return EdgeValidationCode.AT_CHAIN_NOT_ALLOWED

                # In the original forced-mode code, this check ends up rejecting whenever the child
                # has ANY outgoing "@..." edge (due to a set-vs-str comparison). We preserve that behavior.
                child_out_atcs = self._outgoing_at_components_for(c_idx, allow_prefix=False)
                if child_out_atcs:
                    return EdgeValidationCode.AT_CHAIN_CHILD_INCONSISTENT
                if atc_new not in self.allowed_chain_at:
                    return EdgeValidationCode.AT_CHAIN_NOT_ALLOWED

        # Note: The old constraint "at most one outgoing @-edge per node" has been removed.
        # According to the new word-lingua design, nodes can have multiple PRIMARY_AT_EDGES.
        # The @upper/@lower edges are added in Phase 5 as post-processing.

        # -------------------------
        # pattern constraint (Phase 2/3)
        # -------------------------
        if check_pattern and self.pattern_check is not None and mode in ("phase2", "phase3"):
            if not self.pattern_check(p_idx, lbl):
                return EdgeValidationCode.PATTERN_CONSTRAINT

        return EdgeValidationCode.OK