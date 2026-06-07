from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import os
from collections import defaultdict
from lingua.structure.gpgraph import GPGraph, GPGPhraseNode, GPGEdge, GPGAuxNode
from lingua.learn.wordgraph.decoding.statistics import WordLinguaStatistics
from lingua.learn.wordgraph.decoding.visualizer import ProbabilityAwareVisualizer, GPGraphVisualizer
from lingua.learn.wordgraph.decoding.validator import EdgeValidationCode
from lingua.learn.wordgraph.decoding.validator import EdgeValidator


import logging 
logger = logging.getLogger(__name__)


class GreedyPatternDecoder:
    """
    SDP-style graph decoder that constructs a scored subgraph edge-by-edge.

    Strategy:
    1. Select candidate edges with p(edge) > threshold.
    2. Score each candidate as p(edge) * p(label), optionally reweighted by local
       statistics.
    3. Greedily select the highest-scoring edge that does not violate local graph
       constraints.

    By default this decoder keeps the word-lingua single-root DAG contract by
    adding best-scored root attachments, but it does not rewrite/move existing
    semantic edges. This keeps high-confidence model edges as the primary object
    while avoiding the old Phase 4 failure mode where a good edge was moved to a
    low-confidence alternative.
    """

    # Keep repair-time hierarchy hints on the same probability footing as real
    # model-selected edges: both pieces and their joint confidence must be high.
    MIN_LABEL_PROB = 0.01
    MIN_HIERARCHY_HINT_CONFIDENCE = 0.5
    HIERARCHY_HINT_EDGES = {"@upper", "@lower"}
    
    def __init__(self, stats: Any, id2label: Dict[int, str],
                 label2id: Dict[str, int], arc_threshold: float = 0.5,
                 edge_prob_tables: Optional[WordLinguaStatistics] = None,
                 lam_parent_prob: float = 1.0,
                 lam_sibling_prob: float = 1.0,
                 lam_pos_prob: float = 1.0,
                 lam_parent_given_child_prob: float = 1.0,
                 id2node_type: Optional[Dict[int, str]] = None,
                 debug: bool = False,
                 debug_dir: Optional[str] = None,
                 debug_example_id: Optional[int] = None,
                 enforce_connected: bool = False,
                 enable_phase4_rewrites: bool = False,
                 enforce_single_root: bool = True,
                 decoder_mode: str = "ilp",
                 ilp_max_candidates_per_child: int = 16,
                 ilp_max_labels_per_pair: int = 4,
                 ilp_time_limit: float = 30.0,
                 ilp_mip_rel_gap: float = 0.001,
                 ilp_root_score_weight: float = 1.0,
                 ilp_max_triangle_constraints: int = 200000,
                 ilp_lazy_constraints: bool = True,
                 ilp_max_lazy_rounds: int = 4):
        # NOTE: legacy `edge_pattern_stats.json` is no longer used. `stats` is kept only
        # for backward compatibility and is ignored.
        self.stats = None
        self.id2label = id2label
        self.label2id = label2id
        self.arc_threshold = arc_threshold
        self.edge_prob_tables = edge_prob_tables
        self.lam_parent_prob = float(lam_parent_prob)
        self.lam_sibling_prob = float(lam_sibling_prob)
        self.lam_pos_prob = float(lam_pos_prob)
        self.lam_parent_given_child_prob = float(lam_parent_given_child_prob)
        if id2node_type is None:
            raise ValueError("id2node_type must be provided by the loaded model vocabulary")
        self.id2node_type = id2node_type
        self.debug = debug
        self.debug_dir = debug_dir
        self.debug_example_id = debug_example_id
        self.debug_step_counter = 0
        self.enforce_connected = bool(enforce_connected)
        self.enable_phase4_rewrites = bool(enable_phase4_rewrites)
        self.enforce_single_root = bool(enforce_single_root)
        self.decoder_mode = str(decoder_mode or "ilp")
        if self.decoder_mode not in {"ilp", "sdp_graph"}:
            raise ValueError(f"Unsupported decoder_mode: {self.decoder_mode}")
        self.ilp_max_candidates_per_child = int(ilp_max_candidates_per_child)
        self.ilp_max_labels_per_pair = int(ilp_max_labels_per_pair)
        self.ilp_time_limit = float(ilp_time_limit)
        self.ilp_mip_rel_gap = float(ilp_mip_rel_gap)
        self.ilp_root_score_weight = float(ilp_root_score_weight)
        self.ilp_max_triangle_constraints = int(ilp_max_triangle_constraints)
        self.ilp_lazy_constraints = bool(ilp_lazy_constraints)
        self.ilp_max_lazy_rounds = int(ilp_max_lazy_rounds)
    
    def _debug_visualize_step(self, graph: GPGraph, phase: str, step_desc: str, 
                              arc_probs: Optional[np.ndarray] = None,
                              rel_probs: Optional[np.ndarray] = None,
                              label2id: Optional[Dict[str, int]] = None):
        """Visualize intermediate graph state during decoding."""
        if not self.debug or self.debug_dir is None or self.debug_example_id is None:
            return
        
        self.debug_step_counter += 1
        step_str = f"{self.debug_step_counter:04d}"
        filename = f"{self.debug_example_id:05d}.{step_str}.{phase}.{step_desc}.png"
        filepath = os.path.join(self.debug_dir, filename)
        
        try:
            os.makedirs(self.debug_dir, exist_ok=True)
            visualizer = ProbabilityAwareVisualizer() if (arc_probs is not None and rel_probs is not None) else GPGraphVisualizer()
            
            # Add probability info if available
            if arc_probs is not None and rel_probs is not None:
                node_id_to_word_idx = {}
                for node in graph.nodes():
                    if hasattr(node, 'spans') and node.spans:
                        first_span = node.spans[0]
                        if isinstance(first_span, int):
                            word_idx = first_span
                        elif isinstance(first_span, (tuple, list)) and len(first_span) >= 1:
                            word_idx = first_span[0]
                        else:
                            continue
                        node_id_to_word_idx[node.ID] = word_idx
                
                for start_node, edge, end_node in graph.edges():
                    if start_node.ID in node_id_to_word_idx and end_node.ID in node_id_to_word_idx:
                        parent_idx = node_id_to_word_idx[start_node.ID]
                        child_idx = node_id_to_word_idx[end_node.ID]
                        
                        if (parent_idx < arc_probs.shape[0] and child_idx < arc_probs.shape[1] and
                            parent_idx < rel_probs.shape[0] and child_idx < rel_probs.shape[1]):
                            edge.arc_prob = float(arc_probs[parent_idx, child_idx])
                            
                            if label2id is not None and hasattr(edge, 'label') and edge.label in label2id:
                                label_id = label2id[edge.label]
                                if label_id < rel_probs.shape[2]:
                                    edge.rel_prob = float(rel_probs[parent_idx, child_idx, label_id])
                                else:
                                    edge.rel_prob = 0.0
                            else:
                                edge.rel_prob = float(np.max(rel_probs[parent_idx, child_idx]))
            
            visualizer.visualize(graph, file_name=filepath, format="png")
            logger.debug(f"Debug visualization saved: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save debug visualization {filepath}: {e}")

    def _debug_log_phase34_report(self, words: List[str], decoding_info: Dict[str, Any]) -> None:
        """
        In debug-one-sample mode, print a detailed report for Phase 3 and Phase 4.
        This complements step-by-step visualization by explaining *why* some edges
        were rejected (validator codes, etc.).
        """
        if not self.debug:
            return
        try:
            ex = self.debug_example_id
            logger.info("\n" + "=" * 100)
            logger.info(f"DEBUG Phase 3/4 report (example_id={ex})")
            logger.info("=" * 100)

            # -------------------------
            # Phase 3
            # -------------------------
            phase3_components = decoding_info.get("phase3_components", []) or []
            phase3_failures = decoding_info.get("phase3_failures", []) or []
            logger.info(f"[Phase 3] components_processed={len(phase3_components)} failures={len(phase3_failures)}")
            for k, comp in enumerate(phase3_components):
                nodes = comp.get("component_nodes", [])
                logger.info(f"  - component#{k+1}: size={len(nodes)} nodes={nodes}")
                if comp.get("selected_edge") is not None:
                    sel = comp["selected_edge"]
                    logger.info(
                        "    selected_edge="
                        f"{sel.get('parent')}->{sel.get('child')} label='{sel.get('label')}' "
                        f"mode={sel.get('mode')} arc_prob={sel.get('arc_prob'):.4f} rel_prob={sel.get('rel_prob'):.4f} "
                        f"score={sel.get('score'):.6f}"
                    )
                if comp.get("stop_message"):
                    logger.info(f"    stop_message={comp.get('stop_message')}")
                top_fails = comp.get("top_failures", []) or []
                if top_fails:
                    logger.info(f"    top_rejected_candidates={min(len(top_fails), 10)} (showing up to 10)")
                    for f in top_fails[:10]:
                        p = f.get("parent")
                        c = f.get("child")
                        arc_p = f.get("arc_prob", 0.0)
                        mode = f.get("mode")
                        thr = f.get("enforce_arc_threshold")
                        logger.info(f"      * {p}->{c} mode={mode} enforce_arc_threshold={thr} arc_prob={arc_p:.4f}")
                        attempts = f.get("label_attempts", []) or []
                        for a in attempts[:5]:
                            lbl = a.get("label")
                            rp = a.get("rel_prob", 0.0)
                            code = a.get("validator_code")
                            logger.info(f"          - label='{lbl}' rel_prob={rp:.4f} reject={code}")

            if phase3_failures:
                # include a brief summary of failure messages (detailed info is in the entries above)
                for fail in phase3_failures[:5]:
                    logger.info(f"  [Phase 3 failure] {fail.get('message')}")

            # -------------------------
            # Phase 4
            # -------------------------
            phase4_fixes = decoding_info.get("phase4_multihead_fixes", []) or []
            phase4_add_edge_failures = decoding_info.get("phase4_add_edge_failures", []) or []
            logger.info(f"[Phase 4] multihead_fixes={len(phase4_fixes)} add_edge_failures={len(phase4_add_edge_failures)}")
            if phase4_fixes:
                for fx in phase4_fixes[:10]:
                    logger.info(
                        "  - fix: "
                        f"phase4_root={fx.get('phase4_root')} predicted_root={fx.get('predicted_root')} "
                        f"head={fx.get('head')} common_child={fx.get('common_child')} "
                        f"moved_edges={len(fx.get('moved_parent_edges', []) or [])}"
                    )
            if phase4_add_edge_failures:
                logger.info(f"  top_add_edge_failures={min(len(phase4_add_edge_failures), 30)} (showing up to 30)")
                for f in phase4_add_edge_failures[:30]:
                    p = f.get("parent")
                    c = f.get("child")
                    lbl = f.get("label")
                    reason = f.get("reason")
                    vcode = f.get("validator_code")
                    meta = f.get("meta")
                    extra = ""
                    if isinstance(meta, dict):
                        # keep this compact
                        keys = ["old_child", "head", "common_child", "phase4_root", "predicted_root"]
                        kv = {k: meta.get(k) for k in keys if k in meta}
                        if kv:
                            extra = f" meta={kv}"
                    if vcode is not None:
                        logger.info(f"    * {p}->{c} label='{lbl}' reject={vcode} reason={reason}{extra}")
                    else:
                        logger.info(f"    * {p}->{c} label='{lbl}' reason={reason}{extra}")

            logger.info("=" * 100 + "\n")
        except Exception as e:
            logger.warning(f"Failed to produce Phase 3/4 debug report: {e}", exc_info=True)
    
    def decode(
        self,
        words: List[str],
        arc_logits: np.ndarray,
        rel_logits: np.ndarray,
        node_type_preds: np.ndarray,
        child_of_whether_preds: np.ndarray,
        is_root_logits: Optional[np.ndarray] = None,  # [num_words, 2] (optional)
        arc_probs: Optional[np.ndarray] = None,  # For debug visualization
        rel_probs: Optional[np.ndarray] = None,  # For debug visualization
    ) -> Tuple[GPGraph, Dict[str, Any]]:
        """
        Decode predictions using bottom-up greedy search with pattern constraints.
        
        Returns:
            Tuple of (graph, decoding_info) where decoding_info contains:
            - errors: list of error messages
            - warnings: list of warning messages
            - forced_connections: list of forced edge connections
            - disconnected_nodes: list of nodes that couldn't be connected
        """
        num_words = len(words)
        
        # Track decoding issues
        decoding_info = {
            "decoder_mode": self.decoder_mode,
            "decoder_options": {
                "enforce_connected": bool(self.enforce_connected),
                "enable_phase4_rewrites": bool(self.enable_phase4_rewrites),
                "enforce_single_root": bool(self.enforce_single_root),
                "ilp_max_candidates_per_child": int(self.ilp_max_candidates_per_child),
                "ilp_max_labels_per_pair": int(self.ilp_max_labels_per_pair),
                "ilp_time_limit": float(self.ilp_time_limit),
                "ilp_mip_rel_gap": float(self.ilp_mip_rel_gap),
                "ilp_root_score_weight": float(self.ilp_root_score_weight),
                "ilp_lazy_constraints": bool(self.ilp_lazy_constraints),
                "ilp_max_lazy_rounds": int(self.ilp_max_lazy_rounds),
            },
            "errors": [],
            "warnings": [],
            "forced_connections": [],
            "disconnected_nodes": [],
            "phase3_failures": [],
            "phase4_force_connects": [],
            "phase4_multihead_fixes": [],
            # Debug-only: detailed per-component traces for Phase 3/4
            "phase3_components": [],
            "phase4_add_edge_failures": [],
        }
        
        # Get node types from the model bundle vocabulary.
        node_types = []
        for i in range(num_words):
            type_id = int(node_type_preds[i])
            if type_id not in self.id2node_type:
                raise ValueError(f"Predicted node type id is not in model vocabulary: {type_id}")
            node_types.append(self.id2node_type[type_id])
        
        # Convert logits to probabilities
        arc_probs = self._sigmoid(arc_logits)
        rel_probs = self._softmax(rel_logits, axis=-1)

        def _edge_model_probs(p_idx: int, c_idx: int, label: str) -> Tuple[float, float, float]:
            p_idx = int(p_idx)
            c_idx = int(c_idx)
            arc_prob = float(arc_probs[p_idx, c_idx]) if (p_idx < arc_probs.shape[0] and c_idx < arc_probs.shape[1]) else 0.0
            rel_prob = 0.0
            if label in self.label2id:
                lid = int(self.label2id[label])
                if p_idx < rel_probs.shape[0] and c_idx < rel_probs.shape[1] and lid < rel_probs.shape[2]:
                    rel_prob = float(rel_probs[p_idx, c_idx, lid])
            return arc_prob, rel_prob, float(arc_prob * rel_prob)

        def _hierarchy_hint_skip_reason(
            p_idx: int,
            c_idx: int,
            label: str,
            *,
            arc_prob: Optional[float] = None,
            rel_prob: Optional[float] = None,
        ) -> Optional[str]:
            if str(label) not in self.HIERARCHY_HINT_EDGES:
                return None
            if arc_prob is None or rel_prob is None:
                arc_prob, rel_prob, _confidence = _edge_model_probs(p_idx, c_idx, label)
            if float(arc_prob) < float(self.arc_threshold):
                return "arc_prob_below_threshold"
            if float(rel_prob) < self.MIN_LABEL_PROB:
                return "rel_prob_below_threshold"
            if float(arc_prob) * float(rel_prob) < self.MIN_HIERARCHY_HINT_CONFIDENCE:
                return "confidence_below_threshold"
            return None

        # Choose the real root word index (NO virtual ROOT node exists in the graph).
        # Prefer the model's is_root head when available; otherwise fall back to in-degree heuristic.
        if num_words <= 0:
            root_idx = 0
        elif is_root_logits is not None and isinstance(is_root_logits, np.ndarray) and is_root_logits.shape[0] >= num_words and is_root_logits.shape[-1] >= 2:
            r = is_root_logits[:num_words, :2].astype(np.float32)
            # score = logit(root=1) - logit(root=0)
            root_scores = r[:, 1] - r[:, 0]
            root_idx = int(np.argmax(root_scores))
        else:
            arc_bin = (arc_probs > float(self.arc_threshold))
            # ignore self-loops
            if arc_bin.shape[0] == arc_bin.shape[1]:
                np.fill_diagonal(arc_bin, False)
            indeg = arc_bin.sum(axis=0)
            roots = np.where(indeg == 0)[0]
            root_idx = int(roots.min()) if roots.size > 0 else int(indeg.argmin())

        # Build graph
        graph = GPGraph()
        graph.words = words
        
        # Create all nodes
        word_to_node = {}
        for i, word in enumerate(words):
            node = GPGPhraseNode(
                ID=str(i),
                spans=[(i, i)],
                pos=node_types[i]
            )
            if child_of_whether_preds[i]:
                node.child_of_whether = True
            graph.add_node(node)
            word_to_node[i] = node

        # Debug: visualize initial empty graph
        if self.debug:
            self._debug_visualize_step(
                graph, "init", "empty_graph",
                arc_probs=arc_probs, rel_probs=rel_probs,
                label2id={v: k for k, v in self.label2id.items()}
            )

        if self.decoder_mode == "ilp":
            return self._decode_ilp_single_root_dag(
                words=words,
                graph=graph,
                word_to_node=word_to_node,
                node_types=node_types,
                arc_probs=arc_probs,
                rel_probs=rel_probs,
                root_idx=root_idx,
                is_root_logits=is_root_logits,
                decoding_info=decoding_info,
            )
        
        # Track outgoing edges for each node (for pattern constraint checking)
        node_outgoing_edges = {i: [] for i in range(num_words)}
        # Track incoming edges for each node (for next_word constraint checking)
        node_incoming_edges = {i: [] for i in range(num_words)}
        # Track which nodes are targets of next_word edges (i.e., nodes whose incoming edge includes "next_word")
        next_word_targets = set()
        # Track graph structure for cycle detection
        children_of = {i: set() for i in range(num_words)}
        # Track reverse adjacency (kept in sync with children_of; validator holds a reference)
        parents_of = {i: set() for i in range(num_words)}
        # Track chosen label per directed pair (parent_idx, child_idx) for triangle constraint checks
        edge_labels_by_pair: Dict[Tuple[int, int], str] = {}
        
        # Define constant types that require adjacent next_word edges
        ADJACENT_ONLY_TYPES = {
            "ModificationalConstant", "DeterminerConstant", "OtherConstant",
            "NominalConstant", "SymbolConstant"
        }
        # Note: PunctuationalConstant is excluded (allows non-adjacent for paired punctuation)
        
        # Phase 1: Collect all candidate edges with p(edge) > threshold
        candidate_edges = []
        filtered_edges = []  # Track edges that don't make it into candidates
        
        for i in range(num_words):
            for j in range(num_words):
                if i == j:
                    continue
                arc_prob = arc_probs[i, j]
                
                # Track edges filtered by arc_threshold
                if arc_prob <= self.arc_threshold:
                    if self.debug:
                        # Get the best label for this pair (even if arc_prob is low)
                        best_label_id = int(np.argmax(rel_probs[i, j]))
                        best_label = self.id2label.get(best_label_id, "UNK")
                        best_rel_prob = rel_probs[i, j, best_label_id]
                        filtered_edges.append({
                            "parent": i,
                            "child": j,
                            "label": best_label,
                            "arc_prob": arc_prob,
                            "rel_prob": best_rel_prob,
                            "score": arc_prob * best_rel_prob,
                            "filter_reason": f"arc_prob <= threshold ({arc_prob:.4f} <= {self.arc_threshold:.4f})",
                        })
                    continue
                
                # Get label probabilities
                for label_id in range(rel_probs.shape[2]):
                    label = self.id2label.get(label_id, "UNK")
                    if label == "UNK" or label not in self.label2id:
                        continue
                    rel_prob = rel_probs[i, j, label_id]
                    
                    if rel_prob > 0.01:  # Filter very low probability labels
                        candidate_edges.append({
                            "parent": i,
                            "child": j,
                            "label": label,
                            "label_id": label_id,
                            "arc_prob": arc_prob,
                            "rel_prob": rel_prob,
                            # base score (without structural factors)
                            "score": arc_prob * rel_prob,
                        })
                    elif self.debug:
                        # Track labels filtered by rel_prob threshold
                        filtered_edges.append({
                            "parent": i,
                            "child": j,
                            "label": label,
                            "arc_prob": arc_prob,
                            "rel_prob": rel_prob,
                            "score": arc_prob * rel_prob,
                            "filter_reason": f"rel_prob <= threshold ({rel_prob:.4f} <= 0.01)",
                        })
        
        # If we have structural probability tables, use them to re-weight candidate ranking.
        # Note: at this stage we don't yet know the true incoming edge for each node, so we
        # use p(child_edge_label | node_pos) which doesn't require parent edge information.
        # During Phase 2/3/4 we re-weight again using the current inferred incoming edge + already-chosen sibling edges.
        if self.edge_prob_tables is not None:
            for e in candidate_edges:
                parent_idx = int(e["parent"])
                label = str(e["label"])
                node_pos = node_types[parent_idx]
                # Use p(child | pos) when we don't have parent edge information yet
                struct = self.edge_prob_tables.get_prob_child_given_pos(node_pos, label) ** self.lam_pos_prob
                # no sibling factor yet (no outgoing edges chosen in Phase 1)
                e["score_base"] = float(e["score"])
                e["score"] = float(e["score"] * struct)

        # Sort by score (descending)
        candidate_edges.sort(key=lambda x: -x["score"])
        
        # Debug: log filtered edges (not added to candidate_edges) grouped by node pairs
        if self.debug and filtered_edges:
            # Group filtered edges by (parent_idx, child_idx) pairs
            filtered_by_pair = defaultdict(list)
            for edge in filtered_edges:
                pair = (edge["parent"], edge["child"])
                filtered_by_pair[pair].append(edge)
            
            # Count filter reasons
            arc_filtered = sum(1 for e in filtered_edges if "arc_prob <= threshold" in e["filter_reason"])
            rel_filtered = sum(1 for e in filtered_edges if "rel_prob <= threshold" in e["filter_reason"])
            
            # Sort pairs for consistent output
            sorted_filtered_pairs = sorted(filtered_by_pair.keys())
            
            logger.debug(f"\n{'='*80}")
            logger.debug(f"Phase 1: Filtered edges (NOT added to candidate_edges)")
            logger.debug(f"{'='*80}")
            logger.debug(f"Total candidate edges: {len(candidate_edges)}")
            logger.debug(f"Total filtered edges: {len(filtered_edges)}")
            logger.debug(f"  - Filtered by arc_threshold: {arc_filtered}")
            logger.debug(f"  - Filtered by rel_prob threshold: {rel_filtered}")
            logger.debug(f"Grouped by {len(filtered_by_pair)} node pairs:\n")
            
            for (p_idx, c_idx) in sorted_filtered_pairs:
                parent_word = words[p_idx] if p_idx < len(words) else f"node_{p_idx}"
                child_word = words[c_idx] if c_idx < len(words) else f"node_{c_idx}"
                logger.debug(f"  Pair ({p_idx}->{c_idx}): '{parent_word}' -> '{child_word}'")
                
                # Sort by score (descending)
                edges_for_pair = sorted(filtered_by_pair[(p_idx, c_idx)], 
                                       key=lambda x: -x["score"])
                
                for i, edge in enumerate(edges_for_pair):
                    logger.debug(
                        f"    [{i+1}] label='{edge['label']:20s}' | "
                        f"arc={edge['arc_prob']:.4f} | rel={edge['rel_prob']:.4f} | "
                        f"score={edge['score']:.4f} | {edge['filter_reason']}"
                    )
                logger.debug("")
            
            logger.debug(f"{'='*80}\n")
        
        # Helper function to check if adding edge would create a cycle
        def would_create_cycle(parent: int, child: int) -> bool:
            """Check if child can reach parent through existing edges (would create cycle)."""
            visited = set()
            stack = [child]
            while stack:
                node = stack.pop()
                if node == parent:
                    return True
                if node in visited:
                    continue
                visited.add(node)
                stack.extend(children_of[node])
            return False
        
        # Phase 2: Greedily select edges while respecting constraints
        # After each edge is added, recompute scores for remaining candidates and reorder
        selected_edges = []
        
        # Constraint (chain @->@):
        # nodeA -[@edge1]-> nodeB -[@edge2]-> nodeC must satisfy:
        #   - edge1 == edge2  (compare @-component)
        #   - edge1 is restricted to { @list, @parataxis, @expression }
        ALLOWED_CHAIN_AT = {"@list", "@parataxis", "@expression"}
        
        def _incoming_edge_label_for(node_idx: int) -> str:
            # In training stats, nodes with no incoming edge are treated as having incoming_edge="ROOT".
            
            if node_incoming_edges.get(node_idx):
                # Prefer the first non-next_word incoming edge if present.
                for e in node_incoming_edges[node_idx]:
                    if e != "next_word":
                        return e
                return node_incoming_edges[node_idx][0]
            return "ROOT"

        def _compute_edge_score(edge: Dict) -> float:
            """Compute the current structural score for an edge based on current graph state."""
            parent_idx = edge["parent"]
            child_idx = edge["child"]
            label = edge["label"]
            
            original_score = edge["arc_prob"] * edge["rel_prob"]
            struct_factor = 1.0
            
            if self.edge_prob_tables is not None:
                incoming = _incoming_edge_label_for(parent_idx)
                node_pos = node_types[parent_idx]
                # Use p(child | parent, pos) when we have parent edge information
                struct = self.edge_prob_tables.get_prob_child_given_parent(incoming, node_pos, label) ** self.lam_parent_prob
                # Also incorporate p(child | pos) as an additional factor
                struct *= self.edge_prob_tables.get_prob_child_given_pos(node_pos, label) ** self.lam_pos_prob
                # Sibling factor: p(child | other_child) - this changes as we add edges!
                for other_lbl in node_outgoing_edges[parent_idx]:
                    struct *= self.edge_prob_tables.get_prob_child_given_sibling(other_lbl, label) ** self.lam_sibling_prob
                struct_factor = struct
            
            final_score = original_score * struct_factor
            edge["score_struct"] = float(final_score)
            edge["original_score"] = float(original_score)
            edge["struct_factor"] = float(struct_factor)
            return final_score

        validator = EdgeValidator(
            root_idx=root_idx,
            node_types=node_types,
            node_incoming_edges=node_incoming_edges,
            node_outgoing_edges=node_outgoing_edges,
            children_of=children_of,
            parents_of=parents_of,
            edge_labels_by_pair=edge_labels_by_pair,
            next_word_targets=next_word_targets,
            adjacent_only_types=ADJACENT_ONLY_TYPES,
            allowed_chain_at=ALLOWED_CHAIN_AT,
            would_create_cycle=would_create_cycle,
            pattern_check=lambda p_idx, lbl: self._is_edge_valid_for_pattern(
                p_idx,
                lbl,
                node_types[p_idx],
                node_outgoing_edges[p_idx],
                is_root=(p_idx == root_idx),
            ),
        )
        
        # Keep a list of remaining candidate edges
        remaining_candidates = candidate_edges.copy()
        
        # Greedily select edges: after each addition, recompute scores and reorder
        while remaining_candidates:
            # Recompute scores for all remaining candidates based on current graph state
            for edge in remaining_candidates:
                _compute_edge_score(edge)
            
            # Sort by score (descending)
            remaining_candidates.sort(key=lambda x: -x["score_struct"])
            
            # Find the highest scoring valid edge
            best_edge = None
            best_idx = -1
            for idx, edge in enumerate(remaining_candidates):
                parent_idx = edge["parent"]
                child_idx = edge["child"]
                label = edge["label"]
                
                code = validator.validate(
                    parent_idx, child_idx, label,
                    mode="phase2",
                    check_cycle=True,
                    check_pattern=True,
                )
                
                if code == EdgeValidationCode.OK:
                    best_edge = edge
                    best_idx = idx
                    break
                else:
                    # Log rejection for debugging
                    if code == EdgeValidationCode.NEXT_WORD_NOT_ADJACENT:
                        parent_type = node_types[parent_idx]
                        child_type = node_types[child_idx]
                        distance = abs(parent_idx - child_idx)
                        logger.debug(
                            f"Rejected non-adjacent next_word: parent[{parent_idx}]({parent_type}) "
                            f"-> child[{child_idx}]({child_type}), distance={distance}"
                        )
                    elif code == EdgeValidationCode.NEXT_WORD_CHILD_HAS_NON_NEXTWORD_INCOMING:
                        non_next_word_incoming = [e for e in node_incoming_edges[child_idx] if e != "next_word"]
                        logger.debug(f"Rejected next_word: child[{child_idx}] already has incoming edges: {non_next_word_incoming}")
                    elif code == EdgeValidationCode.NEXT_WORD_CHILD_HAS_NON_NEXTWORD_OUTGOING:
                        non_next_word_outgoing = [e for e in node_outgoing_edges[child_idx] if e != "next_word"]
                        logger.debug(f"Rejected next_word: child[{child_idx}] already has outgoing edges: {non_next_word_outgoing}")
                    elif code == EdgeValidationCode.CHILD_IS_NEXT_WORD_TARGET:
                        logger.debug(f"Rejected {label}: child[{child_idx}] is a next_word target, cannot receive other edge types")
                    elif code == EdgeValidationCode.PARENT_IS_NEXT_WORD_TARGET:
                        logger.debug(f"Rejected {label}: parent[{parent_idx}] is a next_word target, can only have next_word outgoing edges")
                    elif code == EdgeValidationCode.PATTERN_CONSTRAINT and label in node_outgoing_edges[parent_idx]:
                        logger.debug(f"Rejected duplicate edge: parent={parent_idx}, label={label}, existing={node_outgoing_edges[parent_idx]}")
            
            # If no valid edge found, break
            if best_edge is None:
                break
            
            # Debug: log candidate edge list grouped by node pairs
            if self.debug:
                # Group candidates by (parent_idx, child_idx) pairs
                candidates_by_pair = defaultdict(list)
                for edge in remaining_candidates:
                    pair = (edge["parent"], edge["child"])
                    candidates_by_pair[pair].append({
                        "label": edge["label"],
                        "arc_prob": edge["arc_prob"],
                        "rel_prob": edge["rel_prob"],
                        "original_score": edge.get("original_score", edge["arc_prob"] * edge["rel_prob"]),
                        "struct_factor": edge.get("struct_factor", 1.0),
                        "score_struct": edge.get("score_struct", edge["arc_prob"] * edge["rel_prob"]),
                    })
                
                # Sort pairs for consistent output
                sorted_pairs = sorted(candidates_by_pair.keys())
                
                logger.debug(f"\n{'='*80}")
                logger.debug(f"Phase 2: Candidate edges (before adding edge #{len(selected_edges) + 1})")
                logger.debug(f"{'='*80}")
                logger.debug(f"Total remaining candidates: {len(remaining_candidates)}")
                logger.debug(f"Grouped by {len(candidates_by_pair)} node pairs:\n")
                
                for (p_idx, c_idx) in sorted_pairs:
                    parent_word = words[p_idx] if p_idx < len(words) else f"node_{p_idx}"
                    child_word = words[c_idx] if c_idx < len(words) else f"node_{c_idx}"
                    logger.debug(f"  Pair ({p_idx}->{c_idx}): '{parent_word}' -> '{child_word}'")
                    
                    # Sort labels by score (descending)
                    labels_for_pair = sorted(candidates_by_pair[(p_idx, c_idx)], 
                                            key=lambda x: -x["score_struct"])
                    
                    for i, cand in enumerate(labels_for_pair):
                        marker = "  >>> SELECTED" if (cand["label"] == best_edge["label"] and 
                                                      p_idx == best_edge["parent"] and 
                                                      c_idx == best_edge["child"]) else ""
                        logger.debug(
                            f"    [{i+1}] label='{cand['label']:20s}' | "
                            f"arc={cand['arc_prob']:.4f} | rel={cand['rel_prob']:.4f} | "
                            f"orig={cand['original_score']:.4f} | struct={cand['struct_factor']:.4f} | "
                            f"final={cand['score_struct']:.4f}{marker}"
                        )
                    logger.debug("")
                
                logger.debug(f"{'='*80}\n")
            
            # Edge is valid - add it
            parent_idx = best_edge["parent"]
            child_idx = best_edge["child"]
            label = best_edge["label"]
            
            # Remove ALL candidate edges between (parent_idx -> child_idx) from remaining candidates
            # Once we've selected a label for this pair, we shouldn't consider other labels
            remaining_candidates = [
                e for e in remaining_candidates 
                if not (e["parent"] == parent_idx and e["child"] == child_idx)
            ]
            
            selected_edges.append(best_edge)
            node_outgoing_edges[parent_idx].append(label)
            node_incoming_edges[child_idx].append(label)
            children_of[parent_idx].add(child_idx)
            parents_of[child_idx].add(parent_idx)
            edge_labels_by_pair[(int(parent_idx), int(child_idx))] = str(label)
            
            # Track next_word targets
            if label == "next_word":
                next_word_targets.add(child_idx)
            
            # Add edge to graph immediately for debug visualization
            parent_node = word_to_node[parent_idx]
            child_node = word_to_node[child_idx]
            gpg_edge = GPGEdge(label=label)
            gpg_edge.arc_prob = best_edge["arc_prob"]
            gpg_edge.rel_prob = best_edge["rel_prob"]
            if "original_score" in best_edge:
                gpg_edge.original_score = best_edge["original_score"]
                gpg_edge.transformed_score = best_edge.get("score_struct", best_edge["original_score"])
                gpg_edge.struct_factor = best_edge.get("struct_factor", 1.0)
            graph.add_edge(parent_node, child_node, gpg_edge)
            
            # Debug: visualize after each edge addition in Phase 2
            if self.debug:
                step_desc = f"edge_{parent_idx}->{child_idx}_{label.replace('.', '_')}"
                self._debug_visualize_step(
                    graph, "phase2", step_desc,
                    arc_probs=arc_probs, rel_probs=rel_probs,
                    label2id={v: k for k, v in self.label2id.items()}
                )
            
            logger.debug(f"Added edge: parent={parent_idx}, child={child_idx}, label={label}, score={best_edge['score_struct']:.4f}")
        
        # Debug: visualize after Phase 2 completion
        if self.debug:
            self._debug_visualize_step(
                graph, "phase2", "complete",
                arc_probs=arc_probs, rel_probs=rel_probs,
                label2id={v: k for k, v in self.label2id.items()}
            )
        
        # Phase 3: Ensure the graph is weakly connected (one weakly-connected component).
        #
        # IMPORTANT: "weakly connected" is about connectivity in the *undirected* version of the graph.
        # The previous implementation mixed directed reachability with weak components, which could
        # incorrectly leave the final graph disconnected.
        #
        # This phase focuses on connectivity; multi-head issues are handled later (Phase 4).
        # Keep parents_of consistent (don't rebind: EdgeValidator holds a reference).
        for i in range(num_words):
            parents_of[i].clear()
        for p_idx, kids in children_of.items():
            for c_idx in kids:
                parents_of[int(c_idx)].add(int(p_idx))

        def weak_component(start_idx: int, allowed: set) -> set:
            """Return the weakly-connected component containing start_idx, restricted to `allowed`."""
            comp = set()
            stack = [start_idx]
            while stack:
                node = stack.pop()
                if node in comp or node not in allowed:
                    continue
                comp.add(node)
                # Outgoing neighbors
                for child in children_of[node]:
                    if child in allowed and child not in comp:
                        stack.append(child)
                # Incoming neighbors
                for parent in parents_of[node]:
                    if parent in allowed and parent not in comp:
                        stack.append(parent)
            return comp

        def _best_valid_label(
            p_idx: int,
            c_idx: int,
            *,
            mode: str,
            debug_trace: Optional[List[Dict[str, Any]]] = None,
            max_trace: int = 5,
        ) -> Optional[Tuple[str, float]]:
            """Return (label, rel_prob) for the best valid label under validator rules, else None.

            When debug_trace is provided, append up to `max_trace` attempted labels with validator codes.
            """
            # Note: in phase4_forced, validator intentionally relaxes next_word / pattern constraints.
            rel_vec = rel_probs[p_idx, c_idx]
            for lid in np.argsort(rel_vec)[::-1]:
                lbl = self.id2label.get(int(lid), "UNK")
                if lbl == "UNK" or lbl not in self.label2id:
                    continue
                rel_prob = float(rel_vec[lid])
                skip_reason = _hierarchy_hint_skip_reason(p_idx, c_idx, lbl, rel_prob=rel_prob)
                if skip_reason is not None:
                    if debug_trace is not None and len(debug_trace) < int(max_trace):
                        debug_trace.append({
                            "label": str(lbl),
                            "rel_prob": rel_prob,
                            "validator_code": f"LOW_HIERARCHY_{skip_reason.upper()}",
                        })
                    continue
                code = validator.validate(
                    p_idx, c_idx, lbl,
                    mode=mode,
                    check_cycle=True,
                    check_pattern=True,
                )
                if debug_trace is not None and len(debug_trace) < int(max_trace):
                    debug_trace.append({
                        "label": str(lbl),
                        "rel_prob": rel_prob,
                        "validator_code": str(code.name),
                    })
                if code == EdgeValidationCode.OK:
                    return lbl, rel_prob
            return None

        def _edge_score(p_idx: int, c_idx: int, lbl: str, arc_prob: float, rel_prob: float) -> float:
            score = float(arc_prob) * float(rel_prob)
            if self.edge_prob_tables is not None:
                incoming = _incoming_edge_label_for(p_idx)
                node_pos = node_types[p_idx]
                struct = self.edge_prob_tables.get_prob_child_given_parent(incoming, node_pos, lbl) ** self.lam_parent_prob
                struct *= self.edge_prob_tables.get_prob_child_given_pos(node_pos, lbl) ** self.lam_pos_prob
                for other_lbl in node_outgoing_edges[p_idx]:
                    struct *= self.edge_prob_tables.get_prob_child_given_sibling(other_lbl, lbl) ** self.lam_sibling_prob
                score *= struct
            return float(score)

        all_nodes = set(range(num_words))
        main_component = weak_component(root_idx, all_nodes)
        remaining = all_nodes - main_component
        if not self.enforce_connected:
            if remaining:
                decoding_info["phase3_skipped"] = {
                    "reason": "sdp_graph_mode",
                    "root_idx": int(root_idx),
                    "disconnected_nodes": sorted(int(x) for x in remaining),
                }
            remaining = set()

        import heapq
        while remaining:
            start_node = next(iter(remaining))
            component = weak_component(start_node, remaining)
            logger.debug(f"Phase 3: Found disconnected weak component with {len(component)} nodes: {sorted(component)}")

            best = None  # (p_idx, c_idx, label, mode_used, arc_prob, rel_prob, score)
            best_score = -float("inf")
            # Debug: keep a compact trace of rejected candidates so we can explain failures
            phase3_dbg = None
            fail_heap: List[Tuple[float, int, Dict[str, Any]]] = []
            fail_counter = 0
            if self.debug:
                phase3_dbg = {
                    "component_nodes": sorted([int(x) for x in component]),
                    "component_words": [words[i] for i in sorted(component) if i < len(words)],
                    "main_component_size": int(len(main_component)),
                    "attempted_pairs": 0,
                    "selected_edge": None,
                    "stop_message": None,
                    "top_failures": [],
                }

            def _consider_edge(p_idx: int, c_idx: int, *, mode: str, enforce_arc_threshold: bool):
                nonlocal best, best_score, fail_counter
                if phase3_dbg is not None:
                    phase3_dbg["attempted_pairs"] += 1
                arc_prob = float(arc_probs[p_idx, c_idx])
                if enforce_arc_threshold and arc_prob < float(self.arc_threshold):
                    return
                trace = [] if self.debug else None
                found = _best_valid_label(p_idx, c_idx, mode=mode, debug_trace=trace)
                if found is None:
                    # Debug: capture why labels were rejected for this high-scoring arc
                    if self.debug and trace:
                        # score this rejection by arc_prob * best_rel_prob attempted (first trace entry)
                        try:
                            fail_score = float(arc_prob) * float(trace[0].get("rel_prob", 0.0))
                        except Exception:
                            fail_score = float(arc_prob)
                        item = {
                            "parent": int(p_idx),
                            "child": int(c_idx),
                            "mode": str(mode),
                            "enforce_arc_threshold": bool(enforce_arc_threshold),
                            "arc_prob": float(arc_prob),
                            "label_attempts": trace,
                        }
                        # keep only top-N failures to avoid exploding debug JSON/logs
                        max_keep = 25
                        fail_counter += 1
                        if len(fail_heap) < max_keep:
                            heapq.heappush(fail_heap, (fail_score, fail_counter, item))
                        else:
                            if fail_score > fail_heap[0][0]:
                                heapq.heapreplace(fail_heap, (fail_score, fail_counter, item))
                    return
                lbl, rel_prob = found
                score = _edge_score(p_idx, c_idx, lbl, arc_prob, rel_prob)
                if score > best_score:
                    best_score = score
                    best = (p_idx, c_idx, lbl, mode, arc_prob, rel_prob, score)

            # Two-pass search (STRICT):
            # - Pass 1: strict Phase 3 constraints + arc_threshold
            # - Pass 2: strict Phase 3 constraints but allow arcs below threshold (connectivity-first)
            for mode, enforce_thr in (
                ("phase3", True),
                ("phase3", False),
            ):
                # Try both directions; for weak connectivity either direction connects the components.
                for u in main_component:
                    for v in component:
                        _consider_edge(u, v, mode=mode, enforce_arc_threshold=enforce_thr)
                        _consider_edge(v, u, mode=mode, enforce_arc_threshold=enforce_thr)
                if best is not None:
                    break

            if best is None:
                # STRICT mode: do not add an invalid edge. Record and stop Phase 3, leaving the graph disconnected.
                msg = (
                    "Phase 3: STOP (strict). No valid edge found to connect a disconnected component. "
                    f"main_component_size={len(main_component)} component_size={len(component)}"
                )
                logger.debug(msg)
                decoding_info["phase3_failures"].append({
                    "message": msg,
                    "component_nodes": list(component),
                    "component_words": [words[i] for i in component if i < len(words)],
                })
                if phase3_dbg is not None:
                    phase3_dbg["stop_message"] = msg
                    # sort failures by descending score
                    phase3_dbg["top_failures"] = [x for _, _counter, x in sorted(fail_heap, key=lambda t: -t[0])]
                    decoding_info["phase3_components"].append(phase3_dbg)
                break

            parent_idx, child_idx, label, mode_used, arc_prob, rel_prob, score = best

            # Defensive: validate the selected connection edge right before adding it (should already be OK).
            final_code = validator.validate(
                parent_idx, child_idx, label,
                mode=str(mode_used),
                check_cycle=True,
                check_pattern=True,
            )
            if final_code != EdgeValidationCode.OK:
                msg = (
                    "Phase 3: STOP (strict). Selected edge unexpectedly invalid at add-time. "
                    f"edge={parent_idx}->{child_idx} label='{label}' validator_code={final_code.name}"
                )
                logger.debug(msg)
                decoding_info["phase3_failures"].append({
                    "message": msg,
                    "validator_code": int(final_code),
                    "component_nodes": list(component),
                    "component_words": [words[i] for i in component if i < len(words)],
                })
                if phase3_dbg is not None:
                    phase3_dbg["stop_message"] = msg
                    phase3_dbg["top_failures"] = [x for _, _counter, x in sorted(fail_heap, key=lambda t: -t[0])]
                    decoding_info["phase3_components"].append(phase3_dbg)
                break

            # Add the connection edge
            parent_node = word_to_node[parent_idx]
            child_node = word_to_node[child_idx]
            gpg_edge = GPGEdge(label=label)
            gpg_edge.arc_prob = arc_prob
            gpg_edge.rel_prob = rel_prob
            gpg_edge.original_score = arc_prob * rel_prob
            gpg_edge.transformed_score = score
            graph.add_edge(parent_node, child_node, gpg_edge)
            node_outgoing_edges[parent_idx].append(label)
            node_incoming_edges[child_idx].append(label)
            children_of[parent_idx].add(child_idx)
            parents_of[child_idx].add(parent_idx)
            edge_labels_by_pair[(int(parent_idx), int(child_idx))] = str(label)

            # Track next_word targets
            if label == "next_word":
                next_word_targets.add(child_idx)

            logger.debug(
                f"Phase 3: Connected weak component via {parent_idx}->{child_idx} "
                f"label='{label}' (mode={mode_used}, score={score:.4f})"
            )
            if phase3_dbg is not None:
                phase3_dbg["selected_edge"] = {
                    "parent": int(parent_idx),
                    "child": int(child_idx),
                    "label": str(label),
                    "mode": str(mode_used),
                    "arc_prob": float(arc_prob),
                    "rel_prob": float(rel_prob),
                    "score": float(score),
                }
                phase3_dbg["top_failures"] = [x for _, _counter, x in sorted(fail_heap, key=lambda t: -t[0])]
                decoding_info["phase3_components"].append(phase3_dbg)

            # Debug: visualize after Phase 3 connection
            if self.debug:
                step_desc = f"connect_component_{parent_idx}->{child_idx}_{label.replace('.', '_')}"
                self._debug_visualize_step(
                    graph, "phase3", step_desc,
                    arc_probs=arc_probs, rel_probs=rel_probs,
                    label2id={v: k for k, v in self.label2id.items()}
                )

            # Update weak connectivity sets
            main_component |= component
            remaining -= component
        
        # Phase 4: Cope with multi-head problems (multiple roots / "heads")
        #
        # Rule (per user request):
        # - Let root_idx be the model-predicted root.
        # - Let R be the *decoded* root: a node with in-degree 0 that has the shortest directed path to root_idx.
        # - Let "heads" be nodes with in-degree 0 in the current decoded graph (excluding R).
        # - For each head H != R:
        #   - Find a common child/descendant C of H and R that minimizes:
        #       dist(H -> C) + dist(R -> C)
        #     (directed distances over children_of; tie-break deterministically).
        #   - If the path H -> ... -> C uses only edge components in:
        #       {func.arg, as:modification, as:punctuation, variable}
        #     then move the *parent edges of C* (incoming edges into C) to H,
        #     keeping the H-path parent of C unchanged.
        #
        # Interpretation of "move the parent edges of C to H":
        # For every incoming edge P -> C (label L) where P is NOT the predecessor of C on the chosen H->C path,
        # replace it by P -> H with the same label L (if valid and cycle-free). This makes H absorb the extra roots.
        # ALLOWED_PATH_EDGE_COMPONENTS = {"func.arg", "as:modification", "as:punctuation", 
        #     "variable", "body", "pred.arg.1", "pred.arg.2", "pred.arg.3", "pred.arg.4",
        #     "@modification", }
        ALLOWED_PATH_EDGE_COMPONENTS = set(self.label2id.keys())
        def _edge_components(lbl: str) -> set:
            return set(str(lbl).split("+"))

        def _descendants(start: int) -> set:
            seen = set()
            stack = [int(start)]
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                for v in children_of.get(u, set()):
                    if v not in seen:
                        stack.append(v)
            seen.discard(int(start))
            return seen

        def _bfs_tree(start: int) -> Tuple[Dict[int, int], Dict[int, int]]:
            """Return (dist, prev) for directed BFS over children_of."""
            from collections import deque

            s = int(start)
            dist: Dict[int, int] = {s: 0}
            prev: Dict[int, int] = {}
            q = deque([s])
            while q:
                u = q.popleft()
                for v in children_of.get(u, set()):
                    v = int(v)
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        q.append(v)
            return dist, prev

        def _reconstruct_path(prev: Dict[int, int], start: int, end: int) -> Optional[List[int]]:
            s = int(start)
            t = int(end)
            if t == s:
                return [s]
            if t not in prev:
                return None
            path = [t]
            while path[-1] != s:
                cur = path[-1]
                if cur not in prev:
                    return None
                path.append(prev[cur])
            path.reverse()
            return path

        def _remove_edge(p_idx: int, c_idx: int) -> Optional[str]:
            """Remove edge p_idx->c_idx from graph and tracking; return removed label if present."""
            p_idx = int(p_idx)
            c_idx = int(c_idx)
            parent_node = word_to_node[p_idx]
            child_node = word_to_node[c_idx]
            if not graph.g.has_edge(parent_node.ID, child_node.ID):
                return None
            try:
                old_edge = graph.get_edge(parent_node, child_node)
                old_label = str(old_edge.label)
            except Exception:
                old_label = None
            graph.remove_edge_between(parent_node, child_node)

            if old_label is not None:
                if old_label in node_outgoing_edges.get(p_idx, []):
                    node_outgoing_edges[p_idx].remove(old_label)
                if old_label in node_incoming_edges.get(c_idx, []):
                    node_incoming_edges[c_idx].remove(old_label)
                if old_label == "next_word":
                    if "next_word" not in node_incoming_edges.get(c_idx, []):
                        next_word_targets.discard(c_idx)
            children_of.get(p_idx, set()).discard(c_idx)
            parents_of.get(c_idx, set()).discard(p_idx)
            edge_labels_by_pair.pop((int(p_idx), int(c_idx)), None)

            # Remove from selected_edges if present (Phase 2 edges only).
            for e in selected_edges[:]:
                if int(e.get("parent", -1)) == p_idx and int(e.get("child", -1)) == c_idx:
                    selected_edges.remove(e)
                    break
            return old_label

        def _add_edge(
            p_idx: int,
            c_idx: int,
            label: str,
            *,
            context: str = "phase4",
            meta: Optional[Dict[str, Any]] = None,
            validation_mode: str = "phase3",
            check_pattern: bool = True,
            allow_topology_only: bool = False,
        ) -> bool:
            """Add edge p_idx->c_idx with given label if valid; update graph and tracking.

            In debug mode we also record failures (validator code / existing edge) for Phase 4 analysis.
            """
            p_idx = int(p_idx)
            c_idx = int(c_idx)
            parent_node = word_to_node[p_idx]
            child_node = word_to_node[c_idx]
            if graph.g.has_edge(parent_node.ID, child_node.ID):
                if self.debug and context != "phase4_rollback":
                    decoding_info["phase4_add_edge_failures"].append({
                        "parent": int(p_idx),
                        "child": int(c_idx),
                        "label": str(label),
                        "reason": "edge already exists",
                        "validator_code": None,
                        "context": str(context),
                        "meta": meta,
                    })
                return False

            arc_prob, rel_prob, _confidence = _edge_model_probs(p_idx, c_idx, label)
            skip_reason = (
                None
                if context == "phase4_rollback"
                else _hierarchy_hint_skip_reason(p_idx, c_idx, label, arc_prob=arc_prob, rel_prob=rel_prob)
            )
            if skip_reason is not None:
                if self.debug:
                    decoding_info["phase4_add_edge_failures"].append({
                        "parent": int(p_idx),
                        "child": int(c_idx),
                        "label": str(label),
                        "reason": f"hierarchy hint {skip_reason}",
                        "validator_code": None,
                        "context": str(context),
                        "arc_prob": float(arc_prob),
                        "rel_prob": float(rel_prob),
                        "meta": meta,
                    })
                decoding_info.setdefault("skipped_hierarchy_edges", []).append({
                    "from": parent_node.ID,
                    "to": child_node.ID,
                    "from_word_idx": int(p_idx),
                    "to_word_idx": int(c_idx),
                    "label": str(label),
                    "source": str(context),
                    "reason": skip_reason,
                    "arc_prob": float(arc_prob),
                    "rel_prob": float(rel_prob),
                    "confidence": float(arc_prob * rel_prob),
                    "meta": meta or {},
                })
                return False

            # Keep frame-arity constraints hard even when a final root repair is
            # allowed to relax softer constraints. In particular, a predicator
            # must not get duplicate pred.arg.x / variable / body edges.
            code = validator.validate(
                p_idx, c_idx, label,
                mode=validation_mode,
                check_cycle=True,
                check_pattern=check_pattern,
            )
            if code == EdgeValidationCode.DUPLICATED_EDGE_LABELS:
                if self.debug and context != "phase4_rollback":
                    decoding_info["phase4_add_edge_failures"].append({
                        "parent": int(p_idx),
                        "child": int(c_idx),
                        "label": str(label),
                        "reason": "hard frame-arity constraint rejected",
                        "validator_code": str(code.name),
                        "context": str(context),
                        "meta": meta,
                    })
                return False
            if code != EdgeValidationCode.OK and not allow_topology_only:
                if self.debug and context != "phase4_rollback":
                    decoding_info["phase4_add_edge_failures"].append({
                        "parent": int(p_idx),
                        "child": int(c_idx),
                        "label": str(label),
                        "reason": "validator rejected",
                        "validator_code": str(code.name),
                        "context": str(context),
                        "meta": meta,
                    })
                return False
            if code != EdgeValidationCode.OK and allow_topology_only and would_create_cycle(p_idx, c_idx):
                if self.debug and context != "phase4_rollback":
                    decoding_info["phase4_add_edge_failures"].append({
                        "parent": int(p_idx),
                        "child": int(c_idx),
                        "label": str(label),
                        "reason": "topology-only repair would create cycle",
                        "validator_code": str(code.name),
                        "context": str(context),
                        "meta": meta,
                    })
                return False

            score = _edge_score(p_idx, c_idx, label, arc_prob, rel_prob)

            gpg_edge = GPGEdge(label=label)
            gpg_edge.arc_prob = arc_prob
            gpg_edge.rel_prob = rel_prob
            gpg_edge.original_score = arc_prob * rel_prob
            gpg_edge.transformed_score = score
            graph.add_edge(parent_node, child_node, gpg_edge)

            node_outgoing_edges[p_idx].append(label)
            node_incoming_edges[c_idx].append(label)
            children_of[p_idx].add(c_idx)
            parents_of[c_idx].add(p_idx)
            edge_labels_by_pair[(int(p_idx), int(c_idx))] = str(label)
            if label == "next_word":
                next_word_targets.add(c_idx)

            # Keep selected_edges in sync for any later debug use.
            selected_edges.append({
                "parent": p_idx,
                "child": c_idx,
                "label": label,
                "arc_prob": arc_prob,
                "rel_prob": rel_prob,
                "score_struct": score,
            })
            return True

        def _add_hierarchy_hint_edge(
            p_idx: int,
            c_idx: int,
            label: str,
            *,
            source: str,
            case: Optional[int] = None,
            meta: Optional[Dict[str, Any]] = None,
        ) -> Optional[Dict[str, Any]]:
            """Add @upper/@lower only when the model also supports the edge."""
            p_idx = int(p_idx)
            c_idx = int(c_idx)
            label = str(label)
            if label not in self.HIERARCHY_HINT_EDGES or label not in self.label2id:
                return None
            if graph.g.has_edge(word_to_node[p_idx].ID, word_to_node[c_idx].ID):
                return None
            arc_prob, rel_prob, confidence = _edge_model_probs(p_idx, c_idx, label)
            skip_reason = _hierarchy_hint_skip_reason(p_idx, c_idx, label, arc_prob=arc_prob, rel_prob=rel_prob)
            if skip_reason is None and would_create_cycle(p_idx, c_idx):
                skip_reason = "would_create_cycle"

            info = {
                "from": word_to_node[p_idx].ID,
                "to": word_to_node[c_idx].ID,
                "from_word_idx": int(p_idx),
                "to_word_idx": int(c_idx),
                "label": label,
                "source": source,
                "case": case,
                "arc_prob": float(arc_prob),
                "rel_prob": float(rel_prob),
                "confidence": float(confidence),
                "meta": meta or {},
            }
            if skip_reason is not None:
                info["reason"] = skip_reason
                decoding_info.setdefault("skipped_hierarchy_edges", []).append(info)
                return None

            edge = GPGEdge(label=label)
            edge.arc_prob = float(arc_prob)
            edge.rel_prob = float(rel_prob)
            edge.original_score = float(confidence)
            edge.transformed_score = float(confidence)
            edge.decoder_source = source
            edge.postprocess = True
            edge.postprocess_case = case
            graph.add_edge(word_to_node[p_idx], word_to_node[c_idx], edge)

            node_outgoing_edges[p_idx].append(label)
            node_incoming_edges[c_idx].append(label)
            children_of[p_idx].add(c_idx)
            parents_of[c_idx].add(p_idx)
            edge_labels_by_pair[(int(p_idx), int(c_idx))] = label
            return info

        def _current_roots() -> List[int]:
            return [
                i
                for i in range(num_words)
                if graph.g.in_degree(word_to_node[i].ID) == 0
            ]

        def _root_reachable_nodes(root: int) -> set:
            reachable, _ = _bfs_tree(int(root))
            return set(reachable.keys())

        def _best_label_for_pair(
            p_idx: int,
            c_idx: int,
            *,
            mode: str,
            check_pattern_for_mode: bool,
            topology_only: bool = False,
        ) -> Optional[Tuple[str, float, str]]:
            rel_vec = rel_probs[p_idx, c_idx]
            for lid in np.argsort(rel_vec)[::-1]:
                lbl = self.id2label.get(int(lid), "UNK")
                if lbl == "UNK" or lbl == "nonsense" or lbl not in self.label2id:
                    continue
                rel_prob = float(rel_vec[lid])
                if _hierarchy_hint_skip_reason(p_idx, c_idx, lbl, rel_prob=rel_prob) is not None:
                    continue
                code = validator.validate(
                    p_idx,
                    c_idx,
                    lbl,
                    mode=mode,
                    check_cycle=True,
                    check_pattern=check_pattern_for_mode,
                )
                if code == EdgeValidationCode.OK:
                    return lbl, rel_prob, "valid"
                if code == EdgeValidationCode.DUPLICATED_EDGE_LABELS:
                    continue
                if topology_only and not would_create_cycle(p_idx, c_idx):
                    return lbl, rel_prob, f"topology_only:{code.name}"
            return None

        def _choose_primary_decoded_root(roots: List[int]) -> int:
            if not roots:
                return int(root_idx)
            if int(phase4_root_idx) in roots:
                return int(phase4_root_idx)
            if int(root_idx) in roots:
                return int(root_idx)
            dist_to_model_root = _dist_to_pred_root()
            reachable_roots = [int(r) for r in roots if int(r) in dist_to_model_root]
            if reachable_roots:
                return int(min(reachable_roots, key=lambda r: (dist_to_model_root[int(r)], int(r))))
            return int(min(roots))

        def _force_single_root() -> None:
            """Attach remaining decoded roots with the best model-scored edge."""
            max_rounds = max(1, num_words)
            for _ in range(max_rounds):
                roots = _current_roots()
                if len(roots) <= 1:
                    return

                primary = _choose_primary_decoded_root(roots)
                anchored_nodes = _root_reachable_nodes(primary)
                if not anchored_nodes:
                    anchored_nodes = {primary}

                changed = False
                for head_idx in sorted(int(r) for r in roots if int(r) != int(primary)):
                    best = None
                    best_score = -float("inf")

                    # Prefer attaching from the primary root branch, then fall back
                    # to any non-head node. The child is fixed to the extra root so
                    # root count monotonically decreases when an edge is added.
                    parent_groups = [
                        sorted(int(p) for p in anchored_nodes if int(p) != head_idx),
                        sorted(int(p) for p in range(num_words) if int(p) != head_idx),
                    ]
                    seen_parents = set()
                    for parents in parent_groups:
                        for p_idx in parents:
                            if p_idx in seen_parents:
                                continue
                            seen_parents.add(p_idx)
                            if graph.g.has_edge(word_to_node[p_idx].ID, word_to_node[head_idx].ID):
                                continue

                            for mode, check_pattern_for_mode, topology_only in (
                                ("phase3", True, False),
                                ("phase4_forced", False, False),
                                ("phase4_forced", False, True),
                            ):
                                found = _best_label_for_pair(
                                    p_idx,
                                    head_idx,
                                    mode=mode,
                                    check_pattern_for_mode=check_pattern_for_mode,
                                    topology_only=topology_only,
                                )
                                if found is None:
                                    continue
                                lbl, rel_prob, repair_mode = found
                                arc_prob = float(arc_probs[p_idx, head_idx])
                                score = _edge_score(p_idx, head_idx, lbl, arc_prob, rel_prob)
                                if score > best_score:
                                    best_score = score
                                    best = (
                                        p_idx,
                                        head_idx,
                                        lbl,
                                        mode,
                                        check_pattern_for_mode,
                                        topology_only,
                                        repair_mode,
                                        arc_prob,
                                        rel_prob,
                                        score,
                                    )
                                break

                    if best is None:
                        decoding_info["errors"].append(
                            f"Unable to attach extra root {head_idx} ({words[head_idx] if head_idx < len(words) else head_idx})"
                        )
                        continue

                    (
                        p_idx,
                        c_idx,
                        lbl,
                        mode,
                        check_pattern_for_mode,
                        topology_only,
                        repair_mode,
                        arc_prob,
                        rel_prob,
                        score,
                    ) = best
                    added = _add_edge(
                        p_idx,
                        c_idx,
                        lbl,
                        context="single_root_repair",
                        validation_mode=mode,
                        check_pattern=check_pattern_for_mode,
                        allow_topology_only=topology_only,
                        meta={
                            "primary_root": int(primary),
                            "head": int(head_idx),
                            "repair_mode": repair_mode,
                        },
                    )
                    if not added:
                        decoding_info["errors"].append(
                            f"Unable to add selected single-root repair {p_idx}->{c_idx} label='{lbl}'"
                        )
                        continue

                    changed = True
                    decoding_info.setdefault("single_root_repairs", []).append({
                        "parent": int(p_idx),
                        "child": int(c_idx),
                        "label": str(lbl),
                        "primary_root": int(primary),
                        "mode": str(mode),
                        "repair_mode": str(repair_mode),
                        "arc_prob": float(arc_prob),
                        "rel_prob": float(rel_prob),
                        "score": float(score),
                    })
                    anchored_nodes |= _root_reachable_nodes(primary)

                if not changed:
                    return

        # Choose R for Phase 4:
        # among decoded roots (in-degree 0), pick the one with the shortest directed path to model root_idx.
        # Efficiently compute dist(u -> root_idx) by BFS on the reverse graph (parents_of) from root_idx.
        predicted_root_idx = int(root_idx)

        def _dist_to_pred_root() -> Dict[int, int]:
            from collections import deque

            dist: Dict[int, int] = {predicted_root_idx: 0}
            q = deque([predicted_root_idx])
            while q:
                u = q.popleft()
                for p in parents_of.get(u, set()):
                    p = int(p)
                    if p not in dist:
                        dist[p] = dist[u] + 1
                        q.append(p)
            return dist

        dist_to_pred = _dist_to_pred_root()

        decoded_roots_all = [i for i in range(num_words) if graph.g.in_degree(word_to_node[i].ID) == 0]
        reachable_roots = [r for r in decoded_roots_all if int(r) in dist_to_pred]
        if reachable_roots:
            phase4_root_idx = int(min(reachable_roots, key=lambda r: (dist_to_pred[int(r)], int(r))))
        else:
            # Fallback: if no decoded root reaches predicted_root_idx, keep using predicted_root_idx as R.
            phase4_root_idx = predicted_root_idx

        # Detect heads (roots in the decoded graph) excluding R.
        max_rounds = 3 if self.enable_phase4_rewrites else 0
        round_idx = 0
        while round_idx < max_rounds:
            round_idx += 1
            decoded_roots = [i for i in range(num_words) if graph.g.in_degree(word_to_node[i].ID) == 0]
            extra_heads = [h for h in decoded_roots if int(h) != int(phase4_root_idx)]
            if not extra_heads:
                break

            any_change = False
            # Iterate in a stable order for reproducibility
            for head_idx in sorted(extra_heads):
                head_idx = int(head_idx)
                # Recompute root reachability each time because previous rewires can change it.
                dist_r, _prev_r_unused = _bfs_tree(phase4_root_idx)
                dist_h, prev_h = _bfs_tree(head_idx)
                # Common child/descendant: reachable from both R and H (directed),
                # excluding the endpoints themselves.
                common = [
                    n for n in dist_h.keys()
                    if (n in dist_r) and (n != head_idx) and (n != int(phase4_root_idx))
                ]
                if not common:
                    continue

                # Prefer common child C minimizing dist(H->C)+dist(R->C); tie-break by more balanced paths,
                # then by individual distances, then by node index for determinism.
                def _common_score(n: int) -> Tuple[int, int, int, int, int]:
                    dh = int(dist_h.get(n, 10**9))
                    dr = int(dist_r.get(n, 10**9))
                    return (dh + dr, dr, dh)

                c_idx = int(min(common, key=_common_score))
                logger.debug(
                    f"Phase 4: "
                    f"Root {phase4_root_idx} and Head {head_idx} has common child {c_idx} "
                    f"(dist_h={dist_h.get(c_idx)}, dist_r={dist_r.get(c_idx)}, sum={_common_score(c_idx)[0]})"
                )
                path_nodes = _reconstruct_path(prev_h, head_idx, c_idx)
                if not path_nodes or len(path_nodes) < 2:
                    continue

                # Check path-edge labels constraint
                ok_path = True
                path_edges: List[Tuple[int, int, str]] = []
                for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                    try:
                        lbl = str(graph.get_edge(word_to_node[u], word_to_node[v]).label)
                    except Exception:
                        ok_path = False
                        break
                    comps = _edge_components(lbl)
                    if not comps.issubset(ALLOWED_PATH_EDGE_COMPONENTS):
                        ok_path = False
                        break
                    path_edges.append((u, v, lbl))

                if not ok_path:
                    continue

                # Keep the predecessor-on-path -> C incoming edge unchanged
                pred_of_c = int(path_nodes[-2])
                parents_c = set(parents_of.get(c_idx, set()))
                movable_parents = [p for p in parents_c if int(p) != pred_of_c]
                if not movable_parents:
                    continue

                moved = []
                for p_idx in sorted(movable_parents):
                    p_idx = int(p_idx)
                    if not graph.g.has_edge(word_to_node[p_idx].ID, word_to_node[c_idx].ID):
                        continue
                    try:
                        old_lbl = str(graph.get_edge(word_to_node[p_idx], word_to_node[c_idx]).label)
                    except Exception:
                        continue

                    # This is a *move* of an existing edge label from (P->C) to (P->H).
                    # Validate AFTER removing P->C, otherwise pattern multiplicity checks may reject
                    # even though the outgoing-label count for P does not increase.
                    removed_lbl = _remove_edge(p_idx, c_idx)
                    if removed_lbl is None:
                        continue

                    # When moving an @ edge to H, H's existing @ children may need
                    # @lower hints from the moved parent. Collect these first, and
                    # add them only after the move succeeds so failed rewrites do
                    # not leak stray @lower edges.
                    pending_lower_children: List[Tuple[int, str]] = []
                    if "@" in old_lbl:
                        for head_child in set(children_of.get(head_idx, set())):
                            head_child_node = word_to_node[head_child]
                            edge = graph.get_edge(word_to_node[head_idx], head_child_node)
                            if edge is None:
                                continue
                            edge_label = str(edge.label)
                            if edge_label.startswith("@") and edge_label in self.PRIMARY_AT_EDGES:
                                pending_lower_children.append((int(head_child), edge_label))

                    if not _add_edge(
                        p_idx,
                        head_idx,
                        old_lbl,
                        context="phase4_move",
                        meta={
                            "old_child": int(c_idx),
                            "head": int(head_idx),
                            "common_child": int(c_idx),
                            "phase4_root": int(phase4_root_idx),
                            "predicted_root": int(predicted_root_idx),
                        },
                    ):
                        # rollback: restore original edge if we couldn't add new one
                        _add_edge(p_idx, c_idx, old_lbl, context="phase4_rollback")
                        continue

                    added_lower_edges = []
                    for head_child, edge_label in pending_lower_children:
                        added_lower = _add_hierarchy_hint_edge(
                            p_idx,
                            head_child,
                            "@lower",
                            source="phase4_multihead",
                            meta={
                                "head": int(head_idx),
                                "head_child_label": edge_label,
                                "moved_label": old_lbl,
                                "old_child": int(c_idx),
                                "common_child": int(c_idx),
                                "phase4_root": int(phase4_root_idx),
                                "predicted_root": int(predicted_root_idx),
                            },
                        )
                        if added_lower is not None:
                            added_lower_edges.append(added_lower)
                            logger.debug(
                                f"Phase 4: Added @lower edge {p_idx}->{head_child} "
                                f"because parent {p_idx} moved @ edge to head {head_idx} "
                                f"which already has @ child {head_child}"
                            )

                    moved.append({
                        "from_parent": p_idx,
                        "old_child": c_idx,
                        "new_child": head_idx,
                        "label": old_lbl,
                        "added_lower_edges": added_lower_edges,
                    })

                if moved:
                    any_change = True
                    info = {
                        "predicted_root": int(predicted_root_idx),
                        "phase4_root": int(phase4_root_idx),
                        "head": int(head_idx),
                        "common_child": int(c_idx),
                        "path": [int(x) for x in path_nodes],
                        "path_edges": [{"parent": int(u), "child": int(v), "label": lbl} for (u, v, lbl) in path_edges],
                        "moved_parent_edges": moved,
                    }
                    decoding_info["phase4_multihead_fixes"].append(info)
                    logger.debug(
                        f"Phase 4: Rewired {len(moved)} parent edges of common child {c_idx} to head {head_idx} "
                        f"(phase4_root={phase4_root_idx}, predicted_root={predicted_root_idx})."
                    )

                    if self.debug:
                        step_desc = f"multihead_fix_head_{head_idx}_via_{c_idx}"
                        self._debug_visualize_step(
                            graph, "phase4", step_desc,
                            arc_probs=arc_probs, rel_probs=rel_probs,
                            label2id={v: k for k, v in self.label2id.items()}
                        )
            if not any_change:
                break

        # Phase 4b is disabled by default in SDP graph mode. Forcing a single
        # root is tree-decoder behavior and can replace high-confidence semantic
        # edges with low-confidence attachments.
        if self.enforce_single_root:
            _force_single_root()
        else:
            roots = _current_roots()
            if len(roots) > 1:
                decoding_info["single_root_repair_skipped"] = {
                    "reason": "sdp_graph_mode",
                    "roots": [int(r) for r in roots],
                }
        
        # if unreachable:
        #     logger.debug(f"Phase 4: Connecting {len(unreachable)} disconnected nodes")
            
        #     # Find connected components among unreachable nodes
        #     def find_component(start_idx, nodes_set):
        #         """Find all nodes in the same component as start_idx (following edges in both directions)."""
        #         component = set()
        #         stack = [start_idx]
        #         while stack:
        #             node = stack.pop()
        #             if node in component or node not in nodes_set:
        #                 continue
        #             component.add(node)
        #             # Follow outgoing edges
        #             for child in children_of[node]:
        #                 if child in nodes_set:
        #                     stack.append(child)
        #             # Follow incoming edges (find parents)
        #             for potential_parent in nodes_set:
        #                 if node in children_of[potential_parent]:
        #                     stack.append(potential_parent)
        #         return component
            
        #     # Find local roots (nodes with no incoming edges within the component)
        #     def find_local_roots(component):
        #         """Find nodes in component that have no incoming edges from within the component."""
        #         local_roots = []
        #         for node in component:
        #             has_incoming_from_component = False
        #             for potential_parent in component:
        #                 if node in children_of[potential_parent]:
        #                     has_incoming_from_component = True
        #                     break
        #             if not has_incoming_from_component:
        #                 local_roots.append(node)
        #         return local_roots if local_roots else list(component)  # Fallback to any node if cycle
            
        #     # Process disconnected components
        #     remaining_unreachable = set(unreachable)
        #     while remaining_unreachable:
        #         # Pick a node and find its component
        #         start_node = next(iter(remaining_unreachable))
        #         component = find_component(start_node, remaining_unreachable)
                
        #         # Find local roots of this component
        #         local_roots = find_local_roots(component)
                
        #         # Try to connect any local root to the root branch
        #         best_connection = None
        #         best_score = -float('inf')
                
        #         for local_root in local_roots:
        #             for parent_idx in reachable:
        #                 # Check cycle constraint
        #                 if would_create_cycle(parent_idx, local_root):
        #                     continue
                        
        #                 arc_prob = float(arc_probs[parent_idx, local_root])
                        
        #                 # Find best valid label
        #                 for lid in np.argsort(rel_probs[parent_idx, local_root])[::-1]:
        #                     lbl = self.id2label.get(lid, "UNK")
        #                     if is_edge_valid_phase3(parent_idx, local_root, lbl):
        #                         rel_prob = float(rel_probs[parent_idx, local_root, lid])
        #                         score = arc_prob * rel_prob
        #                         if self.edge_prob_tables is not None:
        #                             incoming = _incoming_edge_label_for(parent_idx)
        #                             node_pos = node_types[parent_idx]
        #                             # Use p(child | parent, pos) when we have parent edge information
        #                             struct = self.edge_prob_tables.get_prob_child_given_parent(incoming, node_pos, lbl) ** self.lam_parent_prob
        #                             # Also incorporate p(child | pos) as an additional factor
        #                             struct *= self.edge_prob_tables.get_prob_child_given_pos(node_pos, lbl) ** self.lam_pos_prob
        #                             # Sibling factor: p(child | other_child)
        #                             for other_lbl in node_outgoing_edges[parent_idx]:
        #                                 struct *= self.edge_prob_tables.get_prob_child_given_sibling(other_lbl, lbl) ** self.lam_sibling_prob
        #                             score *= struct
        #                         if score > best_score:
        #                             best_score = score
        #                             best_connection = (parent_idx, local_root, lbl, arc_prob, rel_prob)
        #                         break  # Found best label for this parent-child pair
                
        #         if best_connection:
        #             parent_idx, child_idx, label, arc_prob, rel_prob = best_connection
        #             parent_node = word_to_node[parent_idx]
        #             child_node = word_to_node[child_idx]
        #             gpg_edge = GPGEdge(label=label)
        #             gpg_edge.arc_prob = arc_prob
        #             gpg_edge.rel_prob = rel_prob
        #             graph.add_edge(parent_node, child_node, gpg_edge)
                    
        #             node_outgoing_edges[parent_idx].append(label)
        #             node_incoming_edges[child_idx].append(label)
        #             children_of[parent_idx].add(child_idx)
                    
        #             if label == "next_word":
        #                 next_word_targets.add(child_idx)
                    
        #             logger.debug(f"Phase 4: Connected component root {child_idx} to {parent_idx} with '{label}' (score={best_score:.4f})")
                    
        #             # Debug: visualize after Phase 4 connection
        #             if self.debug:
        #                 step_desc = f"connect_{parent_idx}->{child_idx}_{label.replace('.', '_')}"
        #                 self._debug_visualize_step(
        #                     graph, "phase4", step_desc,
        #                     arc_probs=arc_probs, rel_probs=rel_probs,
        #                     label2id={v: k for k, v in self.label2id.items()}
        #                 )
                    
        #             # Update reachable set with this component
        #             newly_reachable = get_reachable_from(child_idx)
        #             reachable.update(newly_reachable)
        #         else:
        #             # Force connect with any edge (ignoring constraints) as last resort
        #             msg = f"Phase 4: Force connecting component with {len(component)} nodes (no valid edge found)"
        #             logger.warning(msg)
        #             decoding_info["warnings"].append(msg)
                    
        #             # Find any edge with highest score (base arc*rel, optionally re-weighted by structural probabilities)
        #             best_force = None
        #             best_force_score = -1.0
        #             for local_root in local_roots:
        #                 for parent_idx in reachable:
        #                     if would_create_cycle(parent_idx, local_root):
        #                         continue
        #                     arc_prob = float(arc_probs[parent_idx, local_root])

        #                     # Even in forced mode, keep the "@-edge uniqueness" constraint:
        #                     # each node can have at most one outgoing label starting with "@"
        #                     # Also keep: nodes with incoming next_word cannot have outgoing edge containing "@"
        #                     label = None
        #                     rel_prob = 0.0
        #                     for lid in np.argsort(rel_probs[parent_idx, local_root])[::-1]:
        #                         cand_label = self.id2label.get(int(lid), "UNK")
        #                         if validator.validate(
        #                             parent_idx,
        #                             local_root,
        #                             cand_label,
        #                             mode="phase4_forced",
        #                             check_cycle=False,   # already checked above
        #                             check_pattern=False, # forced mode ignores pattern constraints
        #                         ) != EdgeValidationCode.OK:
        #                             continue
        #                         label = cand_label
        #                         rel_prob = float(rel_probs[parent_idx, local_root, int(lid)])
        #                         break
        #                     if label is None:
        #                         continue

        #                     score = arc_prob * rel_prob
        #                     if self.edge_prob_tables is not None:
        #                         incoming = _incoming_edge_label_for(parent_idx)
        #                         node_pos = node_types[parent_idx]
        #                         # Use p(child | parent, pos) when we have parent edge information
        #                         struct = self.edge_prob_tables.get_prob_child_given_parent(incoming, node_pos, label) ** self.lam_parent_prob
        #                         # Also incorporate p(child | pos) as an additional factor
        #                         struct *= self.edge_prob_tables.get_prob_child_given_pos(node_pos, label) ** self.lam_pos_prob
        #                         # Sibling factor: p(child | other_child)
        #                         for other_lbl in node_outgoing_edges[parent_idx]:
        #                             struct *= self.edge_prob_tables.get_prob_child_given_sibling(other_lbl, label) ** self.lam_sibling_prob
        #                         score *= struct
        #                     if score > best_force_score:
        #                         best_force_score = score
        #                         best_force = (parent_idx, local_root, label, arc_prob, rel_prob)
                    
        #             if best_force:
        #                 parent_idx, child_idx, label, arc_prob, rel_prob = best_force
        #                 parent_node = word_to_node[parent_idx]
        #                 child_node = word_to_node[child_idx]
        #                 gpg_edge = GPGEdge(label=label)
        #                 gpg_edge.arc_prob = arc_prob
        #                 gpg_edge.rel_prob = rel_prob
        #                 graph.add_edge(parent_node, child_node, gpg_edge)
                        
        #                 node_outgoing_edges[parent_idx].append(label)
        #                 node_incoming_edges[child_idx].append(label)
        #                 children_of[parent_idx].add(child_idx)
                        
        #                 if label == "next_word":
        #                     next_word_targets.add(child_idx)
                        
        #                 msg = f"Phase 4 (forced): Connected {child_idx} to {parent_idx} with '{label}'"
        #                 logger.debug(msg)
        #                 decoding_info["phase4_force_connects"].append({
        #                     "parent_idx": parent_idx,
        #                     "child_idx": child_idx,
        #                     "parent_word": words[parent_idx] if parent_idx < len(words) else "?",
        #                     "child_word": words[child_idx] if child_idx < len(words) else "?",
        #                     "label": label,
        #                     "arc_prob": arc_prob,
        #                     "rel_prob": rel_prob,
        #                 })
                        
        #                 # Debug: visualize after Phase 4 forced connection
        #                 if self.debug:
        #                     step_desc = f"forced_{parent_idx}->{child_idx}_{label.replace('.', '_')}"
        #                     self._debug_visualize_step(
        #                         graph, "phase4", step_desc,
        #                         arc_probs=arc_probs, rel_probs=rel_probs,
        #                         label2id={v: k for k, v in self.label2id.items()}
        #                     )
                        
        #                 # Update reachable
        #                 newly_reachable = get_reachable_from(child_idx)
        #                 reachable.update(newly_reachable)
        #             else:
        #                 msg = f"Phase 4: Cannot connect component {component} - all edges create cycles"
        #                 logger.error(msg)
        #                 decoding_info["errors"].append(msg)
        #                 decoding_info["disconnected_nodes"].extend(list(component))
                
        #         # Remove processed component from remaining
        #         remaining_unreachable -= component

        # Phase 5: Add @upper and @lower edges as post-processing
        # These edges establish hierarchy relationships between @ children
        self._add_upper_lower_edges(
            graph,
            word_to_node,
            node_outgoing_edges,
            decoding_info,
            add_hierarchy_hint_edge=_add_hierarchy_hint_edge,
        )
        
        # Debug: visualize after Phase 5
        if self.debug:
            self._debug_visualize_step(
                graph, "phase5", "add_upper_lower",
                arc_probs=arc_probs, rel_probs=rel_probs,
                label2id={v: k for k, v in self.label2id.items()}
            )

        # Debug: visualize final word-lingua graph
        if self.debug:
            self._debug_visualize_step(
                graph, "final", "word_lingua",
                arc_probs=arc_probs, rel_probs=rel_probs,
                label2id={v: k for k, v in self.label2id.items()}
            )
        
        # Check if there are any decoding issues
        if (
            decoding_info["errors"]
            or decoding_info["warnings"]
            or decoding_info["phase4_force_connects"]
            or decoding_info.get("phase3_failures")
            or decoding_info.get("phase4_add_edge_failures")
            or decoding_info.get("single_root_repairs")
        ):
            decoding_info["has_issues"] = True
        else:
            decoding_info["has_issues"] = False

        # Debug: emit detailed Phase 3/4 report for single-sample debugging
        if self.debug:
            self._debug_log_phase34_report(words, decoding_info)

        # Validate the word-lingua graph
        validation_errors = self._validate_word_lingua_graph(graph, node_types, decoding_info)
        if validation_errors:
            decoding_info["validation_errors"] = validation_errors
            decoding_info["has_issues"] = True

        return graph, decoding_info

    def _decode_ilp_single_root_dag(
        self,
        *,
        words: List[str],
        graph: GPGraph,
        word_to_node: Dict[int, Any],
        node_types: List[str],
        arc_probs: np.ndarray,
        rel_probs: np.ndarray,
        root_idx: int,
        is_root_logits: Optional[np.ndarray],
        decoding_info: Dict[str, Any],
    ) -> Tuple[GPGraph, Dict[str, Any]]:
        """
        Select a maximum-scoring legal word-lingua graph with a MILP backend.

        The binary edge variables choose labeled edges. Root variables enforce
        exactly one zero-indegree node, and continuous order variables enforce a
        DAG through topological-order inequalities.
        """
        if not words:
            decoding_info["ilp"] = {
                "candidate_edges": 0,
                "raw_candidate_edges": 0,
                "constraints": 0,
                "selected_edges": 0,
                "selected_root": None,
                "status": None,
                "success": True,
                "message": "empty graph",
            }
            decoding_info["has_issues"] = False
            return graph, decoding_info

        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import coo_matrix
        except Exception as e:
            decoding_info["errors"].append(f"ILP decoder requires scipy.optimize.milp: {e}")
            decoding_info["has_issues"] = True
            return graph, decoding_info

        num_words = len(words)
        eps = 1e-9
        frame_components = {"pred.arg.1", "pred.arg.2", "pred.arg.3", "pred.arg.4", "variable", "body"}
        hierarchy_hint_at = {"@lower", "@upper"}
        primary_aux_at = set(self.PRIMARY_AT_EDGES) | {"@whether_of"}
        adjacent_only_types = {
            "ModificationalConstant", "DeterminerConstant", "OtherConstant",
            "NominalConstant", "SymbolConstant",
        }
        allowed_chain_at = {"@list", "@parataxis", "@expression"}

        def _safe_log(x: float) -> float:
            return float(np.log(max(float(x), eps)))

        def _edge_components(label: str) -> Set[str]:
            return set(str(label).split("+"))

        def _is_punctuation_label(label: str) -> bool:
            comps = _edge_components(label)
            return "punctuation" in comps

        def _is_modification_label(label: str) -> bool:
            return str(label).startswith("@modification")

        def _edge_model_probs(p_idx: int, c_idx: int, label: str) -> Tuple[float, float, float]:
            p_idx = int(p_idx)
            c_idx = int(c_idx)
            arc_prob = float(arc_probs[p_idx, c_idx]) if (p_idx < arc_probs.shape[0] and c_idx < arc_probs.shape[1]) else 0.0
            rel_prob = 0.0
            if label in self.label2id:
                lid = int(self.label2id[label])
                if p_idx < rel_probs.shape[0] and c_idx < rel_probs.shape[1] and lid < rel_probs.shape[2]:
                    rel_prob = float(rel_probs[p_idx, c_idx, lid])
            return arc_prob, rel_prob, float(arc_prob * rel_prob)

        def _hierarchy_hint_skip_reason(
            p_idx: int,
            c_idx: int,
            label: str,
            *,
            arc_prob: Optional[float] = None,
            rel_prob: Optional[float] = None,
        ) -> Optional[str]:
            if str(label) not in self.HIERARCHY_HINT_EDGES:
                return None
            if arc_prob is None or rel_prob is None:
                arc_prob, rel_prob, _confidence = _edge_model_probs(p_idx, c_idx, label)
            if float(arc_prob) < float(self.arc_threshold):
                return "arc_prob_below_threshold"
            if float(rel_prob) < self.MIN_LABEL_PROB:
                return "rel_prob_below_threshold"
            if float(arc_prob) * float(rel_prob) < self.MIN_HIERARCHY_HINT_CONFIDENCE:
                return "confidence_below_threshold"
            return None

        root_scores = np.zeros(num_words, dtype=float)
        if num_words > 0:
            if (
                is_root_logits is not None
                and isinstance(is_root_logits, np.ndarray)
                and is_root_logits.shape[0] >= num_words
                and is_root_logits.shape[-1] >= 2
            ):
                r = is_root_logits[:num_words, :2].astype(np.float32)
                root_scores = (r[:, 1] - r[:, 0]).astype(float)
            elif 0 <= int(root_idx) < num_words:
                root_scores[int(root_idx)] = 1.0

        empty_outgoing = {i: [] for i in range(num_words)}
        empty_incoming = {i: [] for i in range(num_words)}
        empty_children = {i: set() for i in range(num_words)}
        empty_parents = {i: set() for i in range(num_words)}
        empty_validator = EdgeValidator(
            root_idx=int(root_idx),
            node_types=node_types,
            node_incoming_edges=empty_incoming,
            node_outgoing_edges=empty_outgoing,
            children_of=empty_children,
            parents_of=empty_parents,
            edge_labels_by_pair={},
            next_word_targets=set(),
            adjacent_only_types=adjacent_only_types,
            allowed_chain_at=allowed_chain_at,
            would_create_cycle=lambda _p, _c: False,
            pattern_check=None,
        )

        max_labels_per_pair = max(1, int(self.ilp_max_labels_per_pair))
        max_candidates_per_child = max(1, int(self.ilp_max_candidates_per_child))
        raw_by_child: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_words)}
        static_reject_counts: Dict[str, int] = defaultdict(int)

        for child_idx in range(num_words):
            for parent_idx in range(num_words):
                if parent_idx == child_idx:
                    continue
                arc_prob = float(arc_probs[parent_idx, child_idx])
                label_order = np.argsort(rel_probs[parent_idx, child_idx])[::-1][:max_labels_per_pair]
                for label_id in label_order:
                    label = self.id2label.get(int(label_id), "UNK")
                    if label == "UNK" or label == "nonsense" or label not in self.label2id:
                        continue
                    rel_prob = float(rel_probs[parent_idx, child_idx, int(label_id)])
                    if rel_prob <= 0.0:
                        continue

                    skip_reason = _hierarchy_hint_skip_reason(
                        parent_idx,
                        child_idx,
                        label,
                        arc_prob=arc_prob,
                        rel_prob=rel_prob,
                    )
                    if skip_reason is not None:
                        static_reject_counts[f"hierarchy_{skip_reason}"] += 1
                        continue

                    code = empty_validator.validate(
                        parent_idx,
                        child_idx,
                        label,
                        mode="phase2",
                        check_cycle=False,
                        check_pattern=False,
                    )
                    if code != EdgeValidationCode.OK:
                        static_reject_counts[code.name] += 1
                        continue

                    comps = _edge_components(label)
                    if label == "next_word":
                        p_type = node_types[parent_idx]
                        c_type = node_types[child_idx]
                        if (p_type in adjacent_only_types or c_type in adjacent_only_types) and abs(parent_idx - child_idx) != 1:
                            static_reject_counts["NEXT_WORD_NOT_ADJACENT"] += 1
                            continue
                    if node_types[parent_idx] == "FactualPredicator" and "as:modification" in comps:
                        static_reject_counts["FACTUAL_PREDICATOR_CONSTRAINT_VIOLATED"] += 1
                        continue

                    confidence = float(arc_prob * rel_prob)
                    utility = _safe_log(arc_prob) - _safe_log(1.0 - arc_prob) + _safe_log(rel_prob)
                    struct_factor = 1.0
                    if self.edge_prob_tables is not None:
                        struct_factor = float(
                            self.edge_prob_tables.get_prob_child_given_pos(node_types[parent_idx], label)
                            ** self.lam_pos_prob
                        )
                        utility += _safe_log(struct_factor)

                    raw_by_child[child_idx].append({
                        "parent": int(parent_idx),
                        "child": int(child_idx),
                        "label": str(label),
                        "label_id": int(label_id),
                        "arc_prob": float(arc_prob),
                        "rel_prob": float(rel_prob),
                        "confidence": float(confidence),
                        "utility": float(utility),
                        "struct_factor": float(struct_factor),
                        "above_threshold": bool(arc_prob >= float(self.arc_threshold) and rel_prob >= self.MIN_LABEL_PROB),
                    })

        candidates: List[Dict[str, Any]] = []
        truncated_children = []
        for child_idx, child_candidates in raw_by_child.items():
            if not child_candidates:
                continue
            child_candidates.sort(key=lambda e: (-float(e["utility"]), -float(e["confidence"]), int(e["parent"]), str(e["label"])))
            kept = child_candidates[:max_candidates_per_child]
            if len(child_candidates) > len(kept):
                truncated_children.append({
                    "child": int(child_idx),
                    "raw": int(len(child_candidates)),
                    "kept": int(len(kept)),
                })
            candidates.extend(kept)

        edge_count = len(candidates)
        root_offset = edge_count
        order_offset = edge_count + num_words
        total_vars = edge_count + (2 * num_words)

        by_pair: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        incoming_by_child: Dict[int, List[int]] = defaultdict(list)
        outgoing_by_parent: Dict[int, List[int]] = defaultdict(list)
        full_label_by_parent: Dict[Tuple[int, str], List[int]] = defaultdict(list)
        frame_by_node: Dict[Tuple[int, str], List[int]] = defaultdict(list)
        next_in_by_node: Dict[int, List[int]] = defaultdict(list)
        non_next_out_by_node: Dict[int, List[int]] = defaultdict(list)
        punct_in_by_node: Dict[int, List[int]] = defaultdict(list)
        incoming_at_by_node: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        outgoing_at_by_node: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        pair_any_vars: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        pair_mod_vars: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        aux_child_by_parent_child: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        unsupported_aux_child_pair_vars: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        for var_idx, edge in enumerate(candidates):
            parent_idx = int(edge["parent"])
            child_idx = int(edge["child"])
            label = str(edge["label"])
            comps = _edge_components(label)
            pair_key = (parent_idx, child_idx)
            undirected_pair = tuple(sorted((parent_idx, child_idx)))
            by_pair[pair_key].append(var_idx)
            incoming_by_child[child_idx].append(var_idx)
            outgoing_by_parent[parent_idx].append(var_idx)
            full_label_by_parent[(parent_idx, label)].append(var_idx)
            pair_any_vars[undirected_pair].append(var_idx)
            if _is_modification_label(label):
                pair_mod_vars[undirected_pair].append(var_idx)
            if comps & primary_aux_at:
                aux_child_by_parent_child[(parent_idx, child_idx)].append(var_idx)
            if not (comps & hierarchy_hint_at):
                unsupported_aux_child_pair_vars[undirected_pair].append(var_idx)
            if label == "next_word":
                next_in_by_node[child_idx].append(var_idx)
            else:
                non_next_out_by_node[parent_idx].append(var_idx)
            if _is_punctuation_label(label):
                punct_in_by_node[child_idx].append(var_idx)
            at_comp = EdgeValidator._at_component(label, allow_prefix=True)
            if at_comp is not None:
                incoming_at_by_node[child_idx].append((var_idx, at_comp))
                outgoing_at_by_node[parent_idx].append((var_idx, at_comp))
            for comp in frame_components:
                if comp in comps:
                    frame_by_node[(parent_idx, comp)].append(var_idx)
                if f"reverse_{comp}" in comps:
                    frame_by_node[(child_idx, comp)].append(var_idx)

        row_idx: List[int] = []
        col_idx: List[int] = []
        values: List[float] = []
        lower: List[float] = []
        upper: List[float] = []

        def _add_constraint(terms: List[Tuple[int, float]], lb: float, ub: float) -> None:
            row = len(lower)
            for col, val in terms:
                row_idx.append(row)
                col_idx.append(int(col))
                values.append(float(val))
            lower.append(float(lb))
            upper.append(float(ub))

        # One label per directed pair.
        for idxs in by_pair.values():
            _add_constraint([(i, 1.0) for i in idxs], -np.inf, 1.0)

        # Exactly one root.
        if num_words > 0:
            _add_constraint([(root_offset + i, 1.0) for i in range(num_words)], 1.0, 1.0)

        # Root/non-root indegree contract:
        # root: indegree == 0; every non-root: indegree >= 1.
        for node_idx in range(num_words):
            incoming = incoming_by_child.get(node_idx, [])
            root_var = root_offset + node_idx
            max_incoming = max(1, len(incoming))
            _add_constraint([(i, 1.0) for i in incoming] + [(root_var, 1.0)], 1.0, np.inf)
            _add_constraint([(i, 1.0) for i in incoming] + [(root_var, float(max_incoming))], -np.inf, float(max_incoming))

        # Root gets topological order 0. Selected edges must increase order by >= 1.
        if num_words > 0:
            max_order = max(0, num_words - 1)
            for node_idx in range(num_words):
                _add_constraint(
                    [(order_offset + node_idx, 1.0), (root_offset + node_idx, float(max_order))],
                    -np.inf,
                    float(max_order),
                )
            for var_idx, edge in enumerate(candidates):
                parent_idx = int(edge["parent"])
                child_idx = int(edge["child"])
                _add_constraint(
                    [
                        (order_offset + parent_idx, 1.0),
                        (order_offset + child_idx, -1.0),
                        (var_idx, float(num_words)),
                    ],
                    -np.inf,
                    float(num_words - 1),
                )

        # Outgoing full-label cardinality, with root-specific func.arg allowance.
        for (parent_idx, label), idxs in full_label_by_parent.items():
            base_max = int(self._get_max_edge_count(node_types[parent_idx], label, is_root=False))
            root_max = int(self._get_max_edge_count(node_types[parent_idx], label, is_root=True))
            bonus = max(0, root_max - base_max)
            terms = [(i, 1.0) for i in idxs]
            if bonus:
                terms.append((root_offset + parent_idx, -float(bonus)))
            _add_constraint(terms, -np.inf, float(base_max))

        # Frame arity: no repeated pred.arg.x / variable / body for the same logical parent.
        for idxs in frame_by_node.values():
            _add_constraint([(i, 1.0) for i in idxs], -np.inf, 1.0)

        # next_word target constraints: if a node receives next_word, it has no
        # other incoming edge and may only emit next_word edges.
        for node_idx, next_idxs in next_in_by_node.items():
            incoming = incoming_by_child.get(node_idx, [])
            if incoming:
                _add_constraint(
                    [(i, 1.0) for i in incoming] + [(i, float(max(0, len(incoming) - 1))) for i in next_idxs],
                    -np.inf,
                    float(len(incoming)),
                )
            non_next_out = non_next_out_by_node.get(node_idx, [])
            if non_next_out:
                _add_constraint(
                    [(i, 1.0) for i in non_next_out] + [(i, float(len(non_next_out))) for i in next_idxs],
                    -np.inf,
                    float(len(non_next_out)),
                )

        # Punctuation edges attach leaf nodes.
        for node_idx, punct_idxs in punct_in_by_node.items():
            outgoing = outgoing_by_parent.get(node_idx, [])
            if outgoing:
                _add_constraint(
                    [(i, 1.0) for i in outgoing] + [(i, float(len(outgoing))) for i in punct_idxs],
                    -np.inf,
                    float(len(outgoing)),
                )

        # The next three constraint families are high-order and can be much
        # larger than the final selected graph. In lazy mode we do not enumerate
        # them upfront: solve a compact ILP, inspect the selected graph, add only
        # violated cuts, and re-solve.
        lazy_constraint_keys: Set[Tuple[Any, ...]] = set()
        at_conflict_count = 0
        aux_child_subgraph_constraints = 0
        triangle_constraints = 0
        triangle_truncated = False

        def _add_lazy_constraint(
            key: Tuple[Any, ...],
            terms: List[Tuple[int, float]],
            lb: float,
            ub: float,
            family: str,
        ) -> bool:
            nonlocal at_conflict_count, aux_child_subgraph_constraints, triangle_constraints, triangle_truncated
            if key in lazy_constraint_keys:
                return False
            if family == "modification_triangle" and triangle_constraints >= self.ilp_max_triangle_constraints:
                triangle_truncated = True
                return False
            lazy_constraint_keys.add(key)
            _add_constraint(terms, lb, ub)
            if family == "at_chain_conflict":
                at_conflict_count += 1
            elif family == "aux_child_subgraph":
                aux_child_subgraph_constraints += 1
            elif family == "modification_triangle":
                triangle_constraints += 1
            return True

        # word2gp.lower_edge_to_node_hierarchy() treats multiple primary @
        # children of the same parent as an aux-child hierarchy. Edges inside
        # that child-only subgraph must be @lower/@upper hints; ordinary
        # semantic edges there cannot be converted back to Lingua.
        aux_children_by_parent: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
        for (parent_idx, child_idx), idxs in aux_child_by_parent_child.items():
            aux_children_by_parent[int(parent_idx)][int(child_idx)] = idxs

        mod_neighbors: Dict[int, Set[int]] = defaultdict(set)
        for a, b in pair_mod_vars.keys():
            mod_neighbors[int(a)].add(int(b))
            mod_neighbors[int(b)].add(int(a))

        def _at_chain_cut_key(in_var: int, out_var: int) -> Tuple[Any, ...]:
            return ("at_chain_conflict", int(in_var), int(out_var))

        def _add_at_chain_cut(in_var: int, out_var: int) -> bool:
            return _add_lazy_constraint(
                _at_chain_cut_key(in_var, out_var),
                [(int(in_var), 1.0), (int(out_var), 1.0)],
                -np.inf,
                1.0,
                "at_chain_conflict",
            )

        def _aux_child_cut_key(parent_idx: int, first_child: int, second_child: int) -> Tuple[Any, ...]:
            first_child, second_child = sorted((int(first_child), int(second_child)))
            return ("aux_child_subgraph", int(parent_idx), first_child, second_child)

        def _add_aux_child_cut(parent_idx: int, first_child: int, second_child: int) -> bool:
            child_map = aux_children_by_parent[int(parent_idx)]
            bad_pair = tuple(sorted((int(first_child), int(second_child))))
            bad_edge_vars = unsupported_aux_child_pair_vars.get(bad_pair, [])
            if not bad_edge_vars:
                return False
            terms = (
                [(i, 1.0) for i in child_map[int(first_child)]]
                + [(i, 1.0) for i in child_map[int(second_child)]]
                + [(i, 1.0) for i in bad_edge_vars]
            )
            return _add_lazy_constraint(
                _aux_child_cut_key(parent_idx, first_child, second_child),
                terms,
                -np.inf,
                2.0,
                "aux_child_subgraph",
            )

        def _mod_triangle_key(
            mod_pair_1: Tuple[int, int],
            mod_pair_2: Tuple[int, int],
            any_pair: Tuple[int, int],
        ) -> Tuple[Any, ...]:
            mod_pair_1 = tuple(sorted(mod_pair_1))
            mod_pair_2 = tuple(sorted(mod_pair_2))
            any_pair = tuple(sorted(any_pair))
            return ("modification_triangle",) + tuple(sorted((mod_pair_1, mod_pair_2, any_pair)))

        def _add_mod_triangle_cut(
            mod_pair_1: Tuple[int, int],
            mod_pair_2: Tuple[int, int],
            any_pair: Tuple[int, int],
        ) -> bool:
            mod_pair_1 = tuple(sorted(mod_pair_1))
            mod_pair_2 = tuple(sorted(mod_pair_2))
            any_pair = tuple(sorted(any_pair))
            if mod_pair_1 not in pair_mod_vars or mod_pair_2 not in pair_mod_vars or any_pair not in pair_any_vars:
                return False
            terms = (
                [(i, 1.0) for i in pair_mod_vars[mod_pair_1]]
                + [(i, 1.0) for i in pair_mod_vars[mod_pair_2]]
                + [(i, 1.0) for i in pair_any_vars[any_pair]]
            )
            return _add_lazy_constraint(
                _mod_triangle_key(mod_pair_1, mod_pair_2, any_pair),
                terms,
                -np.inf,
                2.0,
                "modification_triangle",
            )

        def _add_all_high_order_constraints() -> Dict[str, int]:
            added = {"at_chain_conflict": 0, "aux_child_subgraph": 0, "modification_triangle": 0}

            for node_idx in range(num_words):
                incoming_at = incoming_at_by_node.get(node_idx, [])
                outgoing_at = outgoing_at_by_node.get(node_idx, [])
                if not incoming_at or not outgoing_at:
                    continue
                for in_var, in_at in incoming_at:
                    if in_at in hierarchy_hint_at:
                        continue
                    for out_var, out_at in outgoing_at:
                        if out_at in hierarchy_hint_at or int(in_var) == int(out_var):
                            continue
                        if in_at != out_at or in_at not in allowed_chain_at:
                            if _add_at_chain_cut(in_var, out_var):
                                added["at_chain_conflict"] += 1

            for parent_idx, child_map in aux_children_by_parent.items():
                children = sorted(child_map)
                if len(children) < 2:
                    continue
                for pos, first_child in enumerate(children):
                    for second_child in children[pos + 1:]:
                        if _add_aux_child_cut(parent_idx, first_child, second_child):
                            added["aux_child_subgraph"] += 1

            for u, neighbors in list(mod_neighbors.items()):
                for v in list(neighbors):
                    if u >= v:
                        continue
                    for w in mod_neighbors.get(u, set()):
                        if w == v:
                            continue
                        if _add_mod_triangle_cut((u, v), (u, w), (v, w)):
                            added["modification_triangle"] += 1
                    for w in mod_neighbors.get(v, set()):
                        if w == u:
                            continue
                        if _add_mod_triangle_cut((u, v), (v, w), (u, w)):
                            added["modification_triangle"] += 1
            return added

        def _selected_pairs(selected: Set[int], pair_map: Dict[Tuple[int, int], List[int]]) -> Set[Tuple[int, int]]:
            return {
                tuple(pair)
                for pair, idxs in pair_map.items()
                if any(int(i) in selected for i in idxs)
            }

        def _add_violated_high_order_constraints(x: np.ndarray) -> Dict[str, int]:
            selected = {idx for idx in range(edge_count) if float(x[idx]) >= 0.5}
            added = {"at_chain_conflict": 0, "aux_child_subgraph": 0, "modification_triangle": 0}

            for node_idx in range(num_words):
                incoming_at = [(i, at) for i, at in incoming_at_by_node.get(node_idx, []) if int(i) in selected]
                outgoing_at = [(i, at) for i, at in outgoing_at_by_node.get(node_idx, []) if int(i) in selected]
                if not incoming_at or not outgoing_at:
                    continue
                for in_var, in_at in incoming_at:
                    if in_at in hierarchy_hint_at:
                        continue
                    for out_var, out_at in outgoing_at:
                        if out_at in hierarchy_hint_at or int(in_var) == int(out_var):
                            continue
                        if in_at != out_at or in_at not in allowed_chain_at:
                            if _add_at_chain_cut(in_var, out_var):
                                added["at_chain_conflict"] += 1

            for parent_idx, child_map in aux_children_by_parent.items():
                selected_children = [
                    child_idx
                    for child_idx, idxs in child_map.items()
                    if any(int(i) in selected for i in idxs)
                ]
                if len(selected_children) < 2:
                    continue
                for pos, first_child in enumerate(sorted(selected_children)):
                    for second_child in sorted(selected_children)[pos + 1:]:
                        bad_pair = tuple(sorted((int(first_child), int(second_child))))
                        if not any(int(i) in selected for i in unsupported_aux_child_pair_vars.get(bad_pair, [])):
                            continue
                        if _add_aux_child_cut(parent_idx, first_child, second_child):
                            added["aux_child_subgraph"] += 1

            selected_mod_pairs = _selected_pairs(selected, pair_mod_vars)
            selected_any_pairs = _selected_pairs(selected, pair_any_vars)
            selected_mod_neighbors: Dict[int, Set[int]] = defaultdict(set)
            for a, b in selected_mod_pairs:
                selected_mod_neighbors[int(a)].add(int(b))
                selected_mod_neighbors[int(b)].add(int(a))
            for u, neighbors in list(selected_mod_neighbors.items()):
                for v in list(neighbors):
                    if u >= v:
                        continue
                    for w in selected_mod_neighbors.get(u, set()):
                        if w == v:
                            continue
                        if tuple(sorted((v, w))) in selected_any_pairs:
                            if _add_mod_triangle_cut((u, v), (u, w), (v, w)):
                                added["modification_triangle"] += 1
                    for w in selected_mod_neighbors.get(v, set()):
                        if w == u:
                            continue
                        if tuple(sorted((u, w))) in selected_any_pairs:
                            if _add_mod_triangle_cut((u, v), (v, w), (u, w)):
                                added["modification_triangle"] += 1
            return added

        if not self.ilp_lazy_constraints:
            _add_all_high_order_constraints()

        c = np.zeros(total_vars, dtype=float)
        for var_idx, edge in enumerate(candidates):
            c[var_idx] = -float(edge["utility"])
        for node_idx in range(num_words):
            c[root_offset + node_idx] = -float(self.ilp_root_score_weight) * float(root_scores[node_idx])

        lb = np.zeros(total_vars, dtype=float)
        ub = np.ones(total_vars, dtype=float)
        if num_words > 0:
            lb[order_offset:order_offset + num_words] = 0.0
            ub[order_offset:order_offset + num_words] = float(max(0, num_words - 1))
        integrality = np.zeros(total_vars, dtype=int)
        integrality[:edge_count + num_words] = 1

        decoding_info["ilp"] = {
            "candidate_edges": int(edge_count),
            "raw_candidate_edges": int(sum(len(v) for v in raw_by_child.values())),
            "constraints": int(len(lower)),
            "base_constraints": int(len(lower) - at_conflict_count - aux_child_subgraph_constraints - triangle_constraints),
            "lazy_constraints": bool(self.ilp_lazy_constraints),
            "max_lazy_rounds": int(self.ilp_max_lazy_rounds),
            "at_chain_conflict_constraints": int(at_conflict_count),
            "aux_child_subgraph_constraints": int(aux_child_subgraph_constraints),
            "modification_triangle_constraints": int(triangle_constraints),
            "triangle_constraints_truncated": bool(triangle_truncated),
            "children_with_truncated_candidates": truncated_children[:20],
            "static_reject_counts": dict(static_reject_counts),
            "lazy_rounds": [],
        }

        def _solve_current_constraints() -> Any:
            if lower:
                A = coo_matrix((values, (row_idx, col_idx)), shape=(len(lower), total_vars)).tocsr()
                constraints = LinearConstraint(A, np.array(lower, dtype=float), np.array(upper, dtype=float))
            else:
                constraints = ()
            return milp(
                c=c,
                integrality=integrality,
                bounds=Bounds(lb, ub),
                constraints=constraints,
                options={
                    "time_limit": float(self.ilp_time_limit),
                    "mip_rel_gap": float(self.ilp_mip_rel_gap),
                    "disp": False,
                },
            )

        result = None
        round_idx = 0
        eager_fallback_solve = False
        max_lazy_rounds = max(0, int(self.ilp_max_lazy_rounds))
        while True:
            constraints_before = len(lower)
            try:
                result = _solve_current_constraints()
            except Exception as e:
                decoding_info["errors"].append(f"ILP solve failed: {e}")
                decoding_info["has_issues"] = True
                return graph, decoding_info

            round_info = {
                "round": int(round_idx),
                "constraints_before": int(constraints_before),
                "status": int(getattr(result, "status", -1)),
                "success": bool(getattr(result, "success", False)),
                "objective": float(getattr(result, "fun", 0.0)) if getattr(result, "fun", None) is not None else None,
                "cuts_added": {"at_chain_conflict": 0, "aux_child_subgraph": 0, "modification_triangle": 0},
            }
            decoding_info["ilp"]["lazy_rounds"].append(round_info)

            if getattr(result, "x", None) is None:
                break

            if not self.ilp_lazy_constraints or eager_fallback_solve:
                break

            added = _add_violated_high_order_constraints(np.asarray(result.x, dtype=float))
            round_info["cuts_added"] = {k: int(v) for k, v in added.items()}
            round_info["constraints_after"] = int(len(lower))
            if sum(added.values()) == 0:
                break

            if round_idx >= max_lazy_rounds:
                eager_added = _add_all_high_order_constraints()
                round_info["eager_fallback_cuts_added"] = {k: int(v) for k, v in eager_added.items()}
                decoding_info["warnings"].append(
                    f"ILP lazy constraints exceeded {max_lazy_rounds} rounds; solving once with all high-order cuts"
                )
                eager_fallback_solve = True

            round_idx += 1

        decoding_info["ilp"].update({
            "constraints": int(len(lower)),
            "at_chain_conflict_constraints": int(at_conflict_count),
            "aux_child_subgraph_constraints": int(aux_child_subgraph_constraints),
            "modification_triangle_constraints": int(triangle_constraints),
            "triangle_constraints_truncated": bool(triangle_truncated),
        })

        if triangle_truncated:
            decoding_info["warnings"].append(
                f"ILP modification-triangle constraints truncated at {self.ilp_max_triangle_constraints}"
            )

        if result is None:
            decoding_info["errors"].append("ILP did not run")
            decoding_info["has_issues"] = True
            return graph, decoding_info

        decoding_info["ilp"].update({
            "status": int(getattr(result, "status", -1)),
            "success": bool(getattr(result, "success", False)),
            "message": str(getattr(result, "message", "")),
            "objective": float(getattr(result, "fun", 0.0)) if getattr(result, "fun", None) is not None else None,
        })

        if getattr(result, "x", None) is None:
            decoding_info["errors"].append(f"ILP returned no solution: {getattr(result, 'message', '')}")
            decoding_info["has_issues"] = True
            return graph, decoding_info

        x = np.asarray(result.x, dtype=float)
        selected_edge_indices = [idx for idx in range(edge_count) if x[idx] >= 0.5]
        root_values = x[root_offset:root_offset + num_words] if num_words > 0 else np.array([])
        ilp_root_idx = int(np.argmax(root_values)) if root_values.size else None
        decoding_info["ilp"].update({
            "selected_edges": int(len(selected_edge_indices)),
            "selected_root": ilp_root_idx,
            "selected_root_word": words[ilp_root_idx] if ilp_root_idx is not None and ilp_root_idx < len(words) else None,
        })

        node_outgoing_edges = {i: [] for i in range(num_words)}
        node_incoming_edges = {i: [] for i in range(num_words)}
        children_of = {i: set() for i in range(num_words)}
        parents_of = {i: set() for i in range(num_words)}
        edge_labels_by_pair: Dict[Tuple[int, int], str] = {}

        for var_idx in sorted(selected_edge_indices, key=lambda i: (int(candidates[i]["parent"]), int(candidates[i]["child"]), str(candidates[i]["label"]))):
            edge_info = candidates[var_idx]
            parent_idx = int(edge_info["parent"])
            child_idx = int(edge_info["child"])
            label = str(edge_info["label"])
            gpg_edge = GPGEdge(label=label)
            gpg_edge.arc_prob = float(edge_info["arc_prob"])
            gpg_edge.rel_prob = float(edge_info["rel_prob"])
            gpg_edge.original_score = float(edge_info["confidence"])
            gpg_edge.transformed_score = float(edge_info["utility"])
            gpg_edge.struct_factor = float(edge_info.get("struct_factor", 1.0))
            gpg_edge.decoder_source = "ilp"
            graph.add_edge(word_to_node[parent_idx], word_to_node[child_idx], gpg_edge)
            node_outgoing_edges[parent_idx].append(label)
            node_incoming_edges[child_idx].append(label)
            children_of[parent_idx].add(child_idx)
            parents_of[child_idx].add(parent_idx)
            edge_labels_by_pair[(parent_idx, child_idx)] = label

        def _would_create_cycle(parent: int, child: int) -> bool:
            visited = set()
            stack = [int(child)]
            while stack:
                node = stack.pop()
                if node == int(parent):
                    return True
                if node in visited:
                    continue
                visited.add(node)
                stack.extend(children_of.get(node, set()))
            return False

        def _add_hierarchy_hint_edge(
            p_idx: int,
            c_idx: int,
            label: str,
            *,
            source: str,
            case: Optional[int] = None,
            meta: Optional[Dict[str, Any]] = None,
        ) -> Optional[Dict[str, Any]]:
            p_idx = int(p_idx)
            c_idx = int(c_idx)
            label = str(label)
            if label not in self.HIERARCHY_HINT_EDGES or label not in self.label2id:
                return None
            if graph.g.has_edge(word_to_node[p_idx].ID, word_to_node[c_idx].ID):
                return None
            arc_prob, rel_prob, confidence = _edge_model_probs(p_idx, c_idx, label)
            skip_reason = _hierarchy_hint_skip_reason(p_idx, c_idx, label, arc_prob=arc_prob, rel_prob=rel_prob)
            if skip_reason is None and _would_create_cycle(p_idx, c_idx):
                skip_reason = "would_create_cycle"

            info = {
                "from": word_to_node[p_idx].ID,
                "to": word_to_node[c_idx].ID,
                "from_word_idx": int(p_idx),
                "to_word_idx": int(c_idx),
                "label": label,
                "source": source,
                "case": case,
                "arc_prob": float(arc_prob),
                "rel_prob": float(rel_prob),
                "confidence": float(confidence),
                "meta": meta or {},
            }
            if skip_reason is not None:
                info["reason"] = skip_reason
                decoding_info.setdefault("skipped_hierarchy_edges", []).append(info)
                return None

            edge = GPGEdge(label=label)
            edge.arc_prob = float(arc_prob)
            edge.rel_prob = float(rel_prob)
            edge.original_score = float(confidence)
            edge.transformed_score = float(confidence)
            edge.decoder_source = source
            edge.postprocess = True
            edge.postprocess_case = case
            graph.add_edge(word_to_node[p_idx], word_to_node[c_idx], edge)
            node_outgoing_edges[p_idx].append(label)
            node_incoming_edges[c_idx].append(label)
            children_of[p_idx].add(c_idx)
            parents_of[c_idx].add(p_idx)
            edge_labels_by_pair[(p_idx, c_idx)] = label
            return info

        self._add_upper_lower_edges(
            graph,
            word_to_node,
            node_outgoing_edges,
            decoding_info,
            add_hierarchy_hint_edge=_add_hierarchy_hint_edge,
        )

        if self.debug:
            self._debug_visualize_step(
                graph,
                "final",
                "word_lingua_ilp",
                arc_probs=arc_probs,
                rel_probs=rel_probs,
                label2id={v: k for k, v in self.label2id.items()},
            )

        validation_errors = self._validate_word_lingua_graph(graph, node_types, decoding_info)
        if validation_errors:
            decoding_info["validation_errors"] = validation_errors

        decoding_info["has_issues"] = bool(
            decoding_info["errors"]
            or decoding_info["warnings"]
            or validation_errors
            or not bool(getattr(result, "success", False))
        )
        return graph, decoding_info
    
    # PRIMARY_AT_EDGES: symbolic edges that represent aux node relationships
    # These are allowed to have multiple occurrences per node (unlike the old constraint)
    PRIMARY_AT_EDGES = {
        "@modification_of", "@appositive_of", "@attribute_of", 
        "@copula", "@discourse_of", "@expression", "@list", 
        "@parataxis", "@ref_of", "@vocative_of", "@whether_of"
    }
    
    # List-like edges that can form chains
    LIST_LIKE_EDGES = {"@list", "@parataxis", "@expression"}
    
    def _is_edge_valid_for_pattern(
        self, 
        parent_idx: int, 
        new_label: str, 
        node_type: str,
        current_outgoing: List[str],
        is_root: bool = False
    ) -> bool:
        """
        Check if adding a new edge with the given label is valid for the node's pattern.
        
        Based on training data analysis:
        - Only 'func.arg' edges can be duplicated (for ConjunctionalFunctor, ExpressionFunctor)
        - PRIMARY_AT_EDGES (like @modification_of, @list, etc.) can have multiple occurrences
        - Each specific PRIMARY_AT_EDGE label can only appear once per node
        - @upper and @lower are legal decoder labels when they are present in
          the training edge vocabulary.
        
        Uses statistics when available to get max edge counts.
        """
        # First, try to use statistics to get max allowed count for this edge label
        max_count = self._get_max_edge_count(node_type, new_label, is_root)
        
        current_count = current_outgoing.count(new_label)
        if current_count >= max_count:
            return False
        
        return True
    
    def _get_max_edge_count(self, node_type: str, edge_label: str, is_root: bool = False) -> int:
        """
        Max allowed multiplicity for an outgoing edge label.

        Updated to allow multiple PRIMARY_AT_EDGES per node, but each specific
        PRIMARY_AT_EDGE label can only appear once.
        """
        if edge_label == "func.arg":
            if is_root:
                return 8
            if node_type == "ConjunctionalFunctor":
                return 8
            if node_type == "ExpressionFunctor":
                return 3
            return 1
        
        # PRIMARY_AT_EDGES: each specific label can appear once per node
        # but a node can have multiple different @ edges
        if edge_label in self.PRIMARY_AT_EDGES:
            return 1
        
        return 1
    
    def _get_offsprings(self, graph: GPGraph, node) -> set:
        """Get all offspring (descendants) of a node in the graph."""
        offsprings = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child, _ in list(graph.children(current)):
                if child not in offsprings:
                    offsprings.add(child)
                    stack.append(child)
        return offsprings
    
    def _add_upper_lower_edges(
        self, 
        graph: GPGraph, 
        word_to_node: Dict[int, Any],
        node_outgoing_edges: Dict[int, List[str]],
        decoding_info: Dict[str, Any],
        *,
        add_hierarchy_hint_edge: Callable[..., Optional[Dict[str, Any]]],
    ) -> None:
        """
        Phase 5: Add @upper and @lower edges as post-processing.
        
        These edges are added in two cases:
        
        Case 1: When a node has multiple @ children
        - For each @ child A's @ grandchildren gc
        - If another @ child B is in gc's offsprings: add gc → B with @upper
        - Otherwise: add B → gc with @lower
        
        Case 2: For list-like chains (list/expression/parataxis)
        - If child has @ edge with same label pointing to grandchild
        - Add parent → grandchild with @lower
        """
        phase5_edges_added = []
        node_to_word_idx = {node.ID: idx for idx, node in word_to_node.items()}
        node_to_word_idx.update({str(node.ID): idx for idx, node in word_to_node.items()})

        def _add_phase5_edge(parent: Any, child: Any, label: str, case: int, meta: Dict[str, Any]) -> None:
            p_idx = node_to_word_idx.get(parent.ID, node_to_word_idx.get(str(parent.ID)))
            c_idx = node_to_word_idx.get(child.ID, node_to_word_idx.get(str(child.ID)))
            if p_idx is None or c_idx is None:
                return
            added = add_hierarchy_hint_edge(
                int(p_idx),
                int(c_idx),
                label,
                source="phase5",
                case=case,
                meta=meta,
            )
            if added is not None:
                phase5_edges_added.append(added)
        
        # Case 1: Handle nodes with multiple @ children
        # Convert to list before iterating to avoid 'dictionary changed size during iteration'
        for node in list(graph.nodes()):
            at_children = []  # List of (child_node, edge_labels_set)
            for child, edge in list(graph.children(node)):
                edge_labels = set(edge.label.split("+"))
                primary_labels = edge_labels & self.PRIMARY_AT_EDGES
                if primary_labels:
                    at_children.append((child, primary_labels, edge.label))
            
            if len(at_children) <= 1:
                continue  # Only one @ child, no hierarchy needed
            
            # For each pair of @ children, check grandchildren relationships
            for i, (child_i, labels_i, full_label_i) in enumerate(at_children):
                for j, (child_j, labels_j, full_label_j) in enumerate(at_children):
                    if i == j:
                        continue
                    
                    # Check child_i's @ grandchildren
                    for grandchild, gc_edge in list(graph.children(child_i)):
                        gc_labels = set(gc_edge.label.split("+"))
                        if not (gc_labels & self.PRIMARY_AT_EDGES):
                            continue
                        
                        # Case 1 logic
                        offsprings_of_gc = self._get_offsprings(graph, grandchild)
                        if child_j in offsprings_of_gc:
                            # grandchild → child_j: @upper
                            if not graph.has_edge(grandchild, child_j):
                                _add_phase5_edge(
                                    grandchild,
                                    child_j,
                                    "@upper",
                                    1,
                                    {
                                        "node": node.ID,
                                        "child_i": child_i.ID,
                                        "child_j": child_j.ID,
                                        "gc_edge_label": gc_edge.label,
                                    },
                                )
                        else:
                            # child_j → grandchild: @lower
                            if not graph.has_edge(child_j, grandchild):
                                _add_phase5_edge(
                                    child_j,
                                    grandchild,
                                    "@lower",
                                    1,
                                    {
                                        "node": node.ID,
                                        "child_i": child_i.ID,
                                        "child_j": child_j.ID,
                                        "gc_edge_label": gc_edge.label,
                                    },
                                )
        
        # Case 2: Handle list-like chains
        # Convert to list before iterating to avoid 'dictionary changed size during iteration'
        for node in list(graph.nodes()):
            for child, edge in list(graph.children(node)):
                edge_labels = edge.label.split("+")
                
                # Find list-like edge labels
                list_like_in_edge = [lbl for lbl in edge_labels if lbl in self.LIST_LIKE_EDGES]
                if not list_like_in_edge:
                    continue
                
                aux_edge_label = list_like_in_edge[0]
                
                # Check child's @ children
                for grandchild, gc_edge in list(graph.children(child)):
                    gc_edge_labels = gc_edge.label.split("+")
                    
                    # Case 2: if child → grandchild uses the same list-like label
                    if aux_edge_label in gc_edge_labels:
                        # node → grandchild: @lower
                        if not graph.has_edge(node, grandchild):
                            _add_phase5_edge(
                                node,
                                grandchild,
                                "@lower",
                                2,
                                {
                                    "child": child.ID,
                                    "aux_edge_label": aux_edge_label,
                                    "child_edge_label": edge.label,
                                    "grandchild_edge_label": gc_edge.label,
                                },
                            )
        
        # Record in decoding_info
        decoding_info["phase5_upper_lower_edges"] = phase5_edges_added
        
        if self.debug and phase5_edges_added:
            logger.debug(f"Phase 5: Added {len(phase5_edges_added)} @upper/@lower edges")
            for edge_info in phase5_edges_added:
                logger.debug(f"  {edge_info['from']} -> {edge_info['to']}: {edge_info['label']} (case {edge_info['case']})")
    
    def _validate_word_lingua_graph(
        self, 
        graph: GPGraph, 
        node_types: List[str],
        decoding_info: Dict[str, Any]
    ) -> List[str]:
        """
        Validate the word-lingua graph to ensure:
        1. All full edge labels are in the training label vocabulary
        2. All node types are in the allowed set
        3. The graph is a DAG. In SDP graph mode multiple roots/components are
           legal; legacy single-root/connectivity checks are opt-in.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        import networkx as nx
        
        errors = []
        
        # 1. Check edge labels
        allowed_labels = set(self.label2id.keys())
        for parent, edge, child in list(graph.edges()):
            edge_label = str(edge.label)
            if edge_label not in allowed_labels:
                errors.append(
                    f"Invalid edge label not in training vocabulary: "
                    f"'{graph.node_text(parent)}' --{edge_label}--> '{graph.node_text(child)}'"
                )
        
        # 2. Check node types
        allowed_node_types = set(self.id2node_type.values())
        for node in list(graph.nodes()):
            if hasattr(node, 'pos') and node.pos:
                if node.pos not in allowed_node_types:
                    errors.append(f"Invalid node type: '{node.pos}' for node '{graph.node_text(node)}'")
        
        # 3. Check DAG/topology
        # 3a. Check for cycles (DAG requirement)
        try:
            if nx.is_directed_acyclic_graph(graph.g):
                pass  # OK, it's a DAG
            else:
                errors.append("Graph contains cycles (not a DAG)")
        except Exception as e:
            errors.append(f"Error checking DAG property: {e}")
        
        # 3b. Check single root only when tree-like legacy repair is enabled.
        roots = [node for node in list(graph.nodes()) if graph.g.in_degree(node.ID) == 0]
        if len(roots) == 0:
            errors.append("Graph has no root (all nodes have incoming edges - possible cycle)")
        elif self.enforce_single_root and len(roots) > 1:
            root_texts = [graph.node_text(r) for r in roots]
            errors.append(f"Graph has multiple roots ({len(roots)}): {root_texts[:5]}{'...' if len(roots) > 5 else ''}")

        # 3c. Check weak connectivity only when explicitly requested.
        if self.enforce_connected and graph.g.number_of_nodes() > 0:
            try:
                if not nx.is_weakly_connected(graph.g):
                    errors.append("Graph is not weakly connected")
            except Exception as e:
                errors.append(f"Error checking weak connectivity: {e}")
        
        # Log validation errors in debug mode
        if self.debug and errors:
            logger.warning(f"Word-lingua graph validation errors: {errors}")
        
        return errors
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x64 = np.asarray(x, dtype=np.float64)
        return 1 / (1 + np.exp(-np.clip(x64, -40, 40)))
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
