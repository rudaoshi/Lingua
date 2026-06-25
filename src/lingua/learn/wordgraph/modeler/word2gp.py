from graphlib import TopologicalSorter
from numpy import var
from lingua.structure.gpgraph import GPGraph, GPGPhraseNode, GPGAuxNode, GPGEdge, GPGNode, GPGraphVisualizer
from collections import defaultdict
import itertools
import logging

import networkx as nx

from lingua.structure.utils import positions2spans
logger = logging.getLogger(__name__)

from lingua.utils.topology_sorter import LinguaGraphTopologySorter

HIERARCHY_EDGE_LABELS = {"@upper", "@lower"}


def split_edge_label(label: str):
    return [edge_label for edge_label in str(label).split("+") if edge_label]


def strip_hierarchy_edge_components(lingua_graph: GPGraph):
    """Remove word-lingua hierarchy hints from final Lingua edge labels."""
    updated = False

    for parent, edge, child in list(lingua_graph.edges()):
        edge_labels = split_edge_label(edge.label)
        remaining_edge_labels = [
            edge_label for edge_label in edge_labels
            if edge_label not in HIERARCHY_EDGE_LABELS
        ]

        if len(remaining_edge_labels) == len(edge_labels):
            continue

        confidence = edge.confidence
        lingua_graph.remove_relation(parent, child)
        if remaining_edge_labels:
            lingua_graph.add_relation(
                parent,
                child,
                "+".join(remaining_edge_labels),
                confidence=confidence,
            )
        updated = True

    return lingua_graph, updated


def _node_sort_key(lingua_graph: GPGraph, node: GPGNode):
    if isinstance(node, GPGPhraseNode):
        starts = [
            span[0]
            for span in node.spans
            if isinstance(span, tuple) and isinstance(span[0], int)
        ]
        start = min(starts) if starts else 10**9
    else:
        start = 10**9
    return (start, str(node.ID))


def _node_start(node: GPGNode) -> int:
    if not isinstance(node, GPGPhraseNode):
        return 10**9
    starts = [
        span[0]
        for span in node.spans
        if isinstance(span, tuple) and isinstance(span[0], int)
    ]
    return min(starts) if starts else 10**9


def _has_as_shorthand_outgoing(lingua_graph: GPGraph, node: GPGNode) -> bool:
    return any(str(edge.label).startswith("as:") for _, edge in lingua_graph.children(node))


def _as_punctuation_children(lingua_graph: GPGraph, node: GPGNode) -> list[GPGNode]:
    return [
        child
        for child, edge in lingua_graph.children(node)
        if str(edge.label) == "as:punctuation"
    ]


def _is_punctuation_shorthand_root(lingua_graph: GPGraph, node: GPGNode) -> bool:
    return (
        getattr(node, "pos", None) == "PunctuationalConstant"
        and bool(_as_punctuation_children(lingua_graph, node))
    )


def _punctuation_restore_order(parent: GPGNode, children: list[GPGNode]) -> list[GPGNode]:
    """Order repeated ``@punctuation_of`` children from inner to outer."""

    parent_start = _node_start(parent)

    def key(child: GPGNode):
        child_start = _node_start(child)
        is_punctuation = int(getattr(child, "pos", None) == "PunctuationalConstant")
        distance = abs(parent_start - child_start)
        return (1 - is_punctuation, distance, child_start, str(child.ID))

    return sorted(children, key=key)


def _first_symbolic_label(labels: list[str]) -> str | None:
    for label in labels:
        if label in symbol_edge2nodelabel:
            return label
    return None


def _mixed_aux_restore_order(
    parent: GPGNode,
    children: list[GPGNode],
    aux_edge_labels_by_child: dict[int, list[str]],
) -> list[GPGNode]:
    """Order mixed aux labels from inner wrappers to outer wrappers."""

    punctuation_order = _punctuation_restore_order(parent, children)
    punctuation_rank = {node.ID: index for index, node in enumerate(punctuation_order)}

    def key(child: GPGNode):
        aux_label = _first_symbolic_label(aux_edge_labels_by_child[child.ID])
        is_punctuation_aux = int(aux_label == "@punctuation_of")
        return (
            is_punctuation_aux,
            punctuation_rank.get(child.ID, len(punctuation_rank)),
            _node_start(child),
            str(child.ID),
        )

    return sorted(children, key=key)


def _aux_hierarchy_topological_order(
    hierarchy_graph: nx.DiGraph,
    parent: GPGNode,
    children: list[GPGNode],
    aux_edge_labels_by_child: dict[int, list[str]],
) -> list[int]:
    aux_labels = {
        label
        for labels in aux_edge_labels_by_child.values()
        for label in labels
        if label in symbol_edge2nodelabel
    }
    if "@punctuation_of" not in aux_labels:
        return list(nx.topological_sort(hierarchy_graph))

    restore_order = _mixed_aux_restore_order(parent, children, aux_edge_labels_by_child)
    restore_rank = {node.ID: index for index, node in enumerate(restore_order)}

    return list(
        nx.lexicographical_topological_sort(
            hierarchy_graph,
            key=lambda node_id: (-restore_rank.get(node_id, len(restore_rank)), str(node_id)),
        )
    )


def choose_primary_root(lingua_graph: GPGraph, roots):
    """Pick the root to preserve when a decoded graph has multiple roots."""
    root_scores = []
    for node in roots:
        try:
            descendant_count = len(nx.descendants(lingua_graph.g, node.ID))
        except Exception:
            descendant_count = 0
        is_sentence_punctuation_root = int(_is_punctuation_shorthand_root(lingua_graph, node))
        has_as_shorthand_outgoing = int(_has_as_shorthand_outgoing(lingua_graph, node))
        is_punctuation = int(getattr(node, "pos", None) == "PunctuationalConstant")
        root_scores.append(
            (
                is_sentence_punctuation_root,
                has_as_shorthand_outgoing,
                is_punctuation,
                -descendant_count,
                *_node_sort_key(lingua_graph, node),
                node,
            )
        )
    return min(root_scores)[-1]


def choose_fallback_parent(lingua_graph: GPGraph, primary_root: GPGNode, extra_roots=None):
    """Find a node in the primary component that can emit fallback edges."""
    candidate_ids = {primary_root.ID}
    candidate_ids.update(nx.descendants(lingua_graph.g, primary_root.ID))
    candidates = [lingua_graph.get_node(node_id) for node_id in candidate_ids]
    candidates = [node for node in candidates if node is not None]
    if not candidates:
        return primary_root
    if extra_roots:
        safe_candidates = [
            node
            for node in candidates
            if all(not nx.has_path(lingua_graph.g, root.ID, node.ID) for root in extra_roots)
        ]
        if safe_candidates:
            candidates = safe_candidates

    def score(node: GPGNode):
        return (
            int(_has_as_shorthand_outgoing(lingua_graph, node)),
            int(getattr(node, "pos", None) == "PunctuationalConstant"),
            *_node_sort_key(lingua_graph, node),
        )

    return min(candidates, key=score)


def ensure_single_rooted_dag(lingua_graph: GPGraph, fallback_label="nonsense"):
    """
    Ensure final converted Lingua output is a single-root DAG.

    The neural decoder can occasionally leave several roots after word-lingua
    hierarchy hints are consumed. `nonsense` is used only as a legal structural
    fallback so the invalid topology stays visible as a low-information edge.
    """
    if lingua_graph.number_of_nodes() == 0:
        return lingua_graph, False

    if not nx.is_directed_acyclic_graph(lingua_graph.g):
        raise ValueError("Converted Lingua graph is not a DAG")

    roots = [node for node in list(lingua_graph.nodes()) if lingua_graph.g.in_degree(node.ID) == 0]
    if len(roots) <= 1:
        lingua_graph.root_ = None
        return lingua_graph, False

    primary_root = choose_primary_root(lingua_graph, roots)
    extra_roots = [root for root in roots if root != primary_root]
    fallback_parent = choose_fallback_parent(lingua_graph, primary_root, extra_roots)
    repair_info = {
        "type": "multiple_roots",
        "fallback_label": fallback_label,
        "primary_root": lingua_graph.node_text(primary_root),
        "fallback_parent": lingua_graph.node_text(fallback_parent),
        "extra_roots": [],
    }

    for root in sorted(extra_roots, key=lambda n: _node_sort_key(lingua_graph, n)):
        if nx.has_path(lingua_graph.g, root.ID, fallback_parent.ID):
            raise ValueError("Converted Lingua graph has roots that would create a cycle if repaired")
        if not lingua_graph.has_edge(fallback_parent, root):
            lingua_graph.add_relation(fallback_parent, root, fallback_label)
            repair_info["extra_roots"].append(lingua_graph.node_text(root))

    lingua_graph.meta.setdefault("word2gp_repairs", []).append(repair_info)
    lingua_graph.root = primary_root

    return lingua_graph, True


def edge2children(lingua_graph: GPGraph, visiting_node: GPGNode):
    edge2children_map = defaultdict(list)
    for child, edge in lingua_graph.children(visiting_node):
        labels = tuple(split_edge_label(edge.label))
        # Compound labels are allowed (the forward transform uses "+" to preserve
        # multiple relations between the same node pair).
        # if len(labels) > 1 and any(label.startswith("as:") for label in labels[1:]):
        #     raise ValueError(f"Unsupported edge label: {edge.label}")
        edge2children_map[labels].append(child)
    # NOTE: A symbolic label component (containing "@") does NOT imply a single child.
    # `gp2word.remove_aux_nodes()` can generate multiple edges with the same symbolic label
    # from one parent to different children (e.g. multiple @modification bodies).
    return edge2children_map

def safe_remove_relation(lingua_graph: GPGraph, parent: GPGNode, child: GPGNode):
    """Remove relation if it exists"""
    if lingua_graph.has_edge(parent, child):
        lingua_graph.remove_relation(parent, child)

def safe_remove_aux_relation(lingua_graph: GPGraph, parent: GPGNode, child: GPGNode, aux_label: str):
    """Remove relation if it exists"""
    edge = lingua_graph.get_edge(parent, child)
    if edge is None:
        return
    edge_labels = split_edge_label(edge.label)
    if aux_label not in edge_labels:
        return
    edge_labels.remove(aux_label)

    confidence = edge.confidence
    lingua_graph.remove_relation(parent, child)
    if edge_labels:
        new_edge_label = "+".join(edge_labels)
        lingua_graph.add_relation(parent, child, new_edge_label, confidence=confidence)

def get_this_layer_offsprings(lingua_graph: GPGraph, target_node: GPGNode, visited=None):

    this_layer_offsprings = set()
    this_layer_offsprings.add(target_node)
    if visited is None:
        visited = set()
    visited.add(target_node)
    for child, edge in list(lingua_graph.children(target_node)):
        if "upper_" in edge.label:
            continue 
        if child in visited:
            continue
        this_layer_offsprings.update(get_this_layer_offsprings(lingua_graph, child, visited))

    return this_layer_offsprings


def restore_reversed_edges_in_this_layer(lingua_graph: GPGraph, source_node: GPGNode, target_node: GPGNode):

    this_layer_offsprings = get_this_layer_offsprings(lingua_graph, target_node)

    for child, edge in list(lingua_graph.children(source_node)):

        if child not in this_layer_offsprings:
            continue

        edge_labels = split_edge_label(edge.label)
        remained_edges = []
        reversed_edges = []
        for edge_label in edge_labels:
            if edge_label.startswith("reverse_"):
                reversed_edges.append(edge_label[len("reverse_"):])
            else:
                remained_edges.append(edge_label)
        
        safe_remove_relation(lingua_graph, source_node, child)
        if remained_edges:
            remained_edge_label = "+".join(remained_edges)
            lingua_graph.add_relation(source_node, child, remained_edge_label)
        if reversed_edges:
            reversed_edge_label = "+".join(reversed_edges)
            lingua_graph.add_relation(child, source_node, reversed_edge_label)
            yield child, reversed_edge_label


def transfer_child_of_whether(current_root: GPGNode, aux_node: GPGAuxNode) -> None:
    """Lift the in-memory Whether marker when a word-lingua node is rewrapped."""

    if hasattr(current_root, "child_of_whether") and current_root.child_of_whether:
        aux_node.child_of_whether = True
        delattr(current_root, "child_of_whether")


def _find_modification_covering(lingua_graph: GPGraph, body_source: GPGNode, variable_target: GPGNode) -> GPGNode | None:
    candidates = []
    for node in lingua_graph.nodes():
        if not isinstance(node, GPGAuxNode) or node.label != "Modification":
            continue
        children = edge2children(lingua_graph, node)
        variables = children.get(("variable",), [])
        bodies = children.get(("body",), [])
        if variable_target not in variables or not bodies:
            continue
        for body in bodies:
            if body == body_source or body_source in lingua_graph.offsprings(body):
                try:
                    distance = nx.shortest_path_length(lingua_graph.g, node.ID, body_source.ID)
                except nx.NetworkXNoPath:
                    distance = 0
                candidates.append((distance, str(node.ID), node))

    if not candidates:
        return None
    return max(candidates)[-1]


def _add_aux_node_for_symbolic_edge(
    lingua_graph: GPGraph,
    source: GPGNode,
    target: GPGNode,
    aux_label: str,
    *,
    body_override: GPGNode | None = None,
    rewire_from: GPGNode | None = None,
):
    aux_node_label = symbol_edge2nodelabel[aux_label]
    aux_node_pos = symbol_nodelabel2pos[aux_node_label]
    aux_node = GPGAuxNode(aux_node_label, pos=aux_node_pos)
    lingua_graph.add_node(aux_node)

    if aux_node_label == "Punctuation":
        lingua_graph.add_relation(aux_node, target, "pred.arg.1")
        lingua_graph.add_relation(aux_node, source, "pred.arg.2")
    elif aux_node_pos == "ModificationalFunctor":
        lingua_graph.add_relation(aux_node, target, "variable")
        lingua_graph.add_relation(aux_node, body_override or source, "body")
    elif aux_node_pos.endswith("Predicator"):
        lingua_graph.add_relation(aux_node, target, "pred.arg.1")
        lingua_graph.add_relation(aux_node, source, "pred.arg.2")
    elif aux_node_label in {"Parataxis", "List", "Expression"}:
        lingua_graph.add_relation(aux_node, source, "func.arg")
        lingua_graph.add_relation(aux_node, target, "func.arg")
    else:
        raise Exception(f"Unsupported residual aux edge: {aux_label}")

    rewire_node = rewire_from or source
    for parent, edge in list(lingua_graph.parents(rewire_node)):
        if parent == aux_node:
            continue
        confidence = edge.confidence
        label = edge.label
        lingua_graph.remove_relation(parent, rewire_node)
        lingua_graph.add_relation(parent, aux_node, label, confidence=confidence)

    return aux_node


def restore_remaining_aux_edges(lingua_graph: GPGraph):
    """Consume residual symbolic word-lingua edges after aux-chain recovery."""

    updated = False
    for source, edge, target in list(lingua_graph.edges()):
        edge_labels = split_edge_label(edge.label)
        symbolic_labels = [label for label in edge_labels if label in symbol_edge2nodelabel]
        if not symbolic_labels:
            continue

        remaining_labels = list(edge_labels)
        for aux_label in symbolic_labels:
            body_override = None
            rewire_from = None
            if aux_label == "@modification_of":
                covering = _find_modification_covering(lingua_graph, source, target)
                if covering is not None:
                    body_override = covering
                    rewire_from = covering

            _add_aux_node_for_symbolic_edge(
                lingua_graph,
                source,
                target,
                aux_label,
                body_override=body_override,
                rewire_from=rewire_from,
            )
            remaining_labels.remove(aux_label)
            updated = True

        confidence = edge.confidence
        if not lingua_graph.has_edge(source, target):
            continue
        lingua_graph.remove_relation(source, target)
        if remaining_labels:
            lingua_graph.add_relation(source, target, "+".join(remaining_labels), confidence=confidence)

    return lingua_graph, updated


def _single_child_for_label(lingua_graph: GPGraph, node: GPGNode, label: str) -> GPGNode | None:
    children = edge2children(lingua_graph, node).get((label,), [])
    if len(children) != 1:
        return None
    return children[0]


def restore_whether_scope_around_aux_nodes(lingua_graph: GPGraph):
    updated = False

    for whether_node in list(lingua_graph.nodes()):
        if not isinstance(whether_node, GPGAuxNode) or whether_node.label != "Whether":
            continue

        func_arg = _single_child_for_label(lingua_graph, whether_node, "func.arg")
        if not isinstance(func_arg, GPGAuxNode):
            continue

        if func_arg.label in {"Attribute", "Copula", "Apposition"}:
            pred_arg1 = _single_child_for_label(lingua_graph, func_arg, "pred.arg.1")
            pred_arg2 = _single_child_for_label(lingua_graph, func_arg, "pred.arg.2")
            if (
                pred_arg2 is None
                or not isinstance(pred_arg1, GPGAuxNode)
                or pred_arg1.label != "Punctuation"
            ):
                continue

            parents = list(lingua_graph.parents(whether_node))
            if not any(
                edge.label == "pred.arg.1"
                and isinstance(parent, GPGAuxNode)
                and parent.label == "Punctuation"
                for parent, edge in parents
            ):
                continue

            for parent, edge in parents:
                confidence = edge.confidence
                label = edge.label
                lingua_graph.remove_relation(parent, whether_node)
                lingua_graph.add_relation(parent, func_arg, label, confidence=confidence)

            whether_edge = lingua_graph.get_edge(whether_node, func_arg)
            pred_arg2_edge = lingua_graph.get_edge(func_arg, pred_arg2)
            if whether_edge is None or pred_arg2_edge is None:
                continue

            whether_confidence = whether_edge.confidence
            pred_arg2_confidence = pred_arg2_edge.confidence
            lingua_graph.remove_relation(whether_node, func_arg)
            lingua_graph.remove_relation(func_arg, pred_arg2)
            lingua_graph.add_relation(func_arg, whether_node, "pred.arg.2", confidence=pred_arg2_confidence)
            lingua_graph.add_relation(whether_node, pred_arg2, "func.arg", confidence=whether_confidence)
            updated = True

        elif func_arg.label == "Discourse":
            discourse_body = _single_child_for_label(lingua_graph, func_arg, "body")
            if discourse_body is None:
                continue

            parents = list(lingua_graph.parents(whether_node))
            vocative_parents = [
                parent
                for parent, edge in parents
                if edge.label == "variable"
                and isinstance(parent, GPGAuxNode)
                and parent.label == "Vocative"
            ]
            if not vocative_parents:
                continue

            for parent, edge in parents:
                confidence = edge.confidence
                label = edge.label
                lingua_graph.remove_relation(parent, whether_node)
                if parent in vocative_parents:
                    lingua_graph.add_relation(parent, discourse_body, label, confidence=confidence)
                else:
                    lingua_graph.add_relation(parent, func_arg, label, confidence=confidence)

            whether_edge = lingua_graph.get_edge(whether_node, func_arg)
            body_edge = lingua_graph.get_edge(func_arg, discourse_body)
            if whether_edge is None or body_edge is None:
                continue

            whether_confidence = whether_edge.confidence
            body_confidence = body_edge.confidence
            lingua_graph.remove_relation(whether_node, func_arg)
            lingua_graph.remove_relation(func_arg, discourse_body)
            lingua_graph.add_relation(func_arg, whether_node, "body", confidence=body_confidence)
            lingua_graph.add_relation(whether_node, discourse_body, "func.arg", confidence=whether_confidence)
            updated = True

    return lingua_graph, updated


def restore_vocative_variable_from_discourse_body(lingua_graph: GPGraph):
    updated = False

    for vocative_node in list(lingua_graph.nodes()):
        if not isinstance(vocative_node, GPGAuxNode) or vocative_node.label != "Vocative":
            continue

        children = edge2children(lingua_graph, vocative_node)
        variables = children.get(("variable",), [])
        if len(variables) != 1:
            continue

        discourse_node = variables[0]
        if not isinstance(discourse_node, GPGAuxNode) or discourse_node.label != "Discourse":
            continue

        discourse_children = edge2children(lingua_graph, discourse_node)
        bodies = discourse_children.get(("body",), [])
        if len(bodies) != 1:
            continue

        variable_target = bodies[0]
        if not isinstance(variable_target, GPGAuxNode) or variable_target.label != "Whether":
            continue
        whether_children = edge2children(lingua_graph, variable_target)
        func_args = whether_children.get(("func.arg",), [])
        if len(func_args) != 1:
            continue
        variable_target = func_args[0]

        variable_edge = lingua_graph.get_edge(vocative_node, discourse_node)
        if variable_edge is None:
            continue

        confidence = variable_edge.confidence
        lingua_graph.remove_relation(vocative_node, discourse_node)
        lingua_graph.add_relation(vocative_node, variable_target, "variable", confidence=confidence)
        updated = True

    return lingua_graph, updated


symbol_nodelabel2edge = {
    "Apposition":"@appositive_of",
    "Attribute":"@attribute_of",
    "Copula":"@copula",
    "Discourse":"@discourse_of",
    "Expression":"@expression",
    "List":"@list",
    "Modification":"@modification_of",
    "Parataxis":"@parataxis",
    "Punctuation":"@punctuation_of",
    "Ref":"@ref_of",
    "Vocative":"@vocative_of",
    "Whether":"@whether_of",
}
    

symbol_edge2nodelabel = {edge: label for label, edge in symbol_nodelabel2edge.items()}


symbol_nodelabel2pos = {
    "Apposition": "AppositionalPredicator",
    "Attribute": "AttributePredicator",
    "Copula": "FactualPredicator",
    "Discourse": "ModificationalFunctor",
    "Expression": "ExpressionFunctor",
    "List": "ListFunctor",
    "Modification": "ModificationalFunctor",
    "Parataxis": "ConjunctionalFunctor",
    "Punctuation": "PunctuationalConstant",
    "Ref": "ReferentialPredicator",
    "Vocative": "ModificationalFunctor",
    "Whether": "GeneralFunctor",
}


def restore_whether_aux(lingua_graph: GPGraph, debug=False):
    """Restore Whether auxiliary nodes based on child_of_whether attribute"""
    updated = False
    
    for node in list(lingua_graph.nodes()):
        if hasattr(node, "child_of_whether") and node.child_of_whether:
            # Create Whether aux node
            whether_node = GPGAuxNode(label="Whether", pos="GeneralFunctor")
            lingua_graph.add_node(whether_node)
            
            # Move all parents of the func_child to whether_node
            for parent, edge in list(lingua_graph.parents(node)):
                safe_remove_relation(lingua_graph, parent, node)
                lingua_graph.add_relation(parent, whether_node, edge.label)
            
            # Connect whether_node to func_child
            lingua_graph.add_relation(whether_node, node, "func.arg")
            
            # Remove the attribute
            delattr(node, "child_of_whether")
            updated = True
    
    return lingua_graph, updated

def find_aux_chain(lingua_graph: GPGraph, target_node: GPGNode):

    # print(f"find_aux_chain: {lingua_graph.node_text(target_node)}")
    aux_chain = []

    if not hasattr(target_node, "aux_hierarchy") or not target_node.aux_hierarchy:
        return aux_chain

    for aux_node_child_id in target_node.aux_hierarchy:
        aux_child_node = lingua_graph.get_node(aux_node_child_id)
        assert aux_child_node is not None 
        
        edge = lingua_graph.get_edge(target_node, aux_child_node)
        edge_labels = split_edge_label(edge.label)

        assert edge_labels[0].startswith("@")

        aux_edge_label = edge_labels[0]

        if aux_edge_label in {"@parataxis", "@list", "@expression"}:
            aux_child_nodes = [aux_child_node] 
            current_node = aux_child_node
            while True:
                if not hasattr(current_node, "aux_hierarchy") or not current_node.aux_hierarchy:
                    break

                next_elem_node = lingua_graph.get_node(current_node.aux_hierarchy[0])
                assert next_elem_node is not None 

                this_edge = lingua_graph.get_edge(current_node, next_elem_node)
                this_edge_labels = split_edge_label(this_edge.label)
                if this_edge_labels[0] == aux_edge_label:
                    nex_elem_node_parents = set([n for n, e in  lingua_graph.parents(next_elem_node)])
                    current_node_parents = set([n for n, e in lingua_graph.parents(current_node)])
                    intersection = nex_elem_node_parents.intersection(current_node_parents)
                    if intersection:
                        intersect_node = intersection.pop() 
                        if "@lower" in  lingua_graph.get_edge(intersect_node, next_elem_node).label:
                            safe_remove_aux_relation(lingua_graph, intersect_node, next_elem_node, "@lower")
                            break
                    aux_child_nodes.append(next_elem_node)
                    current_node.aux_hierarchy.pop(0)
                    current_node = next_elem_node
                else:
                    break

            aux_chain.append({"aux_node_label": symbol_edge2nodelabel[aux_edge_label], "nodes": aux_child_nodes})

        else:
            aux_chain.append({"aux_node_label": symbol_edge2nodelabel[aux_edge_label], "nodes": [aux_child_node]})

    delattr(target_node, "aux_hierarchy")

    return list(reversed(aux_chain))

# def clean_lower_tag(lingua_graph: GPGraph, visiting_node: GPGNode):
#     has_parent_at_edge = False 
#     for parent, edge in list(lingua_graph.parents(visiting_node)):
#         if "@" in edge.label:
#             has_parent_at_edge = True
#             break
#     if has_parent_at_edge:
#         return

#     edge2children_map = edge2children(lingua_graph, visiting_node)
#     for edge_labels, children in edge2children_map.items():
#         new_edge_labels = []
#         for edge_label in edge_labels:
#             if edge_label.startswith("lower_"):
#                 edge_label = edge_label[len("lower_"):]
#             new_edge_labels.append(edge_label)

#         new_edge_label = "+".join(new_edge_labels)

#         for child in children:
#             lingua_graph.remove_relation(visiting_node, child)
#             lingua_graph.add_relation(visiting_node, child, new_edge_label)


def lower_edge_to_node_hierarchy(lingua_graph: GPGraph, debug=False):

    for current_node in list(lingua_graph.nodes()):
        
        edge2children_map = edge2children(lingua_graph, current_node)

        aux_children = []

        for edge_labels, children in edge2children_map.items():
            if any(edge_label in symbol_edge2nodelabel for edge_label in edge_labels):
                aux_children.extend(children)

        if not aux_children:
            continue
        elif len(aux_children) == 1:
            current_node.aux_hierarchy = [aux_children[0].ID]
        else:
            aux_edge_labels_by_child = {
                child.ID: split_edge_label(lingua_graph.get_edge(current_node, child).label)
                for child in aux_children
            }
            sub_graph = lingua_graph.subgraph(aux_children)
            hierarchy_graph = nx.DiGraph()
            hierarchy_graph.add_nodes_from(child.ID for child in aux_children)
            # Convert to list before iterating to avoid 'dictionary changed size during iteration'
            for start, edge, end in list(sub_graph.edges()):
                edge_labels = split_edge_label(edge.label)
                if "@lower" in edge_labels:
                    hierarchy_graph.add_edge(start.ID, end.ID)
                    remaining_edge_label = "+".join([edge_label for edge_label in edge_labels if edge_label != "@lower"])

                    confidence = edge.confidence
                    lingua_graph.remove_relation(start, end)
                    if remaining_edge_label:
                        lingua_graph.add_relation(start, end, remaining_edge_label, confidence=confidence)
                elif "@upper" in edge_labels:
                    hierarchy_graph.add_edge(end.ID, start.ID)
                    remaining_edge_label = "+".join([edge_label for edge_label in edge_labels if edge_label != "@upper"])
                    confidence = edge.confidence
                    lingua_graph.remove_relation(start, end)
                    if remaining_edge_label:
                        lingua_graph.add_relation(start, end, remaining_edge_label, confidence=confidence)
                else:
                    raise Exception(f"Unsupported edge label: {edge.label}")

            if hierarchy_graph.number_of_edges() == 0:
                aux_labels = {
                    label
                    for labels in aux_edge_labels_by_child.values()
                    for label in labels
                    if label in symbol_edge2nodelabel
                }
                if aux_labels == {"@punctuation_of"}:
                    restore_order = _punctuation_restore_order(current_node, aux_children)
                    current_node.aux_hierarchy = [node.ID for node in reversed(restore_order)]
                elif "@punctuation_of" in aux_labels:
                    restore_order = _mixed_aux_restore_order(
                        current_node,
                        aux_children,
                        aux_edge_labels_by_child,
                    )
                    current_node.aux_hierarchy = [node.ID for node in reversed(restore_order)]
                else:
                    current_node.aux_hierarchy = list(nx.topological_sort(hierarchy_graph))
            else:
                current_node.aux_hierarchy = _aux_hierarchy_topological_order(
                    hierarchy_graph,
                    current_node,
                    aux_children,
                    aux_edge_labels_by_child,
                )

        if debug:
            logger.debug(
                "lower_edge_to_node_hierarchy: %s - %s",
                lingua_graph.node_text(current_node),
                current_node.aux_hierarchy,
            )


def restore_aux_nodes(lingua_graph: GPGraph, debug=False):
    """
    Restore all auxiliary nodes removed by `gp2word.remove_aux_nodes()`.

    Forward pass (`remove_aux_nodes`) encodes aux structure using symbolic edge components:
      - "@..." (and sometimes "upper_@..." / "lower_@...")
    plus possible cycle-breaking "reverse_" prefixes (handled separately by
    `normalize_reverse_components()`).

    This function is the *single* inverse step for that forward transform. Internally it
    restores:
      - list-like functors: Expression/List/Parataxis
      - single-layer functors/predicators: Modification/Apposition/Attribute/Copula/Discourse/Ref/Vocative
    """
    

    lower_edge_to_node_hierarchy(lingua_graph, debug=debug)

    updated = False
    update_idx = 0
    # Convert to list before iterating to avoid 'dictionary changed size during iteration'
    for visiting_node in list(lingua_graph.topological_sort()):

        aux_chain = find_aux_chain(lingua_graph, visiting_node)

        if not aux_chain:
            continue 

        
        if debug:
            logger.debug("restore_aux_nodes: %s - %s", lingua_graph.node_text(visiting_node), aux_chain)

        previous_tail_node = None 
        current_root = visiting_node
        for idx, layer in enumerate(aux_chain):

            parents_of_current_root = list(n for n, _ in lingua_graph.parents(current_root))

            aux_node_label = layer["aux_node_label"]
            nodes = layer["nodes"]
            aux_node_pos = symbol_nodelabel2pos[aux_node_label]
            aux_node = GPGAuxNode(aux_node_label, pos=aux_node_pos)

            lingua_graph.add_node(aux_node)

            if aux_node_pos == "ModificationalFunctor":
                assert len(nodes) == 1
                body = current_root
                variable = nodes[0]
                lingua_graph.add_relation(aux_node, variable, "variable")
                lingua_graph.add_relation(aux_node, body, "body")
                safe_remove_aux_relation(lingua_graph, visiting_node, variable, symbol_nodelabel2edge[aux_node_label])
                for new_parent, new_edge_label in restore_reversed_edges_in_this_layer(lingua_graph, body, variable):
                    parents_of_current_root.append(new_parent) 
                    # lingua_graph.add_relation(new_parent, aux_node, new_edge_label)
                
            elif aux_node_pos.endswith("Predicator"):
                assert len(nodes) == 1
                arg2 = current_root
                arg1 = nodes[0]
                lingua_graph.add_relation(aux_node, arg1, "pred.arg.1")
                lingua_graph.add_relation(aux_node, arg2, "pred.arg.2")
                safe_remove_aux_relation(lingua_graph, visiting_node, arg1, symbol_nodelabel2edge[aux_node_label])
                for new_parent, new_edge_label in restore_reversed_edges_in_this_layer(lingua_graph, arg2, arg1):
                    parents_of_current_root.append(new_parent) 
                    #lingua_graph.add_relation(new_parent, aux_node, new_edge_label)
            elif aux_node_label == "Punctuation":
                assert len(nodes) == 1
                punct = current_root
                child = nodes[0]
                lingua_graph.add_relation(aux_node, child, "pred.arg.1")
                lingua_graph.add_relation(aux_node, punct, "pred.arg.2")
                safe_remove_aux_relation(lingua_graph, visiting_node, child, symbol_nodelabel2edge[aux_node_label])
                for new_parent, new_edge_label in restore_reversed_edges_in_this_layer(lingua_graph, punct, child):
                    parents_of_current_root.append(new_parent)
                
            elif aux_node_label in {"Parataxis", "List", "Expression"}:

                if debug:
                    logger.debug("processing: %s", lingua_graph.node_text(visiting_node))
                    logger.debug(
                        "previous_tail_node: %s",
                        lingua_graph.node_text(previous_tail_node) if previous_tail_node is not None else None,
                    )
                children = [current_root] + nodes
                for node in children:
                    if debug:
                        logger.debug("child: %s", lingua_graph.node_text(node))
    
                    lingua_graph.add_relation(aux_node, node, "func.arg")

                safe_remove_aux_relation(lingua_graph, visiting_node, nodes[0], symbol_nodelabel2edge[aux_node_label])
                    
                for prev, next in itertools.pairwise(children):
                    if debug:
                        logger.debug(
                            "remove_aux_relation: %s -[%s]> %s",
                            lingua_graph.node_text(prev),
                            symbol_nodelabel2edge[aux_node_label],
                            lingua_graph.node_text(next),
                        )
                    safe_remove_aux_relation(lingua_graph, prev, next, symbol_nodelabel2edge[aux_node_label])
                    for new_parent, new_edge_label in restore_reversed_edges_in_this_layer(lingua_graph, prev, next):
                        parents_of_current_root.append(new_parent) 
                        # lingua_graph.add_relation(new_parent, aux_node, new_edge_label)
                
            else:
                raise Exception(f"Unsupported aux node: label {aux_node_label}, pos {aux_node_pos}")

            for child, edge in list(lingua_graph.children(current_root)):
                edge_labels = split_edge_label(edge.label)
                if not(any(edge_label.startswith("reverse_") for edge_label in edge_labels)):
                    continue
                remain_edges =[]
                reversed_edges =[]
                for edge_label in edge_labels:
                    if edge_label.startswith("reverse_"):
                        reversed_edges.append(edge_label)
                    else:
                        remain_edges.append(edge_label)
                
                lingua_graph.remove_relation(current_root, child)
                if remain_edges:
                    remain_edge_label = "+".join(remain_edges)
                    lingua_graph.add_relation(current_root, child, remain_edge_label)
                if reversed_edges:
                    reversed_edge_label = "+".join(reversed_edges)
                    lingua_graph.add_relation(aux_node, child, reversed_edge_label)   

            
            this_layer_offsprings = get_this_layer_offsprings(lingua_graph, aux_node)
            if debug:
                logger.debug("aux_node: %s", lingua_graph.node_text(aux_node))
                logger.debug(
                    "this_layer_offsprings: %s",
                    [lingua_graph.node_text(n) for n in this_layer_offsprings],
                )
            for parent in parents_of_current_root:
                if parent in this_layer_offsprings:
                    continue 

                edge = lingua_graph.get_edge(parent, current_root)
                edge_labels = split_edge_label(edge.label)
                
                lingua_graph.remove_relation(parent, current_root)
                carry_duplicate_parent_edge = (
                    idx < len(aux_chain) - 1
                    and len(edge_labels) != len(set(edge_labels))
                )
                if carry_duplicate_parent_edge:
                    lingua_graph.add_relation(parent, aux_node, edge.label)
                    continue

                lingua_graph.add_relation(parent, aux_node, edge_labels[0])
                remaining_edge = "+".join(edge_labels[1:])
                if remaining_edge:
                    lingua_graph.add_relation(parent, current_root, remaining_edge)

            transfer_child_of_whether(current_root, aux_node)
            current_root = aux_node

            # visualizer = GPGraphVisualizer()
            # visualizer.visualize(lingua_graph, file_name=f"{visiting_node.ID}.{idx}.png", format="png")

        
        updated = True 


    return lingua_graph, updated


def collapse_multi_word_nodes(lingua_graph: GPGraph, debug=False):
    """Collapse multiple word nodes connected by next_word back into multi-word nodes"""
    updated = False
    
    for node in list(lingua_graph.nodes()):
        if not lingua_graph.has_node(node):
            continue
        
        # Check if this is the start of a word chain
        edge2children_map = edge2children(lingua_graph, node)
        if ("next_word",) not in edge2children_map:
            continue
        
        # Collect all words in chain
        word_chain = [node]
        assert len(edge2children_map[("next_word",)]) == 1, "should have exactly one next_word"
        current = edge2children_map[("next_word",)][0]
        
        while current:
            word_chain.append(current)
            current_children = edge2children(lingua_graph, current)
            if ("next_word",) in current_children:
                assert len(current_children[("next_word",)]) == 1, "should have exactly one next_word"
                current = current_children[("next_word",)][0]
            else:
                break
        
        # Merge into multi-word node
        if len(word_chain) > 1:
            positions = list()
            for word_node in word_chain:
                positions.extend(word_node.words(with_aux=True))
            
            # Create new multi-word node
            multi_node = GPGPhraseNode(spans=positions2spans(positions), pos=node.pos)
            if hasattr(node, "child_of_whether"):
                multi_node.child_of_whether = node.child_of_whether
            
            # Add to graph
            lingua_graph.add_node(multi_node)
            
            # Replace first node with multi_node
            lingua_graph.replace(node, multi_node)
            
            # Remove chain nodes and edges
            # for i in range(len(word_chain) - 1):
            #     lingua_graph.remove_edge_between(word_chain[i], word_chain[i+1])
            
            for word_node in word_chain[1:]:
                if lingua_graph.has_node(word_node):
                    lingua_graph.remove_node(word_node)
            
            updated = True
    
    return lingua_graph, updated


import copy
def wordlingua2lingua(word_lingua_graph: GPGraph, debug=False):
    """
    Convert word-lingua back to lingua format.
    This reverses the transformations done in `gp2word.lingua2wordlingua()`.
    
    Args:
        word_lingua_graph: The word-lingua format graph
        debug: If True, returns debug information
    
    Returns:
        tuple: (converted graph, updated flag, debug_info dict)
    """
    lingua_graph = word_lingua_graph
    
    if debug:
        debug_info = {}

    try:
        updated_any = False
        # Reverse the transformations in opposite order of gp2word.lingua2wordlingua():
        #
        #   gp2word:
        #     1) remove_whether_aux
        #     2) remove_aux_nodes   (may also introduce reverse_* components when breaking cycles)
        #
        # So here:
        #   0) normalize_reverse_components (undo reverse_* artifacts)
        #   1) restore_aux_nodes            (inverse of remove_aux_nodes)
        #   2) restore_whether_aux          (inverse of remove_whether_aux)

        # 0) Normalize reverse_* components introduced by cycle-breaking in gp2word.py

        # 1) Restore all aux nodes removed by gp2word.remove_aux_nodes()
        lingua_graph, updated = restore_aux_nodes(lingua_graph, debug=debug)
        updated_any = updated_any or updated
        if debug and updated:
            debug_info["restore_aux_nodes"] = copy.deepcopy(lingua_graph)

        # 2) Restore Whether auxiliary nodes
        lingua_graph, updated = restore_whether_aux(lingua_graph, debug=debug)
        updated_any = updated_any or updated
        if debug and updated:
            debug_info["restore_whether_aux"] = copy.deepcopy(lingua_graph)

        lingua_graph, updated = collapse_multi_word_nodes(lingua_graph, debug=debug)
        updated_any = updated_any or updated
        if debug and updated:
            debug_info["collapse_multi_word_nodes"] = copy.deepcopy(lingua_graph)

        lingua_graph, updated = restore_whether_scope_around_aux_nodes(lingua_graph)
        updated_any = updated_any or updated
        if debug and updated:
            debug_info["restore_whether_scope_around_aux_nodes"] = copy.deepcopy(lingua_graph)

        lingua_graph, updated = restore_vocative_variable_from_discourse_body(lingua_graph)
        updated_any = updated_any or updated
        if debug and updated:
            debug_info["restore_vocative_variable_from_discourse_body"] = copy.deepcopy(lingua_graph)

        lingua_graph, updated = restore_remaining_aux_edges(lingua_graph)
        updated_any = updated_any or updated
        if debug and updated:
            debug_info["restore_remaining_aux_edges"] = copy.deepcopy(lingua_graph)

        # Final Lingua graphs must not expose word-lingua hierarchy hints.
        lingua_graph, updated = strip_hierarchy_edge_components(lingua_graph)
        updated_any = updated_any or updated
        if debug and updated:
            debug_info["strip_hierarchy_edge_components"] = copy.deepcopy(lingua_graph)

        lingua_graph, updated = ensure_single_rooted_dag(lingua_graph)
        updated_any = updated_any or updated
        if debug and updated:
            debug_info["ensure_single_rooted_dag"] = copy.deepcopy(lingua_graph)
    
    except Exception as e:
        if debug:
            e.debug_info = debug_info
        raise e
    
    if debug:
        return lingua_graph, updated_any, debug_info
    else:
        return lingua_graph, updated_any, {}
