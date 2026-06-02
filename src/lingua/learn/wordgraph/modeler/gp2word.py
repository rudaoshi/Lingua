from graphlib import TopologicalSorter
from pydoc import resolve
from lingua.structure.gpgraph import GPGraph, GPGPhraseNode, GPGAuxNode, GPGEdge, GPGNode
from collections import defaultdict
import itertools
import logging
import copy
logger = logging.getLogger(__name__)

from lingua.utils.topology_sorter import LinguaGraphTopologySorter



symbol_nodelabel2edge = {
    "Apposition":"@appositive_of",
    "Attribute":"@attribute_of",
    "Copula":"@copula",
    "Discourse":"@discourse_of",
    "Expression":"@expression",
    "List":"@list",
    "Modification":"@modification_of",
    "Parataxis":"@parataxis",
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
    "Ref": "ReferentialPredicator",
    "Vocative": "ModificationalFunctor",
    "Whether": "GeneralFunctor",
}



def edge2parents(lingua_graph: GPGraph, node: GPGNode):
    label2nodes = defaultdict(list)
    for parent, edge in lingua_graph.parents(node):
        label2nodes[edge.label].append(parent)
    return label2nodes

def edge2children(lingua_graph: GPGraph, node: GPGNode):
    label2nodes = defaultdict(list)
    for child, edge in lingua_graph.children(node):
        label2nodes[edge.label].append(child)
    return label2nodes


def add_relation(parent, child, edge_label, lingua_graph: GPGraph):
    existing_edge = lingua_graph.get_edge(parent, child)
    if existing_edge:
        new_edge_label = edge_label + "+" + existing_edge.label
        #logger.warning(f"Adding compound relation {new_edge_label}")
    else:
        new_edge_label = edge_label
    lingua_graph.add_relation(parent, child, new_edge_label)
    return new_edge_label


def add_hierarchy_relation(parent, child, edge_label, lingua_graph: GPGraph):
    if parent == child:
        return None
    return add_relation(parent, child, edge_label, lingua_graph)

def symbolic_level_prefix(lingua_graph: GPGraph, node: GPGNode):

    var_edge2_children = edge2children(lingua_graph, node)
    symbolic_level = max((label.count("@") for label in var_edge2_children.keys()), default=0)
    prefix = "@" * symbolic_level

    return prefix

def remove_whether_aux(lingua_graph: GPGraph, ordered_nodes: list, debug=False):

    updated = False
    for visiting_node in reversed(ordered_nodes):

        # if debug:
            # print(f"visiting_node: {visiting_node.ID}, text: {lingua_graph.node_text(visiting_node)}")
        if isinstance(visiting_node, GPGPhraseNode):
            continue

        if not lingua_graph.has_node(visiting_node):
            continue 

        if visiting_node.label == "Whether":
            edge2children_map = edge2children(lingua_graph, visiting_node)
            if "func.arg" not in edge2children_map:
                continue
            func_child = edge2children_map["func.arg"][0]

            for parent, edge in list(lingua_graph.parents(visiting_node)):
                lingua_graph.remove_relation(parent, visiting_node)
                add_relation(parent, func_child, edge.label, lingua_graph)

            lingua_graph.remove_node(visiting_node)
            func_child.child_of_whether = True
            updated = True

    return lingua_graph, updated


def get_head_and_tail_node(lingua_graph: GPGraph, target_node: GPGAuxNode, sorted_nodes: list):

    assert isinstance(target_node, GPGAuxNode), "target node should be an aux node"

    edge2children_map = edge2children(lingua_graph, target_node)
    if "pred.arg.1" in edge2children_map:
        head_node = edge2children_map["pred.arg.1"][0]
        tail_node = edge2children_map["pred.arg.2"][0]
    elif "variable" in edge2children_map and "body" in edge2children_map:
        head_node = edge2children_map["variable"][0]
        tail_node = edge2children_map["body"][0]
    elif "func.arg" in edge2children_map:
        sorted_fun_arg_nodes = [node for node in sorted_nodes if node in edge2children_map["func.arg"]]
        head_node = sorted_fun_arg_nodes[0]
        tail_node = sorted_fun_arg_nodes[1:]
    else:
        raise Exception(f"No head node and other node found. node:{target_node.label} edges: {edge2children_map.keys()}")
    return head_node, tail_node


def get_upper_layer_aux(lingua_graph: GPGraph, target_node: GPGNode, sorted_nodes: list, filter_func=None):

    edge2parents_map = edge2parents(lingua_graph, target_node)

    upper_aux_nodes = []
    if "pred.arg.1" in edge2parents_map:
        upper_aux_nodes.extend([node for node in edge2parents_map["pred.arg.1"] if isinstance(node, GPGAuxNode)])
    if "variable" in edge2parents_map:
        upper_aux_nodes.extend([node for node in edge2parents_map["variable"] if isinstance(node, GPGAuxNode)])
    if "func.arg" in edge2parents_map:
        for parent in edge2parents_map["func.arg"]:
            if isinstance(parent, GPGPhraseNode):
                continue 
            head, tail_node = get_head_and_tail_node(lingua_graph, parent, sorted_nodes)
            if head == target_node:
                upper_aux_nodes.append(parent)

    if filter_func is not None:
        upper_aux_nodes = [node for node in upper_aux_nodes if filter_func(node)]

    # Shared structures can be referenced by multiple upper aux nodes, e.g. a
    # name Expression used by both Ref and Modification. Callers only need to
    # know whether any upper layer exists so the lower aux can be processed later.
    if len(upper_aux_nodes) >= 1:
        return upper_aux_nodes[0]
    else:
        return None

def get_lower_layer_aux(lingua_graph: GPGraph, target_node: GPGNode, sorted_nodes: list):
    edge2children_map = edge2children(lingua_graph, target_node)
    lower_aux_nodes = []
    if "pred.arg.1" in edge2children_map:
        lower_aux_nodes.extend([node for node in edge2children_map["pred.arg.1"] if isinstance(node, GPGAuxNode)])
    if "variable" in edge2children_map:
        lower_aux_nodes.extend([node for node in edge2children_map["variable"] if isinstance(node, GPGAuxNode)])
    if "func.arg" in edge2children_map:
        head, tail_node = get_head_and_tail_node(lingua_graph, target_node, sorted_nodes)
        if isinstance(head, GPGAuxNode):
            lower_aux_nodes.append(head)
        
    assert len(lower_aux_nodes) <= 1, "lower aux nodes should be at most one"
    if len(lower_aux_nodes) == 1:
        return lower_aux_nodes[0]
    else:
        return None

def get_mod_aux_chain_layer(lingua_graph: GPGraph, target_node: GPGNode):
    
    root = target_node
    while True:
        edge2children_map = edge2children(lingua_graph, root)
        aux_chain_child = [edge2children_map[label] for label in edge2children_map.keys() if label.startswith("@modification_of")]
        assert len(aux_chain_child) <= 1, "aux chain number should be exactly one"
        if len(aux_chain_child) == 0:
            break 

        aux_chain_child = aux_chain_child[0]
        assert len(aux_chain_child) == 1, "aux chain child should be exactly one"
        root = aux_chain_child[0]

    return root 
        


def find_aux_chain(lingua_graph: GPGraph, aux_chain_root: GPGAuxNode, 
                sorted_nodes: list, filter_func=None):

    aux_chain = []
    current_node = aux_chain_root

    if filter_func is not None and not filter_func(current_node):
        return aux_chain

    while True:

        edge2children_map = edge2children(lingua_graph, current_node)
        if "pred.arg.1" in edge2children_map:
            aux_head = [node for node in edge2children_map["pred.arg.1"]]
            aux_tail = [node for node in edge2children_map["pred.arg.2"]]
        elif "variable" in edge2children_map:
            aux_head = [node for node in edge2children_map["variable"]]
            aux_tail = [node for node in edge2children_map["body"]]
        elif "func.arg" in edge2children_map:
            aux_head, aux_tail = get_head_and_tail_node(lingua_graph, current_node, sorted_nodes)
            aux_head = [aux_head]
        else:
            break 
        
        if len(aux_head) == 0:
            break
        elif len(aux_head) == 1:

            aux_chain.append({"aux_node": current_node, "head": aux_head[0], "tail": aux_tail})
            aux_head = aux_head[0]
            if isinstance(aux_head, GPGAuxNode):
                current_node = aux_head
            else:
                break 

            if filter_func is not None and not filter_func(current_node):
                break 
            
        else:
            raise Exception(f"Multiple aux parents found for node {current_node.ID}")
    return aux_chain

def find_ambiguous_hierarchy(lingua_graph: GPGraph, target_node: GPGNode):

    for child, edge in lingua_graph.children(target_node):
        edge_labels = edge.label.split("+")
        if any(edge_label in symbol_edge2nodelabel for edge_label in edge_labels):
            yield child, edge 

def lower_edge_level(lingua_graph: GPGraph, target_node: GPGNode):
    edge2children_map = edge2children(lingua_graph, target_node)
    for label in edge2children_map.keys():
        if "@" not in label:
            continue 

        # assert not (label.startswith("lower_") or label.endswith("upper_")), "lower_ or upper_ label should not be used"
        new_label = "lower_" + label
        for child in edge2children_map[label]:
            lingua_graph.remove_relation(target_node, child)
            lingua_graph.add_relation(target_node, child, new_label)

def resolve_cycle_before_add_edge(lingua_graph: GPGraph, source_node: GPGNode, target_node: GPGNode):


    target_offsprings = set(lingua_graph.offsprings(target_node))
    if source_node in target_offsprings:
        for parent, edge in list(lingua_graph.parents(source_node)):
            if parent in target_offsprings:
                lingua_graph.remove_relation(parent, source_node)
                add_relation(source_node, parent, "reverse_" + edge.label, lingua_graph)


def move_children_to_new_node(lingua_graph: GPGraph, old_node: GPGNode, new_node: GPGNode, debug=False):

    if debug:
        debug_info = []

    for child, edge in list(lingua_graph.children(old_node)):
        
        edge_labels = edge.label.split("+")

        edges_to_move = "+".join([edge for edge in edge_labels if "@" in edge])
        edges_to_keep = "+".join([edge for edge in edge_labels if not "@" in edge])

        lingua_graph.remove_relation(old_node, child)

        if edges_to_move:
            new_edge_label = add_relation(new_node, child, edges_to_move, lingua_graph)
            if debug:
                debug_info.append({
                    "from": lingua_graph.node_text(new_node),
                    "to": lingua_graph.node_text(child),
                    "label": new_edge_label
                })
        
        if edges_to_keep:
            new_edge_label = add_relation(old_node, child, edges_to_keep, lingua_graph)
            if debug:
                debug_info.append({
                    "from": lingua_graph.node_text(old_node),
                    "to": lingua_graph.node_text(child),
                    "label": new_edge_label
                })

    if debug:
        return lingua_graph, debug_info
    else:
        return lingua_graph

def move_parents_to_new_node(lingua_graph: GPGraph, old_node: GPGNode, new_node: GPGNode, filter_func=None, debug=False):
    
    if debug:
        debug_info = []

    for parent, edge in list(lingua_graph.parents(old_node)):
        if filter_func is not None and not filter_func(parent, edge):
            continue

        if debug:
            debug_info.append({
                "from": lingua_graph.node_text(parent),
                "to": lingua_graph.node_text(old_node),
                "label": edge.label
            })
        
        lingua_graph.remove_relation(parent, old_node)
        new_edge_label = add_relation(parent, new_node, edge.label, lingua_graph)

        if debug:
            debug_info.append({
                "from": lingua_graph.node_text(parent),
                "to": lingua_graph.node_text(new_node),
                "label": new_edge_label
            })
            
    if debug:
        return lingua_graph, debug_info
    else:
        return lingua_graph


def remove_aux_nodes(lingua_graph: GPGraph, ordered_nodes: list, debug=False):


    updated = False
    update_idx = 0
    debug_info = {} if debug else None
    
    for visiting_node in reversed(ordered_nodes):

        # if debug:
            # print(f"visiting_node: {visiting_node.ID}, text: {lingua_graph.node_text(visiting_node)}")
        if isinstance(visiting_node, GPGPhraseNode):
            continue

        if not lingua_graph.has_node(visiting_node):
            continue 

        if get_upper_layer_aux(lingua_graph, visiting_node, ordered_nodes) is not None:
            continue 

        aux_chain = list(reversed(find_aux_chain(lingua_graph, visiting_node, ordered_nodes)))
        
        if not aux_chain:
            continue
        
        if debug:
            node_key = f"node_{visiting_node.ID}"
            debug_info[node_key] = {
                "visiting_node_text": lingua_graph.node_text(visiting_node),
                "aux_chain_length": len(aux_chain),
                "layers": []
            }

        first_aux_layer = aux_chain[0]
        first_aux_node = first_aux_layer["aux_node"]
        first_head_node = first_aux_layer["head"]
        first_tail_node = first_aux_layer["tail"]
        # lower_edge_level(lingua_graph, global_head_node)
        tail_nodes = first_tail_node if isinstance(first_tail_node, (tuple, list)) else [first_tail_node]
        first_group = [first_head_node] + tail_nodes

        if debug:
            layer_debug = {
                "layer_index": 0,
                "graph_before": copy.deepcopy(lingua_graph),
                "aux_node": {
                    "label": first_aux_node.label,
                    "pos": first_aux_node.pos,
                    "text": lingua_graph.node_text(first_aux_node)
                },
                "changes": {
                    "nodes_removed": [],
                    "edges_added": [],
                    "edges_removed": [],
                    "edges_modified": []
                }
            }

        current_head_node = aux_chain[0]["head"]
        for idx, this_aux_layer in enumerate(aux_chain):
            this_aux_node = this_aux_layer["aux_node"]
            this_head_node = this_aux_layer["head"]
            this_tail_node = this_aux_layer["tail"]

            if debug:
                layer_debug = {
                    "layer_index": idx + 1,
                    "graph_before": copy.deepcopy(lingua_graph),
                    "aux_node": {
                        "label": this_aux_node.label,
                        "pos": this_aux_node.pos,
                        "text": lingua_graph.node_text(this_aux_node)
                    },
                    "changes": {
                        "nodes_removed": [],
                        "edges_added": [],
                        "edges_removed": [],
                        "edges_modified": []
                    }
                }

            if this_aux_node.pos == "ModificationalFunctor":

                variable = current_head_node
                body = this_tail_node[0]

                resolve_cycle_before_add_edge(lingua_graph, body, variable)
                for child, edge in find_ambiguous_hierarchy(lingua_graph, this_tail_node[0]):
                    if variable in lingua_graph.offsprings(child):
                        add_hierarchy_relation(child, variable, "@upper", lingua_graph)
                    else:
                        add_hierarchy_relation(variable, child, "@lower", lingua_graph)
                new_edge_label = add_relation(body, current_head_node,  symbol_nodelabel2edge[this_aux_node.label], lingua_graph)
                    
                if debug:
                    layer_debug["changes"]["edges_added"].append({
                        "from": lingua_graph.node_text(this_tail_node[0]),
                        "to": lingua_graph.node_text(current_head_node),
                        "label": new_edge_label
                    })

                current_head_node = body

            elif this_aux_node.pos.endswith("Predicator"):

                arg1 = current_head_node
                arg2 = this_tail_node[0]

                resolve_cycle_before_add_edge(lingua_graph, arg2, arg1)
                for child, edge in find_ambiguous_hierarchy(lingua_graph, arg2):
                    if arg1 in lingua_graph.offsprings(child):
                        add_hierarchy_relation(child, arg1, "@upper", lingua_graph)
                    else:
                        add_hierarchy_relation(arg1, child, "@lower", lingua_graph)
                new_edge_label = add_relation(arg2, arg1, symbol_nodelabel2edge[this_aux_node.label], lingua_graph)
                
                if debug:
                    layer_debug["changes"]["edges_added"].append({
                        "from": lingua_graph.node_text(arg1),
                        "to": lingua_graph.node_text(arg2),
                        "label": new_edge_label   
                    })
                
                current_head_node = arg2
            else:
                # lower_edge_level(lingua_graph, current_tail_node)
                aux_children = [current_head_node] + this_tail_node

                aux_edge_label = symbol_nodelabel2edge[this_aux_node.label]
                # next_connecting_node = this_tail_node[0]

                # next_connecting_node = this_tail_node[0]
                for p, n in itertools.pairwise(aux_children):

                    resolve_cycle_before_add_edge(lingua_graph, p, n)
                    for child, edge in find_ambiguous_hierarchy(lingua_graph, p):
                        if n in lingua_graph.offsprings(child):
                            add_hierarchy_relation(child, n, "@upper", lingua_graph)
                        else:
                            add_hierarchy_relation(n, child, "@lower", lingua_graph)

                    for child, edge in find_ambiguous_hierarchy(lingua_graph, n):
                        if aux_edge_label in edge.label.split("+"):
                            add_hierarchy_relation(p, child, "@lower", lingua_graph)
                        
                    new_edge_label = add_relation(p, n, aux_edge_label, lingua_graph)
                        
                    if debug:
                        layer_debug["changes"]["edges_added"].append({
                            "from": lingua_graph.node_text(p),
                            "to": lingua_graph.node_text(n),
                            "label": new_edge_label
                        })
                    
                
            for parent, edge in list(lingua_graph.parents(this_aux_node)):
                if debug:
                    layer_debug["changes"]["edges_removed"].append({
                        "from": lingua_graph.node_text(parent),
                        "to": lingua_graph.node_text(this_aux_node),
                        "label": edge.label
                    })
                
                lingua_graph.remove_relation(parent, this_aux_node)

                new_edge_label = add_relation(parent, current_head_node, edge.label, lingua_graph)
                    
                # resolve_cycle_before_add_edge(lingua_graph, parent, global_head_node)
                if debug:
                    layer_debug["changes"]["edges_added"].append({
                        "from": lingua_graph.node_text(parent),
                        "to": lingua_graph.node_text(current_head_node),
                        "label": new_edge_label
                    })

            if hasattr(this_aux_node, "child_of_whether") and this_aux_node.child_of_whether:
                current_head_node.child_of_whether = True

            if debug:
                layer_debug["changes"]["nodes_removed"].append({
                    "node_text": lingua_graph.node_text(this_aux_node),
                    "node_label": this_aux_node.label,
                    "node_pos": this_aux_node.pos
                })
            
            lingua_graph.remove_node(this_aux_node)
            
            if debug:
                layer_debug["graph_after"] = copy.deepcopy(lingua_graph)
                layer_debug["current_head_text"] = lingua_graph.node_text(current_head_node)
                debug_info[node_key]["layers"].append(layer_debug)

        #     final_root = next

        # for parent, edge in list(lingua_graph.parents(final_root)):
        #     lingua_graph.remove_relation(parent, final_root)
        #     add_relation(parent, global_head_node, edge.label, lingua_graph)

        # lingua_graph.remove_node(final_root)

        updated = True 

    if debug:
        return lingua_graph, updated, debug_info
    else:
        return lingua_graph, updated

# def remove_cycle(lingua_graph: GPGraph):
#     import networkx as nx

#     updated = False
    
#     while True: 
#         cycles = list(nx.simple_cycles(lingua_graph.g))
#         if len(cycles) == 0:
#             break
#         for cycle in cycles:
#             aux_source_nodes = []

#             cycle_nodes = [lingua_graph.get_node(n) for n in cycle + [cycle[0]]]
            
#             for prev, next in itertools.pairwise(cycle_nodes):
#                 edge = lingua_graph.get_edge(prev, next)
#                 if edge and "@" in edge.label:
#                     aux_source_nodes.append(prev)

#             assert len(aux_source_nodes) == 1, "aux source nodes should be exactly one"

#             for aux_souce_node in aux_source_nodes:
#                 for parent, edge in list(lingua_graph.parents(aux_souce_node)):
#                     if parent in cycle_nodes:
#                         lingua_graph.remove_relation(parent, aux_souce_node)
#                         add_relation(aux_souce_node, parent, "reverse_" + edge.label, lingua_graph)
#             updated = True

#    return lingua_graph, updated

def expand_multi_word_node(lingua_graph: GPGraph, debug=False):

    updated = False
    update_idx = 0
    for visiting_node in list(lingua_graph.nodes()):

        if not isinstance(visiting_node, GPGPhraseNode):
            continue

        words = list(visiting_node.words(with_aux=False))
        if len(words) > 1:
            word_nodes = []
            for word in words:
                node = GPGPhraseNode(spans=[[word, word]], pos=visiting_node.pos)
                word_nodes.append(node)
                lingua_graph.add_node(node)
            for prev, next in itertools.pairwise(word_nodes):
                add_relation(prev, next, "next_word", lingua_graph)

            lingua_graph.replace(visiting_node, word_nodes[0])
            if hasattr(visiting_node, "child_of_whether") and visiting_node.child_of_whether:
                word_nodes[0].child_of_whether = True
            updated = True
    return lingua_graph, updated

def lingua2wordlingua(lingua_graph: GPGraph, debug=False):

    ordered_nodes = list(LinguaGraphTopologySorter().nodes(lingua_graph))

    debug_info = {}

    try:
        lingua_graph, updated = remove_whether_aux(lingua_graph, ordered_nodes, debug=debug)
        if debug and updated:
            debug_info["remove_whether_aux"] = copy.deepcopy(lingua_graph)

        if debug:
            lingua_graph, updated, remove_aux_debug  = remove_aux_nodes(lingua_graph, ordered_nodes, debug=debug)
            if updated:
                debug_info["remove_aux_nodes"] = copy.deepcopy(lingua_graph)
        else:
            lingua_graph, updated = remove_aux_nodes(lingua_graph, ordered_nodes, debug=debug)


        # if debug:
        #     lingua_graph, updated, remove_aux_debug = remove_aux_parallel_structure(lingua_graph, ordered_nodes, debug=debug)
        #     if updated:
        #         debug_info["remove_aux_parallel"] = remove_aux_debug
        # else:
        #     lingua_graph, updated = remove_aux_parallel_structure(lingua_graph, ordered_nodes, debug=debug)

        # if debug:
        #     lingua_graph, updated, remove_aux_debug = remove_aux_predicator(lingua_graph, ordered_nodes, debug=debug)
        #     if updated:
        #         debug_info["remove_aux_predicator"] = remove_aux_debug
        # else:
        #     lingua_graph, updated = remove_aux_predicator(lingua_graph, ordered_nodes, debug=debug)

        # if debug:
        #     lingua_graph, updated, remove_aux_debug = remove_aux_modification(lingua_graph, ordered_nodes, debug=debug)
        #     if updated:
        #         debug_info["remove_aux_modification"] = remove_aux_debug
        # else:
        #     lingua_graph, updated = remove_aux_modification(lingua_graph, ordered_nodes, debug=debug)

        # if debug:
        #     lingua_graph, updated, remove_aux_debug = remove_aux_nodes(lingua_graph, ordered_nodes, debug=debug)
        #     if updated:
        #         debug_info["remove_aux_nodes"] = remove_aux_debug
        # else:
        #     lingua_graph, updated = remove_aux_nodes(lingua_graph, ordered_nodes, debug=debug)

        # lingua_graph, updated = remove_aux_chain(lingua_graph, ordered_nodes, debug=debug)
        # if debug and updated:
        #     debug_info["remove_aux_chain"] = copy.deepcopy(lingua_graph)

        # lingua_graph, updated = remove_cycle(lingua_graph)
        # if debug and updated:
        #     debug_info["remove_cycle"] = copy.deepcopy(lingua_graph)

        # lingua_graph, updated = remove_list_auxnodes(lingua_graph, ordered_nodes, debug=debug)
        # if debug and updated:
        #     debug_info["remove_list_auxnodes"] = copy.deepcopy(lingua_graph)

        lingua_graph, updated = expand_multi_word_node(lingua_graph, debug=debug)

        if debug and updated:
            debug_info["expand_multi_word_node"] = copy.deepcopy(lingua_graph)
    
    except Exception as e:
        if debug:
            e.debug_info = debug_info
        raise e

    return lingua_graph, updated, debug_info
