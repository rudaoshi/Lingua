from copy import deepcopy
from enum import Enum
from pathlib import Path
import abc
import copy
from collections import defaultdict, OrderedDict
import sys
import traceback

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lingua.structure.gpgraph import GPGraph, GPGPhraseNode, GPGAuxNode, GPGNode

import logging
logger = logging.getLogger(__name__)


class GraphNodeVisitor(abc.ABC):

    @abc.abstractmethod
    def visit_before_child(self, node):
        pass

    @abc.abstractmethod
    def visit_after_child(self, node):
        pass

    def is_visited(self, node):
        return False

    def set_visited(self, node):
        pass


def deep_first_visit_dag(node, graph, visitor: GraphNodeVisitor):
    if visitor.is_visited(node):
        return

    visitor.visit_before_child(node)

    for child, _edge in graph.children(node):
        deep_first_visit_dag(child, graph, visitor)

    visitor.visit_after_child(node)
    visitor.set_visited(node)


class LinguaNodeType(Enum):
    ReferentialPredicator = "ReferentialPredicator"
    ModificationalFunctor = "ModificationalFunctor"
    ConjunctionalFunctor = "ConjunctionalFunctor"
    LogicalPredicator = "LogicalPredicator"
    AttributePredicator = "AttributePredicator"
    FactualPredicator = "FactualPredicator"
    GeneralFunctor = "GeneralFunctor"
    DeterminerConstant = "DeterminerConstant"
    PunctuationalPredicator = "PunctuationalPredicator"
    AppositionalPredicator = "AppositionalPredicator"


PREDEFINED_PREDICATE_NAME_TO_TYPE = {
    "Apposition": LinguaNodeType.AppositionalPredicator,
    "Attribute": LinguaNodeType.AttributePredicator,
    "Copula": LinguaNodeType.FactualPredicator,
    "Discourse": LinguaNodeType.ModificationalFunctor,
    "Index": LinguaNodeType.FactualPredicator,
    "Modification": LinguaNodeType.ModificationalFunctor,
    "Parataxis": LinguaNodeType.ConjunctionalFunctor,
    "Punctuation": LinguaNodeType.PunctuationalPredicator,
    "Ref": LinguaNodeType.ReferentialPredicator,
    "Repeat": LinguaNodeType.ModificationalFunctor,
    "Vocative": LinguaNodeType.ModificationalFunctor,
}


def remove_cycle(lingua_graph: GPGraph):
    """Remove back-edges until the graph becomes acyclic."""

    def get_back_edges(graph):
        back_edges = set()
        visited = set()
        rec_stack = set()

        def dfs(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor_id in graph[node_id]:
                if neighbor_id not in visited:
                    dfs(neighbor_id)
                elif neighbor_id in rec_stack:
                    back_edges.add((node_id, neighbor_id))

            rec_stack.remove(node_id)

        for node_id in graph:
            if node_id not in visited:
                dfs(node_id)

        return back_edges

    for source_id, target_id in get_back_edges(lingua_graph.g):
        lingua_graph.remove_edge_between(
            lingua_graph.get_node(source_id),
            lingua_graph.get_node(target_id),
        )

"""
algorithm:
* standardize 
* process function as:pred.arg.
* process modification 
    * rank modification by priority
    * convert paralleled modification to hierarchical structure according to priority
"""


AUXILIARY_VERB = ['do', 'does', 'is', 'are', 'have', 'has', 'did', 'had', 'shall', 'should', 'will', 'would', 'were', 'was']
def add_root(lingua_graph: GPGraph):

    v_root = lingua_graph.root
    if isinstance(v_root, GPGAuxNode) and v_root.label == 'Root':
        return

    root = [n for n in lingua_graph.nodes() if lingua_graph.g.in_degree[n.ID] == 0]
    assert len(root) == 1

    root = root[0]
    virtual_root = GPGAuxNode(label="Root")
    virtual_root.pos = 'Root'
    lingua_graph.add_node(virtual_root)
    lingua_graph.add_relation(virtual_root, root, "root")

def remove_root(lingua_graph: GPGraph):
    root = [n for n in lingua_graph.nodes() if lingua_graph.g.in_degree[n.ID] == 0]
    assert len(root) == 1

    root = root[0]
    if isinstance(root, GPGAuxNode) and root.label == 'Root':   
        lingua_graph.remove_node(root)


def process_function(lingua_graph):
    """
    two classes of functions:
    - ModificationalFunction:  where, when, why, how, etc, funtor -(func.arg)-> predicate
    - NominalFunction: that which, who, what, etc, funtor -(as:pred.arg.x)-> predicate
    """
    applied = False
    edge_list = list(lingua_graph.edges())
    for s, edge, e in edge_list:
        if s.pos != 'GeneralFunctor':
            continue 

        if not edge.label.startswith('as:pred.arg.'):
            continue 

        assert e.pos.endswith('Predicator')

        for parent, parent_edge in list(lingua_graph.parents(s)):
            lingua_graph.remove_relation(parent, s)
            lingua_graph.add_relation(parent, e, parent_edge.label)

        lingua_graph.remove_edge(edge)
        s.pos = "GeneralFunctor"
        lingua_graph.add_relation(e, s, edge.label[3:])
        applied = True

    return applied


        


priority_map = {

    LinguaNodeType.ReferentialPredicator.value: 80,
    LinguaNodeType.ModificationalFunctor.value: 80,
    LinguaNodeType.ConjunctionalFunctor.value: 80,
    LinguaNodeType.LogicalPredicator.value: 80,
    LinguaNodeType.AttributePredicator.value: 80,
    LinguaNodeType.FactualPredicator.value: 80,
    LinguaNodeType.GeneralFunctor.value: 60,

    LinguaNodeType.DeterminerConstant.value: 80,
    LinguaNodeType.PunctuationalPredicator.value: 80,
    
#    LinguaNodeType.DeterminerConstant.value: 3

}



def get_modification_priority(lingua_graph: GPGraph, node, modification_node):

    if modification_node.pos in priority_map:
        base_priority = priority_map[modification_node.pos]
    else:
        base_priority = 20

    if isinstance(node, GPGAuxNode) or all(isinstance(c, GPGAuxNode) for c in lingua_graph.offsprings(modification_node)):
        return base_priority
    else:

        node_words = list(node.words(with_aux=False))
        min_node_pos = min(node_words)
        min_modification_pos = 10000
        min_dist = 10000
        for c in lingua_graph.offsprings(modification_node):
            min_dist = min(min_dist, min_node_distance(node, c))
            if isinstance(c, GPGPhraseNode):
                c_words = list(c.words(with_aux=False))
                min_modification_pos = min(min_modification_pos, min(c_words))
        
        positional_bonus = 100 if min_modification_pos < min_node_pos else 0
        
        distance_penalty_factor = 5
        distance_penalty = (min_dist - 1) * distance_penalty_factor if min_dist > 1 else 0

        effective_score = base_priority + positional_bonus - distance_penalty
        return effective_score
    

def back_track_variable_ancestor(lingua_graph: GPGraph, node):

    this_node = node 
    while True:
        variable_parent_found = False
        for parent, edge in lingua_graph.parents(this_node):
            if edge.label == "variable":
                this_node = parent
                variable_parent_found = True
                break 
        if not variable_parent_found:
            return this_node
        


class Modification2HierarchyVisitor(GraphNodeVisitor):

    def __init__(self, graph: GPGraph, debug_info=None):
        self.graph = graph
        self.target_graph = deepcopy(graph)

        self.target_graph.cached_relations = graph.cached_relations
        self.applied = False
        self.debug_info = debug_info

        if self.debug_info is not None:
            self.debug_info.step_results["modification_to_hierarchy"] = dict()

        self.visited_nodes = set()

    def visit_before_child(self, n):
        pass

    def is_visited(self, n):
        return n.ID in self.visited_nodes

    def set_visited(self, n):
        self.visited_nodes.add(n.ID)

    def visit_after_child(self, this_node):

        # if this_node.pos == 'GeneralFunctor':
        #     modification_children = [p for p, e in self.target_graph.children(this_node) if e.label.startswith("as:pred.arg.")]
        #     if len(modification_children) == 0:
        #         return False
            
        #     if len(modification_children) != 1:
        #         raise ValueError(f"Modification Function {this_node.ID} has multiple modification children: {modification_children}")
            
        #     modification_child = modification_children[0]

        #     if modification_child.pos not in  {'ModificationalFunctor', 'FactualPredicator', 'ModificationalFunction', 
        #                                        'ConjunctionalPredicate', 'LogicalPredicator'}:
        #         raise ValueError(f"Modification Function {this_node.ID} has invalid modification pos: {modification_child.pos}")
            
        #     parents = list(self.target_graph.parents(this_node))

        #     function_node = GPGAuxNode(label="GeneralFunctor", pos="GeneralFunctor")
        #     self.target_graph.add_node(function_node)
        #     self.target_graph.add_relation(function_node, this_node, "variable")
        #     self.target_graph.add_relation(function_node, modification_child, "body")

        #     edge = self.target_graph.get_edge(this_node, modification_child)
        #     self.target_graph.remove_relation(this_node, modification_child)
        #     self.target_graph.add_relation(modification_child, this_node, edge.label[3:])

        #     for parent, e in parents:
        #         self.target_graph.add_relation(parent, function_node, e.label)
        #         self.target_graph.remove_relation(parent, modification_child)

        #     return True

        
        responsible_predicates = {'ModificationalFunctor', "FactualPredicator", 
                                  "PunctuationalPredicator", "LogicalPredicator",
                                  "ReferentialPredicator", "AppositionalPredicator",
                                  "AttributePredicator", "ModificationalFunction"}

        if self.debug_info is not None:
            logs = []
        try:

            this_target_node = self.target_graph.get_node(this_node.ID)

            this_node_text = self.target_graph.node_text(this_target_node)
            
            this_node_signature = this_node_text + "_{}".format(this_node.ID)

            this_target_children = [p for p, e in self.target_graph.children(this_target_node) 
                            if e.label.startswith("as:pred.arg.") and p.pos in responsible_predicates]
            other_children = [p for p, e in self.target_graph.children(this_target_node) 
                            if e.label.startswith("as:pred.arg.") and p not in this_target_children]

            if len(this_target_children) == 0:
                return
            
            if self.debug_info is not None:
                self.debug_info.step_results["modification_to_hierarchy"][this_node_signature] = {"logs": "", "steps": [], "error": None}

            groups = []
            # reverse conjunction of modifications. Need post-processing to find the root of the modification

            # shared_grandson_to_child_map = defaultdict(list)   
            # for c in this_target_children:
            #     for p, e in self.target_graph.children(c):
            #         if e.label.startswith("as:pred.arg."):
            #             shared_grandson_to_child_map[p].append(c)

            
            # for grandson, children in shared_grandson_to_child_map.items():

            #     if len(children) > 1:
            #         assert grandson.pos == 'ConjunctionalPredicate'
                
            #     poses = [c.pos for c in children]
            #     assert len(set(poses)) == 1
            #     group_pos = poses[0]
            #     group_priority = get_modification_priority(self.target_graph, this_node, children, group_pos)
            #     groups.append((group_priority, children, grandson))
            #     nodes_in_group.update(children)

            for c in this_target_children:
                group_priority = get_modification_priority(self.target_graph, this_node, c)
                groups.append((group_priority, c))
            
            groups.sort(key=lambda x: x[0], reverse=True)
            if self.debug_info is not None:
                group_expr_texts = []
                for priority, child in groups:
                    group_expr_texts.append(str(priority) + "<:>" + str(child.ID))
                logs.append("groups: \n" + "\n".join(group_expr_texts))

            this_target_node = self.target_graph.get_node(this_node.ID)
            target_parents = list(self.target_graph.parents(this_target_node))

            current_target_sub_root = this_target_node
            # prev_target_sub_root = None
            # processed_children = set()

            

            for _, child in groups:
                try:
                    if self.debug_info is not None:
                        group_logs = []

                    if self.debug_info is not None:
                        group_logs.append(f"processing child: {child.ID}")
                        group_logs.append(f"current_target_sub_root: {current_target_sub_root.ID}")
                        

                    if self.debug_info is not None:
                        group_logs.append("reversing the as:pred.arg. edge")

                        group_logs.append(f"\treversing target_child: {child.ID}")

                    edge = self.target_graph.get_edge(this_target_node, child)

                    self.target_graph.remove_relation(this_target_node, child)
                    self.target_graph.add_relation(child, current_target_sub_root, edge.label[3:])

                    if child in self.target_graph.cached_relations:
                        for c, p, l in self.target_graph.cached_relations[child]:
                            self.target_graph.add_relation(c, current_target_sub_root, l)

                    # add mod function
                    mod_func_node = GPGAuxNode("ComplexModification", pos="ModificationalFunction")
                    self.target_graph.add_node(mod_func_node)
                    self.target_graph.add_relation(mod_func_node, current_target_sub_root, "variable")
                    variable_ancestor = back_track_variable_ancestor(self.target_graph, child)
                    self.target_graph.add_relation(mod_func_node, variable_ancestor, "body")

                    if self.debug_info is not None:
                        group_logs.append(f"making the mod-function node {mod_func_node.ID} the new root")

                    # prev_target_sub_root = current_target_sub_root
                    current_target_sub_root = mod_func_node

                    if self.debug_info is not None:
                        group_logs.append("." * 100)

                    # processed_children.add(child)

                    # if self.debug_info is not None:
                    #     group_logs.append("3. moving the remaining modification relations to the new root")

                    # for child in this_target_children:
                    #     if child in processed_children:
                    #         continue

                    #     if self.debug_info is not None:
                    #         group_logs.append(f"\tprocessing child: {self.graph.node_text(child)}")
                        
                    #     edge = None
                    #     if prev_target_sub_root:
                    #         edge = self.target_graph.get_edge(prev_target_sub_root, child)
                    #     else:
                    #         edge = self.target_graph.get_edge(this_target_node, child)
                    #     if not edge:
                    #         edge = self.graph.get_edge(this_target_node, child)

                    #     self.target_graph.add_relation(current_target_sub_root, child, edge.label)
                    #     if prev_target_sub_root is not None:
                    #         self.target_graph.remove_relation(prev_target_sub_root, child)
                        
                    # if self.debug_info is not None:
                    #     group_logs.append("." * 100)

                    if self.debug_info is not None:
                        self.debug_info.step_results["modification_to_hierarchy"][this_node_signature]["steps"].append(
                            {
                                "graph": copy.deepcopy(self.target_graph),
                                "logs": "\n".join(group_logs)
                            }
                        )

                    
                except Exception as e:
                    if self.debug_info is not None:
                        self.debug_info.step_results["modification_to_hierarchy"][this_node_signature]["steps"].append(
                            {
                                "graph": None,
                                "logs": "\n".join(group_logs),
                                "error": traceback.format_exc()
                            }
                        )
                    raise e
            
            if self.debug_info is not None:
                logs.append("Modification hierarchy applied, moving other children to the new root")

            for child in other_children:
                if self.debug_info is not None:
                    logs.append(f"\tmove child {child.ID} to new root {current_target_sub_root.ID}")
                edge = self.target_graph.get_edge(this_target_node, child)

                self.target_graph.add_relation(current_target_sub_root, child, edge)
                self.target_graph.remove_relation(this_target_node, child)

            for parent, e in target_parents:    
                if e.label.startswith("as:"):
                    continue 
                if self.debug_info is not None:
                    logs.append(f"\tadding parent {parent.ID} to the new root: {current_target_sub_root.ID}")
            
                self.target_graph.add_relation(parent, current_target_sub_root, e.label)
                self.target_graph.remove_relation(parent, this_target_node)

            # if this_target_node.pos == 'GeneralFunctor':
            #     current_target_sub_root.label = "GeneralFunction"
            #     current_target_sub_root.pos = "GeneralFunctor"
            #     this_target_node.pos = "GeneralFunctor"

            #     current_var = [x for x, e in self.target_graph.children(current_target_sub_root) if e.label == "variable"]
            #     assert len(current_var) == 1
            #     current_var = current_var[0]
            #     self.target_graph.remove_relation(current_target_sub_root, current_var)

            #     self.target_graph.add_relation(current_target_sub_root, this_target_node, "variable")

                
            if self.debug_info is not None:
                self.debug_info.step_results["modification_to_hierarchy"][this_node_signature]["logs"] = "\n".join(logs)
                self.debug_info.step_results["modification_to_hierarchy"][this_node_signature]["graph"] = copy.deepcopy(self.target_graph)
            # if self.debug_info is not None:
            #     logs.append("Modification hierarchy applied, moving parent relations to the new root")
                
            # for parent, e in target_parents:
            #     if self.debug_info is not None:
            #         logs.append(f"\tadding parent relation to target_grandson: {self.target_graph.node_text(parent)}")
            #     self.target_graph.add_relation(parent, current_target_sub_root, e.label)
            #     self.target_graph.remove_relation(parent, this_target_node)

            # if self.debug_info is not None:
            #     self.debug_info.step_results["modification_to_hierarchy"][this_node_text]["steps"].append(
            #         {
            #             "graph": copy.deepcopy(self.target_graph),
            #             "logs": "\n".join(logs)
            #         }
            #     )

            self.applied = True

        except Exception as e:
            stack_trace_message = traceback.format_exc()
            if self.debug_info is not None:
                self.debug_info.step_results["modification_to_hierarchy"][this_node_signature]["error"] = stack_trace_message  
                self.debug_info.step_results["modification_to_hierarchy"][this_node_signature]["logs"] = "\n".join(logs)
            raise e
                
        
    
def modification_to_hierarchy(lingua_graph: GPGraph, debug_info=None):


    visitor = Modification2HierarchyVisitor(lingua_graph, debug_info)

    deep_first_visit_dag(lingua_graph.root, lingua_graph, visitor)

    result_graph = visitor.target_graph
    lingua_graph.g = result_graph.g
    lingua_graph.node_id_base = result_graph.node_id_base

    # currently, the graph has multiple roots, we need to set the root to the original root
    #lingua_graph.root = lingua_graph.get_node(root_id)

    lingua_graph.update_root()

    if visitor.applied:
        return True
    else:
        return False
    



class ModificationSimplificationVisitor(GraphNodeVisitor):

    def __init__(self, graph: GPGraph, debug_info=None):

        super().__init__()

        self.graph = graph
        self.target_graph = deepcopy(graph)

        self.applied = False
        self.debug_info = debug_info

        if self.debug_info is not None:
            self.debug_info.step_results["modification_function_simplification"] = dict()

        self.visited_nodes = set()

    def visit_before_child(self, n):
        pass

    def is_visited(self, n):
        return n.ID in self.visited_nodes

    def set_visited(self, n):
        self.visited_nodes.add(n.ID)

    def visit_after_child(self, this_node):

        if this_node.ID in self.visited_nodes:
            return

        self.visited_nodes.add(this_node.ID)

        if this_node.pos != "ModificationalFunction":
            return
    
        logs = []
        error = []


        this_target_node = self.target_graph.get_node(this_node.ID)
        this_node_text = self.target_graph.node_text(this_target_node)

        this_node_signature = this_node_text + "_{}".format(this_node.ID)

        if self.debug_info is not None:
            self.debug_info.step_results["modification_function_simplification"][this_node_signature] = dict()
        

        try:
            variable_child = [p for p, e in self.target_graph.children(this_target_node) if e.label == "variable"]
            body_child = [p for p, e in self.target_graph.children(this_target_node) if e.label == "body"]

            if len(variable_child) != 1:
                message = f"ModificationalFunction {this_node.ID} is not simplified, multiple variable children: {[x.ID for x in variable_child]}"
                logger.error(message)
                error.append(message)
                return 
            
            if len(body_child) != 1:
                message = f"ModificationalFunction {this_node.ID} is not simplified, multiple body children: {[x.ID for x in body_child]}"
                logger.error(message)
                error.append(message)
                return 

            variable_child = variable_child[0]
            body_child = body_child[0]

            if variable_child.pos == "GeneralFunctor":
                message = f"ModificationalFunction {this_node.ID} is not simplified, variable child is a function: {variable_child.ID}"
                logger.info(message)
                logs.append(message)
                return 

            other_mod_children = [(p, e) for p, e in self.target_graph.children(this_target_node) if p not in [variable_child, body_child]]

            if body_child.pos not in {
                "ModificationalFunctor", "FactualPredicator",
                "PunctuationalPredicator", "LogicalPredicator",
                "ReferentialPredicator", "AppositionalPredicator",
                "AttributePredicator"
            }:
                message = f"ModificationalFunction {this_node.ID} is not simplified, body child is not ModificationalFunctor or FactualPredicator: {body_child.ID}"
                logger.info(message)
                logs.append(message)
                return 

            body_arg1_child = [p for p, e in self.target_graph.children(body_child) if e.label == "pred.arg.1"]
            if len(body_arg1_child) != 1:
                message = f"ModificationalFunction {this_node.ID} is not simplified, multiple body arg1 children: {[x.ID for x in body_arg1_child]}"
                logger.error(message)
                error.append(message)
                return 
            body_arg1_child = body_arg1_child[0]

            body_arg2_children = [p for p, e in self.target_graph.children(body_child) if e.label == "pred.arg.2"]
            if len(body_arg2_children) > 1:
                message = f"ModificationalFunction {this_target_node.ID} is not simplified, multiple body arg2 children: {[x.ID for x in body_arg2_children]}"
                logger.error(message)
                error.append(message)
                return 

            if body_arg1_child != variable_child:
                message = f"ModificationalFunction {this_target_node.ID} is not simplified, arg1_child is not variable_child"
                logger.info(message)
                logs.append(message)
                return 
            

            
            modfunc_parent = list(self.target_graph.parents(this_target_node))

            new_mod_root = None 

            if isinstance(body_child, GPGAuxNode):
                if len(body_arg2_children) != 1:
                    # means len(body_arg2_children) == 0 
                    message = f"ModificationalFunction {this_target_node.ID} is not simplified, multiple or zero body arg2 children: {[x.ID for x in body_arg2_children]}"
                    logger.error(message)
                    new_mod_root = body_arg1_child
                    self.target_graph.remove_node(body_child)
                else:
                    body_arg2_child = body_arg2_children[0]

                    # TODO: If arg2 has multiple offsprings, the modification cannot be simplified
                    body_arg2_offsprings = list(self.target_graph.offsprings(body_arg2_child))
                    if len(body_arg2_offsprings) > 1:
                        message = f"ModificationalFunction {this_target_node.ID} is not simplified, body arg2 has multiple offsprings: {[x.ID for x in body_arg2_offsprings]}"
                        logger.info(message)
                        new_mod_root = body_child
                    else:
                        new_mod_root = body_arg2_child
                        self.target_graph.add_relation(body_arg2_child, variable_child, "as:" + body_child.label.lower())
                        
                        self.target_graph.remove_node(body_child)
            else:
                new_mod_root = body_child

            for parent, e in modfunc_parent:
                self.target_graph.add_relation(parent, new_mod_root, e.label)
                self.target_graph.remove_relation(parent, this_target_node)

            for p, e in other_mod_children:
                self.target_graph.add_relation(new_mod_root, p, e.label)
                self.target_graph.remove_relation(p, this_target_node)

            self.target_graph.remove_node(this_target_node)

            self.applied = True
        except Exception as e:
            error.append(traceback.format_exc())
            raise e

        finally:
            if self.debug_info is not None:
                self.debug_info.step_results["modification_function_simplification"][this_node_signature]["logs"] = "\n".join(logs)
                self.debug_info.step_results["modification_function_simplification"][this_node_signature]["error"] = "\n".join(error)

                if self.applied:
                    self.debug_info.step_results["modification_function_simplification"][this_node_signature]["graph"] = copy.deepcopy(self.target_graph)
                



def modification_simplification(lingua_graph: GPGraph, debug_info=None):

    root_id = lingua_graph.root.ID

    visitor = ModificationSimplificationVisitor(lingua_graph, debug_info)

    deep_first_visit_dag(lingua_graph.root, lingua_graph, visitor)

    result_graph = visitor.target_graph
    lingua_graph.g = result_graph.g

    # currently, the graph has multiple roots, we need to set the root to the original root
    # lingua_graph.root = lingua_graph.get_node(root_id)


    if visitor.applied:
        return True
    else:
        return False
    

def function_simplification(lingua_graph: GPGraph, debug_info=None):

    # case: uri 67

    updated = False

    for node in list(reversed(list(lingua_graph.topological_sort()))):

        if node.pos != "GeneralFunctor":
            continue

        func_arg_children = [p for p, e in lingua_graph.children(node) if e.label == "func.arg"]
        if len(func_arg_children) != 1:
            continue

        func_arg_child = func_arg_children[0]

        if func_arg_child.pos != "ModificationalFunction":
            continue

        variable_child = [p for p, e in lingua_graph.children(func_arg_child) if e.label == "variable"]
        body_child = [p for p, e in lingua_graph.children(func_arg_child) if e.label == "body"]

        assert len(variable_child) == 1
        assert len(body_child) == 1

        variable_child = variable_child[0]
        body_child = body_child[0]

        other_mod_children = [(p, e) for p, e in lingua_graph.children(func_arg_child) if p not in [variable_child, body_child]]


        body_arg1_child = [p for p, e in lingua_graph.children(body_child) if e.label == "pred.arg.1"]
        if len(body_arg1_child) != 1:
            continue 

        body_arg1_child = body_arg1_child[0]

        body_arg2_children = [p for p, e in lingua_graph.children(body_child) if e.label == "pred.arg.2"]
        if len(body_arg2_children) != 1:
            continue

        body_arg2_child = body_arg2_children[0]

        if body_arg1_child == variable_child:

            # this should not happen, since it should be processed by modification_simplification
            continue 

        lingua_graph.add_relation(node, variable_child, "variable")
        lingua_graph.add_relation(node, body_child, "body")

        for p, e in other_mod_children:
            lingua_graph.add_relation(body_child, p, e.label)

        lingua_graph.remove_node(func_arg_child)

        updated = True

    return updated




def min_node_distance(n1: GPGNode, n2: GPGNode):
    if isinstance(n1, GPGAuxNode) or isinstance(n2, GPGAuxNode):
        return 10000

    assert isinstance(n1, GPGPhraseNode) and isinstance(n2, GPGPhraseNode)

    import numpy as np

    n1_words = np.array(list(n1.words(with_aux=False)))
    n2_words = np.array(list(n2.words(with_aux=False)))

    if n1_words.shape[0] == 0 or n2_words.shape[0] == 0:
        return 10000

    from scipy.spatial.distance import cdist

    return np.min(cdist(n1_words[..., np.newaxis], n2_words[..., np.newaxis])) # word index difference between n1 and n2

def max_node_distance(n1: GPGNode, n2: GPGNode):
    if isinstance(n1, GPGAuxNode) or isinstance(n2, GPGAuxNode):
        return -10000

    assert isinstance(n1, GPGPhraseNode) and isinstance(n2, GPGPhraseNode)

    import numpy as np

    n1_words = np.array(list(n1.words(with_aux=False)))
    n2_words = np.array(list(n2.words(with_aux=False)))

    if n1_words.shape[0] == 0 or n2_words.shape[0] == 0:
        return -10000

    from scipy.spatial.distance import cdist

    return np.max(cdist(n1_words[..., np.newaxis], n2_words[..., np.newaxis])) # word index difference between n1 and n2

def is_subset(small_list, large_list):
    for node in small_list:
        if node not in large_list:
            return False
    return True



def span_dist(node1, node2):
    import numpy as np
    node1_words = np.expand_dims(list(node1.words()), 1)
    node2_words = np.expand_dims(list(node2.words()), 1)

    from scipy.spatial import distance
    dist_mat = distance.cdist(node1_words, node2_words)
    return dist_mat.min()


def punct2pred(lingua_graph: GPGraph):

    for node in list(lingua_graph.nodes()):

        punct_children = [child for child, edge in lingua_graph.children(node)
                           if isinstance(child, GPGPhraseNode) and child.pos in ['PunctuationalConstant']]
        if not punct_children:
            continue

        parent_list = list(lingua_graph.parents(node))
        punct_children.sort(key=lambda x: span_dist(node, x), reverse=True)

        cur_parent = None
        for punct_child in punct_children:

            punct_child.pos = 'PunctuationalPredicator'
            lingua_graph.remove_relation(node, punct_child)
            lingua_graph.add_relation(punct_child, node, 'pred.arg.1')

            if cur_parent:
                lingua_graph.add_relation(cur_parent, punct_child, 'pred.arg.1')
                lingua_graph.remove_relation(cur_parent, node)

            cur_parent = punct_child

        for p, edge in parent_list:
            lingua_graph.remove_relation(p, node)
            lingua_graph.add_relation(p, punct_children[0], edge.label)

    #
    #
    # node_list = [node for node in lingua_graph.nodes() if node.pos in ['PunctuationalConstant']]
    # #node_list = [node for node in lingua_graph.nodes() if node.pos == 'PunctuationalPredicator']
    #
    # for node in node_list:
    #     #assert node.label == 'Punctuation'
    #     # assert node.label in ['Tail', 'Local', 'Global', 'Part', 'Punctuation']
    #     parent_list = list(lingua_graph.parents(node))
    #     child_list = [child for child, child_edge in lingua_graph.children(node)]
    #     arg1_list = [(child, child_edge) for child, child_edge in lingua_graph.children(node) if child_edge.label in ['pred.arg.1']]
    #     arg1_list_fix = [(child, child_edge) for child, child_edge in lingua_graph.children(node) if child_edge.label in ['pred.arg.1', 'direct:pred.arg.1']]
    #     arg2_list = [child for child, child_edge in lingua_graph.children(node) if child_edge.label == 'pred.arg.2']
    #     # if len(arg1_list_fix) == 1 and len(arg1_list) == 0:
    #     #     continue
    #     assert len(child_list) == 2 and len(arg1_list) == 1 and len(arg2_list) == 1, 'check child {}'.format(lingua_graph.node_text(node))
    #     punct_node = arg2_list[0]
    #     #punct_node.pos = 'PunctuationalPredicator'
    #     punct_node.pos = node.pos
    #     lingua_graph.remove_node(node)
    #     for parent, parent_edge in parent_list:
    #         lingua_graph.add_relation(parent, punct_node, parent_edge)
    #     lingua_graph.add_relation(punct_node, arg1_list[0][0], arg1_list[0][1])
    #     # if arg1_list[0].pos == 'GeneralFunctor':
    #     #     punct_node.pos = 'GeneralFunctor'



def is_prepositional_node(graph, node):
    if isinstance(node, GPGPhraseNode) and node.pos == 'ModificationalFunctor':
        parents = list(graph.parents(node))
        assert len(parents) == 1, "modification predicate has multiple parents {}".format(node.spans)
        p, pe = parents[0]
        if p.pos != 'NominalConstant':
            return True
    return False


def move_shared_modification_to_conjunction(lingua_graph: GPGraph):
    """
    move the shared modification to conjunction functor 
    """
    applied = False
    for node in lingua_graph.nodes():

        if node.pos not in {'ModificationalFunctor', 'LogicalPredicator'}:
            continue

        parents = list(lingua_graph.parents(node))
        if len(parents) <= 1:
            continue

        parent_edges = set([e.label for p, e in parents])

        if len(parent_edges) > 1:
            continue 

        parent_edge = parent_edges.pop()

        if not parent_edge.startswith("as:pred.arg.") or parent_edge == "modification":
            continue

        grandpar = set([g for p, e in parents for g, _ in lingua_graph.parents(p) ])
        if len(grandpar) != 1:
            continue

        grandpar = grandpar.pop()

        if grandpar.pos != 'ConjunctionalFunctor':
            logger.warning(f"Grandparent of {node.ID} is not a conjunctional function: {grandpar.ID}")
            continue

        for p, e in parents:
            lingua_graph.remove_relation(p, node)

        lingua_graph.add_relation(grandpar, node, parent_edge)

        applied = True

    return applied

def find_nearest_common_ancestor(lingua_graph: GPGraph, nodes):

    all_ancestors = set()
    root_to_parents_paths = []
    for p in nodes:
        paths = list(lingua_graph.paths(lingua_graph.root, p))
        root_to_parents_paths.extend(paths)
        for path in paths:
            all_ancestors.update(path)

    common_ancestors = set() 
    for ancestor in all_ancestors:
        if all(ancestor in path for path in root_to_parents_paths):
            common_ancestors.add(ancestor.ID)

    topo_order = nx.topological_sort(lingua_graph.g)
    nearest_common_ancestor = None
    for node_id in topo_order:
        if node_id in common_ancestors:
            nearest_common_ancestor = node_id

    if nearest_common_ancestor is None:
        return None 

    nearest_common_ancestor = lingua_graph.get_node(nearest_common_ancestor)
    return nearest_common_ancestor


def process_reverse_conjunction(lingua_graph: GPGraph):
    applied = False
    lingua_graph.cached_relations = defaultdict(list)

    for node in list(lingua_graph.nodes()):

        if node.pos not in {"ConjunctionalFunctor"}:
            continue

        this_conj_node = node

        parents = list(lingua_graph.parents(this_conj_node))
        children = list(lingua_graph.children(this_conj_node))

        if not any(edge.label.startswith("as:func.arg") for p, edge in parents):
            continue

        if len(children) != 0:
            continue

        if not all(edge.label.startswith("as:func.arg.") for p, edge in parents):
            continue

        if len(parents) <= 1:
            continue

        nearest_common_ancestor = find_nearest_common_ancestor(lingua_graph, [p for p, _ in parents])

        if nearest_common_ancestor is None:
            logger.warning(f"No nearest common ancestor found for {this_conj_node.ID}")
            continue


        conj_children = []
        relation_between_common_ancestor_and_conj = set()

        for p, edge in parents:
            paths = list(lingua_graph.paths(nearest_common_ancestor, p))

            if len(paths) == 0:
                logger.warning(f"No path found from {nearest_common_ancestor.ID} to {p.ID}")
                continue

            assert len(paths) == 1
            path = paths[0]
            conj_child = path[1]
            conj_children.append(conj_child)

            rel = lingua_graph.get_edge(nearest_common_ancestor, conj_child)
            relation_between_common_ancestor_and_conj.add(rel.label)

        if len(relation_between_common_ancestor_and_conj) != 1:
            logger.warning(f"Multiple relations found between {nearest_common_ancestor.ID} and {this_conj_node.ID}: {relation_between_common_ancestor_and_conj}")
            continue

        relation_between_common_ancestor_and_conj = relation_between_common_ancestor_and_conj.pop()

        if relation_between_common_ancestor_and_conj == "as:pred.arg.1":
            mod_node = GPGAuxNode("Modification", pos=LinguaNodeType.ModificationalFunctor.value)
            lingua_graph.add_node(mod_node)
            lingua_graph.add_relation(nearest_common_ancestor, mod_node, "as:pred.arg.1")
            lingua_graph.add_relation(mod_node, this_conj_node, "pred.arg.2")
            for c in conj_children:
                lingua_graph.remove_relation(nearest_common_ancestor, c)

                lingua_graph.cached_relations[mod_node].append((c, nearest_common_ancestor, "pred.arg.1"))
                #lingua_graph.add_relation(c, nearest_common_ancestor, "pred.arg.1")
        # elif relation_between_common_ancestor_and_conj.startswith("as:"):
        #     raise Exception("Cannot process relation {}".format(relation_between_common_ancestor_and_conj))
        else:
            lingua_graph.add_relation(nearest_common_ancestor, this_conj_node, relation_between_common_ancestor_and_conj)
            for c in conj_children:
                lingua_graph.remove_relation(nearest_common_ancestor, c)

        for child, (conj_parent, edge) in zip(conj_children, parents):
            lingua_graph.remove_relation(conj_parent, this_conj_node)
            lingua_graph.add_relation(this_conj_node, child, "func.arg")

        applied = True

    return applied

def apply_cached_relations(lingua_graph: GPGraph):
    for relation_list in lingua_graph.cached_relations.values():
        for child, parent, edge_label in relation_list:
            lingua_graph.add_relation(parent, child, edge_label)



def not_prepositional_node(grpah, node):
    return not is_prepositional_node(grpah, node)

def expand_modification_relations(lingua_graph: GPGraph):
    applied = False
    
    for start, edge, end in list(lingua_graph.edges()):
        if edge.label.startswith("pred.arg") or edge.label.startswith("as:pred.arg") \
            or edge.label.startswith("func.arg") or edge.label.startswith("as:func.arg"):
            continue
        
        reverse_relation = False
        relation = edge.label
        if edge.label.startswith("as:"):
            reverse_relation = True
            relation = edge.label[3:]

        predicate_type = PREDEFINED_PREDICATE_NAME_TO_TYPE.get(relation.capitalize())
        if predicate_type is None:
            pos = "unknown"
        else:
            pos = predicate_type.value

        new_auxnode = GPGAuxNode(relation.capitalize(), pos=pos)

        if not reverse_relation:

            lingua_graph.add_node(new_auxnode)
            lingua_graph.remove_relation(start, end)
            lingua_graph.add_relation(start, new_auxnode, 'as:pred.arg.1')
            lingua_graph.add_relation(new_auxnode, end, 'pred.arg.2')
            applied = True
        else:
            lingua_graph.add_node(new_auxnode)
            lingua_graph.remove_relation(start, end)
            lingua_graph.add_relation(start, new_auxnode, 'as:pred.arg.2')
            lingua_graph.add_relation(new_auxnode, end, 'pred.arg.1')
            applied = True
        
    return applied

def hierarchical_view(original_lingua_graph: GPGraph, debug_info=None):

    lingua_graph = copy.deepcopy(original_lingua_graph)

    # add arg1 for copula
    if debug_info is not None:
        debug_info.step_results = OrderedDict()
        debug_info.step_results["original"] = copy.deepcopy(lingua_graph)

    applied = expand_modification_relations(lingua_graph)
    if applied and debug_info is not None:
        debug_info.step_results["expand_modification_relations"] = copy.deepcopy(lingua_graph)

    # Processing reverse conjunctions
    applied = process_reverse_conjunction(lingua_graph)
    if applied and debug_info is not None:
        debug_info.step_results["process_reverse_conjunction"] = copy.deepcopy(lingua_graph)

    
    applied = move_shared_modification_to_conjunction(lingua_graph)
    if applied and debug_info is not None:
        debug_info.step_results["move_shared_modification_to_conjunction"] = copy.deepcopy(lingua_graph)

    lingua_graph.update_root()

    # apply_cached_relations(lingua_graph)

    remove_cycle(lingua_graph)

    applied = modification_to_hierarchy(lingua_graph, debug_info)
    remove_cycle(lingua_graph)

    applied = modification_simplification(lingua_graph, debug_info)
    remove_cycle(lingua_graph)

    applied = function_simplification(lingua_graph)
    remove_cycle(lingua_graph)

    lingua_graph.update_root()

    if applied and debug_info is not None:
        debug_info.step_results["function_simplification"] = copy.deepcopy(lingua_graph)

    if debug_info is not None:
        debug_info.step_results["final"] = copy.deepcopy(lingua_graph)

    return lingua_graph

