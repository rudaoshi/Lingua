from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lingua.structure.gpgraph import TextGPGraph, GPGTextNode, GPGAuxNode
from collections import defaultdict, OrderedDict
import logging
logger = logging.getLogger(__name__)
import re
import traceback
from typing import Dict, List

# Global debug information dictionary
debug_info_dict: Dict[str, List[Dict[str, str]]] = defaultdict(list)

class DebugInfoHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        debug_info = {
            'level': record.levelname,
            'message': msg,
            'chunk': getattr(record, 'chunk', ''),
            'parsing_type': getattr(record, 'parsing_type', ''),
            'timestamp': record.created
        }
        debug_info_dict[record.levelname.lower()].append(debug_info)

# Add the custom handler to the logger
debug_handler = DebugInfoHandler()
logger.addHandler(debug_handler)

def log_with_context(level, msg, chunk='', parsing_type=''):
    extra = {'chunk': chunk, 'parsing_type': parsing_type}
    logger.log(level, msg, extra=extra)

def first_word(chunk: str):
    cleaned_chunk = re.sub(r"\{\d+\}", " ", chunk)
    cleaned_chunk = re.sub(r"\{\{\d+\}\}", " ", cleaned_chunk)

    words = filter(None, cleaned_chunk.split(" "))
    return next(words)


class LLMParsingResultVisitor:

    def __init__(self):
        self.lingua_graph = TextGPGraph()

    def make_role_type_view(self, parsing_result: dict):
        role_view = OrderedDict()
        type_view = OrderedDict()

        if "COMPONENTS" not in parsing_result:
            return role_view, type_view

        for sub_chunk, sub_parsing in parsing_result["COMPONENTS"].items():
            sub_parsing_type = sub_parsing["TYPE"]
            role = sub_parsing["ROLE"] if "ROLE" in sub_parsing else None
            
            if role is None:
                role = sub_parsing_type

            if role not in role_view:
                role_view[role] = []
            if sub_parsing_type not in type_view:
                type_view[sub_parsing_type] = []

            sub_parsing["CHUNK"] = sub_chunk
            role_view[role].append(sub_parsing)
            type_view[sub_parsing_type].append(sub_parsing)

        return role_view, type_view

    def parse(self, chunk: str, parsing_type: str, parsing_result: dict):
        # print("parsing ", chunk, " type ", parsing_type)
        role_view, type_view = self.make_role_type_view(parsing_result)
        exception_occurred = False
        try:
            match parsing_type:
                case "SS":
                    node = parse_ss(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "RSC" | "SUBSC" | "SC" | "NPC" | "VPC" | "SVPC" | "ADVMC" | "PPC" |"ADJC" | "ACM_OBJC" | "ACMC":
                    node = parse_conjunction(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "LIST":
                    node = parse_list(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "EXPRESSION":
                    node = parse_expression(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "RSMS":
                    node = parse_rsms(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "SUBSMS":
                    node = parse_subsms(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                # case "RS":
                #     return parse_rs(chunk, parsing_type, parsing_result, lingua_graph)
                case "SUBS":
                    node = parse_subs(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "SCP":
                    node = parse_scp(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "NP":
                    node = parse_np(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "PP":
                    node = parse_pp(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "VP":
                    node = parse_vp(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "SNP":
                    node = parse_snp(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "ACM_OBJ":
                    node = parse_acm_obj(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "ACM":
                    node = parse_acm(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                # case "SVPING" | "SVPED":
                #     node = parse_svp_ing_ed(chunk, parsing_type, parsing_result, lingua_graph)
                case "POSS_SNP":
                    node = parse_poss_snp(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "AMPW":
                    node = parse_ampw(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "CCP":
                    node = parse_ccp(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "SADV" | "ADV":
                    node = parse_sadv(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "SADJ":
                    node = parse_sadj(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "SVP":
                    node = parse_svp(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "SLVP":
                    node = parse_slvp(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "BE_VERB":
                    node = parse_be_verb(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "AUX_VERB":
                    node = parse_aux_verb(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "NS":
                    node = parse_ns(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "WH_PHRASE":
                    node = parse_wh_phrase(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "WH_NS" | "WH_RS":
                    node = parse_wh_sentence(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "COMP_NS":
                    node = parse_comp_ns(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "NS_COMPLEMENTIZER":
                    node = parse_ns_complementizer(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "THAT_WORD":
                    node = parse_that_word(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "THAT_RS":
                    node = parse_that_rs(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "APPOS":
                    node = parse_appos(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "APPOS_CONJ":
                    node = parse_appos_conj(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "COMP_NS":
                    node = parse_comp_ns(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "DET" | "NUMBER" | "INTJ" | "SYM" | "OTHER":
                    node = parse_constant(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "OP":
                    node = parse_op(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "POSS_MARKER":
                    node = parse_poss_marker(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "ACM_MARKER":
                    node = parse_acm_marker(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "ATTR":
                    node = parse_attr(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "Q_MARK":
                    node = parse_q_mark(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case "YN_QUESTION":
                    node = parse_yn_question(chunk, parsing_type, parsing_result, role_view, type_view, self.lingua_graph)
                case _:
                    log_with_context(logging.WARNING, f"Unsupported type: {parsing_type}", chunk, parsing_type)
                    node = GPGTextNode(chunk, pos=parsing_type.capitalize() + "Constant")
                    self.lingua_graph.add_node(node)

        except Exception as e:
            print(f"Error parsing {chunk} with type {parsing_type}: {traceback.format_exc()}")

            # log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: {e}", chunk, parsing_type)

            if "COMPONENTS" not in parsing_result:
                log_with_context(logging.ERROR, f"Error parsing {chunk} with type {parsing_type}: {e}", chunk, parsing_type)
                node = GPGTextNode(chunk, pos=parsing_type.capitalize() + "Constant")
                self.lingua_graph.add_node(node)
            else:

                # assert all("SUBTREE" in parsing_result["COMPONENTS"][sub_chunk] for sub_chunk in parsing_result["COMPONENTS"]), f"Error parsing {chunk} with type {parsing_type}: {e}"
                # for sub_chunk in parsing_result["COMPONENTS"]:
                #     if "SUBTREE" not in parsing_result["COMPONENTS"][sub_chunk]:
                #         sub_type = parsing_result["COMPONENTS"][sub_chunk]["TYPE"]
                #         log_with_context(logging.ERROR, f"Error parsing {sub_chunk} with type {sub_type}: {e}", sub_chunk, sub_type)
                #         sub_node = GPGTextNode(sub_chunk, pos=sub_type.capitalize() + "Constant")
                #         lingua_graph.add_node(sub_node)
                #         parsing_result["COMPONENTS"][sub_chunk]["SUBTREE"] = sub_node

                all_sub_nodes = []
                for sub_chunk in parsing_result["COMPONENTS"]:
                    if "SUBTREE" in parsing_result["COMPONENTS"][sub_chunk]:
                        subtree = parsing_result["COMPONENTS"][sub_chunk]["SUBTREE"]
                        if subtree is None:
                            continue
                        if isinstance(subtree, (list, tuple)):
                            all_sub_nodes.extend([n for n in subtree if self.lingua_graph.g.has_node(n.ID)])
                        elif self.lingua_graph.g.has_node(subtree.ID):
                            all_sub_nodes.append(subtree)

                nodes = sorted(all_sub_nodes, key=lambda x: len(list(self.lingua_graph.offsprings(x))))
                node = nodes[-1]

            exception_occurred = True


        if "COMPONENTS" in parsing_result:
            unknown_relation_occurred = process_disconnected_subcomponents(node, parsing_type, parsing_result, role_view, type_view, self.lingua_graph, exception_occurred)
            if unknown_relation_occurred:
                sub_types = [parsing_result["COMPONENTS"][sub_chunk]["TYPE"] for sub_chunk in parsing_result["COMPONENTS"]]
                sub_types_pattern = ", ".join(sub_types)
                log_with_context(logging.ERROR, f"Unknown relation found by pattern [{parsing_type}: {sub_types_pattern}] for chunk [{chunk}]", chunk, parsing_type)

        return node

    def visit(self, chunk: str, parsing_type: str, parsing_result: dict):
        try:
            if "COMPONENTS" in parsing_result and len(parsing_result["COMPONENTS"]) == 1:
                sub_chunk, sub_parsing = next(iter(parsing_result["COMPONENTS"].items()))
                sub_parsing_type = sub_parsing["TYPE"]
                node = sub_parsing["SUBTREE"]
            else:
                node = self.parse(chunk, parsing_type, parsing_result)

        except Exception as e:
            log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: {e}", chunk, parsing_type)
            # print(f"Error parsing {chunk} with type {parsing_type}: {traceback.format_exc()}")
            raise e
        
        punct_host = None

        if isinstance(node, (list, tuple)):
            for n in node:
                # print("node: ", n.pos, n.text if isinstance(n, GPGTextNode) else n.label)
                if not hasattr(n, "chunk"):
                    n.chunk = chunk
                    
            common_children = set()
            for idx, n in enumerate(node):
                if idx == 0:
                    common_children = set([n for n, _ in self.lingua_graph.children(n)])
                else:
                    common_children.intersection_update(set([n for n, _ in self.lingua_graph.children(n)]))

            if len(common_children) == 0:
                log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: No common children found for node {node.pos} {node.text if isinstance(node, GPGTextNode) else node.label}", chunk, parsing_type)
                raise ValueError(f"No common children found for node {node.pos} {node.text if isinstance(node, GPGTextNode) else node.label}")
            elif len(common_children) > 1:
                log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Multiple common children found for node {node.pos} {node.text if isinstance(node, GPGTextNode) else node.label}", chunk, parsing_type)
                raise ValueError(f"Multiple common children found for node {node.pos} {node.text if isinstance(node, GPGTextNode) else node.label}")
            else:
                punct_host = common_children.pop()
        else:
            
            # print("node: ", node.pos, node.text if isinstance(node, GPGTextNode) else node.label)
            if not hasattr(node, "chunk"):
                node.chunk = chunk

            punct_host = node

        if "TerminalPunct" in parsing_result:
            punct_node = GPGTextNode(parsing_result["TerminalPunct"], pos="PunctuationalConstant")
            self.lingua_graph.add_node(punct_node)
            self.lingua_graph.add_relation(punct_host, punct_node, "punctuation")

        if "StartPunct" in parsing_result:
            punct_node = GPGTextNode(parsing_result["StartPunct"], pos="PunctuationalConstant")
            self.lingua_graph.add_node(punct_node)
            self.lingua_graph.add_relation(punct_host, punct_node, "punctuation")

        if "BracketPunct" in parsing_result:
            punct_node = GPGTextNode(parsing_result["BracketPunct"], pos="PunctuationalConstant")
            self.lingua_graph.add_node(punct_node)
            self.lingua_graph.add_relation(punct_host, punct_node, "punctuation")
        
        # print("after process: chunk ", chunk, " type ", parsing_type, " result ", parsing_result)

        parsing_result["SUBTREE"] = node

import collections
def parse_llm_parsing_result(chunk: str, parsing_result: dict, visitor: LLMParsingResultVisitor):

    if "COMPONENTS" in parsing_result:
        sub_chunks = list(parsing_result["COMPONENTS"].keys())

        try:
            for sub_chunk in sub_chunks:
                assert first_word(sub_chunk) in chunk, f"Subchunk {sub_chunk} not found in chunk {chunk}"

            sorted_sub_chunks = sorted(sub_chunks, key=lambda x: chunk.index(first_word(x)))
        except Exception as e:
            log_with_context(logging.WARNING, f"Error sorting subchunks: {e}", chunk, parsing_result.get("TYPE", ""))
            sorted_sub_chunks = sub_chunks

        components = collections.OrderedDict()
        for sub_chunk in sorted_sub_chunks:
            components[sub_chunk] = parsing_result["COMPONENTS"][sub_chunk]

        parsing_result["COMPONENTS"] = components
        for sub_chunk, sub_parsing in components.items():
            sub_parsing_type = sub_parsing["TYPE"]
            parse_llm_parsing_result(sub_chunk, sub_parsing, visitor)

    visitor.visit(chunk, parsing_result["TYPE"], parsing_result)


def process_subj_objs(pred_node, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    processed_roles = []

    for role, parsing_results in role_view.items():
        
        match role:
            case "SBJ":
                assert len(parsing_results) == 1
                sub_tree_root = parsing_results[0]["SUBTREE"]
                if pred_node.pos == "ConjunctionalFunctor":
                    for real_verb, _ in lingua_graph.children(pred_node):
                        lingua_graph.add_relation(real_verb, sub_tree_root, "pred.arg.1")
                else:
                    lingua_graph.add_relation(pred_node, sub_tree_root, "pred.arg.1")

                processed_roles.append(role)

            case "DOBJ":
                assert len(parsing_results) == 1
                sub_tree_root = parsing_results[0]["SUBTREE"]
                if pred_node.pos == "ConjunctionalFunctor":
                    for real_verb, _ in lingua_graph.children(pred_node):
                        lingua_graph.add_relation(real_verb, sub_tree_root, "pred.arg.2")
                else:
                    lingua_graph.add_relation(pred_node, sub_tree_root, "pred.arg.2")

                processed_roles.append(role)
            case "IOBJ":
                assert len(parsing_results) == 1
                sub_tree_root = parsing_results[0]["SUBTREE"]
                if pred_node.pos == "ConjunctionalFunctor":
                    for real_verb, _ in lingua_graph.children(pred_node):
                        lingua_graph.add_relation(real_verb, sub_tree_root, "pred.arg.3")
                else:
                    lingua_graph.add_relation(pred_node, sub_tree_root, "pred.arg.3")

                processed_roles.append(role)

            case "OBJ_COMPLEMENT":
                assert len(parsing_results) == 1
                sub_tree_root = parsing_results[0]["SUBTREE"]
                if pred_node.pos == "ConjunctionalFunctor":
                    for real_verb, _ in lingua_graph.children(pred_node):
                        lingua_graph.add_relation(real_verb, sub_tree_root, "pred.arg.4")
                else:
                    lingua_graph.add_relation(pred_node, sub_tree_root, "pred.arg.4")

                processed_roles.append(role)

    return processed_roles

def process_modifications(root_node, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    processed_roles = set()

    for role, parsing_results in role_view.items():

        match role:
            case "PP" | "PPC":
                for parsing_result in parsing_results:
                    lingua_graph.add_relation(root_node, parsing_result["SUBTREE"], "as:pred.arg.1")

                processed_roles.add(role)

            case "MOD":
                for parsing_result in parsing_results:
                    sub_root = parsing_result["SUBTREE"]
                    if sub_root == root_node:
                        continue
                    if sub_root.pos.endswith("Predicator"):
                        lingua_graph.add_relation(root_node, sub_root, "as:pred.arg.1")
                    else:
                        lingua_graph.add_relation(root_node, sub_root, "modification")
                
                processed_roles.add(role)

            case "ADVM" | "ADVMC":
                for parsing_result in parsing_results:
                    lingua_graph.add_relation(root_node, parsing_result["SUBTREE"], "modification")

                processed_roles.add(role)

            case "ACM":
                for parsing_result in parsing_results:
                    if parsing_result["SUBTREE"].pos.endswith("Predicator"):
                        lingua_graph.add_relation(root_node, parsing_result["SUBTREE"], "as:pred.arg.1")
                    else:
                        lingua_graph.add_relation(root_node, parsing_result["SUBTREE"], "modification")

                processed_roles.add(role)

            

    return processed_roles


def parse_attr(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """

    if "ATTR_NAME" not in role_view or len(role_view["ATTR_NAME"]) == 0:
        log_with_context(logging.WARNING, f"No attribute name components found in {chunk}", chunk, parsing_type)
        raise ValueError(f"No attribute name components found in {chunk}")

    if "ATTR_VALUE" not in role_view or len(role_view["ATTR_VALUE"]) == 0:
        log_with_context(logging.WARNING, f"No attribute value components found in {chunk}", chunk, parsing_type)
        raise ValueError(f"No attribute value components found in {chunk}")

    if len(role_view["ATTR_NAME"]) > 1:
        log_with_context(logging.WARNING, f"Multiple attribute name components found in {chunk}", chunk, parsing_type)
        raise ValueError(f"Multiple attribute name components found in {chunk}")
    
    if len(role_view["ATTR_VALUE"]) > 1:
        log_with_context(logging.WARNING, f"Multiple attribute value components found in {chunk}", chunk, parsing_type)
        raise ValueError(f"Multiple attribute value components found in {chunk}")
    
    attr_name_node = role_view["ATTR_NAME"][0]["SUBTREE"]
    attr_value_node = role_view["ATTR_VALUE"][0]["SUBTREE"]

    attr_name_node.pos = "AttributePredicator"

    lingua_graph.add_relation(attr_name_node, attr_value_node, "pred.arg.2")
    return attr_name_node
    
    
        
        

def process_other_relations(root_node, parsing_results: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    processed_roles = set()

    for role, parsing_results in role_view.items():

        match role:
            case "APPOS":
                for parsing_result in parsing_results:
                    appos_root = parsing_result["SUBTREE"]
                    if appos_root.pos == "AppositionalPredicator":
                        lingua_graph.add_relation(root_node, appos_root, "as:pred.arg.1")
                    else:
                        lingua_graph.add_relation(root_node, appos_root, "appositive")

                processed_roles.add(role)

            case "DISCOURSE":
                for parsing_result in parsing_results:
                    lingua_graph.add_relation(root_node, parsing_result["SUBTREE"], "discourse")

                processed_roles.add(role)

            case "INDEX":
                for parsing_result in parsing_results:
                    lingua_graph.add_relation(root_node, parsing_result["SUBTREE"], "index")

                processed_roles.add(role)

            case "ATTR":
                for parsing_result in parsing_results:
                    lingua_graph.add_relation(root_node, parsing_result["SUBTREE"], "as:pred.arg.1")

                processed_roles.add(role) 

            case "ATTR_VALUE":
                for parsing_result in parsing_results:
                    lingua_graph.add_relation(root_node, parsing_result["SUBTREE"], "attribute")

                processed_roles.add(role)

    return processed_roles

def process_disconnected_subcomponents(root_node, parsing_type: str, parsing_results: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph, exception_occurred: bool = False):

    import networkx as nx
    all_node = []
    for sub_chunk in parsing_results["COMPONENTS"]:
        if "SUBTREE" in parsing_results["COMPONENTS"][sub_chunk]:
            subtree = parsing_results["COMPONENTS"][sub_chunk]["SUBTREE"]
            if subtree is None:
                continue
            if isinstance(subtree, (list, tuple)):
                all_node.extend(subtree)
            else:
                all_node.append(subtree)

    if isinstance(root_node, (list, tuple)):
        root_nodes = root_node
    else:
        root_nodes = [root_node]

    unknown_relation_occurred = False
    for r in root_nodes:
        for node in all_node:
            if not lingua_graph.g.has_node(node.ID):
                continue
            if node == r:
                continue

            if not nx.has_path(lingua_graph.g.to_undirected(), r.ID, node.ID):
                if node.pos == "ModificationalConstant":
                    lingua_graph.add_relation(r, node, "modification")
                elif node.pos == "PunctuationalConstant":
                    lingua_graph.add_relation(r, node, "punctuation")
                else:
                    if not exception_occurred:
                        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for missing processing some component chunk []: Add unknown relation to connect sub-component", "", "process_disconnected_subcomponents")
                    lingua_graph.add_relation(r, node, "unknown")
                    unknown_relation_occurred = True

    return unknown_relation_occurred

def parse_expression(chunk: str, parsing_type: str, parsing_result: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "EXPRESSION"


    if "OP" not in type_view or len(type_view["OP"]) == 0:
        op = GPGAuxNode("Expression", pos="ExpressionFunctor")
        lingua_graph.add_node(op)
    else:
        if len(type_view["OP"]) > 1:
            log_with_context(logging.WARNING, f"Multiple operators found in the parsing result {type_view} for chunk {chunk}", chunk, parsing_type)
   
        op = type_view["OP"][0]["SUBTREE"]

    children_components = list(itertools.chain(*[type_view[x] for x in type_view if x != "OP"]))


    sorted_components = sorted(children_components, key=lambda x: chunk.find(x["CHUNK"]))

    for idx, component in enumerate(sorted_components):
        lingua_graph.add_relation(op, component["SUBTREE"], f"func.arg")

    return op
        


def parse_list(chunk: str, parsing_type: str, parsing_result: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "LIST"


    root = GPGAuxNode("List", pos="ListFunctor")
    lingua_graph.add_node(root)

    for role, parsing_results in type_view.items():
        for parsing_result in parsing_results:
            lingua_graph.add_relation(root, parsing_result["SUBTREE"], "func.arg")
        
    return root

def parse_constant(chunk: str, parsing_type: str, parsing_result: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """


    node = GPGTextNode(chunk, pos=parsing_type.capitalize() + "Constant")
    lingua_graph.add_node(node)

    return node
        

# when parsing each parent type, we need to check:
# 1. whether it is already parsed as a more specific type only with name mis-generated
# 2. whether it is parsed to contain a sub-component of a more specific type


def parse_s(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert sentence parsing to Lingua format.
    """

    if any(sub_chunk_result["TYPE"] == "CCP" 
           for sub_chunk, sub_chunk_result in parsing["COMPONENTS"].items()):
        return parse_conjunction(chunk, parsing_type, parsing, role_view, type_view, lingua_graph)
    elif any(sub_chunk_result["TYPE"] in ["SUBS", "SUBSC"] 
             for sub_chunk, sub_chunk_result in parsing["COMPONENTS"].items()):
        return parse_subsms(chunk, parsing_type, parsing, role_view, type_view, lingua_graph)
    elif any(sub_chunk_result["TYPE"] in ["RSC", "RS", "THAT_RS", "WH_RS"] 
             for sub_chunk, sub_chunk_result in parsing["COMPONENTS"].items()):
        return parse_rsms(chunk, parsing_type, parsing, role_view, type_view, lingua_graph)
    elif any("ROLE" in sub_chunk_result and sub_chunk_result["ROLE"] == "PRED" 
             for sub_chunk, sub_chunk_result in parsing["COMPONENTS"].items()):
        return parse_ss(chunk, parsing_type, parsing, role_view, type_view, lingua_graph)
    else:
        
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: unknown how to parse S", chunk, parsing_type)
        raise ValueError(f"unknown how to parse S")

    

def parse_ns(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "NS"

    if any(sub_chunk_result["TYPE"] == "NS_COMPLEMENTIZER" 
           for sub_chunk, sub_chunk_result in parsing["COMPONENTS"].items()):
        return parse_comp_ns(chunk, parsing_type, parsing, role_view, type_view, lingua_graph)
    elif any(sub_chunk_result["TYPE"] == "WH_NS" 
             for sub_chunk, sub_chunk_result in parsing["COMPONENTS"].items()):
        return parse_wh_sentence(chunk, parsing_type, parsing, role_view, type_view, lingua_graph)
    elif any("ROLE" in sub_chunk_result and sub_chunk_result["ROLE"] == "PRED" 
             for sub_chunk, sub_chunk_result in parsing["COMPONENTS"].items()):
        return parse_ss(chunk, parsing_type, parsing, role_view, type_view, lingua_graph)
    else:
        
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: unknown how to parse NS", chunk, parsing_type)
        raise ValueError(f"unknown how to parse NS")

def parse_cs(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "CS"

    if any(sub_chunk_result["TYPE"] == "CCP" 
           for sub_chunk, sub_chunk_result in parsing["COMPONENTS"].items()):
        return parse_conjunction(chunk, parsing_type, parsing, role_view, type_view, lingua_graph)
    elif any(sub_chunk_result["TYPE"] == "SUBS" 
             for sub_chunk, sub_chunk_result in parsing["COMPONENTS"].items()):
        return parse_subs(chunk, parsing_type, parsing, role_view, type_view, lingua_graph)
    elif any(sub_chunk_result["TYPE"] in ["RS", "WH_RS", "THAT_RS"] 
             for sub_chunk, sub_chunk_result in parsing["COMPONENTS"].items()):
        return parse_rsms(chunk, parsing_type, parsing, role_view, type_view, lingua_graph)
    else:
        
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: unknown how to parse CS", chunk, parsing_type)
        raise ValueError(f"unknown how to parse CS")

        
def parse_wh_phrase(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "WH_PHRASE"

    function_node = GPGTextNode(chunk, pos="GeneralFunctor")
    lingua_graph.add_node(function_node)

    return function_node

def process_noun_relations(noun_node, relations, clause_node, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):

    if clause_node.pos.endswith("GeneralFunctor"):
        if not hasattr(clause_node, "ref"):
            return 
        if not (clause_node.ref == noun_node.chunk or clause_node.ref == noun_node.text):
            log_with_context(logging.WARNING, f"Unexpected ref {clause_node.ref} for noun node {noun_node.chunk}", "", "noun_relations")

        lingua_graph.add_relation(noun_node, clause_node, "ref")
    else:

        for relation, target in relations:

            related_node = None
            for node in lingua_graph.offsprings(clause_node):
                
                if isinstance(node, GPGAuxNode):
                    continue
                if node.chunk == target or node.text == target:
                    related_node = node

            if not related_node:
                continue 

            if relation == "as:DOBJ":
                lingua_graph.add_relation(noun_node, related_node, "as:pred.arg.2")
            elif relation == "as:IOBJ":
                lingua_graph.add_relation(noun_node, related_node, "as:pred.arg.3")
            elif relation == "as:SBJ":
                lingua_graph.add_relation(noun_node, related_node, "as:pred.arg.1")
            else:
                log_with_context(logging.ERROR, f"Error parsing type NounRelation : Unexpected relation {relation} for noun node {noun_node.chunk}", chunk, parsing_type)
                raise ValueError(f"Unexpected relation {relation} for noun node {noun_node.chunk}")



def process_that_wh_relations(that_wh_node, relations, ampw_node, clause_node, lingua_graph: TextGPGraph):

    relation_added = False
    relations = [tuple(x) for x in relations]
    if ampw_node is not None:
        
        if not (("as:DOBJ", ampw_node.text) in relations or ("as:DOBJ", ampw_node.chunk) in relations):
            log_with_context(logging.ERROR, f"Error parsing type ThatWhRelation : Unexpected wh relations {relations} for ampw node {ampw_node.chunk} ", "", "that_wh_relations")
            raise ValueError(f"Unexpected wh relations {relations} for ampw node {ampw_node.chunk} ")
        lingua_graph.add_relation(that_wh_node, ampw_node, "as:pred.arg.2")
        relation_added = True
        relations = [x for x in relations if x != ("as:DOBJ", ampw_node.chunk)]

    for relation, target in relations:
        if relation == "as:REF":
            that_wh_node.ref = target
            continue 

        related_node = None
        for node in lingua_graph.offsprings(clause_node):
        
            if isinstance(node, GPGAuxNode):
                continue
            
            if target in node.chunk or target in node.text:
                related_node = node

        if not related_node:
            log_with_context(logging.ERROR, f"Error parsing type ThatWhRelation : No node found for relation {relation} and target {target}", "", "that_wh_relations")
            raise ValueError(f"No node found for relation {relation} and target {target}")
        
        if relation in ["as:DOBJ", "as:OBJ"]:
            lingua_graph.add_relation(that_wh_node, related_node, "as:pred.arg.2")
            relation_added = True
        elif relation == "as:IOBJ":
            lingua_graph.add_relation(that_wh_node, related_node, "as:pred.arg.3")
            relation_added = True
        elif relation == "as:SBJ":
            lingua_graph.add_relation(that_wh_node, related_node, "as:pred.arg.1")
            relation_added = True
        elif relation == "as:MOD":
            assert related_node == clause_node, f"Unexpected related node {related_node} for relation {relation}"
            lingua_graph.add_relation(that_wh_node, related_node, "func.arg")
            relation_added = True
        else:
            log_with_context(logging.ERROR, f"Error parsing type ThatWhRelation : Unexpected relation {relation} for that wh node {that_wh_node.chunk}", "", "that_wh_relations")
            raise ValueError(f"Unexpected relation {relation} for that wh node {that_wh_node.chunk}")

    if not relation_added:
        lingua_graph.add_relation(that_wh_node, clause_node, "func.arg")




def parse_that_rs(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "THAT_RS"

    pred_node = find_pred(chunk, parsing, role_view, type_view)
    print(f"Pred node: {pred_node}")

    if pred_node is not None:
        clause_root = pred_node
    elif "SS" in type_view or "SC" in type_view:
        clause_root = type_view["SS"][0]["SUBTREE"] if "SS" in type_view else type_view["SC"][0]["SUBTREE"]
    else:
        clause_components = list(itertools.chain(*[type_view[x] for x in type_view if x not in ["WH_PHRASE", "AMPW"]]))
        if len(clause_components) == 1:
            clause_component = clause_components[0]
            clause_root = clause_component["SUBTREE"]
        else:
            log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Multiple clauses found in the parsing result {type_view}", chunk, parsing_type)
            raise ValueError(f"Multiple clauses found in the parsing result {type_view} for chunk {chunk}")

    process_subj_objs(clause_root, role_view, type_view, lingua_graph)
    process_modifications(clause_root, role_view, type_view, lingua_graph)
    process_other_relations(clause_root, parsing, role_view, type_view, lingua_graph)

    that_components = list(itertools.chain(*[type_view[x] for x in type_view if x in ["THAT_WORD", "NS_COMPLEMENTIZER"]]))
    
    assert len(that_components) <= 1, f"Expected at most 1 that component, got {len(that_components)} in {type_view}"

    if that_components:
        that_component = that_components[0]
        that_node = that_component["SUBTREE"]

        log_with_context(logging.INFO, f"That node: {that_node}", chunk, parsing_type)
        log_with_context(logging.INFO, f"Clause root: {clause_root}", chunk, parsing_type)

        if "RELATION" in that_component:
            relations = that_component["RELATION"]
            process_that_wh_relations(that_node, relations, None, clause_root, lingua_graph)

        else:
            lingua_graph.add_relation(that_node, clause_root, "func.arg")

        root = that_node
    else:
        root = clause_root

    return root

def parse_q_mark(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "Q_MARK"

    return GPGTextNode(chunk, pos="PunctuationalConstant")

def parse_yn_question(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    YN_QUESTION (yes/no question) = (AUX_VERB|BE_VERB) -> S -> (Q_MARK)? 
    """
    assert parsing_type == "YN_QUESTION"

    type_pattern = tuple([parsing["COMPONENTS"][sub_chunk]["TYPE"] for sub_chunk in parsing["COMPONENTS"]])

    root = GPGAuxNode("Whether", pos="GeneralFunctor")
    lingua_graph.add_node(root)
    match type_pattern:
        case ("AUX_VERB", "SS"):
            aux_verb = type_view["AUX_VERB"][0]["SUBTREE"]
            pred = type_view["SS"][0]["SUBTREE"]
            lingua_graph.add_relation(root, pred, "func.arg")
            lingua_graph.add_relation(pred, aux_verb, "aux_verb")
            return root
        case ("BE_VERB", "SS"):
            be_verb = type_view["BE_VERB"][0]["SUBTREE"]
            pred = type_view["SS"][0]["SUBTREE"]
            lingua_graph.add_relation(root, pred, "func.arg")
            lingua_graph.add_relation(pred, be_verb, "aux_verb")
            return root
        case _:
            log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Unexpected type pattern {type_pattern} for chunk {chunk}", chunk, parsing_type)
            raise ValueError(f"Unexpected type pattern {type_pattern} for chunk {chunk}")
    

def parse_wh_sentence(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type in ["WH_NS", "WH_RS"]

    wh_ampw_node = None
    if "WH_PHRASE" not in type_view:
        wh_phrase = GPGAuxNode("Missing", pos="GeneralFunctor")
        lingua_graph.add_node(wh_phrase)
    else:
        wh_phrase = type_view["WH_PHRASE"][0]["SUBTREE"]

        if "AMPW" in type_view: 
            assert len(type_view["AMPW"]) == 1
            ampw_node = type_view["AMPW"][0]["SUBTREE"]
            if chunk.index(ampw_node.chunk) < chunk.index(wh_phrase.chunk):
                wh_ampw_node = ampw_node
                del type_view["AMPW"]
                del role_view["AMPW"]


    pred_node = find_pred(chunk, parsing, role_view, type_view)

    if pred_node is not None:
        clause_root = pred_node
    elif "SS" in type_view or "SC" in type_view:
        clause_root = type_view["SS"][0]["SUBTREE"] if "SS" in type_view else type_view["SC"][0]["SUBTREE"]
    else:
        clause_components = list(itertools.chain(*[type_view[x] for x in type_view if x not in ["WH_PHRASE", "AMPW"]]))
        if len(clause_components) == 1:
            clause_component = clause_components[0]
            clause_root = clause_component["SUBTREE"]
        else:
            log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Multiple clauses found in the parsing result {type_view}", chunk, parsing_type)
            raise ValueError(f"Multiple clauses found in the parsing result {type_view} for chunk {chunk}")

    process_subj_objs(clause_root, role_view, type_view, lingua_graph)
    process_modifications(clause_root, role_view, type_view, lingua_graph)
    process_other_relations(clause_root, parsing, role_view, type_view, lingua_graph)
        
    if wh_ampw_node:
        lingua_graph.add_relation(wh_phrase, wh_ampw_node, "as:pred.arg.2")

        lingua_graph.add_relation(wh_ampw_node, clause_root, "pred.arg.1")


    if "WH_PHRASE" in type_view and "RELATION" in type_view["WH_PHRASE"][0]:
        wh_relations = type_view["WH_PHRASE"][0]["RELATION"]

        process_that_wh_relations(wh_phrase, wh_relations, wh_ampw_node, clause_root, lingua_graph)
    else:
        lingua_graph.add_relation(wh_phrase, clause_root, "func.arg")

    return wh_phrase

def parse_that_word(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "THAT_WORD"

    that_word = GPGTextNode(chunk, pos="GeneralFunctor")
    lingua_graph.add_node(that_word)

    return that_word


def parse_appos(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """

    assert parsing_type == "APPOS"

    if "APPOS_CONJ" not in type_view:
        appos_conj = GPGAuxNode("Appositive", pos="AppositionalPredicator")
        lingua_graph.add_node(appos_conj)
    else:
        appos_conj = type_view["APPOS_CONJ"][0]["SUBTREE"]

    other_components = list(itertools.chain(*[type_view[x] for x in type_view if x != "APPOS_CONJ"]))
    if len(other_components) > 1:
        log_with_context(logging.WARNING, f"Multiple appos components found in the parsing result {type_view} for chunk {chunk}", chunk, parsing_type)

    for other_component in other_components:
        child = other_component["SUBTREE"]
        lingua_graph.add_relation(appos_conj, child, "pred.arg.2")
    
    return appos_conj

def parse_appos_conj(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "APPOS_CONJ"

    node = GPGTextNode(chunk, pos="AppositionalPredicator")  
    lingua_graph.add_node(node)

    return node


def parse_ns_complementizer(chunk, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "NS_COMPLEMENTIZER"

    node = GPGTextNode(chunk, pos="GeneralFunctor")
    lingua_graph.add_node(node)

    return node


def parse_comp_ns(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type in ["COMP_NS", "NS"] 

    if any(sub_chunk_result["TYPE"] == "NS_COMPLEMENTIZER" 
           for sub_chunk, sub_chunk_result in parsing["COMPONENTS"].items()):

        other_types = [x for x in type_view if x != "NS_COMPLEMENTIZER"]

        assert len(other_types) == 1
        other_type = other_types[0]

        child = type_view[other_type][0]["SUBTREE"]

        if "NS_COMPLEMENTIZER" in type_view:
            complementizer = type_view["NS_COMPLEMENTIZER"][0]["SUBTREE"]

            lingua_graph.add_relation(complementizer, child, "func.arg")

            return complementizer
        else:
            return child
    elif any(sub_chunk_result["TYPE"] in ["S", "SS", "CS"]
           for sub_chunk, sub_chunk_result in parsing["COMPONENTS"].items()):
        
        if len(type_view) != 1:
            log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Multiple roles found in the parsing result {type_view}", chunk, parsing_type)
            raise ValueError(f"Multiple roles found in the parsing result {type_view} for chunk {chunk}")

        role, parse_result = next(iter(type_view.items()))
        if len(parse_result) != 1:
            log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Multiple sentences found in the parsing result {type_view}", chunk, parsing_type)
            raise ValueError(f"Multiple sentences found in the parsing result {type_view} for chunk {chunk}")

        # TODO: add puncts here 
        return parse_result[0]["SUBTREE"]
        

    elif any("ROLE" in sub_chunk_result and sub_chunk_result["ROLE"] == "PRED" 
             for sub_chunk, sub_chunk_result in parsing["COMPONENTS"].items()):
        parsing_type = "S"
        return parse_s(chunk, parsing_type, parsing, role_view, type_view, lingua_graph)
    else:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: No NS_COMPLEMENTIZER or PRED found in the parsing result {parsing}", chunk, parsing_type)
        raise ValueError(f"No NS_COMPLEMENTIZER or PRED found in the parsing result {parsing} for chunk {chunk}")
            

    


def parse_rsms(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type in ["RSMS", "CS", "S"]


    rs = [type_view[x] for x in ["RSS", "RCS"]]
    if len(rs) != 1:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Multiple RS found in the parsing result {type_view}", chunk, parsing_type)
        raise ValueError(f"Multiple RS found in the parsing result {type_view} for chunk {chunk}")
    rs_node = rs[0]["SUBTREE"]

    s = [type_view[x] for x in type_view if x not in ["RSMS", "SUBSMRS", "RSMRS", "RCS"]]
    if len(s) != 1:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Multiple S found in the parsing result {type_view}", chunk, parsing_type)
        raise ValueError(f"Multiple S found in the parsing result {type_view} for chunk {chunk}")
    s_node = s[0]["SUBTREE"]

    if isinstance(rs_node, (list, tuple)):
        for n in rs_node:
            lingua_graph.add_relation(s_node, n, "modification")
    else:
        lingua_graph.add_relation(s_node, rs_node, "modification")

    return s_node

    
        
def parse_subsms(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type in ["SUBSMS", "CS", "S"], f"parsing type {parsing_type} is not supported for parse_subsms"

    # if len(type_view) != 2:
    #     log_with_context(logging.ERROR, f"Multiple roles found in the parsing result {type_view} for chunk {chunk}", chunk, parsing_type)
    #     raise ValueError(f"Multiple roles found in the parsing result {type_view} for chunk {chunk}")

    if "SUBS" not in type_view:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: No subordinate sentence found in the parsing result {parsing}", chunk, parsing_type)
        raise ValueError(f"No subordinate sentence found in the parsing result {parsing} for chunk {chunk}")
    
    other_components = list(itertools.chain(*[role_view[x] for x in role_view if x not in {"SUBS", "MOD", "VOC", "DISCOURSE", "INDEX"}]))

    if len(other_components) != 1:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Multiple roles found in the parsing result {type_view}", chunk, parsing_type)
        raise ValueError(f"Multiple roles found in the parsing result {type_view} for chunk {chunk}")

    root = other_components[0]["SUBTREE"]

    for parsing_results in type_view["SUBS"]:
        assert parsing_results["TYPE"] == "SUBS"
        sub_tree_root = parsing_results["SUBTREE"]
        if sub_tree_root.pos == "GeneralFunctor":
            lingua_graph.add_relation(root, sub_tree_root, "modification")
        else:
            lingua_graph.add_relation(root, sub_tree_root, "as:pred.arg.1")
    
    return root

def parse_subs(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "SUBS"

    if "SCP" not in type_view:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: No subordinating conjunction phrase found in the parsing result {parsing}", chunk, parsing_type)
        raise ValueError(f"No subordinating conjunction phrase found in the parsing result {parsing} for chunk {chunk}")
    
    if len(type_view["SCP"]) != 1:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Multiple subordinating conjunction phrases found in the parsing result {type_view}", chunk, parsing_type)
        raise ValueError(f"Multiple subordinating conjunction phrases found in the parsing result {type_view} for chunk {chunk}")

    root = type_view["SCP"][0]["SUBTREE"]

    for type_for_lingua, parsing_results in type_view.items():

        if type_for_lingua in ["SCP"]:
            continue

        for parsing_result in parsing_results:
            sub_tree_root = parsing_result["SUBTREE"]
            place_holder = parsing_result["PLACEHOLDER"]
            if place_holder !=  "{2}":
                log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Unexpected place holder {place_holder} in the parsing result {parsing_result}", chunk, parsing_type)
                raise ValueError(f"Unexpected place holder {place_holder} in the parsing result {parsing_result} for chunk {chunk}")
            if sub_tree_root.pos == "GeneralFunctor":
                lingua_graph.add_relation(root, sub_tree_root, "func.arg")
            else:
                lingua_graph.add_relation(root, sub_tree_root, "pred.arg.2")

    return root

def parse_scp(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "SCP"

    # Parse the subordinating conjunction phrase
    # Check if the chunk contains wh-words
    wh_words = ["when", "where", "why", "how", "what", "which", "who", "whom", "whose"]
    
    # Convert to lowercase for case-insensitive comparison
    chunk_lower = chunk.lower()
    
    # Check if any wh-word is in the chunk
    contains_wh = any(wh_word in chunk_lower.split() for wh_word in wh_words)
    
    if contains_wh:
        # If it contains wh-words, mark it as an InterrogativePredicate
        node = GPGTextNode(chunk, pos="GeneralFunctor")
    else:
        # Otherwise, use the default ModificationalFunctor
        node = GPGTextNode(chunk, pos="LogicalPredicator")
    
    lingua_graph.add_node(node)

    return node

def parse_relative_noun(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "RELATIVE_NOUN"

    node = GPGTextNode(chunk, pos="GeneralFunctor")
    lingua_graph.add_node(node)

    return node
    
    
        
def parse_snp(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "SNP"

    if "COMPONENTS" not in parsing:
        node = GPGTextNode(chunk, pos="NominalConstant")
        lingua_graph.add_node(node)

        return node
    else:
        return parse_np(chunk, "NP", parsing, role_view, type_view, lingua_graph)

def parse_ampw(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "AMPW"
    node = GPGTextNode(chunk, pos="ModificationalFunctor")
    lingua_graph.add_node(node)

    return node

def parse_ccp(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "CCP"
    node = GPGTextNode(chunk.replace("{{", "{").replace("}}", "}"), pos="ConjunctionalFunctor")
    lingua_graph.add_node(node)

    return node


def parse_op(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "OP"
    node = GPGTextNode(chunk.replace("{{", "{").replace("}}", "}"), pos="ExpressionFunctor")
    lingua_graph.add_node(node)

    return node

def parse_sadv(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "SADV"
    node = GPGTextNode(chunk, pos="ModificationalConstant")
    lingua_graph.add_node(node)

    return node

def parse_sadj(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "SADJ"
    node = GPGTextNode(chunk, pos="ModificationalConstant")
    lingua_graph.add_node(node)

    return node

def check_if_svp(chunk: str, parsing: dict, role_view: dict, type_view: dict):
    """
    Check if the chunk is an SVPC.
    """
    if "COMPONENTS" not in parsing:
        return True
    elif any(type in ["NP", "SNP", "SYM", "NPC", "NUMBER", "EXPRESSION", "PP"]
             for type in type_view):
        return False
    else:
        return True
    
def parse_be_verb(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "BE_VERB"    
    node = GPGTextNode(chunk, pos="FactualPredicator")
    lingua_graph.add_node(node)

    return node
    
def parse_aux_verb(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "AUX_VERB"   
    node = GPGTextNode(chunk, pos="FactualPredicator")
    lingua_graph.add_node(node)

    return node
    
    

def parse_slvp(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "SLVP"

    if check_if_svp(chunk, parsing, role_view, type_view):
        node = GPGTextNode(chunk, pos="FactualPredicator")
        lingua_graph.add_node(node)

        if "COMPONENTS" in parsing:
            for sub_chunk in parsing["COMPONENTS"]:
                if "SUBTREE" in parsing["COMPONENTS"][sub_chunk]:
                    parsing["COMPONENTS"][sub_chunk]["SUBTREE"] = None 

        return node
    else:
        return parse_vp(chunk, "VP", parsing, role_view, type_view, lingua_graph)
    

def parse_svp(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type == "SVP"
    if check_if_svp(chunk, parsing, role_view, type_view):

        sub_components = list(itertools.chain(*[role_view[x] for x in role_view]) )
        for comp in sub_components:
            
            child_node = comp["SUBTREE"]
            # print(child_node.text)
            lingua_graph.remove_node(child_node)

        node = GPGTextNode(chunk, pos="FactualPredicator")
        lingua_graph.add_node(node)

        return node
    else:
        node = parse_vp(chunk, "VP", parsing, role_view, type_view, lingua_graph)

    return node

def parse_ss(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the parsing result to the lingua format.
    """
    assert parsing_type in ["SS", "S", "NS"], f"parsing type {parsing_type} is not supported for parse_ss"

    if "SS" in type_view:
        root_node = type_view["SS"][0]["SUBTREE"]
        process_modifications(root_node, role_view, type_view, lingua_graph)
        process_other_relations(root_node, parsing, role_view, type_view, lingua_graph)
        return root_node
    else:

        pred_node = find_pred(chunk, parsing_type, role_view, type_view)

        if pred_node is None:
            log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: No predicate found in the parsing result {type_view}", chunk, parsing_type)
            raise ValueError(f"No predicate found in the parsing result {type_view} for chunk {chunk}")

        root = pred_node

        process_subj_objs(root, role_view, type_view, lingua_graph)
        process_modifications(root, role_view, type_view, lingua_graph)
        process_other_relations(root, parsing, role_view, type_view, lingua_graph)

        if "AUX_VERB" in type_view:
            aux_verb_node = type_view["AUX_VERB"][0]["SUBTREE"]
            if aux_verb_node != root:
                lingua_graph.add_relation(root, aux_verb_node, "aux_verb")

        if "BE_VERB" in type_view:
            be_verb_node = type_view["BE_VERB"][0]["SUBTREE"]
            if be_verb_node != root:
                lingua_graph.add_relation(root, be_verb_node, "aux_verb")

    return root 
                    

def parse_conjunction(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert the sentence conjunction to the lingua format.
    """
    # assert parsing_type in ["RSC", "SUBSC", "SC", "NPC", "VPC", "SVPC", "ADVMC", "PPC"]

    if "CCP" not in type_view:
        root = GPGAuxNode("Parataxis", pos="ConjunctionalFunctor")
        lingua_graph.add_node(root)
    else:
        
        if len(type_view["CCP"]) != 1:
            log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Multiple coordinating conjunction phrases found in the parsing result {parsing}", chunk, parsing_type)
            
        root = type_view["CCP"][0]["SUBTREE"]
    reverse_conj = parsing["TYPE"] in ["RSC", "SUBSC", "ADVMC", "PPC"]
    conj_components = []

    arg_index = 0
    for type_for_lingua, parsing_results in type_view.items():
        if type_for_lingua == "CCP":
            continue

        for parsing_result in parsing_results:
            sub_tree_root = parsing_result["SUBTREE"]
            if "PLACEHOLDER" in parsing_result:
                place_holder = parsing_result["PLACEHOLDER"]
                arg_index = int(place_holder.strip("{}"))
            else:
                arg_index += 1

            if reverse_conj:
                conj_components.append(sub_tree_root)
                lingua_graph.add_relation(sub_tree_root, root,  f"as:func.arg")
            else:
                lingua_graph.add_relation(root, sub_tree_root, f"func.arg")

    return root if not reverse_conj else conj_components

import itertools

def find_head_np(chunk: str, role_view: dict, type_view: dict):

    if "HEAD" in role_view:
        all_possible_head_components = role_view["HEAD"]
    else:
        all_possible_head_components = list(itertools.chain(*[type_view[x] for x in type_view if x in ["SNP", "NP", "SYM", "NPC", "NUMBER", "EXPRESSION"]]))

    if len(all_possible_head_components) == 0:
        log_with_context(logging.ERROR, f"Error parsing type FindHeadNP for chunk [{chunk}]: No center noun phrase found in the parsing result {role_view}", chunk, "find_head_np")
        raise ValueError(f"No center noun phrase found in the parsing result {role_view} for chunk {chunk}")
    elif len(all_possible_head_components) == 1:
        return all_possible_head_components[0]
    else:
        # find last np component as the head
        # if not all(np_component["CHUNK"] in chunk for np_component in all_possible_head_components):
        #     log_with_context(logging.ERROR, f"The sub chunks are not found in the chunk {chunk}", chunk, "find_head_np")
        #     raise ValueError(f"The sub chunks are not found in the chunk {chunk}")
        # Sort NP components by their occurrence index in the chunk
        sorted_np_components = sorted(
            all_possible_head_components, 
            key=lambda np_component: chunk.find(np_component["CHUNK"])
        )
        
        # Return the last NP component as the head
        return sorted_np_components[-1]
        
def parse_acm_marker(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert deverbal marker parsing to Lingua format.
    """
    assert parsing_type == "ACM_MARKER"

    node = GPGTextNode(chunk, pos="PunctuationalConstant")
    lingua_graph.add_node(node)
    return node

def parse_poss_marker(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert possessive marker parsing to Lingua format.
    """
    assert parsing_type == "POSS_MARKER"

    node = GPGTextNode(chunk, pos="ModificationalFunctor")
    lingua_graph.add_node(node)

    return node

def parse_acm_obj(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert deverbal object parsing to Lingua format.
    """
    assert parsing_type == "ACM_OBJ"

    head_component = find_head_np(chunk, role_view, type_view)

    root = head_component["SUBTREE"]

    process_subj_objs(root, role_view, type_view, lingua_graph)
    process_modifications(root, role_view, type_view, lingua_graph)
    process_other_relations(root, parsing, role_view, type_view, lingua_graph)

    if "ACM_MARKER" in type_view:
        
        acm_marker_components = role_view["ACM_MARKER"]
        acm_marker_node = acm_marker_components[0]["SUBTREE"]
        lingua_graph.add_relation(root, acm_marker_node, "punctuation")

    return root


# def parse_svp_ing_ed(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
#     """
#     Convert deverbal subject parsing to Lingua format.
#     """
#     assert parsing_type in ["SVPING", "SVPED"]

#     if "COMPONENTS" in parsing:
#         return parse_vp(chunk, parsing_type, parsing, role_view, type_view, lingua_graph)

#     else:
#         vp_node = GPGTextNode(chunk, pos="FactualPredicator")
#         lingua_graph.add_node(vp_node)

#         return vp_node
    

def parse_acm(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert deverbal subject parsing to Lingua format.
    """
    assert parsing_type == "ACM"


    type_pattern = tuple([parsing["COMPONENTS"][sub_chunk]["TYPE"] for sub_chunk in parsing["COMPONENTS"]])
    nodes = [parsing["COMPONENTS"][sub_chunk]["SUBTREE"] for sub_chunk in parsing["COMPONENTS"]]
    if len(type_pattern) == 2:
        if type_pattern[1] in ["SVP", "VP", "SLVP", "ADJ", "SADJ"]:
            relation_node = nodes[1]
            if type_pattern[1] in {"ADJ", "SADJ"}:
                relation_node.text = "(be) " + relation_node.text
                relation_node.pos = "FactualPredicator"
            root = relation_node
            if type_pattern[0] == "ACM_OBJ" or type_pattern[0] == "ACM_OBJC":
                acm_obj_component = nodes[0]
                lingua_graph.add_relation(root, acm_obj_component, "pred.arg.2")
            else:
                log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: No deverbal object found in the parsing result {parsing}", chunk, parsing_type)
                raise ValueError(f"No deverbal object found in the parsing result {parsing} for chunk {chunk}")
        elif type_pattern[1] in ["NP", "SNP"]:
            root_node = nodes[1]
            root = root_node
            modifications = nodes[0]
            lingua_graph.add_relation(root, modifications, "modification")
            
        else:
            log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Unsupported type pattern {type_pattern} for chunk {chunk}", chunk, parsing_type)
            raise ValueError(f"Unsupported type pattern {type_pattern} for chunk {chunk}")

    elif len(type_pattern) == 3:
        if type_pattern[1] != "ACM_MARKER":
            log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Unsupported type pattern {type_pattern} for chunk {chunk}", chunk, parsing_type)
            raise ValueError(f"Unsupported type pattern {type_pattern} for chunk {chunk}")
        
        marker_node = nodes[1]
        if type_pattern[2] in ["SVP", "VP", "SLVP", "ADJ", "SADJ"]:
            relation_node = nodes[2]
            if type_pattern[1] in {"ADJ", "SADJ"}:
                relation_node.text = "(be) " + relation_node.text
                relation_node.pos = "FactualPredicator"
            acm_obj_component = nodes[0]
            root = relation_node
            lingua_graph.add_relation(root, acm_obj_component, "pred.arg.2")
            lingua_graph.add_relation(acm_obj_component, marker_node, "punctuation")
        elif type_pattern[2] in ["NP", "SNP"]:
            root_node = nodes[2]
            root = root_node
            modifications = nodes[0]
            lingua_graph.add_relation(root, modifications, "modification")
            lingua_graph.add_relation(modifications, marker_node, "punctuation")
        else:
            log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Unsupported type pattern {type_pattern} for chunk {chunk}", chunk, parsing_type)
            raise ValueError(f"Unsupported type pattern {type_pattern} for chunk {chunk}")
        
    else:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Unsupported type pattern {type_pattern} for chunk {chunk}", chunk, parsing_type)
        raise ValueError(f"Unsupported type pattern {type_pattern} for chunk {chunk}")

    return root
    
    
def parse_poss_snp(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert possessive noun phrase parsing to Lingua format.
    """
    assert parsing_type == "POSS_SNP"


    if "POSS_MARKER" not in role_view:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: No possessive marker found in the parsing result {parsing}", chunk, parsing_type)
        raise ValueError(f"No possessive marker found in the parsing result {parsing} for chunk {chunk}")
    
    poss_marker_components = role_view["POSS_MARKER"]

    if len(poss_marker_components) != 1:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Multiple possessive markers found in the parsing result {parsing}", chunk, parsing_type)
        raise ValueError(f"Multiple possessive markers found in the parsing result {parsing} for chunk {chunk}")
    
    poss_marker_component = poss_marker_components[0]
    poss_marker_node = poss_marker_component["SUBTREE"]
    
    owner_components = role_view["OWNER"]

    if len(owner_components) != 1:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Multiple owners found in the parsing result {parsing}", chunk, parsing_type)
        raise ValueError(f"Multiple owners found in the parsing result {parsing} for chunk {chunk}")
    
    owner_component = owner_components[0]
    owner_node = owner_component["SUBTREE"]

    possession_components = role_view["POSSESSION"]

    if len(possession_components) > 1:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: Multiple possessions found in the parsing result {parsing}", chunk, parsing_type)
        raise ValueError(f"Multiple possessions found in the parsing result {parsing} for chunk {chunk}")
    
    if len(possession_components) == 0:
        possession_node = GPGTextNode("Missing", pos="NominalConstant")
    else:
        possession_component = possession_components[0]
        possession_node = possession_component["SUBTREE"]

    root = possession_node

    lingua_graph.add_relation(root, poss_marker_node, "as:pred.arg.1")
    lingua_graph.add_relation(poss_marker_node, owner_node, "pred.arg.2")
    
    return root
    
    
def parse_np(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert noun phrase parsing to Lingua format.
    """
    assert parsing_type in ["NP"], f"parsing type {parsing_type} is not supported"

    head_component = find_head_np(chunk, role_view, type_view)

    root = head_component["SUBTREE"]

    process_subj_objs(root, role_view, type_view, lingua_graph)
    process_modifications(root, role_view, type_view, lingua_graph)
    process_other_relations(root, parsing, role_view, type_view, lingua_graph)

    if "THAT_RS" in type_view:
        that_rs_node = type_view["THAT_RS"][0]["SUBTREE"]
        lingua_graph.add_relation(root, that_rs_node, "ref")
    
    if "WH_RS" in type_view:
        wh_rs_node = type_view["WH_RS"][0]["SUBTREE"]
        lingua_graph.add_relation(root, wh_rs_node, "ref")

    for sub_chunk in parsing["COMPONENTS"]:
        if "SUBTREE" in parsing["COMPONENTS"][sub_chunk]:
            subtree = parsing["COMPONENTS"][sub_chunk]["SUBTREE"]
            subtree_type = parsing["COMPONENTS"][sub_chunk]["TYPE"]
            if subtree == root or subtree is None:
                continue
            if not lingua_graph.has_edge(root, subtree):
                if subtree_type in {"SYM", "SNP", "NP", "EXPRESSION", "NUMBER", "OTHER", "OP", "LIST"}:
                    lingua_graph.add_relation(root, subtree, "modification")
            

    if "RELATION" in head_component:
        relations = head_component["RELATION"]
    else:
        relations = []

    # process_disconnected_subcomponents(root, parsing_type, parsing, role_view, type_view, lingua_graph)

    sub_nodes = []
    for sub_chunk in parsing["COMPONENTS"]:
        if "SUBTREE" in parsing["COMPONENTS"][sub_chunk]:
            subtree = parsing["COMPONENTS"][sub_chunk]["SUBTREE"]
            if subtree == root:
                continue
            if isinstance(subtree, (list, tuple)):
                sub_nodes.extend(subtree)
            else:
                sub_nodes.append(subtree)

    for child in sub_nodes:
        process_noun_relations(root, relations, child, role_view, type_view, lingua_graph)

    return root

def find_pred(chunk: str, parsing_type: str, role_view: dict, type_view: dict):

    possible_pred_types = {"SVP", "SLVP", "VP", "VPC", "BE_VERB", "AUX_VERB"}
    if "PRED" in role_view or "SPRED" in role_view:
        pred_components = role_view["PRED"] if "PRED" in role_view else role_view["SPRED"]
    elif "HEAD" in role_view and role_view["HEAD"][0]["TYPE"] in possible_pred_types:
        pred_components = role_view["HEAD"]
    else:
        pred_components = list(itertools.chain(*[role_view[x] for x in role_view if x in possible_pred_types]))

    if len(pred_components) == 0:
        return None
    elif len(pred_components) > 1:
        log_with_context(logging.WARNING, f"Multiple predicates found in the parsing result {type_view} for chunk {chunk}", chunk, parsing_type)
    
    pred_component = pred_components[-1]["SUBTREE"]

    return pred_component

def parse_vp(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert verb phrase parsing to Lingua format.
    """
    assert parsing_type in ["VP"]
    
    pred_node = find_pred(chunk, parsing_type, role_view, type_view)

    if pred_node is None:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: No predicate found in the parsing result {type_view}", chunk, parsing_type)
        raise ValueError(f"No predicate found in the parsing result {type_view} for chunk {chunk}")

    root = pred_node

    process_subj_objs(root, role_view, type_view, lingua_graph)
    process_modifications(root, role_view, type_view, lingua_graph)
    process_other_relations(root, parsing, role_view, type_view, lingua_graph)

    if "AUX_VERB" in type_view:
        aux_verb_node = type_view["AUX_VERB"][0]["SUBTREE"]
        if aux_verb_node != root:
            lingua_graph.add_relation(root, aux_verb_node, "aux_verb")

    if "BE_VERB" in type_view:
        be_verb_node = type_view["BE_VERB"][0]["SUBTREE"]
        if be_verb_node != root:
            lingua_graph.add_relation(root, be_verb_node, "aux_verb")

    return root

def parse_pp(chunk: str, parsing_type: str, parsing: dict, role_view: dict, type_view: dict, lingua_graph: TextGPGraph):
    """
    Convert prepositional phrase parsing to Lingua format.
    """
    assert parsing_type in ["PP"]
    
    # Handle simple PP
    if "AMPW" not in type_view:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: No prepositional word found in the parsing result {parsing}", chunk, parsing_type)
        raise ValueError(f"No prepositional word found in the parsing result {parsing} for chunk {chunk}")
    
    root = type_view["AMPW"][0]["SUBTREE"]

    other_components = list(itertools.chain(*[type_view[x] for x in type_view if x not in ["AMPW"]]))

    if len(other_components) == 0:
        log_with_context(logging.ERROR, f"Error parsing type {parsing_type} for chunk [{chunk}]: No other components found in the parsing result {parsing}", chunk, parsing_type)
        raise ValueError(f"No other components found in the parsing result {parsing} for chunk {chunk}")

    other_component = other_components[0]
    other_node = other_component["SUBTREE"]

    lingua_graph.add_relation(root, other_node, "pred.arg.2")
    
    return root

def post_process_lingua_graph(lingua_graph: TextGPGraph):
    """
    Post process the Lingua graph.
    """
    for node in lingua_graph.nodes():
        if node.pos == "DetConstant":
            node.pos = "DeterminerConstant"
        elif node.pos == "IntjConstant":
            node.pos = "InterjectionConstant"
        elif node.pos == "SymConstant":
            node.pos = "SymbolConstant"
        elif node.pos == "OtherConstant":
            node.pos = "OtherConstant"
        elif node.pos == "NumberConstant":
            node.pos = "NominalConstant"
        elif node.pos == "ModificationalFunctor":
            node.pos = "ModificationalFunctor"

        if isinstance(node, GPGTextNode) and node.text.strip().endswith("{2}") and "{1}" not in node.text \
            and len(node.text.replace("{2}", "").strip().split(" ")) == 1:
            print("missing {1} in ", node.text)
            node.text = "{1} " + node.text 
        
        for child, edge in lingua_graph.children(node):
            if child.pos == "FactualPredicator" and edge.label == "aux_verb":
                node.text = child.text + " " + node.text
                lingua_graph.remove_node(child)

def llmparsing2lingua(sentence: str, parsing_result: dict, retrun_debug_info: bool = False):
    """
    Convert the parsing result to the lingua format.
    """
    clear_debug_info()

    logger.info(f"Parsing sentence: [{sentence}]")

    visitor = LLMParsingResultVisitor()
    parse_llm_parsing_result(sentence, parsing_result, visitor)

    lingua_graph = visitor.lingua_graph
    post_process_lingua_graph(lingua_graph)

    logger.info(f"Finished parsing sentence: [{sentence}]")

    if retrun_debug_info:
        return lingua_graph, get_debug_info()
    else:
        return lingua_graph

def get_debug_info():
    """
    Get the current debug information dictionary.
    Returns:
        Dict[str, List[Dict[str, str]]]: Dictionary containing debug information organized by log level
    """
    return dict(debug_info_dict)

def clear_debug_info():
    """
    Clear all debug information.
    """
    debug_info_dict.clear()

def get_debug_info_by_level(level: str):
    """
    Get debug information for a specific log level.
    Args:
        level (str): Log level to get information for (e.g., 'error', 'warning', 'info')
    Returns:
        List[Dict[str, str]]: List of debug information entries for the specified level
    """
    return debug_info_dict.get(level.lower(), [])

def get_debug_info_by_chunk(chunk: str):
    """
    Get debug information for a specific chunk.
    Args:
        chunk (str): The chunk to get debug information for
    Returns:
        List[Dict[str, str]]: List of debug information entries for the specified chunk
    """
    all_info = []
    for level_info in debug_info_dict.values():
        for entry in level_info:
            if entry.get('chunk') == chunk:
                all_info.append(entry)
    return all_info
