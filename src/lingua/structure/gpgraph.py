"""
lingua graph
"""
# import numpy as np

# logger = logging.getLogger(__name__)
import copy
import json

import re
from enum import Enum
from typing import Union

from .basegraph import Node, Edge, DirectedGraph, LearnableGraph, GraphVisualizer
from .utils import positions2spans, CompactJSONEncoder

arg_placeholder_pattern = re.compile(r'{\d+}')

def sent_spans2list(spans):
    spans_list = list()
    for span in spans:
        if isinstance(span, str):
            spans_list.append({'label': span})
        else:
            assert isinstance(span, tuple) and len(span) == 2
            spans_list.append({'start': span[0], 'end': span[1]})
    return spans_list

def is_arg_placeholder(string):
    """

    @param string:
    @return:
    """
    return re.match(arg_placeholder_pattern, string) is not None



class GPGNode(Node):
    """
    GPGNode
    """

    def __init__(self, id=None, pos=None, confidence=None, *args, **kwargs):
        
        self.id = id
        self.pos = pos
        self.confidence = confidence

    @property
    def ID(self):
        """

        :return:
        """
        return self.id

    @ID.setter
    def ID(self, id):
        """
        setter
        """
        self.id = id

    def __hash__(self):
        """

        :return:
        """
        return hash(self.ID)

    def __eq__(self, another):
        """

        :param another:
        :return:
        """
        return self.ID == another.ID

    def copy(self):
        return copy.deepcopy(self)


def get_start(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    elif isinstance(x, int):
        return x
    else:
        raise ValueError('unexpected span')

def standardize_spans(spans):
    """

    @param spans:
    @type spans:
    @return:
    @rtype:
    """
    standardized = []
    # deduplicated = []
    # span_set = set()
    # for span in spans:
    #     if isinstance(span, int):
    #         span = (span, span)
    #     if tuple(span) not in span_set:
    #         deduplicated.append(span)
    #         span_set.add(tuple(span))
    #     else:
    #         continue
    
    # spans = deduplicated
    
    idx = 0
    while idx < len(spans):
        span = spans[idx]
        if isinstance(span, int):
            standardized.append((span, span))
        elif isinstance(span, str):
            standardized.append(span)
        # elif isinstance(span, tuple) and isinstance(span[0], str):
        #     standardized.append(span[0])
        elif isinstance(span, (tuple, list)):
            assert len(span) == 2
            standardized.append(tuple(span))
            # we merge next span if it is continuous
            # to_merge = idx
            # break_continuous = False
            # for j in range(idx + 1, len(spans)):
            #     if not isinstance(spans[j], (tuple, list)):
            #         break
            #     if spans[j][0] == spans[j-1][1] + 1 and not break_continuous:
            #         to_merge = j
            #     else:
            #         break_continuous = True
            #
            # merged_span = (span[0], spans[to_merge][1])
            # idx += to_merge - idx
            # standardized.append(tuple(merged_span))
        else:
            raise ValueError('Invalid span: {}'.format(span))
        idx += 1

    return tuple(standardized)


def readable_spans(spans):
    """

    @param spans:
    @type spans:
    @return:
    @rtype:
    """

    readable = []
    for span in spans:
        if isinstance(span, int):
            readable.append(span)
        elif isinstance(span, str):
            readable.append(span)
        else:
            start, end = span
            if start == end:
                span = start

            readable.append(span)

    return tuple(readable)


class GPGPhraseNode(GPGNode):
    """
    GPGPhraseNode
    """

    def __init__(self, spans=None, id=None, pos=None, confidence=None, *args, **kwargs):
        super().__init__(id=id, pos=pos, confidence=confidence, *args, **kwargs)
        
        if spans is not None:
            self._spans = standardize_spans(spans)
        else:
            self._spans = None
        self.contexts = list()
    #

    @staticmethod
    def merge_continuous_spans(spans):
        new_spans = list()
        for span in spans:
            if isinstance(span, int):
                new_spans.append((span, span))
            else:
                new_spans.append(span)
        spans = new_spans
        span_list = list()
        start, end = None, None
        for idx, span in enumerate(spans):
            if isinstance(span, str):
                if start is not None:
                    span_list.append((start, end))
                span_list.append(span)
                start, end = None, None
            else:
                s, e = span
                if len(spans) == 1:
                    span_list.append(span)
                    break
                elif idx == len(spans) - 1:
                    if start is None:
                        span_list.append(span)
                    else:
                        if end + 1 == s:
                            span_list.append((start, e))
                        else:
                            span_list.append((start, end))
                            span_list.append(span)
                else:
                    if start is None:
                        start, end = s, e
                    else:
                        if end + 1 == s:
                            end = e
                        else:
                            span_list.append((start, end))
                            start, end = s, e
        return tuple(span_list)
    #
    # @staticmethod
    # def sort_non_str_spans(spans):
    #     if not any(isinstance(x, str) for x in spans):
    #         sorted_spans = sorted(spans, key=lambda x:get_start(x))
    #         return sorted_spans
    #     else:
    #         return spans

    def has_symbols(self):
        """

        @return:
        @rtype:
        """

        for span in self._spans:
            if isinstance(span, str):
                return True

        return False
    #
    # def add_span_to_head(self, span):
    #     """
    #
    #     @param span:
    #     @type span:
    #     @return:
    #     @rtype:
    #     """
    #     if isinstance(span, str):
    #         self._spans = list(self._spans)
    #         self._spans.insert(0, span)
    #         self._spans = tuple(self._spans)
    #         return
    #
    #     if isinstance(span, int):
    #         span = [span, span]
    #     if isinstance(self._spans[0], (list, tuple)) and \
    #             span[1] == self._spans[0][0] - 1:
    #         self._spans = tuple([(span[0], self._spans[0][1])] + list(self._spans)[1:])
    #     else:
    #         x = list(self._spans)
    #         x.insert(0, span)
    #         self._spans = tuple(x)
    #
    # def remove_span_from_head(self, span):
    #     """
    #
    #     @param span:
    #     @type span:
    #     @return:
    #     @rtype:
    #     """
    #     if isinstance(span, tuple):
    #         span_list = list(self._spans)
    #         for s in list(span):
    #             span_list.remove(s)
    #             self._spans = standardize_spans(span_list)
    #
    # def add_span_to_tail(self, span):
    #     """
    #
    #     @param span:
    #     @type span:
    #     @return:
    #     @rtype:
    #     """
    #     if isinstance(span, str):
    #         self._spans = list(self._spans)
    #         self._spans.append(span)
    #         self._spans = tuple(self._spans)
    #         return
    #
    #     if isinstance(span, int):
    #         span = [span, span]
    #
    #     if isinstance(self._spans[-1], (list, tuple)) and \
    #             span[0] == self._spans[-1][1] + 1:
    #         self._spans = tuple(list(self._spans)[:-1] + [(self._spans[-1][0], span[1])])
    #     else:
    #         x = list(self._spans)
    #         x.append(span)
    #         self._spans = tuple(x)
    #
    # def add_spans_to_head(self, spans):
    #     """
    #
    #     @param span:
    #     @type span:
    #     @return:
    #     @rtype:
    #     """
    #
    #     for span in reversed(spans):
    #         self.add_span_to_head(span)
    #
    # def add_spans_to_tail(self, spans):
    #     """
    #
    #     @param span:
    #     @type span:
    #     @return:
    #     @rtype:
    #     """
    #
    #     for span in spans:
    #         self.add_span_to_tail(span)

    @property
    def spans(self):
        """

        @return:
        @rtype:
        """
        return self._spans

    @spans.setter
    def spans(self, spans):
        """

        @param spans:
        @type spans:
        @return:
        @rtype:
        """
        raise Exception("spans should not be set directly. please use GPGraph.modify_node_spans")

        # self._spans = standardize_spans(spans)

    @property
    def readable_spans(self):
        """

        @return:
        @rtype:
        """
        return readable_spans(self._spans)

    def __str__(self):
        """

        :return:
        """
        return ",".join(map(str, self.spans))

    def __contains__(self, word_id):
        """

        :param word_id:
        :return:
        """

        for span in self._spans:
            if isinstance(span, str):
                if span == word_id:
                    return True
                else:
                    continue
            else:
                start, end = span
                if start <= word_id <= end:
                    return True
        return False

    def words(self, with_aux=True):
        """

        :return:
        """
        for span in self._spans:
            if isinstance(span, str):
                if with_aux:
                    yield span
            else:
                start, end = span
                for i in range(start, end + 1):
                    yield i

    def indexes(self, with_aux=True):
        """

        :return:
        """
        for span in self._spans:
            if isinstance(span, str):
                pass
            else:
                start, end = span
                for i in range(start, end + 1):
                    yield i


class GPGAuxNode(GPGNode):
    """
    GPGNode
    """

    def __init__(self, label=None, id=None, pos=None, confidence=None, *args, **kwargs):
        
        super().__init__(id=id, pos=pos, confidence=confidence, *args, **kwargs)
        self.label = label
        self.contexts = list()

    def __str__(self):
        """

        :return:
        """
        return self.label

class GPGTextNode(GPGNode):

    def __init__(self, text, pos, confidence=None):
        super().__init__()
        self.text = text
        self.pos = pos
        self.confidence = confidence

    def __str__(self):
        """

        :return:
        """
        return self.text



import networkx as nx


class GPGEdge(Edge):
    """
    a set of relations between a pair of nodes for multiple edges.
    behaves like a single relation (that is, a string)
    for code compatability
    """

    def __init__(self, label=None, mod=False, confidence=None):
        super().__init__()
        self.label = label
        self.confidence = None
        self.mod = mod
        self.contexts = []
        self.confidence = confidence

    def __str__(self):

        return self.label

    def __bool__(self):
        """

        :return:
        """
        return self.label is not None

    @property
    def value(self):
        """
        :return:
        :rtype:
        """
        return self.label

    @value.setter
    def value(self, value):
        """
        :return:
        :rtype:
        """
        self.label = value


def _empty_hook():
    """
    _empty_hook
    """

    pass

class GraphRootMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_ = None

    @property
    def root(self):
        """
        Return the root
            * roots = [nodes with zero in-degree].
            * if only one root found, return it, whether it is virtual or not
            * else return the node with label == 'Root', that is, the virtual root
        Contracts:
            * if no virtual root, the program is running on the training data, and there is only one
              node with zero in-degree
        Returns:
        """
        if self.root_ is None:
            topo_roots = [node for node in self.nodes()
                          if self.g.in_degree[node.ID] == 0]

            if len(topo_roots) == 1:
                self.root_ = topo_roots[0]
            else:
                virtual_roots = [node for node in self.nodes()
                                 if isinstance(node, GPGAuxNode) and node.label == 'Root']
                if len(virtual_roots) == 1:
                    self.root_ = virtual_roots[0]
                else:
                    raise Exception("Bad Graph with multiple roots or no roots. ")

        return self.root_

    @root.setter
    def root(self, value):
        self.root_ = value

    def update_root(self):
        topo_roots = [node for node in self.nodes()
                          if self.g.in_degree[node.ID] == 0]

        if len(topo_roots) == 1:
            self.root_ = topo_roots[0]
        else:
            virtual_roots = [node for node in self.nodes()
                                if isinstance(node, GPGAuxNode) and node.label == 'Root']
            if len(virtual_roots) == 1:
                self.root_ = virtual_roots[0]
            else:
                raise Exception("Bad Graph with multiple roots or no roots. ")


class TextGPGraph(DirectedGraph, LearnableGraph, GraphRootMixin):

    def add_node(self, node: Union[GPGTextNode, GPGAuxNode], reuse_id=False):
        """
        Add a node to the graph
        """
        assert isinstance(node, (GPGTextNode, GPGAuxNode)), type(node)
        return super().add_node(node, reuse_id=reuse_id)

    def add_edge(self, ni, nj, e):
        """
        Add an edge to the graph
        """
        if isinstance(e, str):
            e = GPGEdge(label=e)

        return super().add_edge(ni, nj, e)

    def node_text(self, node, interval=" "):
        """
        Return the text of a node
        """
        if isinstance(node, GPGAuxNode):
            return node.label
        else:
            return node.text
        
    def add_relation(self, ni, nj, rel):
        """
        Add an edge to the graph
        """
        if isinstance(rel, str):
            edge = GPGEdge(rel)
        elif isinstance(rel, GPGEdge):
            confidence = rel.confidence
            edge = GPGEdge(rel.label, confidence=confidence)
        else:
            raise Exception("Unknown rel type")
        return super().add_edge(ni, nj, edge)
    
    def remove_relation(self, ni, nj):
        """
        Remove an edge from the graph
        """
        super().remove_edge_between(ni, nj)

class GPGraph(DirectedGraph, LearnableGraph, GraphRootMixin):
    """
    Dependency graph
    """

    def __init__(self, g=None):

        super().__init__(g=g)

        self.meta = dict()
        self.words = []
        self.context_hook = _empty_hook


        self.spans2node = dict()

        if isinstance(g, GPGraph):
            self.meta = g.meta
            self.words = g.words
            self._root = g.root_
            self.spans2node = g.spans2node

    def __copy__(self):
        """

        :return:
        """
        from copy import copy
        copied = DirectedGraph.__copy__(self)
        copied.meta = copy(self.meta)
        copied.words = copy(self.words)
        copied.context_hook = self.context_hook
        return copied

    def __deepcopy__(self, memodict={}):
        """

        :param memodict:
        :type memodict:
        :return:
        :rtype:
        """

        from copy import deepcopy

        copied = DirectedGraph.__deepcopy__(self)
        copied.meta = deepcopy(self.meta)
        
        copied.words = deepcopy(self.words)
        copied.context_hook = deepcopy(self.context_hook)
        return copied



    def set_words(self, words):
        """

        :param words:
        :return:
        """
        if isinstance(words, list):
            self.words = words
        else:
            raise Exception("words input not list.")
    #
    # def modify_node_spans(self, node, spans):
    #     original_spans = node.spans
    #     node.spans = spans
    #     del self.spans2node[original_spans]
    #     self.spans2node[node.spans] = node

    def add_words(self, words, pos=None):
        """

        :param head:
        :return:
        """
        #
        # if len(words) == 1 and isinstance(words[0], float):
        #     words = tuple(words)
        #     if words not in self.spans2node:
        #         self.spans2node[words] = self.add_aux("(be)")
        #
        #     return self.spans2node[words]
        #
        # if any(isinstance(x, float) for x in words):
        #     raise Exception("float found")

        spans = positions2spans(words)

        return self.add_spans(spans, pos)

    def set_context_hook(self, hook):
        """

        :param hook:
        :return:
        """
        self.context_hook = hook

    def clear_context_hook(self):
        """

        :param hook:
        :return:
        """
        self.context_hook = _empty_hook
    
    def add_node(self, node, reuse_id=False):
        """

        :param node:
        :return:
        """
        assert isinstance(node, GPGPhraseNode) or isinstance(node, GPGAuxNode), f"Invalid node type: {type(node)}"

        context = self.context_hook()
        if context:
            node.contexts.extend(context)

        if isinstance(node, GPGPhraseNode):
            node._spans = standardize_spans(node.spans)

            if node.spans in self.spans2node:
                raise Exception("Repeated node found: {}".format(node.spans))
            self.spans2node[node.spans] = super().add_node(node, reuse_id=reuse_id)
            return self.spans2node[node.spans]
        else:
            node = super().add_node(node, reuse_id=reuse_id)
            return node

    def modify_node_spans(self, node, spans):
        """
        Modify the spans of a node
        """
        assert node.spans in self.spans2node, f"Node spans {node.spans} not found in spans2node: {self.spans2node}"
        if self.spans2node[node.spans] == node:
            del self.spans2node[node.spans]
        node._spans = standardize_spans(spans)
        self.spans2node[node.spans] = node

        
    def remove_node(self, node):
        if isinstance(node, GPGPhraseNode):
            # node.spans = standardize_spans(node.spans)
            if node.spans in self.spans2node:
                del self.spans2node[node.spans]
        node = super().remove_node(node)
        return node

    def add_spans(self, spans, pos=None):
        """

        :param node:
        :return:
        """

        spans = standardize_spans(spans)

        if spans not in self.spans2node:
            added_node = GPGPhraseNode(spans)
            added_node.pos = pos
            self.spans2node[spans] = super().add_node(added_node)

        return self.spans2node[spans]

    def has_node(self, node: GPGNode):
        """

        @param node:
        @type node:
        @return:
        @rtype:
        """
        if self.g.has_node(node.ID):
            return True
        else:
            return False

    def has_word(self, words):
        """

        :param word:
        :return:
        """
        spans = positions2spans(words)
        for node in self.nodes():
            if isinstance(node, GPGPhraseNode) and node.spans == spans:
                added_node = node
                return True
        return False

    def get_node_by_words(self, positions):
        """

        @param positions:
        @type positions:
        @return:
        @rtype:
        """
        spans = positions2spans(positions)

        spans = standardize_spans(spans)

        if spans in self.spans2node:
            return self.spans2node[spans]

        return None
    
    def get_node_by_spans(self, spans):
        spans = standardize_spans(spans)
        if spans in self.spans2node:
            return self.spans2node[spans]
        return None

    def has_relation(self, node1: GPGNode,
                     node2: GPGNode,
                     direct_link=True):
        """

        :return:
        @param node1:
        @type node1:
        @param node2:
        @type node2:
        """

        if node1 is None or node2 is None:
            return False

        if direct_link and (self.g.has_edge(node1.ID, node2.ID) or self.g.has_edge(node2.ID, node1.ID)):
            return True
        elif not direct_link and (
                nx.algorithms.shortest_paths.generic.has_path(self.g, node1.ID, node2.ID) or
                nx.algorithms.shortest_paths.generic.has_path(self.g, node2.ID, node1.ID)):
            return True

        return False

    def add_aux(self, label, pos=None):
        """

        :param label:
        :return:
        """

        node = GPGAuxNode(label)
        node.pos = pos
        self.add_node(node)

        return node

    def get_aux(self, label):
        """

        :param label:
        :return:
        """
        for node_id in self.g.nodes:
            node = self.get_node(node_id)
            if isinstance(node, GPGAuxNode) and node.label == label:
                yield node

    def get_edge(self, node1, node2):
        """

        :param node1:
        :param node2:
        :return:
        """
        try:
            return super().get_edge(node1, node2)
        except:
            return None

    def spans(self):
        """

        @return:
        @rtype:
        """

        spans = []

        for x in self.nodes():
            if isinstance(x, GPGPhraseNode):
                for span in x.spans:
                    if isinstance(span, str):
                        continue
                    elif isinstance(span, int):
                        span = (span, span)
                        spans.append(span)
                    else:
                        spans.append(tuple(span))

        spans.sort(key=lambda x: x[0])

        return spans

    def parents_on_path(self, node, ancestor):
        """

        :param node:
        :param ancestor:
        :return:
        """
        for path in nx.all_simple_paths(self.g, ancestor.ID, node.ID):
            yield self.get_node(path[-2])

    def paths(self, node1, node2):
        """

        :param node1:
        :param node2:
        :return:
        """
        for path in nx.all_simple_paths(self.g, node1.ID, node2.ID):
            yield [self.get_node(x) for x in path]

    def replace(self, old_node, new_node):
        """

        :param old_node:
        :param new_node:
        :return:
        """

        if new_node.ID == old_node.ID:
            raise Exception("Bad business logic: cannot replace a node with itself")

        # if self.g.has_node(new_node.ID):
        #     raise Exception("Bad business logic: the new node is already in the graph")

        if not self.has_node(new_node):
            self.add_node(new_node)

        for child, rel in self.children(old_node):
            self.g.add_edge(new_node.ID, child.ID, Edge=rel)

        for parent, rel in self.parents(old_node):
            self.g.add_edge(parent.ID, new_node.ID, Edge=rel)

        self.remove_node(old_node)

    def add_edge(self, start_node, end_node, edge):
        """

        @param start_node:
        @type start_node:
        @param end_node:
        @type end_node:
        @param edge:
        @type edge:
        @return:
        @rtype:
        """
        context = self.context_hook()
        if context:
            edge.contexts.extend(context)

        if start_node.ID not in self.g.nodes():
            raise ValueError('start node not in graph')
        if end_node.ID not in self.g.nodes():
            raise ValueError('end node not in graph')

        if isinstance(edge, str):
            edge = GPGEdge(label=edge)

        # self.add_node(start_node, reuse_id=True)
        # self.add_node(end_node, reuse_id=True)
        edge.start = start_node.ID
        edge.end = end_node.ID
        self.g.add_edge(start_node.ID, end_node.ID, Edge=edge)

    # def add_argument(self, pred_node, arg_node, index, mod=False):
    #     """

    #     :param node1:
    #     :param node2:
    #     :param rel:
    #     :return:
    #     """

    #     # if isinstance(pred_node.ID, int) or isinstance(pred_node.ID, str):
    #     #     raise Exception("Bad ID")
    #     # if isinstance(arg_node.ID, int) or isinstance(arg_node.ID, str):
    #     #     raise Exception("Bad ID")

    #     if mod:

    #         if any(self.node_text(arg_node).lower().startswith(x)
    #                for x in {"what ", "which ", "where ", "who ",
    #                          "whom ", "when ", "why ", "how "}):
    #             edge_label = "func.arg"
    #         else:
    #             edge_label = "as:pred.arg.{0}".format(index)

    #         edge = GPGEdge(label=edge_label, mod=mod)

    #         self.add_edge(arg_node, pred_node, edge)
    #     else:

    #         edge_label = "pred.arg.{0}".format(index)

    #         edge = GPGEdge(label=edge_label, mod=mod)
    #         self.add_edge(pred_node, arg_node, edge)

    # def add_mod(self, modifier, center):
    #     """

    #     :param target:
    #     :param source:
    #     :return:
    #     """

    #     edge = GPGEdge(label="modification", mod=False)

    #     self.add_edge(center, modifier, edge)

    # def add_function(self, functor, argument, index=None):
    #     """

    #     :param functor:
    #     :param argument:
    #     :return:
    #     """
    #     if index is None:
    #         index = "1"
    #     edge_label = "func.arg" #.format(index)

    #     #        if isinstance(functor.ID, int) or isinstance(functor.ID, str):
    #     #            raise Exception("Bad ID")
    #     #        if isinstance(argument.ID, int) or isinstance(argument.ID, str):
    #     #            raise Exception("Bad ID")

    #     edge = GPGEdge(label=edge_label, mod=False)

    #     functor.is_func = True

    #     self.add_edge(functor, argument, edge)

    # def add_ref(self, source, ref):
    #     """

    #     :param target:
    #     :param source:
    #     :return:
    #     """

    #     edge = GPGEdge(label="ref", mod=False)

    #     self.add_edge(source, ref, edge)

    def add_relation(self, node1, node2, rel, confidence=None):
        """

        :param node1:
        :param node2:
        :param rel:
        :return:
        """
        if isinstance(rel, str):
            edge = GPGEdge(rel, confidence=confidence)
        elif isinstance(rel, GPGEdge):
            if confidence is None:
                confidence = rel.confidence
            edge = GPGEdge(rel.label, confidence=confidence)
        else:
            raise Exception("Unknown rel type")
        self.add_edge(node1, node2, edge)

    def merge_continuous_spans(self):
        nodes = list(self.nodes())
        for node in nodes:
            if isinstance(node, GPGAuxNode):
                continue
    
            spans = node.spans
            span_tuple = GPGPhraseNode.merge_continuous_spans(spans)
            # print('old', spans)
            # print('new', span_tuple)
            #del self.spans2node[node.spans]
            self.modify_node_spans(node, span_tuple)
    
            #node.spans = span_list
            #self.spans2node[node.spans] = node

    def remove_relation(self, node1, node2):
        """

        :param node1:
        :param node2:
        :return:
        """
        super().remove_edge_between(node1, node2)

    def relations(self):
        """

        :return:
        """
        return super().edges()

    def node_text(self, node, interval=""):
        """

        @param node:
        @type node:
        @return:
        @rtype:
        """

        if isinstance(node, GPGPhraseNode):
            node_texts = []
            for span in node.spans:
                if isinstance(span, str):
                    node_texts.append(span)
                elif isinstance(span, tuple) and isinstance(span[0], str):
                    node_texts.append(span[0])
                else:
                    start, end = span
                    for i in range(start, end + 1):
                        node_texts.append(self.words[i])

            node_text = interval.join(node_texts)
        else:
            node_text = node.label

        return node_text

    def topological_sort(self):
        """

        :return:
        """

        for id in nx.topological_sort(self.g):
            yield self.get_node(id)

    @staticmethod
    def parse(json_obj):
        """
        Parse a JSON object into an GPGraph.

        :param json_obj: JSON object representing the graph
        :param check_valid: Whether to validate the graph after parsing
        :return: GPGraph instance
        """
        if isinstance(json_obj, str):
            json_obj = json.loads(json_obj)

        assert isinstance(json_obj, dict)

        graph = GPGraph()
        graph.meta = json_obj["meta"]
        graph.words = [0] * len(json_obj["words"])
        if isinstance(json_obj["words"][0], list):
            for id, word in json_obj["words"]:
                graph.words[id] = word
        elif isinstance(json_obj["words"][0], str):
            for id, word in enumerate(json_obj["words"]):
                graph.words[id] = word
        else:
            raise TypeError("Invalid word format")
        
        if len(json_obj["graph"]["nodes"]) == 0:
            return graph

        nodes = dict()

        for node_info in json_obj["graph"]["nodes"]:
            if isinstance(node_info, list):
                id, spans, pos = node_info[:3]
                confidence = node_info[3] if len(node_info) > 3 else None
            elif isinstance(node_info, dict):
                spans = node_info['spans']
                pos = node_info['type']
                id = len(nodes)
                confidence = node_info.get('confidence')
            else:
                raise TypeError("Invalid node format")

            if isinstance(spans, str) or (isinstance(spans, (list, tuple)) and len(spans) == 1 and isinstance(spans[0], str)):
                aux_node = GPGAuxNode(spans[0] if isinstance(spans, (list, tuple)) else spans)
                aux_node.ID = id
                node = graph.add_node(aux_node, reuse_id=True)
            else:
                node = GPGPhraseNode(spans)
                node.ID = id
                graph.add_node(node, reuse_id=True)
            
            node.pos = pos
            node.confidence = confidence
            nodes[id] = node

        graph.node_id_base = max([node.ID for node in graph.nodes()]) + 1

        for edge_info in json_obj["graph"]["edges"]:
            if isinstance(edge_info, list):
                n1_id, edge_label, n2_id = edge_info[:3]
                confidence = edge_info[3] if len(edge_info) > 3 else None
            elif isinstance(edge_info, dict):
                n1_id = edge_info['start']
                n2_id = edge_info['end']
                edge_label = edge_info['label']
                confidence = edge_info.get('confidence')
            else:
                raise TypeError("Invalid edge format")

            node1 = nodes[n1_id]
            node2 = nodes[n2_id]
            graph.add_relation(node1, node2, edge_label, confidence)

        return graph


    def data(self):
        """

        @return:
        @rtype:
        """

        graph = dict()

        graph["nodes"] = []
        node2idx = dict()
        for idx, node in enumerate(self.nodes()):
            node2idx[node.ID] = idx
            if isinstance(node, GPGPhraseNode):
                if node.confidence is None:
                    graph["nodes"].append({'spans': node.spans, 'type': node.pos})
                else:
                    graph["nodes"].append({'spans': node.spans, 'type': node.pos, 'confidence': node.confidence})
            else:
                if node.confidence is None:
                    graph["nodes"].append({'spans': node.label, 'type': node.pos})
                else:
                    graph["nodes"].append({'spans': node.label, 'type': node.pos, 'confidence': node.confidence})

        graph["edges"] = []
        for idx, (n1, edge, n2) in enumerate(self.relations()):
            n1_id = node2idx[n1.ID]
            n2_id = node2idx[n2.ID]
            if edge.confidence is None:
                graph["edges"].append({'start': n1_id, 'end': n2_id, 'label': edge.label})
            else:
                graph["edges"].append({'start': n1_id, 'end': n2_id, 'label': edge.label, 'confidence': edge.confidence})

        data = dict()
        data["meta"] = self.meta
        data['words'] = self.words
        data['graph'] = graph

        return data
    
    def save(self, output_file_path):
        """

        :param output_file_path:
        :return:
        """

        data = self.data()

        with open(output_file_path, "w", encoding="UTF8") as output_file:
            json.dump(data, output_file, cls=CompactJSONEncoder, ensure_ascii=False)



class GPGraphVisualizer(GraphVisualizer):
    """
    GPGraphVisualizer
    """

    def __init__(self, debug=False):

        self.debug = debug

    def escape(self, node_text):
        """

        @param node_text:
        @return:
        """
        special_tokens = '{}<>"'

        for token in special_tokens:
            node_text = node_text.replace(token, "\\" + token)

        return node_text

    def node_label(self, graph, node, no_text=False, *args, **kwargs):
        """

        :param node:
        :param dep_graph:
        :return:
        """
        components = []
        components.append(str(node.ID))
        if no_text:
            node_text = ""
        else:
            node_text = self.escape(graph.node_text(node))
        components.append(node_text)

        if isinstance(node, GPGPhraseNode):
            span_str = self.escape(str(tuple(node.readable_spans)))
            components.append(span_str)

        x = node.pos
        if x is None:
            x = 'None'
        components.append(x)

        if hasattr(node, 'concept'):
            components.append(node.concept)

        label = "{0}".format(" | ".join(components))

        if self.debug and node.contexts:
            label = "{{{0}}}".format(" | ".join([label, "\n".join(node.contexts)]))

        return label

    def node_style(self, graph, node, *args, **kwargs):
        """

        @param node:
        @return:

        """
        style = {}

        style['shape'] = "record"
        style['fillcolor'] = "grey"
        style['style'] = "filled"
        
        return style

    def edge_label(self, graph, edge, *args, **kwargs):
        """

        @param edge:
        @param debug:
        @return:
        """

        edge_label = edge.label

        if self.debug and edge.contexts:
            edge_label = "{{{0}}}".format("|".join([edge_label, "\n".join(edge.contexts)]))

        return edge_label

    def edge_style(self, graph, edge, *args, **kwargs):
        """

        @param node:
        @return:

        """
        style = {}

        return style
    


from typing import List
import string


def get_word_positions(sentence: str, words: List[str]) -> List[tuple]:
    """
    Get the starting and ending character positions for each word in the sentence.

    Args:
        sentence (str): The original sentence.
        words (List[str]): The list of words in the sentence.

    Returns:
        List[tuple]: A list of tuples where each tuple contains the starting and ending positions of a word.
    """
    positions = []
    current_pos = 0

    for word in words:
        start_pos = sentence.find(word, current_pos)
        if start_pos == -1:
            raise ValueError(f"Word '{word}' in [{words}]not found in the sentence [{sentence}] starting from position {current_pos}.")
        end_pos = start_pos + len(word)
        positions.append((start_pos, end_pos))
        current_pos = end_pos

    return positions


def get_node_text(node: GPGPhraseNode, sentence: str, word_positions: List[tuple]):

    word_indexes = list(node.words(with_aux=True))

    node_label = ""
    for w in word_indexes:
        if isinstance(w, int):
            word_pos = word_positions[w]
            word_text = sentence[word_pos[0]: word_pos[1]]
            if word_pos[0] > 0 and sentence[word_pos[0] - 1] in string.whitespace:
                word_text = " " + word_text
            node_label += word_text
        else:
            node_label += w # should add space ? += (" " + " ")?
    return node_label


class GraphValidator(object):

    @property
    def name(self):
        return None

    """
    check whether the graph is valid, and return the severity of the error, and details of the error
    The severity of the error:
    *  is an float between 0 and 1. The larger the value, the more severe the error.
    *  0 means the graph is perfect in the aspect of the check
    """
    def validate(self, graph: GPGraph):
        pass
