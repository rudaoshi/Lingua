"""
Graph data problem designed for graph2graph learning
"""
from typing import List

import networkx as nx

class Node:
    """
    Node
    """


    # def __init__(self, ID=None):
    #     """
    #
    #     :param id:
    #     :param object:
    #     :param rep:
    #     :param state:
    #     :param prob:
    #     """
    #
    #     self.ID = ID

    @property
    def ID(self):
        """

        :return:
        :rtype:
        """

        pass

    @ID.setter
    def ID(self, id):
        """

        :param id:
        :type id:
        :return:
        :rtype:
        """
        pass


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


class Edge(object):
    """
    Edge
    """

    def __init__(self, start=None, end=None):
        """

        :param object:
        :param rep:
        :param state:
        :param prob:
        """

        self.start = start
        self.end = end


class Graph(object):
    """
    Graph
    """

    def __init__(self, g=None):
        """
        init graph
        """
        super().__init__()
        if g is None:
            self.g = nx.Graph()
            self.node_id_base = 0
        else:
            import copy
            assert (isinstance(g, nx.Graph) or isinstance(g, Graph))
            if isinstance(g, Graph):
                self.g = copy.deepcopy(g.g)
                self.node_id_base = g.node_id_base
            else:
                self.g = copy.deepcopy(g)
                self.node_id_base = max(g.nodes, default=0) + 1

    def nodes(self):
        """

        :return:
        """

        for node in self.g.nodes:

            yield self.get_node(node)

    def edges(self):
        """

        :return:
        """

        for s, e in self.g.edges():
            edge = self.g[s][e]["Edge"]

            s_node = self.get_node(s)
            e_node = self.get_node(e)

            yield (s_node, edge, e_node)

    def has_node(self, node):
        """

        :param node:
        :return:
        """

        return node.ID in self.g

    def has_edge(self, node1, node2):
        """

        :param node1:
        :param node2:
        :return:
        """

        return node2.ID in self.g[node1.ID]

    def number_of_nodes(self):
        """

        :return:
        """

        return nx.number_of_nodes(self.g)

    def number_of_edges(self):
        """

        :return:
        """

        return nx.number_of_edges(self.g)


    def get_node(self, node_id):
        """

        :param node_id:
        :return:
        """

        return self.g.nodes[node_id]["Node"]

    def remove_node(self, node):
        """

        :param node:
        :return:
        """
        self.g.remove_node(node.ID)

    def remove_edge(self, edge):
        """

        :param node:
        :return:
        """
        self.g.remove_edge(edge.start, edge.end)

    def remove_edge_between(self, node1, node2):
        """

        :param node1:
        :type node1:
        :param node2:
        :type node2:
        :return:
        :rtype:
        """
        if self.g.has_edge(node1.ID, node2.ID):
            self.g.remove_edge(node1.ID, node2.ID)

    def get_edge(self, node1, node2):
        """

        :param node1_id:
        :param node2_id:
        :return:
        """

        if isinstance(node1, Node):
            node1 = node1.ID
        if isinstance(node2, Node):
            node2 = node2.ID
        """
        if type(node1) is not int:
            node1 = node1.ID
        if type(node2) is not int:
            node2 = node2.ID
        """

        try:
            edge = self.g[node1][node2]["Edge"]
        except KeyError as e:
            raise Exception("There is no edge between node {0} and {1}".format(node1, node2))

        return edge

    def add_node(self, n, reuse_id=False):
        """

        :param n:
        :param id:
        :return:
        """

        if reuse_id:
            node_id = n.ID
        else:
            node_id = self.node_id_base
            self.node_id_base += 1

        n.ID = node_id

        self.g.add_node(node_id, Node=n)

        return n

    def add_edge(self, ni, nj, e):
        """

        :param ni:
        :param eij:
        :param nj:
        :return:
        """

        if not isinstance(ni, Node):
            ni = self.get_node(ni)
        if not isinstance(nj, Node):
            nj = self.get_node(nj)

        e.start = ni.ID
        e.end = nj.ID

        self.g.add_edge(ni.ID, nj.ID, Edge=e)

    def neighbors(self, node):
        """

        :param ni:
        :return:
        """

        for nj in self.g[node.ID]:

            eij = self.g[node.ID][nj]["Edge"]

            yield eij, self.get_node(nj)


    def connected_components(self):
        """

        :return:
        """
        components = nx.algorithms.components.connected_components(self.g)

        for component in components:
            yield [self.get_node(x) for x in component]

    def breadth_first_dag(self, start_node):
        """

        :return:
        """

        dag = DirectedGraph()

        for node in self.nodes():

            dag.add_node(node.copy(), reuse_id=True)

        edges = nx.bfs_edges(self.g, start_node.ID)
        orderd_nodes = [start_node.ID] + [v for u, v in edges]
        for i, u in enumerate(orderd_nodes):
            for j, v in enumerate(orderd_nodes):

                if j <= i:
                    continue
                node_u = self.get_node(u)
                node_v = self.get_node(v)

                if self.has_edge(node_u, node_v):
                    edge = self.get_edge(node_u, node_v).copy()
                    dag.add_edge(node_u, node_v, edge)
        assert self.number_of_nodes() == dag.number_of_nodes()
        assert self.number_of_edges() == dag.number_of_edges()

        return dag

    def breadth_first_tree(self, start_node):
        """

        :return:
        """

        dag = DirectedGraph()

        dag.add_node(start_node.copy(), reuse_id=True)

        edges = nx.bfs_edges(self.g, start_node.ID)

        def __get_or_copy_node(u):

            try:
                node_u = dag.get_node(u)
            except:
                node_u = self.get_node(u).copy()
                dag.add_node(node_u, reuse_id=True)
            return node_u

        for u, v in edges:
            node_u = __get_or_copy_node(u)
            node_v = __get_or_copy_node(v)

            edge = self.get_edge(u, v)
            dag.add_edge(node_u, node_v, edge)

        return dag

    def __copy__(self):
        """

        :return:
        """
        copied = type(self)()
        copied.g = self.g.copy()
        copied.node_id_base = self.node_id_base
        return copied

    def __deepcopy__(self, memodict={}):
        """

        :param memodict:
        :type memodict:
        :return:
        :rtype:
        """

        from copy import deepcopy

#        copied_g = type(self.g)()
#
        copied = type(self)()

        memodict[id(self)] = copied

        for node in self.nodes():
            new_node = deepcopy(node)
            new_node = copied.add_node(new_node, reuse_id=True)
            assert new_node.ID == node.ID, "Node ID is not copied correctly {0} {1}".format(new_node.ID, node.ID)

        for (s_node, edge, e_node) in self.edges():
            copied.add_edge(s_node, e_node, deepcopy(edge))

        copied.node_id_base = self.node_id_base
        return copied

    def offsprings(self, node, filter=None):
        """

        :param node:
        :return:
        """

        for node_id in nx.dfs_postorder_nodes(self.g, node.ID):
            node = self.get_node(node_id)
            if not filter or filter(node):
                yield self.get_node(node_id)

    def subgraph(self, nodes):
        """

        :param nodes:
        :return:
        """
        node_ids = [n.ID if isinstance(n, Node) else n for n in nodes]

        subgraph = self.g.subgraph(node_ids).copy()

        result = self.__class__()
        result.g = subgraph

        return result

    def has_path(self, node1, node2):
        """

        :param node1:
        :param node2:
        :return:
        """
        return nx.algorithms.shortest_paths.has_path(self.g, node1.ID, node2.ID)


    def dual(self):
        """
        return the dual graph
        the dual graph is the graph with edges corresponding nodes and
        nodes corresponding edges
        """

        dual = Graph()

        edge_node_map = dict()

        for edge in self.edges():

            node = Node(value=edge.value)

            dual.add_node(node)

            edge_node_map[(edge.start, edge.end)] = node
            edge_node_map[(edge.end, edge.start)] = node

        for node in self.nodes():

            edges = list(self.g.edges(node.ID))

            # since the end node is added
            assert len(edges) >= 2, "Edge number should larger than 2 " \
                                    "since the end node is added"

            for idx1, (edge1_start, edge1_end) in enumerate(edges):

                for (edge2_start, edge2_end) in edges[idx1 + 1:]:

                    node1 = edge_node_map[(edge1_start, edge1_end)]
                    node2 = edge_node_map[(edge2_start, edge2_end)]

                    dual.add_edge(node1, node2, Edge(value=node.value))

        return dual

    #
    # def visualize(self, file_name=None):
    #     """
    #
    #     :return:
    #     """
    #
    #     visual_g = type(self.g)()
    #
    #     for node in self.nodes():
    #         visual_g.add_node(node.ID, label=self.node_label(node))
    #
    #     for node_s, edge, node_e in self.edges():
    #
    #         visual_g.add_edge(node_s.ID, node_e.ID, label=self.edge_label(edge))
    #
    #     from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
    #
    #     A = to_agraph(visual_g)
    #     if file_name:
    #         A.draw(file_name, prog="dot")
    #
    #     return A.to_string()



class DirectedGraph(Graph):
    """
    Directed Graph
    """

    def __init__(self, g=None):
        """

        :param edge_identifier:
        """

        if g is None:
            g = nx.DiGraph()

        super().__init__(g=g)

    def is_connected(self):
        """

        :return:
        """
        return nx.algorithms.components.is_weakly_connected(self.g)


    def connected_components(self):
        """

        :return:
        """
        components = nx.algorithms.components.weakly_connected_components(self.g)

        for component in components:
            yield [self.get_node(x) for x in component]

    def is_leaf(self, node):
        if len(list(self.children(node))) == 0:
            return True
        return False

    def children(self, node):
        """

        :param node:
        :return:
        """
        for child_id in self.g.successors(node.ID):
            child = self.get_node(child_id)
            rel = self.get_edge(node, child)

            yield child, rel

    def offsprings(self, node, filter=None):
        """

        :param node:
        :return:
        """
        yield node
        for node_id in nx.descendants(self.g, node.ID):
            node = self.get_node(node_id)
            if not filter or filter(node):
                yield self.get_node(node_id)

    def ancestors(self, node, filter=None):
        """

        :param node:
        :return:
        """
        yield node
        for node_id in nx.ancestors(self.g, node.ID):
            node = self.get_node(node_id)
            if not filter or filter(node):
                yield self.get_node(node_id)

    def parents(self, node):
        """

        :param node:
        :return:
        """
        for parent_id in self.g.predecessors(node.ID):
            parent = self.get_node(parent_id)
            rel = self.get_edge(parent, node)

            yield parent, rel

    def topological_sort(self):
        """

        :return:
        """

        for id in nx.topological_sort(self.g):
            yield self.get_node(id)


class LearnableGraph(object):
    """
    LearnableGraph
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def node_types(self, orders: List[Node], node_voc_functor):
        """

        :return:
        :rtype:
        """

        import numpy as np

        return np.narray([node_voc_functor(x) for x in orders])


    def adjmatrix(self, node_orders: List[Node], edge_voc_functor, empty_id=0):
        """

        :return:
        :rtype:
        """

        n_nodes = len(node_orders)
        node2index = dict((node, idx) for idx, node in enumerate(node_orders))
        import numpy as np

        in_a = np.ones([n_nodes, n_nodes], dtype=np.int32) * empty_id
        out_a = np.ones([n_nodes, n_nodes], dtype=np.int32) * empty_id 

        for u, edge, v in self.edges():

            u_idx = node2index[u]
            v_idx = node2index[v]

            e_idx = edge_voc_functor(edge)  # zero is empty type

            out_a[u_idx][v_idx] = e_idx
            in_a[v_idx][u_idx] = e_idx

            if not nx.is_directed(self.g):
                in_a[u_idx][v_idx] = e_idx
                out_a[v_idx][u_idx] = e_idx

        return (in_a, out_a)

    def to_tensor(self, node_orders: List[Node], node_voc, edge_voc, end_node=None):
        """

        :return:
        """
        node_types = self.node_types(node_orders, node_voc)
        a_in, a_out = self.adjmatrix(node_orders, edge_voc)
        if end_node:
            node_num = len(node_types)
            node_types.resize((node_num + 1,))
            node_types[-1] = end_node

            a_in.resize((node_num + 1, node_num + 1))
            a_out.resize((node_num + 1, node_num + 1))


        return node_types, a_in, a_out



def valid_alignment(choices):
    """

    :param choices:
    :return:
    """
    def _inner(i):
        if i == n:
            yield tuple(result)
            return
        for elt in sets[i] - seen:
            seen.add(elt)
            result[i] = elt
            for t in _inner(i + 1):
                yield t
            seen.remove(elt)

    sets = [set(seq) for seq in choices]
    n = len(sets)
    seen = set()
    result = [None] * n
    for t in _inner(0):
        yield t


def is_valid_topology_sort(dag, node_objs):
    """
    decide whether the order of nodes in dag2 is a valid topological sort order of dag1
    :param dag:
    :param pred_node_objs donot contain the start node:
    :return:
    """
    target_nodes = list(dag.nodes())
    target_node_objects = [n.object for n in target_nodes]

    choices = []
    for i, node_obj in enumerate(node_objs):
        cur_choice = []
        for j, target_node in enumerate(target_node_objects):
            if target_node == node_obj:
                cur_choice.append(j)

        if len(cur_choice) == 0:
            return False

        choices.append(cur_choice)

    for align in valid_alignment(choices):

        if len(set(align)) != len(align):
            continue

        node_ids = [target_nodes[i].ID for i in align]

        bad_align = False
        for id, node_id in enumerate(node_ids):

            if nx.descendants(dag.g, node_id).intersection(set(node_ids[:id])):
                bad_align = True
                break

            if nx.ancestors(dag.g, node_id).intersection(set(node_ids[id + 1:])):
                bad_align = True
                break

        if not bad_align:
            return True

    return False




def not_isomorphic(graph_a, graph_b):
    """

    :param graph_a:
    :param graph_b:
    :return:
    """

    return nx.faster_could_be_isomorphic(graph_a.g, graph_b.g)


def dot2image(dot_string, file_name=None, program="dot", format=None, return_img=False):
    """

    @param g:
    @param file_name:
    @return:
    """

    from PIL import Image
    import os 

    import tempfile
    dot_file = tempfile.NamedTemporaryFile(mode='w', suffix=".dot", delete=False)
    dot_file.write(dot_string)
    dot_file.close()

    if not format:
        format = "svg"

    if not file_name and return_img:
        import tempfile
        fout = tempfile.NamedTemporaryFile(suffix="." + format)
        file_name = fout.name

    return_val = os.system(f'{program} -T {format} "{dot_file.name}" -o "{file_name}"')
    assert return_val == 0

    if return_img:
        return Image.open(file_name)

class GraphVisualizer(object):
    """
    BasicGraphVisualizer
    """


    def node_label(self, graph, node, *args, **kwargs):
        """

        :param node:
        :type node:
        :return:
        :rtype:
        """

        return str(node)

    def node_style(self, graph, node, *args, **kwargs):
        """

        @param graph:
        @param node:
        @param args:
        @param kwargs:
        @return:
        """
        return {}

    def edge_label(self, graph, edge, *args, **kwargs):
        """

        :param edge:
        :type edge:
        :return:
        :rtype:
        """
        return str(edge)

    def edge_style(self, graph, edge, *args, **kwargs):
        """

        @param graph:
        @param edge:
        @param args:
        @param kwargs:
        @return:
        """

        return {}

    def visualize(self, graph, file_name=None, return_img=False, format="svg", no_text=False, *args, **kwargs):
        """

        @return:
        @rtype:
        """
        import io
        dot_string = io.StringIO()

        dot_string.write("strict digraph {\n")

        node2index = dict()
        for index, node_id in enumerate(graph.g.nodes()):
            node = graph.get_node(node_id)

            node_label = self.node_label(graph, node, no_text=no_text, *args, **kwargs)
            node_style = self.node_style(graph, node, *args, **kwargs)
            node2index[node.ID] = index

            node_attr = ['label="{0}"'.format(node_label)]
            for k, v in node_style.items():
                node_attr.append('{0}="{1}"'.format(k, v))

            vis_node_label = '{0}\t[{1}]; \n'.format(
                index, ", ".join(node_attr)
            )

            dot_string.write(vis_node_label)
            # if simple:
            #     g.add_node(id2index[node_id], label=node_text, shape=shape)
            # else:
            #     g.add_node(node_id, label=node_text, shape=shape)

        for s, e in graph.g.edges():
            edge = graph.g[s][e]["Edge"]

            edge_label = self.edge_label(graph, edge, *args, **kwargs)
            edge_style = self.edge_style(graph, edge, *args, **kwargs)

            edge_attr = ['label="{0}"'.format(edge_label)]
            for k, v in edge_style.items():
                edge_attr.append('{0}="{1}"'.format(k, v))

            s = node2index[s]
            e = node2index[e]

            dot_string.write('{0}\t->\t{1}\t[{2}];\n'.format(
                s, e, ", ".join(edge_attr)
            ))

        dot_string.write("}\n")

        dot_string = dot_string.getvalue()

        result = dot_string

        if file_name or return_img:
            image = dot2image(dot_string, file_name=file_name, return_img=return_img,
                              format=format)

        if return_img:
            result = image

        return result
