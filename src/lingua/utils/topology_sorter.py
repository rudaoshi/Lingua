from lingua.structure.gpgraph import GPGraph, GPGNode, GPGEdge, GPGPhraseNode, GPGAuxNode
from lingua.concept.standard import LinguaGraphEdges

from typing import List, Tuple
from collections import deque






class LinguaGraphTopologySorter:


    """
    Sorter a LinguaGraph by un-ambiguous topology order .
    """

    def __init__(self):
        """Define edge type priorities"""

        self.edge_priorities = {
            # Core semantic relations
            'pred.arg.1': 0, 'pred.arg.2': 1, 'pred.arg.3': 2, 'pred.arg.4': 3,
            'func.arg': 4, 'variable': 5, 'body': 6,
            
            # Predicate relations
            'copula': 10, 'appositive': 11, 'discourse': 12, 'ref': 13,
            'repeat': 14, 'vocative': 15, 'nonsense': 16, 'modification': 17,
            'parataxis': 18, 'index': 19, 'attribute': 20,
            
            # As-predicate relations
            'as:pred.arg': 30, 'as:pred.arg.1': 31, 'as:pred.arg.2': 32,
            'as:pred.arg.3': 33, 'as:pred.arg.4': 34, 'as:func.arg': 35,
            'as:copula': 40, 'as:appositive': 41, 'as:discourse': 42,
            'as:ref': 43, 'as:repeat': 44, 'as:vocative': 45,
            'as:nonsense': 46, 'as:parataxis': 47, 'as:index': 48,
            'as:attribute': 49, 'as:modification': 50, 'as:punctuation': 51, 'punctuation': 52,
            'as:body': 53, 'as:variable': 54,
            
            # Punctuation and structural
            'local': 60, 'tail': 61, 
        }

        missing_labels = set(LinguaGraphEdges) - set(self.edge_priorities.keys())
        if missing_labels:
            raise ValueError(f"Missing edge labels: {missing_labels}")

    def _get_edge_rank(self, edge_label: str) -> int:
        """
        Rank edge types by importance.
        """

        return self.edge_priorities[edge_label]

    def _get_node_rank_info(self, gpg_graph: GPGraph) -> int:
        """
        Get node rank info for a GPGraph.
        """
        noderank_info = dict()
        node_id_to_sorted_words = dict()

        for node in reversed(list(gpg_graph.topological_sort())):
            node_words = set()
            if isinstance(node, GPGPhraseNode):
                node_words.update(node.words(with_aux=False))
            else:
                for child, _ in gpg_graph.children(node):
                    if child.ID in node_id_to_sorted_words:
                         node_words.update(set(node_id_to_sorted_words[child.ID]))
            
            sorted_words = tuple(sorted(node_words))
            node_id_to_sorted_words[node.ID] = sorted_words

            if sorted_words:
                noderank_info[node.ID] = (sorted_words, "PHRASE")
            else:
                assert isinstance(node, GPGAuxNode), "node must be a aux node"
                noderank_info[node.ID] = ((-100,), node.label)
        return noderank_info


    def edges(self, lingua_graph: GPGraph) -> List[Tuple[GPGNode, GPGEdge, GPGNode]]:
        """
        Sort edges based on breadth-first search traversal with edge priorities.
        Ensures uniqueness by processing nodes level-by-level and sorting children
        by edge priority (with node ID as tie-breaker) before processing.
        """
        root = lingua_graph.root
        visited = set()
        queue = deque([root])
        visited.add(root.ID)

        noderank_info = self._get_node_rank_info(lingua_graph)
        
        while queue:
            node = queue.popleft()
            
            # Get all children and sort by edge priority, node rank, then node ID for uniqueness
            children = list(lingua_graph.children(node))
            children.sort(key=lambda child_edge: (
                self._get_edge_rank(child_edge[1].label),  # Primary: edge priority
                noderank_info[child_edge[0].ID],            # Secondary: node rank (position/label)
                child_edge[0].ID,                           # Tie-breaker: node ID
            ))
            
            # Yield edges and enqueue children
            for child, edge in children:
                yield (node, edge, child)
                
                # Add child to queue if not already visited
                if child.ID not in visited:
                    visited.add(child.ID)
                    queue.append(child)

    def nodes(self, lingua_graph: GPGraph) -> List[GPGNode]:
        """
        Sort nodes by topology, ensuring each node is yielded only after all its parents.
        Maintains the same ordering as edges() for nodes at the same level.
        """
        # 1. Calculate in-degrees
        in_degree = {node.ID: 0 for node in lingua_graph.nodes()}
        for node in lingua_graph.nodes():
            for child, _ in lingua_graph.children(node):
                in_degree[child.ID] += 1

        # 2. Get rank info
        noderank_info = self._get_node_rank_info(lingua_graph)

        # 3. BFS-based Kahn's Algorithm
        queue = deque([lingua_graph.root])
        visited = {lingua_graph.root.ID}
        
        while queue:
            node = queue.popleft()
            yield node
            
            # Get children and sort exactly like in edges()
            children = list(lingua_graph.children(node))
            children.sort(key=lambda child_edge: (
                self._get_edge_rank(child_edge[1].label),
                noderank_info[child_edge[0].ID],
                child_edge[0].ID,
            ))
            
            for child, edge in children:
                in_degree[child.ID] -= 1
                if in_degree[child.ID] == 0:
                    if child.ID not in visited:
                        visited.add(child.ID)
                        queue.append(child)