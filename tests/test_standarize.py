from lingua.structure.gpgraph import GPGraph, GPGPhraseNode
from lingua.utils.segement import segment
from lingua.utils.standarize import standardize_gpgraph_words


def lexical_coverage(graph):
    coverage = [0] * len(graph.words)
    for node in graph.nodes():
        if not isinstance(node, GPGPhraseNode):
            continue
        for token_index in node.indexes(with_aux=False):
            coverage[token_index] += 1
    return coverage


def test_standardize_gpgraph_words_keeps_segment_token_for_shared_old_nodes():
    sentence = "A 10(-8)"
    graph = GPGraph()
    graph.words = ["A", " 10", "(", "-", "8", ")"]

    a_node = graph.add_node(GPGPhraseNode(spans=[(0, 0)], pos="NominalConstant"))
    ten_node = graph.add_node(GPGPhraseNode(spans=[(1, 1)], pos="NominalConstant"))
    open_node = graph.add_node(
        GPGPhraseNode(spans=[(2, 2)], pos="PunctuationalConstant")
    )
    minus_node = graph.add_node(
        GPGPhraseNode(spans=[(3, 3)], pos="ExpressionFunctor")
    )
    eight_node = graph.add_node(GPGPhraseNode(spans=[(4, 4)], pos="NominalConstant"))
    close_node = graph.add_node(
        GPGPhraseNode(spans=[(5, 5)], pos="PunctuationalConstant")
    )

    graph.add_relation(ten_node, a_node, "as:modification")
    graph.add_relation(open_node, minus_node, "as:punctuation")
    graph.add_relation(close_node, minus_node, "as:punctuation")
    graph.add_relation(minus_node, eight_node, "func.arg")

    standardize_gpgraph_words(graph, sentence)

    assert graph.words == segment(sentence)
    assert "(-" in graph.words
    assert graph.has_node(minus_node)
    assert not graph.has_node(open_node)
    assert graph.node_text(minus_node) == "(-"
    assert graph.node_text(close_node) == ")"
    assert graph.get_edge(minus_node, eight_node).label == "func.arg"
    assert lexical_coverage(graph) == [1] * len(graph.words)
