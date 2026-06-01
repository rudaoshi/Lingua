"""
convert a token gpgraph to word gpgraph 
"""

import re
from lingua.structure.gpgraph import (
    GPGraph,
    GPGAuxNode,
    GPGPhraseNode,
    GPGTextNode,
    TextGPGraph,
    standardize_spans,
)

from typing import List

# def add_prefix_space(words: List[str], sentence: str) -> List[str]:
#     """
#     Add a space prefix to words if there is a space before them in the sentence.
    
#     Args:
#         words: List of words (without spacing information), which are tokenization of sentence
#         sentence: The original sentence string
        
#     Returns:
#         List of words with space prefixes where appropriate
#     """
#     if not words or not sentence:
#         return words
    
#     result = []
#     sent_idx = 0  # Current position in sentence
    
#     for word in words:
#         # Strip any existing spaces from the word for matching
#         word_stripped = word.strip()
        
#         if not word_stripped:
#             # Empty word, keep as is
#             result.append(word)
#             continue

#         word_start_pos = sentence.find(word_stripped, sent_idx)
#         if word_start_pos != -1:
#             if word_start_pos > 1 and sentence[word_start_pos - 1] == ' ':
#                 result.append(' ' + word_stripped)
#             else:
#                 result.append(word_stripped)
#             sent_idx = word_start_pos + len(word_stripped)
#         else:
#             raise ValueError(f"Word [{word_stripped}] not found in sentence [{sentence}] starting from position {sent_idx}.")
#     return result



# def merge_words(gpgraph: GPGraph, sentence: str):
#     """
#     convert a token gpgraph to word gpgraph 
#     """

#     assert isinstance(gpgraph, GPGraph)

#     # Collect all individual spans from all nodes and flatten them
#     # Separate string spans (like "{1}" argument placeholders) from tuple spans
#     all_spans = []
#     for node in gpgraph.nodes():
#         if isinstance(node, GPGPhraseNode):
#             # node.spans is a tuple of spans like ((0, 0),) or ((0, 1), (3, 4)) or with string placeholders
#             for span in node.spans:
#                 if not isinstance(span, str):
#                     all_spans.append(span)

#     # Remove duplicates and sort tuple spans by start position
#     all_spans = list(set(all_spans))
#     all_spans.sort(key=lambda x: x[0] if isinstance(x, (tuple, list)) else x)

#     words = []
#     tokenspan2wordspan_mapping = dict()

#     split_pattern =  re.compile(rf"([{string.punctuation} ])")
    
#     for span in all_spans:
#         if isinstance(span, (tuple, list)) and len(span) == 2:
#             start, end = span
#             # token_gpgraph.words contains token strings, concatenate them to get phrase
#             phrase = "".join(gpgraph.words[start: end+1])
#             key = (start, end)
#         elif isinstance(span, int):
#             # Single token span
#             phrase = gpgraph.words[span]
#             key = span 
#         else:
#             raise ValueError(f"Invalid span format: {span}")

#                 # Split phrase into words and filter empty ones
#         cur_words = list(filter(None, [x.strip() for x in re.split(split_pattern, phrase)]))
#         assert len(cur_words) > 0, f"Empty words are found in phrase: {phrase}"
#         if len(cur_words) == 1:
#             tokenspan2wordspan_mapping[key] = len(words)
#         else:
#             tokenspan2wordspan_mapping[key] = (len(words), len(words) + len(cur_words) - 1)
#         words.extend(cur_words)

#     gpgraph.words = add_prefix_space(words, sentence)
    
#     # Create new nodes with updated word spans
#     for node in gpgraph.nodes():
#         if isinstance(node, GPGPhraseNode):

#             assert node.spans in gpgraph.spans2node, f"Node spans {node.spans} not found in spans2node: {gpgraph.spans2node}"
#             # Convert each span in node.spans from token spans to word spans
#             new_spans = []
#             for span in node.spans:
#                 if isinstance(span, str):
#                     new_spans.append(span)
#                 else:
#                     new_spans.append(tokenspan2wordspan_mapping[span])
#             # Create new node with updated spans
#             gpgraph.modify_node_spans(node, new_spans)
        
#     return gpgraph

# import string
# from lingua.structure.utils import positions2spans

# def split_symbol_words(gpgraph: GPGraph, sentence: str):
#     """
#     Split words by non-alphanumeric characters in gpgraph and update node spans accordingly.
    
#     Args:
#         gpgraph: A GPGraph with words that may contain non-alphanumeric characters
        
#     Returns:
#         A new GPGraph with words split by non-alphanumeric characters and updated spans
#     """
#     assert isinstance(gpgraph, GPGraph)
    
#     new_gpgraph = GPGraph()
#     if hasattr(gpgraph, "sentence"):
#         new_gpgraph.sentence = gpgraph.sentence
    
#     # Pattern to split on any character except A-Za-z0-9 while keeping them
#     punct_pattern = re.compile(rf"([{string.punctuation+string.digits} ])")
    
#     # Create mapping from old word indices/spans to new word indices/spans
#     old2new_mapping = {}
    
#     # Process string spans (argument placeholders like "{1}")
#     string_spans = set()
#     for node in gpgraph.nodes():
#         if isinstance(node, GPGPhraseNode):
#             for span in node.spans:
#                 if isinstance(span, str):
#                     string_spans.add(span)
#                     old2new_mapping[span] = span
    
#     # Build new words list and create mapping
#     new_words = []
    
#     for i, word in enumerate(gpgraph.words):
#         # Split word by punctuation

#         word = word.strip()
        
#         parts = re.split(punct_pattern, word)
#         # Filter out empty strings
#         parts = list(filter(None, [p.strip() for p in parts]))
        
#         assert len(parts) > 0, f"Empty words are found in word: {word}"
        
#         if len(parts) == 1:
#             # No punctuation found, keep as is
#             old2new_mapping[i] = len(new_words)
#             new_words.append(word)
#         else:
#             # Multiple parts (word + punctuation), create span
#             start_idx = len(new_words)
#             new_words.extend(parts)
#             end_idx = len(new_words) - 1
#             old2new_mapping[i] = list(range(start_idx, end_idx + 1))
    
#     # Now handle tuple spans (start, end) from nodes
#     for node in gpgraph.nodes():
#         if isinstance(node, GPGPhraseNode):
#             node_positions = []
#             for old_word in node.words(with_aux=True):
#                 if isinstance(old_word, str):
#                     # Auxiliary node - keep as string
#                     node_positions.append(old_word)
#                 elif isinstance(old_word, int):
#                     # Map old word index to new word indices
#                     mapped = old2new_mapping.get(old_word, [])
#                     if isinstance(mapped, int):
#                         node_positions.append(mapped)
#                     elif isinstance(mapped, list):
#                         node_positions.extend(mapped)
#                 else:
#                     raise ValueError(f"Invalid word type: {type(old_word)}")
#             gpgraph.modify_node_spans(node, positions2spans(node_positions))
    
#     gpgraph.words = add_prefix_space(new_words, sentence)
    
#     return gpgraph

from .segement import segment

def get_word_char_spans(words: List[str], sentence: str):
    """
    Get character spans (start, end) for each word in the list relative to the sentence.
    Ignores whitespace during matching but assumes order.
    """
    spans = []
    cur = 0
    for word in words:
        w = word.strip()
        if not w:
            spans.append(None)
            continue
            
        start = sentence.find(w, cur)
        if start == -1:
            # Fallback: try searching from beginning if not found (though assume order)
            # Or strict order? strict order is safer for "a a" cases.
            raise ValueError(f"Word '{w}' not found in sentence starting from index {cur}")
            
        end = start + len(w)
        spans.append((start, end))
        cur = end
    return spans

def _indices_to_spans(indices):
    indices = sorted(set(indices))
    if not indices:
        return []

    spans = []
    start = end = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            spans.append((start, end))
            start = end = idx
    spans.append((start, end))
    return spans


def _choose_target_owner(candidates, source_indices, source_idx2node, node2source_indices):
    def score(node):
        node_source_indices = [
            idx for idx in source_indices if source_idx2node.get(idx) == node
        ]
        return (
            node.pos != "PunctuationalConstant",
            max(node_source_indices),
            len(node_source_indices),
            len(node2source_indices[node]),
            node.ID,
        )

    return max(candidates, key=score)


def _mapped_spans_for_source_span(span, source2target_map, target_owner, node):
    if isinstance(span, str):
        return [span]

    if isinstance(span, int):
        source_indices = [span]
    elif isinstance(span, (tuple, list)) and len(span) == 2:
        start, end = span
        source_indices = range(start, end + 1)
    else:
        raise ValueError(f"Invalid span: {span}")

    target_indices = []
    for source_idx in source_indices:
        for target_idx in source2target_map.get(source_idx, []):
            if target_owner.get(target_idx) == node:
                target_indices.append(target_idx)

    return _indices_to_spans(target_indices)


def standardize_gpgraph_words(gpgraph: GPGraph, sentence: str):
    """
    Standardize gpgraph words to exactly match segment(sentence).

    If several old phrase nodes map into one segment token, keep the segment
    token intact and assign it to a single owner node. Empty loser nodes are
    removed and their edges are redirected to the owner.
    """
    assert isinstance(gpgraph, GPGraph)

    source_words = gpgraph.words
    target_words = segment(sentence)

    source_spans = get_word_char_spans(source_words, sentence)
    target_spans = get_word_char_spans(target_words, sentence)

    source2target_map = {}
    for source_idx, source_span in enumerate(source_spans):
        if source_span is None:
            continue

        source_start, source_end = source_span
        target_indices = []
        for target_idx, target_span in enumerate(target_spans):
            if target_span is None:
                continue

            target_start, target_end = target_span
            if max(source_start, target_start) >= min(source_end, target_end):
                continue

            source_contains_target = (
                source_start <= target_start and source_end >= target_end
            )
            target_contains_source = (
                target_start <= source_start and target_end >= source_end
            )
            if not (source_contains_target or target_contains_source):
                raise ValueError(
                    "Word alignment cross boundary: "
                    f"Source '{source_words[source_idx]}' ({source_span}) vs "
                    f"Target '{target_words[target_idx]}' ({target_span})"
                )

            target_indices.append(target_idx)

        if target_indices:
            source2target_map[source_idx] = target_indices

    target2source_map = {}
    for source_idx, target_indices in source2target_map.items():
        for target_idx in target_indices:
            target2source_map.setdefault(target_idx, []).append(source_idx)

    phrase_nodes = [node for node in gpgraph.nodes() if isinstance(node, GPGPhraseNode)]
    node2source_indices = {
        node: list(node.indexes(with_aux=False)) for node in phrase_nodes
    }
    source_idx2node = {}
    for node, indices in node2source_indices.items():
        for source_idx in indices:
            source_idx2node[source_idx] = node

    target_owner = {}
    redirect_candidates = {node: [] for node in phrase_nodes}
    for target_idx in range(len(target_words)):
        source_indices = sorted(target2source_map.get(target_idx, []))
        candidates = {
            source_idx2node[source_idx]
            for source_idx in source_indices
            if source_idx in source_idx2node
        }
        if not candidates:
            raise ValueError(
                f"No phrase node can own target word {target_idx}: "
                f"{target_words[target_idx]!r}"
            )

        owner = _choose_target_owner(
            candidates, source_indices, source_idx2node, node2source_indices
        )
        target_owner[target_idx] = owner
        for node in candidates:
            if node != owner:
                redirect_candidates[node].append(owner)

    node2new_spans = {}
    removed_nodes = []
    for node in phrase_nodes:
        new_spans = []
        for span in node.spans:
            new_spans.extend(
                _mapped_spans_for_source_span(
                    span, source2target_map, target_owner, node
                )
            )

        if new_spans:
            node2new_spans[node] = standardize_spans(new_spans)
        else:
            removed_nodes.append(node)

    node_redirect = {}
    for node in removed_nodes:
        candidates = redirect_candidates[node]
        if not candidates:
            continue

        counts = {candidate: candidates.count(candidate) for candidate in candidates}
        node_redirect[node] = max(
            counts,
            key=lambda candidate: (
                counts[candidate],
                len(node2source_indices.get(candidate, [])),
                candidate.ID,
            ),
        )

    original_edges = list(gpgraph.relations())
    gpgraph.g.remove_edges_from(list(gpgraph.g.edges()))

    for node in removed_nodes:
        if gpgraph.has_node(node):
            gpgraph.remove_node(node)

    gpgraph.words = target_words
    gpgraph.spans2node = {}
    for node in phrase_nodes:
        if not gpgraph.has_node(node):
            continue

        node._spans = node2new_spans[node]
        if node.spans in gpgraph.spans2node:
            raise ValueError(f"Repeated node spans after standardization: {node.spans}")
        gpgraph.spans2node[node.spans] = node

    def resolve_node(node):
        seen = set()
        while node in node_redirect and node not in seen:
            seen.add(node)
            node = node_redirect[node]
        return node

    for start_node, edge, end_node in original_edges:
        start_node = resolve_node(start_node)
        end_node = resolve_node(end_node)

        if start_node == end_node:
            continue
        if not gpgraph.has_node(start_node) or not gpgraph.has_node(end_node):
            continue
        if gpgraph.get_edge(start_node, end_node) is not None:
            continue

        gpgraph.add_relation(start_node, end_node, edge.label, edge.confidence)

    gpgraph.root_ = None

    return gpgraph
