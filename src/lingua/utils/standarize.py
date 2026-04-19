"""
convert a token gpgraph to word gpgraph 
"""

import re
from lingua.structure.gpgraph import GPGraph, GPGAuxNode, GPGPhraseNode, GPGTextNode, TextGPGraph

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

def standardize_gpgraph_words(gpgraph: GPGraph, sentence: str):
    """
    Standardize the words in the gpgraph and update the node spans accordingly.
    """
    assert isinstance(gpgraph, GPGraph)

    source_words = gpgraph.words
    target_words = segment(sentence)
    
    # 1. Compute char spans
    source_spans = get_word_char_spans(source_words, sentence)
    target_spans = get_word_char_spans(target_words, sentence)
    
    # 2. Align source words to target words
    # Map source_index -> list of target_indices
    source2target_map = {}
    
    for s_idx, s_span in enumerate(source_spans):
        if s_span is None:
            continue
            
        s_start, s_end = s_span
        matched_t_indices = []
        
        for t_idx, t_span in enumerate(target_spans):
            if t_span is None:
                continue
                
            t_start, t_end = t_span
            
            # Check for overlap
            # Overlap if max(start) < min(end)
            if max(s_start, t_start) < min(s_end, t_end):
                # We have an overlap. Check boundary conditions.
                # Valid if: Source contains Target OR Target contains Source
                
                s_contains_t = (s_start <= t_start) and (s_end >= t_end)
                t_contains_s = (t_start <= s_start) and (t_end >= s_end)
                
                if not (s_contains_t or t_contains_s):
                    raise Exception(f"Word alignment cross boundary: Source '{source_words[s_idx]}' ({s_span}) vs Target '{target_words[t_idx]}' ({t_span})")
                
                matched_t_indices.append(t_idx)
        
        if matched_t_indices:
            # If multiple targets, they should be continuous intervals?
            # It's possible to skip if there are gaps in sentence not covered by targets?
            # But just store the list.
            source2target_map[s_idx] = matched_t_indices

    # 3. Conflict Resolution: Check for shared target indices across DIFFERENT nodes
    # If multiple nodes map to the same target index, we must split that target word.
    
    # Build reverse map: target_idx -> list of source_indices
    target2source_map = {}
    for s_idx, t_indices in source2target_map.items():
        for t_idx in t_indices:
            if t_idx not in target2source_map:
                target2source_map[t_idx] = []
            target2source_map[t_idx].append(s_idx)

    # Determine which target words need splitting
    # A target word needs splitting if it is mapped to by source words that belong to DIFFERENT nodes.
    # We need to look up which node owns which source word.
    
    source_idx2node = {}
    for node in gpgraph.nodes():
        if isinstance(node, GPGPhraseNode):
            for span in node.spans:
                if isinstance(span, int):
                    source_idx2node[span] = node
                elif isinstance(span, (tuple, list)):
                    for i in range(span[0], span[1] + 1):
                        source_idx2node[i] = node
    
    new_target_words = []
    # Map old_target_idx -> [new_target_indices]
    target_split_map = {} 
    
    current_new_idx = 0
    
    for t_idx, word in enumerate(target_words):
        if t_idx not in target2source_map:
            # Not used by any source (or at least no alignment found), keep as is
            new_target_words.append(word)
            target_split_map[t_idx] = [current_new_idx]
            current_new_idx += 1
            continue
            
        associated_source_indices = sorted(target2source_map[t_idx])
        
        # Check if these source indices belong to different nodes
        associated_nodes = set()
        for s_idx in associated_source_indices:
            if s_idx in source_idx2node:
                associated_nodes.add(source_idx2node[s_idx])
        
        if len(associated_nodes) > 1:
            # Conflict! Multiple nodes share this target word.
            # We must split this target word back into pieces corresponding to the source words.
            # Since we know the char spans of source words, we can extract them from the target word?
            # Or from the original sentence? 
            # Actually, simplify: Just take the corresponding source words as the new words for this segment?
            # But we want to respect the 'segment' normalization (e.g. spacing).
            # The issue is 'hello' (target) coming from 'hel', 'lo'.
            # If we split, we probably want to revert to 'hel', 'lo' OR split 'hello' at the correct boundary.
            # Since 'segment' did the merge, maybe we should just use the source words for this segment?
            # BUT source words might be unnormalized.
            # Ideally, we split 'hello' based on the character lengths of the source words.
            
            # Let's try to split 'word' based on the character lengths of aligned source words.
            # 1. Get char spans of source words
            s_spans = [source_spans[i] for i in associated_source_indices]
            
            # Validate coverage: do source spans cover the target span?
            # Target span
            t_span = target_spans[t_idx]
            
            # Reconstruct parts
            parts = []
            
            # We assume order is preserved in associated_source_indices
            
            # There might be gaps or unmapped chars?
            # Strategy: Use source words' text as the split parts?
            # If we trust source words to be the "truth" for splitting.
            # But we should normalize them if possible (add spaces).
            
            # Actually, simpler: If conflict, just use the source words corresponding to each conflicting node group.
            # Correct. Use source words for these indices.
            
            split_indices = []
            for s_idx in associated_source_indices:
                # Add source word to new list
                # Use source text? Or try to extract from sentence?
                # Extracting from sentence preserves original formatting better?
                # s_span = source_spans[s_idx]
                # part = sentence[s_span[0]:s_span[1]]
                # Actually, simple: use source_words[s_idx] but maybe add space if needed?
                # The 'segment' function manages spacing.
                # If we revert to source words, we lose 'segment' benefits for this word.
                # But it's better than invalid graph.
                
                # Let's use source_words[s_idx] and try running add_prefix_space on the chunk later?
                # Or just append.
                
                part = source_words[s_idx]
                
                # Heuristic: if original source word didn't have space, but target did...
                # Current simple fix: use source word.
                new_target_words.append(part)
                split_indices.append(current_new_idx)
                current_new_idx += 1
                
            target_split_map[t_idx] = split_indices
            
        else:
            # No conflict (only 1 node or no nodes)
            new_target_words.append(word)
            target_split_map[t_idx] = [current_new_idx]
            current_new_idx += 1
            
    # Update target_words to the split version
    target_words = new_target_words
    
    # Update gpgraph.words to the new target words
    gpgraph.words = target_words
    
    # Re-map source2target_map using target_split_map
    # old source2target_map: source_idx -> [old_target_indices]
    # new source2target_map: source_idx -> [new_target_indices]
    
    new_source2target_map = {}
    for s_idx, t_indices in source2target_map.items():
        new_t_indices = []
        
        # If this source idx was part of a conflict, it should map to ONLY its specific split part.
        # How do we know which split part?
        # In the loop above, we iterated associated_source_indices in order.
        # So split_indices[k] corresponds to associated_source_indices[k].
        
        # We need check if t_idx was split.
        for t_idx in t_indices:
            if t_idx in target_split_map:
                mapped_news = target_split_map[t_idx]
                
                if len(mapped_news) > 1:
                    # It was split! Find which one corresponds to s_idx
                    # Re-derive associated_source_indices for this t_idx
                    associated = sorted(target2source_map[t_idx])
                    try:
                        pos = associated.index(s_idx)
                        new_t_indices.append(mapped_news[pos])
                    except ValueError:
                        # s_idx not in associated? Should not happen if logic is consistent
                        pass
                else:
                    new_t_indices.extend(mapped_news)
        
        if new_t_indices:
            new_source2target_map[s_idx] = sorted(list(set(new_t_indices)))

    source2target_map = new_source2target_map

    for node in gpgraph.nodes():
        if isinstance(node, GPGPhraseNode):
            new_spans = []
            for span in node.spans:
                if isinstance(span, str):
                    new_spans.append(span)
                elif isinstance(span, (tuple, list)) and len(span) == 2:
                    start, end = span
                    # Range inclusive [start, end]
                    # Collect all target indices mapped from source indices in this range
                    t_indices = []
                    for s_i in range(start, end + 1):
                        if s_i in source2target_map:
                            t_indices.extend(source2target_map[s_i])
                    
                    if t_indices:
                        t_indices = sorted(list(set(t_indices)))
                        # Convert to spans (start, end)
                        # We might have gaps if source words mapped to discontinuous targets? 
                        # Assuming continuous for now or let standardize_spans handle it.
                        # But modify_node_spans takes list of elements or (start, end).
                        # Let's group continuous indices.
                        
                        rng_start = t_indices[0]
                        rng_end = t_indices[0]
                        for idx in t_indices[1:]:
                            if idx == rng_end + 1:
                                rng_end = idx
                            else:
                                new_spans.append((rng_start, rng_end))
                                rng_start = idx
                                rng_end = idx
                        new_spans.append((rng_start, rng_end))
                        
                elif isinstance(span, int):
                    # Single source index
                    if span in source2target_map:
                        t_indices = source2target_map[span]
                        # Add as spans
                        # If single target: int ok? standardize_spans handles it.
                        # But better be explicit tuple if range.
                        if len(t_indices) == 1:
                            new_spans.append(t_indices[0])
                        else:
                             new_spans.append((min(t_indices), max(t_indices)))
            
            if new_spans:
                gpgraph.modify_node_spans(node, new_spans)
                
    return gpgraph