"""
convert a token gpgraph to word gpgraph 
"""

import re
from lingua.structure.gpgraph import GPGraph, GPGAuxNode, GPGPhraseNode, GPGTextNode, TextGPGraph

from typing import List

def add_prefix_space(words: List[str], sentence: str) -> List[str]:
    """
    Add a space prefix to words if there is a space before them in the sentence.
    
    Args:
        words: List of words (without spacing information), which are tokenization of sentence
        sentence: The original sentence string
        
    Returns:
        List of words with space prefixes where appropriate
    """
    if not words or not sentence:
        return words
    
    result = []
    sent_idx = 0  # Current position in sentence
    
    for word in words:
        # Strip any existing spaces from the word for matching
        word_stripped = word.strip()
        
        if not word_stripped:
            # Empty word, keep as is
            result.append(word)
            continue

        word_start_pos = sentence.find(word_stripped, sent_idx)
        if word_start_pos != -1:
            if word_start_pos > 1 and sentence[word_start_pos - 1] == ' ':
                result.append(' ' + word_stripped)
            else:
                result.append(word_stripped)
            sent_idx = word_start_pos + len(word_stripped)
        else:
            raise ValueError(f"Word [{word_stripped}] not found in sentence [{sentence}] starting from position {sent_idx}.")
    return result



import string

import spacy
nlp = spacy.blank("en")          # 不加载模型，只创建英文 tokenizer
    

def split_mixed_word(word: str) -> List[str]:
    """
    Split a word if it contains a mix of digits, characters, and punctuations.
    
    Args:
        word: A word that may contain mixed content
        
    Returns:
        List of split parts if the word contains mixed content, otherwise [word]
    """
    if not word:
        return [word]
    
    # Check if word contains digits, letters, and punctuation
    has_digit = any(c.isdigit() for c in word)
    has_letter = any(c.isalpha() for c in word)
    has_punct = any(c in string.punctuation for c in word)
    # Check for other characters (e.g. en-dash, em-dash, other unicode symbols not in string.punctuation)
    has_other = any(not (c.isdigit() or c.isalpha() or c in string.punctuation) for c in word)
    
    # Count how many types are present
    type_count = sum([has_digit, has_letter, has_punct, has_other])
    
    # If word contains mixed types (2 or more), split it
    if type_count >= 2:
        # Split on boundaries between different character types
        # This regex splits on transitions between digits, letters, and punctuation
        parts = []
        current_part = []
        current_type = None
        
        for char in word:
            if char.isdigit():
                char_type = 'digit'
            elif char.isalpha():
                char_type = 'letter'
            elif char in string.punctuation:
                char_type = 'punct'
            else:
                char_type = 'other'
            
            if current_type is None:
                current_type = char_type
                current_part.append(char)
            elif char_type == current_type:
                current_part.append(char)
            else:
                # Type changed, save current part and start new one
                if current_part:
                    parts.append(''.join(current_part))
                current_part = [char]
                current_type = char_type
        
        # Add the last part
        if current_part:
            parts.append(''.join(current_part))
        
        return parts if len(parts) > 1 else [word]
    
    return [word]


def segment(sentence: str) -> List[str]:
    """
    Segment a sentence into words.
    
    Steps:
    1. Use NLTK tokenizer to tokenize the sentence to words
    2. Check whether each word contains a mix of digit, char and punctuations, if so split them
    3. Add prefix spaces of the word
    
    Args:
        sentence: The input sentence string
        
    Returns:
        List of words with prefix spaces where appropriate
    """
    # Step 1: Use Spacy tokenizer to tokenize the sentence

    doc = nlp.tokenizer(sentence)
    words = [t.text for t in doc]
    
    # Step 2: Split words that contain mixed content (digits, chars, punctuations)
    split_words = []
    for word in words:
        
        split_parts = split_mixed_word(word)
        split_words.extend(split_parts)
    
    # Step 3: Add prefix spaces
    result = add_prefix_space(split_words, sentence)
    
    return result