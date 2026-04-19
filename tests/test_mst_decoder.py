#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for MST decoder.
"""

import unittest
import numpy as np
from lingua.learn.wordgraph.decoding.mst_decoder import (
    MSTDecoder,
    decode_batch
)


class TestMSTDecoder(unittest.TestCase):
    """Test cases for MST decoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.id2label = {
            0: "nsubj",
            1: "det",
            2: "case",
            3: "obj",
            4: "root"
        }
        self.decoder = MSTDecoder(id2label=self.id2label)
    
    def test_simple_tree(self):
        """Test decoding a simple tree structure."""
        # Sentence: "The cat sat"
        # Expected: ROOT -> sat(2), sat -> cat(1), cat -> The(0)
        length = 3
        arc_scores = np.zeros((length, length))
        
        # Set up tree structure
        arc_scores[0, 2] = 10.0  # ROOT -> sat
        arc_scores[2, 1] = 8.0   # sat -> cat
        arc_scores[1, 0] = 6.0   # cat -> The
        
        heads, labels = self.decoder.decode(arc_scores, length=length)
        
        # Verify tree structure
        self.assertEqual(len(heads), length)
        self.assertEqual(heads[0], 1)   # The -> cat
        self.assertEqual(heads[1], 2)   # cat -> sat
        self.assertEqual(heads[2], -1)  # sat -> ROOT
    
    def test_with_labels(self):
        """Test decoding with edge labels."""
        length = 3
        arc_scores = np.zeros((length, length))
        arc_scores[0, 2] = 10.0
        arc_scores[2, 1] = 8.0
        arc_scores[1, 0] = 6.0
        
        # Label scores: [length, length, num_labels]
        num_labels = len(self.id2label)
        label_scores = np.zeros((length, length, num_labels))
        
        # Assign specific labels
        label_scores[2, 1, 0] = 10.0  # sat -> cat : nsubj
        label_scores[1, 0, 1] = 10.0  # cat -> The : det
        
        heads, labels = self.decoder.decode(arc_scores, label_scores, length=length)
        
        # Verify labels
        self.assertEqual(labels[0], 1)  # The: det
        self.assertEqual(labels[1], 0)  # cat: nsubj
    
    def test_with_root_scores(self):
        """Test decoding with explicit root scores."""
        length = 4
        arc_scores = np.random.randn(length, length) * 0.1
        
        # Make word 2 the clear root
        root_scores = np.array([-5.0, -5.0, 10.0, -5.0])
        
        # Set up some edges
        arc_scores[2, 0] = 5.0
        arc_scores[2, 1] = 5.0
        arc_scores[2, 3] = 5.0
        
        heads, labels = self.decoder.decode(arc_scores, root_scores=root_scores, length=length)
        
        # Verify that word 2 is the root
        self.assertEqual(heads[2], -1)
        # Other words should have word 2 as ancestor
    
    def test_cycle_detection(self):
        """Test that cycles are properly detected and resolved."""
        length = 3
        arc_scores = np.zeros((length, length))
        
        # Create a potential cycle: 0 -> 1, 1 -> 2, 2 -> 0
        arc_scores[0, 1] = 5.0
        arc_scores[1, 2] = 5.0
        arc_scores[2, 0] = 5.0
        
        # Add root edge
        arc_scores[0, 0] = 1.0  # Make 0 slightly prefer to be root
        
        heads, labels = self.decoder.decode(arc_scores, length=length)
        
        # Verify no cycles exist
        visited = set()
        for i in range(length):
            path = []
            current = i
            while current >= 0 and current not in visited:
                if current in path:
                    self.fail(f"Cycle detected: {path}")
                path.append(current)
                current = heads[current]
            visited.update(path)
    
    def test_single_root(self):
        """Test that exactly one word is assigned as root."""
        length = 5
        arc_scores = np.random.randn(length, length)
        
        heads, labels = self.decoder.decode(arc_scores, length=length)
        
        # Count roots (heads with -1)
        num_roots = sum(1 for h in heads if h == -1)
        self.assertEqual(num_roots, 1, f"Expected 1 root, got {num_roots}")
    
    def test_all_words_have_heads(self):
        """Test that all words (except root) have exactly one head."""
        length = 6
        arc_scores = np.random.randn(length, length)
        
        heads, labels = self.decoder.decode(arc_scores, length=length)
        
        # All words should have a head
        self.assertEqual(len(heads), length)
        
        # Each word should appear as head at most once or be root
        for i in range(length):
            if heads[i] >= 0:
                self.assertLess(heads[i], length)
                self.assertGreaterEqual(heads[i], 0)
    
    def test_decode_to_dict(self):
        """Test decode_to_dict output format."""
        length = 3
        words = ["The", "cat", "sat"]
        arc_scores = np.zeros((length, length))
        arc_scores[0, 2] = 10.0
        arc_scores[2, 1] = 8.0
        arc_scores[1, 0] = 6.0
        
        label_scores = np.zeros((length, length, len(self.id2label)))
        label_scores[2, 1, 0] = 10.0
        label_scores[1, 0, 1] = 10.0
        
        result = self.decoder.decode_to_dict(
            arc_scores=arc_scores,
            label_scores=label_scores,
            length=length,
            words=words
        )
        
        # Verify output format
        self.assertIn("heads", result)
        self.assertIn("labels", result)
        self.assertIn("label_ids", result)
        self.assertIn("words", result)
        self.assertIn("edges", result)
        
        # Verify content
        self.assertEqual(len(result["heads"]), length)
        self.assertEqual(len(result["labels"]), length)
        self.assertEqual(result["words"], words)
        self.assertIsInstance(result["edges"], list)
        
        # Each edge should be a tuple
        for edge in result["edges"]:
            self.assertEqual(len(edge), 3)  # (head, dependent, label)
    
    def test_batch_decoding(self):
        """Test batch decoding."""
        batch_size = 3
        seq_len = 4
        num_labels = len(self.id2label)
        
        arc_scores = np.random.randn(batch_size, seq_len, seq_len)
        label_scores = np.random.randn(batch_size, seq_len, seq_len, num_labels)
        root_scores = np.random.randn(batch_size, seq_len)
        lengths = [3, 4, 2]
        
        batch_heads, batch_labels = decode_batch(
            arc_scores=arc_scores,
            label_scores=label_scores,
            root_scores=root_scores,
            lengths=lengths,
            id2label=self.id2label
        )
        
        # Verify batch size
        self.assertEqual(len(batch_heads), batch_size)
        self.assertEqual(len(batch_labels), batch_size)
        
        # Verify each sentence
        for i, (heads, labels, length) in enumerate(zip(batch_heads, batch_labels, lengths)):
            self.assertEqual(len(heads), length, f"Sentence {i} heads length mismatch")
            self.assertEqual(len(labels), length, f"Sentence {i} labels length mismatch")
            
            # Each sentence should have exactly one root
            num_roots = sum(1 for h in heads if h == -1)
            self.assertEqual(num_roots, 1, f"Sentence {i} should have 1 root, got {num_roots}")
    
    def test_projective_tree(self):
        """Test decoding a projective tree."""
        # Projective tree: no crossing arcs
        length = 5
        arc_scores = np.zeros((length, length))
        
        # Create projective structure: ROOT -> 2, 2 -> 1, 2 -> 3, 1 -> 0, 3 -> 4
        arc_scores[0, 2] = 10.0
        arc_scores[2, 1] = 8.0
        arc_scores[2, 3] = 7.0
        arc_scores[1, 0] = 6.0
        arc_scores[3, 4] = 5.0
        
        heads, labels = self.decoder.decode(arc_scores, length=length)
        
        # Verify structure
        self.assertEqual(heads[0], 1)
        self.assertEqual(heads[1], 2)
        self.assertEqual(heads[2], -1)
        self.assertEqual(heads[3], 2)
        self.assertEqual(heads[4], 3)
    
    def test_non_projective_tree(self):
        """Test decoding a non-projective tree (with crossing arcs)."""
        # Non-projective trees are valid in dependency parsing
        length = 5
        arc_scores = np.zeros((length, length))
        
        # Create non-projective structure: ROOT -> 1, 1 -> 0, 1 -> 3, 3 -> 2, 3 -> 4
        # This creates a crossing between (1,3) and (3,2)
        arc_scores[0, 1] = 10.0
        arc_scores[1, 0] = 8.0
        arc_scores[1, 3] = 7.0
        arc_scores[3, 2] = 6.0
        arc_scores[3, 4] = 5.0
        
        heads, labels = self.decoder.decode(arc_scores, length=length)
        
        # Verify it's still a valid tree (no cycles, single root)
        num_roots = sum(1 for h in heads if h == -1)
        self.assertEqual(num_roots, 1)
        
        # Check connectivity
        for i in range(length):
            if heads[i] >= 0:
                self.assertGreaterEqual(heads[i], 0)
                self.assertLess(heads[i], length)
    
    def test_empty_sentence(self):
        """Test handling of empty/zero-length sentences."""
        arc_scores = np.array([[]])
        heads, labels = self.decoder.decode(arc_scores, length=0)
        self.assertEqual(len(heads), 0)
        self.assertEqual(len(labels), 0)
    
    def test_single_word(self):
        """Test single word sentence."""
        arc_scores = np.array([[0.0]])
        heads, labels = self.decoder.decode(arc_scores, length=1)
        self.assertEqual(len(heads), 1)
        self.assertEqual(heads[0], -1)  # Single word is root
    
    def test_uniform_scores(self):
        """Test with uniform scores (all edges equally likely)."""
        length = 4
        arc_scores = np.ones((length, length))
        
        heads, labels = self.decoder.decode(arc_scores, length=length)
        
        # Should still produce valid tree
        self.assertEqual(len(heads), length)
        num_roots = sum(1 for h in heads if h == -1)
        self.assertEqual(num_roots, 1)
    
    def test_negative_scores(self):
        """Test with negative scores."""
        length = 3
        arc_scores = -np.ones((length, length)) * 10
        
        # Make some edges less negative
        arc_scores[0, 1] = -1.0
        arc_scores[1, 2] = -1.0
        
        heads, labels = self.decoder.decode(arc_scores, length=length)
        
        # Should still find best tree even with negative scores
        self.assertEqual(len(heads), length)
        num_roots = sum(1 for h in heads if h == -1)
        self.assertEqual(num_roots, 1)


class TestMSTAlgorithm(unittest.TestCase):
    """Test the MST algorithm implementation details."""
    
    def test_greedy_best_heads(self):
        """Test greedy head selection."""
        decoder = MSTDecoder()
        scores = np.array([
            [0, 1, 2],
            [3, 0, 5],
            [6, 7, 0]
        ])
        
        heads = decoder._greedy_best_heads(scores)
        
        # Node 0 has no head (root)
        self.assertEqual(heads[0], -1)
        # Node 1's best incoming edge is from node 2 (score 7)
        self.assertEqual(heads[1], 2)
        # Node 2's best incoming edge is from node 1 (score 5)
        self.assertEqual(heads[2], 1)
    
    def test_cycle_finding(self):
        """Test cycle detection."""
        decoder = MSTDecoder()
        
        # No cycle
        heads = [-1, 0, 1, 2]
        cycle = decoder._find_cycle(heads)
        self.assertIsNone(cycle)
        
        # Simple cycle: 1 -> 2 -> 1
        heads = [-1, 2, 1]
        cycle = decoder._find_cycle(heads)
        self.assertIsNotNone(cycle)
        self.assertEqual(len(cycle), 2)
        self.assertIn(1, cycle)
        self.assertIn(2, cycle)
        
        # Cycle not involving root: 1 -> 2 -> 3 -> 1
        heads = [-1, 2, 3, 1]
        cycle = decoder._find_cycle(heads)
        self.assertIsNotNone(cycle)
        self.assertEqual(len(cycle), 3)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_large_sentence(self):
        """Test with a larger sentence."""
        length = 50
        decoder = MSTDecoder()
        
        arc_scores = np.random.randn(length, length)
        heads, labels = decoder.decode(arc_scores, length=length)
        
        self.assertEqual(len(heads), length)
        num_roots = sum(1 for h in heads if h == -1)
        self.assertEqual(num_roots, 1)
    
    def test_very_negative_scores(self):
        """Test with very negative scores."""
        length = 5
        decoder = MSTDecoder()
        
        arc_scores = np.full((length, length), -1e6)
        arc_scores[0, 1] = -100
        arc_scores[1, 2] = -100
        arc_scores[2, 3] = -100
        arc_scores[3, 4] = -100
        
        heads, labels = decoder.decode(arc_scores, length=length)
        
        self.assertEqual(len(heads), length)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        length = 4
        decoder = MSTDecoder()
        
        # Mix of very large and very small values
        arc_scores = np.array([
            [0, 1e10, 1e-10, 1e5],
            [1e-10, 0, 1e10, 1e-10],
            [1e5, 1e-10, 0, 1e10],
            [1e10, 1e5, 1e-10, 0]
        ])
        
        heads, labels = decoder.decode(arc_scores, length=length)
        
        # Should still produce valid tree
        num_roots = sum(1 for h in heads if h == -1)
        self.assertEqual(num_roots, 1)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)

