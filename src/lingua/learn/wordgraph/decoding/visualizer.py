from lingua.structure.gpgraph import GPGraphVisualizer


class ProbabilityAwareVisualizer(GPGraphVisualizer):
    """
    A visualizer that shows edge probabilities (arc_prob, rel_prob) on edges.
    """
    
    def edge_label(self, graph, edge, *args, **kwargs):
        """Override to include probability information."""
        edge_label = edge.label
        
        # Check if edge has probability info stored
        if hasattr(edge, 'arc_prob') and hasattr(edge, 'rel_prob'):
            arc_prob = edge.arc_prob
            rel_prob = edge.rel_prob
            
            # Show original score (arc_prob * rel_prob)
            original_score = arc_prob * rel_prob
            score_parts = [f"a={arc_prob:.2f}", f"r={rel_prob:.2f}", f"orig={original_score:.3f}"]
            
            # Show transformed score if available
            if hasattr(edge, 'transformed_score') and hasattr(edge, 'original_score'):
                transformed_score = edge.transformed_score
                struct_factor = getattr(edge, 'struct_factor', 1.0)
                score_parts.append(f"trans={transformed_score:.3f}")
                if struct_factor != 1.0:
                    score_parts.append(f"struct={struct_factor:.3f}")
            elif hasattr(edge, 'struct_factor') and edge.struct_factor != 1.0:
                # Compute transformed score if we have struct_factor but not transformed_score
                transformed_score = original_score * edge.struct_factor
                score_parts.append(f"trans={transformed_score:.3f}")
                score_parts.append(f"struct={edge.struct_factor:.3f}")
            
            edge_label = f"{edge_label}\\n" + ",".join(score_parts)
        elif hasattr(edge, 'confidence') and edge.confidence is not None:
            edge_label = f"{edge_label}\\nc={edge.confidence:.2f}"
        
        if self.debug and edge.contexts:
            edge_label = "{{{0}}}".format("|".join([edge_label, "\\n".join(edge.contexts)]))
        
        return edge_label