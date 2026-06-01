"""
Evaluation script for Word-Lingua Graph Parser

This script uses the WordLinguaInference class from the lingua to:
1. Load a trained model from a HuggingFace format directory
2. Run inference on samples with ground-truth graphs from JSONL
3. Compute evaluation metrics

Usage:
    python inference_eval.py --model_dir /path/to/model --data_path /path/to/sample.jsonl
"""

import os
import sys
import json
import copy
import logging
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import jsonlines
from tqdm import tqdm

from lingua.structure.gpgraph import GPGraph, GPGPhraseNode, GPGEdge, GPGraphVisualizer
from lingua.utils.metric import graph_match_metric
from lingua.learn.wordgraph.modeler.gp2word import lingua2wordlingua
from lingua.learn.wordgraph.modeler.word2gp import wordlingua2lingua
from lingua.learn.wordgraph.decoding.inference import WordLinguaInference, InferenceConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def edge_f1_from_counts(matched_edge_num: int, pred_edge_num: int, true_edge_num: int) -> float:
    if pred_edge_num + true_edge_num == 0:
        return 1.0
    return 2 * matched_edge_num / (pred_edge_num + true_edge_num)


def load_samples(data_path: str) -> List[Dict[str, Any]]:
    """
    Load samples from JSONL file.
    
    Each sample should have:
    - sentence: The input sentence string
    - lingua: The ground truth lingua graph data
    
    Returns:
        List of samples with parsed graphs
    """
    samples = []
    with jsonlines.open(data_path, "r") as reader:
        for idx, item in enumerate(reader):
            try:
                sentence = item["sentence"]
                
                # Parse ground truth lingua graph
                gold_lingua_graph = GPGraph.parse(item["lingua"])
                
                # Optionally convert to word-lingua for word-level comparison
                gold_word_lingua_graph = None
                try:
                    wl_graph, _, _ = lingua2wordlingua(copy.deepcopy(gold_lingua_graph))
                    gold_word_lingua_graph = wl_graph
                except Exception as e:
                    logger.debug(f"Failed to convert ground truth lingua to word-lingua {idx}: {e}")
                
                samples.append({
                    "idx": idx,
                    "sentence": sentence,
                    "gold_lingua_graph": gold_lingua_graph,
                    "gold_word_lingua_graph": gold_word_lingua_graph,  # May be None
                })
            except Exception as e:
                logger.warning(f"Error loading sample {idx}: {e}")
                continue
    
    logger.info(f"Loaded {len(samples)} samples from {data_path}")
    return samples


def visualize_graph_pair(
    gold_graph: GPGraph,
    pred_graph: GPGraph,
    sentence: str,
    output_dir: str,
    example_id: Any,
    format: str = "png",
) -> Tuple[str, str]:
    """Visualize both ground truth and predicted graphs."""
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = GPGraphVisualizer()
    
    # Format ID string
    id_str = f"{example_id:05d}" if isinstance(example_id, int) else str(example_id)
    
    # Visualize ground truth
    gold_path = os.path.join(output_dir, f"{id_str}.gold.{format}")
    try:
        visualizer.visualize(gold_graph, file_name=gold_path, format=format)
    except Exception as e:
        logger.warning(f"Failed to visualize gold graph for example {example_id}: {e}")
        gold_path = None
    
    # Visualize prediction
    pred_path = os.path.join(output_dir, f"{id_str}.pred.{format}")
    try:
        visualizer.visualize(pred_graph, file_name=pred_path, format=format)
    except Exception as e:
        logger.warning(f"Failed to visualize pred graph for example {example_id}: {e}")
        pred_path = None
    
    return gold_path, pred_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate Word-Lingua Parser using WordLinguaInference")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to test data file (JSONL with 'sentence' and 'lingua' fields)")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Path to HuggingFace model directory")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save evaluation metrics (JSON)")
    
    # Inference arguments
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run inference on (cuda, mps, cpu). Auto-detected if not provided.")
    parser.add_argument("--statistics", type=str, default=None,
                       help="Path to wordlingua_statistics.json (optional)")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length (tokens)")
    
    # Decoder arguments
    parser.add_argument("--arc_threshold", type=float, default=0.5,
                       help="Arc probability threshold for decoding")
    parser.add_argument("--lam_parent_prob", type=float, default=1.0,
                       help="Weight for parent probability constraint")
    parser.add_argument("--lam_sibling_prob", type=float, default=1.0,
                       help="Weight for sibling probability constraint")
    parser.add_argument("--lam_pos_prob", type=float, default=1.0,
                       help="Weight for POS probability constraint")
    parser.add_argument("--lam_parent_given_child_prob", type=float, default=1.0,
                       help="Weight for parent given child probability constraint")
    
    # Evaluation arguments
    parser.add_argument("--edge_match", type=str, default="exact",
                       choices=["exact", "loose"],
                       help="Edge matching mode for graph metric")
    parser.add_argument("--max_samples", type=int, default=0,
                       help="Maximum number of samples to evaluate (0 for all)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for inference")
    
    # Visualization arguments
    parser.add_argument("--visualize_dir", type=str, default=None,
                       help="Directory to save graph visualizations")
    parser.add_argument("--visualize_mode", type=str, default="mismatch",
                       choices=["all", "mismatch", "match", "worst"],
                       help="Which examples to visualize")
    parser.add_argument("--visualize_format", type=str, default="png",
                       choices=["png", "svg", "pdf"],
                       help="Output format for visualizations")
    parser.add_argument("--max_visualize", type=int, default=100,
                       help="Maximum number of examples to visualize (0 for unlimited)")
    parser.add_argument("--visualize_lingua", action="store_true",
                       help="Also visualize linguagraph")
    
    # Debug arguments
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--debug_dir", type=str, default=None,
                       help="Directory for debug outputs")
    
    args = parser.parse_args()
    
    # Set up device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Create inference configuration
    config = InferenceConfig(
        max_length=args.max_length,
        arc_threshold=args.arc_threshold,
        lam_parent_prob=args.lam_parent_prob,
        lam_sibling_prob=args.lam_sibling_prob,
        lam_pos_prob=args.lam_pos_prob,
        lam_parent_given_child_prob=args.lam_parent_given_child_prob,
        debug=args.debug,
        debug_dir=args.debug_dir,
    )
    
    # Load inference engine using the new WordLinguaInference class
    logger.info(f"Loading model from {args.model_dir}...")
    inference = WordLinguaInference.from_pretrained(
        model_dir=args.model_dir,
        device=device,
        statistics_path=args.statistics,
        config=config,
    )
    logger.info("Model loaded successfully!")
    
    # Load samples
    samples = load_samples(args.data_path)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        logger.info(f"Evaluating on {len(samples)} samples (limited by --max_samples)")
    
    # Setup visualization directories
    wordlingua_vis_dir = None
    lingua_vis_dir = None
    error_vis_dir = None
    if args.visualize_dir:
        import shutil
        if os.path.exists(args.visualize_dir):
            shutil.rmtree(args.visualize_dir, ignore_errors=True)
        
        os.makedirs(args.visualize_dir, exist_ok=True)
        wordlingua_vis_dir = os.path.join(args.visualize_dir, "wordlingua")
        lingua_vis_dir = os.path.join(args.visualize_dir, "linguagraph")
        error_vis_dir = os.path.join(args.visualize_dir, "error")
        os.makedirs(wordlingua_vis_dir, exist_ok=True)
        os.makedirs(error_vis_dir, exist_ok=True)
        if args.visualize_lingua:
            os.makedirs(lingua_vis_dir, exist_ok=True)
    
    # Initialize metrics
    total_count = 0
    exact_match_count = 0
    
    # Word-lingua graph metrics
    total_pred_nodes = 0
    total_true_nodes = 0
    total_matched_nodes = 0
    total_pred_edges = 0
    total_true_edges = 0
    total_matched_edges = 0
    
    # Lingua graph metrics
    lingua_total_pred_nodes = 0
    lingua_total_true_nodes = 0
    lingua_total_matched_nodes = 0
    lingua_total_pred_edges = 0
    lingua_total_true_edges = 0
    lingua_total_matched_edges = 0
    lingua_exact_match_count = 0
    lingua_conversion_errors = 0
    
    # Visualization tracking
    visualize_count = 0
    lingua_visualize_count = 0
    error_visualize_count = 0
    validation_error_count = 0
    examples_to_visualize = []
    
    # Run evaluation with batch processing
    logger.info(f"Running inference and evaluation (batch_size={args.batch_size})...")
    
    # Process samples in batches
    for batch_start in tqdm(range(0, len(samples), args.batch_size), desc="Evaluating"):
        batch_samples = samples[batch_start:batch_start + args.batch_size]
        batch_sentences = [s["sentence"] for s in batch_samples]
        
        # Run batch inference
        batch_results = inference.parse_batch(batch_sentences, debug=args.debug)
        
        # Process each result in the batch
        for sample, result in zip(batch_samples, batch_results):
            example_idx = sample["idx"]
            sentence = sample["sentence"]
            gold_word_lingua_graph = sample["gold_word_lingua_graph"]
            gold_lingua_graph = sample["gold_lingua_graph"]
            
            pred_word_lingua_graph = result.word_lingua_graph
            pred_lingua_graph = result.lingua_graph
            
            if pred_word_lingua_graph is None:
                logger.warning(f"No prediction for sample {example_idx}: {result.lingua_conversion_error}")
                continue
            
            # Store evaluation results for later visualization
            candidate_item = {
                "idx": example_idx,
                "sentence": sentence,
                "gold_lingua_graph": gold_lingua_graph,
                "pred_lingua_graph": pred_lingua_graph,
                "gold_word_lingua_graph": gold_word_lingua_graph,
                "pred_word_lingua_graph": pred_word_lingua_graph,
                "lingua_metric": None,
                "word_lingua_metric": None,
                "validation_errors": result.decoding_info.get("validation_errors") if result.decoding_info else None
            }
            
            # Primary evaluation: Lingua graph comparison
            try:
                if pred_lingua_graph is not None:
                    lingua_metric = graph_match_metric(pred_lingua_graph, gold_lingua_graph, edge_match=args.edge_match)
                    
                    lingua_total_pred_nodes += lingua_metric["pred_node_num"]
                    lingua_total_true_nodes += lingua_metric["true_node_num"]
                    lingua_total_matched_nodes += lingua_metric["matched_node_num"]
                    lingua_total_pred_edges += lingua_metric["pred_edge_num"]
                    lingua_total_true_edges += lingua_metric["true_edge_num"]
                    lingua_total_matched_edges += lingua_metric["matched_edge_num"]
                    
                    if lingua_metric["exact_same"]:
                        lingua_exact_match_count += 1
                    
                    # Store metric for sorting
                    # Calculate Edge F1 for sorting
                    edge_f1 = edge_f1_from_counts(
                        lingua_metric["matched_edge_num"],
                        lingua_metric["pred_edge_num"],
                        lingua_metric["true_edge_num"],
                    )
                    lingua_metric["edge_f1"] = edge_f1
                    candidate_item["lingua_metric"] = lingua_metric
                    
                else:
                    lingua_conversion_errors += 1
                    if result.lingua_conversion_error:
                        logger.debug(f"Lingua conversion error for sample {example_idx}: {result.lingua_conversion_error}")
                        
            except Exception as e:
                lingua_conversion_errors += 1
                logger.debug(f"Error computing lingua metric for sample {example_idx}: {e}")
            
            # Secondary evaluation: Word-lingua graph comparison (only if ground truth conversion succeeded)
            if gold_word_lingua_graph is not None:
                try:
                    metric = graph_match_metric(pred_word_lingua_graph, gold_word_lingua_graph, edge_match=args.edge_match)
                    
                    total_pred_nodes += metric["pred_node_num"]
                    total_true_nodes += metric["true_node_num"]
                    total_matched_nodes += metric["matched_node_num"]
                    total_pred_edges += metric["pred_edge_num"]
                    total_true_edges += metric["true_edge_num"]
                    total_matched_edges += metric["matched_edge_num"]
                    
                    if metric["exact_same"]:
                        exact_match_count += 1
                    
                    candidate_item["word_lingua_metric"] = metric
                            
                except Exception as e:
                    logger.debug(f"Error computing word-lingua metric for sample {example_idx}: {e}")
            
            # Store candidate
            examples_to_visualize.append(candidate_item)
            
            total_count += 1

    # =========================================================================
    # Visualization Phase (After all batches)
    # =========================================================================
    if args.visualize_dir:
        logger.info(f"Visualizing results to {args.visualize_dir}...")
        
        # Sort candidates
        # Default: Sort by Lingua Edge F1 (ascending = worst to best)
        # If lingua metric missing (error), treat as -1.0
        examples_to_visualize.sort(key=lambda x: (x["lingua_metric"]["edge_f1"] if x["lingua_metric"] else -1.0))
        
        visualized_count = 0
        for rank, item in enumerate(examples_to_visualize):
            if args.max_visualize > 0 and visualized_count >= args.max_visualize:
                break
                
            idx = item["idx"]
            lingua_metric = item["lingua_metric"]
            wl_metric = item["word_lingua_metric"]
            
            # Determine if we should visualize based on mode
            should_visualize = False
            if args.visualize_mode == "all" or args.visualize_mode == "worst": # 'worst' is naturally handled by sorting
                should_visualize = True
            elif args.visualize_mode == "mismatch":
                 # Visualize if EITHER lingua OR word-lingua is not exact match
                 if not (lingua_metric and lingua_metric["exact_same"]) or (wl_metric and not wl_metric["exact_same"]):
                     should_visualize = True
            elif args.visualize_mode == "match":
                 # Visualize if BOTH are exact match (strictest definition)
                 if (lingua_metric and lingua_metric["exact_same"]):
                     should_visualize = True

            if not should_visualize:
                continue

            # Construct filename prefix
            score = lingua_metric["edge_f1"] if lingua_metric else 0.0
            file_prefix = f"{rank:04d}_f1_{score:.2f}_id_{idx}"
            
            # Visualize Lingua Graph (Primary)
            if args.visualize_lingua and item["gold_lingua_graph"]:
                visualize_graph_pair(
                    gold_graph=item["gold_lingua_graph"],
                    pred_graph=item["pred_lingua_graph"],
                    sentence=item["sentence"],
                    output_dir=args.visualize_dir,
                    example_id=file_prefix + ".lingua", # Pass as string ID to helper
                    format=args.visualize_format,
                )
            
            # Visualize Word-Lingua Graph (Secondary)
            if item["gold_word_lingua_graph"] or item["pred_word_lingua_graph"]:
                 visualize_graph_pair(
                    gold_graph=item["gold_word_lingua_graph"],
                    pred_graph=item["pred_word_lingua_graph"],
                    sentence=item["sentence"],
                    output_dir=args.visualize_dir,
                    example_id=file_prefix + ".wordlingua",
                    format=args.visualize_format,
                )
            
            # Metric Info File
            if lingua_metric or wl_metric:
                metric_file = os.path.join(args.visualize_dir, f"{file_prefix}.metric.json")
                metric_data = {
                    "idx": idx,
                    "sentence": item["sentence"],
                    "lingua_metric": lingua_metric,
                    "word_lingua_metric": wl_metric,
                    "validation_errors": item["validation_errors"]
                }
                with open(metric_file, "w") as f:
                    json.dump(metric_data, f, indent=2)

            visualized_count += 1

    
    # Compute aggregate metrics
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    
    # Primary: Lingua graph metrics
    lingua_node_precision = lingua_total_matched_nodes / (lingua_total_pred_nodes + 1e-8)
    lingua_node_recall = lingua_total_matched_nodes / (lingua_total_true_nodes + 1e-8)
    lingua_node_f1 = 2 * lingua_node_precision * lingua_node_recall / (lingua_node_precision + lingua_node_recall + 1e-8)
    
    lingua_edge_precision = lingua_total_matched_edges / (lingua_total_pred_edges + 1e-8)
    lingua_edge_recall = lingua_total_matched_edges / (lingua_total_true_edges + 1e-8)
    lingua_edge_f1 = 2 * lingua_edge_precision * lingua_edge_recall / (lingua_edge_precision + lingua_edge_recall + 1e-8)
    
    valid_lingua_count = total_count - lingua_conversion_errors
    lingua_exact_match_rate = lingua_exact_match_count / (valid_lingua_count + 1e-8)
    
    logger.info(f"Total examples evaluated: {total_count}")
    logger.info(f"")
    logger.info(f"=== Lingua Graph Metrics (Primary) ===")
    logger.info(f"Conversion errors: {lingua_conversion_errors}")
    logger.info(f"Exact match: {lingua_exact_match_count} ({lingua_exact_match_rate:.2%})")
    logger.info(f"Node P/R/F1: {lingua_node_precision:.4f} / {lingua_node_recall:.4f} / {lingua_node_f1:.4f}")
    logger.info(f"Edge P/R/F1: {lingua_edge_precision:.4f} / {lingua_edge_recall:.4f} / {lingua_edge_f1:.4f}")
    
    # Secondary: Word-lingua graph metrics (only if any samples have word-lingua ground truth)
    node_precision = total_matched_nodes / (total_pred_nodes + 1e-8)
    node_recall = total_matched_nodes / (total_true_nodes + 1e-8)
    node_f1 = 2 * node_precision * node_recall / (node_precision + node_recall + 1e-8)
    
    edge_precision = total_matched_edges / (total_pred_edges + 1e-8)
    edge_recall = total_matched_edges / (total_true_edges + 1e-8)
    edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall + 1e-8)
    
    exact_match_rate = exact_match_count / (total_count + 1e-8)
    
    if total_pred_nodes > 0:  # Only show if word-lingua comparison was possible
        logger.info(f"")
        logger.info(f"=== Word-Lingua Graph Metrics (Secondary) ===")
        logger.info(f"Exact match: {exact_match_count} ({exact_match_rate:.2%})")
        logger.info(f"Node P/R/F1: {node_precision:.4f} / {node_recall:.4f} / {node_f1:.4f}")
        logger.info(f"Edge P/R/F1: {edge_precision:.4f} / {edge_recall:.4f} / {edge_f1:.4f}")
    
    # Save results
    results = {
        "total_examples": total_count,
        "word_lingua_metrics": {
            "exact_match_count": exact_match_count,
            "exact_match_rate": exact_match_rate,
            "node_precision": node_precision,
            "node_recall": node_recall,
            "node_f1": node_f1,
            "edge_precision": edge_precision,
            "edge_recall": edge_recall,
            "edge_f1": edge_f1,
        },
        "lingua_metrics": {
            "conversion_errors": lingua_conversion_errors,
            "exact_match_count": lingua_exact_match_count,
            "exact_match_rate": lingua_exact_match_rate,
            "node_precision": lingua_node_precision,
            "node_recall": lingua_node_recall,
            "node_f1": lingua_node_f1,
            "edge_precision": lingua_edge_precision,
            "edge_recall": lingua_edge_recall,
            "edge_f1": lingua_edge_f1,
        },
        "config": {
            "model_dir": args.model_dir,
            "data_path": args.data_path,
            "device": device,
            "max_length": args.max_length,
            "arc_threshold": args.arc_threshold,
            "edge_match": args.edge_match,
        },
    }
    
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {args.output_path}")
    
    return results


if __name__ == "__main__":
    main()
