"""Gradio demo for local or Space-style Lingua inference."""

import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import gradio as gr
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import Lingua packages
from lingua.structure.gpgraph import GPGraph, GPGraphVisualizer
from lingua.learn.wordgraph.decoding.inference import WordLinguaInference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model ID for HuggingFace Hub
MODEL_ID = os.getenv("LINGUA_MODEL_ID", "rudaoshi/lingua")
HF_TOKEN = os.getenv("HF_TOKEN")

# ============================================================================
# Model Loading Singleton
# ============================================================================

_INFERENCE_ENGINE: Optional[WordLinguaInference] = None

def get_inference_engine(model_id: str = MODEL_ID) -> WordLinguaInference:
    """Get or load the inference engine singleton."""
    global _INFERENCE_ENGINE
    if _INFERENCE_ENGINE is not None:
        return _INFERENCE_ENGINE
    
    try:
        logger.info(f"Loading model from {model_id}...")
        
        # Determine if it's a local path or HF Hub ID
        if os.path.exists(model_id):
            model_dir = model_id
        else:
            logger.info("Downloading model from HuggingFace Hub...")
            model_dir = snapshot_download(repo_id=model_id, token=HF_TOKEN)
            
        logger.info(f"Model directory: {model_dir}")
        
        # Load inference engine
        _INFERENCE_ENGINE = WordLinguaInference.from_pretrained(
            model_dir=model_dir,
            device="cpu"  # Force CPU for Spaces usually, or check torch.cuda.is_available() inside from_pretrained defaults
        )
        
        logger.info("Model loaded successfully!")
        return _INFERENCE_ENGINE
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e

# ============================================================================
# Processing Logic
# ============================================================================

def visualize_graph(graph: GPGraph) -> Optional[str]:
    """Visualize graph and return path to temporary image file."""
    if graph is None:
        return None
    
    try:
        # Create temporary file
        temp_fd, temp_file = tempfile.mkstemp(suffix=".png")
        os.close(temp_fd)
        
        # Visualize
        visualizer = GPGraphVisualizer()
        visualizer.visualize(graph, file_name=temp_file, format="png")
        
        return temp_file
    except Exception as e:
        logger.error(f"Error visualizing graph: {e}")
        return None

def process_sentence(sentence: str) -> Tuple[Optional[str], str, Optional[Dict]]:
    """Process a sentence and return the visualization and graph data."""
    if not sentence.strip():
        return None, "Please enter a sentence.", None

    try:
        # Get inference engine
        inference = get_inference_engine()
        
        # Run inference
        result = inference.parse(sentence)
        
        if result.lingua_graph:
            graph = result.lingua_graph
            status_msg = f"Graph generated successfully using Lingua pipeline!"
        else:
            # Fallback to word-lingua graph if conversion failed
            graph = result.word_lingua_graph
            status_msg = "Warning: Failed to convert to full Lingua graph. Showing Word-Lingua graph."
        
        # Visualize
        img_path = visualize_graph(graph)
        
        # Get graph data for JSON output
        graph_data = None
        if graph:
            try:
                graph_data = graph.data()
            except Exception as e:
                logger.warning(f"Failed to serialize graph data: {e}")
        
        if img_path:
            return img_path, status_msg, graph_data
        else:
            return None, "Failed to generate visualization.", graph_data

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None, error_msg, None

def load_model_on_startup():
    """Load model when the Space starts up."""
    try:
        get_inference_engine()
        return "Model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {str(e)}"

# ============================================================================
# Gradio Interface
# ============================================================================

with gr.Blocks(title="Lingua Graph Parser") as demo:
    gr.Markdown("""
    # Lingua Graph Parser
    
    Parse English sentences into Lingua graphs using the released parser.
    
    Enter a sentence below to visualize its predicted structure.
    """)
    
    with gr.Column():
        with gr.Row():
            sentence_input = gr.Textbox(
                label="Input Sentence",
                placeholder="Enter a sentence here...",
                lines=3,
                info="Type any English sentence to parse",
                scale=4
            )
            parse_btn = gr.Button("Parse Sentence", variant="primary", size="lg", scale=1)
        
        output_text = gr.Textbox(
            label="Status", 
            lines=3,
            interactive=False
        )
        output_image = gr.Image(
            label="Graph Visualization", 
            type="filepath",
            height=600
        )
        output_json = gr.JSON(
            label="Graph Data (JSON)",
            visible=False
        )
    
    # Load model on startup
    demo.load(
        fn=load_model_on_startup,
        outputs=output_text
    )
    
    # Parse button click handler
    parse_btn.click(
        fn=process_sentence,
        inputs=[sentence_input],
        outputs=[output_image, output_text, output_json]
    )
    
    # Example sentences
    gr.Markdown("### Example Sentences")
    gr.Examples(
        examples=[
            "The cat sat on the mat .",
            "John loves Mary .",
            "I want to go to the store .",
            "The quick brown fox jumps over the lazy dog .",
            "She gave him a book yesterday .",
        ],
        inputs=sentence_input
    )
    
    # gr.Markdown("""
    # ### About
    
    # This parser uses a BERT-based model with biaffine attention to parse sentences into 
    # word-lingua graphs, which represent linguistic structures including:
    # - Predicate-argument relations
    # - Modification relations
    # - Discourse markers
    # - And more...
    
    # """)

if __name__ == "__main__":
    demo.launch()
