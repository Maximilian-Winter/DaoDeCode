"""
Complete training and testing script for MechanismPointsLLM.

This script demonstrates the full pipeline:
1. Preparing training data
2. Training a tokenizer and model
3. Testing text generation with mechanism points analysis
4. Visualizing transformation patterns
"""

import os
import argparse
import logging
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import from the mechanism_llm module
# Assuming the previous code is saved as mechanism_llm.py
from mechanism_llm import (
    MechanismTokenizer,
    MechanismPointsLLM,
    TextDataset,
    Trainer,
    analyze_mechanism_points,
    generate_text,
    interactive_generation,
    visualize_transformation_patterns,
    demonstrate_position_timing,
    batch_analyze_texts
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")]
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def load_training_data(data_path=None):
    """Load training data from file or use sample data."""
    if data_path and os.path.exists(data_path):
        # Load data from file
        logger.info(f"Loading training data from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            if data_path.endswith('.json'):
                texts = json.load(f)
            else:
                texts = f.read().splitlines()
        return texts
    else:
        # Use sample data
        logger.info("Using sample training data")
        return [
            "The mechanism points concept focuses on identifying critical leverage points in complex systems where minimal input creates maximum output.",
            "According to the Five Elements theory, wood represents expansion, fire represents transformation, earth represents stabilization, metal represents refinement, and water represents adaptation.",
            "In Conway's Game of Life, small changes to initial conditions can create dramatically different outcomes, demonstrating sensitivity to mechanism points.",
            "Position-timing unity suggests that 'where' and 'when' to intervene in a system are not separate considerations but aspects of a unified manifold.",
            "The I Ching describes 64 hexagrams representing different transformation patterns between states.",
            "Transformation patterns can be characterized by their dominant element: wood for expansionary patterns, fire for accelerating patterns, earth for stabilizing patterns, metal for refining patterns, and water for adaptive patterns.",
            "Quantum mechanics has parallels with the position-timing unity concept, as particles exist in superpositions across spacetime until measured.",
            "The concept of wu-wei (non-action) in Daoism relates to finding natural mechanism points where minimal effort creates maximal effect.",
            "Complex systems like financial markets, climate systems, and social networks exhibit mechanism points where small interventions can create large effects.",
            "The Five Elements system includes both generating cycles (where each element enhances another) and constraining cycles (where each element controls another).",
            "In strategic games like chess, the position-timing unity concept helps players identify optimal moments for specific moves.",
            "The Gosper glider gun in Conway's Game of Life demonstrates how a simple repeating pattern can generate complex, persistent structures."
        ]

def train_model(args):
    """Train a model with the specified parameters."""
    # Load training data
    texts = load_training_data(args.data_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train tokenizer
    logger.info("Training tokenizer...")
    tokenizer = MechanismTokenizer(vocab_size=args.vocab_size)
    tokenizer_path = tokenizer.train_tokenizer(
        texts, 
        output_path=os.path.join(args.output_dir, "tokenizer")
    )
    
    # Split data for training/evaluation
    import random
    random.shuffle(texts)
    split_idx = int(len(texts) * 0.9)  # 90% for training, 10% for evaluation
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = TextDataset(train_texts, tokenizer, max_length=args.max_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_length=args.max_length) if eval_texts else None
    
    # Initialize model
    logger.info("Initializing model...")
    model = MechanismPointsLLM(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        output_dir=os.path.join(args.output_dir, "checkpoints"),
    )
    
    training_logs = trainer.train()
    
    # Plot training metrics
    if training_logs:
        plt.figure(figsize=(12, 6))
        
        # Extract metrics
        steps = [log["step"] for log in training_logs]
        train_losses = [log["train_loss"] for log in training_logs]
        eval_losses = [log["eval_loss"] for log in training_logs]
        
        # Plot training and evaluation loss
        plt.plot(steps, train_losses, label="Training Loss")
        plt.plot(steps, eval_losses, label="Evaluation Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(args.output_dir, "training_progress.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Training progress plot saved to {plot_path}")
    
    return model, tokenizer

def test_generation(model, tokenizer, args):
    """Test text generation with the trained model."""
    # Test prompts
    test_prompts = [
        "The key insight about mechanism points is",
        "The Five Elements in transformation patterns are",
        "Position-timing unity means that",
    ]
    
    results = []
    
    logger.info("Testing text generation...")
    for prompt in test_prompts:
        logger.info(f"Generating text for prompt: {prompt}")
        
        # Generate text
        result = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_length=100,
            temperature=0.7,
            track_mechanism_points=True
        )
        
        # Log results
        logger.info(f"Generated text: {result['text']}")
        if "top_mechanism_points" in result:
            logger.info("Top mechanism points:")
            for token, strength, idx in result["top_mechanism_points"]:
                logger.info(f"  {token} (position {idx}): {strength:.4f}")
        
        if "dominant_element" in result and result["dominant_element"]:
            logger.info(f"Dominant element: {result['dominant_element']}")
        
        results.append(result)
    
    # Visualize transformation patterns
    if args.visualize:
        logger.info("Visualizing transformation patterns...")
        all_texts = [r["text"] for r in results]
        visualization_results = visualize_transformation_patterns(model, all_texts, tokenizer)
        
        # Demonstrate position-timing unity
        logger.info("Demonstrating position-timing unity...")
        position_timing_results = demonstrate_position_timing(
            model, 
            tokenizer,
            template="The {position} cat sat on the mat.",
            positions=["happy", "angry", "sleepy"]
        )
        
        # Print position-timing results
        logger.info("Position-timing results:")
        for result in position_timing_results:
            logger.info(f"Input: {result['input']}")
            logger.info(f"Generated: {result['generated']}")
            logger.info(f"Dominant element: {result['dominant_element']}")
            logger.info("Top mechanism points:")
            for token, strength, idx in result["top_mechanism_points"]:
                logger.info(f"  {token} (position {idx}): {strength:.4f}")
            logger.info("---")
    
    return results

def interactive_mode(model, tokenizer):
    """Run interactive text generation with mechanism points analysis."""
    logger.info("Starting interactive mode...")
    interactive_generation(model, tokenizer)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train and test MechanismPointsLLM")
    
    # Training parameters
    parser.add_argument("--data_path", type=str, default=None, help="Path to training data file")
    parser.add_argument("--output_dir", type=str, default="model", help="Output directory for model and results")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size for tokenizer")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=1024, help="Intermediate size in feed-forward layers")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_position", type=int, default=1024, help="Maximum position embeddings")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    
    # Mode and options
    parser.add_argument("--mode", type=str, choices=["train", "test", "interactive", "all"], default="all",
                       help="Mode of operation: train, test, interactive, or all")
    parser.add_argument("--model_path", type=str, default=None, 
                       help="Path to pretrained model (for test and interactive modes)")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations of transformation patterns")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # Set device again if needed
    if args.no_cuda:
        global device
        device = torch.device("cpu")
        logger.info("Forcing CPU usage")
    
    # Load or train model based on mode
    model = None
    tokenizer = None
    
    if args.mode in ["train", "all"]:
        # Train model
        model, tokenizer = train_model(args)
    
    if args.mode in ["test", "interactive"] or (args.mode == "all" and model is None):
        # Load model if not already trained
        if model is None:
            model_path = args.model_path or os.path.join(args.output_dir, "checkpoints", "best_model")
            if not os.path.exists(model_path):
                model_path = os.path.join(args.output_dir, "checkpoints", "final_model")
            
            if not os.path.exists(model_path):
                logger.error(f"Model not found at {model_path}. Please train a model first.")
                return
            
            logger.info(f"Loading model from {model_path}")
            model = MechanismPointsLLM.from_pretrained(model_path)
            model.to(device)
            
            # Load tokenizer
            tokenizer_path = os.path.join(args.output_dir, "tokenizer", "mechanism_tokenizer.model")
            if not os.path.exists(tokenizer_path):
                logger.error(f"Tokenizer not found at {tokenizer_path}")
                return
            
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = MechanismTokenizer(model_path=tokenizer_path)
    
    if args.mode in ["test", "all"]:
        # Test generation
        test_generation(model, tokenizer, args)
    
    if args.mode in ["interactive", "all"]:
        # Interactive mode
        interactive_mode(model, tokenizer)
    
    logger.info("Script execution completed")

if __name__ == "__main__":
    main()
