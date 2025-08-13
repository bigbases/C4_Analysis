#!/usr/bin/env python3
"""
Gemma-3-4B-IT model fine-tuning script
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.common import ModelConfig, SystemMonitor, DatasetHandler, console
from utils.trainer import BaseTrainer
import argparse
from pathlib import Path

# google/gemma-3-4b-it
def create_gemma_config(mode="finetune"):
    """Create Gemma model configuration"""
    # QLoRA fine-tuning configuration (unsupervised learning)
    return ModelConfig(
        model_name="gemma3-4b",
        model_id="google/gemma-3-4b-it",  # Use original large model
        max_length=1024,  # Reasonable sequence length
        batch_size=2,  # 1 → 2 (increase throughput when memory allows, maintain performance)
        learning_rate=2e-4,  # Learning rate suitable for QLoRA
        num_epochs=3,  # Fine-tuning
        warmup_steps=100,
        save_steps=1000,  # 500 → 1000 (reduce I/O frequency, performance unrelated)
        eval_steps=500,   # 250 → 500 (reduce evaluation frequency, performance unrelated)
        gradient_accumulation_steps=8,  # 16 → 8 (maintain effective batch = 2*8=16)
        fp16=True,  # FP16 is stable in QLoRA
        use_lora=True,  # Use QLoRA
        lora_r=16,  # Reasonable LoRA rank
        lora_alpha=32,
        lora_dropout=0.1
    )

def main():
    parser = argparse.ArgumentParser(description="Gemma-3-4B-IT model QLoRA fine-tuning")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Training data path (JSONL format)")
    parser.add_argument("--output_dir", type=str, 
                       default="/checkpoints/gemma3-4b",
                       help="Model save path")
    parser.add_argument("--create_sample_data", action="store_true",
                       help="Generate sample data")
    parser.add_argument("--sample_size", type=int, default=1000,
                       help="Size of sample data to generate")
    
    args = parser.parse_args()
    
    console.print(f"[bold blue]Starting Gemma-3-4B-IT model QLoRA fine-tuning[/bold blue]")
    
    # Print system information
    SystemMonitor.print_system_info()
    
    # Generate sample data (if needed)
    if args.create_sample_data:
        console.print(f"[yellow]Generating sample data for fine-tuning...[/yellow]")
        DatasetHandler.create_sample_dataset(args.data_path, args.sample_size)
    
    # Model configuration
    config = create_gemma_config()
    
    # Execute training
    trainer = BaseTrainer(config, args.data_path, args.output_dir, mode="finetune")
    
    try:
        trainer.train()
        console.print(f"[bold green]Gemma-3-4B-IT QLoRA fine-tuning completed![/bold green]")
        console.print(f"[green]Model save location: {args.output_dir}[/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error during training: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main() 