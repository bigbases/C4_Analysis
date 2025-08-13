#!/usr/bin/env python3
"""
Simple Political Compass Test
Load one model and ask questions sequentially
"""

import os
import sys
import json
import torch
import pandas as pd
import re
from pathlib import Path


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

class SimplePoliticalTest:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.tokenizer = None
        
        if not self.checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    def _detect_base_model(self) -> str:
        """Detect base model from checkpoint path"""
        # Check adapter_config.json
        adapter_config_path = self.checkpoint_path / "adapter_config.json"
        if adapter_config_path.exists():
            try:
                with open(adapter_config_path, 'r') as f:
                    config = json.load(f)
                    base_model = config.get("base_model_name_or_path")
                    if base_model:
                        print(f"Base model detected: {base_model}")
                        return base_model
            except Exception as e:
                print(f"Failed to read adapter_config.json: {e}")
        
        # Fallback to path inference
        path_str = str(self.checkpoint_path).lower()
        if "gemma3-4b" in path_str:
            return "google/gemma-3-4b-it"
        elif "llama3.2-3b" in path_str:
            return "meta-llama/Llama-3.2-3B-Instruct"
                
        raise ValueError("Cannot detect base model. Please specify --base_model_id")
    
    def load_model(self):
        """Load model and tokenizer"""
        base_model_id = self._detect_base_model()
        print(f"Loading base model: {base_model_id}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            padding_side="left",
            cache_dir="/tmp/huggingface_cache"
        )
        
        # Handle special tokens for different models
        if self.tokenizer.pad_token is None:
            if "gemma" in base_model_id.lower():
                # Gemma has specific pad token handling
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 4-bit quantization config
        compute_dtype = torch.bfloat16 if "gemma" in base_model_id.lower() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
        
        # Load base model
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": {"": 0},  # Explicitly place on GPU 1 (with CUDA_VISIBLE_DEVICES=1 setting)
            "trust_remote_code": True
        }
        
        # Model-specific attention implementation
        if "phi" in base_model_id.lower():
            model_kwargs["attn_implementation"] = "eager"
        elif "gemma" in base_model_id.lower():
            model_kwargs["attn_implementation"] = "eager"  # Use eager for Gemma too
            model_kwargs["torch_dtype"] = torch.bfloat16  # Gemma prefers bfloat16
        
        # Add cache directory
        model_kwargs["cache_dir"] = "/tmp/huggingface_cache"
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            **model_kwargs
        )
        
        print(f"Loading LoRA adapter: {self.checkpoint_path}")
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            str(self.checkpoint_path),
            is_trainable=False
        )
        
        self.model.eval()
        
        # Check GPU usage
        if torch.cuda.is_available():
            print(f"Model loaded successfully on GPU 1 (mapped as: {torch.cuda.get_device_name(0)})")
            print(f"Current device: {next(self.model.parameters()).device}")
        else:
            print("Model loaded successfully (CPU mode)")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate response for prompt"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        
        # Move to device safely
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generation config
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
            "use_cache": False,  # Fix DynamicCache error
            "output_scores": False,
            "return_dict_in_generate": False,
        }
        
        # Generate response
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_config
                )
                
                # Extract response (remove input part)
                input_length = inputs["input_ids"].shape[1]
                response_ids = outputs[0][input_length:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                
            except Exception as e:
                print(f"Generation error: {e}")
                response = "Error in generation"
        
        return response
    
    def generate_filename(self, test_num: int) -> str:
        """Generate filename using checkpoint path + test number"""
        checkpoint_name = self.checkpoint_path.name
        return f"{checkpoint_name}_test{test_num}.csv"
    
    def create_output_directory(self) -> Path:
        """Create output directory based on checkpoint name"""
        checkpoint_name = self.checkpoint_path.name
        output_dir = Path("results") / checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def save_results_to_csv(self, results: list, filename: str, output_dir: Path):
        """Save test results to CSV file in the specified directory"""
        filepath = output_dir / filename
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Results saved to: {filepath}")

    def ask_question(self, question: str, options: list) -> str:
        """Ask a political compass question"""
        # Create English prompt
        options_text = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])
        
        prompt = f"""Please answer the following question by selecting the most appropriate option.

Question: {question}

Options:
{options_text}

Please select the number of your choice and briefly explain your reasoning.
Answer: """
        
        response = self.generate_response(prompt, max_new_tokens=150)
        return response
    
    def load_questions(self, questions_file: str) -> list:
        """Load questions from JSON file"""
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data["questions"]
    
    def run_single_test(self, questions_file: str, round_num: int):
        """Run a single political compass test and save results to CSV"""
        # Load questions
        questions = self.load_questions(questions_file)
        print(f"Loaded {len(questions)} questions")
        print(f"Running round {round_num} test for checkpoint: {self.checkpoint_path.name}")
        print("=" * 80)
        
        # Store results
        results = []
        
        # Process each question
        for i, q_data in enumerate(questions):
            print(f"\nQuestion {i+1}/{len(questions)}")
            print(f"Topic: {q_data['topic']}")
            print(f"Question: {q_data['question']}")
            print("Options:")
            for j, option in enumerate(q_data['options']):
                print(f"  {j+1}. {option}")
            
            try:
                response = self.ask_question(q_data['question'], q_data['options'])
                
                print(f"Model Response: {response}")
                
                # Store result (no parsing, just raw data)
                results.append({
                    'question_id': i + 1,
                    'topic': q_data['topic'],
                    'question': q_data['question'],
                    'raw_answer': response
                })
                
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                results.append({
                    'question_id': i + 1,
                    'topic': q_data.get('topic', 'Unknown'),
                    'question': q_data.get('question', 'Unknown'),
                    'raw_answer': f"Error: {str(e)}"
                })
            
            print("-" * 80)
        
        # Save results to CSV with round number
        filename = f"{self.checkpoint_path.name}_round{round_num}.csv"
        output_dir = self.create_output_directory()
        self.save_results_to_csv(results, filename, output_dir)
        
        return results


def main():
    # Check GPU
    if torch.cuda.is_available():
        print(f"Using GPU 1 (mapped as: {torch.cuda.get_device_name(0)})")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Current CUDA device: {torch.cuda.current_device()} (actual GPU 1)")
    else:
        print("CUDA not available, using CPU")
    print("-" * 50)
    
    # Configuration - List of checkpoint paths to test
    checkpoint_paths = [
        "../Fine_tuning_model/checkpoints/{}",
    ]
    
    questions_file = "political_compass_test_Q/questions.json"
    num_rounds = 10          # Number of rounds to repeat the entire checkpoint sequence
    
    # Check questions file exists
    if not Path(questions_file).exists():
        print(f"Questions file not found: {questions_file}")
        return
    
    print("Starting Political Compass Test Suite")
    print(f"Questions file: {questions_file}")
    print(f"Number of checkpoints: {len(checkpoint_paths)}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Total tests: {len(checkpoint_paths)} × {num_rounds} = {len(checkpoint_paths) * num_rounds}")
    print("=" * 80)
    
    # Repeat the entire checkpoint sequence for num_rounds
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*30} ROUND {round_num}/{num_rounds} {'='*30}")
        
        # Process each checkpoint in this round
        for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths):
            print(f"\n{'='*20} Round {round_num} - Checkpoint {checkpoint_idx+1}/{len(checkpoint_paths)} {'='*20}")
            print(f"Checkpoint: {checkpoint_path}")
            
            # Check if checkpoint exists
            if not Path(checkpoint_path).exists():
                print(f"Checkpoint not found: {checkpoint_path}")
                print("Skipping to next checkpoint...")
                continue
            
            try:
                # Create tester and load model
                print("Loading model...")
                tester = SimplePoliticalTest(checkpoint_path)
                tester.load_model()
                
                # Run single test for this checkpoint in this round
                print(f"Running test for round {round_num}...")
                results = tester.run_single_test(questions_file, round_num)
                
                print(f"Round {round_num} - Checkpoint {checkpoint_path} completed!")
                
                # Clean up model to free memory
                del tester.model
                del tester.tokenizer
                del tester
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"Error processing checkpoint {checkpoint_path} in round {round_num}: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing to next checkpoint...")
                continue
        
        print(f"\nROUND {round_num} COMPLETED!")
    
    print(f"\nAll {num_rounds} rounds completed!")
    print(f"Total tests performed: {len(checkpoint_paths) * num_rounds}")
    print("Results are organized in directories under 'results/' folder")


if __name__ == "__main__":
    main() 