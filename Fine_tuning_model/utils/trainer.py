import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
import time
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import numpy as np
from .common import ModelConfig, TrainingLogger, console, format_time

class UnsupervisedDataset(Dataset):
    """Dataset class for unsupervised learning (natural text learning)"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Process texts for natural learning
        for text in texts:
            if len(text.strip()) < 20:  # Exclude too short texts
                continue
                
            # Tokenization
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
            
            if len(tokens) >= 10:  # Ensure minimum length
                self.examples.append({
                    'input_ids': tokens,
                    'attention_mask': [1] * len(tokens)
                })
        
        console.print(f"[green]Total {len(self.examples)} training samples prepared[/green]")
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

class BaseTrainer:
    """Base training class"""
    
    def __init__(self, config: ModelConfig, data_path: str, output_dir: str, mode: str = "finetune"):
        self.config = config
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode  # "finetune" (unsupervised learning)
        
        # Initialize logger
        self.logger = TrainingLogger(
            log_dir=str(self.output_dir / "logs"),
            model_name=config.model_name
        )
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[green]Using device: {self.device}[/green]")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer (QLoRA support)"""
        console.print(f"[yellow]Loading model: {self.config.model_id}[/yellow]")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 4-bit quantization setup for QLoRA
        if self.config.use_lora:
            console.print("[yellow]QLoRA mode: Applying 4-bit quantization[/yellow]")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # Load model (4-bit quantization)
            # Configure Phi-4-mini to work without flash-attn
            model_kwargs = {
                "quantization_config": bnb_config,
                "device_map": "auto",
                "trust_remote_code": True
            }
            
            # Set attention implementation to eager for Phi models (without flash-attn)
            if "phi" in self.config.model_id.lower():
                model_kwargs["attn_implementation"] = "eager"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs
            )
            
            # Prepare model for QLoRA
            self.model = prepare_model_for_kbit_training(self.model)
            self._apply_lora()
            
        else:
            # Normal mode
            model_kwargs = {
                "torch_dtype": torch.float16 if self.config.fp16 else torch.float32,
                "device_map": "auto",
                "trust_remote_code": True
            }
            
            # Set attention implementation to eager for Phi models
            if "phi" in self.config.model_id.lower():
                model_kwargs["attn_implementation"] = "eager"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs
            )
            
        console.print(f"[green]Model loading completed[/green]")
        
    def _apply_lora(self):
        """Apply QLoRA configuration"""
        console.print("[yellow]Applying QLoRA configuration...[/yellow]")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
    def load_dataset(self):
        """Load dataset (for unsupervised learning)"""
        console.print(f"[yellow]Loading dataset: {self.data_path}[/yellow]")
        
        # Read JSONL file
        texts = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'text' in data:
                        texts.append(data['text'])
                    elif 'content' in data:
                        texts.append(data['content'])
                    else:
                        # Try other field names
                        for key in data.keys():
                            if isinstance(data[key], str) and len(data[key]) > 20:
                                texts.append(data[key])
                                break
                except json.JSONDecodeError:
                    continue
        
        if not texts:
            raise ValueError(f"No text found in data file: {self.data_path}")
        
        console.print(f"[green]Total {len(texts)} texts loaded[/green]")
        
        # Split dataset (90% train, 10% validation)
        split_idx = int(len(texts) * 0.9)
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:] if len(texts) > 10 else texts[:min(10, len(texts))]
        
        # Create datasets
        self.train_dataset = UnsupervisedDataset(
            train_texts, self.tokenizer, self.config.max_length
        )
        self.eval_dataset = UnsupervisedDataset(
            eval_texts, self.tokenizer, self.config.max_length
        )
        
        console.print(f"[green]Training data: {len(self.train_dataset)}, Validation data: {len(self.eval_dataset)}[/green]")
        
    def setup_training_args(self):
        """Setup training arguments"""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=50,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.fp16,
            max_grad_norm=1.0,  # gradient clipping setup
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="tensorboard",
            logging_dir=str(self.output_dir / "tensorboard_logs"),
        )
    
    def train(self):
        """Execute model training"""
        start_time = time.time()
        
        # Load model and dataset
        self.load_model_and_tokenizer()
        self.load_dataset()
        
        # Training setup
        training_args = self.setup_training_args()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
        
        # Training start log
        config_dict = {
            "model_name": self.config.model_name,
            "model_id": self.config.model_id,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "num_epochs": self.config.num_epochs,
            "use_lora": self.config.use_lora
        }
        self.logger.log_training_start(config_dict)
        
        console.print(f"[green]Training started: {self.config.model_name}[/green]")
        
        # Execute training
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        # Training completion log
        total_time = time.time() - start_time
        self.logger.log_training_complete(total_time)
        
        console.print(f"[green]Training completed! Total time: {format_time(total_time)}[/green]")
        console.print(f"[green]Model save location: {self.output_dir}[/green]")
        
        return trainer
    
    def evaluate_model(self, test_data_path: Optional[str] = None):
        """Model evaluation"""
        if test_data_path:
            # Evaluate with test data
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = [json.loads(line) for line in f]
            
            test_dataset = UnsupervisedDataset(
                test_data, self.tokenizer, self.config.max_length
            )
            
            # Execute evaluation
            trainer = Trainer(
                model=self.model,
                eval_dataset=test_dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer, mlm=False
                )
            )
            
            eval_results = trainer.evaluate()
            self.logger.log_evaluation(eval_results)
            
            return eval_results
        
    def generate_text(self, prompt: str, max_length: int = 512):
        """Text generation"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded.")
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        
        # Remove input prompt
        generated_text = generated_text[len(prompt):].strip()
        
        return generated_text 