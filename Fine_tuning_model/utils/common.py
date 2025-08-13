import os
import json
import torch
import logging
import psutil
import GPUtil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import time
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

@dataclass
class ModelConfig:
    """Data class for model configuration"""
    model_name: str
    model_id: str
    max_length: int = 2048
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    save_steps: int = 1000
    eval_steps: int = 500
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

class SystemMonitor:
    """System resource monitoring class"""
    
    @staticmethod
    def get_gpu_info():
        """Return GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total': f"{gpu.memoryTotal}MB",
                    'memory_used': f"{gpu.memoryUsed}MB",
                    'memory_free': f"{gpu.memoryFree}MB",
                    'utilization': f"{gpu.load*100:.1f}%",
                    'temperature': f"{gpu.temperature}°C"
                })
            return gpu_info
        except:
            return []
    
    @staticmethod
    def get_cpu_memory_info():
        """Return CPU and memory information"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        return {
            'cpu_usage': f"{cpu_percent:.1f}%",
            'memory_total': f"{memory.total / (1024**3):.1f}GB",
            'memory_used': f"{memory.used / (1024**3):.1f}GB",
            'memory_available': f"{memory.available / (1024**3):.1f}GB",
            'memory_percent': f"{memory.percent:.1f}%"
        }
    
    @staticmethod
    def print_system_info():
        """Print system information in a nice format"""
        table = Table(title="System Resource Information")
        table.add_column("Category", style="cyan")
        table.add_column("Information", style="green")
        
        # CPU/Memory information
        sys_info = SystemMonitor.get_cpu_memory_info()
        table.add_row("CPU Usage", sys_info['cpu_usage'])
        table.add_row("Memory Usage", sys_info['memory_percent'])
        table.add_row("Available Memory", sys_info['memory_available'])
        
        # GPU information
        gpu_info = SystemMonitor.get_gpu_info()
        if gpu_info:
            for i, gpu in enumerate(gpu_info):
                table.add_row(f"GPU {i} ({gpu['name']})", 
                            f"Usage: {gpu['utilization']}, Memory: {gpu['memory_used']}/{gpu['memory_total']}")
        else:
            table.add_row("GPU", "Not available")
        
        console.print(table)

class DatasetHandler:
    """Class for dataset processing"""
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    @staticmethod
    def save_jsonl(data: List[Dict], file_path: str):
        """Save JSONL file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    @staticmethod
    def create_sample_dataset(output_path: str, num_samples: int = 1000):
        """Create sample dataset"""
        sample_data = []
        for i in range(num_samples):
            sample_data.append({
                "instruction": f"Please answer the following question: Question {i+1}",
                "input": f"This is the {i+1}th input text.",
                "output": f"This is the {i+1}th output text. Provides detailed answer."
            })
        
        DatasetHandler.save_jsonl(sample_data, output_path)
        console.print(f"[green]Sample dataset created: {output_path}[/green]")

class TrainingLogger:
    """Class for training logging"""
    
    def __init__(self, log_dir: str, model_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup log file
        log_file = self.log_dir / f"{model_name}_{int(time.time())}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(model_name)
    
    def log_training_start(self, config: Dict[str, Any]):
        """Training start log"""
        self.logger.info(f"Training started: {config}")
        
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Epoch start log"""
        self.logger.info(f"Epoch {epoch+1}/{total_epochs} started")
        
    def log_step(self, step: int, loss: float, learning_rate: float):
        """Step-by-step log"""
        self.logger.info(f"Step {step} - Loss: {loss:.4f}, LR: {learning_rate:.2e}")
        
    def log_evaluation(self, metrics: Dict[str, float]):
        """Evaluation result log"""
        self.logger.info(f"Evaluation results: {metrics}")
        
    def log_training_complete(self, total_time: float):
        """Training completion log"""
        self.logger.info(f"Training completed. Total time: {total_time:.2f}s")

def setup_device():
    """Device setup"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(f"[green]CUDA available: {torch.cuda.get_device_name()}[/green]")
    else:
        device = torch.device("cpu")
        console.print("[yellow]Running in CPU mode[/yellow]")
    
    return device

def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration as JSON file"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load JSON configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_time(seconds: float) -> str:
    """Format time in readable form"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s" 