#!/usr/bin/env python3
"""
Scenario-specific Model QLoRA Fine-tuning Script

Fine-tune all models with QLoRA for each scenario 
(balanced, left_heavy, right_heavy, neutral_heavy).
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

console = Console()

# Model definitions
MODELS = {
    "llama3.2-3b": {
        "script_path": "models/llama3.2-3b/train_llama.py",
        "checkpoint_dir": "checkpoints/llama3.2-3b",
        "description": "Llama-3.2-3B-Instruct"
    },
    "gemma3-4b": {
        "script_path": "models/gemma3-4b/train_gemma.py",
        "checkpoint_dir": "checkpoints/gemma3-4b", 
        "description": "Gemma-3-4B-IT"
    },
}

# Scenario definitions
SCENARIOS = ["left_extreme", "right_extreme"]

SCENARIOS_DIR = "data/extreme_datasets"
# Number of training repetitions
REPEAT_COUNT = 3

def check_data_files():
    """Check if scenario data files exist"""
    console.print("[yellow]Checking scenario data files...[/yellow]")
    
    missing_files = []
    for scenario in SCENARIOS:
        data_file = f"{SCENARIOS_DIR}/{scenario}_finetune.jsonl"
        if not os.path.exists(data_file):
            missing_files.append(data_file)
    
    if missing_files:
        console.print("[red]The following data files are missing:[/red]")
        for file in missing_files:
            console.print(f"  - {file}")
        console.print("[yellow]Please run 1_extreme_prepare_scenario_datasets.py first.[/yellow]")
        return False
    
    console.print("[green]All scenario data files exist.[/green]")
    return True

def tail_log_file(log_file: Path, stop_event: threading.Event):
    """Function to tail log file in real-time"""
    if not log_file.exists():
        return
    
    with open(log_file, 'r') as f:
        # Move to end of file
        f.seek(0, 2)
        
        while not stop_event.is_set():
            line = f.readline()
            if line:
                # Only output training progress or important information
                if any(keyword in line.lower() for keyword in ['epoch', 'loss', 'learning_rate', 'step', 'training', 'error', 'completed']):
                    console.print(f"[dim]  → {line.strip()}[/dim]")
            else:
                time.sleep(0.5)

def run_training(model_name: str, model_config: dict, scenario: str, repeat_num: int):
    """Execute individual model training"""
    data_path = f"{SCENARIOS_DIR}/{scenario}_finetune.jsonl"
    output_dir = f"{model_config['checkpoint_dir']}_{scenario}_run{repeat_num}"
    
    # Activate virtual environment and run training
    cmd = [
        "bash", "-c",
        f"python {model_config['script_path']} "
        f"--data_path {data_path} "
        f"--output_dir {output_dir}"
    ]
    
    console.print(f"\n[bold blue]🚀 Starting: {model_config['description']} - {scenario} scenario (Run {repeat_num})[/bold blue]")
    console.print(f"[dim]📁 Output directory: {output_dir}[/dim]")
    console.print(f"[dim]📄 Data file: {data_path}[/dim]")
    
    start_time = time.time()
    
    try:
        # Set up log file
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{model_name}_{scenario}_run{repeat_num}.log"
        
        console.print(f"[dim]📝 Log file: {log_file}[/dim]")
        console.print("[yellow]Training started... (real-time log output)[/yellow]")
        
        # Set up thread for log tailing
        stop_event = threading.Event()
        log_thread = threading.Thread(target=tail_log_file, args=(log_file, stop_event))
        
        # Execute process
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd(),
                bufsize=1,
                universal_newlines=True
            )
            
            # Real-time output
            for line in iter(process.stdout.readline, ''):
                f.write(line)
                f.flush()
                
                # Only output important logs in real-time
                if any(keyword in line.lower() for keyword in ['epoch', 'loss', 'learning_rate', 'step', 'training', 'error', 'completed', 'saving']):
                    console.print(f"[dim]  → {line.strip()}[/dim]")
            
            process.wait()
        
        stop_event.set()
        
        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        if process.returncode == 0:
            console.print(f"[bold green]✅ Completed: {model_config['description']} - {scenario} Run {repeat_num}[/bold green]")
            console.print(f"[green]⏱️  Duration: {hours:02d}:{minutes:02d}:{seconds:02d}[/green]")
            console.print(f"[green]📁 Checkpoint: {output_dir}[/green]")
            return True
        else:
            console.print(f"[bold red]❌ Failed: {model_config['description']} - {scenario} Run {repeat_num}[/bold red]")
            console.print(f"[red]🚨 Exit code: {process.returncode}[/red]")
            console.print(f"[red]📝 Log file: {log_file}[/red]")
            
            # Output last few lines of log
            if log_file.exists():
                console.print("[red]Last log content:[/red]")
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:  # Last 10 lines
                        console.print(f"[red]  {line.strip()}[/red]")
            return False
            
    except Exception as e:
        console.print(f"[bold red]💥 Error: {model_config['description']} - {scenario} Run {repeat_num}[/bold red]")
        console.print(f"[red]🔍 Error details: {str(e)}[/red]")
        return False

def main():
    """Main function"""
    console.print("[bold blue]🎯 Scenario-specific Model QLoRA Repeated Fine-tuning Started[/bold blue]")
    console.print(f"[cyan]📊 Execution plan: {len(MODELS)} models × {len(SCENARIOS)} scenarios × {REPEAT_COUNT} repetitions = {len(MODELS) * len(SCENARIOS) * REPEAT_COUNT} tasks[/cyan]")
    
    # Check data files
    if not check_data_files():
        return
    
    # Result tracking
    results = {}
    total_tasks = len(MODELS) * len(SCENARIOS) * REPEAT_COUNT
    completed_tasks = 0
    
    # Progress table
    table = Table(title="QLoRA Repeated Fine-tuning Progress")
    table.add_column("Model", style="cyan")
    table.add_column("Scenario", style="magenta")
    table.add_column("Run", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Output Directory", style="blue")
    
    console.print(f"[yellow]Total {total_tasks} training tasks to execute. (Models: {len(MODELS)}, Scenarios: {len(SCENARIOS)}, Repetitions: {REPEAT_COUNT})[/yellow]")
    
    # Initialize result dictionary
    for model_name in MODELS.keys():
        results[model_name] = {}
        for scenario in SCENARIOS:
            results[model_name][scenario] = {}
    
    # Execute all model/scenario combinations for each round
    for repeat_num in range(1, REPEAT_COUNT + 1):
        console.print(f"\n[bold magenta]🎯 ========== Round {repeat_num}/{REPEAT_COUNT} Started ===========[/bold magenta]")
        console.print(f"[yellow]This round tasks: {len(MODELS) * len(SCENARIOS)}[/yellow]")
        
        round_start_time = time.time()
        
        for model_idx, (model_name, model_config) in enumerate(MODELS.items(), 1):
            console.print(f"\n[bold cyan]📱 Model {model_idx}/{len(MODELS)}: {model_config['description']}[/bold cyan]")
            
            for scenario_idx, scenario in enumerate(SCENARIOS, 1):
                console.print(f"[cyan]  📋 Scenario {scenario_idx}/{len(SCENARIOS)}: {scenario}[/cyan]")
                
                success = run_training(model_name, model_config, scenario, repeat_num)
                results[model_name][scenario][repeat_num] = success
                completed_tasks += 1
                
                # Update table
                status = "✅ Success" if success else "❌ Failed"
                output_dir = f"{model_config['checkpoint_dir']}_{scenario}_run{repeat_num}"
                table.add_row(
                    model_config['description'],
                    scenario,
                    f"Run {repeat_num}",
                    status,
                    output_dir
                )
                
                # Calculate overall progress
                progress_percent = (completed_tasks / total_tasks) * 100
                console.print(f"[bold blue]📊 Overall progress: {completed_tasks}/{total_tasks} ({progress_percent:.1f}%)[/bold blue]")
                
                # Short wait for GPU memory cleanup
                if success:
                    console.print("[dim]GPU memory cleanup... (3 seconds wait)[/dim]")
                    time.sleep(3)
        
        # Output intermediate results after each round completion
        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        round_hours = int(round_duration // 3600)
        round_minutes = int((round_duration % 3600) // 60)
        round_seconds = int(round_duration % 60)
        
        console.print(f"\n[bold green]🎉 ========== Round {repeat_num} Completed! ===========[/bold green]")
        console.print(f"[green]⏱️  Round duration: {round_hours:02d}:{round_minutes:02d}:{round_seconds:02d}[/green]")
        
        round_success = sum(sum(1 for run_num, success in runs.items() if run_num == repeat_num and success) for scenarios in results.values() for runs in scenarios.values())
        round_total = len(MODELS) * len(SCENARIOS)
        console.print(f"[cyan]📈 Round {repeat_num} success rate: {round_success}/{round_total} ({(round_success/round_total)*100:.1f}%)[/cyan]")
        
        # Cumulative success rate so far
        total_completed_so_far = completed_tasks
        total_success_so_far = sum(sum(sum(runs.values()) for runs in scenarios.values()) for scenarios in results.values())
        console.print(f"[blue]📊 Cumulative success rate: {total_success_so_far}/{total_completed_so_far} ({(total_success_so_far/total_completed_so_far)*100:.1f}%)[/blue]")
        
        if repeat_num < REPEAT_COUNT:
            console.print(f"[yellow]⏸️  Waiting 10 seconds until next round... (GPU memory cleanup)[/yellow]")
            for i in range(10, 0, -1):
                console.print(f"[dim]  {i} seconds remaining...[/dim]")
                time.sleep(1)
    
    # Final result output
    console.print(table)
    
    # Summary statistics
    total_success = sum(sum(sum(runs.values()) for runs in scenarios.values()) for scenarios in results.values())
    total_failure = total_tasks - total_success
    
    console.print(f"\n[bold]Final Results:[/bold]")
    console.print(f"[green]Success: {total_success}[/green]")
    console.print(f"[red]Failed: {total_failure}[/red]")
    
    if total_failure > 0:
        console.print(f"\n[yellow]Failed tasks:[/yellow]")
        for model_name, scenarios in results.items():
            for scenario, runs in scenarios.items():
                for run_num, success in runs.items():
                    if not success:
                        console.print(f"  - {MODELS[model_name]['description']} - {scenario} - Run {run_num}")
    
    # Model-specific scenario success rate summary
    console.print(f"\n[bold cyan]Success Rate Summary by Model:[/bold cyan]")
    for model_name, scenarios in results.items():
        model_success = sum(sum(runs.values()) for runs in scenarios.values())
        model_total = len(scenarios) * REPEAT_COUNT
        success_rate = (model_success / model_total) * 100 if model_total > 0 else 0
        console.print(f"  {MODELS[model_name]['description']}: {model_success}/{model_total} ({success_rate:.1f}%)")
    
    console.print(f"\n[bold green]Scenario-specific QLoRA Repeated Fine-tuning Completed![/bold green]")
    console.print(f"[blue]Generated models: {len(SCENARIOS) * REPEAT_COUNT} per model for each model[/blue]")

if __name__ == "__main__":
    main() 