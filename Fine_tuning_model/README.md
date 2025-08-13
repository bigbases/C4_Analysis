# Fine-tuning Models

QLoRA fine-tuning system for small language models with scenario-based training.

## Supported Models
- Gemma-3-4B-IT (4.3B parameters)
- LLama-3.2-3B-Instruct (3.21B parameters)

## Files
- `train_scenario_models.py` - Main training orchestrator
- `balanced_sampling.py` - Balanced sampling strategy
- `extreme_sampling.py` - Extreme bias sampling strategy
- `models/` - Model-specific training scripts
- `utils/` - Common utilities for training

## Usage
```bash
python train_scenario_models.py
``` 