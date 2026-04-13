# Political Bias Detection in Language Models

Research project for detecting and analyzing political bias in large language models using C4 web dataset.

## Publication
```
@inproceedings{you2026data,
  title={From Data to Model in Bias: A Statistical Analysis of Political Bias in the C4 Corpus and Its Impact on LLMs},
  author={You, Jaebeom and Lee, Jaewon and Lee, Sehun and Kwon, Hyuk-Yoon},
  booktitle={Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining},
  pages={860--870},
  year={2026}
}
```

## Project Structure

- **`Bias_detection/`** - Statistical bias detection and analysis tools
- **`C4_datat_collection/`** - Political content extraction from C4 dataset  
- **`Fine_tuning_model/`** - QLoRA fine-tuning system for bias scenarios
- **`LLM_based_annotation/`** - Multi-persona annotation using ChatGPT/Claude
- **`Political_compass_test/`** - Political compass evaluation framework

## Data Files

- **`statement.json`** - Query statements for evaluating vanilla models' political stances on various topics. Contains structured questions with topic categories, statements, and polarity indicators for bias assessment.
- **`topics-questions.csv`** - Comprehensive list of political topics and keywords used in our analysis. Includes categorized topics (Economics & Markets, Governance & Civil Rights, Social & Cultural Values) with associated keywords and search queries for data collection.

## Quick Start

1. **Data Collection**: Extract political content from C4
2. **Annotation**: Generate bias annotations with multiple LLM personas
3. **Fine-tuning**: Train models on different bias scenarios
4. **Testing**: Evaluate models using political compass questions
5. **Analysis**: Detect and measure bias using statistical methods

## Requirements

- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- OpenAI API key (for ChatGPT)
- Anthropic API key (for Claude) 
