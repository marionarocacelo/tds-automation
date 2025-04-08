# Table Analyzer

This tool analyzes CSV tables and enhances them by adding reference keys from a context Excel file using AI models.

## Features

- Supports multiple AI providers:
  - Gemini
  - Groq
  - DeepSeek
- Processes CSV files in batch
- Adds reference keys to tables based on content matching

## Setup

1. Install required packages:
```bash
pip install pandas google-generativeai openpyxl groq openai
```

2. Configure API keys in `config.json`:
```json
{
    "google_api_key": "your_gemini_api_key",
    "groq_api_key": "your_groq_api_key",
    "deepseek_api_key": "your_deepseek_api_key"
}
```

## Usage

Run the script with the desired AI model and context file:

```bash
python table_analyzer.py [model] [context_file]
```

Parameters:
- `model`: Choose from "gemini", "groq", or "deepseek" (default: "gemini")
- `context_file`: Path to the Excel file containing reference keys (default: "./context/PIM-ABT.xlsx")

Example:
```bash
python table_analyzer.py deepseek ./context/PIM-ABT.xlsx
```

## Output

The enhanced tables will be saved in the `output/enhanced_tables` directory with the prefix "enhanced_". Each table will have an additional "references" column containing comma-separated matching keys from the context file. 