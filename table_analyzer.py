import os
import pandas as pd
import google.generativeai as genai
from groq import Groq
from openai import OpenAI
from pathlib import Path
import json
import logging
import sys
from typing import List, Dict, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableAnalyzer:
    def __init__(self, context_excel_path: str, output_dir: str = "output", model: str = "gemini"):
        self.context_excel_path = Path(context_excel_path)
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / "tables"
        self.enhanced_dir = self.output_dir / "enhanced_tables"
        self.enhanced_dir.mkdir(exist_ok=True)
        self.model = model.lower()
        
        # Load context data
        self.context_df = pd.read_excel(self.context_excel_path)
        self.context_keys = self.context_df['key'].tolist()
        logger.info(f"Loaded {len(self.context_keys)} context keys from Excel file")
        
        # Initialize AI API
        if self.model == "gemini":
            self.setup_gemini()
        elif self.model == "groq":
            self.setup_groq()
        elif self.model == "deepseek":
            self.setup_deepseek()
        else:
            raise ValueError(f"Unsupported model: {model}. Choose from: gemini, groq, deepseek")
    
    def setup_gemini(self):
        """Setup Gemini API with configuration from environment or config file"""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                config_path = Path('config.json')
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                        api_key = config.get('google_api_key')
            
            if not api_key:
                raise ValueError("Google API key not found. Please set GOOGLE_API_KEY environment variable or add to config.json")
            
            genai.configure(api_key=api_key)
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Error setting up Gemini API: {str(e)}")
            raise

    def setup_groq(self):
        """Setup Groq API"""
        try:
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                config_path = Path('config.json')
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                        api_key = config.get('groq_api_key')
            
            if not api_key:
                raise ValueError("Groq API key not found. Please set GROQ_API_KEY environment variable or add to config.json")
            
            self.groq_client = Groq(api_key=api_key)
            logger.info("Groq API configured successfully")
        except Exception as e:
            logger.error(f"Error setting up Groq API: {str(e)}")
            raise

    def setup_deepseek(self):
        """Setup DeepSeek API"""
        try:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                config_path = Path('config.json')
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                        api_key = config.get('deepseek_api_key')
            
            if not api_key:
                raise ValueError("DeepSeek API key not found. Please set DEEPSEEK_API_KEY environment variable or add to config.json")
            
            self.deepseek_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            logger.info("DeepSeek API configured successfully")
        except Exception as e:
            logger.error(f"Error setting up DeepSeek API: {str(e)}")
            raise
    
    def find_matching_references_gemini(self, row_data: str) -> str:
        """Use Gemini to find matching references for a row"""
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""Analyze the following table row data and find the most relevant reference key from the provided list.
        The reference should be the most appropriate match based on the content of the row.

        Row data: {row_data}
        Available reference keys: {', '.join(self.context_keys)}

        Return ONLY the matching reference key. If no clear match is found, return 'None'.
        Do not include any explanations or additional text."""

        try:
            response = model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                result = response.text.strip()
                logger.info(f"RESULT: {result}")
                if result in self.context_keys:
                    return result
                return 'None'
        except Exception as e:
            logger.error(f"Error getting Gemini response: {str(e)}")
            return 'None'
        
        return 'None'

    def find_matching_references_groq(self, row_data: str) -> str:
        """Use Groq to find matching references for a row"""
        logger.info(f"ROW DATA: {row_data}")
        prompt = f"""Analyze the following table row data and find all relevant reference keys from the provided list.
        The references should be appropriate matches based on the content of the row.

        Row data: {row_data}
        Available reference keys: {', '.join(self.context_keys)}

        Return a comma-separated list of matching reference keys. If no matches are found, return 'None'.
        Do not include any explanations or additional text."""

        try:
            completion = self.groq_client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            result = completion.choices[0].message.content.strip()
            logger.info(f"GROQ RESULT: {result}")
            
            if result.lower() == 'none':
                return 'None'
                
            # Split the result by commas and clean up each key
            matching_keys = [key.strip() for key in result.split(',')]
            # Filter to only include keys that exist in context_keys
            valid_keys = [key for key in matching_keys if key in self.context_keys]
            
            if valid_keys:
                return ','.join(valid_keys)
            return 'None'
            
        except Exception as e:
            logger.error(f"Error getting Groq response: {str(e)}")
            return 'None'

    def find_matching_references_deepseek(self, row_data: str) -> str:
        """Use DeepSeek to find matching references for a row"""
        prompt = f"""Analyze the following table row data and find the most relevant reference key from the provided list.
        The reference should be the most appropriate match based on the content of the row.

        Row data: {row_data}
        Available reference keys: {', '.join(self.context_keys)}

        Return ONLY the matching reference key. If no clear match is found, return 'None'.
        Do not include any explanations or additional text."""

        try:
            completion = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
                stream=False
            )
            
            result = completion.choices[0].message.content.strip()
            if result in self.context_keys:
                return result
            return 'None'
        except Exception as e:
            logger.error(f"Error getting DeepSeek response: {str(e)}")
            return 'None'
    
    def find_matching_references(self, row_data: str) -> str:
        """Find matching references using the selected AI provider"""
        if self.model == "gemini":
            return self.find_matching_references_gemini(row_data)
        elif self.model == "groq":
            return self.find_matching_references_groq(row_data)
        elif self.model == "deepseek":
            return self.find_matching_references_deepseek(row_data)
        else:
            raise ValueError(f"Unsupported model: {self.model}")
    
    def analyze_table(self, csv_path: Path) -> pd.DataFrame:
        """Analyze a single table and find matching references"""
        logger.info(f"Analyzing table: {csv_path.name}")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"DF 1: {df}")
        # Create a new column for references
        df['references'] = None
        
        logger.info(f"DF 2: {df}")

        # Process each row
        for idx, row in df.iterrows():
            try:
                # Convert row to string for analysis
                logger.info(f"ROW: {row}")
                
                row_text = row.to_string()
                
                logger.info(f"ROW TEXT: {row_text}")
                logger.info(f"idx: {idx}")

                # Find matching references
                references = self.find_matching_references(row_text)
                
                # Update the references column
                df.at[idx, 'references'] = references
                
                # Add small delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                continue
        
        return df
    
    def process_all_tables(self):
        """Process all CSV files in the tables directory"""
        csv_files = list(self.tables_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in tables directory")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        for csv_file in csv_files:
            try:
                logger.info(f"\nProcessing {csv_file.name}...")
                
                # Analyze the table
                enhanced_df = self.analyze_table(csv_file)
                
                # Save enhanced table
                output_path = self.enhanced_dir / f"enhanced_{csv_file.name}"
                enhanced_df.to_csv(output_path, index=False)
                
                logger.info(f"Saved enhanced table to {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {str(e)}")
                continue

def main():
    # Check if model is provided
    if len(sys.argv) > 1:
        model = sys.argv[1].lower()
        if model not in ["gemini", "groq", "deepseek"]:
            logger.error("Invalid model. Choose from: gemini, groq, deepseek")
            sys.exit(1)
    else:
        model = "gemini"  # Default to Gemini
    
    # Check if context file is provided
    if len(sys.argv) > 2:
        context_path = sys.argv[2]
    else:
        context_path = "./context/PIM-ABT.xlsx"  # Default context file name
    
    try:
        analyzer = TableAnalyzer(context_path, model=model)
        analyzer.process_all_tables()
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main() 