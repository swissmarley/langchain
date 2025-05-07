import os
import pandas as pd
import sqlite3
import json
import requests
from typing import Dict, Any, Optional, List
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, create_react_agent 
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List


class DatasetCreationTool(BaseTool):
    """
    A tool for creating datasets from web sources with flexible output formats.
    """
    name: str = Field(default="dataset_creator")
    description: str = Field(default="Creates datasets from web sources with flexible output formats")
    brave_api_key: str = Field(description="API key for Brave Search")
    openai_api_key: str = Field(description="API key for OpenAI")

    class Config:
        arbitrary_types_allowed = True

    def _run(self, tool_input: str = "") -> str:
        """
        Primary method to run dataset creation workflow
        """
        try:
            # Get dataset details from user
            dataset_name = input("Enter the name of the dataset: ")
            source_url = input("Enter the source URL to scrape: ")
            
            # Choose file type
            print("Choose output file type:")
            print("1. CSV")
            print("2. JSON")
            print("3. SQLite")
            file_type_choice = int(input("Enter your choice (1-3): "))
            
            # Perform web scraping and data analysis
            print("Scraping data from URL...")
            raw_data = self._scrape_data(source_url)
            
            if not raw_data:
                return "Error: No data was scraped from the URL"
            
            print(f"Found {len(raw_data)} items of data")
            cleaned_data = self._clean_data(raw_data)
            print(f"Cleaned data contains {len(cleaned_data)} items")
            
            # Save in all formats for demonstration
            results = []
            for i in range(1, 4):
                file_path = self._save_dataset(
                    dataset_name,
                    cleaned_data,
                    self._get_file_type(i)
                )
                results.append(file_path)
            
            return f"Datasets created successfully at:\n" + "\n".join(results)
            
        except Exception as e:
            return f"Error in dataset creation: {str(e)}"

    def _scrape_data(self, url: str) -> List[Dict[str, Any]]:
        """
        Enhanced scraping function with better data extraction
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # More comprehensive data extraction
            data = []
            
            # Try different common patterns for data
            # 1. Try tables first
            tables = soup.find_all('table')
            if tables:
                for table in tables:
                    headers = []
                    rows = table.find_all('tr')
                    
                    # Get headers
                    header_row = rows[0]
                    headers = [header.get_text(strip=True) for header in header_row.find_all(['th', 'td'])]
                    
                    # Get data rows
                    for row in rows[1:]:
                        cells = row.find_all(['td', 'th'])
                        row_data = {}
                        for i, cell in enumerate(cells):
                            header = headers[i] if i < len(headers) else f'column_{i}'
                            row_data[header] = cell.get_text(strip=True)
                        if row_data:
                            data.append(row_data)
            
            # 2. Try structured divs if no tables found
            if not data:
                containers = soup.find_all(['div', 'article'], class_=lambda x: x and any(term in str(x).lower() for term in ['container', 'item', 'card', 'product']))
                for container in containers:
                    item_data = {}
                    
                    # Try to find title/header
                    title_elem = container.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    if title_elem:
                        item_data['title'] = title_elem.get_text(strip=True)
                    
                    # Try to find description
                    desc_elem = container.find(['p', 'div'], class_=lambda x: x and 'description' in str(x).lower())
                    if desc_elem:
                        item_data['description'] = desc_elem.get_text(strip=True)
                    
                    # Try to find price
                    price_elem = container.find(class_=lambda x: x and 'price' in str(x).lower())
                    if price_elem:
                        item_data['price'] = price_elem.get_text(strip=True)
                    
                    if item_data:
                        data.append(item_data)
            
            # 3. If still no data, try a more generic approach
            if not data:
                # Get all text content organized by sections
                sections = soup.find_all(['section', 'article', 'div'], class_=True)
                for section in sections:
                    section_data = {}
                    section_class = section.get('class', ['unknown'])[0]
                    section_text = section.get_text(strip=True)
                    if section_text:
                        section_data[section_class] = section_text
                        data.append(section_data)
            
            print(f"Scraped {len(data)} items from {url}")
            return data
            
        except requests.RequestException as e:
            print(f"Error scraping data: {e}")
            return []

    def _clean_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhanced data cleaning function
        """
        cleaned_data = []
        for item in data:
            # Remove empty or None values
            cleaned_item = {}
            for k, v in item.items():
                # Clean the key
                clean_key = k.strip().lower().replace(' ', '_')
                
                # Clean the value
                if isinstance(v, str):
                    clean_value = v.strip()
                    # Remove excessive whitespace
                    clean_value = ' '.join(clean_value.split())
                    # Remove special characters if needed
                    # clean_value = re.sub(r'[^\w\s-]', '', clean_value)
                    
                    if clean_value:  # Only add non-empty values
                        cleaned_item[clean_key] = clean_value
                else:
                    cleaned_item[clean_key] = v
            
            if cleaned_item:  # Only add items that have data
                cleaned_data.append(cleaned_item)
        
        return cleaned_data

    def _save_dataset(self, name: str, data: List[Dict[str, Any]], file_type: str) -> str:
        """
        Enhanced save function with better error handling and data validation
        """
        if not data:
            raise ValueError("No data to save")

        os.makedirs('datasets', exist_ok=True)
        file_path = f'datasets/{name}.{file_type}'
        
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            
            # Clean column names
            df.columns = df.columns.str.replace('[^\w\s]', '_')  # Replace special characters with underscore
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
            df.columns = df.columns.str.lower()  # Convert to lowercase
            df.columns = [f'column_{i}' if not col else col for i, col in enumerate(df.columns)]  # Replace empty column names
            
            if file_type == 'csv':
                df.to_csv(file_path, index=False, encoding='utf-8')
                print(f"Saved CSV file with {len(df)} rows")
            
            elif file_type == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"Saved JSON file with {len(data)} records")
            
            elif file_type == 'sqlite':
                conn = sqlite3.connect(file_path)
                
                # Ensure column names are SQLite-compatible
                df.columns = [f'col_{i}' if not col or col.startswith('_') else col 
                            for i, col in enumerate(df.columns)]
                
                # Replace any remaining invalid characters
                df.columns = df.columns.str.replace('[^\w]', '_')
                
                # Ensure no duplicate column names
                seen_columns = set()
                new_columns = []
                for col in df.columns:
                    if col in seen_columns:
                        i = 1
                        while f"{col}_{i}" in seen_columns:
                            i += 1
                        col = f"{col}_{i}"
                    seen_columns.add(col)
                    new_columns.append(col)
                df.columns = new_columns
                
                df.to_sql('dataset_table', conn, if_exists='replace', index=False)
                conn.close()
                print(f"Saved SQLite database with {len(df)} records")
            
            return file_path
        
        except Exception as e:
            print(f"Error saving dataset: {e}")
            raise

    def _get_file_type(self, choice: int) -> str:
        """
        Convert numeric choice to file extension
        """
        file_types = {1: 'csv', 2: 'json', 3: 'sqlite'}
        return file_types.get(choice, 'csv')

def main():
    # Load environment variables
    load_dotenv()
    
    # Get API keys from environment variables
    BRAVE_API_KEY = os.getenv('BRAVE_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    if not BRAVE_API_KEY or not OPENAI_API_KEY:
        raise ValueError("Missing required API keys in environment variables")
    
    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0, 
        model_name="gpt-3.5-turbo", 
        openai_api_key=OPENAI_API_KEY
    )
    
    # Initialize tool
    dataset_tool = DatasetCreationTool(
        brave_api_key=BRAVE_API_KEY,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Create the prompt template
    prompt = PromptTemplate.from_template(
        "Answer the following questions as best you can. You have access to the following tools:\n\n{tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nQuestion: {input}\nThought:{agent_scratchpad}"
    )

    # Initialize the agent
    agent = create_react_agent(llm, [dataset_tool], prompt)
    
    # Create an agent executor
    agent_executor = AgentExecutor(agent=agent, tools=[dataset_tool], verbose=True)
    
    # Run the dataset creation workflow
    print("Welcome to Dataset Creation Workflow")
    result = agent_executor.invoke({"input": "Create a new dataset"})
    print(result)

if __name__ == '__main__':
    main()
