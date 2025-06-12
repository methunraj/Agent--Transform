# app/services.py
import os
import json
import uuid
import time
import glob
import logging
import pandas as pd
from pathlib import Path
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.python import PythonTools
from agno.tools.googlesearch import GoogleSearchTools
from .core.config import settings

logger = logging.getLogger(__name__)

def create_agno_agent(model: str, temp_dir: str) -> Agent:
    """Creates and configures the Agno agent with robust, clear instructions."""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in the environment.")

    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

    # --- FINAL, SIMPLIFIED AND ROBUST INSTRUCTIONS ---
    instructions = [
        "You are an expert financial analyst. Your goal is to create a professional, multi-sheet Excel report from JSON data.",
        
        "**Core Task: Generate and Execute a Python Script**",
        "1.  Your primary output is a single Python script that performs the entire data transformation and Excel generation process.",
        "2.  **Crucially, when you save this script using the `save_to_file_and_run` tool, you MUST provide a `file_name` argument. Name the script 'excel_report_generator.py'.** This is a mandatory step.",

        "**Script Requirements:**",
        "1.  **Currency Conversion:** The script must detect the original currency, use the `Google Search` tool to find the current USD exchange rate, and then create two columns for every financial figure: one for the original currency and an adjacent one for the converted USD value. Add a note in the summary sheet citing the rate used.",
        "2.  **Data Integrity:** The script must not invent or alter historical data. Forecasts are allowed but must be clearly labeled and based on the source data.",
        "3.  **Excel Structure:** The script should create an Excel file with multiple sheets:",
        "    - A 'Summary' sheet with a narrative, key insights, and an embedded forecast chart (in USD).",
        "    - Separate sheets for 'Income Statement', 'Balance Sheet', and 'Cash Flow'.",
        "    - Additional sheets for any data breakdowns found, like 'Business Segments' or 'Geographic Revenue'.",
        "4.  **No Raw JSON:** The final Excel file must NOT contain any raw JSON data.",

        "Review your generated Python code for correctness before calling the tool to save and run it.",
    ]
    # --- END OF FINAL INSTRUCTIONS ---

    agent = Agent(
        model=Gemini(id=model, api_key=settings.GOOGLE_API_KEY),
        tools=[
            PythonTools(run_code=True, pip_install=True, base_dir=Path(temp_dir)),
            GoogleSearchTools()
        ],
        show_tool_calls=True,
        instructions=instructions
    )
    logger.info(f"Created Agno agent with model: {model} and final, robust instructions.")
    return agent

# The rest of the file remains the same...
def direct_json_to_excel(json_data: str, file_name: str, chunk_size: int, temp_dir: str) -> tuple:
    try:
        decoder = json.JSONDecoder()
        pos = 0
        data_objects = []
        clean_json_data = json_data.strip()
        while pos < len(clean_json_data):
            obj, end_pos = decoder.raw_decode(clean_json_data[pos:])
            data_objects.append(obj)
            pos = end_pos
            while pos < len(clean_json_data) and clean_json_data[pos].isspace():
                pos += 1
        
        data = data_objects[0] if len(data_objects) == 1 else data_objects

        file_id = str(uuid.uuid4())
        safe_filename = "".join(c for c in file_name if c.isalnum() or c in (' ', '-', '_')).strip()
        xlsx_filename = f"{safe_filename}_direct.xlsx"
        file_path = os.path.join(temp_dir, f"{file_id}_{xlsx_filename}")

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            if isinstance(data, list):
                if len(data) > chunk_size:
                    for i in range(0, len(data), chunk_size):
                        df = pd.json_normalize(data[i:i + chunk_size])
                        df.to_excel(writer, sheet_name=f'Data_Chunk_{i//chunk_size + 1}', index=False)
                else:
                    pd.json_normalize(data).to_excel(writer, sheet_name='Data', index=False)
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        pd.json_normalize(value).to_excel(writer, sheet_name=str(key)[:31], index=False)
            else:
                 pd.DataFrame([{'value': data}]).to_excel(writer, sheet_name='Data', index=False)

        return file_id, xlsx_filename, file_path
    except Exception as e:
        logger.error(f"Direct conversion failed: {e}")
        raise

def convert_with_agno(json_data: str, file_name: str, description: str, model: str, temp_dir: str) -> str:
    try:
        agent = create_agno_agent(model, temp_dir)
        prompt = f"""
        Please process the following JSON data from an annual report according to your specialized instructions.
        The final Excel file should be named based on this base name: {file_name}
        Description of the report: {description}

        JSON Data:
        {json_data}
        """
        response = agent.run(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Agno AI processing failed: {e}")
        raise

def find_newest_file(directory: str, files_before: set) -> str | None:
    files_after = set(glob.glob(os.path.join(directory, "*.xlsx")))
    new_files = files_after - files_before
    if not new_files:
        return None
    return max(new_files, key=os.path.getmtime)