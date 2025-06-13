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
from .core.config import settings

logger = logging.getLogger(__name__)

def create_agno_agent(model: str, temp_dir: str) -> Agent:
    """Creates and configures the Agno agent, accepting the temp_dir path."""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in the environment.")

    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

    instructions = [
        "You are an advanced data processing AI.",
        "Analyze JSON data and create an optimal Excel layout.",
        "Execute your Python code immediately after writing it.",
        "Save files in the current working directory.",
        "Handle complex nested data intelligently.",
        "Create multiple sheets for distinct data categories.",
    ]

    agent = Agent(
        model=Gemini(id=model, api_key=settings.GOOGLE_API_KEY),
        # Pass the temp_dir to the agent's tools
        tools=[PythonTools(run_code=True, pip_install=True, base_dir=Path(temp_dir))],
        show_tool_calls=True,
        instructions=instructions
    )
    logger.info(f"Created Agno agent with model: {model}")
    return agent

def direct_json_to_excel(json_data: str, file_name: str, chunk_size: int, temp_dir: str) -> tuple:
    """
    Directly converts JSON data to an Excel file.
    This version robustly handles single JSON objects, arrays, and streams of objects.
    """
    try:
        # --- FIX for JSONDecodeError ---
        # Handle multiple concatenated or newline-delimited JSON objects.
        decoder = json.JSONDecoder()
        pos = 0
        data_objects = []
        clean_json_data = json_data.strip()
        while pos < len(clean_json_data):
            obj, end_pos = decoder.raw_decode(clean_json_data[pos:])
            data_objects.append(obj)
            pos = end_pos
            # Move past any whitespace before the next object
            while pos < len(clean_json_data) and clean_json_data[pos].isspace():
                pos += 1
        
        # If there was only one object and it wasn't a list, use it directly.
        # Otherwise, process the list of all objects found.
        data = data_objects[0] if len(data_objects) == 1 else data_objects
        # --- End of FIX ---

        file_id = str(uuid.uuid4())
        safe_filename = "".join(c for c in file_name if c.isalnum() or c in (' ', '-', '_')).strip()
        xlsx_filename = f"{safe_filename}_direct.xlsx"
        file_path = os.path.join(temp_dir, f"{file_id}_{xlsx_filename}")

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            if isinstance(data, list):
                if len(data) > chunk_size:
                    logger.info(f"Large dataset: processing {len(data)} records in chunks of {chunk_size}.")
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
    """
    Uses the Agno agent to convert JSON, accepting the temp_dir path.
    """
    try:
        # Pass temp_dir to the agent creator
        agent = create_agno_agent(model, temp_dir)
        prompt = f"""
        Convert the following JSON data to a well-structured Excel workbook.
        Base filename: {file_name}
        Description: {description}

        JSON Data:
        {json_data}

        Instructions:
        1. Analyze the JSON structure.
        2. Design an optimal, professional Excel layout.
        3. Save the result as: {file_name}_ai.xlsx
        """
        response = agent.run(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Agno AI processing failed: {e}")
        raise

def find_newest_file(directory: str, files_before: set) -> str | None:
    """Finds the newest .xlsx file in a directory that is not in the 'files_before' set."""
    files_after = set(glob.glob(os.path.join(directory, "*.xlsx")))
    new_files = files_after - files_before
    if not new_files:
        return None
    return max(new_files, key=os.path.getmtime)