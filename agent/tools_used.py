import os
import math
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List, Dict, Any, Optional
import tempfile
import requests
from urllib.parse import urlparse
import numexpr

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Performs a web search using Tavily and returns formatted documents.

    Args:
        query: The search query string.
        max_results: Number of top results to return.

    Returns:
        A formatted string of search results.
    """
    try:
        search_docs = TavilySearchResults(max_results=max_results).invoke({"query": query})

        formatted = "\n\n---\n\n".join([
            f'<Document source="{doc.get("url", "")}">\n{doc.get("content", "")}\n</Document>'
            for doc in search_docs
        ])

        return formatted if formatted else "[No relevant results found.]"

    except Exception as e:
        return f"[Error during web search: {str(e)}]"
    

@tool
def calculator(expression: str) -> str:
    """Calculate expression using Python's numexpr library.

    Expression should be a single line mathematical expression
    that solves the problem.

    Examples:
        "37593 * 67" for "37593 times 67"
        "37593**(1/5)" for "37593^(1/5)"
    """
    local_dict = {"pi": math.pi, "e": math.e}
    return str(
        numexpr.evaluate(
            expression.strip(),
            global_dict={},  # restrict access to globals
            local_dict=local_dict,  # add common mathematical functions
        )
    )
    

@tool
def code_executor(code: str) -> str:
    """
    Executes simple Python code safely and returns the result of the last variable.

    Args:
        code: A string of Python code (e.g., "a = 2 * 3\\nb = a + 5").

    Returns:
        The result of the last variable assigned, or a message if no output.
    """
    try:
        local_vars = {}
        exec(code, {"__builtins__": {}}, local_vars)

        if not local_vars:
            return "Code executed successfully but no variables were defined."

        # Return the last variable's value
        last_key = list(local_vars)[-1]
        return f"{last_key} = {local_vars[last_key]}"
    except Exception as e:
        return f"Error executing code: {str(e)}"

@tool
def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a file and return the path.
    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w") as f:
        f.write(content)

    return f"File saved to {filepath}. You can read this file to process its contents."

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

@tool
def download_file_from_url(task_id: str, save_dir: str = "downloads") -> str:
    """
    Downloads the file associated with a GAIA task_id.
    Automatically handles proper file naming and extension.
    """
    if not task_id:
        raise ValueError("task_id is required to download the file.")
    print("Task ID found:", task_id)
    url = f"https://agents-course-unit4-scoring.hf.space/files/{task_id}"
    os.makedirs(save_dir, exist_ok=True)

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise ValueError(f"Failed to download file for task_id {task_id}. HTTP {response.status_code}")

    # Try to extract filename from response headers
    content_disp = response.headers.get("Content-Disposition", "")
    filename = task_id  # default fallback
    if "filename=" in content_disp:
        filename = content_disp.split("filename=")[-1].strip('"; ')

    local_path = os.path.join(save_dir, filename)

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_path




TOOLS = [web_search, download_file_from_url, save_and_read_file, calculator, code_executor]
