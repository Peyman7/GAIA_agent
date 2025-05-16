import os
import math
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import YoutubeLoader
from typing import List, Dict, Any, Optional
import tempfile
import requests
from urllib.parse import urlparse
import numexpr
import openai
import pandas as pd
import subprocess
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

        return {"results": formatted or "No relevant results found."}
    
    except Exception as e:
        return {"results": f"[Error during web search: {str(e)}]"}
    

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
def run_code_file(file_path: str) -> str:
    """
    Runs a code file based on its extension (.py, .cpp, .java).
    Returns the standard output or error from execution.
    """
    if not os.path.exists(file_path):
        return "Error: File does not exist."

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == ".py":
            result = subprocess.run(["python", file_path], capture_output=True, text=True, check=True)
            return result.stdout

        elif file_ext == ".cpp":
            exe_path = file_path.replace(".cpp", "")
            compile_result = subprocess.run(["g++", file_path, "-o", exe_path], capture_output=True, text=True)
            if compile_result.returncode != 0:
                return f"Compilation Error:\n{compile_result.stderr}"
            run_result = subprocess.run([exe_path], capture_output=True, text=True)
            return run_result.stdout

        elif file_ext == ".java":
            class_dir = os.path.dirname(file_path)
            compile_result = subprocess.run(["javac", file_path], capture_output=True, text=True)
            if compile_result.returncode != 0:
                return f"Compilation Error:\n{compile_result.stderr}"
            class_name = os.path.splitext(os.path.basename(file_path))[0]
            run_result = subprocess.run(["java", "-cp", class_dir, class_name], capture_output=True, text=True)
            return run_result.stdout

        else:
            return "Error: Unsupported file extension. Supported: .py, .cpp, .java"

    except Exception as e:
        return f"Execution Error:\n{str(e)}"

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
def download_file_from_url(task_id: str, url: str = None, save_dir: str = "downloads") -> str:
    """
    Downloads a file from a given URL (if provided) or builds a URL using task_id (if url is not available).
    Saves the file with its original name if available in headers, else uses task_id.
    
    Parameters:
    - task_id: Unique identifier used to generate the URL if none is provided.
    - url: Direct URL to the file. If None, the function builds a URL using task_id.
    - save_dir: Directory where the file will be saved.

    Returns:
    - The local path to the downloaded file.
    """
    if not url:
        if not task_id:
            raise ValueError("Either a valid URL or task_id must be provided.")
        url = f"https://agents-course-unit4-scoring.hf.space/files/{task_id}"

    os.makedirs(save_dir, exist_ok=True)

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise ValueError(f"Failed to download file from {url}. HTTP {response.status_code}")

    # Extract filename from headers if possible
    content_disp = response.headers.get("Content-Disposition", "")
    filename = task_id  # Default filename
    if "filename=" in content_disp:
        filename = content_disp.split("filename=")[-1].strip('"; ')

    local_path = os.path.join(save_dir, filename)

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return local_path

@tool
def transcribe_file(file_path: str) -> str:
    """
    Transcribes audio or video file using OpenAI Whisper API.

    Args:
        file_path: Path to the local audio file.

    Returns:
        Transcribed text or error message.
    """
    try:
        # Check extension to see if it's a video file
        audio_extensions = [".mp3", ".wav", ".m4a"]
        _, ext = os.path.splitext(file_path)

        if ext.lower() not in audio_extensions:
            print("Extracting audio from video...")
            audio_path = "temp_audio.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", file_path, "-ar", "16000", "-ac", "1", audio_path
            ], check=True)
        else:
            audio_path = file_path

        # Transcribe using OpenAI Whisper API
        client = openai.OpenAI()
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
            )

        # Clean up temporary audio if created
        if audio_path != file_path:
            os.remove(audio_path)

        return transcript.text

    except Exception as e:
        return f"Error transcribing file: {str(e)}"

@tool    
def analyze_table_file(file_path: str, query: str) -> str:
    """
    Analyze an CSV and Excel file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the Excel file.
        query (str): Question about the data
    """
    try:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
    
        if ext == ".csv":
            df = pd.read_csv(file_path)
        
        elif ext in [".xls", ".xlsx"]:
        # Read the Excel file
            df = pd.read_excel(file_path)

        result = (
            f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        )
        result += f"Columns: {', '.join(df.columns)}\n\n"

        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"


@tool
def wikipedia_tool(query: str, load_max_docs: int=2, doc_content_chars_max: int = 4000) -> str:
    """Looks up a topic on Wikipedia and returns a summary and additional content for reasoning."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted = "\n\n---\n\n".join([
        f'<Document source="{doc.get("url", "")}">\n{doc.get("content", "")}\n</Document>'
        for doc in search_docs
    ])
    return {"results": formatted or "No relevant results found."}


@tool
def youtube_transcript(url: str, language: str = "en") -> str:
    """
    Fetches the transcript of a YouTube video in the specified language.
    
    Args:
        url (str): The URL of the YouTube video.
        language (str): The language code for the transcript (default is "en").
    
    Returns:
        str: The transcript of the video.
    """
    try:
        loader = YoutubeLoader.from_youtube_url(url=url, add_video_info=False, language=language)
        transcript = loader.load()
        formatted = "\n\n---\n\n".join([
        f'<Document source="{doc.get("url", "")}">\n{doc.get("content", "")}\n</Document>'
        for doc in transcript
        ])
        return {"results": formatted or "No relevant results found."}        
    except Exception as e:
        return f"Error fetching YouTube transcript: {str(e)}"

# List of all tools
TOOLS = [web_search, download_file_from_url, save_and_read_file, calculator, 
        analyze_table_file, run_code_file, transcribe_file,
        wikipedia_tool, youtube_transcript
        ]
