import os
import math
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, YoutubeLoader
import requests
import numexpr
import openai
import pandas as pd
import subprocess
import re
import sys 
from PIL import Image
import pytesseract
import base64
from pytubefix import YouTube
import io 

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def _ensure_python_deps(py_file: str):
    """
    Scan a Python file for imports, and pip‑install any that aren’t already importable.
    """
    missing = set()
    pattern = re.compile(r'^\s*(?:import\s+([a-zA-Z0-9_]+)|from\s+([a-zA-Z0-9_]+)\s+import)', re.MULTILINE)

    with open(py_file, 'r', encoding='utf-8') as f:
        code = f.read()

    for imp, frm in pattern.findall(code):
        pkg = imp or frm
        try:
            __import__(pkg)
        except ImportError:
            missing.add(pkg)

    for pkg in missing:
        print(f"[auto-install] {pkg} not found, installing…")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

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
    Runs a code file based on its extension.
    Supported extensions:
      - .py   : Python
      - .cpp  : C++
      - .java : Java
      - .sql  : SQL (runs via sqlite3 against an in-memory DB)
      - .cs   : C# (compiles via csc)
    Args:
        file_path: Path to the code file.
    Returns:
        The captured stdout, or compilation/runtime error text.
    """
    if not os.path.exists(file_path):
        return "Error: File does not exist."

    ext = os.path.splitext(file_path)[1].lower()
    try:
        # Python
        if ext == ".py":
            _ensure_python_deps(file_path)
            proc = subprocess.run(
                [sys.executable, file_path],
                capture_output=True, text=True
            )
            return proc.stdout or proc.stderr

        # C++
        elif ext == ".cpp":
            exe = file_path[:-4]
            compile_proc = subprocess.run(
                ["g++", file_path, "-o", exe],
                capture_output=True, text=True
            )
            if compile_proc.returncode:
                return f"Compilation Error:\n{compile_proc.stderr}"
            run_proc = subprocess.run([exe], capture_output=True, text=True)
            return run_proc.stdout or run_proc.stderr

        # Java
        elif ext == ".java":
            dir_ = os.path.dirname(file_path) or "."
            compile_proc = subprocess.run(
                ["javac", file_path],
                capture_output=True, text=True
            )
            if compile_proc.returncode:
                return f"Compilation Error:\n{compile_proc.stderr}"
            cls = os.path.splitext(os.path.basename(file_path))[0]
            run_proc = subprocess.run(
                ["java", "-cp", dir_, cls],
                capture_output=True, text=True
            )
            return run_proc.stdout or run_proc.stderr

        # SQL (SQLite)
        elif ext == ".sql":
            # Requires sqlite3 CLI installed.
            # By default runs against an in-memory DB.
            # To run against a file-based DB, replace ":memory:" with a path.
            cmd = ["sqlite3", ":memory:", "-batch", f".read {file_path}"]
            run_proc = subprocess.run(
                " ".join(cmd),
                shell=True, capture_output=True, text=True
            )
            return run_proc.stdout or run_proc.stderr

        # C#
        elif ext == ".cs":
            exe = file_path[:-3] + (".exe" if os.name == "nt" else "")
            # Compile with csc (Windows) or mcs (Mono)
            compiler = "csc" if os.name == "nt" else "mcs"
            compile_proc = subprocess.run(
                [compiler, "-out:" + exe, file_path],
                capture_output=True, text=True
            )
            if compile_proc.returncode:
                return f"Compilation Error:\n{compile_proc.stderr}"
            run_proc = subprocess.run(
                [exe],
                capture_output=True, text=True
            )
            return run_proc.stdout or run_proc.stderr

        else:
            return (
                "Error: Unsupported file extension.\n"
                "Supported: .py, .cpp, .java, .sql, .cs"
            )

    except Exception as e:
        return f"Execution Error:\n{e}"


@tool
def download_video_from_url(url: str = None, save_dir: str = "downloads") -> str:
    """
    Download a video file from a given link provided in the human message. If the user question DIRECTLY includes a link to a video file (e.g. a YouTube or HTTP URL), pass it to the 'url' argument.
    
    Args:
    - url: The link or url to the file or video (e.g., YouTube) indicated DIRECTLY in the user question.
    - save_dir: Directory where the file will be saved.

    Returns:
    - The local path to the downloaded file.
    """
    os.makedirs(save_dir, exist_ok=True)

    if url and "youtube.com" in url.lower():
        try:
            filename = url 
            local_path = os.path.join(save_dir, filename)
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution()
            video_path = stream.download(output_path=local_path)
            print(f"Video downloaded to: {video_path}")
            return video_path
        except Exception as e:
            raise ValueError(f"Failed to download YouTube video: {str(e)}")
    else:
        raise ValueError("Either a valid URL or task_id must be provided.")

@tool
def download_file_from_task_id(task_id: str,  save_dir: str = "downloads") -> str:
    """
    Download a file from the given task_id. if the user query refers to an attached file or document, pass the question task_id to the 'task_id' argument.
    
    Args:
    - task_id: Unique identifier used to generate the URL if none is provided.
    - save_dir: Directory where the file will be saved.

    Returns:
    - The local path to the downloaded file.
    """

    if not task_id:
        raise ValueError("Either a valid URL or task_id must be provided.")
    url = f"https://agents-course-unit4-scoring.hf.space/files/{task_id}"

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise ValueError(f"Failed to download file from {url}. HTTP {response.status_code}")

    # Extract filename from headers if possible
    os.makedirs(save_dir, exist_ok=True)

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
    Returns:
        The result of the analysis.
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
    """Looks up a topic on Wikipedia and returns a summary and additional content for reasoning.
    Args:
        query: input query string
    Returns:
        A formatted string of search results.    
    """
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted = "\n\n---\n\n".join([
        f'<Document source="{doc.metadata.get("url", "")}">\n{doc.page_content[:doc_content_chars_max]}\n</Document>'
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
        The transcript of the video.
    """
    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=language)
        transcript = loader.load()
        formatted = "\n\n---\n\n".join([
        f'<Document source="{doc.get("url", "")}">\n{doc.get("content", "")}\n</Document>'
        for doc in transcript
        ])
        return {"results": formatted or "No relevant results found."}        
    except Exception as e:
        return f"Error fetching YouTube transcript: {str(e)}"


@tool
def preprocess_image(image_path):
    """
    Preprocesses an image by resizing and encoding it in base64 format.

    Args:
        image_path (str): The file path to the input image.

    Returns:
        A base64-encoded string representation of the resized PNG image.
    """
    with Image.open(image_path) as img:
        img.thumbnail((128, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
@tool
def processs_image_text(image_path: str) -> str:
    """
    Given the local path to an image file, performs OCR to extract text information from an image.
    Args:
        image_path (str): The file path to the input image.
    Returns:
        The extracted text from the image.
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        return f"Error: file not found at {image_path}"
    except Exception as e:
        return f"Error opening image: {e}"

    try:
        text = pytesseract.image_to_string(img)
    except Exception as e:
        return f"OCR failed: {e}"
    return text


# List of all tools
TOOLS = [web_search, download_video_from_url, download_file_from_task_id, calculator, 
        analyze_table_file, run_code_file, transcribe_file,
        wikipedia_tool, youtube_transcript, preprocess_image, processs_image_text
        ]
