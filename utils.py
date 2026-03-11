"""
utils.py
========
Utility/helper functions used across the project.
Keeps the main files clean and readable.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def log_step(step_number: int, description: str):
    """
    Print a formatted step header — makes console output easy to follow.
    Example output:
       ──────────────────────────────────
       📌 STEP 3 | Creating embeddings model
       ──────────────────────────────────
    """
    print(f"\n{'─'*45}")
    print(f"   📌 STEP {step_number} | {description}")
    print(f"{'─'*45}")


def get_env_variable(key: str) -> str:
    """
    Safely retrieve an environment variable.
    Raises a clear error message if it's missing — avoids confusing crashes.
    """
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"\n❌ Missing environment variable: '{key}'\n"
            f"   → Please add it to your .env file:\n"
            f"     {key}=your_token_here\n"
        )
    return value


def format_sources(source_documents: list) -> str:
    """
    Format the source document chunks used to generate the answer.
    Displayed in the Streamlit UI so users can see where the answer came from.
    """
    if not source_documents:
        return "No sources found."

    sources_text = "📄 **Sources Used:**\n"
    for i, doc in enumerate(source_documents, start=1):
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content[:200].replace("\n", " ").strip()
        sources_text += f"\n**[{i}] Page {page}:** _{snippet}..._\n"

    return sources_text


def validate_pdf_path(pdf_path: str) -> bool:
    """
    Check that the given PDF path exists and is a valid .pdf file.
    Returns True if valid, raises FileNotFoundError otherwise.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF not found: '{pdf_path}'")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"❌ File is not a PDF: '{pdf_path}'")
    return True
