import os

from dotenv import load_dotenv

load_dotenv()


APP_TITLE = "CareGuide RAG API"
APP_DESCRIPTION = (
    "A healthcare document question-answering application "
    "using TF-IDF retrieval, LangGraph and Claude"
)
APP_VERSION = "1.0.0"

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

MIN_RELEVANCE_SCORE = 0.05

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5"
CLAUDE_MAX_TOKENS = 500

FALLBACK_ANSWER = "I could not find relevant information in the uploaded documents."

MEDICAL_DISCLAIMER = (
    "This application provides document-based information only "
    "and is not a substitute for professional medical advice."
)

HEALTHCARE_SYSTEM_PROMPT = (
    "You are CareGuide, a healthcare document assistant. "
    "Answer only using the supplied document context. "
    "Do not provide a diagnosis or invent medical facts. "
    "Cite relevant sources using their exact square-bracket "
    "labels. If the context does not contain the answer, say "
    "'I could not find that information in the uploaded documents.'"
)


def get_anthropic_api_key() -> str | None:
    """Read the Anthropic API key from the environment at call time."""

    return os.getenv("ANTHROPIC_API_KEY")


def get_claude_model() -> str:
    """Read the configured Claude model from the environment at call time."""

    return os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
