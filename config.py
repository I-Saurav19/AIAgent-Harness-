"""Environment configuration for the agent."""

import os
from dotenv import load_dotenv

load_dotenv()


def get_api_key(provider):
    variable = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "xai": "XAI_API_KEY",
    }[provider]
    value = os.getenv(variable)
    if not value:
        raise ValueError(f"Set {variable} in your environment or .env file.")
    return value
