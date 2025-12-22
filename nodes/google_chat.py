import os
from typing import Tuple


def _google_chat_models():
    return [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash-exp",
    ]


class GoogleChatLLM:
    @classmethod
    def INPUT_TYPES(cls):
        models = _google_chat_models()
        return {
            "required": {
                "model": (models, {"default": models[0]}),
            },
            "optional": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "AnyLLM/Google"

    def run(self, model: str, prompt: str = "", api_key: str = "") -> Tuple[str]:
        effective_key = (api_key or "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
        if not effective_key:
            return (
                "Error: Missing Google API key. Provide `api_key` or set environment variable GOOGLE_API_KEY.",
            )

        user_prompt = (prompt or "").strip() or "Hello"

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as e:
            return (
                "Error: langchain-google-genai is not installed. Install this custom node's requirements.txt. "
                f"Details: {type(e).__name__}: {e}",
            )

        try:
            llm = ChatGoogleGenerativeAI(model=model, google_api_key=effective_key)
            result = llm.invoke(user_prompt)
            content = getattr(result, "content", result)
            return (str(content),)
        except Exception as e:
            return (f"Error calling Google model: {type(e).__name__}: {e}",)


NODE_CLASS_MAPPINGS = {
    "GoogleChatLLM": GoogleChatLLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleChatLLM": "Google Chat (LangChain)",
}
