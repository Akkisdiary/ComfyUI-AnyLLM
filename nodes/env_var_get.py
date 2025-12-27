import os
from typing import Tuple


class GetEnvVar:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": (
                    "STRING",
                    {"default": "GOOGLE_API_KEY", "multiline": False}
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = "AnyLLM/Utils"

    def run(self, name: str) -> Tuple[str]:
        key = (name or "").strip()
        if not key:
            raise ValueError("Env var name is empty")

        val = os.getenv(key) or ""
        if not val.strip():
            raise RuntimeError(f"Env var '{key}' is not set or is empty")

        return (str(val),)


NODE_CLASS_MAPPINGS = {
    "GetEnvVar": GetEnvVar,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetEnvVar": "Get Environment Variable (AnyLLM)",
}
