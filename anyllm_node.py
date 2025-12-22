import base64
import io
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


def _tensor_to_png_base64(image_tensor) -> str:
    if image_tensor is None:
        raise ValueError("image_tensor is None")

    image_np = image_tensor.detach().cpu().numpy()
    if image_np.ndim != 3:
        raise ValueError(f"Expected image tensor with shape [H,W,C], got {image_np.shape}")

    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(image_np, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_first_batch_image(images) -> List[Any]:
    if images is None:
        return []
    if hasattr(images, "shape") and len(images.shape) == 4:
        batch = images.shape[0]
        return [images[i] for i in range(batch)]
    return [images]


def _get_provider_model_choices(provider: str) -> List[str]:
    if provider == "OpenAI":
        return [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1",
        ]
    if provider == "Anthropic":
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]
    if provider == "Google (Gemini)":
        return [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash-exp",
        ]
    if provider == "OpenAI-Compatible":
        return [
            "gpt-4o-mini",
            "gpt-4o",
        ]
    return ["gpt-4o-mini"]


def _get_all_model_choices() -> List[str]:
    models = set()
    for p in ["OpenAI", "Anthropic", "Google (Gemini)", "OpenAI-Compatible"]:
        for m in _get_provider_model_choices(p):
            models.add(m)
    return sorted(models)


class AnyLLMVisionChat:
    @classmethod
    def INPUT_TYPES(cls):
        providers = ["OpenAI", "Anthropic", "Google (Gemini)", "OpenAI-Compatible"]
        default_provider = providers[0]
        models = _get_all_model_choices()

        return {
            "required": {
                "images": ("IMAGE",),
                "provider": (providers, {"default": default_provider}),
                "model": (models, {"default": models[0]}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "base_url": ("STRING", {"default": "", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 8192}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "AnyLLM"

    def run(
        self,
        images,
        provider: str,
        model: str,
        api_key: str,
        system_prompt: str,
        prompt: str,
        base_url: str = "",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> Tuple[str]:
        if not api_key:
            return ("Error: API key is required.",)

        allowed_models = set(_get_provider_model_choices(provider))
        if model not in allowed_models:
            return (
                f"Error: Model '{model}' is not in the supported list for provider '{provider}'. "
                f"Allowed: {sorted(allowed_models)}",
            )

        image_tensors = _extract_first_batch_image(images)
        image_b64_list = []
        try:
            for it in image_tensors:
                image_b64_list.append(_tensor_to_png_base64(it))
        except Exception as e:
            return (f"Error converting input images: {type(e).__name__}: {e}",)

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except Exception as e:
            return (
                "Error: langchain is not installed in this Python environment. "
                "Install this custom node's requirements.txt. "
                f"Details: {type(e).__name__}: {e}",
            )

        try:
            llm = None
            messages = []
            if system_prompt and system_prompt.strip():
                messages.append(SystemMessage(content=system_prompt))

            if provider in ("OpenAI", "OpenAI-Compatible"):
                from langchain_openai import ChatOpenAI

                kwargs: Dict[str, Any] = {
                    "model": model,
                    "api_key": api_key,
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                }
                if provider == "OpenAI-Compatible":
                    if base_url and base_url.strip():
                        kwargs["base_url"] = base_url.strip()
                    else:
                        return ("Error: base_url is required for OpenAI-Compatible provider.",)

                llm = ChatOpenAI(**kwargs)

                parts: List[Dict[str, Any]] = []
                if prompt and prompt.strip():
                    parts.append({"type": "text", "text": prompt})
                for b64 in image_b64_list:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    )
                messages.append(HumanMessage(content=parts))

            elif provider == "Anthropic":
                from langchain_anthropic import ChatAnthropic

                llm = ChatAnthropic(
                    model=model,
                    anthropic_api_key=api_key,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )

                parts = []
                if prompt and prompt.strip():
                    parts.append({"type": "text", "text": prompt})
                for b64 in image_b64_list:
                    parts.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": b64},
                        }
                    )
                messages.append(HumanMessage(content=parts))

            elif provider == "Google (Gemini)":
                from langchain_google_genai import ChatGoogleGenerativeAI

                llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=api_key,
                    temperature=float(temperature),
                    max_output_tokens=int(max_tokens),
                )

                parts = []
                if prompt and prompt.strip():
                    parts.append({"type": "text", "text": prompt})
                for b64 in image_b64_list:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{b64}",
                        }
                    )
                messages.append(HumanMessage(content=parts))

            else:
                return (f"Error: Unknown provider '{provider}'.",)

            result = llm.invoke(messages)
            content = getattr(result, "content", result)
            if isinstance(content, list):
                content = "\n".join([str(x) for x in content])
            return (str(content),)

        except Exception as e:
            return (f"Error calling LLM: {type(e).__name__}: {e}",)


NODE_CLASS_MAPPINGS = {
    "AnyLLMVisionChat": AnyLLMVisionChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyLLMVisionChat": "Any LLM (LangChain) - Vision Chat",
}
