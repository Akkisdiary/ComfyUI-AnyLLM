import logging
import base64
import io
from typing import Any, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image


logger = logging.getLogger("ComfyUI-AnyLLM")


def _log(msg: str) -> None:
    try:
        logger.info(msg)
    except Exception:
        print(f"[ComfyUI-AnyLLM] {msg}")


def _split_batch_images(images: Any) -> List[torch.Tensor]:
    if images is None:
        return []
    if isinstance(images, torch.Tensor) and images.dim() == 4:
        return [images[i] for i in range(images.shape[0])]
    if isinstance(images, torch.Tensor) and images.dim() == 3:
        return [images]
    raise ValueError("Invalid IMAGE input")


def _image_tensor_to_png_bytes(image_tensor: torch.Tensor) -> bytes:
    image_np = image_tensor.detach().cpu().numpy()
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError(f"Expected image tensor [H,W,3], got {image_np.shape}")

    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (image_np * 255.0).astype(np.uint8)
    pil = Image.fromarray(image_np, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _png_bytes_to_image_tensor(data: bytes) -> torch.Tensor:
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _stack_image_tensors(images: List[torch.Tensor]) -> torch.Tensor:
    if not images:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    h0, w0 = int(images[0].shape[0]), int(images[0].shape[1])
    resized: List[torch.Tensor] = []
    for t in images:
        if int(t.shape[0]) == h0 and int(t.shape[1]) == w0:
            resized.append(t)
            continue
        pil = Image.fromarray((np.clip(t.detach().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
        pil = pil.resize((w0, h0), resample=Image.BILINEAR)
        resized.append(torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0))
    return torch.stack(resized, dim=0)


def _iter_response_parts(resp: Any) -> Iterable[Any]:
    parts = getattr(resp, "parts", None)
    if parts:
        for p in parts:
            yield p

    candidates = getattr(resp, "candidates", None)
    if not candidates:
        return
    for c in candidates:
        content = getattr(c, "content", None)
        cparts = getattr(content, "parts", None) if content else None
        if not cparts:
            continue
        for p in cparts:
            yield p


def _google_gemini_models():
    return [
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-image",
        "gemini-3-pro-image-preview",
    ]


class GoogleGemini:
    @classmethod
    def INPUT_TYPES(cls):
        models = _google_gemini_models()
        return {
            "required": {
                "model": (models, {"default": models[0]}),
            },
            "optional": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "response_images": ("BOOLEAN", {"default": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "images")
    FUNCTION = "run"
    CATEGORY = "AnyLLM/Google"

    def run(
        self,
        model: str,
        images=None,
        api_key: str = "",
        prompt: str = "",
        response_images: bool = False,
    ) -> Tuple[str, torch.Tensor]:
        _log(
            f"GoogleGemini: start | model={model} | response_images={bool(response_images)} | "
            f"prompt_len={len((prompt or '').strip())}"
        )
        prompt = (prompt or "").strip() or "Hello"

        if not api_key:
            raise ValueError("Google API key is required")

        try:
            from google import genai
            from google.genai import types
        except Exception as e:
            raise RuntimeError(
                "google-genai is not installed. "
                "Install this custom node's requirements.txt. "
                f"Details: {type(e).__name__}: {e}"
            )

        try:
            client = genai.Client(api_key=api_key)
            batch_images = _split_batch_images(images)
            _log(f"GoogleGemini: preparing request | input_images={len(batch_images)}")

            in_parts: List[Any] = [types.Part.from_text(text=prompt)]
            for t in batch_images:
                in_parts.append(
                    types.Part.from_bytes(
                        data=_image_tensor_to_png_bytes(t),
                        mime_type="image/png",
                    )
                )

            config = None
            if response_images:
                config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])

            resp = client.models.generate_content(
                model=model,
                contents=in_parts,
                config=config,
            )

            _log("GoogleGemini: response received")

            out_text = getattr(resp, "text", None)
            if not out_text:
                for part in _iter_response_parts(resp):
                    ptext = getattr(part, "text", None)
                    if ptext:
                        out_text = str(ptext)
                        break
            if not out_text:
                out_text = str(resp)

            _log(f"GoogleGemini: parsed text | text_len={len(out_text)}")

            out_images: List[torch.Tensor] = []
            for part in _iter_response_parts(resp):
                inline_data = getattr(part, "inline_data", None)
                if not inline_data:
                    continue

                mime = getattr(inline_data, "mime_type", "")
                data = getattr(inline_data, "data", None)

                if data is None and isinstance(inline_data, dict):
                    mime = inline_data.get("mime_type", mime)
                    data = inline_data.get("data")

                if data is None:
                    continue
                if isinstance(data, str):
                    data = base64.b64decode(data)

                if mime.startswith("image/"):
                    out_images.append(_png_bytes_to_image_tensor(data))

            if out_images:
                return (out_text, _stack_image_tensors(out_images))

            passthrough = images if isinstance(images, torch.Tensor) else _stack_image_tensors([])
            return (out_text, passthrough)
        except Exception as e:
            _log(f"GoogleGemini: error | {type(e).__name__}: {e}")
            raise RuntimeError(
                f"Error calling Google model: {type(e).__name__}: {e}"
            )


NODE_CLASS_MAPPINGS = {
    "GoogleGemini": GoogleGemini,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleGemini": "Google Gemini (AnyLLM)",
}
