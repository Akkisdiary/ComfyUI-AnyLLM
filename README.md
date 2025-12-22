# ComfyUI-AnyLLM (LangChain)

A ComfyUI node pack that exposes dedicated provider nodes implemented with LangChain.

## Install

### Install via ComfyUI Manager

1. Open **ComfyUI Manager**
2. Install this repository as a custom node

If the repository is not listed yet, you can still install it by URL using Manager's "Install via Git URL" feature.

### Manual install (git)

Clone into your ComfyUI `custom_nodes` folder:

```bash
git clone <YOUR_GIT_URL> ComfyUI/custom_nodes/ComfyUI-AnyLLM
```

Then install Python dependencies in the same Python environment ComfyUI uses:

```bash
pip install -r ComfyUI/custom_nodes/ComfyUI-AnyLLM/requirements.txt
```

Restart ComfyUI.

## Node

- **Google Chat (LangChain)**

Inputs:
- `model` (dropdown)
- `prompt` (STRING, optional)
- `api_key` (STRING, optional; defaults to env `GOOGLE_API_KEY`)

Output:
- `text` (STRING)

## Notes

- Provider/model capabilities differ; not every model supports images.
- API keys are passed only at runtime (not stored).

## ComfyUI Manager listing

To be discoverable in the default ComfyUI Manager registry, submit a PR to the registry list (ComfyUI-Manager maintains a public JSON list of repositories).
