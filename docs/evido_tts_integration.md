# Evido Integration Notes for Qwen3-TTS

## 1) Current capabilities and gaps in this repository

- **Model + Python API only:** The repository exposes Qwen3-TTS as a Python package with `Qwen3TTSModel.from_pretrained(...)` and generation APIs for custom voice, voice design, and voice clone. This is the core inference surface you can embed in a service, but it is not an HTTP API by itself. The key entrypoints are `generate_custom_voice(...)`, `generate_voice_design(...)`, and `generate_voice_clone(...)`.
- **Interactive demo, not a production API:** The only server-like entrypoint in the repo is the Gradio demo launcher (`qwen-tts-demo`), which spins up a UI and is meant for manual usage. It is not an API with a stable OpenAI-like schema or authentication hooks for production integration.
- **Environment and model prerequisites:** The README documents the Python package usage, required dependencies, and model identifiers. This is the baseline for any containerized deployment and must be replicated in Docker builds (Python, torch, model weights, optional FlashAttention).

**Implication:** To connect Evido’s Live-Translate TTS routing to Qwen3‑TTS, you must add a thin HTTP adapter service (FastAPI/Flask/etc.) around the Python model APIs. The existing repo does not provide an HTTP endpoint or Docker image for that purpose, so you will need to implement and containerize it yourself. (This is an architectural gap, not a small config change.)

## 2) What Evido needs (high-level fit)

Evido’s TTS routing expects an HTTP provider endpoint with config keys like `endpoint`, optional `auth_token`/headers, and structured parameters (e.g., `chatterbox_parameters` or `piper_parameters`). That is compatible with a **custom HTTP provider** as long as your service exposes a stable URL (e.g., `/v1/audio/speech`) and returns audio in a format Evido can consume.

**Architecture note (conflict/risk):** Qwen3‑TTS does **not** implement a request/response schema. If Evido expects an OpenAI‑style schema, you must decide whether to:
1. **Emulate OpenAI audio endpoints** (`/v1/audio/speech`, JSON input, audio bytes response), or
2. **Define a custom schema** and register it in Evido’s provider registry.

Option (1) is typically safer because it aligns with existing Evido configuration fields (endpoint + auth token), but it imposes a contract you must implement carefully. Option (2) is more flexible but requires new provider adapter logic in Evido (bigger architectural change).

## 3) Recommended adapter design (service architecture)

### 3.1 API contract (suggested)

**Minimal request model (OpenAI-like):**
- `input`: text (string)
- `model`: optional string to select which Qwen3‑TTS model to load
- `voice`: map to Qwen3‑TTS speaker (for CustomVoice)
- `language`: map to Qwen3‑TTS language
- `instruct`: optional natural language instruction (VoiceDesign or CustomVoice)
- `response_format`: `wav` or `mp3` (must be supported by your service)

**Response:** audio bytes + `Content-Type: audio/wav` (or `audio/mpeg`).

This contract maps cleanly onto Qwen3‑TTS generation APIs:
- `generate_custom_voice(...)` for predefined speakers (CustomVoice models).
- `generate_voice_design(...)` when you want natural‑language voice description control.
- `generate_voice_clone(...)` if Evido supports providing reference audio in its pipeline (requires extra input fields for `ref_audio`, `ref_text`, or `voice_clone_prompt`).

**Architectural caution:** Voice clone is a different request shape (audio input). Unless Evido can provide reference audio, it is safer to expose only CustomVoice or VoiceDesign in the HTTP adapter and register those in Evido’s mapping. Mixing modes in one endpoint can lead to complex validation and security concerns.

### 3.2 Model lifecycle

Use the repository’s documented `Qwen3TTSModel.from_pretrained(...)` as the single model-loader entrypoint so that config, processor, and generate defaults are set up correctly. The README lists the available model identifiers and installation steps; those must be baked into the container image or pulled at runtime with a persistent cache volume.

**Operational recommendation:**
- Load the model once at process startup and reuse it per request (avoid per‑request initialization).
- Guard against GPU/CPU device mismatch by validating `device_map` and `dtype` at startup.

### 3.3 Containerization (Docker) requirements

Because the repo has no existing Dockerfile, you must create one for the adapter service. The container should include:

- **CUDA runtime + torch** appropriate to your GPU stack (e.g., `nvidia/cuda` base image).
- **Python 3.12** (as recommended in README).
- **qwen-tts** installed from source or PyPI, depending on whether you need local modifications. The README shows both options.
- **Model weights** either baked into the image (large) or downloaded at runtime into a mounted cache volume.

**Architecture fit:** If you must deploy in Evido’s Docker environment, align GPU access with `nvidia-container-toolkit` and keep model cache in a writable volume to avoid re-downloading.

### 3.4 Evido mapping configuration

A typical mapping entry in Evido should point to your adapter service URL and define any additional parameters the adapter expects (e.g., `voice`, `language`, `instruct`). Since Qwen3‑TTS uses **speaker names** (for CustomVoice) and **language names** (e.g., `English`, `Chinese`), your adapter should translate Evido’s language codes and voice keys into Qwen3‑TTS inputs. The supported languages/speakers are validated inside `Qwen3TTSModel`, so mismatched mappings will fail early with a `ValueError`.

**Example mapping concept (Evido side):**
- `endpoint`: `http://qwen3-tts-adapter:8000/v1/audio/speech`
- `auth_token`: bearer token if you enable auth in your service
- `chatterbox_parameters` (or equivalent) could be repurposed to store: `voice`, `language`, `instruct`, `response_format`

**Architecture caveat:** Evido’s configuration schema has provider‑specific key validation. If it only accepts `chatterbox_parameters`/`piper_parameters`, you may need to re‑use those keys and parse them in your adapter service to avoid backend schema conflicts.

## 4) Security considerations (must-have)

- **Auth + TLS:** If Evido calls the TTS service over the network, require a bearer token and terminate TLS at a reverse proxy or inside the service. Otherwise, TTS endpoints can become an open proxy for GPU time.
- **Input validation:** Validate text length, supported language/speaker values (Qwen3‑TTS already enforces this) and reject unsupported or excessively long inputs to reduce prompt‑based resource abuse.
- **PII / logging:** Avoid logging raw text or voice prompts in production logs.
- **Model file integrity:** Verify model checksums or load from an internal model registry to avoid malicious weights.

## 5) Risks, conflicts, and alternatives

- **No built‑in HTTP API:** The biggest architectural gap is that Qwen3‑TTS ships only a Python API and a Gradio demo, not a production HTTP server. Plan for a dedicated adapter service in the architecture; do not attempt to retrofit the demo into production usage.
- **Schema mismatch risk:** Evido’s TTS mapping UI expects provider‑specific schema validation. If your custom adapter does not fit existing `chatterbox`/`piper` schemas, you may need a new provider key or a validation bypass (bigger change).
- **Streaming expectations:** If Evido expects streaming audio responses, you will need to implement chunked transfer or a streaming endpoint. The Qwen3‑TTS API is currently batch oriented; it returns full waveforms after generation, not a token stream.

## 6) Next steps (practical plan)

1. Implement a small FastAPI adapter that wraps `Qwen3TTSModel` and exposes `/v1/audio/speech`.
2. Add a Dockerfile + runtime config (env vars for model id, device, dtype, output format).
3. Define a mapping template in Evido (endpoint + token + voice/language fields) and validate with health checks.
4. Add structured logging and metrics (latency, error codes) for Evido health checks.

> This repo currently lacks the adapter and container runtime glue, so these steps are **required** before Evido can route to Qwen3‑TTS.
