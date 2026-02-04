import io
import logging
import os
from typing import Optional

import soundfile as sf
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

logger = logging.getLogger(__name__)


class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to synthesize.")
    model: Optional[str] = Field(None, description="Optional model id override.")
    voice: Optional[str] = Field(None, description="Speaker name for CustomVoice models.")
    language: Optional[str] = Field(None, description="Language name (e.g., English, Chinese, Auto).")
    instruct: Optional[str] = Field(None, description="Instruction text for voice style/design.")
    response_format: str = Field("wav", description="Output audio format. Only wav is supported.")
    ref_audio: Optional[str] = Field(None, description="Reference audio path/URL/base64 for voice clone.")
    ref_text: Optional[str] = Field(None, description="Reference transcript for voice clone when ICL is enabled.")
    x_vector_only_mode: bool = Field(
        False,
        description="If true, only speaker embedding is used and ref_text is optional.",
    )


class AppConfig(BaseModel):
    model_id: str
    device_map: Optional[str]
    dtype: Optional[str]
    max_input_chars: int
    auth_token: Optional[str]
    default_language: str
    default_voice: Optional[str]


app = FastAPI(title="Qwen3-TTS HTTP API", version="0.1.0")


DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@app.on_event("startup")
def _load_model() -> None:
    config = _load_config()
    app.state.config = config
    model_kwargs = {}
    resolved_device_map = _resolve_device_map(config.device_map)
    if resolved_device_map:
        model_kwargs["device_map"] = resolved_device_map
    resolved_dtype = _resolve_dtype(config.dtype)
    if resolved_dtype is not None:
        model_kwargs["dtype"] = DTYPE_MAP[resolved_dtype]
    logger.info("Loading Qwen3-TTS model: %s", config.model_id)
    app.state.model = Qwen3TTSModel.from_pretrained(config.model_id, **model_kwargs)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _load_config() -> AppConfig:
    model_id = os.getenv("QWEN_TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    device_map = os.getenv("QWEN_TTS_DEVICE_MAP")
    dtype_raw = os.getenv("QWEN_TTS_DTYPE")
    dtype = _normalize_dtype(dtype_raw)
    max_input_chars = int(os.getenv("QWEN_TTS_MAX_INPUT_CHARS", "2000"))
    auth_token = os.getenv("QWEN_TTS_AUTH_TOKEN")
    default_language = os.getenv("QWEN_TTS_DEFAULT_LANGUAGE", "Auto")
    default_voice = os.getenv("QWEN_TTS_DEFAULT_VOICE")
    return AppConfig(
        model_id=model_id,
        device_map=device_map,
        dtype=dtype,
        max_input_chars=max_input_chars,
        auth_token=auth_token,
        default_language=default_language,
        default_voice=default_voice,
    )


def _normalize_dtype(dtype_raw: Optional[str]) -> Optional[str]:
    if not dtype_raw:
        return None
    normalized = dtype_raw.strip().lower()
    if normalized in DTYPE_MAP:
        return normalized
    raise ValueError(f"Unsupported QWEN_TTS_DTYPE: {dtype_raw}")


def _resolve_device_map(device_map: Optional[str]) -> Optional[str]:
    if not device_map:
        return None
    normalized = device_map.strip()
    if torch.cuda.is_available():
        return normalized
    if "cuda" in normalized.lower():
        logger.warning("CUDA requested via QWEN_TTS_DEVICE_MAP=%s, but no GPU detected; using CPU instead.", device_map)
        return "cpu"
    return normalized


def _resolve_dtype(dtype: Optional[str]) -> Optional[str]:
    if dtype is None:
        return None
    if torch.cuda.is_available():
        return dtype
    if dtype in {"float16", "fp16", "bfloat16", "bf16"}:
        logger.warning(
            "Low-precision dtype %s requested without CUDA; falling back to float32 for CPU stability.",
            dtype,
        )
        return "float32"
    return dtype


def _authorize(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    config: AppConfig = app.state.config
    if not config.auth_token:
        return
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
    else:
        token = x_api_key
    if not token or token != config.auth_token:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _resolve_mode(payload: SpeechRequest) -> str:
    if payload.ref_audio:
        return "voice_clone"
    if payload.voice:
        return "custom_voice"
    if payload.instruct:
        return "voice_design"
    return "unknown"


@app.post("/v1/audio/speech", dependencies=[Depends(_authorize)])
def synthesize(payload: SpeechRequest) -> Response:
    config: AppConfig = app.state.config
    model: Qwen3TTSModel = app.state.model

    text = payload.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="input must not be empty")
    if len(text) > config.max_input_chars:
        raise HTTPException(status_code=400, detail="input exceeds QWEN_TTS_MAX_INPUT_CHARS")
    if payload.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="Only wav response_format is supported")
    if payload.model and payload.model != config.model_id:
        raise HTTPException(status_code=400, detail="Requested model is not allowed")

    language = payload.language or config.default_language
    mode = _resolve_mode(payload)

    if mode == "custom_voice":
        speaker = payload.voice or config.default_voice
        if not speaker:
            raise HTTPException(status_code=400, detail="voice is required for custom_voice mode")
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=payload.instruct,
        )
    elif mode == "voice_design":
        if not payload.instruct:
            raise HTTPException(status_code=400, detail="instruct is required for voice_design mode")
        wavs, sample_rate = model.generate_voice_design(
            text=text,
            instruct=payload.instruct,
            language=language,
        )
    elif mode == "voice_clone":
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=payload.ref_audio,
            ref_text=payload.ref_text,
            x_vector_only_mode=payload.x_vector_only_mode,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide ref_audio (voice_clone), voice (custom_voice), or instruct (voice_design).",
        )

    if not wavs:
        raise HTTPException(status_code=500, detail="No audio generated")
    buffer = io.BytesIO()
    sf.write(buffer, wavs[0], sample_rate, format="WAV")
    return Response(content=buffer.getvalue(), media_type="audio/wav")
