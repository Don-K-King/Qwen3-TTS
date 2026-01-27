import os

import uvicorn


def main() -> None:
    host = os.getenv("QWEN_TTS_HOST", "0.0.0.0")
    port = int(os.getenv("QWEN_TTS_PORT", "8000"))
    log_level = os.getenv("QWEN_TTS_LOG_LEVEL", "info")
    uvicorn.run("qwen_tts.api.app:app", host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
