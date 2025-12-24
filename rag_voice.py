import base64
import time
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydub import AudioSegment
import subprocess
from faster_whisper import WhisperModel

# -------- IMPORT YOUR RAG CHATBOT LOGIC --------
from neo_RAG1 import chat_response   # your existing chatbot function

# -----------------------------------------------

app = FastAPI(title="Voice RAG Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("voice-rag")
logger.setLevel(logging.INFO)

TEMP_DIR = Path("temp_voice")
TEMP_DIR.mkdir(exist_ok=True)



# ----------- LOAD WHISPER (STT) ------------
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
logger.info("Whisper STT loaded.")

# ----------- SETUP PIPER (TTS) -------------
PIPER_EXE = "/usr/bin/piper"  # change if needed
PIPER_MODEL = "piper_models/en_US-lessac-medium.onnx"

def convert_to_wav(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ar", "16000",       # Whisper recommended
        "-ac", "1",           # Mono
        "-f", "wav",
        str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# ----------- SPEECH TO TEXT ----------------
def transcribe(audio_bytes: bytes):
    ts_path = TEMP_DIR / "input.webm"
    wav_path = TEMP_DIR / "input.wav"

    with open(ts_path, "wb") as f:
        f.write(audio_bytes)

    print("Received file size:", ts_path.stat().st_size)

    # audio = AudioSegment.from_file(ts_path)
    # audio.export(wav_path, format="wav")
    convert_to_wav(ts_path, wav_path)

    if not wav_path.exists():
        print("FFMPEG FAILED - WAV FILE NOT CREATED")
        return ""

    segments, info = whisper_model.transcribe(str(wav_path))
    text = " ".join([s.text for s in segments]).strip()

    ts_path.unlink(missing_ok=True)
    wav_path.unlink(missing_ok=True)

    return text

# ----------- TEXT TO SPEECH ------------------
def synthesize(text: str):
    out_path = TEMP_DIR / "output.wav"

    subprocess.run(
        [
            PIPER_EXE,
            "--model", PIPER_MODEL,
            "--output_file", str(out_path)
        ],
        input=text.encode("utf-8"),
        check=True
    )

    with open(out_path, "rb") as f:
        audio = f.read()

    out_path.unlink(missing_ok=True)
    return audio


# ----------- WEBSOCKET ENDPOINT --------------
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected.")

    try:
        while True:
            raw = await websocket.receive()

            # Handle text OR binary websocket frames
            if raw["type"] == "websocket.receive":
                if "text" in raw:
                    msg = json.loads(raw["text"])
                elif "bytes" in raw:
                    msg = json.loads(raw["bytes"].decode())
                else:
                    continue

                if msg["type"] == "audio":
                    start = time.time()

                    audio_bytes = base64.b64decode(msg["audio"])
                    user_text = transcribe(audio_bytes)

                    await websocket.send_json({
                        "type": "transcription",
                        "text": user_text
                    })

                    bot_reply = chat_response(user_text)

                    await websocket.send_json({
                        "type": "response",
                        "text": bot_reply
                    })

                    audio_reply = synthesize(bot_reply)
                    audio_b64 = base64.b64encode(audio_reply).decode()

                    await websocket.send_json({
                        "type": "audio",
                        "audio": audio_b64
                    })

                    logger.info(f"Voice pipeline done in {int((time.time() - start) * 1000)} ms")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
        return


@app.get("/")
def root():
    return {"status": "Voice RAG Bot running"}

