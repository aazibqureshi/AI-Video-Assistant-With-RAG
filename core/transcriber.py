import os
import requests
from groq import Groq

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL", "small"
)  # kept for reference, no longer used

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")  # fastest option

_groq_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in environment / .env")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def transcribe_chunk_groq(chunk_path: str) -> str:
    client = get_groq_client()

    with open(chunk_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(chunk_path), f.read()),
            model=GROQ_STT_MODEL,
            response_format="text",
        )

    # response_format="text" -> transcription is a plain string
    return transcription if isinstance(transcription, str) else transcription.text


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(chunk_path, "rb") as f:
        files = {"file": (os.path.basename(chunk_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=300,
        )

    if response.status_code != 200:
        print("SARVAM ERROR RESPONSE:", response.text)
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk depending on language choice.
     - english  -> Groq (Whisper API, fast, no local model)
     - hinglish -> Sarvam (translates Hindi to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_groq(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Groq Whisper API"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription completed")
    return full_transcript.strip()
