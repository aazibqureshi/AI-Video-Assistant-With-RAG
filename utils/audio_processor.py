import os
import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """
    Download YouTube audio and convert it to WAV.

    Important:
    YouTube may block cloud/server IP addresses.
    This function does not force a specific YouTube
    player client.
    """

    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")

    ydl_opts = {
        # Let yt-dlp choose an available audio format.
        # Fall back to a combined format if necessary.
        "format": "ba/b",
        "outtmpl": output_path,
        "noplaylist": True,
        # Convert downloaded media to WAV.
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        # Retry temporary network failures.
        "retries": 3,
        "fragment_retries": 3,
        # Do NOT force player_client.
        # Current YouTube extraction is handled
        # by yt-dlp's default client selection.
        "quiet": False,
        "no_warnings": False,
        "keepvideo": False,
    }

    try:
        print("Extracting YouTube video...")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            filename = ydl.prepare_filename(info)

            base, _ = os.path.splitext(filename)

            wav_path = base + ".wav"

            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"WAV file was not created: {wav_path}")

            print(f"YouTube audio downloaded: {wav_path}")

            return wav_path

    except yt_dlp.utils.DownloadError as e:
        error_message = str(e)

        # Give a useful error instead of hiding
        # the actual YouTube problem.
        if "Sign in to confirm" in error_message:
            raise RuntimeError(
                "YouTube blocked this request because "
                "the server appears to be a bot.\n\n"
                "This is a YouTube authentication/rate-limit "
                "problem, not an audio processing problem."
            ) from e

        if "Requested format is not available" in error_message:
            raise RuntimeError(
                "YouTube did not provide a downloadable "
                "audio/video format for this request.\n\n"
                f"yt-dlp error: {error_message}"
            ) from e

        raise RuntimeError(f"YouTube audio download failed.\n\n{error_message}") from e

    except Exception as e:
        raise RuntimeError(f"Unexpected YouTube download error: {str(e)}") from e


def convert_to_wav(input_path: str) -> str:
    """
    Convert local audio/video file to WAV.

    Output:
    - mono
    - 16 kHz
    """

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)

    audio = audio.set_channels(1).set_frame_rate(16000)

    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_seconds: int = 25) -> list:
    """
    Split WAV into chunks of <= 25 seconds.
    """

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_seconds * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]

        chunk_path = f"{wav_path}_chunk_{i}.wav"

        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    """
    Process either:

    1. YouTube URL
    2. Local audio/video file

    Returns a list of WAV chunks.
    """

    if source.startswith(("http://", "https://")):
        print("Detected YouTube URL. Downloading audio...")

        wav_path = download_youtube_audio(source)

    else:
        print("Detected local file. Converting to WAV...")

        wav_path = convert_to_wav(source)

    print("Chunking audio...")

    chunks = chunk_audio(wav_path)

    print(f"Audio ready — {len(chunks)} chunk(s) created.")

    return chunks
