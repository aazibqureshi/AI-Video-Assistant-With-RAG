import os
import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """
    Download YouTube video/audio and convert it to WAV.

    Uses multiple YouTube player clients so that yt-dlp has
    a better chance of finding an audio-capable format.
    """

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        # Prefer any format containing audio.
        # If separate video/audio streams are available,
        # yt-dlp can merge them using FFmpeg.
        "format": "bv*+ba/b",

        "outtmpl": output_path,

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],

        # Download only one video
        "noplaylist": True,

        # Show useful errors in Streamlit logs
        "quiet": False,
        "no_warnings": False,

        # Try more than only the web player client.
        # YouTube currently exposes different formats
        # through different clients.
        "extractor_args": {
            "youtube": {
                "player_client": [
                    "web_embedded",
                    "web",
                    "android_vr",
                ]
            }
        },

        # Retry temporary network failures
        "retries": 3,
        "fragment_retries": 3,

        # Don't keep unnecessary files
        "keepvideo": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            print("Extracting YouTube information...")

            info = ydl.extract_info(
                url,
                download=True
            )

            filename = ydl.prepare_filename(info)

            # FFmpegExtractAudio converts the downloaded
            # file extension to .wav
            base, _ = os.path.splitext(filename)
            wav_path = base + ".wav"

            if not os.path.exists(wav_path):
                raise FileNotFoundError(
                    f"WAV file was not created: {wav_path}"
                )

            print(f"YouTube audio downloaded: {wav_path}")

            return wav_path

    except yt_dlp.utils.DownloadError as e:

        error_message = str(e)

        raise RuntimeError(
            "YouTube audio download failed.\n\n"
            f"{error_message}\n\n"
            "Possible causes:\n"
            "1. YouTube did not expose an audio format.\n"
            "2. The video requires authentication.\n"
            "3. YouTube temporarily blocked the request.\n"
            "4. FFmpeg is not installed on the server."
        ) from e


def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to WAV format.

    The output is:
    - mono
    - 16 kHz
    """

    output_path = (
        os.path.splitext(input_path)[0]
        + "_converted.wav"
    )

    audio = AudioSegment.from_file(input_path)

    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )

    audio.export(
        output_path,
        format="wav"
    )

    return output_path


def chunk_audio(
    wav_path: str,
    chunk_seconds: int = 25
) -> list:
    """
    Split WAV into chunks of <= 25 seconds.
    """

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_seconds * 1000

    chunks = []

    for i, start in enumerate(
        range(0, len(audio), chunk_ms)
    ):

        chunk = audio[
            start:start + chunk_ms
        ]

        chunk_path = (
            f"{wav_path}_chunk_{i}.wav"
        )

        chunk.export(
            chunk_path,
            format="wav"
        )

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    """
    Process either:

    1. YouTube URL
    2. Local audio/video file

    Returns a list of WAV chunks.
    """

    if source.startswith(
        ("http://", "https://")
    ):

        print(
            "Detected YouTube URL. "
            "Downloading audio..."
        )

        wav_path = download_youtube_audio(
            source
        )

    else:

        print(
            "Detected local file. "
            "Converting to WAV..."
        )

        wav_path = convert_to_wav(
            source
        )

    print("Chunking audio...")

    chunks = chunk_audio(wav_path)

    print(
        f"Audio ready — "
        f"{len(chunks)} chunk(s) created."
    )

    return chunks