import os
import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """
    Download audio from a YouTube URL and convert it to WAV.

    Important:
    - Do NOT force a specific YouTube player client.
    - Let the current yt-dlp version choose the appropriate client.
    - FFmpeg is required for audio extraction.
    """

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(id)s.%(ext)s"
    )

    ydl_opts = {
        # Prefer audio-only.
        # Fall back to the best available combined format.
        "format": "ba/b",

        "outtmpl": output_path,

        # Convert downloaded audio to WAV
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],

        "noplaylist": True,

        # Keep logs visible on Streamlit Cloud
        "quiet": False,
        "no_warnings": False,

        # Network retries
        "retries": 5,
        "fragment_retries": 5,

        # Don't force web/android clients.
        # yt-dlp's current default client selection
        # is safer because YouTube changes its clients.
        #
        # IMPORTANT:
        # No "extractor_args" here.

        "keepvideo": False,
    }

    try:

        print("Starting YouTube extraction...")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(
                url,
                download=True
            )

            filename = ydl.prepare_filename(info)

            # FFmpegExtractAudio changes extension
            # to .wav.
            base, _ = os.path.splitext(filename)

            wav_path = base + ".wav"

            if not os.path.exists(wav_path):
                raise FileNotFoundError(
                    f"FFmpeg did not create WAV file: "
                    f"{wav_path}"
                )

            print(
                f"YouTube audio successfully "
                f"downloaded: {wav_path}"
            )

            return wav_path

    except yt_dlp.utils.DownloadError as e:

        error_message = str(e)

        print(
            "yt-dlp DownloadError:"
        )
        print(error_message)

        raise RuntimeError(
            "YouTube audio download failed.\n\n"
            f"{error_message}\n\n"
            "Please check the Streamlit logs for "
            "the complete yt-dlp error."
        ) from e

    except Exception as e:

        print(
            "Unexpected YouTube download error:"
        )
        print(repr(e))

        raise RuntimeError(
            "Unexpected error while downloading "
            f"YouTube audio: {str(e)}"
        ) from e


def convert_to_wav(input_path: str) -> str:
    """
    Convert any local audio/video file to WAV.

    Output:
    - mono
    - 16 kHz
    """

    output_path = (
        os.path.splitext(input_path)[0]
        + "_converted.wav"
    )

    audio = AudioSegment.from_file(
        input_path
    )

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

    audio = AudioSegment.from_wav(
        wav_path
    )

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

    Returns:
        List of WAV chunk paths.
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

    chunks = chunk_audio(
        wav_path
    )

    print(
        f"Audio ready — "
        f"{len(chunks)} chunk(s) created."
    )

    return chunks