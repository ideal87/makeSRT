import streamlit as st
import subprocess
import os
import tempfile
import re
import numpy as np
import librosa
from scipy.signal import fftconvolve

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000        # Downsample everything to 16 kHz mono
CLIP_DURATION = 10.0       # Use first N seconds of uploaded audio as template

# ── Helpers ──────────────────────────────────────────────────────────────────

def sanitize_youtube_url(url: str) -> str:
    """Extract and return a clean YouTube URL (watch or youtu.be)."""
    url = url.strip()
    match = re.search(r"(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"(https?://youtu\.be/[\w-]+)", url)
    if match:
        return match.group(1)
    return url


def extract_video_id(url: str) -> str | None:
    """Pull the video ID out of a YouTube URL for building timestamped links."""
    m = re.search(r"(?:v=|youtu\.be/)([\w-]+)", url)
    return m.group(1) if m else None


def download_youtube_audio(url: str, output_path: str) -> None:
    """Download only audio from a YouTube URL as a 16 kHz mono WAV."""
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-x",                          # extract audio
        "--audio-format", "wav",
        "--postprocessor-args",
        f"ffmpeg:-ar {SAMPLE_RATE} -ac 1",   # resample on the fly
        "-o", output_path,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (code {result.returncode}):\n{result.stderr}"
        )


def load_clip(file_path: str, duration: float = CLIP_DURATION) -> np.ndarray:
    """Load the first `duration` seconds of an audio file as a 16 kHz mono array."""
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=duration)
    return y


def load_full_audio(file_path: str) -> np.ndarray:
    """Load an entire audio file as a 16 kHz mono array."""
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return y


def find_offset(full_audio: np.ndarray, clip: np.ndarray):
    """
    Use normalised cross-correlation to find where `clip` starts in
    `full_audio`.

    Returns (offset_seconds, confidence) where confidence ∈ [0, 1].
    """
    # Normalise both signals
    full_audio = full_audio / (np.max(np.abs(full_audio)) + 1e-9)
    clip = clip / (np.max(np.abs(clip)) + 1e-9)

    # Cross-correlate using FFT (fast even for long signals)
    correlation = fftconvolve(full_audio, clip[::-1], mode="full")

    # The peak in the correlation gives the best-match position
    peak_index = np.argmax(np.abs(correlation))

    # Convert index → offset in the original full_audio
    offset_samples = peak_index - (len(clip) - 1)
    offset_seconds = offset_samples / SAMPLE_RATE

    # Confidence: peak value normalised by the energy of both signals
    energy = np.sqrt(np.sum(clip ** 2) * np.sum(full_audio ** 2))
    confidence = np.abs(correlation[peak_index]) / (energy + 1e-9)

    return max(0.0, offset_seconds), confidence


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Audio Timestamp Finder", page_icon="🎯")

st.title("🎯 Audio Timestamp Finder")
st.markdown(
    "Upload a sermon audio clip and paste the YouTube URL of the full worship "
    "service. The app will find the exact timestamp where your clip begins."
)

st.divider()

# Inputs
uploaded_file = st.file_uploader(
    "Upload audio clip (MP3 or WAV)",
    type=["mp3", "wav"],
    help="The first ~10 seconds will be used as the search template.",
)

youtube_url = st.text_input(
    "YouTube URL of the full worship service",
    placeholder="https://www.youtube.com/watch?v=...",
)

find_btn = st.button("🔍 Find Timestamp", type="primary", use_container_width=True)

# ── Processing ───────────────────────────────────────────────────────────────

if find_btn:
    # -- Validate inputs --
    if uploaded_file is None:
        st.error("Please upload an audio file first.")
        st.stop()
    if not youtube_url.strip():
        st.error("Please enter a YouTube URL.")
        st.stop()

    clean_url = sanitize_youtube_url(youtube_url)
    video_id = extract_video_id(clean_url)
    if not video_id:
        st.error("Could not parse a valid YouTube video ID from the URL.")
        st.stop()

    # -- Save uploaded file to temp --
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        clip_path = tmp.name

    yt_audio_path = os.path.join(tempfile.gettempdir(), f"yt_{video_id}.wav")

    try:
        # Step 1: Download YouTube audio
        status = st.status("Working…", expanded=True)
        status.write("⬇️  Downloading audio from YouTube…")

        download_youtube_audio(clean_url, yt_audio_path)
        status.write("✅  YouTube audio downloaded.")

        # Step 2: Load audio signals
        status.write("🎵  Loading audio signals…")
        clip = load_clip(clip_path, duration=CLIP_DURATION)
        full_audio = load_full_audio(yt_audio_path)
        status.write(
            f"✅  Loaded clip ({len(clip)/SAMPLE_RATE:.1f}s) and full audio "
            f"({len(full_audio)/SAMPLE_RATE:.1f}s)."
        )

        # Step 3: Cross-correlation matching
        status.write("🔍  Matching audio fingerprint…")
        offset_sec, confidence = find_offset(full_audio, clip)
        status.write("✅  Match found!")
        status.update(label="Done!", state="complete")

        # -- Display results --
        st.divider()
        st.subheader("Result")

        col1, col2 = st.columns(2)
        col1.metric("⏱️ Timestamp", seconds_to_hms(offset_sec))
        col2.metric("📊 Confidence", f"{confidence:.2%}")

        yt_link = f"https://www.youtube.com/watch?v={video_id}&t={int(offset_sec)}s"
        st.markdown(f"▶️ [Open YouTube at this timestamp]({yt_link})")

        if confidence < 0.15:
            st.warning(
                "The confidence score is low. The match may not be accurate. "
                "Try uploading a longer or cleaner audio clip."
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        # Clean up temp files
        if os.path.exists(clip_path):
            os.remove(clip_path)
        if os.path.exists(yt_audio_path):
            os.remove(yt_audio_path)
