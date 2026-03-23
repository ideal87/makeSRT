import streamlit as st
import subprocess
import os
import tempfile
import re
import numpy as np
import librosa
from scipy.signal import fftconvolve
import time

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000        # Downsample everything to 16 kHz mono
CLIP_DURATION = 30.0       # Use first N seconds of uploaded audio as template
SEARCH_WINDOW = 1800       # 30-minute search window in seconds

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
    """Download full audio from a YouTube URL as a 16 kHz mono WAV."""
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-x",                          # extract audio
        "--audio-format", "wav",
        "--postprocessor-args",
        f"ffmpeg:-ar {SAMPLE_RATE} -ac 1",   # resample on the fly
        "-o", output_path,
    ]
    cmd.append(url)
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


def seconds_to_hms(seconds: float, ms: bool = False) -> str:
    """Convert seconds to HH:MM:SS (or HH:MM:SS.mmm) string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if ms:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{h:02d}:{m:02d}:{int(s):02d}"


def hms_to_seconds(hms: str) -> float:
    """Convert HH:MM:SS or MM:SS string to seconds."""
    parts = hms.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Audio Timestamp Finder", page_icon="🎯")

st.title("🎯 Audio Timestamp Finder")
st.markdown(
    "Select your sources for the **Short Clip** and the **Full Audio**, then click find! "
    "The app will locate the exact timestamp where the short clip begins within the full audio."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Short Clip (Search Template)")
    clip_src = st.radio("Source", ["Local File", "YouTube URL"], key="clip_src", label_visibility="collapsed")
    if clip_src == "Local File":
        clip_input = st.file_uploader(
            "Upload audio clip (MP3 or WAV)",
            type=["mp3", "wav"],
            help="The first ~30 seconds will be used as the search template.",
            key="clip_uploader"
        )
    else:
        clip_input = st.text_input(
            "YouTube URL of short clip",
            placeholder="https://www.youtube.com/watch?v=...",
            help="The first ~30 seconds of this video will be used as the search template.",
            key="clip_url"
        )

with col2:
    st.subheader("2. Full Audio (Search Target)")
    full_src = st.radio("Source", ["YouTube URL", "Local File"], key="full_src", label_visibility="collapsed")
    if full_src == "YouTube URL":
        full_input = st.text_input(
            "YouTube URL of full audio",
            placeholder="https://www.youtube.com/watch?v=...",
            key="full_url"
        )
    else:
        full_input = st.file_uploader(
            "Upload full audio file (MP3/WAV)",
            type=["mp3", "wav"],
            key="full_uploader"
        )

st.divider()

window_start = st.text_input(
    "Search window start time (HH:MM:SS)",
    value="00:00:00",
    help="The app will search a 30-minute window starting at this time. "
         "Change this and click Find again to search a different range "
         "(the audio won't be re-downloaded).",
)

find_btn = st.button("🔍 Find Timestamp", type="primary", use_container_width=True)

# ── Processing ───────────────────────────────────────────────────────────────

if find_btn:
    # -- Validate inputs --
    if not clip_input:
        st.error("Please provide a short clip source.")
        st.stop()
    if not full_input:
        st.error("Please provide a full audio source.")
        st.stop()

    clip_url = None
    full_url = None

    if clip_src == "YouTube URL":
        clip_url = sanitize_youtube_url(clip_input)
        if not extract_video_id(clip_url):
            st.error("Invalid YouTube URL for Short Clip.")
            st.stop()
    
    if full_src == "YouTube URL":
        full_url = sanitize_youtube_url(full_input)
        if not extract_video_id(full_url):
            st.error("Invalid YouTube URL for Full Audio.")
            st.stop()

    try:
        window_start_sec = hms_to_seconds(window_start)
    except Exception:
        st.error("Invalid time format for Search Window Start. Use HH:MM:SS.")
        st.stop()
        
    window_end_sec = window_start_sec + SEARCH_WINDOW

    status = st.status("Working…", expanded=True)

    try:
        t0 = time.time()

        def elapsed() -> str:
            return f"[{time.time() - t0:.1f}s]"

        # ── Step 1: Load Full Audio ─────────────────────────────────────────
        if full_src == "YouTube URL":
            assert full_url is not None
            video_id = extract_video_id(full_url)
            cached = st.session_state.get("cached_full_id")
            if cached == video_id and "full_audio" in st.session_state:
                status.write(f"{elapsed()}  ✅  Using cached full YouTube audio.")
                full_audio = st.session_state["full_audio"]
            else:
                status.write(f"{elapsed()}  ⬇️  Downloading full audio from YouTube…")
                yt_audio_path = os.path.join(tempfile.gettempdir(), f"yt_full_{video_id}.wav")
                assert full_url is not None
                download_youtube_audio(full_url, yt_audio_path)
                status.write(f"{elapsed()}  🎵  Loading full audio into memory…")
                full_audio = load_full_audio(yt_audio_path)
                st.session_state["full_audio"] = full_audio
                st.session_state["cached_full_id"] = video_id
                if os.path.exists(yt_audio_path):
                    os.remove(yt_audio_path)
                status.write(f"{elapsed()}  ✅  Full audio loaded ({len(full_audio)/SAMPLE_RATE:.1f}s).")
        else:
            file_id = f"full_{full_input.name}_{full_input.size}"
            if st.session_state.get("cached_full_id") == file_id and "full_audio" in st.session_state:
                status.write(f"{elapsed()}  ✅  Using cached full local audio.")
                full_audio = st.session_state["full_audio"]
            else:
                status.write(f"{elapsed()}  🎵  Loading full local audio into memory…")
                suffix = os.path.splitext(full_input.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(full_input.read())
                    full_path = tmp.name
                
                full_audio = load_full_audio(full_path)
                os.remove(full_path)
                st.session_state["full_audio"] = full_audio
                st.session_state["cached_full_id"] = file_id
                status.write(f"{elapsed()}  ✅  Full audio loaded ({len(full_audio)/SAMPLE_RATE:.1f}s).")

        # ── Step 2: Load the Short Clip ──────────────────────────────────────
        if clip_src == "YouTube URL":
            assert clip_url is not None
            video_id = extract_video_id(clip_url)
            cached = st.session_state.get("cached_clip_id")
            if cached == video_id and "clip_audio" in st.session_state:
                status.write(f"{elapsed()}  ✅  Using cached short clip audio.")
                clip = st.session_state["clip_audio"]
            else:
                status.write(f"{elapsed()}  ⬇️  Downloading short clip from YouTube…")
                yt_clip_path = os.path.join(tempfile.gettempdir(), f"yt_clip_{video_id}.wav")
                assert clip_url is not None
                download_youtube_audio(clip_url, yt_clip_path)
                status.write(f"{elapsed()}  🎵  Loading clip audio into memory (first {CLIP_DURATION}s)…")
                clip = load_clip(yt_clip_path, duration=CLIP_DURATION)
                if os.path.exists(yt_clip_path):
                    os.remove(yt_clip_path)
                st.session_state["clip_audio"] = clip
                st.session_state["cached_clip_id"] = video_id
        else:
            file_id = f"clip_{clip_input.name}_{clip_input.size}"
            if st.session_state.get("cached_clip_id") == file_id and "clip_audio" in st.session_state:
                status.write(f"{elapsed()}  ✅  Using cached short local clip.")
                clip = st.session_state["clip_audio"]
            else:
                suffix = os.path.splitext(clip_input.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(clip_input.read())
                    clip_path = tmp.name
                
                status.write(f"{elapsed()}  🎵  Loading clip audio into memory (first {CLIP_DURATION}s)…")
                clip = load_clip(clip_path, duration=CLIP_DURATION)
                os.remove(clip_path)
                st.session_state["clip_audio"] = clip
                st.session_state["cached_clip_id"] = file_id

        status.write(f"{elapsed()}  ✅  Clip loaded ({len(clip)/SAMPLE_RATE:.1f}s).")

        # ── Step 3: Slice to search window & match ───────────────────────
        total_duration = len(full_audio) / SAMPLE_RATE

        start_sample = int(window_start_sec * SAMPLE_RATE)
        end_sample = int(window_end_sec * SAMPLE_RATE)

        # Clamp to actual audio length
        start_sample = max(0, min(start_sample, len(full_audio)))
        end_sample = max(start_sample, min(end_sample, len(full_audio)))

        window_audio = full_audio[start_sample:end_sample]

        if len(window_audio) == 0:
            status.update(label="Done — bad range", state="error")
            st.error(
                f"The search window ({seconds_to_hms(window_start_sec)} → "
                f"{seconds_to_hms(window_end_sec)}) is outside the audio "
                f"(total length {seconds_to_hms(total_duration)}). "
                "Please adjust the start time."
            )
            st.stop()

        status.write(
            f"{elapsed()}  🔍  Searching in window "
            f"{seconds_to_hms(window_start_sec)} → "
            f"{seconds_to_hms(min(window_end_sec, total_duration))} "
            f"({len(window_audio)/SAMPLE_RATE:.1f}s)…"
        )

        offset_in_window, confidence = find_offset(window_audio, clip)

        # Adjust offset to be relative to the full video
        offset_sec = offset_in_window + window_start_sec

        # -- Display results --
        st.divider()

        if confidence < 0.01:
            status.write(f"{elapsed()}  ❌  No match found.")
            status.update(label=f"Done — no match ({elapsed()})", state="error")
            st.subheader("Result")
            st.error(
                "No matching audio found in the search window. "
                "Try adjusting the start time and clicking **Find Timestamp** "
                "again (the audio won't be re-downloaded)."
            )
        else:
            status.write(f"{elapsed()}  ✅  Match found!")
            status.update(label=f"Done! ({elapsed()})", state="complete")
            st.subheader("Result")

            col1, col2 = st.columns(2)
            col1.metric("⏱️ Timestamp", seconds_to_hms(offset_sec, ms=True))
            col2.metric("📊 Confidence", f"{confidence:.2%}")

            if full_src == "YouTube URL":
                assert full_url is not None
                full_video_id = extract_video_id(full_url)
                yt_link = f"https://www.youtube.com/watch?v={full_video_id}&t={int(offset_sec)}s"
                st.markdown(f"▶️ [Open YouTube at this timestamp]({yt_link})")
            else:
                st.info("Since the full audio is a local file, navigate to this timestamp in your local media player.")

            if confidence < 0.15:
                st.warning(
                    "The confidence score is low. The match may not be accurate. "
                    "Try using a longer or cleaner audio clip."
                )

    except Exception as e:
        status.update(label="Error occurred", state="error")
        st.error(f"An error occurred: {e}")

