"""
Headless audio timestamp finder (no UI).

Given a local audio clip (e.g. an .m4a whose filename contains a YYYYMMDD date),
find the matching full-length YouTube video by title, download its audio, and
locate the timestamp where the clip begins. This is the "Local File clip +
YouTube URL full audio" workflow from app.py, except nobody is around to adjust
the search window: if the default window has no confident match, the windows at
-30 min, +30 min, -60 min and +60 min are tried automatically.

Usage:
    python find_timestamp.py path/to/20260329_sermon.m4a

Progress is logged to stderr; the result is printed to stdout.
Exit codes: 0 = match found, 1 = no confident match, 2 = error.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import warnings

import librosa
import numpy as np
from scipy.signal import fftconvolve

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000                 # Downsample everything to 16 kHz mono
CLIP_DURATION = 30.0                # Use first N seconds of the clip as template
SEARCH_WINDOW = 1800                # 30-minute search window in seconds
DEFAULT_WINDOW_START = 3 * 3600 + 30 * 60   # 03:30:00, same default as app.py
WINDOW_SHIFTS_MIN = (0, -30, 30, -60, 60)   # Tried in order, relative to the start
MIN_CONFIDENCE = 0.02               # lower than app.py's 3% human-review cutoff; noise windows stay under ~1%
CHANNEL_URL = "https://www.youtube.com/@nwtvmedia"

_T0 = time.time()


class FinderError(Exception):
    """A failure the user should see as a plain message rather than a traceback."""


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}", file=sys.stderr, flush=True)


def get_subprocess_env() -> dict[str, str]:
    """Ensure child processes (especially yt-dlp on Windows) use UTF-8 I/O encoding."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONLEGACYWINDOWSSTDIO"] = "0"
    return env


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
    parts = [float(p) for p in hms.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def watch_url(video_id: str, start_sec: float | None = None) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    if start_sec is not None:
        url += f"&t={int(start_sec)}s"
    return url


# ── YouTube search & download ────────────────────────────────────────────────

def extract_date(filename: str) -> str | None:
    """Pull a YYYYMMDD date out of a filename like '2026-03-29_x.m4a' or '20260329.m4a'."""
    m = re.search(r"(\d{4})[-._]?(\d{2})[-._]?(\d{2})", filename)
    return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else None


def date_title_patterns(date: str) -> list[str]:
    """Every way the date might be written in a video title."""
    y, m, d = date[0:4], date[4:6], date[6:8]
    patterns = [
        f"{y}-{m}-{d}", f"{y}.{m}.{d}",
        f"{m}/{d}/{y}", f"{m}-{d}-{y}", f"{m}.{d}.{y}",
    ]
    try:
        dt = datetime.datetime.strptime(date, "%Y%m%d")
        patterns.append(dt.strftime("%B %d, %Y").replace(" 0", " "))
        patterns.append(dt.strftime("%b %d, %Y").replace(" 0", " "))
    except ValueError:
        pass  # not a real calendar date; the numeric patterns just won't match
    return patterns


def score_full_audio(title: str) -> int:
    """Rank titles so the main Sunday worship recording beats praise/prayer/clips."""
    score = 0
    # Primary Sunday worship service: 주일예배 / 주일2부 without 저녁 / 찬양
    if "주일" in title and "예배" in title and "저녁" not in title:
        score += 100
    elif "화요집회" in title or ("예배" in title and "저녁" not in title and "기도회" not in title):
        score += 50
    if "저녁" in title:
        score -= 20
    if "기도회" in title or "중보기도회" in title:
        score -= 30
    if "찬양" in title or "Praise" in title:
        score -= 40
    if "[" in title:
        score -= 50
    return score


def list_channel_videos(channel_url: str, limit: int = 35) -> list[dict]:
    """List the newest `limit` entries of a channel tab as [{'title', 'videoId'}]."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist",
        "--playlist-end", str(limit),
        "--encoding", "utf-8",
        "--extractor-args", "youtube:lang=ko;player_client=web,android",
        "--print", "%(title)s | %(id)s",
        channel_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=get_subprocess_env())
    if result.returncode != 0:
        log(f"⚠️  yt-dlp could not list {channel_url}: {result.stderr.strip()[-300:]}")
        return []
    videos = []
    for line in result.stdout.strip().split("\n"):
        if "|" in line:
            parts = line.split("|")
            videos.append({"title": "|".join(parts[:-1]).strip(), "videoId": parts[-1].strip()})
    return videos


def find_full_audio_video(date: str) -> dict | None:
    """
    Find the full-length recording for `date` by title match: livestreams first,
    then regular videos, preferring the main Sunday worship title.
    """
    patterns = date_title_patterns(date)
    for tab in ("streams", "videos"):
        videos = list_channel_videos(f"{CHANNEL_URL}/{tab}")
        matched = [v for v in videos if any(p in v["title"] for p in patterns)]
        if matched:
            matched.sort(key=lambda v: score_full_audio(v["title"]), reverse=True)
            return matched[0]
    return None


def download_youtube_audio(url: str, output_path: str) -> None:
    """Download full audio from a YouTube URL as a 16 kHz mono MP3 CBR."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--encoding", "utf-8",
        "--js-runtimes", "node",
        "--extractor-args", "youtube:player_client=android,web,mweb,ios",
        "-x",                          # extract audio
        "--audio-format", "mp3",
        "--postprocessor-args",
        f"ffmpeg:-ar {SAMPLE_RATE} -ac 1 -b:a 128k",   # resample and set constant bitrate
        "-o", output_path,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=get_subprocess_env())
    if result.returncode != 0:
        raise FinderError(f"yt-dlp failed (code {result.returncode}):\n{result.stderr}")


# ── Signal matching ──────────────────────────────────────────────────────────

def load_audio(file_path: str, duration: float | None = None) -> np.ndarray:
    """Load an audio file (optionally just the first `duration` seconds) as 16 kHz mono."""
    # librosa falls back from soundfile to audioread for .m4a/.mp4 and warns about it every time
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PySoundFile failed")
        warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=duration)
    return y


def find_offset(full_audio: np.ndarray, clip: np.ndarray) -> tuple[float, float]:
    """
    Use normalised cross-correlation to find where `clip` starts in `full_audio`.

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

    return max(0.0, offset_seconds), float(confidence)


def search_windows(
    full_audio: np.ndarray,
    clip: np.ndarray,
    window_start: float,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """
    Search the default window, then each shifted window in WINDOW_SHIFTS_MIN,
    stopping at the first whose confidence reaches `min_confidence`.

    Returns (match, best): `match` is (offset_sec, confidence) in full-audio time,
    or None if no window was confident enough; `best` is the highest-confidence
    candidate seen across all windows (None if no window could be searched).
    """
    total = len(full_audio) / SAMPLE_RATE
    clip_len = len(clip) / SAMPLE_RATE
    best = None

    for shift_min in WINDOW_SHIFTS_MIN:
        # Clamp to the real audio; a window that starts before 0 loses its head
        lo = max(0.0, window_start + shift_min * 60)
        hi = min(total, window_start + shift_min * 60 + SEARCH_WINDOW)
        label = f"{shift_min:+d} min" if shift_min else "default"

        if hi - lo < clip_len:
            log(f"⏭️  Window {label} is outside the audio "
                f"(length {seconds_to_hms(total)}); skipping.")
            continue

        window = full_audio[int(lo * SAMPLE_RATE):int(hi * SAMPLE_RATE)]
        offset_in_window, confidence = find_offset(window, clip)
        offset_sec = lo + offset_in_window
        log(f"🔍  Window {label} ({seconds_to_hms(lo)} → {seconds_to_hms(hi)}): "
            f"best at {seconds_to_hms(offset_sec, ms=True)}, confidence {confidence:.2%}")

        if best is None or confidence > best[1]:
            best = (offset_sec, confidence)
        if confidence >= min_confidence:
            return (offset_sec, confidence), best

    return None, best


# ── CLI ──────────────────────────────────────────────────────────────────────

def _hms_arg(value: str) -> float:
    try:
        return hms_to_seconds(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid time {value!r}; use HH:MM:SS")


def run(audio_path: str, window_start: float, min_confidence: float) -> int:
    if not os.path.isfile(audio_path):
        raise FinderError(f"File not found: {audio_path}")

    date = extract_date(os.path.basename(audio_path))
    if not date:
        raise FinderError(
            f"No YYYYMMDD date found in the filename {os.path.basename(audio_path)!r}; "
            "it is needed to match the YouTube video title."
        )
    log(f"📅  Detected date {date[0:4]}-{date[4:6]}-{date[6:8]} from the filename.")

    # Load the clip first so a bad file fails before any network work
    clip = load_audio(audio_path, duration=CLIP_DURATION)
    if len(clip) == 0:
        raise FinderError(f"Could not read any audio from {audio_path}")
    log(f"🎵  Clip loaded ({len(clip) / SAMPLE_RATE:.1f}s).")

    log("🔎  Searching YouTube for a title matching the date…")
    video = find_full_audio_video(date)
    if not video:
        raise FinderError(f"No video or stream with the date {date} in its title was found on {CHANNEL_URL}.")
    video_url = watch_url(video["videoId"])
    log(f"✅  Matched video: {video['title']}  ({video_url})")

    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = os.path.join(tmp, f"yt_full_{video['videoId']}.mp3")
        log("⬇️  Downloading full audio from YouTube…")
        download_youtube_audio(video_url, mp3_path)
        log("🎵  Loading full audio into memory…")
        full_audio = load_audio(mp3_path)
    log(f"✅  Full audio loaded ({seconds_to_hms(len(full_audio) / SAMPLE_RATE)}).")

    match, best = search_windows(full_audio, clip, window_start, min_confidence)

    if match is None:
        if best is None:
            log("❌  None of the search windows fall inside the audio.")
        else:
            log(f"❌  No window reached {min_confidence:.2%} confidence. Best candidate "
                f"was {seconds_to_hms(best[0], ms=True)} at {best[1]:.2%} (not trusted).")
        return 1

    offset_sec, confidence = match
    print(f"Video:      {video['title']}")
    print(f"URL:        {video_url}")
    print(f"Timestamp:  {seconds_to_hms(offset_sec, ms=True)}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Link:       {watch_url(video['videoId'], offset_sec)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Titles are often Korean; don't let a legacy Windows console codepage crash printing
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Find where a local audio clip starts inside its full YouTube recording."
    )
    parser.add_argument("audio", help="Short clip (.m4a/.mp3/.wav/.mp4); its filename must contain a YYYYMMDD date.")
    parser.add_argument("--window-start", type=_hms_arg, default=DEFAULT_WINDOW_START, metavar="HH:MM:SS",
                        help="Start of the default search window (default 03:30:00). "
                             "The ±30/±60 min windows are relative to this.")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE, metavar="FRACTION",
                        help=f"Confidence needed to accept a window (default {MIN_CONFIDENCE}).")
    args = parser.parse_args(argv)

    try:
        return run(args.audio, args.window_start, args.min_confidence)
    except FinderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
