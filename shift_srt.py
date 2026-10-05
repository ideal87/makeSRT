"""
Headless SRT shifter (no UI).

Takes the full-length service audio (.m4a) and an SRT that was made for it, finds
the matching *edited short video* on YouTube by title date, works out where that
video cut material out of the full audio, and writes a new SRT whose timestamps
fit the short video. This is app.py's "YouTube short clip + local full audio"
workflow, with the cut positions detected automatically instead of typed in.

How the cuts are found: the short video's audio is walked in 20 s steps, each step
matched against the full audio where the previous step said it should be. While
the video follows the original unbroken, every step lands exactly where expected.
When material was cut out, the match jumps forward in the full audio; that jump is
a cut. `--cuts auto` applies every jump found. `--cuts N` is how many cuts to
expect: if more jumps are seen the N largest are kept, if fewer a warning is
printed and the exit code is 1.

Usage:
    python shift_srt.py "20261004 주일예배.m4a" sermon.srt --cuts auto
    python shift_srt.py "20261004 주일예배.m4a" sermon.srt --cuts 2

Progress is logged to stderr; the result is printed to stdout.
Exit codes: 0 = done, 1 = could not align / fewer cuts found than expected
(an SRT is still written in the latter case), 2 = error.
"""
import argparse
import os
import re
import sys
import tempfile
import traceback

import numpy as np
from scipy.signal import fftconvolve

from find_timestamp import (
    CHANNEL_URL, CLIP_DURATION, MIN_CONFIDENCE, SAMPLE_RATE, FinderError,
    date_title_patterns, download_youtube_audio, extract_date, hms_to_seconds,
    list_channel_videos, load_audio, log, search_windows, seconds_to_hms, watch_url,
)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_WINDOW_START = 10 * 60   # 00:10:00, app.py's default start for a local full audio
SCAN_LEN = 20.0                  # seconds of the short clip matched per step
SCAN_HOP = 10.0                  # step between matches
MAX_GAP = 300.0                  # furthest a cut may jump ahead in the full audio (as in app.py)
BACK = 10.0                      # how far behind the expected spot a wide search looks
TRACK_TOL = 0.25                 # drift tolerated when following an unbroken stretch
MIN_GAP = 1.0                    # a smaller jump is drift, not a cut
MIN_RHO = 0.5                    # normalised correlation needed to trust a step (1.0 = identical)


# ── YouTube short-clip search ────────────────────────────────────────────────

def find_short_clip_video(date: str) -> dict | None:
    """
    Find the edited short video for `date` on the channel's Videos tab. Titles look
    like "2026-10-04 [series ...] ..."; the date followed by '[' is the strongest
    signal, praise/live/worship uploads are excluded.
    """
    patterns = date_title_patterns(date)
    bracket = [re.compile(rf"{re.escape(p)}\s*\[") for p in patterns]
    videos = list_channel_videos(f"{CHANNEL_URL}/videos")

    for v in videos:
        t = v["title"]
        if any(r.search(t) for r in bracket) and not any(w in t for w in ("Live recording", "찬양", "Praise")):
            return v

    matched = []
    for v in videos:
        t = v["title"]
        if any(p in t for p in patterns):
            if not any(w in t for w in ("Live recording", "Praise", "찬양", "예배", "기도회")):
                return v
            matched.append(v)
    return matched[0] if matched else None


# ── Cut detection ────────────────────────────────────────────────────────────

def match_template(window: np.ndarray, template: np.ndarray) -> tuple[int, float]:
    """
    Locate `template` in `window` by normalised cross-correlation.
    Returns (offset_in_samples, rho) with rho ≈ 1 for an identical stretch and
    ≈ 0 for unrelated audio, independent of how long the window is.
    """
    n, L = len(window), len(template)
    if n < L:
        return 0, 0.0
    window = window.astype(np.float64)
    template = template.astype(np.float64)
    corr = fftconvolve(window, template[::-1], mode="valid")            # length n - L + 1
    cs = np.concatenate(([0.0], np.cumsum(window ** 2)))
    e_local = cs[L:] - cs[:-L]                                           # energy under each lag
    e_t = float(np.sum(template ** 2))
    ncc = np.abs(corr) / np.sqrt(e_t * (e_local + 1e-3 * float(np.mean(e_local)) + 1e-9))
    peak = int(np.argmax(ncc))
    return peak, float(ncc[peak])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-12))


def localize_cut(clip: np.ndarray, full: np.ndarray, s_old: int, s_new: int, lo: int, hi: int) -> int:
    """
    Within clip samples [lo, hi), find where the alignment switches from `s_old` to
    `s_new` (full position = clip position + shift), to one-second precision: pick
    the split that makes the seconds before it agree with the old shift and the
    seconds after it agree with the new one.
    """
    starts = list(range(lo, max(lo, hi - SAMPLE_RATE) + 1, SAMPLE_RATE))

    def agree(a: int, shift: int) -> float:
        b = a + shift
        if a < 0 or b < 0 or a + SAMPLE_RATE > len(clip) or b + SAMPLE_RATE > len(full):
            return 0.0
        return _pearson(clip[a:a + SAMPLE_RATE], full[b:b + SAMPLE_RATE])

    old = np.array([agree(a, s_old) for a in starts])
    new = np.array([agree(a, s_new) for a in starts])
    # score(k) = old agreement of the first k seconds + new agreement of the rest
    scores = [old[:k].sum() + new[k:].sum() for k in range(len(starts) + 1)]
    k = int(np.argmax(scores))
    return starts[k] if k < len(starts) else starts[-1] + SAMPLE_RATE


def detect_cuts(clip: np.ndarray, full: np.ndarray, start_offset: float) -> list[dict]:
    """
    Walk the short clip and return every jump in alignment as
    {'clip_sec', 'offset_sec', 'gap_sec', 'rho'}: the clip time where the new
    stretch begins and the full-audio time it corresponds to.
    """
    L, H = int(SCAN_LEN * SAMPLE_RATE), int(SCAN_HOP * SAMPLE_RATE)
    tol, back, max_gap = int(TRACK_TOL * SAMPLE_RATE), int(BACK * SAMPLE_RATE), int(MAX_GAP * SAMPLE_RATE)
    shift = int(round(start_offset * SAMPLE_RATE))      # full position = clip position + shift
    last_good = 0                                        # clip sample of the last step confirmed at `shift`
    cuts = []

    for t in range(0, len(clip) - L + 1, H):
        tpl = clip[t:t + L]
        if np.max(np.abs(tpl)) < 1e-4:
            continue                                     # digital silence tells us nothing
        expected = t + shift

        # Fast path: still following the original unbroken?
        lo, hi = max(0, expected - tol), min(len(full), expected + L + tol)
        if hi - lo >= L:
            off, rho = match_template(full[lo:hi], tpl)
            if rho >= MIN_RHO:
                shift, last_good = lo + off - t, t
                continue

        # Wide search: did material get cut out just before this step?
        lo, hi = max(0, expected - back), min(len(full), expected + L + max_gap)
        off, rho = match_template(full[lo:hi], tpl)
        if rho < MIN_RHO:
            continue                                     # music/noise/unmatched; try the next step
        new_shift = lo + off - t
        if new_shift - shift <= int(MIN_GAP * SAMPLE_RATE):
            continue                                     # same alignment (or backwards jitter): not a cut
        cut = localize_cut(clip, full, shift, new_shift, last_good, t + L)
        cuts.append({
            "clip_sec": cut / SAMPLE_RATE,
            "offset_sec": (cut + new_shift) / SAMPLE_RATE,
            "gap_sec": (new_shift - shift) / SAMPLE_RATE,
            "rho": rho,
        })
        shift, last_good = new_shift, t
    return cuts


# ── SRT shifting (copied from app.py) ────────────────────────────────────────

def shift_srt_content_piecewise(srt_content: str, cut_points: list[float], offsets: list[float]) -> str:
    """
    Shift all timestamps in an SRT string based on piecewise linear mapping
    determined by cut points in the short clip and their offsets in the full audio.
    """
    time_pattern = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")

    def parse_srt_time(t_str: str) -> float:
        h, m, s, ms = int(t_str[0:2]), int(t_str[3:5]), int(t_str[6:8]), int(t_str[9:12])
        return h * 3600 + m * 60 + s + ms / 1000.0

    def format_srt_time(sec: float) -> str:
        if sec < 0: sec = 0.0
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int(round((sec % 1) * 1000))
        if ms >= 1000:
            s += 1
            ms -= 1000
            if s >= 60:
                m += 1
                s -= 60
                if m >= 60:
                    h += 1
                    m -= 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def map_time(t: float) -> float | None:
        # Find the appropriate segment index
        for i in range(len(offsets) - 1, -1, -1):
            if t >= offsets[i]:
                t_mapped = cut_points[i] + (t - offsets[i])
                if i < len(offsets) - 1:
                    # If this timestamp falls within the cut region (past the start of the next segment in the clip)
                    if t_mapped >= cut_points[i+1]:
                        return None
                return t_mapped
        return None

    blocks = re.split(r'\n\s*\n', srt_content.strip())

    # ── Snap Cut Points and Offsets to Subtitle Boundaries ────────────────────
    sub_boundaries = []
    for block in blocks:
        block_lines = block.split('\n')
        if not block_lines: continue
        time_line_idx = -1
        for idx, line in enumerate(block_lines):
            if '-->' in line:
                time_line_idx = idx
                break
        if time_line_idx != -1:
            match = time_pattern.search(block_lines[time_line_idx])
            if match:
                t_start = parse_srt_time(match.group(1))
                t_end = parse_srt_time(match.group(2))
                sub_boundaries.extend([t_start, t_end])

    if sub_boundaries:
        sub_boundaries = sorted(list(set(sub_boundaries)))
        refined_cut_points = [cut_points[0]]
        refined_offsets = [offsets[0]]

        for j in range(1, len(cut_points)):
            C_j = cut_points[j]
            O_j = offsets[j]
            prev_C = refined_cut_points[-1]
            prev_O = refined_offsets[-1]

            E_j = prev_O + (C_j - prev_C)
            gap_j = O_j - E_j

            best_t1 = E_j
            best_t2 = O_j
            min_loss = float('inf')

            # Search within 15 seconds of expected and actual offsets
            candidates_t1 = [t for t in sub_boundaries if abs(t - E_j) <= 15.0]
            candidates_t2 = [t for t in sub_boundaries if abs(t - O_j) <= 15.0]

            for t1 in candidates_t1:
                for t2 in candidates_t2:
                    if t2 > t1:
                        error_c1 = t1 - E_j
                        error_c2 = t2 - O_j
                        loss = abs((t2 - t1) - gap_j) + abs(error_c1 - error_c2)
                        if error_c1 < -2.0:
                            loss += 5.0 * (abs(error_c1) - 2.0)
                        if loss < min_loss:
                            min_loss = loss
                            best_t1 = t1
                            best_t2 = t2

            if min_loss < 2.0:
                refined_O_j = best_t2
                refined_C_j = prev_C + (best_t1 - prev_O)
                refined_cut_points.append(refined_C_j)
                refined_offsets.append(refined_O_j)
            else:
                refined_cut_points.append(C_j)
                refined_offsets.append(O_j)

        cut_points = refined_cut_points
        offsets = refined_offsets

    out_lines = []
    counter = 1

    for block in blocks:
        block_lines = block.split('\n')
        if not block_lines: continue

        time_line_idx = -1
        for i, line in enumerate(block_lines):
            if '-->' in line:
                time_line_idx = i
                break

        if time_line_idx != -1:
            match = time_pattern.search(block_lines[time_line_idx])
            if match:
                start_str, end_str = match.group(1), match.group(2)
                t_start = parse_srt_time(start_str)
                t_end = parse_srt_time(end_str)

                new_start_sec = map_time(t_start)
                new_end_sec = map_time(t_end)

                if new_start_sec is None and new_end_sec is None:
                    continue

                # Drop blocks whose original start falls before the segment boundary
                # to avoid residual partial lines at cut boundaries
                if new_start_sec is None:
                    continue

                if new_end_sec is None:
                    for i in range(len(offsets) - 1, -1, -1):
                        if t_start >= offsets[i]:
                            if i < len(cut_points) - 1:
                                new_end_sec = cut_points[i+1]
                            else:
                                new_end_sec = new_start_sec + (t_end - t_start)
                            break
                    if new_end_sec is None:
                        new_end_sec = new_start_sec + (t_end - t_start)

                if new_end_sec - new_start_sec <= 0.05:
                    continue

                new_start = format_srt_time(new_start_sec)
                new_end = format_srt_time(new_end_sec)

                new_time_line = block_lines[time_line_idx].replace(start_str, new_start).replace(end_str, new_end)
                new_block = [str(counter), new_time_line] + block_lines[time_line_idx+1:]
                out_lines.append('\n'.join(new_block))
                counter += 1

    return '\n\n'.join(out_lines) + '\n'


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_segments(start_offset: float, cuts: list[dict], n_cuts: int | None) -> tuple[list[float], list[float]]:
    """cut_points/offsets for the piecewise shift, keeping the `n_cuts` largest jumps (None = all)."""
    kept = cuts if n_cuts is None else sorted(cuts, key=lambda c: c["gap_sec"], reverse=True)[:n_cuts]
    kept = sorted(kept, key=lambda c: c["clip_sec"])
    return [0.0] + [c["clip_sec"] for c in kept], [start_offset] + [c["offset_sec"] for c in kept]


def _hms_arg(value: str) -> float:
    try:
        return hms_to_seconds(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid time {value!r}; use HH:MM:SS")


def _cuts_arg(value: str) -> int | None:
    """--cuts: a count, or 'auto' (returned as None) to apply every cut found."""
    if value.lower() == "auto":
        return None
    try:
        n = int(value)
    except ValueError:
        n = -1
    if n < 0:
        raise argparse.ArgumentTypeError(f"invalid value {value!r}; use 'auto' or a number 0 or more")
    return n


def run(m4a_path: str, srt_path: str, n_cuts: int | None, out_path: str | None,
        window_start: float, min_confidence: float) -> int:
    for p in (m4a_path, srt_path):
        if not os.path.isfile(p):
            raise FinderError(f"File not found: {p}")
    date = extract_date(os.path.basename(m4a_path))
    if not date:
        raise FinderError(f"No YYYYMMDD date found in {os.path.basename(m4a_path)!r}; "
                          "it is needed to match the short video's title.")
    with open(srt_path, encoding="utf-8-sig") as f:
        srt_text = f.read()

    log("🎵  Loading the full audio…")
    full = load_audio(m4a_path)
    log(f"✅  Full audio loaded ({seconds_to_hms(len(full) / SAMPLE_RATE)}).")

    log(f"🔎  Searching YouTube for the short video of {date[0:4]}-{date[4:6]}-{date[6:8]}…")
    video = find_short_clip_video(date)
    if not video:
        raise FinderError(f"No short video with the date {date} in its title was found on {CHANNEL_URL}.")
    video_url = watch_url(video["videoId"])
    log(f"✅  Matched video: {video['title']}  ({video_url})")

    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = os.path.join(tmp, f"yt_clip_{video['videoId']}.mp3")
        log("⬇️  Downloading the short video's audio…")
        download_youtube_audio(video_url, mp3_path)
        clip = load_audio(mp3_path)
    log(f"✅  Short video loaded ({seconds_to_hms(len(clip) / SAMPLE_RATE)}).")

    match, best = search_windows(full, clip[:int(CLIP_DURATION * SAMPLE_RATE)], window_start, min_confidence)
    if match is None:
        log("❌  The start of the short video was not found in the full audio"
            + (f" (best {best[1]:.2%} at {seconds_to_hms(best[0], ms=True)}, not trusted)." if best else "."))
        return 1
    start_offset = match[0]
    log(f"📍  Short video starts at {seconds_to_hms(start_offset, ms=True)} in the full audio.")

    cuts = []
    if n_cuts != 0:
        log("✂️  Scanning for cuts" + (" (auto)…" if n_cuts is None else f" (expecting {n_cuts})…"))
        cuts = detect_cuts(clip, full, start_offset)
        for c in cuts:
            log(f"    jump at short-video {seconds_to_hms(c['clip_sec'], ms=True)}: "
                f"+{c['gap_sec']:.1f}s → full audio {seconds_to_hms(c['offset_sec'], ms=True)} (rho {c['rho']:.2f})")
    cut_points, offsets = build_segments(start_offset, cuts, n_cuts)

    status = 0
    if n_cuts is None:
        log(f"✂️  Applying all {len(cuts)} cuts found." if cuts else "✂️  No cuts found; applying a single shift.")
    elif len(cuts) < n_cuts:
        log(f"⚠️  Expected {n_cuts} cuts but only found {len(cuts)}.")
        status = 1
    elif len(cuts) > n_cuts:
        log(f"⚠️  Found {len(cuts)} cuts; using the {n_cuts} with the largest gaps.")

    shifted = shift_srt_content_piecewise(srt_text, cut_points, offsets)
    if out_path is None:
        base, ext = os.path.splitext(srt_path)
        out_path = f"{base}_shifted{ext or '.srt'}"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(shifted)
    log(f"💾  Wrote {out_path}")

    print(f"Video:   {video['title']}")
    print(f"URL:     {video_url}")
    print(f"Cuts:    {len(cut_points) - 1}")
    print(f"SRT:     {out_path}")
    return status


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Shift an SRT made for the full service audio onto its edited short YouTube video.")
    parser.add_argument("audio", help="Full-length audio (.m4a); its filename must contain a YYYYMMDD date.")
    parser.add_argument("srt", help="SRT whose timestamps match the full audio.")
    parser.add_argument("--cuts", type=_cuts_arg, required=True, metavar="N|auto",
                        help="Number of cuts to expect in the short video (0 = a single shift), "
                             "or 'auto' to apply every cut found.")
    parser.add_argument("--out", metavar="PATH", help="Output SRT (default: <srt>_shifted.srt).")
    parser.add_argument("--window-start", type=_hms_arg, default=DEFAULT_WINDOW_START, metavar="HH:MM:SS",
                        help="Start of the default window searched for the video's opening (default 00:10:00).")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE, metavar="FRACTION",
                        help=f"Confidence needed to accept the opening match (default {MIN_CONFIDENCE}).")
    args = parser.parse_args(argv)

    try:
        return run(args.audio, args.srt, args.cuts, args.out, args.window_start, args.min_confidence)
    except FinderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
