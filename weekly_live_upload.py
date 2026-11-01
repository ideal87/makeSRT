"""
Weekly livestream caption upload (headless): fetch -> find timestamp -> check -> upload.

  1. From the VVC translation site, download the sermon's clip .m4a
     (e.g. "20261101va converted.m4a") and its "*_final_en.srt" (the non-TTS one,
     already shifted onto the livestream's timeline by the site).
  2. Find the full livestream on @nwtvmedia by the m4a's date and locate where the
     clip starts in it (find_timestamp.py's search).
  3. Check that the SRT's first cue sits at that timestamp. An SRT that still starts
     near 00:00 (site not shifted yet) is shifted by the found offset here; one that
     starts somewhere else entirely is refused.
  4. Upload it as the livestream's English caption track (upload_captions.py).

Meant to be run by a scheduler that may fire more than once per Sunday: a successful
upload writes downloads/live_YYYYMMDD.done with the SRT's hash, and a later run with
the same SRT stops there without touching YouTube.

Usage:
    python weekly_live_upload.py                   # today's sermon
    python weekly_live_upload.py --date 20261101
    python weekly_live_upload.py --dry-run         # everything except the upload
    python weekly_live_upload.py --force           # upload even if already done

Progress goes to stderr and downloads/live_YYYYMMDD.log; a short summary goes to stdout.
Exit codes: 0 = uploaded (or already done), 1 = timestamp not found or SRT doesn't
match it, 2 = error, 3 = not ready yet (sermon, m4a, SRT or video missing; retry later).
"""
import argparse
import datetime
import hashlib
import os
import re
import sys
import tempfile
import traceback

import requests
from googleapiclient.discovery import build

import fetch_sermon_m4a as site
import find_timestamp as ft
import upload_captions as yt

HERE = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS = os.path.join(HERE, "downloads")
SHIFT_TOLERANCE = 30.0   # seconds the first cue may sit after the clip start (seen: 1-3 s)
UNSHIFTED_MAX = 60.0     # a first cue earlier than this means the SRT is still on the clip's timeline
TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")

_log_file = None


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
    if _log_file:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log_file.write(f"[{stamp}] {msg}\n")
        _log_file.flush()


# ── SRT helpers ──────────────────────────────────────────────────────────────

def parse_cues(text: str) -> list[tuple[float, float, str]]:
    """[(start_sec, end_sec, text)] for every cue in an SRT."""
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = block.splitlines()
        for i, line in enumerate(lines):
            times = TIME_RE.findall(line)
            if "-->" in line and len(times) == 2:
                start, end = (int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000 for h, m, s, ms in times)
                cues.append((start, end, "\n".join(lines[i + 1:]).strip()))
                break
    return cues


def first_spoken_start(text: str) -> float | None:
    """Start of the earliest cue with text; the site adds a blank placeholder cue at 00:20:00."""
    starts = [start for start, _, body in parse_cues(text) if body]
    return min(starts) if starts else None


def shift_srt_text(text: str, offset: float) -> str:
    """Add `offset` seconds to every timestamp in an SRT."""
    def bump(m: re.Match) -> str:
        h, mi, s, ms = (int(g) for g in m.groups())
        total = round((h * 3600 + mi * 60 + s) * 1000 + ms + offset * 1000)
        h, rest = divmod(total, 3_600_000)
        mi, rest = divmod(rest, 60_000)
        s, ms = divmod(rest, 1000)
        return f"{h:02d}:{mi:02d}:{s:02d},{ms:03d}"
    return TIME_RE.sub(bump, text)


# ── Steps ────────────────────────────────────────────────────────────────────

def fetch_inputs(date: str) -> tuple[str, str]:
    """Download the clip .m4a and *_final_en.srt for `date`; returns their local paths."""
    email, password = os.environ.get("VVC_EMAIL"), os.environ.get("VVC_PASSWORD")
    if not email or not password:
        raise site.FetchError("VVC_EMAIL and VVC_PASSWORD must be set (environment or .env file).")
    log("🔑  Signing in to the VVC site…")
    token = site.sign_in(email, password)
    sermon = site.pick_sermon(site.list_sermons(token, site.DEFAULT_SERIES_ID), date, date)
    if not sermon:
        raise site.NotReady(f"No sermon titled {date} on the site yet.")
    log(f"📖  Sermon: {sermon['title']}")

    files = site.list_files(token, sermon["id"])
    m4a = site.pick_m4a([f for f in files if "-tts" not in f["name"].lower()])
    srt = next((f for f in files if f["name"].endswith("_final_en.srt") and f["storagePath"]), None)
    if not m4a:
        raise site.NotReady(f"'{sermon['title']}' has no .m4a uploaded yet.")
    if not srt:
        raise site.NotReady(f"'{sermon['title']}' has no *_final_en.srt exported yet.")

    os.makedirs(DOWNLOADS, exist_ok=True)
    m4a_path = os.path.join(DOWNLOADS, os.path.basename(m4a["name"]))
    if os.path.isfile(m4a_path) and (not m4a["size"] or os.path.getsize(m4a_path) == m4a["size"]):
        log(f"✅  Already downloaded: {m4a_path}")
    else:
        log(f"⬇️  Downloading {m4a['name']} ({m4a['size'] / 1e6:.1f} MB)…")
        site.download_file(token, m4a["storagePath"], m4a_path)
    # Always re-download the SRT: a re-export on the site replaces it under the same name
    srt_path = os.path.join(DOWNLOADS, os.path.basename(srt["name"]))
    site.download_file(token, srt["storagePath"], srt_path)
    log(f"✅  Saved {m4a_path} and {srt_path}")
    return m4a_path, srt_path


def find_clip_start(m4a_path: str, date: str) -> tuple[dict, float, float]:
    """Find the livestream for `date` and where the clip starts in it: (video, offset_sec, confidence)."""
    clip = ft.load_audio(m4a_path, duration=ft.CLIP_DURATION)
    if len(clip) == 0:
        raise ft.FinderError(f"Could not read any audio from {m4a_path}")
    video = ft.find_full_audio_video(date)
    if not video:
        raise site.NotReady(f"No livestream with {date} in its title on {ft.CHANNEL_URL} yet.")
    log(f"📺  Livestream: {video['title']}  ({ft.watch_url(video['videoId'])})")

    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = os.path.join(tmp, f"yt_full_{video['videoId']}.mp3")
        log("⬇️  Downloading the livestream's audio (a few minutes)…")
        ft.download_youtube_audio(ft.watch_url(video["videoId"]), mp3_path)
        full_audio = ft.load_audio(mp3_path)
    log(f"🎵  Livestream audio is {ft.seconds_to_hms(len(full_audio) / ft.SAMPLE_RATE)} long.")

    match, best = ft.search_windows(full_audio, clip, ft.DEFAULT_WINDOW_START, ft.MIN_CONFIDENCE)
    if match is None:
        hint = f" Best guess {ft.seconds_to_hms(best[0], ms=True)} at {best[1]:.2%}." if best else ""
        raise ValueError(f"No confident match for the clip in the livestream.{hint}")
    return video, match[0], match[1]


def upload(srt_text: str, video_id: str, dry_run: bool) -> str:
    youtube = build("youtube", "v3", credentials=yt.load_credentials(yt.DEFAULT_TOKEN))
    existing = yt.find_caption_track(youtube, video_id)
    if dry_run:
        log(f"🔍  Dry run: would {'REPLACE track ' + existing['id'] if existing else 'add a new English track'}"
            " on the livestream. Nothing was changed.")
        return ""
    log(f"📤  Uploading ({'replacing' if existing else 'adding'} the English track)…")
    caption_id = yt.upload_caption_track(youtube, video_id, srt_text.encode("utf-8"))
    log(f"✅  Uploaded. Caption ID: {caption_id}")
    return caption_id


# ── CLI ──────────────────────────────────────────────────────────────────────

def run(date: str, dry_run: bool, force: bool) -> int:
    log(f"=== Livestream caption upload for {date}{' (dry run)' if dry_run else ''}")
    m4a_path, srt_path = fetch_inputs(date)
    with open(srt_path, encoding="utf-8-sig") as f:
        srt_text = f.read()
    if not parse_cues(srt_text):
        raise ValueError(f"{srt_path} has no subtitle cues.")

    done_path = os.path.join(DOWNLOADS, f"live_{date}.done")
    srt_hash = hashlib.sha256(srt_text.encode("utf-8")).hexdigest()
    if not force and os.path.isfile(done_path):
        with open(done_path, encoding="utf-8") as f:
            if f.readline().strip() == srt_hash:
                log("✅  This exact SRT was already uploaded for this date; nothing to do.")
                print(f"Already done for {date} (same SRT as the last upload).")
                return 0
        log("🔁  The site's SRT changed since the last upload; uploading again.")

    video, offset, confidence = find_clip_start(m4a_path, date)
    log(f"⏱️  Clip starts at {ft.seconds_to_hms(offset, ms=True)} (confidence {confidence:.2%}).")

    first = first_spoken_start(srt_text)
    if first is None:
        raise ValueError(f"{srt_path} has no cue with text.")
    note = "already shifted"
    if first < UNSHIFTED_MAX and offset > UNSHIFTED_MAX:
        srt_text = shift_srt_text(srt_text, offset)
        note = f"shifted here by +{ft.seconds_to_hms(offset, ms=True)} (the site's SRT was not shifted)"
        shifted_path = srt_path[:-4] + "_live.srt"
        with open(shifted_path, "w", encoding="utf-8") as f:
            f.write(srt_text)
        log(f"↪️  First cue was at {ft.seconds_to_hms(first, ms=True)}; {note}. Saved {shifted_path}")
    elif not (offset - 1 <= first <= offset + SHIFT_TOLERANCE):
        log(f"❌  The SRT's first cue is at {ft.seconds_to_hms(first, ms=True)}, but the clip starts at "
            f"{ft.seconds_to_hms(offset, ms=True)}. Re-export the SRT on the site with that offset.")
        return 1
    else:
        log(f"✅  SRT's first cue {ft.seconds_to_hms(first, ms=True)} matches the clip start.")

    caption_id = upload(srt_text, video["videoId"], dry_run)
    if not dry_run:
        with open(done_path, "w", encoding="utf-8") as f:
            f.write(f"{srt_hash}\n{video['videoId']}\n{caption_id}\n")

    print(f"Video:      {video['title']}")
    print(f"URL:        {ft.watch_url(video['videoId'])}")
    print(f"Timestamp:  {ft.seconds_to_hms(offset, ms=True)} (confidence {confidence:.2%})")
    print(f"SRT:        {os.path.basename(srt_path)}, {len(parse_cues(srt_text))} cues, {note}")
    print(f"Upload:     {'dry run, nothing changed' if dry_run else 'caption ID ' + caption_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    global _log_file
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    site.load_dotenv(os.path.join(HERE, ".env"))

    parser = argparse.ArgumentParser(description="Fetch, check and upload the sermon SRT to the livestream.")
    parser.add_argument("--date", default=datetime.date.today().strftime("%Y%m%d"), metavar="YYYYMMDD",
                        help="Sermon date (default: today).")
    parser.add_argument("--dry-run", action="store_true", help="Do everything except the upload.")
    parser.add_argument("--force", action="store_true", help="Upload even if this SRT was already uploaded.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{8}", args.date):
        parser.error("--date must be YYYYMMDD")

    os.makedirs(DOWNLOADS, exist_ok=True)
    _log_file = open(os.path.join(DOWNLOADS, f"live_{args.date}.log"), "a", encoding="utf-8")
    try:
        return run(args.date, args.dry_run, args.force)
    except site.NotReady as e:
        log(f"⏳  Not ready: {e}")
        return 3
    except requests.RequestException as e:   # before ValueError: requests' JSON errors are both
        log(f"error: network problem: {e}")
        return 2
    except ValueError as e:
        log(f"❌  {e}")
        return 1
    except (site.FetchError, ft.FinderError, yt.UploadError) as e:
        log(f"error: {e}")
        return 2
    except Exception as e:
        if "invalid_grant" in str(e):
            log("error: YouTube login was revoked or has expired. Run `python auth.py` again.")
        else:
            log("error: " + traceback.format_exc())
        return 2
    finally:
        _log_file.close()


if __name__ == "__main__":
    sys.exit(main())
