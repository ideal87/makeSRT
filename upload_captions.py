"""
Upload an SRT to a YouTube video as its English caption track (headless).

Same behaviour as app.py's "Upload to YouTube": any existing caption track with
language "en" and name "English" is replaced. Uses the OAuth login in token.json
(created once by `python auth.py`); an expired access token is refreshed and
saved automatically, so scheduled runs need no interaction.

Usage:
    python upload_captions.py sermon_shifted.srt https://www.youtube.com/watch?v=VIDEO_ID
    python upload_captions.py sermon_shifted.srt VIDEO_ID --dry-run

--dry-run checks the SRT, the login and the video's current caption tracks, and
reports what would be replaced, without changing anything on YouTube.

Exit codes: 0 = done, 2 = error.
"""
import argparse
import io
import os
import re
import sys
import traceback

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]   # same as auth.py
LANGUAGE = "en"
NAME = "English"
DEFAULT_TOKEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token.json")


class UploadError(Exception):
    """A failure the user should see as a plain message rather than a traceback."""


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def extract_video_id(url: str) -> str | None:
    """Pull the video ID out of a YouTube URL (a bare 11-character ID also works)."""
    url = url.strip()
    m = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/)([\w-]+)", url)
    if m:
        return m.group(1)
    return url if re.fullmatch(r"[\w-]{11}", url) else None


def load_credentials(token_path: str) -> Credentials:
    """Load token.json, refreshing (and saving) it if the access token has expired."""
    if not os.path.isfile(token_path):
        raise UploadError(f"{token_path} not found. Run `python auth.py` once to log in to YouTube.")
    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds.valid:
        if not (creds.expired and creds.refresh_token):
            raise UploadError("YouTube login is invalid and cannot be refreshed. Run `python auth.py` again.")
        try:
            creds.refresh(Request())
        except Exception as e:
            if "invalid_grant" in str(e):
                raise UploadError("YouTube login was revoked or has expired. Run `python auth.py` again.")
            raise
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        log("🔄  Refreshed the YouTube access token.")
    return creds


def find_caption_track(youtube, video_id: str) -> dict | None:
    """The video's existing caption track with our language and name, if any."""
    items = youtube.captions().list(part="snippet", videoId=video_id).execute().get("items", [])
    for item in items:
        snippet = item.get("snippet", {})
        if snippet.get("language") == LANGUAGE and snippet.get("name") == NAME:
            return item
    return None


def upload_caption_track(youtube, video_id: str, srt_bytes: bytes) -> str:
    """Replace the video's English caption track with `srt_bytes`; returns the new track ID."""
    existing = find_caption_track(youtube, video_id)
    if existing:
        youtube.captions().delete(id=existing["id"]).execute()
    media = MediaIoBaseUpload(io.BytesIO(srt_bytes), mimetype="application/octet-stream", resumable=True)
    response = youtube.captions().insert(
        part="snippet",
        body={"snippet": {"videoId": video_id, "language": LANGUAGE, "name": NAME, "isDraft": False}},
        media_body=media,
    ).execute()
    return response["id"]


def run(srt_path: str, video: str, token_path: str, dry_run: bool) -> int:
    if not os.path.isfile(srt_path):
        raise UploadError(f"File not found: {srt_path}")
    video_id = extract_video_id(video)
    if not video_id:
        raise UploadError(f"Could not extract a YouTube video ID from {video!r}.")

    # Validate before touching YouTube: replacing a good track with an empty file would be a loss
    with open(srt_path, encoding="utf-8-sig") as f:
        text = f.read()
    cues = len(re.findall(r"-->", text))
    if cues == 0:
        raise UploadError(f"{srt_path} contains no subtitle cues; refusing to upload.")
    log(f"📄  {os.path.basename(srt_path)}: {cues} cues.")

    creds = load_credentials(token_path)
    youtube = build("youtube", "v3", credentials=creds)

    if dry_run:
        existing = find_caption_track(youtube, video_id)
        log(f"🔍  Dry run for video {video_id}: "
            + (f"would REPLACE the existing '{NAME}' ({LANGUAGE}) track {existing['id']}." if existing
               else f"no existing '{NAME}' ({LANGUAGE}) track; would add a new one.")
            + " Nothing was changed.")
        return 0

    log(f"📤  Uploading to https://www.youtube.com/watch?v={video_id} …")
    # Normalise to UTF-8 without BOM, as app.py does
    caption_id = upload_caption_track(youtube, video_id, text.encode("utf-8"))
    log(f"✅  Uploaded. Caption ID: {caption_id}")
    print(caption_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Upload an SRT to YouTube as the video's English captions.")
    parser.add_argument("srt", help="SRT file to upload.")
    parser.add_argument("video", help="YouTube video URL or ID.")
    parser.add_argument("--token", default=DEFAULT_TOKEN, metavar="PATH",
                        help="OAuth token file made by auth.py (default: token.json next to this script).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check everything and report what would be replaced, without uploading.")
    args = parser.parse_args(argv)

    try:
        return run(args.srt, args.video, args.token, args.dry_run)
    except UploadError as e:
        log(f"error: {e}")
        return 2
    except Exception as e:
        if "invalid_grant" in str(e):
            log("error: YouTube login was revoked or has expired. Run `python auth.py` again.")
        else:
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
