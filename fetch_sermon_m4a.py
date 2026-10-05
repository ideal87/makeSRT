"""
Download the latest sermon's .m4a from the VVC translation site (headless).

The site is a Firebase app, so this talks to Firebase's REST endpoints directly
instead of driving a browser: sign in with a dedicated email/password account,
find the newest "YYYYMMDD ..." sermon in the series, and download its .m4a from
Cloud Storage. Signing in on every run means there is no session to expire.

Credentials come from the environment or a git-ignored `.env` file next to this
script (see .env.example):
    VVC_EMAIL=...
    VVC_PASSWORD=...

Usage:
    python fetch_sermon_m4a.py                  # newest sermon dated today or earlier
    python fetch_sermon_m4a.py --date 20261004  # a specific sermon
    python fetch_sermon_m4a.py --out D:\\sermons
    python fetch_sermon_m4a.py --srt            # also the *_final_en_TTS.srt

Prints the downloaded file's path to stdout (with --srt, the SRT's path on a
second line); progress goes to stderr. The SRT is always re-downloaded, since a
re-export on the site replaces it under the same name.
Exit codes: 0 = file(s) ready, 2 = error, 3 = not ready yet (no sermon for the
date, or no .m4a / TTS SRT on it) — a weekly scheduler can simply retry later.
"""
import argparse
import datetime
import os
import re
import sys
import time
import traceback
import urllib.parse

import requests

# ── Constants (public web-app config, visible in the site's own JS) ───────────
API_KEY = "AIzaSyBAPFuwscWWolkfw_H_MQSywmsehJIRekk"
PROJECT_ID = "vvc-translation-88034"
STORAGE_BUCKET = "vvc-translation-88034.firebasestorage.app"
FIRESTORE_DB = "vvc-translation"             # named database, not "(default)"
DEFAULT_SERIES_ID = "UkqS5VZ7SCTZ2VG05mbH"   # "Sunday Sermons"

FIRESTORE = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/{FIRESTORE_DB}/documents"
TIMEOUT = 60


class FetchError(Exception):
    """A failure the user should see as a plain message rather than a traceback."""


class NotReady(Exception):
    """The sermon or its .m4a isn't on the site yet; try again later."""


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_dotenv(path: str) -> None:
    """Minimal .env reader: KEY=VALUE lines; real environment variables win."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


# ── Firebase REST ────────────────────────────────────────────────────────────

def sign_in(email: str, password: str) -> str:
    """Email/password sign-in; returns a Firebase ID token (valid for an hour)."""
    r = requests.post(
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword",
        params={"key": API_KEY},
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=TIMEOUT,
    )
    if r.status_code != 200:
        reason = r.json().get("error", {}).get("message", r.text[:200])
        raise FetchError(f"Sign-in failed ({reason}). Check VVC_EMAIL / VVC_PASSWORD.")
    return r.json()["idToken"]


def _str(fields: dict, name: str) -> str:
    return fields.get(name, {}).get("stringValue", "")


def list_sermons(token: str, series_id: str) -> list[dict]:
    """All sermons in a series as [{'id', 'title'}] (Firestore runQuery)."""
    r = requests.post(
        f"{FIRESTORE}:runQuery",
        headers={"Authorization": f"Bearer {token}"},
        json={"structuredQuery": {
            "from": [{"collectionId": "sermons"}],
            "where": {"fieldFilter": {
                "field": {"fieldPath": "seriesId"}, "op": "EQUAL",
                "value": {"stringValue": series_id},
            }},
            "limit": 500,
        }},
        timeout=TIMEOUT,
    )
    if r.status_code != 200:
        raise FetchError(f"Could not list sermons (HTTP {r.status_code}): {r.text[:300]}")
    return [
        {"id": row["document"]["name"].rsplit("/", 1)[-1],
         "title": _str(row["document"].get("fields", {}), "title")}
        for row in r.json() if "document" in row
    ]


def list_files(token: str, sermon_id: str) -> list[dict]:
    """Files attached to a sermon as [{'name', 'type', 'storagePath', 'size'}]."""
    files, page_token = [], None
    while True:
        params = {"pageSize": 300}
        if page_token:
            params["pageToken"] = page_token
        r = requests.get(f"{FIRESTORE}/sermons/{sermon_id}/files",
                         headers={"Authorization": f"Bearer {token}"}, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            raise FetchError(f"Could not list files (HTTP {r.status_code}): {r.text[:300]}")
        body = r.json()
        for doc in body.get("documents", []):
            f = doc.get("fields", {})
            files.append({
                "name": _str(f, "name"),
                "type": _str(f, "type"),
                "storagePath": _str(f, "storagePath"),
                "size": int(f.get("sizeBytes", {}).get("integerValue", 0) or 0),
            })
        page_token = body.get("nextPageToken")
        if not page_token:
            return files


def download_file(token: str, storage_path: str, dest: str) -> None:
    """Stream a Cloud Storage object to `dest` (same request the site's button makes)."""
    url = (f"https://firebasestorage.googleapis.com/v0/b/{STORAGE_BUCKET}/o/"
           f"{urllib.parse.quote(storage_path, safe='')}")
    tmp = dest + ".part"
    with requests.get(url, params={"alt": "media"}, headers={"Authorization": f"Firebase {token}"},
                      stream=True, timeout=TIMEOUT) as r:
        if r.status_code != 200:
            raise FetchError(f"Download failed (HTTP {r.status_code}): {r.text[:300]}")
        with open(tmp, "wb") as out:
            for chunk in r.iter_content(chunk_size=1 << 20):
                out.write(chunk)
    os.replace(tmp, dest)   # never leave a half-written file under the real name


# ── Selection logic ──────────────────────────────────────────────────────────

def title_date(title: str) -> str | None:
    """'20261004 Sunday Sermon' -> '20261004' (None if the title doesn't start with a date)."""
    m = re.match(r"\s*(\d{8})\b", title)
    return m.group(1) if m else None


def pick_sermon(sermons: list[dict], date: str | None, today: str) -> dict | None:
    """The sermon for `date`, or the newest one dated `today` or earlier."""
    dated = [(title_date(s["title"]), s) for s in sermons]
    dated = [(d, s) for d, s in dated if d]
    if date:
        dated = [(d, s) for d, s in dated if d == date]
    else:
        dated = [(d, s) for d, s in dated if d <= today]
    if not dated:
        return None
    # Newest date wins; if two sermons share a date, prefer the shorter (unsuffixed) title
    return max(dated, key=lambda ds: (ds[0], -len(ds[1]["title"])))[1]


def pick_m4a(files: list[dict]) -> dict | None:
    m4as = [f for f in files if f["name"].lower().endswith(".m4a") and f["storagePath"]]
    return max(m4as, key=lambda f: f["size"]) if m4as else None


def pick_tts_srt(files: list[dict]) -> dict | None:
    """The 'Final Outputs' English TTS SRT, e.g. 20261004_Sunday_Sermon_final_en_TTS.srt."""
    srts = [f for f in files if f["name"].endswith("_final_en_TTS.srt") and f["storagePath"]]
    return srts[0] if srts else None


# ── CLI ──────────────────────────────────────────────────────────────────────

def run(date: str | None, series_id: str, out_dir: str, want_srt: bool = False) -> list[str]:
    email, password = os.environ.get("VVC_EMAIL"), os.environ.get("VVC_PASSWORD")
    if not email or not password:
        raise FetchError("VVC_EMAIL and VVC_PASSWORD must be set (environment or .env file).")

    log("🔑  Signing in…")
    token = sign_in(email, password)

    today = datetime.date.today().strftime("%Y%m%d")
    sermons = list_sermons(token, series_id)
    log(f"📚  {len(sermons)} sermons in the series.")
    sermon = pick_sermon(sermons, date, today)
    if not sermon:
        raise NotReady(f"No sermon titled {date} yet." if date
                       else f"No sermon dated {today} or earlier found.")
    log(f"📖  Sermon: {sermon['title']}")

    files = list_files(token, sermon["id"])
    m4a = pick_m4a(files)
    if not m4a:
        raise NotReady(f"'{sermon['title']}' has no .m4a uploaded yet.")
    srt = pick_tts_srt(files) if want_srt else None
    if want_srt and not srt:
        raise NotReady(f"'{sermon['title']}' has no *_final_en_TTS.srt exported yet.")

    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, os.path.basename(m4a["name"]))
    if os.path.isfile(dest) and (not m4a["size"] or os.path.getsize(dest) == m4a["size"]):
        log(f"✅  Already downloaded: {dest}")
    else:
        log(f"⬇️  Downloading {m4a['name']} ({m4a['size'] / 1e6:.1f} MB)…")
        t0 = time.time()
        download_file(token, m4a["storagePath"], dest)
        log(f"✅  Saved to {dest} ({time.time() - t0:.0f}s).")
    paths = [dest]

    if srt:
        # Always re-download: re-exporting on the site replaces the file under the same name
        srt_dest = os.path.join(out_dir, os.path.basename(srt["name"]))
        download_file(token, srt["storagePath"], srt_dest)
        log(f"✅  Saved {srt['name']} to {srt_dest}.")
        paths.append(srt_dest)
    return paths


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")   # Korean filenames

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

    parser = argparse.ArgumentParser(description="Download the latest sermon .m4a from the VVC translation site.")
    parser.add_argument("--date", metavar="YYYYMMDD", help="Sermon date to fetch (default: newest dated today or earlier).")
    parser.add_argument("--series", default=DEFAULT_SERIES_ID, help="Series ID (default: Sunday Sermons).")
    parser.add_argument("--out", default="downloads", help="Output directory (default: ./downloads).")
    parser.add_argument("--srt", action="store_true",
                        help="Also download the sermon's *_final_en_TTS.srt (printed on a second line).")
    args = parser.parse_args(argv)
    if args.date and not re.fullmatch(r"\d{8}", args.date):
        parser.error("--date must be YYYYMMDD")

    try:
        for path in run(args.date, args.series, args.out, args.srt):
            print(path)
        return 0
    except NotReady as e:
        log(f"⏳  Not ready: {e}")
        return 3
    except FetchError as e:
        log(f"error: {e}")
        return 2
    except requests.RequestException as e:
        log(f"error: network problem: {e}")
        return 2
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
