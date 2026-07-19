import streamlit as st
import subprocess
import sys
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

def extract_video_id(url: str) -> str | None:
    """Pull the video ID out of a YouTube URL for building timestamped links."""
    m = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/)([\w-]+)", url)
    return m.group(1) if m else None


def sanitize_youtube_url(url: str) -> str:
    """Extract and return a clean YouTube watch URL."""
    url = url.strip()
    video_id = extract_video_id(url)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return url


def download_youtube_audio(url: str, output_path: str) -> None:
    """Download full audio from a YouTube URL as a 16 kHz mono MP3 CBR."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--js-runtimes", "node",
        "--extractor-args", "youtube:player_client=default,-android_sdkless",
        "-x",                          # extract audio
        "--audio-format", "mp3",
        "--postprocessor-args",
        f"ffmpeg:-ar {SAMPLE_RATE} -ac 1 -b:a 128k",   # resample and set constant bitrate
        "-o", output_path,
    ]
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (code {result.returncode}):\n{result.stderr}"
        )


def upload_caption_track(youtube, video_id: str, language: str, name: str, media) -> str:
    """
    Uploads a caption track to a YouTube video.
    If a caption track with the same language and name already exists, deletes it first.
    Returns the new caption track ID.
    """
    # List existing caption tracks for the video
    captions_list = youtube.captions().list(part="snippet", videoId=video_id).execute()
    
    # Check if a track with the same language and name exists
    for item in captions_list.get('items', []):
        snippet = item.get('snippet', {})
        if snippet.get('language') == language and snippet.get('name') == name:
            # Delete the existing caption track
            youtube.captions().delete(id=item['id']).execute()
            break
            
    # Insert the new caption track
    request = youtube.captions().insert(
        part="snippet",
        body={
            "snippet": {
                "videoId": video_id,
                "language": language,
                "name": name,
                "isDraft": False
            }
        },
        media_body=media
    )
    response = request.execute()
    return response['id']


def check_youtube_url(url: str) -> tuple[bool, str]:
    """Check if a YouTube URL is valid and accessible using yt-dlp."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--simulate",
        "--print", "title",
        "--no-playlist",
        "--js-runtimes", "node",
        "--extractor-args", "youtube:player_client=default,-android_sdkless",
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode == 0:
        lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        title = lines[-1] if lines else "Unknown Title"
        return True, title
    else:
        err = result.stderr.strip()
        m = re.search(r"ERROR:\s*(.*)", err)
        error_msg = m.group(1) if m else "Unknown error occurred"
        return False, error_msg

def fetch_matching_youtube_url(date_str: str, target_type: str = "full_audio") -> str | None:
    """
    Search NWTVMedia for a video or stream matching the YYYYMMDD date prefix.
    If target_type is "short_clip", searches for videos.
    If target_type is "full_audio", searches for streams, falling back to videos.
    """
    if len(date_str) != 8 or not date_str.isdigit():
        return None
    
    y, m, d = date_str[0:4], date_str[4:6], date_str[6:8]

    # Generate matching pattern candidates
    dash_match_ymd = f"{y}-{m}-{d}"
    dot_match_ymd = f"{y}.{m}.{d}"
    slash_match_mdy = f"{m}/{d}/{y}"
    dash_match_mdy = f"{m}-{d}-{y}"
    dot_match_mdy = f"{m}.{d}.{y}"
    
    import datetime
    try:
        dt = datetime.datetime.strptime(date_str, "%Y%m%d")
        eng_date = dt.strftime("%B %d, %Y").replace(" 0", " ")
        eng_date_short = dt.strftime("%b %d, %Y").replace(" 0", " ")
    except Exception:
        eng_date = "---"
        eng_date_short = "---"

    patterns = [dash_match_ymd, dot_match_ymd, slash_match_mdy, dash_match_mdy, dot_match_mdy, eng_date, eng_date_short]
    patterns = [p for p in patterns if p and p != "---"]

    def title_matches(title: str) -> bool:
        for p in patterns:
            if p in title:
                return True
        return False

    def get_channel_videos_yt_dlp(channel_url: str, limit: int = 30) -> list[dict]:
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--flat-playlist",
            "--playlist-end", str(limit),
            "--extractor-args", "youtube:player_client=default,-android_sdkless",
            "--print", "%(title)s | %(id)s",
            channel_url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        videos = []
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    title = parts[0].strip()
                    video_id = parts[-1].strip()
                    videos.append({"title": title, "videoId": video_id})
        return videos

    if target_type == "short_clip":
        try:
            videos = get_channel_videos_yt_dlp("https://www.youtube.com/@nwtvmedia/videos", limit=30)
            matched_videos = []
            for v in videos:
                title = v.get('title', '')
                if title_matches(title):
                    # Prefer videos that look like sermons/short clips
                    if "Live recording" not in title and "Praise" not in title and "예배" not in title and "기도회" not in title:
                        return f"https://www.youtube.com/watch?v={v['videoId']}"
                    matched_videos.append(f"https://www.youtube.com/watch?v={v['videoId']}")
            if matched_videos:
                return matched_videos[-1]
        except Exception:
            pass
        return None

    else:
        # target_type == "full_audio"
        matched_streams = []
        try:
            streams = get_channel_videos_yt_dlp("https://www.youtube.com/@nwtvmedia/streams", limit=30)
            for v in streams:
                title = v.get('title', '')
                if title_matches(title):
                    matched_streams.append(f"https://www.youtube.com/watch?v={v['videoId']}")
        except Exception:
            pass

        if matched_streams:
            return matched_streams[-1]

        matched_videos = []
        try:
            videos = get_channel_videos_yt_dlp("https://www.youtube.com/@nwtvmedia/videos", limit=30)
            for v in videos:
                title = v.get('title', '')
                if title_matches(title):
                    matched_videos.append(f"https://www.youtube.com/watch?v={v['videoId']}")
        except Exception:
            pass

        if matched_videos:
            return matched_videos[-1]

    return None

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


def find_top_offsets(full_audio: np.ndarray, clip: np.ndarray, num_candidates: int = 3, min_distance_seconds: float = 30.0):
    """
    Use normalised cross-correlation to find up to `num_candidates` peak positions.
    Returns a list of tuples: (offset_seconds, confidence) sorted by confidence descending.
    """
    # Normalise both signals
    full_audio_norm = full_audio / (np.max(np.abs(full_audio)) + 1e-9)
    clip_norm = clip / (np.max(np.abs(clip)) + 1e-9)

    # Cross-correlate using FFT
    correlation = fftconvolve(full_audio_norm, clip_norm[::-1], mode="full")
    corr_abs = np.abs(correlation)

    # Energy calculations for confidence normalization
    energy = np.sqrt(np.sum(clip_norm ** 2) * np.sum(full_audio_norm ** 2)) + 1e-9

    candidates = []
    min_distance_samples = int(min_distance_seconds * SAMPLE_RATE)

    for _ in range(num_candidates):
        peak_index = np.argmax(corr_abs)
        peak_val = corr_abs[peak_index]
        if peak_val <= 0.0:
            break
        
        # Convert index → offset
        offset_samples = peak_index - (len(clip_norm) - 1)
        offset_seconds = max(0.0, offset_samples / SAMPLE_RATE)
        confidence = peak_val / energy
        
        candidates.append((offset_seconds, confidence))
        
        # Suppress the neighborhood of this peak to find other distinct peaks
        start_suppress = max(0, peak_index - min_distance_samples)
        end_suppress = min(len(corr_abs), peak_index + min_distance_samples)
        corr_abs[start_suppress:end_suppress] = 0.0
        
        if np.max(corr_abs) <= 0.0:
            break
            
    return candidates


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


def shift_srt_content(srt_content: str, shift_seconds: float) -> str:
    """Shift all timestamps in an SRT string backwards by shift_seconds."""
    import re
    time_pattern = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")
    
    def shift_time(t_str: str, offset: float) -> str | None:
        h, m, s, ms = int(t_str[0:2]), int(t_str[3:5]), int(t_str[6:8]), int(t_str[9:12])
        sec = h * 3600 + m * 60 + s + ms / 1000.0 - offset
        if sec < 0: return None
        return f"{int(sec//3600):02d}:{int((sec%3600)//60):02d}:{int(sec%60):02d},{int(round((sec%1)*1000)):03d}"

    blocks = re.split(r'\n\s*\n', srt_content.strip())
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
                new_start = shift_time(start_str, shift_seconds)
                new_end = shift_time(end_str, shift_seconds)
                
                if new_end is None:
                    continue
                # Drop blocks whose original start is before the cut point
                # (i.e. shifted start is negative) to avoid residual lines
                if new_start is None:
                    continue
                
                new_time_line = block_lines[time_line_idx].replace(start_str, new_start).replace(end_str, new_end)
                new_block = [str(counter), new_time_line] + block_lines[time_line_idx+1:]
                out_lines.append('\n'.join(new_block))
                counter += 1
                
    return '\n\n'.join(out_lines) + '\n'


def shift_srt_content_piecewise(srt_content: str, cut_points: list[float], offsets: list[float]) -> str:
    """
    Shift all timestamps in an SRT string based on piecewise linear mapping
    determined by cut points in the short clip and their offsets in the full audio.
    """
    import re
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


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Audio Timestamp Finder", page_icon="🎯")

# ── Sidebar: YouTube Authentication Status ────────────────────────────────────
with st.sidebar:
    st.subheader("🔑 YouTube Authentication")
    
    import os
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    
    SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
    creds = None
    status = "NOT_AUTHENTICATED"
    
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            if creds:
                if creds.valid:
                    status = "VALID"
                elif creds.expired:
                    status = "EXPIRED"
        except Exception:
            status = "ERROR"
            
    if status == "VALID":
        st.success("🟢 Connected to YouTube")
        st.caption("Your API credentials are valid and active.")
        
        if st.button("🔄 Force Refresh Token", key="sb_refresh", use_container_width=True):
            try:
                with st.spinner("Refreshing credentials..."):
                    creds.refresh(Request())
                    with open('token.json', 'w') as token_file:
                        token_file.write(creds.to_json())
                st.success("Token refreshed successfully!")
                st.rerun()
            except Exception as e:
                if "invalid_grant" in str(e):
                    if os.path.exists('token.json'):
                        os.remove('token.json')
                    st.error("❌ Your session has been revoked or expired. Please click 'Authenticate' to log in again.")
                    st.rerun()
                else:
                    st.error(f"Failed to refresh: {e}")
                
        if st.button("❌ Disconnect Account", key="sb_disconnect", use_container_width=True):
            if os.path.exists('token.json'):
                os.remove('token.json')
            st.success("Account disconnected.")
            st.rerun()
            
    elif status == "EXPIRED":
        st.warning("🟡 Credentials Expired")
        st.caption("Your session has expired. Renew to continue uploading captions.")
        
        if st.button("🔄 Renew Credentials", key="sb_renew", use_container_width=True):
            try:
                with st.spinner("Renewing credentials..."):
                    creds.refresh(Request())
                    with open('token.json', 'w') as token_file:
                        token_file.write(creds.to_json())
                st.success("Credentials renewed successfully!")
                st.rerun()
            except Exception as e:
                if "invalid_grant" in str(e):
                    if os.path.exists('token.json'):
                        os.remove('token.json')
                    st.error("❌ Your session has been revoked or expired. Please click 'Authenticate' to log in again.")
                    st.rerun()
                else:
                    st.error(f"Failed to renew: {e}")
                
        if st.button("❌ Disconnect Account", key="sb_disconnect_expired", use_container_width=True):
            if os.path.exists('token.json'):
                os.remove('token.json')
            st.success("Account disconnected.")
            st.rerun()
            
    else:
        st.error("🔴 Disconnected")
        st.caption("No valid YouTube account linked.")
        
        if not os.path.exists('client_secret.json'):
            st.info("💡 To authenticate, please place your `client_secret.json` credentials file in the project directory.")
        else:
            if st.button("🔑 Authenticate", key="sb_auth", use_container_width=True):
                try:
                    with st.spinner("Complete auth in browser..."):
                        flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
                        auth_url, _ = flow.authorization_url(prompt='select_account consent', access_type='offline')
                        
                        st.info("If the auth page does not open automatically, visit this URL:")
                        st.code(auth_url)
                        
                        creds = flow.run_local_server(
                            port=8090, 
                            timeout_seconds=300, 
                            authorization_prompt_kwargs={'prompt': 'select_account consent', 'access_type': 'offline'}
                        )
                        with open('token.json', 'w') as token_file:
                            token_file.write(creds.to_json())
                    st.success("Authenticated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Authentication failed: {e}")

if "_pending_clip_src" in st.session_state:
    st.session_state.clip_src = st.session_state._pending_clip_src
    st.session_state.clip_url = st.session_state._pending_clip_url
    if st.session_state.clip_src == "YouTube URL":
        st.session_state.window_start_input = "00:10:00"
    del st.session_state["_pending_clip_src"]
    del st.session_state["_pending_clip_url"]

if "window_start_input" not in st.session_state:
    st.session_state["window_start_input"] = "03:30:00"

if "trigger_find" not in st.session_state:
    st.session_state["trigger_find"] = False

st.title("🎯 Audio Timestamp Finder")
st.markdown(
    "Select your sources for the **Short Clip** and the **Full Audio**, then click find! "
    "The app will locate the exact timestamp where the short clip begins within the full audio."
)

# ── Ad-hoc YouTube Caption Upload Expander ────────────────────────────────────
with st.expander("📤 Ad-hoc YouTube Caption Upload (Skip Audio Finding)"):
    st.markdown(
        "Directly upload an existing SRT subtitle file to a YouTube video. "
        "This skips the audio timestamp search and shifts."
    )
    adhoc_url = st.text_input("YouTube Video URL", key="adhoc_url", placeholder="https://www.youtube.com/watch?v=...")
    adhoc_srt_file = st.file_uploader("Upload SRT Subtitle File", type=["srt"], key="adhoc_srt")
    
    col_adhoc_lang, col_adhoc_name = st.columns(2)
    with col_adhoc_lang:
        adhoc_lang = st.selectbox("Caption Language", ["en", "ko"], index=0, key="adhoc_lang")
    with col_adhoc_name:
        adhoc_caption_name = st.text_input("Caption Track Name", value="English" if adhoc_lang == "en" else "English", key="adhoc_caption_name")
        
    adhoc_upload_btn = st.button("📤 Upload Ad-hoc Subtitles", type="primary", key="adhoc_upload_btn", use_container_width=True)
    if adhoc_upload_btn:
        import os
        if not os.path.exists('token.json'):
            st.error("❌ Not authenticated. Please connect your YouTube account in the sidebar first.")
        elif not adhoc_url:
            st.error("❌ Please enter a YouTube video URL.")
        elif not adhoc_srt_file:
            st.error("❌ Please upload an SRT file.")
        else:
            try:
                from google.oauth2.credentials import Credentials
                from google.auth.transport.requests import Request
                from googleapiclient.discovery import build
                from googleapiclient.http import MediaIoBaseUpload
                import io

                SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
                
                if creds and creds.expired and creds.refresh_token:
                    with st.spinner("Refreshing credentials..."):
                        creds.refresh(Request())
                        with open('token.json', 'w') as token_file:
                            token_file.write(creds.to_json())
                            
                video_id = extract_video_id(adhoc_url)
                if not video_id:
                    st.error("❌ Could not extract YouTube Video ID from the URL.")
                else:
                    with st.spinner("Uploading ad-hoc subtitles to YouTube..."):
                        content = adhoc_srt_file.read().decode("utf-8")
                        media = MediaIoBaseUpload(
                            io.BytesIO(content.encode("utf-8")),
                            mimetype="application/octet-stream",
                            resumable=True
                        )
                        
                        youtube = build('youtube', 'v3', credentials=creds)
                        caption_id = upload_caption_track(
                            youtube,
                            video_id=video_id,
                            language=adhoc_lang,
                            name=adhoc_caption_name,
                            media=media
                        )
                        st.success(f"✅ Ad-hoc subtitles uploaded successfully! Caption ID: {caption_id}")
            except Exception as e:
                if "invalid_grant" in str(e):
                    if os.path.exists('token.json'):
                        os.remove('token.json')
                    st.error("❌ Your session has been revoked or expired. Please re-authenticate in the sidebar.")
                    st.rerun()
                else:
                    st.error(f"❌ YouTube API error: {e}")

st.divider()

col1, col2 = st.columns(2)

with col1:
    def on_clip_change():
        if st.session_state.clip_src == "YouTube URL":
            st.session_state.window_start_input = "00:10:00"

    def on_full_change():
        if st.session_state.full_src == "YouTube URL":
            st.session_state.window_start_input = "03:30:00"
        else:
            st.session_state.window_start_input = "00:10:00"

    st.subheader("1. Short Clip (Search Template)")
    clip_src = st.radio("Source", ["Local File", "YouTube URL"], key="clip_src", label_visibility="collapsed", on_change=on_clip_change)
    if clip_src == "Local File":
        clip_input = st.file_uploader(
            "Upload clip (MP3, WAV, M4A, or MP4)",
            type=["mp3", "wav", "mp4", "m4a"],
            help="The first ~30 seconds will be used as the search template.",
            key="clip_uploader"
        )
        if clip_input:
            import re
            m = re.match(r"^(\d{8})", clip_input.name)
            if m:
                date_prefix = m.group(1)
                st.info(f"Detected date prefix: {date_prefix}")
                
                # --- AUTO-FETCH LOGIC ---
                file_id = f"clip_{clip_input.name}_{clip_input.size}"
                if st.session_state.get("last_clip_file") != file_id:
                    st.session_state["last_clip_file"] = file_id
                    with st.spinner("Automatically searching YouTube for a matching date..."):
                        matched_url = fetch_matching_youtube_url(date_prefix, target_type="full_audio")
                    if matched_url:
                        st.success("Found a matching full audio URL! Starting find process...")
                        st.session_state.full_src = "YouTube URL"
                        st.session_state.full_url = matched_url
                        st.session_state.window_start_input = "03:30:00"
                        st.session_state.trigger_find = True
                        st.rerun()
                # ------------------------

                if st.button("Fetch URL", key="fetch_url_btn"):
                    with st.spinner("Searching YouTube for a matching date..."):
                        matched_url = fetch_matching_youtube_url(date_prefix, target_type="full_audio")
                    if matched_url:
                        st.success("Found a matching full audio URL!")
                        st.session_state.full_src = "YouTube URL"
                        st.session_state.full_url = matched_url
                        st.session_state.window_start_input = "03:30:00"
                        st.rerun()
                    else:
                        st.error("Could not find a matching video/stream on NWTVMedia.")
    else:
        clip_input = st.text_input(
            "YouTube URL of short clip",
            placeholder="https://www.youtube.com/watch?v=...",
            help="The first ~30 seconds of this video will be used as the search template.",
            key="clip_url"
        )
        if st.button("Validate Clip URL"):
            if not clip_input:
                st.warning("Please enter a URL first.")
            else:
                clean_url = sanitize_youtube_url(clip_input)
                if not extract_video_id(clean_url):
                    st.error("Invalid YouTube URL format.")
                else:
                    with st.spinner("Validating..."):
                        is_valid, msg = check_youtube_url(clean_url)
                    if is_valid:
                        st.success(f"✅ Valid! Title: {msg}")
                    else:
                        st.error(f"❌ Cannot access video: {msg}")

with col2:
    st.subheader("2. Full Audio (Search Target)")

    full_src = st.radio("Source", ["YouTube URL", "Local File"], key="full_src", label_visibility="collapsed", on_change=on_full_change)
    if full_src == "YouTube URL":
        full_input = st.text_input(
            "YouTube URL of full audio",
            placeholder="https://www.youtube.com/watch?v=...",
            key="full_url"
        )
        if st.button("Validate Full Audio URL"):
            if not full_input:
                st.warning("Please enter a URL first.")
            else:
                clean_url = sanitize_youtube_url(full_input)
                if not extract_video_id(clean_url):
                    st.error("Invalid YouTube URL format.")
                else:
                    with st.spinner("Validating..."):
                        is_valid, msg = check_youtube_url(clean_url)
                    if is_valid:
                        st.success(f"✅ Valid! Title: {msg}")
                    else:
                        st.error(f"❌ Cannot access video: {msg}")
    else:
        full_input = st.file_uploader(
            "Upload full file (MP3/WAV/M4A/MP4)",
            type=["mp3", "wav", "mp4", "m4a"],
            key="full_uploader"
        )
        if full_input:
            import re
            m = re.match(r"^(\d{8})", full_input.name)
            if m:
                date_prefix = m.group(1)
                st.info(f"Detected date prefix: {date_prefix}")

                # --- AUTO-FETCH LOGIC ---
                file_id = f"full_{full_input.name}_{full_input.size}"
                if st.session_state.get("last_full_file") != file_id:
                    st.session_state["last_full_file"] = file_id
                    with st.spinner("Automatically searching YouTube for a matching date..."):
                        matched_url = fetch_matching_youtube_url(date_prefix, target_type="short_clip")
                    if matched_url:
                        st.success("Found a matching short clip URL! Starting find process...")
                        st.session_state._pending_clip_src = "YouTube URL"
                        st.session_state._pending_clip_url = matched_url
                        st.session_state.trigger_find = True
                        st.rerun()
                # ------------------------

                if st.button("Fetch URL", key="fetch_url_btn_full"):
                    with st.spinner("Searching YouTube for a matching date..."):
                        matched_url = fetch_matching_youtube_url(date_prefix, target_type="short_clip")
                    if matched_url:
                        st.success("Found a matching short clip URL!")
                        st.session_state._pending_clip_src = "YouTube URL"
                        st.session_state._pending_clip_url = matched_url
                        st.rerun()
                    else:
                        st.error("Could not find a matching video/stream on NWTVMedia.")

st.divider()

window_start = st.text_input(
    "Search window start time (HH:MM:SS)",
    key="window_start_input",
    help="The app will search a 30-minute window starting at this time. "
         "Change this and click Find again to search a different range "
         "(the audio won't be re-downloaded).",
)

col_find, col_stop = st.columns(2)
with col_find:
    # Manual button or automatic trigger
    find_btn = st.button("🔍 Find Timestamp", type="primary", use_container_width=True)
    if st.session_state.get("trigger_find"):
        find_btn = True
        st.session_state["trigger_find"] = False
with col_stop:
    stop_btn = st.button("🛑 Stop Processing", use_container_width=True)

if stop_btn:
    st.warning("Process was manually stopped.")
    st.stop()

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
                yt_audio_path = os.path.join(tempfile.gettempdir(), f"yt_full_{video_id}.mp3")
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
            if cached == video_id and "clip_audio" in st.session_state and "clip_full_audio" in st.session_state:
                status.write(f"{elapsed()}  ✅  Using cached short clip audio.")
                clip = st.session_state["clip_audio"]
                clip_full = st.session_state["clip_full_audio"]
            else:
                status.write(f"{elapsed()}  ⬇️  Downloading short clip from YouTube…")
                yt_clip_path = os.path.join(tempfile.gettempdir(), f"yt_clip_{video_id}.mp3")
                assert clip_url is not None
                download_youtube_audio(clip_url, yt_clip_path)
                status.write(f"{elapsed()}  🎵  Loading full clip audio into memory…")
                clip_full = load_full_audio(yt_clip_path)
                clip = clip_full[:int(CLIP_DURATION * SAMPLE_RATE)]
                if os.path.exists(yt_clip_path):
                    os.remove(yt_clip_path)
                st.session_state["clip_audio"] = clip
                st.session_state["clip_full_audio"] = clip_full
                st.session_state["cached_clip_id"] = video_id
        else:
            file_id = f"clip_{clip_input.name}_{clip_input.size}"
            if st.session_state.get("cached_clip_id") == file_id and "clip_audio" in st.session_state and "clip_full_audio" in st.session_state:
                status.write(f"{elapsed()}  ✅  Using cached short local clip.")
                clip = st.session_state["clip_audio"]
                clip_full = st.session_state["clip_full_audio"]
            else:
                suffix = os.path.splitext(clip_input.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(clip_input.read())
                    clip_path = tmp.name
                
                status.write(f"{elapsed()}  🎵  Loading full clip audio into memory…")
                clip_full = load_full_audio(clip_path)
                clip = clip_full[:int(CLIP_DURATION * SAMPLE_RATE)]
                os.remove(clip_path)
                st.session_state["clip_audio"] = clip
                st.session_state["clip_full_audio"] = clip_full
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

        is_low_conf_local = (clip_src == "Local File" and confidence < 0.03)

        if is_low_conf_local:
            status.write(f"{elapsed()}  ⚠️  Confidence is low ({confidence:.2%}). Finding top candidates…")
            raw_candidates = find_top_offsets(window_audio, clip, num_candidates=3)
            candidates = [(o + window_start_sec, c) for o, c in raw_candidates]
            
            if not candidates:
                status.write(f"{elapsed()}  ❌  No match found.")
                status.update(label=f"Done — no match ({elapsed()})", state="error")
                st.subheader("Result")
                st.error("No matching audio found in the search window.")
                if "low_confidence_candidates" in st.session_state:
                    del st.session_state["low_confidence_candidates"]
                if "match_result" in st.session_state:
                    del st.session_state["match_result"]
            else:
                status.write(f"{elapsed()}  ✅  Candidate timestamps located.")
                status.update(label=f"Done! ({elapsed()})", state="complete")
                
                st.session_state["low_confidence_candidates"] = {
                    "candidates": candidates,
                    "clip_src": clip_src,
                    "full_src": full_src,
                    "clip_url": clip_url,
                    "full_url": full_url
                }
                if "match_result" in st.session_state:
                    del st.session_state["match_result"]
                for k in ["cut_points", "offsets", "cut_results_table", "additional_cuts_input_val"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        else:
            if "low_confidence_candidates" in st.session_state:
                del st.session_state["low_confidence_candidates"]

            if confidence < 0.01:
                status.write(f"{elapsed()}  ❌  No match found.")
                status.update(label=f"Done — no match ({elapsed()})", state="error")
                st.subheader("Result")
                st.error(
                    "No matching audio found in the search window. "
                    "Try adjusting the start time and clicking **Find Timestamp** "
                    "again (the audio won't be re-downloaded)."
                )
                if "match_result" in st.session_state:
                    del st.session_state["match_result"]
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
                    
                st.session_state["match_result"] = {
                    "offset_sec": offset_sec,
                    "clip_src": clip_src,
                    "full_src": full_src,
                    "clip_url": clip_url if clip_src == "YouTube URL" else None,
                }
                for k in ["cut_points", "offsets", "cut_results_table", "additional_cuts_input_val"]:
                    if k in st.session_state:
                        del st.session_state[k]

                if confidence < 0.15:
                    st.warning(
                        "The confidence score is low. The match may not be accurate. "
                        "Try using a longer or cleaner audio clip."
                    )

    except Exception as e:
        status.update(label="Error occurred", state="error")
        st.error(f"An error occurred: {e}")

# ── Low Confidence Candidates Display ────────────────────────────────────────
if "low_confidence_candidates" in st.session_state:
    lc_info = st.session_state["low_confidence_candidates"]
    candidates = lc_info["candidates"]
    full_url = lc_info["full_url"]
    
    st.subheader("🎯 Low Confidence Candidates")
    st.warning(
        "The highest confidence score was below 3.00%. "
        "Showing up to 3 candidate timestamps matching the short clip:"
    )
    
    for idx, (offset_sec, confidence) in enumerate(candidates):
        hms_str = seconds_to_hms(offset_sec, ms=True)
        line = f"**Candidate {idx+1}**: `{hms_str}` (Confidence: `{confidence:.2%}`)"
        if lc_info["full_src"] == "YouTube URL" and full_url:
            full_video_id = extract_video_id(full_url)
            if full_video_id:
                yt_link = f"https://www.youtube.com/watch?v={full_video_id}&t={int(offset_sec)}s"
                line += f" — [▶️ Open YouTube at this timestamp]({yt_link})"
        st.markdown(line)
        
    options = []
    for idx, (offset_sec, confidence) in enumerate(candidates):
        hms_str = seconds_to_hms(offset_sec, ms=True)
        options.append(f"Candidate {idx+1}: {hms_str} (Conf: {confidence:.2%})")
        
    selected_option = st.radio(
        "Select candidate timestamp to use for subtitle shifting:",
        options=options,
        index=0,
        key="selected_lc_candidate_radio"
    )
    
    selected_idx = options.index(selected_option)
    chosen_offset, chosen_conf = candidates[selected_idx]
    
    st.session_state["match_result"] = {
        "offset_sec": chosen_offset,
        "clip_src": lc_info["clip_src"],
        "full_src": lc_info["full_src"],
        "clip_url": lc_info["clip_url"] if lc_info["clip_src"] == "YouTube URL" else None,
    }


# ── Feature: Subtitle Shifting ───────────────────────────────────────────────

if "match_result" in st.session_state:
    res = st.session_state["match_result"]
    if res["clip_src"] == "YouTube URL" and res["full_src"] == "Local File":
        st.divider()
        st.subheader("✂️ Detect Additional Cuts in Short Clip")
        st.markdown(
            "If the short clip has cuts (omitted sections of the full audio), "
            "enter the cut timestamps in the short clip format (`HH:MM:SS` or `MM:SS`, comma-separated, up to 7). "
            "We will search the full audio after each cut point to establish a piecewise shift."
        )
        
        additional_cuts_str = st.text_input(
            "Cut timestamps in short clip (comma-separated, up to 7)",
            value=st.session_state.get("additional_cuts_input_val", ""),
            placeholder="e.g. 00:05:30, 00:15:45",
            key="additional_cuts_input"
        )
        
        analyze_cuts = st.button("🔍 Find Cuts & Update Shifts", type="secondary")
        
        if analyze_cuts:
            raw_cuts = [c.strip() for c in additional_cuts_str.split(",") if c.strip()]
            raw_cuts = raw_cuts[:7]
            
            try:
                cut_seconds = []
                for c in raw_cuts:
                    cut_seconds.append(hms_to_seconds(c))
            except Exception:
                st.error("Invalid timestamp format. Use HH:MM:SS or MM:SS.")
                st.stop()
                
            cut_seconds = sorted([c for c in cut_seconds if c > 0])
            
            full_audio = st.session_state.get("full_audio")
            clip_full_audio = st.session_state.get("clip_full_audio")
            
            if full_audio is None or clip_full_audio is None:
                st.error("Audio cache missing. Please run the initial Find Timestamp again.")
                st.stop()
                
            cut_points = [0.0]
            offsets = [res["offset_sec"]]
            cut_results = []
            
            with st.spinner("Analyzing cuts..."):
                for i, c_sec in enumerate(cut_seconds):
                    prev_c = cut_points[i]
                    prev_o = offsets[i]
                    expected_o = prev_o + (c_sec - prev_c)
                    
                    template_start_sample = int(c_sec * SAMPLE_RATE)
                    template_end_sample = min(len(clip_full_audio), template_start_sample + int(CLIP_DURATION * SAMPLE_RATE))
                    
                    if template_start_sample >= len(clip_full_audio):
                        st.warning(f"Cut timestamp {seconds_to_hms(c_sec)} is beyond short clip length.")
                        break
                        
                    template = clip_full_audio[template_start_sample:template_end_sample]
                    
                    search_start_sec = max(0.0, expected_o - 10.0)
                    search_end_sec = min(len(full_audio) / SAMPLE_RATE, expected_o + 300.0)
                    
                    start_sample = int(search_start_sec * SAMPLE_RATE)
                    end_sample = int(search_end_sec * SAMPLE_RATE)
                    
                    if start_sample >= len(full_audio) or start_sample >= end_sample:
                        st.warning(f"Search window for cut {seconds_to_hms(c_sec)} is outside full audio range.")
                        break
                        
                    search_window = full_audio[start_sample:end_sample]
                    
                    offset_in_window, confidence = find_offset(search_window, template)
                    actual_o = search_start_sec + offset_in_window
                    
                    if confidence < 0.02:
                        st.warning(f"Could not find matching audio for cut at {seconds_to_hms(c_sec)} (low confidence: {confidence:.2%}). Assuming no gap.")
                        actual_o = expected_o
                        gap_sec = 0.0
                    else:
                        gap_sec = actual_o - expected_o
                        
                    cut_points.append(c_sec)
                    offsets.append(actual_o)
                    
                    cut_results.append({
                        "Cut Timestamp": seconds_to_hms(c_sec),
                        "Expected Full Offset": seconds_to_hms(expected_o, ms=True),
                        "Found Full Offset": seconds_to_hms(actual_o, ms=True),
                        "Confidence": f"{confidence:.2%}",
                        "Detected Gap": f"{gap_sec:.3f}s" if gap_sec > 0.01 else "None",
                        "Status": "✅ Matched" if confidence >= 0.02 else "⚠️ Fallback"
                    })
                    
            st.session_state["cut_points"] = cut_points
            st.session_state["offsets"] = offsets
            st.session_state["cut_results_table"] = cut_results
            st.session_state["additional_cuts_input_val"] = additional_cuts_str
            st.success("Cuts analyzed successfully!")

        if "cut_results_table" in st.session_state and st.session_state["cut_results_table"]:
            import pandas as pd
            st.markdown("### Detected Cuts & Segments Mapping")
            df = pd.DataFrame(st.session_state["cut_results_table"])
            st.dataframe(df, use_container_width=True)

        st.divider()
        st.subheader("Shift Subtitles (SRT)")
        
        if "cut_points" in st.session_state and "offsets" in st.session_state and len(st.session_state["cut_points"]) > 1:
            st.info("Piecewise subtitle shifting is active based on the cuts detected above.")
        else:
            st.markdown(
                "Upload the SRT file that corresponds to the local target audio. "
                "This will shift all timestamps backwards by the match offset "
                f"(**{res['offset_sec']:.3f}s**)."
            )
            
        srt_file = st.file_uploader("Upload SRT File", type=["srt"])
        if srt_file:
            content = srt_file.read().decode("utf-8")
            
            if "cut_points" in st.session_state and "offsets" in st.session_state and len(st.session_state["cut_points"]) > 1:
                cut_points = st.session_state["cut_points"]
                offsets = st.session_state["offsets"]
                shifted = shift_srt_content_piecewise(content, cut_points, offsets)
                st.success("Subtitles shifted successfully using piecewise segment mapping!")
            else:
                shifted = shift_srt_content(content, res["offset_sec"])
                st.success(f"Subtitles shifted successfully by -{res['offset_sec']:.3f}s!")
            
            orig_name = srt_file.name
            if orig_name.lower().endswith(".srt"):
                new_name = orig_name[:-4] + "_shifted.srt"
            else:
                new_name = orig_name + "_shifted.srt"
                
            col_dl, col_ul = st.columns(2)
            with col_dl:
                st.download_button(
                    label="📥 Download Shifted SRT",
                    data=shifted,
                    file_name=new_name,
                    mime="text/plain",
                    type="primary",
                    use_container_width=True
                )
                
            with col_ul:
                import os
                if not os.path.exists('token.json'):
                    st.warning("⚠️ 'token.json' not found. Run `python auth.py` in the terminal to authenticate your YouTube account.")
                else:
                    try:
                        from google.oauth2.credentials import Credentials
                        from google.auth.transport.requests import Request
                        from googleapiclient.discovery import build
                        from googleapiclient.http import MediaIoBaseUpload
                        import io

                        # Scopes from auth.py
                        SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
                        
                        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
                        if creds and creds.expired and creds.refresh_token:
                            with st.spinner("Refreshing YouTube credentials..."):
                                creds.refresh(Request())
                                with open('token.json', 'w') as token_file:
                                    token_file.write(creds.to_json())
                                    
                        lang = st.selectbox("Caption Language", ["ko", "en"], index=0)
                        caption_name = st.text_input("Caption Track Name", value="Korean" if lang == "ko" else "English")
                        
                        upload_btn = st.button("📤 Upload to YouTube", type="primary", use_container_width=True)
                        if upload_btn:
                            clip_url = res.get("clip_url") or st.session_state.get("clip_url")
                            video_id = extract_video_id(clip_url) if clip_url else None
                            
                            if not video_id:
                                st.error("❌ Could not extract YouTube video ID from short clip URL.")
                            else:
                                with st.spinner("Uploading subtitle track to YouTube..."):
                                    youtube = build('youtube', 'v3', credentials=creds)
                                    
                                    # In-memory MediaIoBaseUpload
                                    media = MediaIoBaseUpload(
                                        io.BytesIO(shifted.encode("utf-8")),
                                        mimetype="application/octet-stream",
                                        resumable=True
                                    )
                                    
                                    caption_id = upload_caption_track(
                                        youtube,
                                        video_id=video_id,
                                        language=lang,
                                        name=caption_name,
                                        media=media
                                    )
                                    st.success(f"✅ Subtitles uploaded successfully! Caption ID: {caption_id}")
                    except Exception as e:
                        st.error(f"❌ YouTube API error: {e}")

