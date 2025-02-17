import streamlit as st
import subprocess
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from openai import OpenAI

# Global constant for chunk length (in seconds)
CHUNK_LENGTH = 200

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Set your OpenAI API key (ensure it's set in your environment or st.secrets)
client = OpenAI()

# Instead of using a client instance from OpenAI, we call openai's functions directly.
# (If you prefer using a client wrapper, adjust accordingly.)

def split_audio(file_path, chunk_length=200):
    """
    Splits an audio file into chunks of specified length.
    Returns:
        List of tuples: (chunk_index, chunk_filename)
    """
    global CHUNK_LENGTH
    CHUNK_LENGTH = chunk_length

    chunks = []
    base_name = os.path.splitext(file_path)[0]
    output_pattern = f"{base_name}_chunk_%03d.mp3"
    
    command = [
        'ffmpeg',
        '-i', file_path,
        '-f', 'segment',
        '-segment_time', str(chunk_length),
        '-c', 'copy',
        output_pattern
    ]
    subprocess.run(command, check=True)
    
    index = 0
    while True:
        chunk_file = f"{base_name}_chunk_{index:03d}.mp3"
        if os.path.exists(chunk_file):
            chunks.append((index, chunk_file))
            index += 1
        else:
            break

    return chunks

def shift_timestamp(ts, offset):
    """
    Shifts a timestamp (format HH:MM:SS,mmm) by offset seconds.
    """
    h, m, s_ms = ts.split(':')
    s, ms = s_ms.split(',')
    total_seconds = int(h)*3600 + int(m)*60 + int(s) + int(ms) / 1000
    new_total = total_seconds + offset
    new_h = int(new_total // 3600)
    new_m = int((new_total % 3600) // 60)
    new_s = int(new_total % 60)
    new_ms = int(round((new_total - int(new_total)) * 1000))
    return f"{new_h:02}:{new_m:02}:{new_s:02},{new_ms:03}"

def adjust_srt(srt_text, offset):
    """
    Adjusts all timestamp lines in an SRT text by the given offset (in seconds).
    """
    lines = srt_text.splitlines()
    adjusted_lines = []
    for line in lines:
        if '-->' in line:
            parts = line.split('-->')
            start_ts = parts[0].strip()
            end_ts = parts[1].strip()
            new_start = shift_timestamp(start_ts, offset)
            new_end = shift_timestamp(end_ts, offset)
            adjusted_lines.append(f"{new_start} --> {new_end}")
        else:
            adjusted_lines.append(line)
    return "\n".join(adjusted_lines)

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko",
            response_format="srt",
            prompt="""시편, 포도나무교회, 새물결선교회, 영적리더쉽, 비전트립, 임재, 오이코스, 예향교회, 성도, 언약, 1부예배, 십자가복음학교, 헌금, 남전도회, 행함,
            여전도회, 긍휼, 전도사, 일의소명, 찬양과, 선교센터, 이길수, 두드림투게더, 선교사, 복음, 다윗, 복음주의, 새물결대학, 새물결, 마다가스카르, 초대교회, 2부예배"""
        )
    return response

def process_text_with_gpt(transcribed_text):
    prompt = (
        """Revise the following SRT format file so that the segments form smoother and more complete Korean sentences without changing the original meaning.
Revise any Korean typos from the transcription process, combine segments where appropriate.
Do not remove specific examples or phrases even though they may not sound clear to certain audience.
Remove unnecessary filler words or conjunctions like '그리고', '그래서', '그러니까'. 
Keep the SRT format in your response (do not include a file delimiter):"""
        + transcribed_text
    )
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

def process_chunk(chunk_info):
    """
    Processes a single audio chunk: transcribes it and processes the text with GPT.
    Returns:
        Tuple (chunk_index, processed_text) with timestamps adjusted.
    """
    index, chunk = chunk_info
    try:
        logging.info(f"Processing chunk {index}: {chunk}")
        transcribed_text = transcribe_audio(chunk)
        logging.info(f"Transcription for chunk {index} completed.")
        processed_text = process_text_with_gpt(transcribed_text)
        logging.info(f"Processing with GPT for chunk {index} completed.")

        # Adjust the SRT timestamps based on the chunk offset
        offset = index * CHUNK_LENGTH
        processed_text = adjust_srt(processed_text, offset)

        os.remove(chunk)
        logging.info(f"Deleted chunk file: {chunk}")
        return (index, processed_text)
    except Exception as e:
        logging.error(f"An error occurred while processing chunk {index}: {e}")
        return (index, "")

def process_audio(file_path):
    chunks = split_audio(file_path)
    all_processed_text = []
    max_workers = min(8, len(chunks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(process_chunk, chunk_info): chunk_info[0] for chunk_info in chunks}
        for future in as_completed(future_to_chunk):
            index = future_to_chunk[future]
            try:
                chunk_index, processed_text = future.result()
                all_processed_text.append((chunk_index, processed_text))
                logging.info(f"Chunk {chunk_index} processed successfully.")
            except Exception as exc:
                logging.error(f"Chunk {index} generated an exception: {exc}")
    all_processed_text.sort(key=lambda x: x[0])
    sorted_texts = [text for index, text in all_processed_text]
    return sorted_texts

# --- Streamlit App ---

st.title("MP3 to SRT Transcription & Revision")
st.write("Upload an MP3 file to generate a revised SRT transcription.")

uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    if st.button("Process Audio"):
        with st.spinner("Processing audio, please wait..."):
            try:
                processed_texts = process_audio(tmp_file_path)
                # Combine all processed chunks into one SRT content string.
                srt_content = ""
                # Optionally, you could re-number SRT segments here.
                for idx, text in enumerate(processed_texts):
                    srt_content += text + "\n\n"

                st.success("Processing complete!")
                st.download_button(
                    label="Download SRT File",
                    data=srt_content,
                    file_name="processed.srt",
                    mime="text/plain",
                )
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
            finally:
                # Clean up the temporary file
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
else:
    st.info("Awaiting MP3 file upload.")
