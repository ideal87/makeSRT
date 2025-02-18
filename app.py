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
        """
        **Task:** Act as a **subtitle optimization engine** to process an SRT file. Follow these steps **exactly**, and include **detailed reasoning** for all changes.  
        ---
        ### **1. Parse & Clean the Input SRT**  
        - **Actions:**  
          - Read segments, noting start/end times and text.  
          - **Revise Korean typos** (e.g., spacing, spelling).  
          - **Remove filler words**: ‘그리고’, ‘그래서’, ‘그러니까’ (unless critical to meaning).  
          - **Preserve unclear phrases/examples** (e.g., jargon, names) even if ambiguous.  
        **Reasoning Format:**  
        > “Segment [INDEX]:  
        > - Original: ‘[TEXT]’  
        > - Typos Revised: ‘[CORRECTED_TEXT]’  
        > - Removed Fillers: [‘그리고’, ...]  
        > - Retained Unclear Phrase: ‘[PHRASE]’.”  
        ---
        ### **2. Combine Short Segments**  
        **Rules:**  
        - **Short** = Duration **<4s** OR text **<25 characters**.  
        - **Combine Priority**:  
          1. Previous segment if its text ≤70 characters.  
          2. Next segment if previous is too long.  
        **Reasoning Format:**  
        > “Segment [INDEX] is short ([DURATION]s / [TEXT_LENGTH] chars).  
        > - Combined with [PREV/NEXT] segment [INDEX].  
        > - New text: ‘[COMBINED_TEXT]’.  
        > - Adjusted end time to [NEW_END_TIME].”  
        ---
        ### **3. Split Long Text Segments**  
        **Rules:**  
        - **Split** if text **>90 characters**.  
        - **Split Point**: Nearest punctuation (., ,, ") near middle (≥20 chars before split).  
        - **Divide duration evenly** between splits.  

        **Reasoning Format:**  
        > “Segment [INDEX]: Text length [LENGTH].  
        > - Split at [PUNCTUATION] (position [SPLIT_INDEX]):  
        >   - Part 1: ‘[TEXT_PART1]’ ([DURATION_PART1]s).  
        >   - Part 2: ‘[TEXT_PART2]’ ([DURATION_PART2]s).”  
        ---
        ### **4. Revise Korean Text for Grammar, Clarity, and Translation Readiness**
        **Rules:**  
        - Preserve theological meaning and tone while improving structure.
        - Do not change Korean Bible Quotes.  
        - Optimize for accurate English translation by resolving ambiguities.
        - Grammar & Syntax Rules:
        Fix particle errors (은/는 vs. 이/가) and verb ending mismatches.
        Convert sentence fragments to complete sentences (e.g., “하나님의 사랑은 크시니” → “하나님의 사랑은 크시므로”).
        Maintain formal liturgical tone unless original uses intentional colloquialisms.
        ---
        ### **5. Combine short blank segments to the previous segments**
        **Rules:**  
        - For each segment with no text and has duration of 4 or less, combine it to the previous segement.
        ---
        ### **6. Generate Output SRT**  
        - Reindex all segments.  
        - **Format:**  
          ```  
          [INDEX]  
          [START_TIME] --> [END_TIME]  
          [TEXT]  
          ```  
        - **Validation**:  
          - No text >90 chars.  
          - No short segments unless unavoidable.  
        ---  
        **Example Output:**  
        ```  
        1  
        00:00:10,000 --> 00:00:14,500  
        Revised text without fillers. Retained unclear term "ABC123".  

        2  
        00:00:14,500 --> 00:00:16,000  
        First half of a long sentence.  

        3  
        00:00:16,000 --> 00:00:18,000  
        Second half of a long sentence.  
        ```  
        ---  
        **Instructions for GPT:**  
        - **Output Only** the final SRT.  
        - **Do not** add summaries or disclaimers or reasoning annotation or file delimiter like ```"""
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
