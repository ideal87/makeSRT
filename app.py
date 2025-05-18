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

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko",
            response_format="text",
            prompt="""시편, 포도나무교회, 새물결선교회, 영적리더쉽, 비전트립, 임재, 오이코스, 예향교회, 성도, 언약, 1부예배, 십자가복음학교, 헌금, 남전도회, 행함,
            여전도회, 긍휼, 전도사, 일의소명, 찬양과, 선교센터, 이길수, 두드림투게더, 선교사, 복음, 다윗, 다윗의, 복음주의, 새물결대학, 새물결, 마다가스카르, 초대교회, 2부예배,
            성경구절, 권세, 기독학교, 여호와, 도전오십가정, 신약, 시게타,십자가복음, 남녀전도회"""
        )
    return response

def process_text_with_gpt(transcribed_text):
    prompt = (
        """
        Please correct the grammar of the following Korean text in Korean Baptist Sermon context.
        Do not ommit any sentences.
        Remove unnecessary conjunctions like '그리고,' 그래서,' '그러니까' where applicable.
        When using quotations, use double quotations unless nested inside another quotation.
        Do not include introduction or outro in your response:\n"""
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

st.title("MP3 to Text Transcription & Revision")
st.write("Upload an MP3 file to generate a revised text transcription.")

uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

        # Extract the base name from the original uploaded file name
        base_name = os.path.splitext(uploaded_file.name)[0]
        output_file_name = f"{base_name}_processed.txt"

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
                    file_name=output_file_name,
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
