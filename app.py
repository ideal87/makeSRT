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

# OpenAI client (API key should be set via env or st.secrets)
client = OpenAI()

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
        "ffmpeg",
        "-i", file_path,
        "-f", "segment",
        "-segment_time", str(chunk_length),
        "-c", "copy",
        output_pattern,
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
            prompt="""
            VVGA, 시편, 포도나무교회, 새물결선교회, 영적리더쉽, 비전트립, 임재, 오이코스, 예향교회,
            성도, 언약, 1부예배, 십자가복음학교, 헌금, 남전도회, 행함, 여전도회, 긍휼, 전도사,
            일의소명, 찬양과, 선교센터, 이길수, 두드림투게더, 선교사, 복음, 다윗, 다윗의,
            복음주의, 새물결대학, 새물결, 마다가스카르, 초대교회, 2부예배, 성경구절, 권세,
            기독학교, 여호와, 도전오십가정, 신약, 시게타, 십자가복음, 남녀전도회, 야고보,
            은혜, 새물결, 두드림, 투게더
            """
        )
    return response


def process_text_with_gpt(transcribed_text):
    """
    Post-process transcription using GPT-5.2
    """
    prompt = (
        """
Role: Korean Baptist sermon transcription editor

Task: Refine and correct the following Korean sermon transcription.

Mandatory Rules (follow exactly):
- Correct grammar, spacing, and word usage within a Korean Baptist sermon context.
- Do NOT omit, summarize, merge, split, reorder, or add any sentences.
- Remove unnecessary discourse fillers and conjunctions such as “그리고,” “그래서,” “그러니까,” “그런데,” “어…,” “음…,” “그…” only when they are stylistically redundant and do not serve rhetorical emphasis.
- Preserve the original sermon flow, preaching cadence, and emphasis; intentional repetition for emphasis must be kept.
- Maintain a consistent pastoral honorific style (e.g., “여러분,” “성도 여러분”) and avoid casual speech unless clearly intended by the speaker.
- Do NOT add interpretation, theological explanation, paraphrasing, or commentary.
- When the speaker is reading or quoting from a book, Scripture, or written source, preserve a neutral, reading-style tone, even if it differs from the sermon’s exhortative tone.
- Normalize Bible references into standard Korean format (e.g., “에베소서 1장 3절–5절”) without changing spoken meaning.

Quotation Rules:
- Use double quotation marks (" ") for all quotations.
- Use single quotation marks (‘ ’) only for quotations nested inside double quotations.
- All book titles must be enclosed in double quotation marks.
- All Bible verses (direct quotations from Scripture) must be enclosed in double quotation marks (" ").

Formatting Rules (critical):
- Output must be a single continuous line only.
- Do NOT include any line breaks, paragraph breaks, indentation, or bullet points.
- Do NOT add titles, introductions, conclusions, explanations, or notes.

Output Requirement:
- Return only the corrected Korean text, strictly following all rules above.

"""
        + transcribed_text
    )

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()


def process_chunk(chunk_info):
    """
    Processes a single audio chunk:
    1) Transcribe with 4o-mini
    2) Post-process with GPT-5.2
    """
    index, chunk = chunk_info
    try:
        logging.info(f"Processing chunk {index}: {chunk}")

        transcribed_text = transcribe_audio(chunk)
        logging.info(f"Transcription for chunk {index} completed.")

        processed_text = process_text_with_gpt(transcribed_text)
        logging.info(f"GPT post-processing for chunk {index} completed.")

        os.remove(chunk)
        logging.info(f"Deleted chunk file: {chunk}")

        return index, processed_text

    except Exception as e:
        logging.error(f"Error processing chunk {index}: {e}")
        return index, ""


def process_audio(file_path):
    chunks = split_audio(file_path)
    all_processed_text = []

    max_workers = min(8, len(chunks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_chunk, chunk): chunk[0]
            for chunk in chunks
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                chunk_index, text = future.result()
                all_processed_text.append((chunk_index, text))
                logging.info(f"Chunk {chunk_index} processed successfully.")
            except Exception as exc:
                logging.error(f"Chunk {idx} failed: {exc}")

    all_processed_text.sort(key=lambda x: x[0])
    return [text for _, text in all_processed_text]


# --- Streamlit App ---

st.title("MP3 to Text Transcription & Revision")
st.write("Upload an MP3 file to generate a revised text transcription.")

uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    base_name = os.path.splitext(uploaded_file.name)[0]
    output_file_name = f"{base_name}_processed.txt"

    if st.button("Process Audio"):
        with st.spinner("Processing audio, please wait..."):
            try:
                processed_texts = process_audio(tmp_file_path)

                final_text = ""
                for text in processed_texts:
                    final_text += text + "\n\n"

                st.success("Processing complete!")
                st.download_button(
                    label="Download Text File",
                    data=final_text,
                    file_name=output_file_name,
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")

            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
else:
    st.info("Awaiting MP3 file upload.")
