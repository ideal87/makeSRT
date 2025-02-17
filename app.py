import streamlit as st
import ffmpeg
import os
import io

def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name

def extract_first_five_minutes(input_file):
    output_file = f"first_5_minutes_{input_file}"
    (
        ffmpeg
        .input(input_file, t=300)  # Extract first 300 seconds
        .output(output_file, codec='copy')
        .run(overwrite_output=True)
    )
    return output_file

st.title("MP3 Extractor")

uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    input_file = save_uploaded_file(uploaded_file)
    if st.button("Extract First 5 Minutes"):
        output_file = extract_first_five_minutes(input_file)
        with open(output_file, 'rb') as f:
            audio_bytes = f.read()
            st.audio(audio_bytes, format='audio/mp3')
            st.download_button(
                label="Download Extracted Segment",
                data=audio_bytes,
                file_name=output_file,
                mime='audio/mpeg'
            )
