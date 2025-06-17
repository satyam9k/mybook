import streamlit as st
import google.generativeai as genai
import numpy as np
import os
import struct
import mimetypes
from streamlit_audiorecorder import audiorecorder

# --- Page Configuration ---
st.set_page_config(page_title="Voice Notebook AI", page_icon="🎙️", layout="wide")

# --- State Management ---
for key in ["document_processed", "podcast_summary", "podcast_audio", "chat_history", "apis_configured", "chunks", "embeddings"]:
    if key not in st.session_state:
        st.session_state[key] = False if key in ["document_processed", "apis_configured"] else ([] if key == "chat_history" else None)

GEMINI_VOICES = {
    "Male 1 (Puck)": "Puck",
    "Female 1 (Achernar)": "Achernar",
    "Narrator": "gemini-1.5-pro-preview-tts", 
}


def configure_apis():
    """Configures the Gemini API using st.secrets."""
    try:
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("GEMINI_API_KEY not found. Please add it to your Streamlit secrets.")
            return False
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        st.session_state.apis_configured = True
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}"); return False

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for raw audio data."""
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample, sample_rate, num_channels = parameters["bits_per_sample"], parameters["rate"], 1
    data_size, bytes_per_sample = len(audio_data), bits_per_sample // 8
    block_align, byte_rate = num_channels * bytes_per_sample, sample_rate * block_align
    chunk_size = 36 + data_size
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample, b"data", data_size)
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict:
    """Parses bits per sample and rate from an audio MIME type string."""
    match = mimetypes.guess_extension(mime_type)
    bits_per_sample = 16
    rate = 24000
    if match and "L16" in match: bits_per_sample = 16
    if "rate=" in mime_type:
        try: rate = int(mime_type.split("rate=")[1])
        except: pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}

@st.cache_data(show_spinner=False)
def process_document(uploaded_file):
    st.session_state.chat_history = []
    model = genai.GenerativeModel('gemini-1.5-flash')
    file_data = {"mime_type": uploaded_file.type, "data": uploaded_file.getvalue()}
    prompt = "Extract all text content from this document. Preserve the original paragraph structure."
    with st.spinner("🧠 Gemini is reading and analyzing the document..."):
        response = model.generate_content([prompt, file_data])
        full_text = response.text
    chunks = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 100]
    with st.spinner("🔬 Creating semantic index of the document..."):
        embedding_model = 'models/text-embedding-004'
        response = genai.embed_content(model=embedding_model, content=chunks, task_type="RETRIEVAL_DOCUMENT")
        chunk_embeddings = [item['embedding'] for item in response['embedding']]
        return chunks, chunk_embeddings

def transcribe_audio_with_gemini(audio_bytes):
    """Uses Gemini 1.5 Pro to transcribe audio."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        audio_file = genai.upload_file(path=audio_bytes, display_name="user_audio", mime_type="audio/wav")
        response = model.generate_content(["Transcribe this audio.", audio_file])
        return response.text if response.text else ""
    except Exception as e:
        st.error(f"Gemini Speech-to-Text Error: {e}"); return ""

def get_rag_response(query, chunks, embeddings):
    embedding_model, model = 'models/text-embedding-004', genai.GenerativeModel('gemini-1.5-flash')
    query_embedding = genai.embed_content(model=embedding_model, content=query, task_type="RETRIEVAL_QUERY")['embedding']
    dot_products = np.dot(np.array(embeddings), query_embedding)
    top_indices = np.argsort(dot_products)[-5:][::-1]
    relevant_context = "\n---\n".join([chunks[i] for i in top_indices])
    prompt = f"You are an expert AI assistant. Answer the user's question based *only* on the provided context.\nCONTEXT:\n---\n{relevant_context}\n---\nQUESTION: {query}\n\nANSWER:"
    response = model.generate_content(prompt)
    return response.text

def generate_tts_with_gemini(text, voice_name):
    """Converts text to speech using the native Gemini TTS feature."""
    try:
        tts_model = genai.GenerativeModel("gemini-1.5-pro-preview-tts")
        response = tts_model.generate_content(text, stream=True,
            generation_config={"response_mime_type": "audio/wav"})
        
        audio_data = b''
        for chunk in response:
            audio_data += chunk.audio_content

        return audio_data
    except Exception as e:
        st.error(f"Native Gemini TTS Error: {e}"); return None

# --- UI Layout ---
st.title("🎙️ myBook")

# --- Automatic Initialization from Secrets ---
if not st.session_state.apis_configured:
    configure_apis()

with st.sidebar:
    st.header("🔊 Voice Selection")
    qa_voice = st.selectbox("Q&A Voice", options=GEMINI_VOICES.keys(), index=0)
    podcast_voice = st.selectbox("Podcast Voice", options=GEMINI_VOICES.keys(), index=1)
    st.markdown("---")
    st.info("This app uses Gemini 1.5 for document analysis, Q&A, speech-to-text, and text-to-speech.")

if not st.session_state.apis_configured:
    st.error("Assistant is not initialized. Please ensure your GEMINI_API_KEY is set in the Streamlit Cloud secrets.")
else:
    tab1, tab2 = st.tabs(["🎧 Podcast Generator", "💬 Voice Q&A"])

    with tab1:
        st.header("Create a Podcast from Your Document")
        uploaded_file = st.file_uploader("Upload a Document (PDF, PNG, JPG)", type=["pdf", "png", "jpg", "jpeg"], key="podcast_uploader")

        if uploaded_file:
            st.session_state.chunks, st.session_state.embeddings = process_document(uploaded_file)
            st.session_state.document_processed = st.session_state.chunks is not None
            if st.session_state.document_processed:
                st.success(f"Document '{uploaded_file.name}' processed. Generate a podcast or ask questions in the Q&A tab.")

        if st.session_state.document_processed:
            if st.button("✨ Generate Podcast Summary & Audio", use_container_width=True):
                full_text = " ".join(st.session_state.chunks)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = "Summarize the document in a clear, narrative style for a short podcast. Include an intro, key points, and a conclusion."
                with st.spinner("Generating podcast script..."):
                    response = model.generate_content([prompt, full_text])
                    st.session_state.podcast_summary = response.text
                with st.spinner("Converting script to audio..."):
                    st.session_state.podcast_audio = generate_tts_with_gemini(st.session_state.podcast_summary, GEMINI_VOICES[podcast_voice])
            
            if st.session_state.podcast_summary: st.subheader("Podcast Script"); st.markdown(st.session_state.podcast_summary)
            if st.session_state.podcast_audio: st.subheader("Listen to Your Podcast"); st.audio(st.session_state.podcast_audio, format='audio/wav')

    with tab2:
        st.header("Ask Questions with Your Voice")
        if not st.session_state.document_processed:
            st.warning("Please upload and process a document in the 'Podcast Generator' tab first.")
        else:
            for entry in st.session_state.chat_history:
                with st.chat_message(entry["role"]):
                    st.markdown(entry["text"])
                    if "audio" in entry and entry["audio"]: st.audio(entry["audio"], format="audio/wav")

            recorded_audio = audiorecorder("Click to Record Your Question", "Recording...")
            if recorded_audio:
                audio_bytes = recorded_audio.export().read()
                with st.spinner("Transcribing your question..."):
                    user_query = transcribe_audio_with_gemini(audio_bytes)
                
                if user_query:
                    st.session_state.chat_history.append({"role": "user", "text": user_query})
                    with st.spinner("Finding answer in the document..."):
                        answer_text = get_rag_response(user_query, st.session_state.chunks, st.session_state.embeddings)
                    with st.spinner("Generating voice reply..."):
                        answer_audio = generate_tts_with_gemini(answer_text, GEMINI_VOICES[qa_voice])
                    st.session_state.chat_history.append({"role": "assistant", "text": answer_text, "audio": answer_audio})
                    st.rerun()
