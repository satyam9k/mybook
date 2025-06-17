import streamlit as st
import google.generativeai as genai
import numpy as np
import os
import soundfile as sf
import io
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# --- Page Configuration ---
st.set_page_config(page_title="Voice Notebook AI", page_icon="🎙️", layout="wide")

# --- State Management ---
for key in ["document_processed", "podcast_summary", "podcast_audio", "chat_history", "apis_configured", "chunks", "embeddings"]:
    if key not in st.session_state:
        st.session_state[key] = False if key in ["document_processed", "apis_configured"] else ([] if key == "chat_history" else None)

# --- API & Voice Configuration ---
GEMINI_VOICES = {
    "Male 1 (Puck)": "Puck",
    "Female 1 (Achernar)": "Achernar",
    "Narrator": "gemini-1.5-pro-preview-tts",
}

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame):
        self.audio_frames.append(frame)
        return frame

    def get_audio_bytes(self):
        if not self.audio_frames:
            return None
        
        sound_chunk = self.audio_frames[0]
        sample_rate = sound_chunk.sample_rate
        sample_width = sound_chunk.format.bytes
        
        sound_data = np.hstack([frame.to_ndarray() for frame in self.audio_frames])

        buffer = io.BytesIO()
        sf.write(buffer, data=sound_data.T, samplerate=sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()

# --- Backend API Functions ---
def configure_apis():
    try:
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("GEMINI_API_KEY not found. Please add it to your Streamlit secrets.")
            return False
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        st.session_state.apis_configured = True
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}"); return False

@st.cache_data(show_spinner=False)
def process_document(uploaded_file):
    st.session_state.chat_history = []
    model = genai.GenerativeModel('gemini-1.5-flash')
    file_data = {"mime_type": uploaded_file.type, "data": uploaded_file.getvalue()}
    prompt = "Extract all text content from this document. Preserve the original paragraph structure."
    with st.spinner("myBook is reading and analyzing the document..."):
        response = model.generate_content([prompt, file_data])
        full_text = response.text
    chunks = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 100]
    with st.spinner("🔬 Creating semantic index..."):
        embedding_model = 'models/text-embedding-004'
        response = genai.embed_content(model=embedding_model, content=chunks, task_type="RETRIEVAL_DOCUMENT")
        chunk_embeddings = [item['embedding'] for item in response['embedding']]
        return chunks, chunk_embeddings

def transcribe_audio_with_gemini(audio_bytes):
    try:
        audio_file = genai.upload_file(path=audio_bytes, display_name="user_audio", mime_type="audio/wav")
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(["Transcribe this audio.", audio_file])
        genai.delete_file(audio_file.name) # Clean up the uploaded file
        return response.text if response.text else ""
    except Exception as e:
        st.error(f"Gemini Speech-to-Text Error: {e}"); return ""

def get_rag_response(query, chunks, embeddings):
    embedding_model, model = 'models/text-embedding-004', genai.GenerativeModel('gemini-1.5-flash')
    query_embedding = genai.embed_content(model=embedding_model, content=query, task_type="RETRIEVAL_QUERY")['embedding']
    dot_products = np.dot(np.array(embeddings), query_embedding)
    top_indices = np.argsort(dot_products)[-5:][::-1]
    relevant_context = "\n---\n".join([chunks[i] for i in top_indices])
    prompt = f"Answer the user's question based *only* on the provided context.\nCONTEXT:\n---\n{relevant_context}\n---\nQUESTION: {query}\n\nANSWER:"
    response = model.generate_content(prompt)
    return response.text

def generate_tts_with_gemini(text):
    try:
        tts_model = genai.GenerativeModel("gemini-1.5-pro-preview-tts")
        response = tts_model.generate_content(text, stream=True, generation_config={"response_mime_type": "audio/wav"})
        audio_data = b''.join([chunk.audio_content for chunk in response])
        return audio_data
    except Exception as e:
        st.error(f"Native Gemini TTS Error: {e}"); return None

# --- UI Layout ---
st.title("🎙️ Voice Notebook AI")
st.markdown("Your personal AI assistant, powered entirely by the Gemini 1.5 API.")

if not st.session_state.apis_configured: configure_apis()

with st.sidebar:
    st.header("🔊 Voice Selection")
    qa_voice = st.selectbox("Q&A Voice", options=GEMINI_VOICES.keys(), index=0)
    podcast_voice = st.selectbox("Podcast Voice", options=GEMINI_VOICES.keys(), index=1)
    st.markdown("---")
    st.info("This app uses Gemini 1.5 for all AI tasks, including document analysis, Q&A, and voice processing.")

if not st.session_state.apis_configured:
    st.error("Assistant not initialized. Please set your GEMINI_API_KEY in the Streamlit Cloud secrets.")
else:
    tab1, tab2 = st.tabs(["🎧 Podcast Generator", "💬 Voice Q&A"])

    with tab1:
        st.header("Create a Podcast from Your Document")
        uploaded_file = st.file_uploader("Upload Document (PDF, PNG, JPG)", type=["pdf", "png", "jpg", "jpeg"], key="podcast_uploader")

        if uploaded_file:
            st.session_state.chunks, st.session_state.embeddings = process_document(uploaded_file)
            st.session_state.document_processed = st.session_state.chunks is not None
            if st.session_state.document_processed: st.success(f"Document '{uploaded_file.name}' processed.")

        if st.session_state.document_processed:
            if st.button("✨ Generate Podcast Summary & Audio", use_container_width=True):
                full_text = " ".join(st.session_state.chunks)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = "Summarize the document in a clear, narrative style for a short podcast."
                with st.spinner("Generating podcast script..."):
                    st.session_state.podcast_summary = model.generate_content([prompt, full_text]).text
                with st.spinner("Converting script to audio..."):
                    st.session_state.podcast_audio = generate_tts_with_gemini(st.session_state.podcast_summary)
            
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

            # NEW: Robust Audio Recording UI
            webrtc_ctx = webrtc_streamer(
                key="speech-to-text",
                mode=WebRtcMode.SENDONLY,
                audio_processor_factory=AudioRecorder,
                media_stream_constraints={"audio": True, "video": False},
                send_interval=1000,
            )

            if not webrtc_ctx.state.playing:
                if st.button("Process Recorded Question", use_container_width=True):
                    if webrtc_ctx.audio_processor:
                        audio_bytes = webrtc_ctx.audio_processor.get_audio_bytes()
                        if audio_bytes:
                            with st.spinner("Transcribing your question..."):
                                user_query = transcribe_audio_with_gemini(io.BytesIO(audio_bytes))
                            
                            if user_query:
                                st.session_state.chat_history.append({"role": "user", "text": user_query})
                                with st.spinner("Finding answer in document..."):
                                    answer_text = get_rag_response(user_query, st.session_state.chunks, st.session_state.embeddings)
                                with st.spinner("Generating voice reply..."):
                                    answer_audio = generate_tts_with_gemini(answer_text)
                                st.session_state.chat_history.append({"role": "assistant", "text": answer_text, "audio": answer_audio})
                                st.rerun()
                        else:
                            st.warning("No audio was recorded. Please try recording again.")
                    else:
                        st.error("Audio processor not found. Please refresh the page.")
