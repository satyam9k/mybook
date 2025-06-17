import streamlit as st
import google.generativeai as genai
from google import genai as genai_client
from google.genai import types
from PIL import Image
import PyPDF2
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import tempfile
import os
import base64
import mimetypes
import struct
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading
import queue
import time
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="myBook - AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

class AudioRecorder:
    """Audio recorder class using WebRTC for reliable audio capture"""
    
    def __init__(self):
        self.audio_frames = []
        self.audio_queue = queue.Queue()
        
    def audio_frame_callback(self, frame):
        """Callback function to handle audio frames"""
        try:
            sound = frame.to_ndarray()
            self.audio_queue.put(sound)
        except Exception as e:
            logger.error(f"Error in audio frame callback: {e}")
    
    def get_audio_data(self):
        """Get recorded audio data"""
        audio_data = []
        try:
            while not self.audio_queue.empty():
                audio_data.append(self.audio_queue.get())
        except Exception as e:
            logger.error(f"Error getting audio data: {e}")
        return audio_data

class GeminiTTSProcessor:
    """TTS processor using Gemini 2.5 Pro TTS capabilities"""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini TTS client"""
        try:
            self.client = genai_client.Client(api_key=api_key)
            self.tts_model = "gemini-2.5-pro-preview-tts"
            logger.info("Gemini TTS client configured successfully")
        except Exception as e:
            logger.error(f"Error configuring Gemini TTS client: {e}")
            raise
    
    def convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """Generates a WAV file header for the given audio data and parameters."""
        parameters = self.parse_audio_mime_type(mime_type)
        bits_per_sample = parameters["bits_per_sample"]
        sample_rate = parameters["rate"]
        num_channels = 1
        data_size = len(audio_data)
        bytes_per_sample = bits_per_sample // 8
        block_align = num_channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        chunk_size = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",          # ChunkID
            chunk_size,       # ChunkSize
            b"WAVE",          # Format
            b"fmt ",          # Subchunk1ID
            16,               # Subchunk1Size
            1,                # AudioFormat
            num_channels,     # NumChannels
            sample_rate,      # SampleRate
            byte_rate,        # ByteRate
            block_align,      # BlockAlign
            bits_per_sample,  # BitsPerSample
            b"data",          # Subchunk2ID
            data_size         # Subchunk2Size
        )
        return header + audio_data

    def parse_audio_mime_type(self, mime_type: str) -> dict[str, int]:
        """Parses bits per sample and rate from an audio MIME type string."""
        bits_per_sample = 16
        rate = 24000

        parts = mime_type.split(";")
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate_str = param.split("=", 1)[1]
                    rate = int(rate_str)
                except (ValueError, IndexError):
                    pass
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except (ValueError, IndexError):
                    pass

        return {"bits_per_sample": bits_per_sample, "rate": rate}
    
    def text_to_speech(self, text: str, voice_style: str) -> Optional[bytes]:
        """Convert text to speech using Gemini TTS"""
        try:
            # Map voice styles to Gemini voice names
            voice_mapping = {
                "Male": "Puck",
                "Female": "Zephyr", 
                "Narrator": "Sage"
            }
            
            voice_name = voice_mapping.get(voice_style, "Zephyr")
            
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=text),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=1,
                response_modalities=["audio"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    ),
                ),
            )
            
            audio_data = b""
            
            for chunk in self.client.models.generate_content_stream(
                model=self.tts_model,
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    chunk.candidates is None
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue
                    
                if (chunk.candidates[0].content.parts[0].inline_data and 
                    chunk.candidates[0].content.parts[0].inline_data.data):
                    inline_data = chunk.candidates[0].content.parts[0].inline_data
                    data_buffer = inline_data.data
                    
                    # Convert to WAV if needed
                    if inline_data.mime_type != "audio/wav":
                        data_buffer = self.convert_to_wav(inline_data.data, inline_data.mime_type)
                    
                    audio_data += data_buffer
            
            return audio_data if audio_data else None
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            st.error(f"TTS Error: {str(e)}")
            return None

class GeminiDocumentProcessor:
    """Main class for processing documents using Gemini API"""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini client"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            self.embedding_model = 'models/embedding-004'
            self.tts_processor = GeminiTTSProcessor(api_key)
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_text_from_image(self, image_file) -> str:
        """Extract text from image using Gemini Vision"""
        try:
            image = Image.open(image_file)
            prompt = """
            Please extract all the text content from this image. 
            Provide the text exactly as it appears, maintaining the original structure and formatting as much as possible.
            If there are multiple columns or sections, please preserve that organization.
            """
            
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def process_document(self, uploaded_file) -> str:
        """Process uploaded document and extract text"""
        try:
            file_type = uploaded_file.type
            
            if file_type == "application/pdf":
                return self.extract_text_from_pdf(uploaded_file)
            elif file_type in ["image/jpeg", "image/png", "image/jpg"]:
                return self.extract_text_from_image(uploaded_file)
            else:
                st.error(f"Unsupported file type: {file_type}")
                return ""
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            st.error(f"Error processing document: {str(e)}")
            return ""
    
    def generate_podcast_script(self, text: str, voice_style: str) -> str:
        """Generate podcast script from document text"""
        try:
            prompt = f"""
            You are an expert podcast scriptwriter. Create an engaging, conversational podcast script based on the following document content.
            
            Voice Style: {voice_style}
            
            Guidelines:
            1. Create a narrative that flows naturally as if being spoken aloud
            2. Include natural transitions and conversational elements
            3. Highlight key insights and interesting points from the document
            4. Make it engaging and easy to follow when listened to
            5. Aim for approximately 3-5 minutes of spoken content
            6. Use a tone appropriate for the {voice_style.lower()} voice style
            7. Include brief pauses and emphasis where appropriate (use punctuation to indicate)
            
            Document Content:
            {text}
            
            Please create a compelling podcast script that captures the essence of this document in an audio-friendly format.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating podcast script: {e}")
            return ""
    
    def text_to_speech(self, text: str, voice_style: str) -> Optional[bytes]:
        """Convert text to speech using Gemini TTS"""
        return self.tts_processor.text_to_speech(text, voice_style)
    
    def speech_to_text(self, audio_data: bytes) -> str:
        """Convert speech to text using Gemini STT"""
        try:
            # Note: Gemini doesn't have native STT yet
            st.warning("""
            **Note**: Gemini API doesn't currently support native Speech-to-Text functionality.
            
            To implement STT, you would need to:
            1. Use Google Cloud Speech-to-Text API
            2. Use an alternative STT service
            3. Wait for Gemini to add native STT support
            
            For now, please type your questions in the text input below.
            """)
            return ""
        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            return ""
    
    def create_embeddings(self, text_chunks: List[str]) -> List[List[float]]:
        """Create embeddings for text chunks using Gemini"""
        try:
            embeddings = []
            for chunk in text_chunks:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=chunk,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return []
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for RAG"""
        try:
            words = text.split()
            chunks = []
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk = ' '.join(words[i:i + chunk_size])
                if len(chunk.strip()) > 0:
                    chunks.append(chunk.strip())
            
            return chunks
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            return []
    
    def find_relevant_chunks(self, query: str, text_chunks: List[str], embeddings: List[List[float]], top_k: int = 3) -> List[str]:
        """Find most relevant text chunks for a query using cosine similarity"""
        try:
            # Create embedding for the query
            query_result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = query_result['embedding']
            
            # Calculate similarities
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            
            # Get top k most similar chunks
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            relevant_chunks = [text_chunks[i] for i in top_indices]
            
            return relevant_chunks
        except Exception as e:
            logger.error(f"Error finding relevant chunks: {e}")
            return []
    
    def generate_answer(self, query: str, relevant_chunks: List[str]) -> str:
        """Generate answer using RAG with relevant chunks"""
        try:
            context = "\n\n".join(relevant_chunks)
            
            prompt = f"""
            Based on the following context from the document, please answer the user's question.
            
            Context:
            {context}
            
            Question: {query}
            
            Instructions:
            1. Answer based primarily on the provided context
            2. If the context doesn't contain enough information, say so clearly
            3. Be concise but comprehensive
            4. Maintain accuracy and avoid speculation beyond the document content
            
            Answer:
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I'm sorry, I encountered an error while generating the answer."

def initialize_session_state():
    """Initialize session state variables"""
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""
    if 'text_chunks' not in st.session_state:
        st.session_state.text_chunks = []
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processor' not in st.session_state:
        st.session_state.processor = None

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">📚 myBook - AI Document Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize Gemini processor
    try:
        if st.session_state.processor is None:
            api_key = st.secrets["GEMINI_API_KEY"]
            st.session_state.processor = GeminiDocumentProcessor(api_key)
    except Exception as e:
        st.error(f"Error initializing Gemini API. Please check your API key configuration: {str(e)}")
        st.stop()
    
    # Sidebar for voice style selection
    with st.sidebar:
        st.header("🎙️ Voice Settings")
        voice_style = st.selectbox(
            "Select Voice Style:",
            ["Male", "Female", "Narrator"],
            index=0,
            help="Male: Puck voice, Female: Zephyr voice, Narrator: Sage voice"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Upload Document**: Start by uploading a PDF or image file
        2. **Generate Podcast**: Create an AI-generated podcast summary with TTS
        3. **Voice Q&A**: Ask questions about your document and hear the answers
        """)
        
        if st.session_state.document_processed:
            st.success("✅ Document processed successfully!")
        else:
            st.info("📄 Please upload a document to begin")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🎧 Podcast Generator", "💬 Voice Q&A"])
    
    with tab1:
        st.header("🎧 Podcast Generator")
        st.markdown("Upload a document and generate an engaging podcast summary with voice narration.")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Upload a PDF document or image file (PNG, JPG, JPEG)"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Process document button
            if st.button("📖 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        # Extract text from document
                        document_text = st.session_state.processor.process_document(uploaded_file)
                        
                        if document_text:
                            st.session_state.document_text = document_text
                            st.session_state.document_processed = True
                            
                            # Prepare for RAG
                            st.session_state.text_chunks = st.session_state.processor.split_text_into_chunks(document_text)
                            st.session_state.embeddings = st.session_state.processor.create_embeddings(st.session_state.text_chunks)
                            
                            st.markdown('<div class="success-message">✅ Document processed successfully!</div>', unsafe_allow_html=True)
                            
                            # Show document preview
                            with st.expander("📄 Document Preview", expanded=False):
                                st.text_area("Extracted Text:", document_text[:1000] + "..." if len(document_text) > 1000 else document_text, height=200)
                        else:
                            st.error("Failed to extract text from the document. Please try again.")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
            
            # Generate podcast script and audio
            if st.session_state.document_processed:
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🎙️ Generate Podcast Script", type="secondary"):
                        with st.spinner("Generating podcast script..."):
                            try:
                                script = st.session_state.processor.generate_podcast_script(
                                    st.session_state.document_text, 
                                    voice_style
                                )
                                
                                if script:
                                    st.session_state.podcast_script = script
                                    st.markdown("### 📝 Generated Podcast Script")
                                    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                                    st.markdown(script)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Download button for script
                                    st.download_button(
                                        label="📥 Download Podcast Script",
                                        data=script,
                                        file_name=f"podcast_script_{uploaded_file.name}.txt",
                                        mime="text/plain"
                                    )
                                else:
                                    st.error("Failed to generate podcast script. Please try again.")
                            except Exception as e:
                                st.error(f"Error generating podcast script: {str(e)}")
                
                with col2:
                    if hasattr(st.session_state, 'podcast_script'):
                        if st.button("🔊 Generate Audio", type="secondary"):
                            with st.spinner("Generating audio from script..."):
                                try:
                                    audio_data = st.session_state.processor.text_to_speech(
                                        st.session_state.podcast_script, 
                                        voice_style
                                    )
                                    
                                    if audio_data:
                                        st.success("✅ Audio generated successfully!")
                                        st.audio(audio_data, format="audio/wav")
                                        
                                        # Download button for audio
                                        st.download_button(
                                            label="📥 Download Audio",
                                            data=audio_data,
                                            file_name=f"podcast_audio_{uploaded_file.name}.wav",
                                            mime="audio/wav"
                                        )
                                    else:
                                        st.error("Failed to generate audio. Please try again.")
                                except Exception as e:
                                    st.error(f"Error generating audio: {str(e)}")
    
    with tab2:
        st.header("💬 Voice Q&A")
        st.markdown("Ask questions about your document and hear the answers spoken aloud.")
        
        if not st.session_state.document_processed:
            st.warning("⚠️ Please upload and process a document in the 'Podcast Generator' tab first.")
        else:
            st.success("✅ Document is ready for Q&A!")
            
            # Voice recording section
            st.markdown("### 🎤 Voice Input")
            
            # WebRTC audio recorder
            rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            
            audio_recorder = AudioRecorder()
            
            webrtc_ctx = webrtc_streamer(
                key="speech-to-text",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration=rtc_configuration,
                audio_frame_callback=audio_recorder.audio_frame_callback,
                media_stream_constraints={"video": False, "audio": True},
                async_processing=True,
            )
            
            if webrtc_ctx.audio_receiver:
                st.info("🎙️ Audio recorder is active. Click 'START' to begin recording your question.")
            
            # Text input as alternative
            st.markdown("### ⌨️ Text Input")
            user_question = st.text_input("Or type your question here:", placeholder="What is this document about?")
            
            # Process question
            if st.button("❓ Ask Question", type="primary"):
                question_to_process = user_question
                
                # If we have audio data, attempt to transcribe it
                if webrtc_ctx.audio_receiver:
                    audio_data = audio_recorder.get_audio_data()
                    if audio_data:
                        # Attempt STT (will show warning about limitation)
                        transcribed_text = st.session_state.processor.speech_to_text(audio_data)
                        if transcribed_text:
                            question_to_process = transcribed_text
                
                if question_to_process:
                    with st.spinner("Finding relevant information and generating answer..."):
                        try:
                            # Find relevant chunks using RAG
                            relevant_chunks = st.session_state.processor.find_relevant_chunks(
                                question_to_process,
                                st.session_state.text_chunks,
                                st.session_state.embeddings
                            )
                            
                            # Generate answer
                            answer = st.session_state.processor.generate_answer(question_to_process, relevant_chunks)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                "question": question_to_process,
                                "answer": answer,
                                "timestamp": time.strftime("%H:%M:%S")
                            })
                            
                            # Clear text input and rerun
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")
                else:
                    st.warning("Please provide a question either by typing or using voice input.")
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("### 💭 Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.container():
                        st.markdown(f"**🙋 Question ({chat['timestamp']}):** {chat['question']}")
                        st.markdown(f"**🤖 Answer:** {chat['answer']}")
                        
                        # TTS for answer
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if st.button(f"🔊 Play Answer", key=f"tts_{i}"):
                                with st.spinner("Generating audio..."):
                                    try:
                                        audio_data = st.session_state.processor.text_to_speech(chat['answer'], voice_style)
                                        if audio_data:
                                            st.audio(audio_data, format="audio/wav")
                                        else:
                                            st.error("Failed to generate audio")
                                    except Exception as e:
                                        st.error(f"Error generating audio: {str(e)}")
                        
                        st.markdown("---")
                
                # Clear chat history button
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()

if __name__ == "__main__":
    main()
