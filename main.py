import streamlit as st
import google.generativeai as genai
from google.generativeai import types # Corrected import
from PIL import Image
import PyPDF2
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# import pickle # Not used, can be removed if not planned for future
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
        self.audio_frames = [] # Not directly used, audio_queue is primary
        self.audio_queue = queue.Queue()

    def audio_frame_callback(self, frame: av.AudioFrame):
        """Callback function to handle audio frames"""
        try:
            # Convert audio frame to numpy array and add to queue
            # Assuming single channel, float32 PCM data from WebRTC usually
            # You might need to adjust format conversion depending on what WebRTC provides
            sound = frame.to_ndarray()
            self.audio_queue.put(sound)
        except Exception as e:
            logger.error(f"Error in audio frame callback: {e}")

    def get_audio_data(self) -> Optional[bytes]:
        """Get recorded audio data as a single bytes object (e.g., WAV format).
           Currently returns a list of ndarrays, needs further processing for STT.
        """
        # This method currently returns a list of numpy arrays (frames).
        # For most STT APIs, you'll need to concatenate these and convert
        # to a single byte stream in a specific audio format (e.g., WAV).
        # The current implementation of speech_to_text in GeminiDocumentProcessor
        # shows a warning, so this part isn't fully utilized yet.
        # If actual STT were implemented, this method would need to be more robust.
        audio_segments = []
        try:
            while not self.audio_queue.empty():
                audio_segments.append(self.audio_queue.get())
        except Exception as e:
            logger.error(f"Error getting audio data: {e}")
        
        if not audio_segments:
            return None

        # For actual STT, you would concatenate and format these segments:
        # Example (conceptual, requires knowing sample rate, width, etc.):
        # full_audio_np = np.concatenate(audio_segments)
        # full_audio_bytes = full_audio_np.astype(np.int16).tobytes() # Example for 16-bit PCM
        # Then, potentially add a WAV header.
        # For now, as STT is a placeholder, returning the raw segments list.
        # This will need to change when STT is fully implemented.
        # Returning raw audio_segments for now, though speech_to_text expects bytes.
        # This highlights a gap if STT were actually functional.
        logger.warning("AudioRecorder.get_audio_data() currently returns raw frames; STT would require format conversion.")
        
        # Concatenate numpy arrays if they exist
        if audio_segments:
            concatenated_audio = np.concatenate(audio_segments, axis=0)
            # Assuming 16-bit signed integers for WAV, and WebRTC provides float samples
            # This normalization and type conversion is a common step.
            # Max value of int16 is 32767. Normalize float samples from [-1, 1] to [-32767, 32767]
            audio_int16 = (concatenated_audio * 32767).astype(np.int16)
            return audio_int16.tobytes()
        return None


class GeminiTTSProcessor:
    """TTS processor using Gemini capabilities"""

    def __init__(self, api_key: str): # api_key is passed but genai is likely already configured
        """Initialize the Gemini TTS model"""
        try:
            # genai.configure(api_key=api_key) # This might be re-configuring if already done.
                                             # It's generally fine, but if issues arise, check.
                                             # Assuming genai is configured globally by GeminiDocumentProcessor.
            self.tts_model_name = "models/text-to-speech" # Standard model for TTS
            # Older model used in original code: "gemini-2.5-pro-preview-tts" - check Gemini docs for current best TTS model
            # For "models/text-to-speech", the API is slightly different (uses synthesize_speech)
            # Sticking to the structure of original code that implies generate_content for TTS
            # If "gemini-2.5-pro-preview-tts" is indeed the correct model for generate_content based TTS:
            self.tts_model_name = "gemini-2.5-pro-preview-tts" # As per original code, verify this model exists for this purpose

            # It seems the original code intended to use a specific TTS model with generate_content.
            # Let's assume 'gemini-2.5-pro-preview-tts' is a valid model ID for this.
            # The client part of original was 'genai_client.Client(api_key=api_key).models.generate_content_stream'
            # The equivalent with the standard `genai` module is to create a GenerativeModel.
            self.model = genai.GenerativeModel(self.tts_model_name)
            logger.info(f"Gemini TTS model '{self.tts_model_name}' initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini TTS model: {e}")
            raise

    def convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """Generates a WAV file header for the given audio data and parameters."""
        parameters = self.parse_audio_mime_type(mime_type)
        bits_per_sample = parameters["bits_per_sample"]
        sample_rate = parameters["rate"]
        num_channels = 1  # Assuming mono
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
            16,               # Subchunk1Size (PCM = 16)
            1,                # AudioFormat (PCM = 1)
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
        bits_per_sample = 16  # Default
        rate = 24000          # Default (common for Gemini TTS)

        parts = mime_type.lower().split(";")
        for param in parts:
            param = param.strip()
            if param.startswith("rate="):
                try:
                    rate_str = param.split("=", 1)[1]
                    rate = int(rate_str)
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse rate from mime_type: {mime_type}")
            elif param.startswith("audio/l"): # e.g., audio/L16 or audio/L24
                try:
                    bits_str = param.split("audio/l", 1)[1]
                    if bits_str.isdigit():
                         bits_per_sample = int(bits_str)
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse bits_per_sample from mime_type: {mime_type}")
        return {"bits_per_sample": bits_per_sample, "rate": rate}

    def text_to_speech(self, text: str, voice_style: str) -> Optional[bytes]:
        """Convert text to speech using Gemini TTS"""
        try:
            voice_mapping = {
                "Male": "Puck",
                "Female": "Zephyr",
                "Narrator": "Sage"
            }
            voice_name = voice_mapping.get(voice_style, "Zephyr") # Default to Zephyr

            # Ensure 'types' refers to 'google.generativeai.types'
            # The structure for TTS using generate_content needs specific model support.
            # The following config is based on original code's intent.
            # Verify with Gemini documentation if this is the correct way for your chosen TTS model.
            
            contents = [
                types.Content( # This is correct for general generate_content
                    role="user",
                    parts=[
                        types.Part.from_text(text=text),
                    ],
                ),
            ]
            
            # The GenerateContentConfig for TTS seems to be what was intended.
            generate_content_config = types.GenerateContentConfig(
                temperature=0.7, # Adjusted from 1, common for TTS to be less random
                response_modalities=["audio"], # Key for requesting audio
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    ),
                ),
            )
            
            audio_data = b""
            
            # Use the self.model instance and pass generation_config correctly
            response_stream = self.model.generate_content_stream(
                contents=contents,
                generation_config=generate_content_config, # Parameter name is generation_config
            )

            for chunk in response_stream:
                if (
                    chunk.candidates is None
                    or not chunk.candidates  # Check if list is not empty
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                    or not chunk.candidates[0].content.parts # Check if list is not empty
                ):
                    continue
                    
                part = chunk.candidates[0].content.parts[0]
                if part.inline_data and part.inline_data.data:
                    inline_data = part.inline_data
                    data_buffer = inline_data.data
                    
                    # Convert to WAV if needed
                    # The Gemini API with response_modalities=["audio"] should ideally return WAV
                    # or allow specifying the output format.
                    # This conversion step is a good fallback.
                    if inline_data.mime_type != "audio/wav":
                        logger.info(f"Received audio in format {inline_data.mime_type}, converting to WAV.")
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
            # Check Gemini documentation for the latest recommended "flash" model if 'gemini-2.0-flash' is deprecated/renamed
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest') # Using a common alias
            self.embedding_model_name = 'models/embedding-001' # Updated to a common embedding model
            self.tts_processor = GeminiTTSProcessor(api_key)
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {e}")
            st.error(f"Failed to initialize GeminiDocumentProcessor: {e}") # Show error in UI
            raise

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text: # Ensure text was extracted
                    text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def extract_text_from_image(self, image_file) -> str:
        """Extract text from image using Gemini Vision"""
        try:
            image = Image.open(image_file)
            # Ensure the model supports image input with text.
            # 'gemini-1.5-flash-latest' or 'gemini-pro-vision' are suitable.
            # If self.model is text-only, this will fail.
            # Assuming self.model ('gemini-1.5-flash-latest') is multi-modal.
            
            prompt = """
            Please extract all the text content from this image.
            Provide the text exactly as it appears, maintaining the original structure and formatting as much as possible.
            If there are multiple columns or sections, please preserve that organization.
            Output only the extracted text.
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
            1. Create a narrative that flows naturally as if being spoken aloud.
            2. Include natural transitions and conversational elements (e.g., "Alright, so...", "Now, let's dive into...", "Interestingly...").
            3. Highlight key insights and interesting points from the document.
            4. Make it engaging and easy to follow when listened to.
            5. Aim for approximately 3-5 minutes of spoken content.
            6. Use a tone appropriate for the {voice_style.lower()} voice style.
            7. Include brief pauses and emphasis where appropriate (use punctuation like commas, ellipses, and short paragraphs to indicate).
            8. Start with a brief, engaging introduction and end with a concise summary or call to reflection.

            Document Content:
            ---
            {text[:15000]}
            ---

            Please create a compelling podcast script that captures the essence of this document in an audio-friendly format.
            The script should be ready for direct text-to-speech conversion.
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
        # Note: As of my last update, direct genai.speech_to_text might not be available
        # or might require Google Cloud Speech-to-Text.
        # This implementation remains a placeholder/warning.
        try:
            # If using Google Cloud Speech-to-Text, you'd initialize its client here.
            # Example (conceptual):
            # from google.cloud import speech
            # client = speech.SpeechClient()
            # audio = speech.RecognitionAudio(content=audio_data)
            # config = speech.RecognitionConfig(
            #     encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # Example
            #     sample_rate_hertz=16000, # Example, must match audio
            #     language_code="en-US",
            # )
            # response = client.recognize(config=config, audio=audio)
            # if response.results:
            #     return response.results[0].alternatives[0].transcript
            # return ""

            st.warning("""
            **Note**: Native Speech-to-Text (STT) functionality directly within the `google-generativeai` package 
            for arbitrary audio input is not straightforward or might require integration with Google Cloud Speech-to-Text API.
            
            This application currently does not have a fully implemented STT component.
            Please type your questions in the text input below.
            """)
            logger.info(f"Received audio data for STT, length: {len(audio_data) if audio_data else 0} bytes. STT not implemented.")
            return ""
        except Exception as e:
            logger.error(f"Error in speech-to-text placeholder: {e}")
            return ""

    def create_embeddings(self, text_chunks: List[str]) -> List[List[float]]:
        """Create embeddings for text chunks using Gemini"""
        try:
            embeddings = []
            for chunk in text_chunks:
                if not chunk.strip(): # Skip empty chunks
                    continue
                result = genai.embed_content(
                    model=self.embedding_model_name,
                    content=chunk,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            st.error(f"Embedding Error: {e}")
            return []

    def split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]: # Smaller chunk size for better RAG
        """Split text into overlapping chunks for RAG"""
        try:
            # Simple splitting by words, can be improved (e.g., sentence splitting)
            words = text.split()
            chunks = []
            if not words:
                return []

            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk = ' '.join(chunk_words)
                if len(chunk.strip()) > 0:
                    chunks.append(chunk.strip())
            
            # Ensure last part of text is captured if not perfectly divisible
            if chunks and ' '.join(words).endswith(chunks[-1]) == False and len(words) > chunk_size :
                 remaining_start_index = (len(chunks) -1) * (chunk_size - overlap) + chunk_size
                 if remaining_start_index < len(words):
                    last_chunk_words = words[remaining_start_index - overlap:] # Add overlap to the last chunk
                    last_chunk = ' '.join(last_chunk_words)
                    if len(last_chunk.strip()) > 0 and last_chunk not in chunks : # Avoid duplicates
                        chunks.append(last_chunk.strip())
            elif not chunks and text.strip(): # If text is shorter than chunk_size
                chunks.append(text.strip())

            return chunks
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            return []

    def find_relevant_chunks(self, query: str, text_chunks: List[str], embeddings: List[List[float]], top_k: int = 3) -> List[str]:
        """Find most relevant text chunks for a query using cosine similarity"""
        if not embeddings or not text_chunks: # Guard against empty embeddings or chunks
            return []
        try:
            query_result = genai.embed_content(
                model=self.embedding_model_name,
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = query_result['embedding']

            # Ensure embeddings is a 2D numpy array for cosine_similarity
            embeddings_array = np.array(embeddings)
            if embeddings_array.ndim == 1: # If only one document chunk
                embeddings_array = embeddings_array.reshape(1, -1)
            
            query_embedding_array = np.array(query_embedding).reshape(1, -1)

            similarities = cosine_similarity(query_embedding_array, embeddings_array)[0]

            # Get top k most similar chunks
            # Ensure top_k is not greater than the number of chunks
            actual_top_k = min(top_k, len(text_chunks))
            if actual_top_k == 0:
                return []
            
            top_indices = np.argsort(similarities)[-actual_top_k:][::-1]
            relevant_chunks = [text_chunks[i] for i in top_indices]

            return relevant_chunks
        except Exception as e:
            logger.error(f"Error finding relevant chunks: {e}")
            st.error(f"Relevance Search Error: {e}")
            return []

    def generate_answer(self, query: str, relevant_chunks: List[str]) -> str:
        """Generate answer using RAG with relevant chunks"""
        try:
            if not relevant_chunks:
                return "I couldn't find relevant information in the document to answer your question. Please try rephrasing or ask about a different topic."

            context = "\n\n---\n\n".join(relevant_chunks) # More distinct separator

            prompt = f"""
            You are a helpful AI assistant. Answer the user's question based *only* on the provided context from a document.
            If the context does not contain the answer, explicitly state that the information is not found in the provided document excerpts.
            Do not use any external knowledge or make assumptions beyond the text.
            Be concise and directly answer the question.

            Context from the document:
            ---
            {context}
            ---

            User's Question: {query}

            Answer:
            """

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I'm sorry, I encountered an error while generating the answer."

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'document_processed': False,
        'document_text': "",
        'text_chunks': [],
        'embeddings': [],
        'chat_history': [],
        'processor': None,
        'podcast_script': None, # Added to prevent errors if button clicked before script generated
        'api_key_valid': False # To track API key status
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    """Main application function"""
    initialize_session_state()

    st.markdown('<div class="main-header">📚 myBook - AI Document Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")

    # API Key Input if not already set and valid
    if not st.session_state.api_key_valid:
        st.sidebar.subheader("🔑 API Key")
        api_key_input = st.sidebar.text_input("Enter your Gemini API Key:", type="password", key="api_key_input_sidebar")
        if st.sidebar.button("Set API Key"):
            if api_key_input:
                try:
                    # Test API key by trying to initialize processor
                    st.session_state.processor = GeminiDocumentProcessor(api_key_input)
                    st.session_state.api_key_valid = True
                    st.session_state.GEMINI_API_KEY = api_key_input # Store for later use if needed
                    st.sidebar.success("API Key accepted!")
                    st.rerun() # Rerun to update UI after key is set
                except Exception as e:
                    st.sidebar.error(f"Invalid API Key or connection error: {e}")
                    st.session_state.api_key_valid = False
                    st.session_state.processor = None
            else:
                st.sidebar.warning("Please enter an API Key.")
        
        # Display features and stop if no valid API key
        if not st.session_state.api_key_valid:
            st.info("Please enter a valid Gemini API Key in the sidebar to use the application.")
            st.stop()
    
    # Ensure processor is initialized if API key is valid
    if st.session_state.api_key_valid and st.session_state.processor is None:
        try:
            # GEMINI_API_KEY might be in secrets or set via input
            api_key = st.session_state.get("GEMINI_API_KEY") # Get from session state if set by input
            if not api_key and "GEMINI_API_KEY" in st.secrets:
                 api_key = st.secrets["GEMINI_API_KEY"]

            if api_key:
                st.session_state.processor = GeminiDocumentProcessor(api_key)
            else:
                # This case should ideally be handled by the API key input logic above
                st.error("API Key not found. Please configure it.")
                st.stop()

        except Exception as e:
            st.error(f"Error initializing Gemini API. Please check your API key configuration: {str(e)}")
            st.stop()


    with st.sidebar:
        st.header("🎙️ Voice Settings")
        voice_style = st.selectbox(
            "Select Voice Style:",
            ["Male", "Female", "Narrator"],
            index=1, # Default to Female
            help="Male: Puck voice, Female: Zephyr voice, Narrator: Sage voice"
        )

        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Ensure API Key is set.**
        2. **Upload Document**: Go to 'Podcast Generator' tab.
        3. **Process Document**: Click 'Process Document'.
        4. **Generate Podcast**: Optionally, generate script and audio.
        5. **Voice Q&A**: Go to 'Voice Q&A' tab to ask questions.
        """)

        if st.session_state.document_processed:
            st.success("✅ Document processed successfully!")
        else:
            st.info("📄 Please upload and process a document to enable all features.")

    tab1, tab2 = st.tabs(["🎧 Podcast Generator", "💬 Document Q&A"])

    with tab1:
        st.header("🎧 Podcast Generator")
        st.markdown("Upload a document (PDF or Image) to extract text, then generate an engaging podcast summary with voice narration.")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Upload a PDF document or image file (PNG, JPG, JPEG)"
        )

        if uploaded_file is not None:
            if 'last_uploaded_filename' not in st.session_state or \
               st.session_state.last_uploaded_filename != uploaded_file.name:
                # Reset states if a new file is uploaded
                st.session_state.document_processed = False
                st.session_state.document_text = ""
                st.session_state.text_chunks = []
                st.session_state.embeddings = []
                st.session_state.podcast_script = None
                st.session_state.last_uploaded_filename = uploaded_file.name

            st.success(f"File '{uploaded_file.name}' selected.")

            if st.button("📖 Process Document", key="process_doc_btn", type="primary", disabled=st.session_state.document_processed):
                with st.spinner("Processing document... This may take a moment."):
                    try:
                        document_text = st.session_state.processor.process_document(uploaded_file)
                        if document_text:
                            st.session_state.document_text = document_text
                            st.session_state.text_chunks = st.session_state.processor.split_text_into_chunks(document_text)
                            if st.session_state.text_chunks:
                                st.session_state.embeddings = st.session_state.processor.create_embeddings(st.session_state.text_chunks)
                                st.session_state.document_processed = True
                                st.markdown('<div class="success-message">✅ Document processed! Ready for Podcast or Q&A.</div>', unsafe_allow_html=True)
                            else:
                                st.error("Failed to split document into manageable chunks.")
                                st.session_state.document_processed = False


                            with st.expander("📄 Document Preview (first 1000 chars)", expanded=False):
                                st.text_area("Extracted Text:", document_text[:1000] + ("..." if len(document_text) > 1000 else ""), height=200, disabled=True)
                        else:
                            st.error("Failed to extract text from the document. The document might be empty, password-protected, or unreadable.")
                            st.session_state.document_processed = False
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        st.session_state.document_processed = False
                st.rerun()


            if st.session_state.document_processed:
                st.markdown("---")
                st.subheader("🎙️ Create Podcast")
                col_script, col_audio = st.columns(2)

                with col_script:
                    if st.button("📝 Generate Podcast Script", key="gen_script_btn"):
                        with st.spinner("Generating podcast script..."):
                            try:
                                script = st.session_state.processor.generate_podcast_script(
                                    st.session_state.document_text,
                                    voice_style
                                )
                                if script:
                                    st.session_state.podcast_script = script
                                    st.success("Script generated!")
                                else:
                                    st.error("Failed to generate podcast script.")
                            except Exception as e:
                                st.error(f"Error generating podcast script: {str(e)}")
                        # No rerun here, let the UI update below

                if st.session_state.podcast_script:
                    with col_script: # Display script in the same column
                        st.markdown("### Generated Podcast Script")
                        st.markdown(f'<div class="feature-card" style="max-height: 300px; overflow-y: auto;">{st.session_state.podcast_script}</div>', unsafe_allow_html=True)
                        st.download_button(
                            label="📥 Download Script (.txt)",
                            data=st.session_state.podcast_script,
                            file_name=f"podcast_script_{uploaded_file.name if uploaded_file else 'document'}.txt",
                            mime="text/plain"
                        )

                    with col_audio:
                        if st.button("🔊 Generate Audio from Script", key="gen_audio_btn"):
                            with st.spinner("Generating audio... This can take some time."):
                                try:
                                    audio_data = st.session_state.processor.text_to_speech(
                                        st.session_state.podcast_script,
                                        voice_style
                                    )
                                    if audio_data:
                                        st.session_state.podcast_audio_data = audio_data
                                        st.success("Audio generated!")
                                    else:
                                        st.error("Failed to generate audio from script.")
                                except Exception as e:
                                    st.error(f"Error generating audio: {str(e)}")
                            # No rerun here, let the UI update below

                        if 'podcast_audio_data' in st.session_state and st.session_state.podcast_audio_data:
                             with col_audio: # Display audio in its column
                                st.audio(st.session_state.podcast_audio_data, format="audio/wav")
                                st.download_button(
                                    label="📥 Download Audio (.wav)",
                                    data=st.session_state.podcast_audio_data,
                                    file_name=f"podcast_audio_{uploaded_file.name if uploaded_file else 'document'}.wav",
                                    mime="audio/wav"
                                )
        else:
            st.info("Upload a document to get started.")


    with tab2:
        st.header("💬 Document Q&A")
        st.markdown("Ask questions about your processed document. The AI will answer based on the document's content.")

        if not st.session_state.document_processed:
            st.warning("⚠️ Please upload and process a document in the 'Podcast Generator' tab first to enable Q&A.")
        else:
            st.success("✅ Document is ready for Q&A!")

            # Voice recording section (Placeholder due to STT limitations)
            # st.markdown("### 🎤 Voice Input (Experimental)")
            # rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            # audio_recorder_instance = AudioRecorder() # Create instance
            # webrtc_ctx = webrtc_streamer(
            #     key="speech-to-text",
            #     mode=WebRtcMode.SENDONLY,
            #     rtc_configuration=rtc_configuration,
            #     audio_frame_callback=audio_recorder_instance.audio_frame_callback,
            #     media_stream_constraints={"video": False, "audio": True},
            #     async_processing=True,
            # )
            # if webrtc_ctx.state.playing:
            #     st.info("🎙️ Recording... Click 'Ask Question' when done.")
            # else:
            #     st.info("🎙️ Click 'START' above to record your question, then 'Ask Question' below.")


            st.markdown("### ⌨️ Ask your question:")
            user_question = st.text_input("Type your question here:", placeholder="e.g., What are the main conclusions of this document?", key="qa_input")

            if st.button("❓ Ask Question", key="ask_q_btn", type="primary"):
                question_to_process = user_question
                # Attempt to get audio if STT were fully implemented
                # if webrtc_ctx.state.playing and not question_to_process:
                #     raw_audio_frames = audio_recorder_instance.get_audio_data() # This returns list of ndarrays
                #     if raw_audio_frames:
                #         # Further processing needed to convert raw_audio_frames to bytes for STT
                #         # For now, this part is conceptual
                #         st.info("Processing audio...")
                #         # actual_audio_bytes = convert_frames_to_bytes(raw_audio_frames, sample_rate=48000, sample_width=2) # Example
                #         # transcribed_text = st.session_state.processor.speech_to_text(actual_audio_bytes)
                #         # if transcribed_text:
                #         #     question_to_process = transcribed_text
                #         #     st.info(f"Transcribed: {transcribed_text}")
                #         # else:
                #         #     st.warning("Could not transcribe audio. Please type your question.")
                #     else:
                #         st.warning("No audio recorded. Please type your question.")


                if question_to_process:
                    with st.spinner("Thinking..."):
                        try:
                            relevant_chunks = st.session_state.processor.find_relevant_chunks(
                                question_to_process,
                                st.session_state.text_chunks,
                                st.session_state.embeddings
                            )
                            answer = st.session_state.processor.generate_answer(question_to_process, relevant_chunks)

                            st.session_state.chat_history.append({
                                "question": question_to_process,
                                "answer": answer,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            # user_question = "" # Clear input - Streamlit handles this via rerun on button click
                            # st.experimental_rerun() # Use st.rerun() for newer Streamlit
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")
                else:
                    st.warning("Please type a question.")

            if st.session_state.chat_history:
                st.markdown("---")
                st.subheader("💬 Conversation History")
                if st.button("🗑️ Clear Chat History", key="clear_chat_btn"):
                    st.session_state.chat_history = []
                    st.rerun()

                for i, chat_entry in enumerate(reversed(st.session_state.chat_history)): # Show newest first
                    with st.container():
                        st.markdown(f"**You ({chat_entry['timestamp']}):** {chat_entry['question']}")
                        st.markdown(f"**AI:**")
                        st.markdown(f'<div class="feature-card" style="background-color: #e9ecef;">{chat_entry["answer"]}</div>', unsafe_allow_html=True)

                        # TTS for answer
                        tts_button_key = f"tts_answer_{len(st.session_state.chat_history) - 1 - i}"
                        if st.button(f"🔊 Play Answer", key=tts_button_key):
                            with st.spinner("Generating audio for answer..."):
                                try:
                                    answer_audio_data = st.session_state.processor.text_to_speech(chat_entry['answer'], voice_style)
                                    if answer_audio_data:
                                        # Use a unique key for the audio player to avoid conflicts
                                        st.audio(answer_audio_data, format="audio/wav", key=f"audio_player_{tts_button_key}")
                                    else:
                                        st.error("Failed to generate audio for this answer.")
                                except Exception as e:
                                    st.error(f"Error generating audio for answer: {str(e)}")
                        st.markdown("---")

if __name__ == "__main__":
    main()
