import streamlit as st
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import PyPDF2
import pickle
from datetime import datetime
import logging
import speech_recognition as sr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),  # Save to file
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Set page config as the first Streamlit command
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize Groq client
logger.info(f"Initializing Groq client with API key: {GROQ_API_KEY[:5]}... (truncated)")
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "chatbot-index"

if index_name not in pc.list_indexes().names():
    logger.info(f"Creating new Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
index = pc.Index(index_name)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Persistence functions
def save_chat_sessions(sessions):
    with open("chat_sessions.pkl", "wb") as f:
        pickle.dump(sessions, f)

def load_chat_sessions():
    try:
        with open("chat_sessions.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_uploaded_files(files):
    with open("uploaded_files.pkl", "wb") as f:
        pickle.dump(files, f)

def load_uploaded_files():
    try:
        with open("uploaded_files.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

# Process uploaded files (with Document Viewer support)
def process_file(file):
    logger.info(f"Processing file: {file.name}")
    if file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
    else:
        text = file.read().decode("utf-8")
    
    # Store full text for Document Viewer
    st.session_state.uploaded_docs[file.name] = text
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk)
        index.upsert([(f"{file.name}_chunk_{i}", embedding, {"text": chunk, "file_name": file.name})])
    logger.info(f"Upserted {len(chunks)} chunks to Pinecone for {file.name}")
    return len(chunks)

# Retrieve chunks with metadata
def retrieve_chunks(query, top_k=3):
    logger.info(f"Pinecone query: '{query}' with top_k={top_k}")
    query_embedding = embeddings.embed_query(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    retrieved_chunks = [(match["metadata"]["text"], match["metadata"].get("file_name", "Unknown Document")) 
                        for match in results["matches"]]
    logger.info(f"Retrieved {len(retrieved_chunks)} chunks from Pinecone")
    return retrieved_chunks

# Generate response with streaming
def generate_response(query, context, model):
    logger.info(f"Groq API call: model={model}, query='{query}', context length={len(context)}")
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response_stream = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        stream=True
    )
    return response_stream

# Voice input function (simplified)
def get_voice_input():
    if st.session_state.get("recording", False):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Recording... Speak now!")
            try:
                audio = r.listen(source, timeout=5)  # Record for up to 5 seconds
                text = r.recognize_google(audio)
                logger.info(f"Voice input transcribed: '{text}'")
                st.session_state.recording = False
                return text
            except sr.WaitTimeoutError:
                st.error("No audio detected within 5 seconds")
                logger.error("Voice input timeout")
            except sr.UnknownValueError:
                st.error("Couldn’t understand the audio")
                logger.error("Speech recognition failed: Unknown value")
            except sr.RequestError as e:
                st.error(f"Speech recognition error: {e}")
                logger.error(f"Speech recognition error: {e}")
        st.session_state.recording = False
    return None

# Streamlit UI
st.title("🤖 RAG Chatbot with Groq & Pinecone")

# Initialize session state
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions()
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = load_uploaded_files()
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "mixtral-8x7b-32768"
if "logs" not in st.session_state:
    st.session_state.logs = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}
if "recording" not in st.session_state:
    st.session_state.recording = False

# Log handler to capture logs in session state
class StreamlitLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        st.session_state.logs.append(msg)

logger.addHandler(StreamlitLogHandler())

# Available Groq models
groq_models = [
    "mixtral-8x7b-32768",
    "llama2-70b-4096",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "gemma-7b-it",
    "deepseek-r1-distill-llama-70b"
]

# Sidebar
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_file = st.file_uploader("Upload a document (PDF/TXT)", type=["pdf", "txt"])
    if uploaded_file:
        with st.spinner("Processing..."):
            chunk_count = process_file(uploaded_file)
            if uploaded_file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file.name)
                save_uploaded_files(st.session_state.uploaded_files)
            st.success(f"Processed {chunk_count} chunks from {uploaded_file.name}")

    st.subheader("Uploaded Documents")
    for file_name in st.session_state.uploaded_files:
        st.write(f"📄 {file_name}")

    if st.button("🗑️ Clear Documents & Index"):
        index.delete(delete_all=True)
        st.session_state.uploaded_files = []
        st.session_state.uploaded_docs = {}
        save_uploaded_files([])
        st.success("Documents and index cleared!")

    st.header("💬 Chat History")
    if st.button("➕ New Chat"):
        temp_id = f"New Chat - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        st.session_state.chat_sessions[temp_id] = []
        st.session_state.current_chat_id = temp_id
        save_chat_sessions(st.session_state.chat_sessions)
        st.success(f"Started new chat. Ask a question to name it!")

    if st.session_state.chat_sessions:
        for chat_id in st.session_state.chat_sessions.keys():
            if st.button(f"{chat_id}", key=chat_id):
                st.session_state.current_chat_id = chat_id

        if st.session_state.current_chat_id:
            chat_content = "\n".join([f"{m['role'].capitalize()}: {m['content']}" 
                                    for m in st.session_state.chat_sessions[st.session_state.current_chat_id]])
            st.download_button(
                label="📥 Download Chat",
                data=chat_content,
                file_name=f"{st.session_state.current_chat_id}.txt",
                mime="text/plain"
            )
    else:
        st.write("No chats yet. Start a new one!")

# Model selection dropdown
st.session_state.selected_model = st.selectbox("Select Model", groq_models, index=groq_models.index(st.session_state.selected_model))

# Chat Input (Text with Audio Button)
col1, col2 = st.columns([5, 1])
with col1:
    prompt = st.chat_input("Ask me anything!")
with col2:
    if st.button("🎙️ Record", key="record_audio"):
        st.session_state.recording = True
if st.session_state.recording:
    voice_prompt = get_voice_input()
    if voice_prompt:
        prompt = voice_prompt

# Main Chat Area
if st.session_state.current_chat_id:
    st.subheader(f"Current Chat: {st.session_state.current_chat_id}")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_sessions[st.session_state.current_chat_id]:
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])
else:
    st.write("Ask a question to start a chat, or use the 'New Chat' button in the sidebar!")

# Display logs in an expander
with st.expander("View Logs"):
    st.text("\n".join(st.session_state.logs[-10:]))  # Show last 10 logs for brevity

# Handle chat input with streaming
if prompt:
    if not st.session_state.current_chat_id:
        temp_id = f"New Chat - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        st.session_state.chat_sessions[temp_id] = []
        st.session_state.current_chat_id = temp_id
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    current_messages = st.session_state.chat_sessions[st.session_state.current_chat_id]
    if not current_messages:
        old_id = st.session_state.current_chat_id
        new_id = f"{prompt[:30]} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        st.session_state.chat_sessions[new_id] = st.session_state.chat_sessions.pop(old_id)
        st.session_state.current_chat_id = new_id
        save_chat_sessions(st.session_state.chat_sessions)
    
    st.session_state.chat_sessions[st.session_state.current_chat_id].append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            chunks_with_metadata = retrieve_chunks(prompt)
            context = "\n".join(chunk for chunk, _ in chunks_with_metadata)
            if not context:
                st.markdown("I don’t have enough context. Please upload a document!")
                full_response = "I don’t have enough context. Please upload a document!"
            else:
                response_stream = generate_response(prompt, context, st.session_state.selected_model)
                response_placeholder = st.empty()
                full_response = ""
                for chunk in response_stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                logger.info(f"Groq response completed: '{full_response[:50]}...'")
        
        # Document Viewer with citations
        if chunks_with_metadata:
            with st.expander("📜 Document Preview & Citations"):
                for chunk, file_name in chunks_with_metadata:
                    st.markdown(f"**From {file_name}:** _{chunk[:200]}..._")
                    if st.button(f"View Full {file_name}", key=f"view_{file_name}_{id(chunk)}"):
                        with st.expander(f"Full Document: {file_name}", expanded=True):
                            st.text(st.session_state.uploaded_docs.get(file_name, "Document not found"))
    
    st.session_state.chat_sessions[st.session_state.current_chat_id].append({"role": "assistant", "content": full_response})
    save_chat_sessions(st.session_state.chat_sessions)