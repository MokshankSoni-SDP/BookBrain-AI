import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from retriever import PhysicsRetriever
from ingest import ingest_data
from ui.styles import load_custom_css

# Load env declaration
load_dotenv()

# Config page
st.set_page_config(
    page_title="Physics Textbook RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom Styles
load_custom_css()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_retriever():
    return PhysicsRetriever()

# Initialize resources
if "retriever" not in st.session_state:
    with st.spinner("Initializing Retrieval Engine..."):
        try:
            st.session_state.retriever = get_retriever()
        except Exception as e:
            st.error(f"Failed to initialize retriever: {e}")

if "groq_client" not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.session_state.groq_client = Groq(api_key=api_key)
    else:
        st.error("GROQ_API_KEY not found in environment variables.")

# System Prompt
SYSTEM_PROMPT = """You are a Physics Education AI tutor with expertise in classical mechanics. You answer questions using ONLY the provided textbook excerpts.

**Response Guidelines**:
1. **Citation Requirement**: Always reference the specific section (e.g., "According to Section 6.1.1") when making claims
2. **Technical Accuracy**: Use precise physics terminology from the source material
3. **Pedagogical Tone**: Explain concepts progressively, building on definitions
4. **Admission of Limits**: If the context doesn't contain the answer, say "The provided sections don't cover [topic]. You may need to refer to Chapter X on [related topic]."
5. **Equation Formatting**: Present equations clearly, explaining each variable
"""

def format_contexts(chunks):
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.payload['metadata']
        location = f"Section {meta['section_number']}"
        if meta.get('subsection_number'):
            location += f".{meta['subsection_number']}"
        
        formatted.append(f"""
--- Context {i} ---
**Location**: {location} - {meta.get('subsection_title') or meta['section_title']}
**Content**:
{chunk.payload['text']}
""")
    return "\n".join(formatted)

# UI Layout
st.title("📚 Physics Textbook AI Tutor")
st.caption("Ask questions about Systems of Particles and Rotational Motion")

# Sidebar
with st.sidebar:
    st.title("⚙️ System Configuration")
    
    # 1. Database Status
    st.subheader("📊 Database Status")
    if "retriever" in st.session_state:
        retriever = st.session_state.retriever
        if retriever.check_connection():
            st.success("✅ Qdrant Connected")
            stats = retriever.get_collection_stats()
            st.markdown(f"""
            <div class="metric-container">
                <h3>Total Chunks</h3>
                <h1>{stats.get('vectors_count', 0)}</h1>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Qdrant Disconnected")
    else:
        st.warning("Retriever initializing...")

    # 2. Data Ingestion
    st.subheader("📥 Data Ingestion")
    uploaded_file = st.file_uploader(
        "Upload chapter_structure.json",
        type=['json'],
        help="Upload your textbook JSON file"
    )
    
    if uploaded_file and st.button("🚀 Process & Ingest"):
        with st.spinner("Processing..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Call ingestion pipeline
                ingest_data(
                    uploaded_file, 
                    progress_callback=progress_bar.progress,
                    status_callback=status_text.text
                )
                st.success("✅ Ingestion Complete!")
                st.cache_resource.clear() # Clear cache to reload retriever if needed
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    # 3. Retrieval Settings
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider("Initial Retrieval (K)", 10, 50, 20)
    final_chunks = st.slider("Final Context Chunks", 3, 15, 6)
    
    # 4. Advanced Options
    with st.expander("🛠️ Advanced"):
        # We can expose these if the lower-level functions support them dynamically
        # For now, just placeholder or read-only
        st.info(f"Embedding Model: {os.getenv('EMBEDDING_MODEL', 'nomic-ai/nomic-embed-text-v1.5')}")
        st.info(f"Reranker: {os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')}")

# Main Chat Interface
chat_container = st.container()

with chat_container:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show sources for assistant messages if available
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📖 View Sources"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                        <strong>Source {i}</strong>: Section {source['section_number']} 
                        {f"- {source['subsection_number']}" if source.get('subsection_number') else ''}
                        <br>
                        <em>{source['section_title']}</em>
                        <hr>
                        {source['text'][:300]}...
                        </div>
                        """, unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask a question about the chapter..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if "retriever" not in st.session_state or "groq_client" not in st.session_state:
            st.error("System not fully initialized.")
            st.stop()
            
        # Retrieval phase
        with st.spinner("🔍 Searching textbook..."):
            # Initial retrieval
            results = st.session_state.retriever.retrieve(prompt, top_k=top_k)
            # Reranking
            reranked_results = st.session_state.retriever.rerank(prompt, results, top_k=final_chunks)
            
            context_str = format_contexts(reranked_results)
        
        # Generation phase
        with st.spinner("✍️ Generating answer..."):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"**Question**: {prompt}\n\n**Relevant Textbook Excerpts**:\n{context_str}\n\n**Your Answer**:"}
            ]
            
            try:
                stream = st.session_state.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    stream=True,
                    temperature=0.3,
                    max_tokens=1024
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Extract and store sources
                sources = [
                    {
                        'section_number': c.payload['metadata']['section_number'],
                        'subsection_number': c.payload['metadata'].get('subsection_number'),
                        'section_title': c.payload['metadata']['section_title'],
                        'text': c.payload['text']
                    }
                    for c in reranked_results
                ]
                
                # Save interaction with sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
