import streamlit as st
import os
import time
from dotenv import load_dotenv
from groq import Groq

from qdrant_client import QdrantClient

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

@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    try:
        path = "./qdrant_data"
        print(f"[DEBUG] Initializing QdrantClient with path: {os.path.abspath(path)}")
        client = QdrantClient(path=path)
        print(f"[DEBUG] QdrantClient initialized successfully. Collections: {client.get_collections()}")
        return client
    except Exception as e:
        print(f"[ERROR] Failed to initialize QdrantClient: {e}")
        raise e

@st.cache_resource(show_spinner=False)
def get_retriever():
    client = get_qdrant_client()
    return PhysicsRetriever(client)

# Initialize resources
if "retriever" not in st.session_state:
    with st.spinner("Initializing Retrieval Engine..."):
        try:
            print("[DEBUG] Calling get_retriever()...")
            st.session_state.retriever = get_retriever()
            print("[DEBUG] Retriever stored in session state.")
            st.success("Retriever initialized (Cached if re-running).")
        except Exception as e:
            print(f"[ERROR] Retriever initialization failed: {e}")
            st.error(f"Failed to initialize retriever: {e}")

@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

if "groq_client" not in st.session_state:
    client = get_groq_client()
    if client:
        st.session_state.groq_client = client
    else:
        st.error("GROQ_API_KEY not found in environment variables.")

# System Prompt
# System Prompt
SYSTEM_PROMPT = """
You are a Physics Textbook AI Tutor.

Your role is to generate structured, student-friendly explanations STRICTLY using the provided textbook excerpts.

--------------------------------
CORE RULES
--------------------------------
1. You may ONLY use information present in the provided excerpts.
2. You MAY synthesize and connect ideas across excerpts.
3. You MUST NOT introduce external knowledge, extra formulas, or assumptions.
4. If information is missing, clearly state:
   "The provided textbook excerpts do not contain enough information to answer this completely."

--------------------------------
ANSWER STRUCTURE RULES
--------------------------------
Adapt your response based on the question type:

• If the user asks for a definition:
  - Provide a concise definition.
  - Include formula if given in text.
  - Do NOT include unnecessary diagrams.

• If the user asks to explain:
  - Use structured sections:
      1. Definition
      2. Conceptual Meaning
      3. Mathematical Expression (if present)
      4. Physical Significance
  - Only reference figures if they directly help understanding.

• If derivation or mathematical explanation is requested:
  - Present step-by-step equations exactly as given in the text.
  - Do not invent intermediate steps not present in excerpts.

--------------------------------
FIGURE RULES
--------------------------------
If a figure is explicitly useful for understanding:
- Refer to it exactly as written in the excerpt (e.g., "Fig. 6.7").
- Do NOT describe images that are not mentioned in the context.
- Do NOT force image references.

--------------------------------
STYLE REQUIREMENTS
--------------------------------
• Use headings and subheadings.
• Use LaTeX formatting for equations.
• Be clear, structured, and pedagogical.
• Avoid robotic repetition of chunk text.
• Make explanations readable and engaging for students.

Always begin with:
"Based on [Location]..."

"""

def get_relevant_images(image_refs, image_folder="extract_images"):
    """
    Finds actual image files for a list of image references (e.g., ['fig_6_2']).
    Handles fuzzy matching for sub-parts (e.g., fig_6_2 matches fig_6_2_a.png).
    """
    found_images = []
    if not os.path.exists(image_folder):
        return []
        
    all_files = os.listdir(image_folder)
    
    for ref in image_refs:
        # Normalize ref: fig 6.3 -> fig_6_3 just in case, though ingest does it too
        base_name = ref.lower().replace(" ", "_").replace(".", "_")
        
        # Match files starting with base_name
        # Custom Exact Match Logic
        # e.g. ref="fig_6_2" matches "fig_6_2.png" but NOT "fig_6_20.png"
        for f in all_files:
            # Normalize file name to compare basenames
            f_base = os.path.splitext(f)[0].lower().replace(" ", "_").replace(".", "_")
            if f_base == base_name:
                found_images.append(os.path.join(image_folder, f))
                
    return sorted(list(set(found_images)))

def inject_images_in_text(response_text, image_paths):
    """
     intelligently injects markdown images into the response text 
    where figure references (e.g., "Fig. 6.7") are detected.
    """
    for path in image_paths:
        # Expected filenames: fig_6_7.png -> 6.7
        base_name = os.path.basename(path)
        name_no_ext = os.path.splitext(base_name)[0]
        
        # Create variations to match LLM output
        # e.g., fig_6_7 -> fig. 6.7, Fig. 6.7, Figure 6.7
        parts = name_no_ext.split('_')
        if len(parts) >= 2 and parts[0].lower().startswith('fig'):
            # Construct "6.7" from parts[1:]
            number_part = ".".join(parts[1:]) 
            
            # Patterns to look for in text
            patterns = [
                f"Fig. {number_part}", 
                f"Figure {number_part}",
                f"Fig {number_part}",
                f"Figure {number_part}"
            ]
            
            image_markdown = f"\n\n![{base_name}]({path})\n\n"
            
            injected = False
            for pattern in patterns:
                if pattern in response_text:
                    # Replace the FIRST occurrence to inject image
                    # We use a placeholder to avoid infinite replacements if we just replaced 'pattern'
                    response_text = response_text.replace(pattern, f"{pattern}\n{image_markdown}", 1)
                    injected = True
                    break
            
            # Fallback: if not found but highly relevant (in unique_images), 
            # we might want to append it? 
            # Current instruction: "Inject image when figure tag detected"
            # So if not detected, we DON'T inject.
            
    return response_text

def format_contexts(chunks):
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.payload['metadata']
        location = f"Section {meta['section_number']}"
        if meta.get('subsection_number'):
            location += f".{meta['subsection_number']}"
            
        # Check for images
        image_note = ""
        if meta.get('image_refs'):
            image_note = "\n[Visual Context: Relevant images/diagrams are available to the user for this section.]"
        
        formatted.append(f"""
--- Context {i} ---
**Location**: {location} - {meta.get('subsection_title') or meta['section_title']}
**Content**:
{chunk.payload['text']}{image_note}
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
        "Upload Textbook Chapter (PDF)",
        type=['pdf'],
        help="Upload a PDF chapter to process and ingest"
    )
    
    if uploaded_file and st.button("🚀 Process & Ingest"):
        with st.spinner("Processing Pipeline..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 2. Save uploaded PDF to temp file
                status_text.text("Saving uploaded file...")
                temp_pdf_path = f"temp_{uploaded_file.name}"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 3. Run Pipeline
                status_text.text("Running Extraction & Classification...")
                progress_bar.progress(10)
                
                # Import here to avoid circular imports if any
                from pipeline import run_pdf_pipeline
                import torch
                import gc
                
                # Get shared client
                client = get_qdrant_client()

                # Define callbacks
                def update_status(msg):
                    status_text.text(msg)
                
                def update_progress(val):
                    if isinstance(val, (int, float)) and 0 <= val <= 100:
                        progress_bar.progress(int(val))

                # Execute pipeline using shared client
                run_dir, json_path, images_dir = run_pdf_pipeline(
                    temp_pdf_path, 
                    client=client,
                    status_callback=update_status,
                    progress_callback=update_progress
                )
                
                progress_bar.progress(80)
                status_text.text("Cleaning up resources...")
                
                # CRITICAL: Force GPU to release memory after ingestion
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Clean up temp PDF
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
                    
                status_text.text("Ingestion Complete. Reloading Retriever...")
                progress_bar.progress(100)
                
                # 4. Refresh retriever
                st.session_state.retriever = get_retriever()
                st.success(f"✅ Processing Complete! Images saved to {images_dir}")
                
            except Exception as e:
                st.error(f"Pipeline failed: {e}") 
                # Attempt to re-init retriever if it failed mid-way so app isn't broken
                if "retriever" not in st.session_state:
                     st.session_state.retriever = get_retriever()

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
            if msg["role"] == "assistant":
                if msg.get("sources"):
                    with st.expander("📖 View Sources (Retrieved Chunks)"):
                        for i, source in enumerate(msg["sources"], 1):
                            st.markdown(f"""
                            <div class="source-box">
                            <strong>Source {i}</strong>: Section {source['section_number']} 
                            {f"- {source['subsection_number']}" if source.get('subsection_number') else ''}
                            <br>
                            <em>{source['section_title']}</em>
                            <hr>
                            {source['text']}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show raw context if available
                if msg.get("context"):
                    with st.expander("🔍 Debug: View Context passed to LLM"):
                        st.code(msg["context"])

                # Show raw context if available
                if msg.get("context"):
                    with st.expander("� Debug: View Context passed to LLM"):
                        st.code(msg["context"])

                # Images are now injected into the text, so no need to show them separately here.

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
            t0 = time.time()
            # Initial retrieval
            results = st.session_state.retriever.retrieve(prompt, top_k=top_k)
            t1 = time.time()
            retrieval_time = t1 - t0
            
            # Reranking
            reranked_results = st.session_state.retriever.rerank(prompt, results, top_k=final_chunks)
            t2 = time.time()
            rerank_time = t2 - t1
            
            context_str = format_contexts(reranked_results)
            t3 = time.time()
            context_time = t3 - t2
        
        # Generation phase
        with st.spinner("✍️ Generating answer..."):
            
            # Adaptive Depth Instruction
            question_type_instruction = """
            Classify the question internally as one of:
            - Definition
            - Conceptual Explanation
            - Mathematical Derivation
            - Example Problem

            Then structure the answer accordingly.
            """

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"""
{question_type_instruction}

**Question**: {prompt}

**Relevant Textbook Excerpts**:
{context_str}

**Your Answer**:
"""}
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
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                # Display final response (Processed)
                
                # Manual Token Estimation (Fallback)
                # Approximation: 1 token ~= 4 chars (english)
                input_text = ""
                for m in messages:
                    input_text += m["content"]
                
                prompt_tokens_est = len(input_text) // 4
                completion_tokens_est = len(full_response) // 4
                total_tokens_est = prompt_tokens_est + completion_tokens_est
                
                st.caption(f"🪙 **Token Usage (Est.)**: Input: ~{prompt_tokens_est} | Output: ~{completion_tokens_est} | Total: ~{total_tokens_est}")
                
                 # Performance Metrics
                with st.expander("⏱️ Performance Metrics", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Retrieval (Qdrant)", f"{retrieval_time:.2f}s")
                    col2.metric("Reranking", f"{rerank_time:.2f}s")
                    col3.metric("Context Prep", f"{context_time:.2f}s")
                    col4.metric("LLM Generation", f"{generation_time:.2f}s")
                    st.caption(f"Total Turnaround: {generation_time + context_time + rerank_time + retrieval_time:.2f}s")
                
                # Extract and store sources
                # 1. Extract and store sources including image_paths
                sources = [
                    {
                        'section_number': c.payload['metadata']['section_number'],
                        'subsection_number': c.payload['metadata'].get('subsection_number'),
                        'section_title': c.payload['metadata']['section_title'],
                        'text': c.payload['text'],
                        'image_paths': c.payload['metadata'].get('image_paths', []) # Use paths
                    }
                    for c in reranked_results
                ]
                
                # 2. Fetch images directly from metadata paths
                unique_images = []
                for s in sources:
                    for path in s.get('image_paths', []):
                        if os.path.exists(path):
                            unique_images.append(path)
                
                unique_images = sorted(list(set(unique_images)))

                # 3. Inject Images into Text
                final_response_text = inject_images_in_text(full_response, unique_images)

                # Display sources and other info
                with message_placeholder.container():
                    st.markdown(final_response_text) # Show INJECTED response
                    
                    if sources:
                        with st.expander("📖 View Sources (Retrieved Chunks)"):
                             for i, source in enumerate(sources, 1):
                                st.markdown(f"""
                                <div class="source-box">
                                <strong>Source {i}</strong>: Section {source['section_number']} 
                                {f"- {source['subsection_number']}" if source.get('subsection_number') else ''}
                                <br>
                                <em>{source['section_title']}</em>
                                <hr>
                                {source['text']}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    with st.expander("🔍 Debug: View Context passed to LLM"):
                        st.code(context_str)

                    # Removed "Relevant Diagrams" section (images now inline)
                
                # Save interaction with sources and images
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response_text,
                    "sources": sources,
                    "images": unique_images,
                    "context": context_str
                })
                
            except Exception as e:
                st.error(f"Error generating response: {e}") 
                                
            # Re-render chat to show images if they weren't shown in the streaming loop (for history)
            # Actually, the loop handles the live generation. 
            # We need to ensure history rendering also shows images.
            # See top of file for history loop update.
