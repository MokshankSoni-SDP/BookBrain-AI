import streamlit as st
import json
import os
import time
from dotenv import load_dotenv
from groq import Groq
import re
import base64

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

if "selected_chapters" not in st.session_state:
    st.session_state.selected_chapters = []

if "available_chapters" not in st.session_state:
    st.session_state.available_chapters = None

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

DEBUG_MODE = True

def debug_log(message):
    if DEBUG_MODE:
        print(message)

# System Prompt
# System Prompt
SYSTEM_PROMPT = """
You are a High-Level Physics Professor. Your goal is to weave the provided textbook excerpts into a seamless, conversational, and pedagogical lesson for a student.

1. THE GOLDEN RULE: STRICT GROUNDING
- Your ONLY source of truth is the 'Textbook Excerpts'. You must answer STRICTLY using the provided textbook excerpts.
- You are not allowed to use prior knowledge.
- If a user asks about a specific entity (e.g., 'Example 6.3' or 'Problem 5') and that EXACT term is not mentioned in the excerpts, you MUST NOT answer using general knowledge.
- In such cases, your entire response must be:
  "The provided textbook excerpts do not contain the specific content for [Entity Name]. Please provide the text or verify the section."

→ Do NOT guess.
→ Do NOT generalize.
→ Do NOT assume based on section theme.

2. PEDAGOGICAL ADAPTIVITY (CRITICAL)
You must intelligently adapt your explanation style based on the user's question.

A. If the user asks to "derive", "prove", or "show that":
   - Present a formal derivation format.
   - Clearly state the objective equation first.
   - Progress logically from assumptions → intermediate steps → final expression.
   - Avoid unnecessary storytelling.
   - Emphasize mathematical flow.
   - Conclude with a boxed or highlighted final result using display math.

B. If the user asks a short factual question (e.g., "What is Gauss's law?"):
   - Provide a concise and precise definition.
   - Include the key equation.
   - Do NOT add long derivations unless asked.

C. If the user asks a conceptual "why" or "explain" question:
   - Begin with intuition.
   - Then connect to mathematical expression (if present in text).
   - Focus on physical interpretation.

D. If the user asks to solve a numerical problem:
   - Extract given values from the excerpt.
   - Show substitution clearly.
   - Present final answer with units.

E. If the question is vague or broad:
   - Provide a structured conceptual explanation.
   - Avoid forcing a derivation format unless explicitly requested.

Your tone should resemble a real professor adjusting explanation depth to the student’s demand.

3. MATHEMATICAL PRECISION (LaTeX)
- Use LaTeX for ALL mathematical symbols, variables, and equations.
- INLINE: Use single dollar signs. Example: $v = r \omega$.
- DISPLAY: Use double dollar signs for main equations.
- Never write variables in plain text.
- Do NOT mix text improperly inside math blocks.
- Ensure all formatting renders cleanly.

4. STRUCTURE (FLEXIBLE BUT CLEAN)
- Always use Markdown headings (###) for conceptual sections.
- DO NOT force fixed headings like:
  "Introduction", "Physical Significance", "Conclusion"
- Only include sections that are relevant to the user's question.
- Maintain natural flow instead of template repetition.

5. FIGURE INTEGRATION
- If content recieved has mentioned any kind of figure , you need to try displaying it with the content 
- You MUST explicitly refer to figures mentioned in the text (e.g., "Fig. 6.7") when explaining a related concept. 
- Format: Always use the exact string 'Fig. X.Y' (e.g., Fig. 6.12). 
- Your mention of 'Fig. X.Y' acts as a trigger for the system to display the diagram. Do not describe an image if the text doesn't mention a Figure ID.

6. RESPONSE START FORMAT
Always begin your response with:

"**Source Reference**: [Exact Section or Subsection Title from Metadata]"

7. NO META-COMMENTARY
- Do NOT mention that you are following rules.
- Do NOT mention the prompt.
- Do NOT mention retrieved chunks.
- Only deliver the physics explanation.

8. CONSISTENCY SAFETY
If mathematical expressions in the excerpts appear inconsistent, incomplete, or truncated:
- Present them exactly as provided.
- Do NOT correct them using external knowledge.

Your goal is to behave like an expert physics professor who adapts explanation depth intelligently — while remaining strictly grounded in the provided text.
"""


def normalize_latex(text):
    """
    Makes LaTeX rendering robust:
    1. Converts [ ... ] math blocks into $$ ... $$
    2. Fixes escaped backslashes
    3. Ensures display equations render correctly
    """

    # Fix double-escaped backslashes
    text = text.replace("\\\\", "\\")

    text = re.sub(r"\\\[(.*?)\\\]", r"\n\n$$\1$$\n\n", text, flags=re.DOTALL)

    # Convert bracketed LaTeX blocks to display math
    def replace_brackets(match):
        content = match.group(1)
        if "\\" in content or "^" in content or "_" in content:
            return f"\n\n$$\n{content}\n$$\n\n"
        return match.group(0)

    text = re.sub(r"\[\s*(.*?)\s*\]", replace_brackets, text, flags=re.DOTALL)

    text = re.sub(
        r"\n\$(.*?)=(.*?)\$\n",
        r"\n\n$$\1=\2$$\n\n",
        text
    )

    return text


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
    for path in image_paths:
        filename = os.path.basename(path)
        figure_number = filename.replace("fig_", "").replace(".png", "").replace("_", ".")
        fig_label = f"Fig. {figure_number}"

        if fig_label in response_text:
            placeholder = f"\n\n[[IMAGE::{path}]]\n\n"
            response_text = response_text.replace(fig_label, fig_label + placeholder, 1)

    return response_text

CHAPTER_MAPPING = {
    "1": "U NITS   AND  M EASUREMENT",
    "2": "M OTION   IN   A  S TRAIGHT  L INE",
    "3": "M OTION   IN   A  P LANE",
    "4": "L AWS   OF  M OTION",
    "5": "W ORK , E NERGY   AND  P OWER",
    "6": "S YSTEMS   OF  P ARTICLES   AND  R OTATIONAL  M OTION",
    "7": "G RAVITATION",
    "8": "M ECHANICAL  P ROPERTIES   OF  S OLIDS",
    "9": "M ECHANICAL  P ROPERTIES   OF  F LUIDS",
    "10": "T HERMAL  P ROPERTIES   OF  M ATTER",
    "11": "T HERMODYNAMICS",
    "12": "K INETIC  T HEORY",
    "13": "O SCILLATIONS",
    "14": "W AVES"
}


def render_response(text):
    parts = re.split(r"\[\[IMAGE::(.*?)\]\]", text)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Normal text
            st.markdown(part)
        else:
            # Image path
            if os.path.exists(part):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(part, width=350)
                    fig_name = os.path.basename(part).replace("fig_", "Fig. ").replace(".png", "").replace("_", ".")
                    st.caption(fig_name)

def classify_intent(prompt):

    debug_log("\n================ NEW QUERY =================")
    debug_log(f"[INTENT] Classifier called for prompt: {prompt}")

    classifier_prompt = f"""
You are an AI routing controller for a Physics Textbook Retrieval System.

The system supports two retrieval strategies:
metadata - direct retrieval using indexing on metadata stored of chunks
hybrid - rag approach in which search is done using semantic search

1) "metadata"
   → Use when the query refers to a specific structured entity
     that is indexed in metadata.
   → use when user asks question that needs listing something and in situations when semantic search might fail   

   Structured entities include:
   - Chapter numbers (e.g., Chapter 6)
   - Section numbers (e.g., 6.2)
   - Subsection numbers (e.g., 6.2.1)
   - Example numbers (e.g., Example 6.1)
   - Exercise numbers (e.g., Exercise 6.1)
   - Figure numbers (e.g., Fig. 6.3)

   If the user query clearly targets a specific numbered
   textbook entity, choose "metadata".

   if user query not clearly targets a specific number you need to think of approach can direct metadata retireval answer that question if yes just go for it

   Even if the user says "explain", "solve", or "derive",
   if it references a specific numbered entity,
   choose "metadata".

------------------------------------------------------------

2) "hybrid"
   → Use when the query is conceptual or thematic
     and does NOT refer to a specific numbered entity.

   Examples:
   - explain angular momentum
   - why does torque cause rotation?
   - what is rigid body motion?
   - explain conservation of energy

------------------------------------------------------------

Decision Logic:

If the query contains a specific textbook reference
(example number, exercise number, section number, etc.),
choose "metadata".

Otherwise, choose "hybrid".

If unsure, default to "hybrid".

------------------------------------------------------------

Return STRICT JSON only:

{{
  "retrieval_type": "metadata" OR "hybrid"
}}

User Query:
{prompt}
"""


    response = st.session_state.groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",  # small cheap model
        messages=[{"role": "user", "content": classifier_prompt}],
        temperature=0
    )

    try:
        content = response.choices[0].message.content
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])
    except:
        return {"retrieval_type": "hybrid"}

def generate_metadata_query(prompt):

    # ---- Chapter Mapping (MANUAL for now as you said) ----
    CHAPTER_MAPPING = {
        "11":{
        "1": "U NITS   AND  M EASUREMENT",
        "2": "M OTION   IN   A  S TRAIGHT  L INE",
        "3": "M OTION   IN   A  P LANE",
        "4": "L AWS   OF  M OTION",
        "5": "W ORK , E NERGY   AND  P OWER",
        "6": "S YSTEMS   OF  P ARTICLES   AND  R OTATIONAL  M OTION",
        "7": "G RAVITATION",
        "8": "M ECHANICAL  P ROPERTIES   OF  S OLIDS",
        "9": "M ECHANICAL  P ROPERTIES   OF  F LUIDS",
        "10": "T HERMAL  P ROPERTIES   OF  M ATTER",
        "11": "T HERMODYNAMICS",
        "12": "K INETIC  T HEORY",
        "13": "O SCILLATIONS",
        "14": "W AVES"
    },
    "12":{
        "1": "U NITS   AND  M EASUREMENT",
        "2": "M OTION   IN   A  S TRAIGHT  L INE",
        "3": "M OTION   IN   A  P LANE",
        "4": "L AWS   OF  M OTION",
        "5": "W ORK , E NERGY   AND  P OWER",
        "6": "S YSTEMS   OF  P ARTICLES   AND  R OTATIONAL  M OTION",
        "7": "G RAVITATION",
        "8": "M ECHANICAL  P ROPERTIES   OF  S OLIDS",
        "9": "M ECHANICAL  P ROPERTIES   OF  F LUIDS",
        "10": "T HERMAL  P ROPERTIES   OF  M ATTER",
        "11": "T HERMODYNAMICS",
        "12": "K INETIC  T HEORY",
        "13": "O SCILLATIONS",
        "14": "W AVES"
    }
    }

    # ---- Schema Block (NO f-string, NO braces formatting issues) ----
    schema_block = """
You are generating metadata filters for a vector database.

The system stores textbook content in structured metadata.

Available metadata fields:
- grade ("11" or "12")
- chapter_title
- section_id
- section_title
- subsection_id
- subsection_title
- content_type (section, subsection, summary, points_to_ponder, exercises)
- example_number
- figure_number

VALID grade values:
- "11"
- "12"

If user says:
- class 11 / 11th / grade 11 → use "grade": "11"
- class 12 / 12th / grade 12 → use "grade": "12"

If grade is not mentioned, DO NOT include grade in filters.

VALID chapter_title values:
""" + json.dumps(CHAPTER_MAPPING, indent=2) + """

Rules:
- If user mentions chapter number, map it using chapter_title above.
- If user mentions chapter name directly, use exact chapter_title.
- content_type must be EXACTLY one of:
  section, subsection, summary, points_to_ponder, exercises

Return STRICT JSON only:

{
  "filters": {
     "field": "value"
  }
}

If no metadata filter applies, return:
{
  "filters": {}
}
"""

    planner_prompt = schema_block + "\n\nUser Query:\n" + prompt

    response = st.session_state.groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": planner_prompt}],
        temperature=0
    )

    try:
        content = response.choices[0].message.content
        start = content.find("{")
        end = content.rfind("}") + 1
        parsed = json.loads(content[start:end])
        print(f"[PLANNER] Metadata Plan: {parsed}")
        return parsed
    except Exception as e:
        print(f"[PLANNER ERROR] {e}")
        return {"filters": {}}



def format_contexts(chunks):
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.payload['metadata']
        
        # Determine Header based on structural metadata
        header_label = ""
        if meta.get("example_number"):
            header_label = f"[Primary Example {meta['example_number']}]"
        elif meta.get("section_id"):
            header_label = f"[Section {meta['section_id']}]"
            if meta.get('subsection_id'):
                header_label += f".{meta['subsection_id']}"
        
        location_title = meta.get('subsection_title') or meta.get('section_title') or "Unknown Section"

        # Check for images
        image_note = ""
        if meta.get('image_refs'):
            image_note = "\n[Visual Context: Relevant images/diagrams are available to the user for this section.]"
        
        formatted.append(f"""
--- Context {i} {header_label} ---
**Location**: {header_label} - {location_title}
**Chunk Metadata**: {meta.get('example_number', '')} {meta.get('figure_number', '')}
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
    
    # Ingestion Method Selection
    ingest_mode = st.radio(
        "Ingestion Method",
        ["Colwise Text (Default)", "Normal Text"],
        help="Choose 'Colwise' for multi-column textbooks. Choose 'Normal' for standard single-column PDFs."
    )
    
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
                
                # Map UI selection to pipeline arg
                mode_arg = "colwise" if "Colwise" in ingest_mode else "normal"
                
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
                    progress_callback=update_progress,
                    extraction_mode=mode_arg
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
                
                # 4. Refresh retriever and chapter cache
                st.session_state.retriever = get_retriever()
                # Clear chapter cache to show newly ingested chapter
                if "available_chapters" in st.session_state:
                    st.session_state.available_chapters = None
                st.success(f"✅ Processing Complete! Images saved to {images_dir}")
                
            except Exception as e:
                st.error(f"Pipeline failed: {e}") 
                # Attempt to re-init retriever if it failed mid-way so app isn't broken
                if "retriever" not in st.session_state:
                     st.session_state.retriever = get_retriever()

    # 3. Knowledge Base Control
    st.subheader("📚 Knowledge Base Control")
    
    client = get_qdrant_client()
    
    # # Fetch all chapters from collection (cached in session state)
    # if st.session_state.available_chapters is None:
    #     chapters = set()
        
    #     try:
    #         scroll_result = client.scroll(
    #             collection_name="physics_textbook",
    #             limit=1000,
    #             with_payload=True
    #         )
        
    #         for point in scroll_result[0]:
    #             cid = point.payload.get("metadata", {}).get("chapter_id")
    #             if cid:
    #                 chapters.add(cid)
        
    #     except:
    #         pass
        
    #     st.session_state.available_chapters = sorted(list(chapters))
    
    # chapters = st.session_state.available_chapters

    if st.session_state.available_chapters is None:
        chapters = set()
        offset = None

        try:
            while True:
                points, offset = client.scroll(
                    collection_name="physics_textbook",
                    limit=1000,
                    offset=offset,
                    with_payload=True
                )

                for point in points:
                    cid = point.payload.get("metadata", {}).get("chapter_id")
                    if cid:
                        chapters.add(cid)

                if offset is None:
                    break

        except Exception as e:
            print("Chapter discovery error:", e)

        st.session_state.available_chapters = sorted(list(chapters))

    chapters = st.session_state.available_chapters

    if chapters:
        selected_chapters = st.multiselect(
            "Select Chapters to Search",
            options=chapters,
            default=chapters,
            help="Filter search results to selected chapters only"
        )
        
        # Store in session state for use in retrieval
        st.session_state.selected_chapters = selected_chapters if selected_chapters else chapters
        
        # Delete functionality
        st.markdown("---")
        st.markdown("**🗑️ Delete Chapters**")
        
        if selected_chapters:
            if st.button("Delete Selected Chapters", type="secondary"):
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                
                delete_filter = Filter(
                    should=[
                        FieldCondition(
                            key="metadata.chapter_id",
                            match=MatchValue(value=cid)
                        )
                        for cid in selected_chapters
                    ]
                )
                
                client.delete(
                    collection_name="physics_textbook",
                    points_selector=delete_filter
                )
                
                st.success(f"✅ Deleted: {', '.join(selected_chapters)}")
                st.rerun()
        else:
            st.warning("⚠️ No chapters selected - searches will return no results")
            st.info("Select chapters above to enable deletion")
    else:
        st.info("No chapters found in database")
        st.session_state.selected_chapters = []

    # 4. Retrieval Settings
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider("Initial Retrieval (K)", 6, 50, 20)
    final_chunks = st.slider("Final Context Chunks", 1, 15, 2)
    
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

def is_topic_query(prompt: str) -> bool:
    prompt_lower = prompt.lower()

    topic_patterns = [
        "list topics",
        "list all topics",
        "list topic",
        "give all topics",
        "list all topic",
        "which are the topics",
        "which are the topic",
        "what are the topics",
        "what are the topic",
        "topics of chapter",
        "topic of chapter",
        "all topics",
        "all topic"
    ]

    return any(p in prompt_lower for p in topic_patterns)

def normalize_text(text: str) -> str:
    """
    Normalize text for reliable comparison:
    - Remove extra spaces
    - Remove all inner spaces
    - Lowercase
    - Remove commas and special characters
    """
    text = text.lower()
    text = re.sub(r"\s+", "", text)  # remove ALL spaces
    text = re.sub(r"[^\w]", "", text)  # remove punctuation
    return text


def extract_chapter_title_from_prompt(prompt: str):
    prompt_lower = prompt.lower()

    structure_path = "processed_data/knowledge_base/structure.json"

    if not os.path.exists(structure_path):
        return None

    with open(structure_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chapters = data.get("chapters", data)

    normalized_prompt = normalize_text(prompt)

    # Case 1 — If user says "chapter 5"
    match = re.search(r"chapter\s+(\d+)", prompt_lower)
    if match:
        chapter_title  = match.group(1)

        # Use CHAPTER_MAPPING to map number to title
        mapped_title = CHAPTER_MAPPING.get(chapter_title)
        if mapped_title:
            return mapped_title

    # Case 2 — If user directly writes chapter name
    for chapter in chapters:
        stored_title = chapter.get("chapter_title", "")
        normalized_stored = normalize_text(stored_title)

        if normalized_stored in normalized_prompt:
            return stored_title

    return None


def get_topics_from_structure(chapter_title: str):
    structure_path = "processed_data/knowledge_base/structure.json"

    if not os.path.exists(structure_path):
        return []

    with open(structure_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chapters = data.get("chapters", data)

    for chapter in chapters:
        if chapter.get("chapter_title") == chapter_title:
            topics = []

            for section in chapter.get("sections", []):
                sec_id = section.get("id")
                sec_title = section.get("title")
                topics.append(f"{sec_id} — {sec_title}")

                for subsection in section.get("subsections", []):
                    sub_id = subsection.get("id")
                    sub_title = subsection.get("title")
                    topics.append(f"{sub_id} — {sub_title}")

            return topics

    return []



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
            # Initial retrieval with chapter filter
            chapter_filter = st.session_state.get('selected_chapters', None)
            
            # -------------------------------
            # DIRECT TOPIC EXTRACTION FLOW
            # -------------------------------
            
            if is_topic_query(prompt):
                debug_log("[ROUTER] Topic query detected. Using direct JSON extraction.")
            
                chapter_title = extract_chapter_title_from_prompt(prompt)
                print(f"[DEBUG] Extracted chapter title: {chapter_title}")
            
                if not chapter_title:
                    st.error("Could not detect chapter number.")
                    st.stop()
            
                topics = get_topics_from_structure(chapter_title)
                print(f"[DEBUG] Extracted topics: {topics}")
            
                if not topics:
                    st.error("No topics found for this chapter.")
                    st.stop()

                # -------------------------------
                # DIRECT STREAMLIT DISPLAY
                # -------------------------------
                with message_placeholder.container():
                    st.markdown(f"### 📘 {chapter_title}")
                    st.markdown("---")

                    for i, topic in enumerate(topics, 1):
                        st.markdown(f"**{i}.** {topic}")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Displayed topics for {chapter_title}",
                    "sources": [],
                    "images": [],
                    "context": "Direct Topic Extraction"
                })

                st.stop()
            
                # context_str = "\n".join(topics)
                # print(f"[DEBUG] Context string: {context_str}")
            
                # # Skip retrieval completely
                intent = {"retrieval_type": "direct_topics"}
            
            else:
                intent = classify_intent(prompt)
                debug_log(f"[INTENT] Result: {intent}")

            if intent["retrieval_type"] == "direct_topics":
                results = []
                reranked_results = []
                retrieval_time = 0
                context_time = 0

            elif intent["retrieval_type"] == "hybrid":
                # Normal RAG
                print(f"[DEBUG] Intent classified as HYBRID. Prompt: {prompt}")
                st.info("🧠 Approach: **Hybrid Search** (Conceptual/Explanatory)")
                
                results = st.session_state.retriever.retrieve(
                    prompt, 
                    top_k=top_k,
                    chapter_filter=chapter_filter
                )
            else:
                # Metadata flow
                print(f"[DEBUG] Intent classified as METADATA. Prompt: {prompt}")
                plan = generate_metadata_query(prompt)
                print(f"[DEBUG] Metadata Plan: {plan}")
                debug_log(f"[PLANNER] Metadata Plan: {plan}")
                st.info(f"🗂️ Approach: **Metadata Search** (Structured)\n\nPlan: `{plan}`")

                results = st.session_state.retriever.metadata_retrieve(
                    filters=plan.get("filters", {}),
                    limit=50
                )
            t1 = time.time()
            retrieval_time = t1 - t0
            
            # Reranking
            
            if intent["retrieval_type"] != "direct_topics":

                if intent["retrieval_type"] == "metadata":
                    reranked_results = results[:final_chunks]
                else:
                    reranked_results = st.session_state.retriever.rerank(prompt, results, top_k=final_chunks)

                t2 = time.time()
                rerank_time = t2 - t1
                
                context_str = format_contexts(reranked_results)
                t3 = time.time()
                context_time = t3 - t2
        
        # Generation phase
        with st.spinner("✍️ Generating answer..."):
            t_gen_start = time.time()
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
                    temperature=0.3
                )
                
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.write(full_response + "▌")

                message_placeholder.markdown(full_response)
                
                generation_time = time.time() - t_gen_start

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
                    #col2.metric("Reranking", f"{rerank_time:.2f}s")
                    col3.metric("Context Prep", f"{context_time:.2f}s")
                    col4.metric("LLM Generation", f"{generation_time:.2f}s")
                
                # Extract and store sources
                # 1. Extract and store sources including image_paths
                sources = [
                    {
                        'section_number': c.payload['metadata']['section_id'],
                        'subsection_number': c.payload['metadata'].get('subsection_id'),
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

                full_response = normalize_latex(full_response)

                
                # 4. Inject Images into Text
                final_response_text = inject_images_in_text(full_response, unique_images)

                # Display sources and other info
                with message_placeholder.container():
                    # st.markdown(final_response_text,unsafe_allow_html=False) # Show INJECTED response
                    render_response(final_response_text)


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
