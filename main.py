import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from retriever import PhysicsRetriever

# Load env
load_dotenv()

# Config page
st.set_page_config(page_title="BookBrain AI", layout="wide")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_retriever():
    return PhysicsRetriever()

if "retriever" not in st.session_state:
    with st.spinner("Initializing Retrieval Engine..."):
        # This will take a moment to load models
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

**Context Structure**:
Each context chunk includes:
- Chapter and section identifiers
- Hierarchical position in the textbook
- The actual content excerpt

Use these metadata tags to provide precise citations."""

def format_contexts(chunks):
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.payload['metadata']
        location = f"Section {meta['section_number']}"
        if meta['subsection_number']:
            location += f".{meta['subsection_number']}"
        
        formatted.append(f"""
--- Context {i} ---
**Location**: {location} - {meta.get('subsection_title') or meta['section_title']}
**Content**:
{chunk.payload['text']}
""")
    return "\n".join(formatted)

# UI
st.title("Physics Textbook AI Tutor")
st.caption("Powered by RAG: Qdrant + Llama-3 + BGE Reranker")

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a physics question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if "retriever" not in st.session_state:
             st.error("Retriever not initialized.")
             st.stop()
             
        if "groq_client" not in st.session_state:
            st.error("Groq client not initialized.")
            st.stop()

        # 1. Retrieve
        results = []
        with st.status("Retrieving knowledge...", expanded=False) as status:
            st.write("Searching vector database...")
            results = st.session_state.retriever.search(prompt)
            st.write(f"Found {len(results)} relevant excerpts.")
            
            # Show citations/debug
            for i, res in enumerate(results, 1):
                meta = res.payload['metadata']
                st.text(f"{i}. Sec {meta['section_number']}: {meta.get('subsection_title') or meta['section_title']} (Score: {res.payload.get('rerank_score', 0):.2f})")
            
            status.update(label="Knowledge retrieved!", state="complete")
            
        # 2. Format Context
        context_str = format_contexts(results)
        
        # 3. Call LLM
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
            
        except Exception as e:
            st.error(f"Error calling LLM: {str(e)}")
            full_response = "I encountered an error while generating the response."
            
        # Save interaction
        st.session_state.messages.append({"role": "assistant", "content": full_response})
