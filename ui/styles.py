import streamlit as st

def load_custom_css():
    st.markdown("""
    <style>
    /* Chat message styling */
    /* Chat message styling */
    .stChatMessage {
        background-color: #131313;
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #333;
    }

    /* Chat Input Styling */
    .stChatInput textarea {
        background-color: #000000 !important;
        color: white !important;
        border: 1px solid white !important;
    }
    
    /* Input container focus state */
    .stChatInput div[data-baseweb="base-input"] {
        background-color: #000000 !important;
        border: 1px solid white !important;
    }
    
    /* Source citation boxes */
    .source-box {
        background-color: #262730; /* Darker background */
        color: #e0e0e0; /* Light text */
        border-left: 4px solid #667eea; /* Matching metric color */
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        border-radius: 4px;
    }
    
    .katex-display {
        font-size: 2em !important;
        margin-top: 1.2em !important;
        margin-bottom: 1.2em !important;
    }

    /* Inline math slightly bigger */
    .katex {
        font-size: 1.35em !important;
}

    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
