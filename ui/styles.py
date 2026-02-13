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
        background-color: #e8eaf6;
        border-left: 4px solid #3f51b5;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
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
