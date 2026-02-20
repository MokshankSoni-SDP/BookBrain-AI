import streamlit as st

def load_custom_css():
    st.markdown("""
    <style>
    /* Root variables for consistency */
    :root {
        --bg-primary: #0a0f1e;
        --bg-surface: #111827;
        --bg-card: #1f2937;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --primary-color: #0f3460;
        --secondary-color: #1e3a8a;
        --accent-success: #10b981;
        --border-color: #334155;
        --shadow-light: 0 4px 6px -1px rgba(0, 0,0, 0.1);
        --shadow-hover: 0 10px 25px -3px rgba(0, 0,0, 0.2);
        --radius: 12px;
    }

    /* App-wide background */
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    /* Main blocks and sections */
    .block-container {
        background-color: var(--bg-primary);
        padding-top: 2rem;
    }

    /* Chat messages - professional cards */
    .stChatMessage {
        background-color: var(--bg-card);
        color: var(--text-primary);
        border-radius: var(--radius);
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-light);
    }

    /* Chat input - sleek dark */
    .stChatInput textarea {
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius);
        padding: 0.75rem;
    }

    /* Buttons - elevated professional */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: var(--radius);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        box-shadow: var(--shadow-light);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
        background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
    }

    /* Sidebar enhancements */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-surface);
        border-right: 1px solid var(--border-color);
    }
    .css-1d391kg {  /* Sidebar header */
        background-color: var(--primary-color);
        padding: 1rem;
        border-radius: 0 0 var(--radius) 0;
    }

    /* Metrics - academic green accent */
    .metric-container {
        background: linear-gradient(135deg, var(--accent-success), #059669);
        color: white;
        border-radius: var(--radius);
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-light);
    }
    .metric-container h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }

    /* Source boxes - refined cards */
    .source-box {
        background-color: var(--bg-card);
        color: var(--text-secondary);
        border-left: 4px solid var(--accent-success);
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: var(--radius);
        box-shadow: var(--shadow-light);
        font-size: 0.95rem;
    }

    /* Math rendering improvements */
    .katex-display {
        font-size: 1.8em !important;
        margin: 1.5em 0 !important;
        padding: 1rem;
        background: var(--bg-card);
        border-radius: var(--radius);
        border-left: 4px solid var(--primary-color);
    }
    .katex {
        font-size: 1.2em !important;
    }

    /* Expanders - subtle */
    .stExpander {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius);
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stChatMessage {
            padding: 1rem;
            margin: 0.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
