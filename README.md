

# 📚 BookBrain AI — NCERT Physics Textbook RAG

**BookBrain AI** is an advanced Retrieval-Augmented Generation (RAG) system designed specifically for NCERT Physics textbooks (Grades 11 & 12). It ingests raw PDF textbooks, extracts and structures their content into a hierarchical JSON format, indexes it into a vector database, and serves an intelligent AI tutor via a Streamlit web interface.

---

## ✨ Features

- 📄 **PDF Ingestion** — Upload 11/12th NCERT Physics chapter PDF directly from the UI
- 🧠 **Intent-Aware Retrieval** — Automatically routes queries between hybrid semantic search and direct metadata-based lookup
- 🔍 **Hybrid Search** — Combines dense semantic embeddings (nomic-embed-text) with BM25 sparse vectors, fused via Reciprocal Rank Fusion (RRF)
- 📌 **Structural Anchor Boosting** — Directly fetches specific Examples, Figures, and Exercises referenced in queries
- 🗃️ **Metadata-Filtered Retrieval** — An LLM planner generates Qdrant metadata filters to answer structured queries (e.g., "list all exercises in Chapter 6")
- 💬 **Physics Professor AI** — Groq-powered LLM (LLaMA 3) responds with LaTeX math, source references, and pedagogical explanations
- 🖼️ **Figure Extraction & Display** — Diagrams are extracted from PDFs and displayed inline with relevant answers

---

## 🔄 Pipeline

The system has two major phases: **Ingestion** and **Retrieval + Generation**.

### Phase 1 — PDF Ingestion Pipeline

```
PDF File
   │
   ▼
[Step 1: Block Extraction]
   │  Uses PyMuPDF (fitz) to extract raw text blocks per page.
   │  Handles two-column NCERT layout by splitting at column midpoint for class 11 .
   |  As well as handles single column text parsing for class 12
   │  Strips headers/footers (top 8%, bottom 8% of page height).
   │  Extracts diagram images when figure captions (e.g. "Fig. 6.2") are detected.
   │
   ▼
[Step 2: Block Classification] 
   │  Classifies each block as: HEADING or CONTENT.
   │  Detects end-of-chapter sections (SUMMARY, POINTS TO PONDER, EXERCISES).
   │  Switches to single-column linear parse mode for end sections.
   │  Returns a flat list of {"type": ..., "value": ...} items.
   │
   ▼
[Step 3: Hierarchy Building]
   │  Assembles classified items into a structured JSON:
   │  {
   │    "chapter_title": "...",
   │    "sections": [ { "id", "title", "content": [], "subsections": [...] } ],
   │    "summary": [...],
   │    "points_to_ponder": [...],
   │    "exercises": [...],
   │    "topics_index": [...]
   │  }
   │  Headings with 1 dot (e.g. "6.1") → sections.
   │  Headings with 2 dots (e.g. "6.1.1") → subsections.
   │
   ▼
[Step 4: Ingestion into Qdrant] ─── ingest.py
   │  Loads JSON structure; iterates sections, subsections.
   │  Cleans noise (page numbers, repeated headers) from text.
   │  Splits long sections into structural blocks, then aggregates into
   │  chunks (soft limit 700 words, hard limit 900 words).
   │  Summary, Points to Ponder, and Exercises are stored as single atomic chunks.
   │  Extracts structural metadata per chunk:
   │    - example_number (e.g. "6.3")
   │    - figure_number  (e.g. "6.2")
   │    - image_refs / image_paths
   │  Generates:
   │    → Dense vector  : thenlper/gte-small(HuggingFace Embeddings)
   │    → Sparse vector : BM25 (hash-based vocabulary, TF-IDF weights)
   │  Upserts both vectors + payload to Qdrant collection 
   │
   ▼
[Qdrant Vector DB] — Local persistent storage (./qdrant_data)
   Named vectors: "dense" (float[]) + "bm25" (sparse)
```

> The pipeline supports two extraction modes selectable from the UI:
> - **Colwise** (default) — Two-column aware, best for standard NCERT layout
> - **Normal V2** — Linear extraction via , for single-column PDFs

---

### Phase 2 — Query Retrieval & Generation Pipeline

```
User Query (Streamlit UI)
   │
   ▼
[Intent Classification] ─── classify_intent() in main.py
   │  Uses Groq LLaMA-3.1-8B-Instant (fast, cheap).
   │  Returns: "metadata" OR "hybrid"
   │
   ├──── "metadata" ──────────────────────────────────────────────────────┐
   │                                                                       │
   ▼                                                                       │
[Metadata Query Generation] ─── generate_metadata_query                    │
   │  Uses Groq LLaMA-3.3-70B-Versatile (high accuracy planner).          │
   │  Converts natural language query into Qdrant filter JSON:             │
   │  e.g., {"filters": {"content_type": "exercises", "section_id": "6.1"}}│
   │                                                                       │
   ▼                                                                       │
[Metadata Retrieval] ───                                                   │
   │  Scrolls Qdrant collection using must[] filter conditions.            │
   │  Retrieves up to 50 matching chunks directly by metadata.             │
   │                                                                       │
   └──── "hybrid" ──────────────────────────────────────────────────────► ▼
                                                                           │
[Hybrid Retrieval] ───                                                     │
   │  Stage A — Structural Anchor Detection:                               │
   │    Regex scans query for "Example X.Y" or "Fig X.Y" patterns.        │
   │    If found, directly fetches that chunk from Qdrant by metadata.     │
   │    Anchor chunks are prepended to results with is_anchor=True flag.   │
   │                                                                       │
   │  Stage B — Hybrid Vector Search:                                      │
   │    Embeds query → dense vector (gte-small-embed-text).                    │
   │    Builds sparse query vector (BM25 hash-based).                      │
   │    Qdrant prefetch: top-20 dense + top-20 sparse.                     │
   │    Fused via Reciprocal Rank Fusion (RRF).                            │
   │    Optional chapter filter applied at query time.                     │
   │    Score threshold: 0.3.                                              │
   │                                                                       │
   │  Stage C — Merge & Deduplicate:                                       │
   │    Anchor chunks first (highest priority), then hybrid results.       │
   │                                                                       │
   ▼                                                                       │
[Context Formatting] ─── ◄──────────────────────────────────────────────── ┘
   │  Formats retrieved chunks into a structured context block with
   │  section/subsection metadata headers for the LLM.
   │
   ▼
[LLM Generation] ─── Groq API (LLaMA-3.3-70B-Versatile)
   │  System prompt enforces: strict grounding, LaTeX math (MathJax),
   │  source reference header, pedagogical explanation style.
   │  Streaming response rendered in Streamlit.
   │
   ▼
[Response Rendering] ─── render_response() + LaTeX normalization
   Normalizes LaTeX delimiters ([ ] → $$ $$), renders with st.markdown.
   Extracts image_refs from context chunks and displays matching figures
   inline using fuzzy path matching.
```

---

## 🧩 Key Components

### `retriever.py` — `PhysicsRetriever`

| Method | Description |
|---|---|
| `retrieve()` | Hybrid search: anchor detection + dense + BM25 + RRF fusion |
| `metadata_retrieve()` | Direct Qdrant scroll with must[] metadata filters |
| `fetch_structural_chunk()` | Exact lookup by `example_number` or `figure_number` metadata |
| `detect_structural_reference()` | Regex to detect "Example X.Y" / "Fig X.Y" in query |
| `build_sparse_query()` | Hash-based BM25 sparse vector for query time |

### `ingest.py` — Key Functions

| Function | Description |
|---|---|
| `load_and_process_data()` | Parses JSON structure into flat list of section/subsection items |
| `ingest_data()` | Orchestrates chunking, embedding, BM25, and Qdrant upsert |
| `split_structural_blocks()` | Splits text by textbook structural patterns (Example, numbered headers) |
| `aggregate_blocks()` | Merges small blocks up to 900-word hard limit |
| `clean_text_noise()` | Removes page numbers, headers, repeated captions |

### `main.py` — Streamlit App

| Function | Description |
|---|---|
| `classify_intent()` | Routes query to "metadata" or "hybrid" strategy via small LLM |
| `generate_metadata_query()` | LLM planner generates Qdrant filter JSON |
| `format_contexts()` | Formats retrieved chunks for LLM context window |
| `normalize_latex()` | Converts `[ ... ]` to `$$ ... $$` for MathJax rendering |
| `get_relevant_images()` | Fuzzy-matches image references to actual extracted image files |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **PDF Parsing** | PyMuPDF (`fitz`) |
| **Embeddings** | `thenlper/gte-small` |
| **Sparse Search** | BM25 (custom hash-based implementation) |
| **Vector Database** | Qdrant (local persistent mode) |
| **Fusion** | Reciprocal Rank Fusion |
| **LLM** | Groq API — LLaMA 3.1 8B (intent) + LLaMA 3.3 70B (generation/planning) |
| **UI Framework** | Streamlit |
| **Backend Helpers** | LangChain Core |
| **Hardware** | Auto-detects CUDA GPU, falls back to CPU |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API Key](https://console.groq.com/)
- Optionally: CUDA-capable GPU for faster embeddings

### Installation

```bash
# Clone the repository
git clone https://github.com/MokshankSoni-SDP/BookBrain-AI.git
cd BookBrain-AI

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
EMBEDDING_MODEL=thenlper/gte-small   # optional override
RERANKER_MODEL=BAAI/bge-reranker-v2-m3           # optional override
```

### Running the App

```bash
streamlit run main.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 Usage

1. **Upload a PDF** — Use the sidebar to upload an 11/12th NCERT Physics book.
2. **Select Extraction Mode** — Choose "Colwise" (two-column, default) or "Normal V2" (single-column).
3. **Click "Process & Ingest"** — The full pipeline runs: extract → classify → JSON → Qdrant.
4. **Ask Questions** — Type any physics question in the chat.
   - *Conceptual*: "Explain the law of conservation of angular momentum."
   - *Structural*: "Solve Example 6.3" , "List all exercises in section 6.2", "List topics of chapter 6 class 11","give the summary of chapter Laws of Motion"
5. **Chapter Filter** — Select specific chapters from the sidebar to restrict retrieval scope.

---

## 📦 Dependencies

```
PyMuPDF          # PDF parsing and image extraction
numpy            # Numerical operations
python-dotenv    # Environment variable management
langchain-core   # Document abstraction
qdrant-client    # Vector database client
torch            # GPU/CPU tensor operations
streamlit        # Web UI framework
groq             # Groq LLM API client
sentence-transformers  # Embedding & reranker models
```

---

## 📁 Data Flow Summary

```
PDF → PyMuPDF blocks → Classified items → Hierarchical JSON
    → Cleaned text chunks → Dense + BM25 vectors → Qdrant

User query → Intent classifier → [Metadata path OR Hybrid path]
          → Retrieved chunks → LLM context → Streamed response → Streamlit UI
```

---

## 🖼️ Image Extraction, Storage & Display Pipeline

Diagrams are a first-class citizen in BookBrain AI. The system extracts figures directly from the PDF during ingestion, stores their file paths in Qdrant metadata, and displays them inline alongside the LLM's explanation at query time.

### Step 1 — Detection: Finding Figure Captions

Before any image is rendered, the system must locate it. Both Class 11 and Class 12 strategies start by **scanning every text block on a page for the `Fig.` / `Figure` pattern** using a compiled regex:

```
FIGURE_PATTERN = r'(?:Fig\.|Figure)\s*(\d+\.\d+)'
```

When a match is found (e.g. `"Fig. 6.2"`), the matching block's bounding box becomes the **caption anchor** — the spatial reference point for the upward image search.

---

### Step 2 — Extraction Strategy (Class 11 vs Class 12)

The two textbook layouts require fundamentally different extraction approaches.

#### 📘 Class 11 — Two-Column Layout (Colwise Mode)

The primary challenge is **preventing horizontal bleed** — accidentally grabbing content from the adjacent column.

| Step | What Happens |
|---|---|
| **Column Split** | Page width is divided at `split_x = page.rect.width × 0.5`. Each block is classified as `left_col` or `right_col` by its `x0` coordinate. |
| **Fenced Sorting** | Blocks within each column are sorted strictly by Y-coordinate (top → bottom), ensuring reading order is preserved per column. |
| **Caption Anchoring** | When a `Fig.` caption is found, its `caption_rect` becomes the spatial anchor for the upward search. |
| **Constrained Search Area** | A search rectangle is projected **300 points upward** from the caption. It is horizontally fenced to `[col_min_x, col_max_x]` — never crossing `split_x`. | |
| **Render** | The Master Box is rendered at **300 DPI (3× zoom matrix)** via `page.get_pixmap()` and saved as `fig_X_Y.png`. |

#### 📗 Class 12 — Single-Column Layout (Normal V2 Mode)

Class 12 textbooks use a full-page-width layout and contain many **bitmap photographs** in addition to vector drawings.

| Step | What Happens |
|---|---|
| **Label Search** | Instead of text blocks, `page.search_for("Fig.")` / `page.search_for("FIGURE")` is used to pinpoint the **exact pixel coordinates** of the figure label. |
| **Full-Width Projection** | A larger search area (**450 points high**) is projected upward, spanning the **entire page width** — no column fencing needed. |
| **Scanning** | Searches for both **bitmaps** (`page.get_images()`) and **vectors** (`page.get_drawings()`). |


---

### Step 3 — Storage: Embedding Image Paths into the Pipeline

After a figure is saved to disk, its path is normalized and embedded **directly into the text content** of the surrounding block as a marker tag:

```
[IMAGE: processed_data/<run_id>/images/fig_6_2.png]
```

This marker travels with the text through `step4_to_json.py` and into `ingest.py`. During ingestion, the `image_pattern` regex extracts all `[IMAGE: ...]` tags from each chunk and populates two metadata fields per Qdrant point:

```json
{
  "image_refs":  ["fig_6_2"],
  "image_paths": ["processed_data/abc12345/images/fig_6_2.png"]
}
```

- **`image_refs`** — basename without extension, used for fuzzy matching at query time  
- **`image_paths`** — full relative path, used for direct file loading in the UI

---

### Step 4 — Display: Showing Figures Inline with LLM Answers

When a query returns chunks that contain image references, the following happens in `main.py`:

```
Retrieved chunks
   │
   ▼
[Image Ref Extraction]
   │  Loops over all returned chunks.
   │  Collects unique image_refs (e.g. ["fig_6_2", "fig_6_3"]) from chunk metadata.
   │
   ▼
[Path Matching] ─── get_relevant_images()
   │  Scans the run's images/ directory for .png files.
   │  For each image_ref, finds files whose basename STARTS WITH the ref string.
   │
   ▼
[LLM Response Generation]
   │  Image paths are included in the context block sent to the LLM.
   │  The LLM references and explains the figure with the help of captions that were stored with the image path in its textual response.
```

> **How the LLM "sees" the image**: The image paths in context allow the LLM to reference the figure by name in its explanation (e.g., *"As shown in Fig. 6.2..."*). The actual PNG is then rendered below the response — creating a seamless professor-style explanation paired with the visual.

---

## 📝 Notes

- The Qdrant collection is named **`physics_textbook`** and stores both dense and sparse named vectors.
- Each chunk's payload contains rich metadata: `chapter_id`, `section_id`, `subsection_id`, `content_type`, `example_number`, `figure_number`, `image_refs`, `image_paths`, `chunk_index`, `estimated_tokens`.
- The system currently targets **NCERT Physics Class 11** chapters by default, but supports multi-grade ingestion via the `grade` metadata field.
- Diagrams are saved as `fig_X_Y.png` in the run's `images/` subdirectory and linked in chunk metadata for inline display.
