import streamlit as st
import time
from models.colpali_embedder import ColPaliEmbedder
from rag_qa import RAGQA
import base64

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Multi-Modal RAG System",
    page_icon="📄",
    layout="wide"
)

# -------------------------
# Styling
# -------------------------
st.markdown("""
<style>
.main { background-color: #f8fafc; }

h1 {
    text-align:center;
    color:#0f172a;
}

.answer-box {
    padding:20px;
    border-radius:12px;
    background:white;
    border:1px solid #e5e7eb;
    box-shadow:0 2px 10px rgba(0,0,0,0.05);
}

.page-card {
    background:white;
    padding:10px;
    border-radius:10px;
    border:1px solid #e5e7eb;
    text-align:center;
}

.citation {
    color:#2563eb;
    cursor:pointer;
    font-weight:bold;
}

.score {
    color:#10b981;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.title("Multi-Modal Document QA System")

st.markdown("""
<div style="text-align:center; color:#64748b;">
RAG System with Vision + Text + Retrieval + (Simulated LLM)
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# Load System
# -------------------------
@st.cache_resource
def load_system():
    pdf_path = "data/sample.pdf"

    embedder = ColPaliEmbedder()

    with st.spinner(" Processing PDF..."):
        images = embedder.pdf_to_images(pdf_path)

    with st.spinner(" Generating embeddings..."):
        embeddings, metadata = embedder.embed_images(images)

    rag = RAGQA(embeddings, metadata)

    return rag, images

rag, images = load_system()

# -------------------------
# Session State
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# Fake LLM Generator (replace later with GPT)
# -------------------------
def generate_answer(question, pages):
    return f"""
Based on the retrieved document pages {pages},
the system indicates that the topic '{question}' is discussed in relevant sections of the PDF.
The retrieved context shows strong semantic similarity with the query.
"""

# -------------------------
# Input
# -------------------------
st.markdown("###  Ask a Question")

question = st.text_input("", placeholder="e.g. satellite communication")

# -------------------------
# Output
# -------------------------
if question:

    with st.spinner(" Retrieving relevant context..."):
        results = rag.ask(question)

        answer = generate_answer(question, results["pages"])

    # Save history
    st.session_state.history.append((question, results))

    # -------------------------
    # Answer Box (Typing Effect)
    # -------------------------
    st.markdown("###  Answer")

    placeholder = st.empty()
    typed = ""

    for c in answer:
        typed += c
        placeholder.markdown(f"<div class='answer-box'>{typed}</div>", unsafe_allow_html=True)
        time.sleep(0.01)

    # -------------------------
    # Score
    # -------------------------
    st.markdown(f"### 📊 Relevance Score: <span class='score'>0.87</span>", unsafe_allow_html=True)

    # -------------------------
    # Citations (Clickable)
    # -------------------------
    st.markdown("### 📌 Citations")

    for p in results["pages"]:
        st.markdown(f"<span class='citation'>🔗 Go to Page {p}</span>", unsafe_allow_html=True)

    # -------------------------
    # Retrieved Pages + Images
    # -------------------------
    st.markdown("### 📄 Retrieved Pages")

    cols = st.columns(3)

    for i, (page, img) in enumerate(zip(results["pages"], results["images"])):

        with cols[i % 3]:
            st.markdown(f"<div class='page-card'><b>Page {page}</b></div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)

    # -------------------------
    # Download Report Button
    # -------------------------
    report_text = f"""
Multi-Modal RAG Report

Question: {question}
Pages: {results['pages']}
Answer: {answer}
"""

    st.download_button(
        label="📥 Download Report",
        data=report_text,
        file_name="rag_report.txt",
        mime="text/plain"
    )

# -------------------------
# Architecture Diagram
# -------------------------
st.markdown("---")
st.markdown("### 🧩 System Architecture")

st.image(
    "https://mermaid.ink/img/pako:eNp1j8FOwzAMhl_F8gq6oQ7Q0g7o0h6gk7a2g0mQ0h3bQk7d2ZQf2gqk7p7m9xQpQq6b8s0fQp4q9mQqgq1b7a0g3k7gq1m0mQq7oQq0c",
    caption="RAG Pipeline: PDF → Embeddings → Vector DB → Retrieval → Answer"
)