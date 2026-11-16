"""Streamlit web interface for the RAG Document Q&A tool (Task 7 bonus).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from rag_qa.pipeline import DEFAULT_SIMILARITY_THRESHOLD, RAGPipeline

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚")
st.title("📚 RAG Document Q&A")
st.write(
    "Ask questions about the sample documents in the `data/` folder. "
    "Answers are grounded in retrieved chunks — and when nothing matches "
    "well enough, the tool honestly says **I don't know**."
)


@st.cache_resource(show_spinner="Indexing documents…")
def load_pipeline(threshold: float) -> RAGPipeline:
    """Build and index the pipeline once, then reuse it."""
    pipeline = RAGPipeline(similarity_threshold=threshold)
    pipeline.index_folder("data")
    return pipeline


threshold = st.slider(
    "Similarity threshold (below this, the answer is \"I don't know\")",
    min_value=0.0,
    max_value=0.9,
    value=DEFAULT_SIMILARITY_THRESHOLD,
    step=0.05,
)

pipeline = load_pipeline(threshold)
st.caption(
    f"Indexed {len(pipeline.chunks)} chunks · "
    f"answer generator: {pipeline.generator.name}"
)

question = st.text_input(
    "Your question",
    placeholder="e.g. When is the one-month internship deadline?",
)
if st.button("Ask") and question.strip():
    answer = pipeline.ask(question.strip())
    if answer.answered:
        st.success(answer.text)
        st.subheader("Sources used")
        for result in answer.citations:
            st.markdown(
                f"**{result.chunk.chunk_id}** "
                f"(similarity {result.score:.2f})"
            )
            st.text(result.chunk.text)
    else:
        st.warning(
            f"I don't know — the best matching chunk scored "
            f"{answer.best_score:.2f}, below the threshold of "
            f"{answer.threshold:.2f}."
        )
