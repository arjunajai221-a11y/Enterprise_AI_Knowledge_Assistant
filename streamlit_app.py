import hashlib
import csv
import io
import time
from pathlib import Path

import streamlit as st
from app.rag_pipeline import RAGPipeline


# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

DOCUMENT_DIR = Path("data/documents")
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# BATCH TEST QUESTIONS
# ==========================================

BATCH_QUESTIONS = [
    "What is data engineering?",
    "What is an ETL pipeline? Explain its three stages.",
    "What is PostgreSQL? Explain its main features.",
    "What is Apache Spark?",
    "What is a data warehouse?",
    "What is the difference between batch processing and stream processing?",
    "Why is Apache Spark used in data engineering?",
    "What are the advantages of using PostgreSQL?",
    "How does data flow through an ETL pipeline?",
    "What is the difference between a database and a data warehouse?",
    "Give a real-world example of an ETL pipeline.",
    "Give one practical example each of batch and stream processing.",
    "How can Apache Spark process large datasets?",
    "Explain how an online store can use ETL for sales analysis.",
    "How does a data warehouse help businesses make decisions?",
    "Can Apache Spark handle real-time data processing?",
    "What happens during the Transform stage if duplicate records exist?",
    "Why is data cleaning important before loading data?",
    "How can PostgreSQL be used as a destination in an ETL pipeline?",
    "Explain how Spark, PostgreSQL and ETL work together."
]


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.2);
}

.hero {
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 24px;
    background: linear-gradient(120deg, #172554, #312e81);
}

.hero h1 {
    color: white;
    font-size: 30px;
}

.hero p {
    color: #dbeafe;
}

.stButton button {
    border-radius: 10px;
    min-height: 42px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# PIPELINE
# ==========================================

@st.cache_resource
def load_pipeline():
    pipeline = RAGPipeline()
    pipeline.load_existing_store()
    return pipeline


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_hash(content):
    return hashlib.sha256(content).hexdigest()


def find_existing_hashes():
    hashes = set()

    for path in DOCUMENT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in [".pdf", ".txt"]:
            hashes.add(calculate_hash(path.read_bytes()))

    return hashes


def get_document_count():
    return sum(
        1 for path in DOCUMENT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in [".pdf", ".txt"]
    )


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🤖 Knowledge Assistant")
    st.caption("Document intelligence powered by AI")

    st.divider()

    st.subheader("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", use_container_width=True):

        if not uploaded_files:
            st.warning("Please upload a document first.")

        else:
            try:
                existing_hashes = find_existing_hashes()
                added_files = 0
                duplicate_files = 0

                for uploaded_file in uploaded_files:

                    content = uploaded_file.getvalue()
                    file_hash = calculate_hash(content)

                    if file_hash in existing_hashes:
                        duplicate_files += 1
                        continue

                    filename = Path(uploaded_file.name).name
                    file_path = DOCUMENT_DIR / filename

                    if file_path.exists():
                        file_path = DOCUMENT_DIR / (
                            f"{file_path.stem}_{file_hash[:8]}"
                            f"{file_path.suffix}"
                        )

                    file_path.write_bytes(content)

                    existing_hashes.add(file_hash)
                    added_files += 1

                if added_files > 0:

                    with st.spinner("Indexing documents..."):

                        pipeline = RAGPipeline()

                        chunk_count = pipeline.ingest_documents(
                            str(DOCUMENT_DIR)
                        )

                    load_pipeline.clear()
                    st.session_state.batch_results = None

                    st.success(
                        f"{added_files} new document(s) added. "
                        f"{chunk_count} chunks indexed."
                    )

                if duplicate_files:
                    st.info(
                        f"{duplicate_files} duplicate document(s) skipped."
                    )

                if added_files == 0 and duplicate_files == 0:
                    st.warning("No documents were processed.")

            except Exception as error:
                st.error(f"Document processing failed: {error}")

    st.divider()

    st.subheader("Workspace")

    st.metric("Stored Documents", get_document_count())
    st.metric("Chat Messages", len(st.session_state.messages))

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("Python | FAISS | Hugging Face | Groq")


# ==========================================
# MAIN HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>Enterprise AI Knowledge Assistant</h1>
    <p>
        Ask questions. Explore documents.
        Get context-grounded AI answers.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# BATCH TESTING
# ==========================================

with st.expander(
    "🧪 Batch Testing - Evaluate 20 Questions",
    expanded=False
):

    st.write(
        "Automatically test your knowledge assistant "
        "using 20 predefined questions."
    )

    st.caption(
        "Measures response generation, retrieval sources "
        "and response time. It does not independently "
        "verify factual correctness."
    )

    st.write(f"Total questions: {len(BATCH_QUESTIONS)}")

    with st.expander("View Questions"):
        for number, question in enumerate(BATCH_QUESTIONS, 1):
            st.write(f"{number}. {question}")

    if st.button("Run All 20 Tests", use_container_width=True):

        try:
            pipeline = load_pipeline()

            results = []
            progress = st.progress(0)
            status = st.empty()

            for index, question in enumerate(BATCH_QUESTIONS):

                status.write(
                    f"Testing question {index + 1} of "
                    f"{len(BATCH_QUESTIONS)}..."
                )

                start_time = time.perf_counter()

                try:
                    response = pipeline.ask(question, k=3)

                    elapsed = time.perf_counter() - start_time

                    answer = response.get("answer", "")
                    sources = response.get("sources", [])

                    result_status = (
                        "Answered" if answer.strip() else "Empty"
                    )

                    source_names = sorted({
                        item.get("source", "Unknown")
                        for item in sources
                    })

                    results.append({
                        "Question": question,
                        "Status": result_status,
                        "Response Time (seconds)": round(elapsed, 2),
                        "Source Count": len(sources),
                        "Sources": ", ".join(source_names),
                        "Answer": answer
                    })

                except Exception as error:

                    elapsed = time.perf_counter() - start_time

                    results.append({
                        "Question": question,
                        "Status": "Error",
                        "Response Time (seconds)": round(elapsed, 2),
                        "Source Count": 0,
                        "Sources": "",
                        "Answer": str(error)
                    })

                progress.progress(
                    (index + 1) / len(BATCH_QUESTIONS)
                )

            status.success("Batch testing completed!")

            st.session_state.batch_results = results

        except Exception as error:
            st.error(f"Batch testing failed: {error}")

    # DISPLAY RESULTS

    if st.session_state.batch_results is not None:

        results = st.session_state.batch_results

        total = len(results)

        answered = sum(
            1 for item in results
            if item["Status"] == "Answered"
        )

        errors = sum(
            1 for item in results
            if item["Status"] == "Error"
        )

        average_time = (
            sum(
                item["Response Time (seconds)"]
                for item in results
            ) / total
            if total else 0
        )

        st.divider()
        st.subheader("Batch Test Results")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Questions", total)
        col2.metric("Answered", answered)
        col3.metric("Errors", errors)
        col4.metric("Average Time", f"{average_time:.2f}s")

        st.dataframe(
            [
                {
                    "Question": item["Question"],
                    "Status": item["Status"],
                    "Time (s)": item["Response Time (seconds)"],
                    "Sources": item["Source Count"]
                }
                for item in results
            ],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Individual Answers")

        for index, item in enumerate(results, 1):

            with st.expander(
                f"{index}. {item['Question']} - {item['Status']}"
            ):

                st.markdown(item["Answer"])

                st.write(
                    f"Response time: "
                    f"{item['Response Time (seconds)']} seconds"
                )

                st.write(
                    f"Sources: {item['Sources'] or 'None'}"
                )

        # CSV EXPORT

        csv_buffer = io.StringIO()

        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=[
                "Question",
                "Status",
                "Response Time (seconds)",
                "Source Count",
                "Sources",
                "Answer"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

        st.download_button(
            "Download Batch Results CSV",
            data=csv_buffer.getvalue(),
            file_name="batch_test_results.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==========================================
# CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 View Sources"):

                for source in message["sources"]:
                    st.write(f"• {source['source']}")


# ==========================================
# CHAT INPUT
# ==========================================

question = st.chat_input(
    "Ask anything about your uploaded documents..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:
                pipeline = load_pipeline()

                response = pipeline.ask(question, k=3)

                answer = response["answer"]
                sources = response["sources"]

                st.markdown(answer)

                if sources:

                    with st.expander("📚 View Sources"):

                        for source in sources:
                            st.write(f"• {source['source']}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:

                st.error(f"Unable to generate answer: {error}")

