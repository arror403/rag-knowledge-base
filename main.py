"""
Streamlit UI for Local RAG Knowledge Base.
"""

import os
import json
from datetime import datetime

import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="Local RAG Knowledge Base", layout="wide")

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]

DEFAULT_LLM_URL = os.environ.get(
    "DEFAULT_LLM_URL",
    "http://llama:8080/v1/chat/completions"
)


@st.cache_resource
def get_engine() -> RAGEngine:
    """Initialize RAG engine with local embedding model."""
    return RAGEngine()


def save_uploaded_files(files, data_dir: str) -> tuple[int, list[str]]:
    saved = []
    for file in files:
        ext = os.path.splitext(file.name)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        dest = os.path.join(data_dir, file.name)
        base, extension = os.path.splitext(dest)
        counter = 1
        while os.path.exists(dest):
            dest = f"{base}_{counter}{extension}"
            counter += 1
        with open(dest, "wb") as f:
            f.write(file.getbuffer())
        saved.append(os.path.basename(dest))
    return len(saved), saved


def list_data_files(data_dir: str) -> list[str]:
    files = []
    for root, _, filenames in os.walk(data_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS:
                rel_path = os.path.relpath(os.path.join(root, fname), data_dir)
                files.append(rel_path)
    return sorted(files)


def delete_file(data_dir: str, filename: str) -> bool:
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def export_history_markdown(history: list[dict]) -> str:
    if not history:
        return "# Q&A History\n\nNo questions asked yet.\n"

    lines = ["# Q&A History\n"]
    lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")

    for i, entry in enumerate(history, 1):
        lines.append(f"## Q{i}: {entry['question']}\n")
        lines.append(f"**Time:** {entry['timestamp']}\n")
        lines.append(f"### Answer\n")
        lines.append(f"{entry['answer']}\n")
        if entry.get("context"):
            lines.append(f"### Retrieved Context\n")
            lines.append(f"```\n{entry['context']}\n```\n")
        lines.append("---\n")

    return "\n".join(lines)


def export_history_json(history: list[dict]) -> str:
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "total_questions": len(history),
        "history": history
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def init_session_state():
    """Initialize all session state variables."""
    if "db_loaded" not in st.session_state:
        st.session_state.db_loaded = False
    if "history" not in st.session_state:
        st.session_state.history = []
    if "auto_loaded" not in st.session_state:
        st.session_state.auto_loaded = False
    if "pending_answer" not in st.session_state:
        st.session_state.pending_answer = None


def auto_load_db(engine: RAGEngine):
    if st.session_state.auto_loaded:
        return
    st.session_state.auto_loaded = True
    if engine.has_db():
        if engine.load_db():
            st.session_state.db_loaded = True


def process_query(engine: RAGEngine, query: str, api_url: str):
    """
    Process a question: retrieve context, query LLM, save to history.
    Called BEFORE sidebar renders so history is up to date.
    """
    if not query or not st.session_state.db_loaded:
        return

    with st.spinner("Searching and generating..."):
        try:
            context = engine.retrieve(query)
            if context is None:
                st.session_state.pending_answer = {
                    "error": None,
                    "empty": True
                }
                return

            answer = engine.query_llm(query, context, api_url=api_url)

            entry = {
                "question": query,
                "answer": answer,
                "context": context,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Add to history IMMEDIATELY so sidebar count is correct
            st.session_state.history.append(entry)

            # Store for display in main area
            st.session_state.pending_answer = {
                "error": None,
                "empty": False,
                "entry": entry
            }

        except RuntimeError as e:
            st.session_state.pending_answer = {"error": f"DB Error: {e}"}
        except ConnectionError as e:
            st.session_state.pending_answer = {"error": f"Connection Error: {e}"}
        except TimeoutError:
            st.session_state.pending_answer = {"error": "Timed out. Model may be overloaded."}
        except Exception as e:
            st.session_state.pending_answer = {"error": f"Error: {e}"}


def main():
    init_session_state()

    try:
        engine = get_engine()
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()

    auto_load_db(engine)

    st.title("🛠️ Local Technical RAG")

    # ── Query Input (rendered first, but we need the values early) ─────────
    # Using a callback ensures history is updated BEFORE the sidebar renders
    col_input, col_button = st.columns([5, 1])

    with col_input:
        query = st.text_input(
            "Enter your technical question:",
            placeholder="e.g., How does the authentication system work?",
            key="query_input",
            label_visibility="visible"
        )

    with col_button:
        st.markdown("<div style='height: 1.65rem'></div>", unsafe_allow_html=True)
        ask_clicked = st.button("🔍 Ask", use_container_width=True)

    # ── Process query BEFORE sidebar so history is current ─────────────────
    if ask_clicked and query:
        if not st.session_state.db_loaded:
            st.warning("Please load or ingest a vector database first.")
        else:
            # Read api_url from session state (set by sidebar input below)
            api_url = st.session_state.get("api_url_input", DEFAULT_LLM_URL)
            process_query(engine, query, api_url)

    # ── Display latest answer ──────────────────────────────────────────────
    pending = st.session_state.pending_answer
    if pending is not None:
        if pending.get("error"):
            st.error(pending["error"])
        elif pending.get("empty"):
            st.info("No relevant documents found.")
        else:
            entry = pending["entry"]
            st.subheader("Answer")
            st.markdown(entry["answer"])
            with st.expander("📎 Retrieved Context"):
                st.text(entry["context"])

        st.session_state.pending_answer = None

    # ── Sidebar (renders AFTER history is updated) ─────────────────────────
    with st.sidebar:

        # ── LLM Server ────────────────────────────────────────────────────
        st.header("🖥️ llama-server")
        st.text_input(
            "API Endpoint",
            value=DEFAULT_LLM_URL,
            key="api_url_input"
        )

        st.divider()

        # ── Documents ─────────────────────────────────────────────────────
        st.header("📁 Documents")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if st.button("💾 Save Uploaded Files", use_container_width=True):
                count, names = save_uploaded_files(uploaded_files, engine.data_dir)
                if count > 0:
                    st.success(f"Saved {count} file(s)")
                    st.info("Click 'Ingest' to index them.")
                else:
                    st.warning("No supported files.")

        existing_files = list_data_files(engine.data_dir)
        if existing_files:
            with st.expander(f"📄 Files ({len(existing_files)})", expanded=False):
                for fname in existing_files:
                    col_name, col_del = st.columns([4, 1])
                    col_name.text(fname)
                    if col_del.button("🗑️", key=f"del_{fname}"):
                        if delete_file(engine.data_dir, fname):
                            st.rerun()
        else:
            st.caption("No documents yet.")

        st.divider()

        # ── Vector DB ─────────────────────────────────────────────────────
        st.header("⚙️ Vector DB")

        status_icon = "✅" if st.session_state.db_loaded else "❌"
        st.metric(
            "Status",
            f"{status_icon} {'Loaded' if st.session_state.db_loaded else 'Not Loaded'}"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Ingest", use_container_width=True):
                with st.spinner("Ingesting..."):
                    try:
                        count = engine.ingest_docs()
                        st.session_state.db_loaded = True
                        st.success(f"{count} chunks indexed.")
                    except Exception as e:
                        st.error(str(e))
        with col2:
            if st.button("📂 Load DB", use_container_width=True):
                if engine.load_db():
                    st.session_state.db_loaded = True
                    st.success("Loaded.")
                else:
                    st.error("No DB found.")

        st.divider()

        # ── History ───────────────────────────────────────────────────────
        st.header("📜 History")

        st.caption(f"{len(st.session_state.history)} question(s)")

        if st.session_state.history:
            col_md, col_json = st.columns(2)
            with col_md:
                st.download_button(
                    "📄 .md",
                    data=export_history_markdown(st.session_state.history),
                    file_name=f"rag_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col_json:
                st.download_button(
                    "📋 .json",
                    data=export_history_json(st.session_state.history),
                    file_name=f"rag_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

    # ── Q&A History Display ────────────────────────────────────────────────
    if st.session_state.history:
        st.divider()
        st.subheader("💬 Q&A History")

        for i, entry in enumerate(reversed(st.session_state.history)):
            q_num = len(st.session_state.history) - i

            with st.container():
                st.markdown(f"**Q{q_num}:** {entry['question']}")
                st.caption(f"🕐 {entry['timestamp']}")
                st.markdown(entry["answer"])

                with st.expander(f"📎 Retrieved Context (Q{q_num})"):
                    st.text(entry["context"])

                st.divider()


if __name__ == "__main__":
    main()
