import streamlit as st
import os
import json
import csv
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Coursera AI/ML Advisor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Lazy-load chatbot to avoid startup crash if data isn't ready
@st.cache_resource(show_spinner="Loading AI advisor...")
def get_chatbot():
    from src.chatbot import CourseraChatbot
    return CourseraChatbot()


def load_stats() -> dict:
    stats = {'pages': 0, 'qa_pairs': 0, 'courses': []}
    try:
        with open('data/raw/all_courses.json', 'r') as f:
            courses = json.load(f)
        stats['pages'] = len(courses)
        stats['courses'] = [c.get('title', '') for c in courses if c.get('title')]
    except FileNotFoundError:
        pass
    try:
        with open('data/processed/qa_dataset.csv', 'r') as f:
            stats['qa_pairs'] = max(0, sum(1 for _ in csv.reader(f)) - 1)
    except FileNotFoundError:
        pass
    return stats


def load_sample_qa(n: int = 5) -> list:
    try:
        with open('data/processed/qa_dataset.csv', 'r') as f:
            return list(csv.DictReader(f))[:n]
    except FileNotFoundError:
        return []


# ── Session state ────────────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'sources' not in st.session_state:
    st.session_state.sources = []
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = (
        os.path.exists('data/raw/all_courses.json')
        and os.path.exists('data/processed/qa_dataset.csv')
        and os.path.exists('data/chroma_db')
    )

# ── Header ───────────────────────────────────────────────────────────────────
stats = load_stats()

title_col, btn_col = st.columns([5, 1])
with title_col:
    st.title("🎓 Coursera AI/ML Course Advisor")
    st.caption("AI-powered course advisor that helps you find the right AI/ML course on Coursera.")
with btn_col:
    st.write("")
    if st.button("🗑️ Clear", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.sources = []
        if st.session_state.data_ready:
            get_chatbot().clear_history()
        st.rerun()

m1, m2, m3 = st.columns(3)
m1.metric("Pages Scraped", stats['pages'])
m2.metric("Q/A Pairs", stats['qa_pairs'])
m3.markdown("[🔗 Coursera AI/ML Courses ↗](https://www.coursera.org/courses?query=machine+learning)")

st.divider()
st.markdown("Ask me anything about AI & Machine Learning courses on Coursera — from beginner to advanced!")

if not st.session_state.data_ready:
    st.warning(
        "**Pipeline not yet run.** Please follow the setup steps:\n\n"
        "```\n"
        "python -m src.scraper          # Phase 1: Scrape Coursera\n"
        "python -m src.qa_generator     # Phase 2: Generate Q/A pairs\n"
        "python -m src.vector_store     # Phase 3: Build vector store\n"
        "streamlit run app.py           # Phase 5: Launch chatbot\n"
        "```\n\n"
        "Or run everything at once:\n```\npython run_pipeline.py\n```"
    )

# Sample Q/A expander
with st.expander("💡 Sample Questions", expanded=False):
    sample_qa = load_sample_qa(5)

    st.markdown("**Try asking:**")
    example_qs = [
        "What is the best course to start learning machine learning?",
        "How long does the Deep Learning Specialization take to complete?",
        "What skills will I gain from Andrew Ng's ML course?",
        "Which course teaches Python for AI and data science?",
        "Is there a course on generative AI and LLMs?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(example_qs):
        if cols[i % 2].button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state._prefill = q


# ── Chat history ─────────────────────────────────────────────────────────────
assistant_idx = 0
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if assistant_idx < len(st.session_state.sources):
                src = st.session_state.sources[assistant_idx]
                with st.expander("📌 Retrieval Details"):
                    badge = "✅ Q/A Match" if src.get('used_qa') else "🔍 Vector Search"
                    st.markdown(f"**Method:** {badge}")
                    st.markdown(f"**Source:** {src.get('source', '')}")
            assistant_idx += 1

# ── Chat input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop('_prefill', None)
prompt = st.chat_input("Ask about Coursera AI/ML courses...", key="chat_input")
if prefill and not prompt:
    prompt = prefill

if prompt:
    if not st.session_state.data_ready:
        st.error("Please run the pipeline first before chatting.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching courses..."):
            result = get_chatbot().chat(prompt)

        st.markdown(result['answer'])
        with st.expander("📌 Retrieval Details"):
            badge = "✅ Q/A Match" if result['used_qa'] else "🔍 Vector Search"
            st.markdown(f"**Method:** {badge}")
            st.markdown(f"**Source:** {result['source']}")

    st.session_state.messages.append({"role": "assistant", "content": result['answer']})
    st.session_state.sources.append({'used_qa': result['used_qa'], 'source': result['source']})
