"""
AI Video Assistant — Streamlit UI
"Studio Deck" theme: a video-editing-console aesthetic for a
download -> transcribe -> summarize -> chat pipeline.

Drop this file in the project root (next to main.py) and run:
    uv run streamlit run app.py
"""

import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
# ... baaki imports

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question


st.set_page_config(
    page_title="AI Video Assistant with RAG",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# THEME — "Studio Deck"
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root{
  --bg:#0B0F17;
  --panel:#131924;
  --panel-alt:#1A2233;
  --border:rgba(255,255,255,0.08);
  --coral:#FF5D3A;
  --coral-glow:rgba(255,93,58,0.35);
  --teal:#33D6C0;
  --teal-glow:rgba(51,214,192,0.30);
  --text:#EDF0F5;
  --muted:#8891A6;
}

@media (prefers-reduced-motion: reduce){
  *{ animation-duration:0.001ms !important; transition-duration:0.001ms !important; }
}

html, body, [class*="css"]{
  font-family:'IBM Plex Sans', sans-serif;
  color:var(--text);
}

.stApp{
  background:
    radial-gradient(ellipse 900px 500px at 15% -10%, rgba(255,93,58,0.08), transparent),
    radial-gradient(ellipse 900px 500px at 100% 0%, rgba(51,214,192,0.07), transparent),
    var(--bg);
}

h1, h2, h3, h4, .studio-heading{
  font-family:'Sora', sans-serif;
  letter-spacing:-0.01em;
}

section[data-testid="stSidebar"]{
  background:var(--panel);
  border-right:1px solid var(--border);
}

section[data-testid="stSidebar"] .block-container{ padding-top:1.6rem; }

/* ── Media Bin label ─────────────────────────────────────────────── */
.bin-label{
  font-family:'Sora', sans-serif;
  font-size:0.95rem;
  font-weight:700;
  color:var(--text);
  margin-bottom:0.15rem;
}
.bin-sub{
  color:var(--muted);
  font-size:0.82rem;
  margin-bottom:1.1rem;
}

/* ── REC indicator ───────────────────────────────────────────────── */
.rec-badge{
  display:inline-flex; align-items:center; gap:0.45rem;
  font-family:'IBM Plex Mono', monospace;
  font-size:0.78rem;
  color:var(--coral);
  border:1px solid rgba(255,93,58,0.35);
  background:rgba(255,93,58,0.08);
  padding:0.28rem 0.7rem;
  border-radius:3px;
}
.rec-dot{
  width:8px; height:8px; border-radius:50%;
  background:var(--coral);
  animation:pulse 1.1s ease-in-out infinite;
}
@keyframes pulse{
  0%,100%{ opacity:1; box-shadow:0 0 0 0 var(--coral-glow); }
  50%{ opacity:0.55; box-shadow:0 0 0 6px transparent; }
}

/* ── Hero waveform ───────────────────────────────────────────────── */
.hero{
  display:flex; align-items:center; gap:1.1rem;
  padding:1.4rem 1.6rem;
  background:linear-gradient(135deg, var(--panel), var(--panel-alt));
  border:1px solid var(--border);
  border-left:3px solid var(--coral);
  margin-bottom:1.4rem;
}
.hero-title{
  font-size:1.7rem; font-weight:700; margin:0; line-height:1.25;
}
.hero-sub{
  color:var(--muted); font-size:0.88rem; margin-top:0.3rem;
}
.wave{ display:flex; align-items:flex-end; gap:3px; height:34px; }
.wave span{
  width:4px; background:var(--teal); border-radius:1px;
  animation:bounce 1.2s ease-in-out infinite;
}
.wave span:nth-child(1){ height:40%; animation-delay:0s; }
.wave span:nth-child(2){ height:80%; animation-delay:0.1s; }
.wave span:nth-child(3){ height:55%; animation-delay:0.2s; }
.wave span:nth-child(4){ height:100%; animation-delay:0.3s; }
.wave span:nth-child(5){ height:65%; animation-delay:0.4s; }
.wave span:nth-child(6){ height:85%; animation-delay:0.5s; }
.wave span:nth-child(7){ height:35%; animation-delay:0.6s; }
@keyframes bounce{
  0%,100%{ transform:scaleY(0.4); }
  50%{ transform:scaleY(1); }
}

/* ── Panels (sharp-cornered console modules) ────────────────────── */
.panel{
  background:var(--panel);
  border:1px solid var(--border);
  padding:1.3rem 1.4rem;
  transition:border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}
.panel:hover{
  border-color:rgba(51,214,192,0.35);
  transform:translateY(-2px);
  box-shadow:0 8px 24px rgba(0,0,0,0.35);
}
.panel h4{ margin-top:0; color:var(--teal); font-size:0.92rem; }
.panel-body{ color:var(--text); font-size:0.93rem; line-height:1.65; white-space:pre-wrap; }

/* ── Transcript (monospace — grounded in subtitle/SRT convention) ── */
.transcript-box{
  background:var(--panel-alt);
  border:1px solid var(--border);
  padding:1.2rem 1.3rem;
  max-height:480px;
  overflow-y:auto;
  font-family:'IBM Plex Mono', monospace;
  font-size:0.85rem;
  line-height:1.75;
  color:#CBD2E1;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button{
  background:var(--coral);
  color:#0B0F17;
  border:none;
  font-weight:600;
  font-family:'Sora', sans-serif;
  padding:0.55rem 1.2rem;
  transition:transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}
.stButton > button:hover{
  background:#FF7856;
  transform:translateY(-2px);
  box-shadow:0 6px 18px var(--coral-glow);
}
.stButton > button:active{ transform:translateY(0); }

/* ── Tabs ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{ gap:0.4rem; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"]{
  color:var(--muted);
  font-family:'Sora', sans-serif;
  font-weight:600;
  font-size:0.88rem;
  padding:0.6rem 0.2rem;
  position:relative;
}
.stTabs [data-baseweb="tab"]::after{
  content:"";
  position:absolute; left:0; bottom:-1px;
  width:0%; height:2px; background:var(--teal);
  transition:width 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover::after{ width:100%; }
.stTabs [aria-selected="true"]{ color:var(--text) !important; }
.stTabs [aria-selected="true"]::after{ width:100%; background:var(--coral); }

/* ── Chat dock ───────────────────────────────────────────────────── */
.dock-label{
  font-family:'Sora', sans-serif; font-weight:700; font-size:0.95rem;
  margin-bottom:0.9rem; color:var(--text);
}
.bubble{
  padding:0.65rem 0.9rem;
  margin-bottom:0.6rem;
  border-radius:10px;
  font-size:0.87rem;
  line-height:1.55;
  animation:slideIn 0.25s ease;
}
@keyframes slideIn{
  from{ opacity:0; transform:translateY(6px); }
  to{ opacity:1; transform:translateY(0); }
}
.bubble-user{
  background:rgba(255,93,58,0.14);
  border:1px solid rgba(255,93,58,0.3);
  margin-left:1.4rem;
}
.bubble-assistant{
  background:var(--panel-alt);
  border:1px solid rgba(51,214,192,0.25);
  margin-right:1.4rem;
}

/* focus visibility for accessibility */
button:focus-visible, input:focus-visible, textarea:focus-visible{
  outline:2px solid var(--teal) !important;
  outline-offset:2px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR — "Media Bin"
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="bin-label">🎞️ AI Video Assistant With RAG </div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="bin-sub">Load a source to begin a session</div>',
        unsafe_allow_html=True,
    )

    source_type = st.radio(
        "Source", ["YouTube URL", "Upload file"], label_visibility="collapsed"
    )

    source_path = None
    if source_type == "YouTube URL":
        url = st.text_input(
            "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
        )
        if url:
            source_path = url.strip()
    else:
        uploaded = st.file_uploader(
            "Audio / video file", type=["wav", "mp3", "m4a", "mp4", "webm"]
        )
        if uploaded is not None:
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            source_path = tmp_path

    language = st.selectbox("Language", ["english", "hinglish"])

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    run_clicked = st.button(
        "▸ Process", use_container_width=True, disabled=not source_path
    )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    if st.session_state.processing:
        st.markdown(
            '<div class="rec-badge"><span class="rec-dot"></span>REC — processing</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────
# PIPELINE — run stage by stage with a visible export-queue style status
# ──────────────────────────────────────────────────────────────────────────
def run_pipeline_with_status(source: str, language: str) -> dict:
    with st.status("Running pipeline…", expanded=True) as status:
        st.write("🎞️ Acquiring & chunking audio…")
        chunks = process_input(source)

        st.write("🗣️ Transcribing…")
        transcript = transcribe_all(chunks, language)

        st.write("🏷️ Generating title…")
        title = generate_title(transcript)

        st.write("📋 Summarizing…")
        summary = summarize(transcript)

        st.write("✅ Extracting action items…")
        action_items = extract_action_items(transcript)

        st.write("🔑 Extracting key decisions…")
        decisions = extract_key_decisions(transcript)

        st.write("❓ Extracting open questions…")
        questions = extract_questions(transcript)

        st.write("🧠 Building chat index…")
        rag_chain = build_rag_chain(transcript)

        status.update(label="Pipeline complete", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if run_clicked and source_path:
    st.session_state.processing = True
    st.session_state.chat_history = []
    try:
        st.session_state.result = run_pipeline_with_status(source_path, language)
    finally:
        st.session_state.processing = False


# ──────────────────────────────────────────────────────────────────────────
# MAIN AREA — "Screening Room" + "Chat Dock"
# ──────────────────────────────────────────────────────────────────────────
result = st.session_state.result

if result is None:
    st.markdown(
        """
        <div class="hero">
            <div class="wave">
                <span></span><span></span><span></span><span></span><span></span><span></span><span></span>
            </div>
            <div>
                <p class="hero-title">Studio Deck</p>
                <p class="hero-sub">Load a video in the Media Bin to transcribe, summarize, and chat with it.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="hero">
            <div class="wave">
                <span></span><span></span><span></span><span></span><span></span><span></span><span></span>
            </div>
            <div>
                <p class="hero-title">{result["title"]}</p>
                <p class="hero-sub">Session ready — browse the tabs or ask the assistant on the right</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    screening_col, dock_col = st.columns([2.1, 1])

    with screening_col:
        tab_summary, tab_transcript, tab_actions, tab_decisions, tab_questions = (
            st.tabs(
                [
                    "Summary",
                    "Transcript",
                    "Action Items",
                    "Key Decisions",
                    "Open Questions",
                ]
            )
        )

        with tab_summary:
            st.markdown(
                f'<div class="panel"><h4>Summary</h4><div class="panel-body">{result["summary"]}</div></div>',
                unsafe_allow_html=True,
            )

        with tab_transcript:
            st.markdown(
                f'<div class="transcript-box">{result["transcript"]}</div>',
                unsafe_allow_html=True,
            )

        with tab_actions:
            st.markdown(
                f'<div class="panel"><h4>Action Items</h4><div class="panel-body">{result["action_items"]}</div></div>',
                unsafe_allow_html=True,
            )

        with tab_decisions:
            st.markdown(
                f'<div class="panel"><h4>Key Decisions</h4><div class="panel-body">{result["key_decisions"]}</div></div>',
                unsafe_allow_html=True,
            )

        with tab_questions:
            st.markdown(
                f'<div class="panel"><h4>Open Questions</h4><div class="panel-body">{result["open_questions"]}</div></div>',
                unsafe_allow_html=True,
            )

    with dock_col:
        st.markdown(
            '<div class="dock-label">💬 Chat Dock</div>', unsafe_allow_html=True
        )

        chat_container = st.container(height=420)
        with chat_container:
            for role, msg in st.session_state.chat_history:
                bubble_class = "bubble-user" if role == "user" else "bubble-assistant"
                speaker = "You" if role == "user" else "Assistant"
                st.markdown(
                    f'<div class="bubble {bubble_class}"><b>{speaker}:</b> {msg}</div>',
                    unsafe_allow_html=True,
                )

        question = st.chat_input("Ask something about this video…")
        if question:
            st.session_state.chat_history.append(("user", question))
            answer = ask_question(result["rag_chain"], question)
            st.session_state.chat_history.append(("assistant", answer))
            st.rerun()
