import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
from pypdf import PdfReader

if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from rag_pipeline import (
    get_answer_detailed,
    prepare_rag_context,
    get_cached_llm,
    get_suggested_followups,
    search_knowledge_base,
    check_system_health,
    reload_vectorstore,
    is_conversational_greeting,
    get_conversational_response,
)
from ingest import build_vector_database

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=f"{config.COLLEGE_SHORT_NAME} Connect AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 2. SESSION STATE
# ─────────────────────────────────────────────
for k, v in {
    "messages": [],
    "page": "AI Assistant",
    "selected_model": config.DEFAULT_LLM_MODEL,
    "top_k": config.DEFAULT_TOP_K,
    "temperature": config.DEFAULT_TEMPERATURE,
    "conversation_history": [],
    "query_analytics": [],
    "feedback_store": {},
    "pending_query": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource(show_spinner=False)
def prewarm_rag_engine():
    """Pre-warms embedding weights and vectorstore on startup to avoid cold start lag."""
    try:
        from rag_pipeline import get_cached_embeddings, get_cached_vectorstore
        get_cached_embeddings()
        get_cached_vectorstore()
        return True
    except Exception:
        return False

# Initialize in-memory RAG singleton on startup
prewarm_rag_engine()

# ─────────────────────────────────────────────
# 3. GLOBAL CSS — FULL PRODUCTION DESIGN SYSTEM
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── GOOGLE FONTS ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── DESIGN TOKENS ── */
:root {
  --bg:          #05080F;
  --bg-1:        #090D19;
  --bg-2:        #0D1220;
  --bg-3:        #111827;
  --border:      rgba(255,255,255,0.07);
  --border-hi:   rgba(99,102,241,0.45);
  --indigo:      #6366F1;
  --indigo-dark: #4F46E5;
  --blue:        #3B82F6;
  --emerald:     #10B981;
  --amber:       #F59E0B;
  --text:        #F1F5F9;
  --text-2:      #94A3B8;
  --text-3:      #64748B;
  --radius-sm:   8px;
  --radius-md:   12px;
  --radius-lg:   16px;
  --radius-xl:   20px;
  --shadow-card: 0 4px 24px rgba(0,0,0,0.35);
  --shadow-glow: 0 0 40px rgba(99,102,241,0.15);
}

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text);
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.6;
}

/* ── WIPE ALL STREAMLIT CHROME ── */
#MainMenu                          { display: none !important; }
footer                             { display: none !important; }
header[data-testid="stHeader"]     { display: none !important; }
[data-testid="stDeployButton"]     { display: none !important; }
[data-testid="stToolbar"]          { display: none !important; }
.stDecoration                      { display: none !important; }
section[data-testid="stSidebar"] > div:first-child > div:first-child > div[data-testid="stVerticalBlock"] > div:last-child { display: none !important; }

/* ── MAIN CONTENT ── */
.block-container {
  max-width: 1280px !important;
  padding: 0 2rem 4rem !important;
  margin: 0 auto !important;
}

/* ── TYPOGRAPHY ── */
h1,h2,h3,h4,h5 {
  font-family: 'Space Grotesk', sans-serif !important;
  color: var(--text) !important;
  letter-spacing: -0.02em;
  margin: 0;
}
p, span, label, li { color: var(--text); }
code, kbd {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px;
  background: rgba(255,255,255,0.06);
  padding: 2px 6px;
  border-radius: 4px;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
  background: var(--bg-1) !important;
  border-right: 1px solid var(--border) !important;
  padding: 0 !important;
}
section[data-testid="stSidebar"] > div {
  padding: 0 !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── SIDEBAR BUTTONS — reset & restyle ── */
section[data-testid="stSidebar"] .stButton { width: 100%; }
section[data-testid="stSidebar"] .stButton > button {
  width: 100% !important;
  background: transparent !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-2) !important;
  font-weight: 500 !important;
  font-size: 13.5px !important;
  text-align: left !important;
  justify-content: flex-start !important;
  padding: 10px 14px !important;
  min-height: 40px !important;
  cursor: pointer;
  transition: background 0.18s, color 0.18s, transform 0.18s;
  box-shadow: none !important;
  outline: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(99,102,241,0.12) !important;
  color: #fff !important;
  transform: none !important;
}
section[data-testid="stSidebar"] .stButton > button:focus {
  box-shadow: none !important;
  outline: none !important;
}
/* Active nav button */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: linear-gradient(90deg, rgba(99,102,241,0.22) 0%, rgba(59,130,246,0.14) 100%) !important;
  border: 1px solid rgba(99,102,241,0.40) !important;
  color: #fff !important;
  font-weight: 600 !important;
}

/* ── MAIN AREA BUTTONS ── */
.stButton > button {
  background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  color: #fff !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  min-height: 36px !important;
  padding: 8px 16px !important;
  cursor: pointer;
  transition: opacity 0.18s, transform 0.18s !important;
  box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
  opacity: 0.88 !important;
  transform: translateY(-1px) !important;
}
.stButton > button:focus {
  box-shadow: 0 0 0 3px rgba(99,102,241,0.35) !important;
}
.stButton > button[disabled] {
  opacity: 0.35 !important;
  cursor: not-allowed !important;
  transform: none !important;
  background: var(--bg-3) !important;
  box-shadow: none !important;
}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-2) !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  min-height: 36px !important;
  box-shadow: none !important;
}
.stDownloadButton > button:hover {
  border-color: var(--indigo) !important;
  color: #fff !important;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 18px 22px !important;
  margin-bottom: 12px !important;
  box-shadow: var(--shadow-card) !important;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] ~ div {
  border-left: 2px solid rgba(99,102,241,0.4);
  padding-left: 12px;
  margin-left: 4px;
}

/* ── CHAT INPUT (STABLE NATIVE CONTRAST) ── */
[data-testid="stChatInput"] {
  background: #0E1726 !important;
  background-color: #0E1726 !important;
  border: 1.5px solid rgba(99, 102, 241, 0.55) !important;
  border-radius: 14px !important;
  box-shadow: 0 6px 28px rgba(0, 0, 0, 0.6) !important;
}

[data-testid="stChatInput"]:focus-within {
  border-color: #818CF8 !important;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.35), 0 8px 30px rgba(0, 0, 0, 0.7) !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input,
[data-testid="stChatInputContainer"] textarea,
div[data-testid="stChatInput"] textarea {
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
  background: transparent !important;
  background-color: transparent !important;
  font-size: 14.5px !important;
  font-weight: 500 !important;
  caret-color: #818CF8 !important;
}

[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputContainer"] textarea::placeholder {
  color: #94A3B8 !important;
  -webkit-text-fill-color: #94A3B8 !important;
  opacity: 1 !important;
}

/* Chat Input Submit button */
[data-testid="stChatInput"] button {
  background: linear-gradient(135deg, #6366F1, #3B82F6) !important;
  color: #FFFFFF !important;
  border-radius: 10px !important;
  border: none !important;
}

/* Bottom floating input container background fix */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stChatFloatingInputContainer"],
div[class*="stBottom"],
div[class*="floating"],
div[class*="bottom"],
footer {
  background: #070B16 !important;
  background-color: #070B16 !important;
  color: #F8FAFC !important;
  border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  padding: 18px 20px !important;
  box-shadow: var(--shadow-card) !important;
}
[data-testid="stMetricLabel"] {
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--text-3) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 28px !important;
  font-weight: 700 !important;
  color: var(--text) !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; color: var(--emerald) !important; }

/* ── CONTAINERS WITH BORDER ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 6px !important;
}

/* ── EXPANDERS ── */
details[data-testid="stExpander"] {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}
details[data-testid="stExpander"] summary {
  font-size: 12.5px !important;
  font-weight: 600 !important;
  color: #A5B4FC !important;
}

/* ── SELECT BOX / SLIDERS / INPUTS ── */
[data-testid="stSelectbox"] > div > div {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
}
.stSlider [data-testid="stSlider"] > div > div > div > div { background: var(--indigo) !important; }
.stTextInput > div > div > input {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--indigo) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.20) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: var(--bg-2) !important;
  border: 2px dashed var(--border-hi) !important;
  border-radius: var(--radius-md) !important;
}

/* ── PROGRESS BAR ── */
[data-testid="stProgress"] > div { background: var(--bg-3) !important; }
[data-testid="stProgress"] > div > div { background: var(--indigo) !important; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}

/* ── DIVIDER ── */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* ── CAPTION / SMALL TEXT ── */
[data-testid="stCaptionContainer"] { color: var(--text-3) !important; font-size: 11.5px !important; }

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: var(--indigo) !important; }

/* ── INFO / SUCCESS / WARNING / ERROR ── */
[data-testid="stAlert"] {
  border-radius: var(--radius-md) !important;
  border: none !important;
  font-size: 13px !important;
}

/* ── POPOVER ── */
[data-testid="stPopoverBody"] {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}

/* ── BARCHART ── */
[data-testid="stVegaLiteChart"] {
  background: transparent !important;
}

/* ── UTILITY CLASSES (injected via markdown) ── */
.page-eyebrow {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #818CF8;
  margin-bottom: 4px;
}
.page-title {
  font-size: 28px;
  font-weight: 700;
  line-height: 1.2;
  color: #fff;
  margin: 0 0 6px 0;
}
.page-desc {
  font-size: 13.5px;
  color: var(--text-2);
  margin: 0;
  line-height: 1.6;
}
/* Sidebar brand */
.sb-brand {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 20px 16px 16px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 10px;
}
.sb-icon {
  width: 40px; height: 40px;
  border-radius: 10px;
  background: linear-gradient(135deg, #4F46E5, #3B82F6);
  display: flex; align-items: center; justify-content: center;
  font-size: 20px;
  box-shadow: 0 6px 20px rgba(79,70,229,0.4);
  flex-shrink: 0;
}
.sb-name { font-family: 'Space Grotesk', sans-serif; font-size: 16px; font-weight: 700; color: #fff; }
.sb-sub  { font-size: 9.5px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #6366F1; }
.sb-section {
  font-size: 9.5px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--text-3);
  padding: 14px 16px 6px;
}
/* Status dot */
.dot-live {
  display: inline-block; width: 7px; height: 7px;
  border-radius: 50%; background: var(--emerald);
  box-shadow: 0 0 8px var(--emerald);
  margin-right: 6px;
}
/* Badge / pill */
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 100px;
  background: rgba(99,102,241,0.12);
  border: 1px solid rgba(99,102,241,0.28);
  font-size: 10.5px; font-weight: 700;
  letter-spacing: 0.07em; text-transform: uppercase;
  color: #A5B4FC;
}
/* Hero */
.hero-wrap {
  text-align: center;
  padding: 18px 16px 6px;
  max-width: 820px;
  margin: 0 auto 28px;
}
.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: clamp(28px, 4vw, 44px);
  font-weight: 700;
  line-height: 1.15;
  background: linear-gradient(135deg, #fff 30%, #c7d2fe 65%, #818CF8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 14px 0 10px;
}
.hero-sub {
  font-size: 14px;
  color: var(--text-2);
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.65;
}
/* Topic cards */
.topic-card {
  background: linear-gradient(145deg, var(--bg-2) 0%, var(--bg-1) 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 16px;
  height: 100%;
  transition: border-color 0.22s, box-shadow 0.22s, transform 0.22s;
  cursor: pointer;
}
.topic-card:hover {
  border-color: rgba(99,102,241,0.50);
  box-shadow: 0 8px 28px rgba(99,102,241,0.16);
  transform: translateY(-2px);
}
.topic-card-icon {
  font-size: 22px;
  margin-bottom: 8px;
}
.topic-card-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 13.5px; font-weight: 700;
  color: #fff;
  margin-bottom: 4px;
}
.topic-card-desc {
  font-size: 11.5px;
  color: var(--text-2);
  line-height: 1.45;
}
/* Source card */
.src-badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 4px 10px; border-radius: 6px;
  background: rgba(99,102,241,0.15);
  border: 1px solid rgba(99,102,241,0.30);
  font-size: 11px; font-weight: 600;
  color: #C7D2FE; margin: 4px 4px 0 0;
}
.src-excerpt {
  background: rgba(5,8,15,0.7);
  border-left: 3px solid var(--indigo);
  padding: 10px 14px;
  margin-top: 8px;
  border-radius: 0 6px 6px 0;
  font-size: 12px;
  color: #CBD5E1;
  line-height: 1.55;
  font-style: italic;
}
/* Sidebar status box */
.sb-status {
  margin: 8px 10px;
  padding: 12px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  font-size: 11.5px;
  line-height: 1.7;
}
.sb-status a { color: #A5B4FC !important; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 4. HELPERS
# ─────────────────────────────────────────────
def set_pending(q: str):
    st.session_state.pending_query = q
    st.session_state.page = "AI Assistant"

def reset_chat():
    st.session_state.messages = []
    st.session_state.pending_query = None
    st.session_state.page = "AI Assistant"


# ─────────────────────────────────────────────
# 5. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown(f"""
    <div class="sb-brand">
      <div class="sb-icon">🎓</div>
      <div>
        <div class="sb-name">{config.COLLEGE_SHORT_NAME} Connect</div>
        <div class="sb-sub">AI Campus Assistant</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="sb-section">Workspaces</div>', unsafe_allow_html=True)
    nav_items = [
        ("💬", "AI Assistant"),
        ("📊", "College Dashboard"),
        ("📁", "Document Vault"),
        ("🔍", "Semantic Search Lab"),
        ("📈", "Usage Analytics"),
        ("⚙️", "Settings & Health"),
    ]
    for icon, label in nav_items:
        active = st.session_state.page == label
        btn_type = "primary" if active else "secondary"
        if st.button(f"{icon}  {label}", key=f"nav_{label}", type=btn_type, use_container_width=True):
            st.session_state.page = label
            st.rerun()

    # Session
    st.markdown('<div class="sb-section">Session</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("＋ New", key="btn_new", use_container_width=True):
            reset_chat(); st.rerun()
    with c2:
        if st.session_state.messages:
            md = f"# MIST AI Chat\n*{datetime.now().strftime('%Y-%m-%d')}*\n\n"
            for m in st.session_state.messages:
                role = "Student" if m["role"] == "user" else "MIST AI"
                md += f"### {role}\n{m['content']}\n\n"
            st.download_button("📥 Export", data=md,
                file_name=f"MIST_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown", use_container_width=True)
        else:
            st.button("📥 Export", disabled=True, use_container_width=True)

    # Recent
    if st.session_state.conversation_history:
        st.markdown('<div class="sb-section">Recent Queries</div>', unsafe_allow_html=True)
        for i, q in enumerate(st.session_state.conversation_history[:5]):
            short = q[:28] + ("…" if len(q) > 28 else "")
            if st.button(f"◦  {short}", key=f"hist_{i}", use_container_width=True):
                set_pending(q); st.rerun()

    # Design Studio Connectors
    st.markdown('<div class="sb-section">Design Studio</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 0 10px; margin-bottom: 6px;">
      <a href="https://www.figma.com/design/hALGvS4Oin6XzaE7sFKV9j" target="_blank" style="text-decoration:none; display:flex; align-items:center; gap:8px; padding:8px 12px; background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.3); border-radius:8px; color:#CAD8FF; font-size:12px; font-weight:600; margin-bottom:6px;">
        <span>🎨</span> Figma UI Architecture ↗
      </a>
      <a href="https://canva.link/wm3n3kn0r5b7khr" target="_blank" style="text-decoration:none; display:flex; align-items:center; gap:8px; padding:8px 12px; background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.3); border-radius:8px; color:#93C5FD; font-size:12px; font-weight:600;">
        <span>✨</span> Canva Portal Banner ↗
      </a>
    </div>
    """, unsafe_allow_html=True)

    # Status card
    st.markdown(f"""
    <div class="sb-status">
      <div style="margin-bottom:8px;">
        <span class="dot-live"></span>
        <strong>System Online</strong>
      </div>
      <div style="color:#64748B; font-size:11px; line-height:1.8;">
        <span style="color:#94A3B8;">Engine:</span> {st.session_state.selected_model.split('/')[-1]}<br>
        <span style="color:#94A3B8;">Docs:</span> 6 PDFs · 22 Vectors<br>
        <span style="color:#94A3B8;">DB:</span> ChromaDB (Cached)
      </div>
      <div style="margin-top:10px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.06); font-size:10.5px; color:#4B5563; line-height:1.8;">
        📞 {config.COLLEGE_PHONE}<br>
        ✉️ {config.COLLEGE_EMAIL}<br>
        🌐 <a href="{config.COLLEGE_WEBSITE}" target="_blank">mist.edu.in</a>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 6. AI ASSISTANT
# ─────────────────────────────────────────────
if st.session_state.page == "AI Assistant":

    # Top bar
    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.markdown(f"""
        <div style="padding:16px 0 10px;">
          <div class="page-eyebrow">🎓 {config.COLLEGE_SHORT_NAME} · Academic Intelligence Platform</div>
          <h2 class="page-title">MIST Connect AI</h2>
          <p class="page-desc">Instant answers grounded in official M24/R22 regulations, fee circulars, and departmental data.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_badge:
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-end; padding-top:22px;">
          <span class="badge">⚡ {st.session_state.selected_model.split('/')[-1]}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── EMPTY STATE ──
    if not st.session_state.messages and not st.session_state.pending_query:
        st.markdown("""
        <div class="hero-wrap">
          <span class="badge">✦ Official Knowledge Platform</span>
          <div class="hero-title">Welcome to MIST Student Assistant 👋</div>
          <p class="hero-sub">Ask anything about admissions, fee structure, attendance rules, exam grading, or academic departments.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p style="font-size:10.5px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#475569;margin-bottom:12px;">Frequently Explored Topics</p>', unsafe_allow_html=True)

        starters = [
            ("💳", "Fee Structure",         "Tuition & program fees",             "What fee amounts are published for B.Tech, M.Tech, MBA, and Diploma?"),
            ("🎓", "Admissions & Entry",    "EAMCET, counselling & eligibility",  "How do I apply for B.Tech admission at MIST and what entrance rank is required?"),
            ("📅", "Attendance Rules",       "Minimum attendance & condonation",   "What are the attendance regulations, condonation rules, and semester timeline under M24?"),
            ("📝", "Exam Regulations",       "CIE, SEE, hall tickets & results",   "What examination regulations apply to MIST students under CIE and SEE?"),
            ("🔬", "Departments",            "CSE, AI&ML, ECE, EEE, Mech, Civil", "What academic departments and specialisations are available at MIST?"),
            ("🏛️", "College Profile",       "NAAC A+, autonomous status",         "Where is MIST located and what accreditations does the institution hold?"),
        ]

        rows = [st.columns(3), st.columns(3)]
        for idx, (emoji, title, desc, query) in enumerate(starters):
            row = rows[0 if idx < 3 else 1]
            col = row[idx % 3]
            with col:
                st.markdown(f"""
                <div class="topic-card">
                  <div class="topic-card-icon">{emoji}</div>
                  <div class="topic-card-title">{title}</div>
                  <div class="topic-card-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Ask this →", key=f"start_{idx}", use_container_width=True):
                    set_pending(query); st.rerun()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── CONVERSATION HISTORY ──
    for idx, msg in enumerate(st.session_state.messages):
        avatar = "👤" if msg["role"] == "user" else "🎓"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("sources"):
                sources = msg["sources"]
                with st.expander(f"📚 {len(sources)} verified source(s) cited", expanded=False):
                    for src in sources:
                        st.markdown(f"<span class='src-badge'>📄 {src['source']} · Page {src['page']}</span>", unsafe_allow_html=True)
                        st.markdown(f"<div class='src-excerpt'>{src['snippet']}</div>", unsafe_allow_html=True)
                        for u in src.get("urls", []):
                            st.markdown(f"🔗 [Official Link]({u})")

                    unique_files = list(dict.fromkeys(s["source"] for s in sources))
                    dl_cols = st.columns(min(len(unique_files), 3))
                    for d_idx, fname in enumerate(unique_files[:3]):
                        fpath = config.DOCUMENTS_DIR / fname
                        if fpath.exists():
                            with dl_cols[d_idx]:
                                st.download_button(
                                    label=f"📥 {fname}",
                                    data=fpath.read_bytes(),
                                    file_name=fname,
                                    mime="application/pdf",
                                    key=f"dl_hist_{idx}_{d_idx}",
                                    use_container_width=True,
                                )

                m_col, f_col = st.columns([3, 1])
                with m_col:
                    st.caption(f"⚡ {msg.get('latency',0)}s · {msg.get('model','').split('/')[-1]} · {msg.get('timestamp','')}")
                with f_col:
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("👍", key=f"up_{idx}"):
                            st.session_state.feedback_store[idx] = "like"; st.toast("Thanks for the feedback!", icon="✨")
                    with b2:
                        if st.button("👎", key=f"dn_{idx}"):
                            st.session_state.feedback_store[idx] = "dislike"; st.toast("Noted! We'll improve.", icon="📝")

                followups = msg.get("followups", [])
                if followups:
                    st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#4F46E5;margin:10px 0 6px;">Suggested Follow-ups</p>', unsafe_allow_html=True)
                    fcols = st.columns(len(followups))
                    for fi, fq in enumerate(followups):
                        with fcols[fi]:
                            if st.button(fq[:48]+"…" if len(fq)>48 else fq, key=f"fup_{idx}_{fi}", use_container_width=True):
                                set_pending(fq); st.rerun()

    # ── STREAMING ANSWER ──
    if st.session_state.pending_query:
        q = st.session_state.pending_query
        st.session_state.pending_query = None

        if q not in st.session_state.conversation_history:
            st.session_state.conversation_history.insert(0, q)
            st.session_state.conversation_history = st.session_state.conversation_history[:10]

        st.session_state.messages.append({"role": "user", "content": q, "timestamp": datetime.now().strftime("%H:%M")})
        with st.chat_message("user", avatar="👤"):
            st.markdown(q)

        t0 = time.perf_counter()
        with st.chat_message("assistant", avatar="🎓"):
            # Instant Greeting Fast-Path
            if is_conversational_greeting(q):
                fast_greeting = get_conversational_response(q)
                def stream_greeting():
                    for word in fast_greeting.split(" "):
                        yield word + " "
                        time.sleep(0.015)
                response = st.write_stream(stream_greeting())
                sources = []
                elapsed = round(time.perf_counter() - t0, 2)
            else:
                with st.spinner("Searching official MIST knowledge base…"):
                    prompt, sources, _ = prepare_rag_context(q, top_k=st.session_state.top_k)
                llm = get_cached_llm(model_name=st.session_state.selected_model, temperature=st.session_state.temperature)
                response = st.write_stream(chunk.content for chunk in llm.stream(prompt) if chunk.content)
                elapsed = round(time.perf_counter() - t0, 2)

            if sources:
                with st.expander(f"📚 {len(sources)} verified source(s) cited", expanded=False):
                    for src in sources:
                        st.markdown(f"<span class='src-badge'>📄 {src['source']} · Page {src['page']}</span>", unsafe_allow_html=True)
                        st.markdown(f"<div class='src-excerpt'>{src['snippet']}</div>", unsafe_allow_html=True)
                        for u in src.get("urls", []):
                            st.markdown(f"🔗 [Official Link]({u})")

                    unique_files = list(dict.fromkeys(s["source"] for s in sources))
                    dl_cols = st.columns(min(len(unique_files), 3))
                    for d_idx, fname in enumerate(unique_files[:3]):
                        fpath = config.DOCUMENTS_DIR / fname
                        if fpath.exists():
                            with dl_cols[d_idx]:
                                st.download_button(
                                    label=f"📥 {fname}",
                                    data=fpath.read_bytes(),
                                    file_name=fname,
                                    mime="application/pdf",
                                    key=f"dl_hist_{idx}_{d_idx}",
                                    use_container_width=True,
                                )

            st.caption(f"⚡ {elapsed}s · {st.session_state.selected_model.split('/')[-1]} · {datetime.now().strftime('%H:%M')}")
            followups = get_suggested_followups(q, response or "")

        st.session_state.messages.append({
            "role": "assistant", "content": response or "",
            "sources": sources, "latency": elapsed,
            "model": st.session_state.selected_model,
            "timestamp": datetime.now().strftime("%H:%M"),
            "followups": followups,
        })
        st.session_state.query_analytics.append({
            "query": q[:60],
            "time": datetime.now().strftime("%H:%M:%S"),
            "latency": elapsed,
            "sources_count": len(sources),
            "model": st.session_state.selected_model.split("/")[-1],
            "status": "✅ Success",
        })
        st.rerun()

    # Chat input
    user_q = st.chat_input("Ask MIST AI… e.g. 'What is B.Tech fee?' or 'What are attendance rules?'")
    if user_q:
        set_pending(user_q); st.rerun()


# ─────────────────────────────────────────────
# 7. COLLEGE DASHBOARD
# ─────────────────────────────────────────────
elif st.session_state.page == "College Dashboard":
    st.markdown("""
    <div style="padding:16px 0 20px;">
      <div class="page-eyebrow">MIST · Overview</div>
      <h2 class="page-title">College Dashboard</h2>
      <p class="page-desc">Live telemetry, institution overview, and fast-access academic portals.</p>
    </div>
    """, unsafe_allow_html=True)

    h = check_system_health()
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Knowledge Docs",   h.get("pdf_document_count", 0), "PDF files indexed")
    m2.metric("Vector Chunks",    h.get("chunks_indexed", 0),     "ChromaDB embeddings")
    m3.metric("Inference Engine", "Groq LPU",                     "Sub-second latency")
    m4.metric("Accreditation",    "NAAC A+",                      "Autonomous · JNTUH")

    st.write("")
    with st.container(border=True):
        st.markdown("### 🎨 Design Studio Assets (Figma & Canva)")
        st.caption("Visual assets, layout architecture, and banners created via connected design tools:")
        c_figma, c_canva = st.columns(2)
        with c_figma:
            st.markdown("""
            **🎨 Figma UI Architecture**  
            Multi-workspace design system, token bindings, and component layouts.  
            [👉 Open Figma Design (hALGvS4Oin6XzaE7sFKV9j)](https://www.figma.com/design/hALGvS4Oin6XzaE7sFKV9j)
            """)
        with c_canva:
            st.markdown("""
            **✨ Canva Campus Portal Banner**  
            Executive dark-theme campus hero layout and promotional visual cards.  
            [👉 Open Canva Design (DAHZlnkLxsY)](https://canva.link/wm3n3kn0r5b7khr)
            """)

    st.write("")
    with st.container(border=True):
        st.markdown(f"### 🏛️ {config.COLLEGE_NAME}")
        st.markdown(f"""
- **Established:** {config.COLLEGE_ESTD}
- **Status:** {config.COLLEGE_ACCREDITATION}
- **Location:** {config.COLLEGE_LOCATION}
- **Helpline:** `{config.COLLEGE_PHONE}` | **Email:** `{config.COLLEGE_EMAIL}`
- **Website:** [{config.COLLEGE_WEBSITE}]({config.COLLEGE_WEBSITE})
        """)

    st.write("")
    st.markdown("### 🚀 Academic Quick-Access Portals")
    portals = [
        ("💳","Fee Structure",       "B.Tech, M.Tech, MBA, Diploma fees.",               "What fee structure is published for MIST?"),
        ("📋","Admissions",          "EAMCET and university counselling routes.",          "Explain the MIST admission process."),
        ("⏱️","M24 Regulations",    "4-year B.Tech, 160 credits, 22-week semesters.",    "What are the M24 attendance requirements?"),
        ("📝","Examinations",        "CIE, SEE, syllabus, results, timetables.",          "What examination regulations apply at MIST?"),
        ("🔬","Departments",         "CSE, AI&ML, ECE, EEE, Mechanical, Civil, MBA.",    "What departments are available at MIST?"),
        ("📍","Campus Facilities",   "Labs, digital library, e-classrooms.",              "What campus facilities are described for MIST?"),
    ]
    for row_start in range(0, len(portals), 3):
        cols = st.columns(3)
        for ci, (emo, title, desc, prompt) in enumerate(portals[row_start:row_start+3]):
            with cols[ci]:
                with st.container(border=True):
                    st.markdown(f"#### {emo} {title}")
                    st.caption(desc)
                    if st.button("Ask AI →", key=f"portal_{row_start+ci}", use_container_width=True):
                        set_pending(prompt); st.rerun()


# ─────────────────────────────────────────────
# 8. DOCUMENT VAULT
# ─────────────────────────────────────────────
elif st.session_state.page == "Document Vault":
    st.markdown("""
    <div style="padding:16px 0 20px;">
      <div class="page-eyebrow">MIST · Knowledge Repository</div>
      <h2 class="page-title">Document Vault</h2>
      <p class="page-desc">Upload official PDFs and rebuild the vector knowledge base in one click.</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### 📤 Upload New Document")
            st.caption("Upload fee circulars, syllabus PDFs, exam notifications, or brochures.")
            up = st.file_uploader("Choose PDF", type=["pdf"], key="uploader")
            if up:
                dest = config.DOCUMENTS_DIR / up.name
                if st.button(f"Save '{up.name}'", key="save_doc", type="primary"):
                    dest.write_bytes(up.getbuffer())
                    st.success(f"Saved '{up.name}'. Rebuild index to activate it.")
                    st.rerun()

    with right:
        with st.container(border=True):
            st.markdown("#### ⚡ Rebuild Index")
            st.caption("Recompute embeddings from all vault PDFs.")
            if st.button("Re-index Knowledge Base 🔄", key="reindex", use_container_width=True, type="primary"):
                pb = st.progress(0, "Starting…")
                info = st.empty()
                result = build_vector_database(progress_callback=lambda m,p: pb.progress(int(p*100), m))
                if result["success"]:
                    reload_vectorstore()
                    info.success(f"✅ Indexed {result['documents_count']} docs · {result['chunks_count']} chunks · {result['elapsed_seconds']}s")
                else:
                    info.error(f"❌ {result.get('error')}")

    st.write("")
    st.markdown("### 📄 Vault Contents")
    pdfs = sorted(config.DOCUMENTS_DIR.glob("*.pdf"))
    if not pdfs:
        st.info("No PDFs found in the documents folder.")
    else:
        st.caption(f"{len(pdfs)} document(s) currently indexed.")
        for pdf in pdfs:
            kb = round(pdf.stat().st_size / 1024, 1)
            mtime = datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            with st.container(border=True):
                ci, cb = st.columns([4.5, 1])
                with ci:
                    st.markdown(f"**📄 {pdf.name}**")
                    st.caption(f"{kb} KB · Modified {mtime}")
                with cb:
                    with st.popover("Preview 👁️", use_container_width=True):
                        try:
                            r = PdfReader(str(pdf))
                            st.caption(f"{len(r.pages)} page(s)")
                            st.text_area("Page 1", r.pages[0].extract_text(), height=240)
                        except Exception as e:
                            st.error(str(e))


# ─────────────────────────────────────────────
# 9. SEMANTIC SEARCH LAB
# ─────────────────────────────────────────────
elif st.session_state.page == "Semantic Search Lab":
    st.markdown("""
    <div style="padding:16px 0 20px;">
      <div class="page-eyebrow">RAG Diagnostics · Vector Search</div>
      <h2 class="page-title">Semantic Search Lab</h2>
      <p class="page-desc">Test ChromaDB retrieval directly — inspect similarity scores, source chunks, and portal links.</p>
    </div>
    """, unsafe_allow_html=True)

    cq, ck = st.columns([4, 1])
    with cq:
        sq = st.text_input("Semantic query", placeholder="e.g. condonation rules, intake capacity, tuition fee…", key="sem_q")
    with ck:
        kv = st.number_input("Top K", 1, 8, 4, key="sem_k")

    if sq:
        with st.spinner("Querying vector index…"):
            hits = search_knowledge_base(sq, top_k=int(kv))
        if not hits:
            st.info("No matching chunks found.")
        else:
            st.success(f"{len(hits)} chunks retrieved for **'{sq}'**")
            for hit in hits:
                with st.container(border=True):
                    sc, dc = st.columns([1.4, 5])
                    with sc:
                        score = hit.get("score")
                        if score is not None:
                            st.metric("Score", f"{score:.3f}")
                        else:
                            st.caption("Match")
                    with dc:
                        st.markdown(f"`{hit['source']}` — Page {hit['page']}")
                        st.markdown(f"<div class='src-excerpt'>{hit['content'][:400]}</div>", unsafe_allow_html=True)
                        for u in hit.get("urls", []):
                            st.markdown(f"🔗 [Portal Link]({u})")


# ─────────────────────────────────────────────
# 10. USAGE ANALYTICS
# ─────────────────────────────────────────────
elif st.session_state.page == "Usage Analytics":
    st.markdown("""
    <div style="padding:16px 0 20px;">
      <div class="page-eyebrow">MIST · Telemetry</div>
      <h2 class="page-title">Usage Analytics</h2>
      <p class="page-desc">Session query telemetry, latency distribution, and audit log.</p>
    </div>
    """, unsafe_allow_html=True)

    msgs = st.session_state.messages
    total = len(msgs)
    student_q = sum(1 for m in msgs if m["role"] == "user")
    ai_r = sum(1 for m in msgs if m["role"] == "assistant")
    lats = [m.get("latency", 0) for m in msgs if m["role"] == "assistant" and m.get("latency")]
    avg_lat = round(sum(lats)/len(lats), 2) if lats else 0

    a1,a2,a3,a4 = st.columns(4)
    a1.metric("Total Messages",  total,    "This session")
    a2.metric("Student Queries", student_q,"Processed")
    a3.metric("AI Responses",    ai_r,     "Generated")
    a4.metric("Avg Latency",     f"{avg_lat}s" if avg_lat else "—", "Groq LPU")

    st.write("")
    qs = st.session_state.query_analytics
    if qs:
        st.markdown("### ⚡ Response Latency (seconds)")
        df = pd.DataFrame(qs)
        if "latency" in df.columns and not df.empty:
            st.bar_chart(df.set_index("time")["latency"], height=200, color="#6366F1")
        st.markdown("### 📋 Query Audit Log")
        st.dataframe(df, use_container_width=True, hide_index=True,
            column_config={
                "query":"Query","time":"Time",
                "latency": st.column_config.NumberColumn("Latency (s)", format="%.2f s"),
                "sources_count":"Sources","model":"Model","status":"Status"
            })
    else:
        st.info("No queries this session yet. Ask a question in the AI Assistant to see analytics.")


# ─────────────────────────────────────────────
# 11. SETTINGS & HEALTH
# ─────────────────────────────────────────────
elif st.session_state.page == "Settings & Health":
    st.markdown("""
    <div style="padding:16px 0 20px;">
      <div class="page-eyebrow">Configuration</div>
      <h2 class="page-title">Settings & System Health</h2>
      <p class="page-desc">Configure inference parameters, switch models, and run live diagnostics.</p>
    </div>
    """, unsafe_allow_html=True)

    cl, cr = st.columns([2.2, 1.5])
    with cl:
        with st.container(border=True):
            st.markdown("#### ⚙️ Inference Parameters")
            st.session_state.selected_model = st.selectbox(
                "LLM Model (Groq)",
                config.AVAILABLE_MODELS,
                index=config.AVAILABLE_MODELS.index(st.session_state.selected_model)
                      if st.session_state.selected_model in config.AVAILABLE_MODELS else 0,
                help="Which Groq LPU model generates the answer from retrieved documents.",
            )
            st.session_state.top_k = st.slider(
                "Top-K Retrieved Chunks", 1, 8, st.session_state.top_k,
                help="Higher K = more context but larger prompt.",
            )
            st.session_state.temperature = st.slider(
                "Temperature", 0.0, 1.0, float(st.session_state.temperature), 0.05,
                help="0.0 = deterministic, 1.0 = creative. Keep ≤0.3 for factual college answers.",
            )
            st.write("")
            if st.button("Save Settings", type="primary", use_container_width=True):
                st.toast("Settings applied!", icon="✅")

    with cr:
        with st.container(border=True):
            st.markdown("#### 🩺 Health Diagnostics")
            h = check_system_health()
            status_map = {
                "groq_key_configured":       ("Groq API Key",      True),
                "documents_folder_exists":   ("Docs Folder",       True),
                "pdf_document_count":        ("PDF Count",         None),
                "chroma_dir_exists":         ("Chroma DB",         True),
                "vector_status":             ("Vector Status",     None),
                "chunks_indexed":            ("Chunks Indexed",    None),
                "embedding_model":           ("Embedding Model",   None),
                "default_llm_model":         ("Default LLM",       None),
            }
            for key, (label, is_bool) in status_map.items():
                val = h.get(key, "—")
                if is_bool:
                    icon = "✅" if val else "❌"
                    st.markdown(f"**{label}:** {icon}")
                else:
                    st.markdown(f"**{label}:** `{val}`")

            st.write("")
            if st.button("📡 Ping Groq API", use_container_width=True):
                with st.spinner("Testing…"):
                    try:
                        from rag_pipeline import get_cached_llm as gcl
                        lm = gcl(model_name=st.session_state.selected_model)
                        t0 = time.time()
                        lm.invoke("ping")
                        ms = round((time.time()-t0)*1000, 1)
                        st.success(f"Connected! Latency: {ms}ms")
                    except Exception as e:
                        st.error(f"Failed: {e}")

    st.write("")
    with st.container(border=True):
        st.markdown("#### ℹ️ About MIST Connect AI")
        st.markdown(f"""
**{config.COLLEGE_NAME}** — Official AI-powered FAQ assistant.

| Component | Detail |
|---|---|
| Vector DB | ChromaDB · `{config.CHROMA_DB_DIR}` |
| Embedding | `{config.EMBEDDING_MODEL}` · {config.EMBEDDING_DIMENSIONS}d |
| LLM Provider | Groq LPUs (Sub-second inference) |
| Framework | LangChain · Streamlit |
| Safety | Strict prompt guardrails — no invented data |
        """)
