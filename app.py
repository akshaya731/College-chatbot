import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
from pypdf import PdfReader

# Ensure stdout uses UTF-8 to prevent charmap errors on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import config
from rag_pipeline import (
    get_answer_detailed,
    search_knowledge_base,
    check_system_health,
    reload_vectorstore,
    get_cached_vectorstore,
    extract_urls,
)
from ingest import build_vector_database

# =========================================================
# 1. PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="MIST Connect AI · College Knowledge Desk",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. SESSION STATE MANAGEMENT
# =========================================================
DEFAULT_SESSION_STATE = {
    "messages": [],
    "page": "AI Assistant",
    "selected_model": config.DEFAULT_LLM_MODEL,
    "top_k": config.DEFAULT_TOP_K,
    "temperature": config.DEFAULT_TEMPERATURE,
    "conversation_history": [],
    "query_analytics": [],
    "feedback_store": {},
}

for key, val in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val


# =========================================================
# 3. HIGH-END UI STYLING & THEMING (CSS)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg-page: #070B16;
    --bg-card: #0D1426;
    --bg-card-hover: #131C35;
    --bg-surface: #101930;
    --border-subtle: rgba(135, 156, 225, 0.14);
    --border-accent: rgba(124, 114, 255, 0.35);
    --text-primary: #F4F7FF;
    --text-secondary: #9EAFD0;
    --text-muted: #6C7F9F;
    --brand-primary: #6366F1;
    --brand-secondary: #3B82F6;
    --brand-gradient: linear-gradient(135deg, #6366F1 0%, #3B82F6 100%);
    --card-glow: 0 8px 30px rgba(0, 0, 0, 0.25);
    --accent-emerald: #10B981;
}

/* App Background */
.stApp {
    background: radial-gradient(circle at 80% 0%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 10% 90%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
                var(--bg-page);
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    max-width: 1300px;
    padding-top: 1.2rem;
    padding-bottom: 3.5rem;
}

/* Typography */
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

p, span, label, div {
    color: var(--text-primary);
}

/* Streamlit Native Chrome Cleanup */
#MainMenu, footer, [data-testid="stDeployButton"] {
    visibility: hidden !important;
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
}

/* Custom Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090E1D 0%, #080D1A 100%) !important;
    border-right: 1px solid var(--border-subtle);
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 6px 18px 6px;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 16px;
}

.brand-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: var(--brand-gradient);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
}

.brand-text-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 19px;
    color: #FFFFFF;
    line-height: 1.2;
}

.brand-text-sub {
    font-size: 10px;
    font-weight: 600;
    color: #8BA1C7;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.nav-section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin: 16px 0 8px 8px;
}

/* Sidebar navigation buttons */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--text-secondary) !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 10px !important;
    min-height: 42px !important;
    padding: 0 14px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99, 102, 241, 0.12) !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
    color: #FFFFFF !important;
    transform: translateX(3px);
}

/* Sidebar Info Card */
.sidebar-status-box {
    margin-top: 14px;
    padding: 12px 14px;
    background: rgba(16, 25, 48, 0.7);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    font-size: 11.5px;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-emerald);
    box-shadow: 0 0 8px var(--accent-emerald);
    margin-right: 6px;
}

/* Hero & Header Section */
.hero-wrapper {
    text-align: center;
    max-width: 900px;
    margin: 20px auto 35px auto;
    padding: 10px 20px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 100px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.28);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.6px;
    color: #A5B4FC;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.hero-title {
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 700;
    line-height: 1.18;
    margin-bottom: 12px;
    background: linear-gradient(135deg, #FFFFFF 30%, #CAD8FF 70%, #A5B4FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 15px;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 680px;
    margin: 0 auto;
}

/* Prompt Starter Cards */
.starter-card-box {
    background: linear-gradient(145deg, #0E162B 0%, #0A1020 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 16px;
    height: 100%;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
}

.starter-card-box:hover {
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
    transform: translateY(-2px);
}

.starter-card-title {
    font-weight: 700;
    font-size: 13.5px;
    color: #FFFFFF;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.starter-card-desc {
    font-size: 11.5px;
    color: var(--text-secondary);
    line-height: 1.45;
}

/* Chat Message Styling */
[data-testid="stChatMessage"] {
    background: rgba(13, 20, 38, 0.85) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    padding: 18px 20px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.18);
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(20, 31, 60, 0.85) !important;
    border-color: rgba(99, 102, 241, 0.22) !important;
}

/* Source Citation Accordions */
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 8px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.28);
    font-size: 11px;
    font-weight: 600;
    color: #C7D2FE;
    margin-right: 6px;
    margin-top: 6px;
}

.source-snippet {
    background: #090E1D;
    border-left: 3px solid var(--brand-primary);
    padding: 10px 14px;
    margin-top: 8px;
    border-radius: 0 8px 8px 0;
    font-size: 12px;
    color: #CBD5E1;
    line-height: 1.55;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 18px;
    box-shadow: var(--card-glow);
}

[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

/* Primary Action Buttons */
.primary-btn button {
    background: var(--brand-gradient) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35) !important;
}

.primary-btn button:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.55) !important;
    transform: translateY(-1px);
}

/* Containers with border */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(13, 20, 38, 0.65);
    border-color: var(--border-subtle) !important;
    border-radius: 16px !important;
    padding: 6px;
}

/* Expander custom styling */
.streamlit-expanderHeader {
    background: rgba(16, 25, 48, 0.6) !important;
    border-radius: 10px !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    color: #CAD8FF !important;
}

/* Chat Input Bar Fix */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    background: #0E162B !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. APPLICATION HELPERS
# =========================================================
def handle_user_query(question: str):
    """Processes a user question, stores query telemetry, and renders the result."""
    q = question.strip()
    if not q:
        return

    # Append user question
    st.session_state.messages.append({
        "role": "user",
        "content": q,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })

    # Save to history
    if q not in st.session_state.conversation_history:
        st.session_state.conversation_history.insert(0, q)
        st.session_state.conversation_history = st.session_state.conversation_history[:10]

    # Generate assistant answer
    with st.spinner("Retrieving official college documents & formulating answer..."):
        result = get_answer_detailed(
            question=q,
            model_name=st.session_state.selected_model,
            top_k=st.session_state.top_k,
            temperature=st.session_state.temperature,
        )

    # Store message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "latency": result.get("latency_seconds", 0.0),
        "model": result.get("model_used", st.session_state.selected_model),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "success": result.get("success", True),
    })

    # Store telemetry for analytics
    st.session_state.query_analytics.append({
        "query": q,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latency": result.get("latency_seconds", 0.0),
        "sources_count": len(result.get("sources", [])),
        "model": result.get("model_used", st.session_state.selected_model),
        "status": "Success" if result.get("success", True) else "Failed",
    })


def reset_conversation():
    """Clears the chat messages and starts a fresh conversation."""
    st.session_state.messages = []
    st.session_state.page = "AI Assistant"


# =========================================================
# 5. SIDEBAR NAVIGATION & CONTROLS
# =========================================================
with st.sidebar:
    # Brand Header
    st.markdown(f"""
    <div class="sidebar-brand">
      <div class="brand-icon">🎓</div>
      <div>
        <div class="brand-text-title">{config.COLLEGE_SHORT_NAME} Connect</div>
        <div class="brand-text-sub">Campus AI Assistant</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Menu
    st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)
    nav_tabs = [
        ("💬", "AI Assistant"),
        ("📊", "College Dashboard"),
        ("📁", "Document Vault"),
        ("🔍", "Semantic Search Lab"),
        ("📈", "Usage Analytics"),
        ("⚙️", "Settings & Health"),
    ]

    for icon, tab_name in nav_tabs:
        is_active = st.session_state.page == tab_name
        label_text = f"{icon}  {tab_name}"
        if st.button(label_text, key=f"nav_{tab_name}", use_container_width=True):
            st.session_state.page = tab_name
            st.rerun()

    # Conversation Management
    st.markdown('<div class="nav-section-label">Session Controls</div>', unsafe_allow_html=True)
    col_new, col_exp = st.columns(2)
    with col_new:
        if st.button("＋ New Chat", key="btn_new_chat", use_container_width=True):
            reset_conversation()
            st.rerun()
    with col_exp:
        # Export chat as markdown
        if st.session_state.messages:
            transcript = f"# MIST Connect AI - Conversation Log\n\n*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            for m in st.session_state.messages:
                role = "Student" if m["role"] == "user" else "MIST AI Assistant"
                transcript += f"### {role} ({m.get('timestamp', '')}):\n{m['content']}\n\n"
            st.download_button(
                "📥 Export",
                data=transcript,
                file_name=f"MIST_Chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.button("📥 Export", disabled=True, use_container_width=True)

    # Recent queries in sidebar
    if st.session_state.conversation_history:
        st.markdown('<div class="nav-section-label">Recent Questions</div>', unsafe_allow_html=True)
        for h_idx, h_query in enumerate(st.session_state.conversation_history[:5]):
            short_q = h_query[:32] + ("..." if len(h_query) > 32 else "")
            if st.button(f"◦ {short_q}", key=f"history_{h_idx}", use_container_width=True):
                st.session_state.page = "AI Assistant"
                handle_user_query(h_query)
                st.rerun()

    # System Status & Quick Contact Card
    st.markdown('<div class="nav-section-label">Campus Helpdesk</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-status-box">
      <div><span class="status-dot"></span><strong>System Status: Active</strong></div>
      <div style="color: #8BA1C7; margin-top: 6px; font-size: 11px;">
        • <strong>Model:</strong> {st.session_state.selected_model.split('/')[-1]}<br>
        • <strong>Docs:</strong> 6 PDFs Indexed<br>
        • <strong>Vector DB:</strong> ChromaDB (Ready)
      </div>
      <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(135,156,225,0.12); font-size: 10.5px; color: #7E93B8;">
        📞 {config.COLLEGE_PHONE}<br>
        ✉️ {config.COLLEGE_EMAIL}<br>
        🌐 <a href="{config.COLLEGE_WEBSITE}" target="_blank" style="color: #A5B4FC;">mist.edu.in</a>
      </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 6. VIEW: AI ASSISTANT
# =========================================================
if st.session_state.page == "AI Assistant":
    # Header bar
    head_left, head_right = st.columns([4, 1.2])
    with head_left:
        st.markdown(f"""
        <div style="margin-bottom: 8px;">
          <div style="font-size: 11px; font-weight: 700; color: #818CF8; letter-spacing: 1.5px; text-transform: uppercase;">
            {config.COLLEGE_SHORT_NAME} · OFFICIAL FAQ & ADMISSIONS DESK
          </div>
          <h2 style="margin: 4px 0; font-size: 26px;">CampusConnect AI Assistant</h2>
          <div style="font-size: 13px; color: #94A3B8;">
            Direct answers verified against official {config.COLLEGE_SHORT_NAME} regulations, fees, and syllabus documents.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with head_right:
        st.markdown(f"""
        <div style="text-align: right; margin-top: 8px;">
          <span class="hero-badge">● MODEL: {st.session_state.selected_model.split('/')[-1]}</span>
        </div>
        """, unsafe_allow_html=True)

    # Empty State: Welcome Banner & Topic Starter Cards
    if not st.session_state.messages:
        st.markdown(f"""
        <div class="hero-wrapper">
          <div class="hero-badge">✦ OFFICIAL CAMPUS INTELLIGENCE PLATFORM</div>
          <div class="hero-title">Welcome to MIST Student Assistant 👋</div>
          <p class="hero-subtitle">
            Ask any question regarding admissions, fee structure, semester attendance, autonomous exam rules, or academic departments.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="font-size: 11px; font-weight: 700; color: #8BA1C7; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 12px;">Frequently Explored Topics</div>', unsafe_allow_html=True)

        starter_prompts = [
            ("💳", "Fee Structure", "What fee amounts are published for B.Tech, M.Tech, MBA, and Diploma?", "Tuition & program fees"),
            ("🎓", "Admissions & Entry", "How do I apply for B.Tech admission at MIST and what is the entrance rank requirement?", "EAMCET, counselling & eligibility"),
            ("📅", "Attendance Rules", "What are the attendance regulations, condonation rules, and semester timeline under M24?", "Minimum attendance & condonation"),
            ("📝", "Exam Regulations", "What examination regulations, CIE, and SEE rules apply to students at MIST?", "Evaluation, hall tickets & results"),
            ("🏛️", "Departments & Labs", "What academic departments and specialisations are available at MIST?", "CSE, AI&ML, ECE, EEE, Mech, Civil"),
            ("📍", "College Profile", "Where is MIST located and what accreditations does the institution hold?", "NAAC A+, autonomous status & campus"),
        ]

        c_rows = [st.columns(3), st.columns(3)]
        for idx, (emoji, title, query, desc) in enumerate(starter_prompts):
            row_idx = 0 if idx < 3 else 1
            col_target = c_rows[row_idx][idx % 3]
            with col_target:
                st.markdown(f"""
                <div class="starter-card-box">
                  <div class="starter-card-title">{emoji} {title}</div>
                  <div class="starter-card-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Ask Assistant →", key=f"starter_btn_{idx}", use_container_width=True):
                    handle_user_query(query)
                    st.rerun()

    # Active Conversation Display
    else:
        for idx, msg in enumerate(st.session_state.messages):
            avatar = "👤" if msg["role"] == "user" else "🎓"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

                # Render sources and citations for assistant messages
                if msg["role"] == "assistant" and msg.get("sources"):
                    sources = msg["sources"]
                    with st.expander(f"📚 Verified Sources & References ({len(sources)} documents)", expanded=False):
                        for s_idx, src in enumerate(sources):
                            st.markdown(
                                f"<span class='source-pill'>📄 <strong>{src['source']}</strong> · Page {src['page']}</span>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"<div class='source-snippet'>\"{src['snippet']}\"</div>", unsafe_allow_html=True)
                            if src.get("urls"):
                                for u in src["urls"]:
                                    st.markdown(f"🔗 [Official Portal Link]({u})")

                    # Metadata badge & Action tools
                    col_meta, col_actions = st.columns([3, 1])
                    with col_meta:
                        st.caption(f"⚡ Latency: {msg.get('latency', 0.0)}s · Engine: {msg.get('model', 'Groq LPU')} · {msg.get('timestamp', '')}")
                    with col_actions:
                        f_col1, f_col2 = st.columns(2)
                        with f_col1:
                            if st.button("👍", key=f"like_{idx}", help="Helpful answer"):
                                st.session_state.feedback_store[idx] = "like"
                                st.toast("Thank you for your feedback!", icon="✨")
                        with f_col2:
                            if st.button("👎", key=f"dislike_{idx}", help="Needs improvement"):
                                st.session_state.feedback_store[idx] = "dislike"
                                st.toast("Feedback noted. We will refine document citations.", icon="📝")

    # Bottom Chat Input Bar (Native, fluid, auto-clearing, Enter-friendly)
    user_input = st.chat_input("Ask MIST AI… e.g., 'What are the attendance requirements?' or 'What is B.Tech fee?'")
    if user_input:
        handle_user_query(user_input)
        st.rerun()


# =========================================================
# 7. VIEW: COLLEGE DASHBOARD
# =========================================================
elif st.session_state.page == "College Dashboard":
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
      <div style="font-size: 11px; font-weight: 700; color: #818CF8; letter-spacing: 1.5px; text-transform: uppercase;">
        {config.COLLEGE_SHORT_NAME} · INSTITUTION OVERVIEW
      </div>
      <h2 style="margin: 4px 0 6px 0;">College Operations Dashboard</h2>
      <p style="color: #94A3B8; font-size: 14px; margin: 0;">
        Real-time telemetry of indexed knowledge assets, autonomous regulation details, and fast inquiry cards.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Health check metrics
    health = check_system_health()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Indexed Documents", health.get("pdf_document_count", 0), "PDF Files")
    m2.metric("Knowledge Chunks", health.get("chunks_indexed", 0), "ChromaDB Vectors")
    m3.metric("LLM Response Engine", "Groq LPU", "High-speed Inference")
    m4.metric("Accreditation", "NAAC A+", "Autonomous · JNTUH")

    st.write("")

    # College Profile Overview Card
    with st.container(border=True):
        st.markdown(f"### 🏛️ {config.COLLEGE_NAME}")
        st.markdown(f"""
        - **Established:** {config.COLLEGE_ESTD}
        - **Status & Recognition:** {config.COLLEGE_ACCREDITATION}
        - **Campus Location:** {config.COLLEGE_LOCATION}
        - **Admissions Helpline:** `{config.COLLEGE_PHONE}` | **Email:** `{config.COLLEGE_EMAIL}`
        - **Official Website:** [{config.COLLEGE_WEBSITE}]({config.COLLEGE_WEBSITE})
        """)

    st.write("")
    st.markdown("### 🎓 Core Academic Portals & Quick Questions")

    portal_cards = [
        ("💳", "Published Fee Structure", "Official fee schedule for B.Tech, M.Tech, MBA, and Diploma.", "What fee structure is published for MIST?"),
        ("📋", "Admissions & Entry Procedure", "Admission paths via Telangana EAMCET and university approval.", "Explain the MIST admission process and eligibility."),
        ("⏱️", "M24 Academic Regulations", "4-year B.Tech program structure, 160 degree credits, and 22-week semesters.", "What are the attendance requirements and regulations under M24?"),
        ("📝", "Examinations & Grading", "Autonomous CIE (internal) and SEE (semester end) exams and syllabus.", "What examination information is available for MIST students?"),
        ("🔬", "Departments & Laboratories", "CSE (AI&ML, Data Science), ECE, EEE, Mechanical, and Civil Engineering.", "What academic departments and specialisations exist at MIST?"),
        ("📍", "Campus Facilities", "Computer centre, digital library, communication labs, and e-classrooms.", "What campus facilities and infrastructure are described for MIST?"),
    ]

    p_cols = st.columns(3)
    for idx, (emoji, title, desc, prompt) in enumerate(portal_cards):
        with p_cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"#### {emoji} {title}")
                st.caption(desc)
                if st.button("Ask Assistant →", key=f"dash_portal_{idx}", use_container_width=True):
                    st.session_state.page = "AI Assistant"
                    handle_user_query(prompt)
                    st.rerun()


# =========================================================
# 8. VIEW: DOCUMENT VAULT & INGESTION
# =========================================================
elif st.session_state.page == "Document Vault":
    st.markdown("""
    <div style="margin-bottom: 20px;">
      <div style="font-size: 11px; font-weight: 700; color: #818CF8; letter-spacing: 1.5px; text-transform: uppercase;">
        MIST · KNOWLEDGE REPOSITORY
      </div>
      <h2 style="margin: 4px 0 6px 0;">Document Vault & Vector Ingestion</h2>
      <p style="color: #94A3B8; font-size: 14px; margin: 0;">
        Manage official PDFs, upload updated circulars, and rebuild the vector database in one click.
      </p>
    </div>
    """, unsafe_allow_html=True)

    action_left, action_right = st.columns([2, 1])

    with action_left:
        with st.container(border=True):
            st.markdown("### 📤 Upload New College Document")
            st.caption("Upload official brochures, exam circulars, syllabus PDFs, or fee notifications.")
            uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")
            if uploaded_file is not None:
                save_dest = config.DOCUMENTS_DIR / uploaded_file.name
                if st.button(f"Save '{uploaded_file.name}' to Vault", key="save_uploaded_doc", type="primary"):
                    with open(save_dest, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"File '{uploaded_file.name}' successfully saved! Click 'Rebuild Vector Index' to index it.")
                    st.rerun()

    with action_right:
        with st.container(border=True):
            st.markdown("### ⚡ Re-index Knowledge Base")
            st.caption("Process all PDFs into text chunks and recompute vector embeddings.")
            if st.button("Rebuild Vector Index 🔄", key="btn_reindex_now", use_container_width=True, type="primary"):
                progress_bar = st.progress(0, text="Initializing ingestion...")
                status_text = st.empty()

                def update_prog(msg: str, pct: float):
                    progress_bar.progress(int(pct * 100), text=msg)

                result = build_vector_database(progress_callback=update_prog)

                if result["success"]:
                    reload_vectorstore()
                    status_text.success(f"✅ Success! Indexed {result['documents_count']} documents into {result['chunks_count']} chunks in {result['elapsed_seconds']}s.")
                else:
                    status_text.error(f"❌ Error: {result.get('error')}")

    st.write("")
    st.markdown("### 📄 Documents Currently Indexed in Vault")

    pdf_files = sorted(list(config.DOCUMENTS_DIR.glob("*.pdf")))
    if not pdf_files:
        st.warning("No PDF documents found in the documents folder.")
    else:
        st.caption(f"Showing {len(pdf_files)} PDF knowledge documents.")
        for pdf_path in pdf_files:
            file_size_kb = round(pdf_path.stat().st_size / 1024, 1)
            modified_time = datetime.fromtimestamp(pdf_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

            with st.container(border=True):
                col_info, col_btn = st.columns([4, 1.2])
                with col_info:
                    st.markdown(f"**📄 {pdf_path.name}**")
                    st.caption(f"Size: {file_size_kb} KB · Last modified: {modified_time} · Path: `{pdf_path.name}`")
                with col_btn:
                    with st.popover("Preview Content 👁️", use_container_width=True):
                        st.markdown(f"**Document Preview: {pdf_path.name}**")
                        try:
                            reader = PdfReader(str(pdf_path))
                            total_pages = len(reader.pages)
                            st.caption(f"Total Pages: {total_pages}")
                            page_text = reader.pages[0].extract_text()
                            st.text_area("Page 1 Content", page_text, height=260)
                        except Exception as e:
                            st.error(f"Could not extract preview: {e}")


# =========================================================
# 9. VIEW: SEMANTIC SEARCH LAB
# =========================================================
elif st.session_state.page == "Semantic Search Lab":
    st.markdown("""
    <div style="margin-bottom: 20px;">
      <div style="font-size: 11px; font-weight: 700; color: #818CF8; letter-spacing: 1.5px; text-transform: uppercase;">
        RAG DIAGNOSTICS · VECTOR EMBEDDINGS
      </div>
      <h2 style="margin: 4px 0 6px 0;">Semantic Vector Search Playground</h2>
      <p style="color: #94A3B8; font-size: 14px; margin: 0;">
        Directly test semantic similarity retrieval against ChromaDB without triggering the LLM generation step.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_q, col_k = st.columns([4, 1])
    with col_q:
        search_query = st.text_input(
            "Enter semantic search test query",
            placeholder="e.g. condonation percentage, tuition fee, entrance rank, intake capacity...",
            key="semantic_test_input",
        )
    with col_k:
        k_val = st.number_input("Top K Chunks", min_value=1, max_value=8, value=4, key="semantic_k")

    if search_query:
        with st.spinner("Searching vector index..."):
            matches = search_knowledge_base(search_query, top_k=int(k_val))

        if not matches:
            st.info("No matching chunks found in the vector database.")
        else:
            st.success(f"Found {len(matches)} relevant chunks for query: **'{search_query}'**")
            for idx, match in enumerate(matches):
                with st.container(border=True):
                    sc_col, doc_col = st.columns([1.5, 4])
                    with sc_col:
                        score_display = match.get("score")
                        if score_display is not None:
                            st.metric("Similarity Score", f"{score_display:.3f}")
                        else:
                            st.caption("Relevance Match")
                    with doc_col:
                        st.markdown(f"**Source Document:** `{match['source']}` (Page {match['page']})")
                        st.markdown(f"<div class='source-snippet'>{match['content']}</div>", unsafe_allow_html=True)
                        if match.get("urls"):
                            st.caption(f"Detected links: {', '.join(match['urls'])}")


# =========================================================
# 10. VIEW: USAGE ANALYTICS
# =========================================================
elif st.session_state.page == "Usage Analytics":
    st.markdown("""
    <div style="margin-bottom: 20px;">
      <div style="font-size: 11px; font-weight: 700; color: #818CF8; letter-spacing: 1.5px; text-transform: uppercase;">
        MIST · TELEMETRY
      </div>
      <h2 style="margin: 4px 0 6px 0;">Chat & Query Analytics</h2>
      <p style="color: #94A3B8; font-size: 14px; margin: 0;">
        Monitor student inquiry activity, average latency, and retrieved sources per query.
      </p>
    </div>
    """, unsafe_allow_html=True)

    total_msgs = len(st.session_state.messages)
    student_q = sum(1 for m in st.session_state.messages if m["role"] == "user")
    ai_resp = sum(1 for m in st.session_state.messages if m["role"] == "assistant")

    latencies = [m.get("latency", 0.0) for m in st.session_state.messages if m["role"] == "assistant" and m.get("latency")]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Messages", total_msgs, "Current Session")
    a2.metric("Student Questions", student_q, "Queries processed")
    a3.metric("AI Responses", ai_resp, "Generated")
    a4.metric("Avg Response Speed", f"{avg_latency}s" if avg_latency else "Sub-sec", "Groq LPU Latency")

    st.write("")
    st.markdown("### 📋 Recent Query Audit Log")

    if not st.session_state.query_analytics:
        st.info("No queries have been submitted in this session yet. Ask a question in the AI Assistant tab to see telemetry.")
    else:
        st.dataframe(
            st.session_state.query_analytics,
            column_config={
                "query": "Student Query",
                "time": "Timestamp",
                "latency": st.column_config.NumberColumn("Latency (s)", format="%.2f s"),
                "sources_count": "Sources Retrieved",
                "model": "Model Used",
                "status": "Inference Status",
            },
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# 11. VIEW: SETTINGS & SYSTEM HEALTH
# =========================================================
elif st.session_state.page == "Settings & Health":
    st.markdown("""
    <div style="margin-bottom: 20px;">
      <div style="font-size: 11px; font-weight: 700; color: #818CF8; letter-spacing: 1.5px; text-transform: uppercase;">
        APPLICATION CONFIGURATION
      </div>
      <h2 style="margin: 4px 0 6px 0;">Settings & System Health</h2>
      <p style="color: #94A3B8; font-size: 14px; margin: 0;">
        Adjust retrieval parameters, choose LLM models, and perform live connectivity checks.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_conf, col_health = st.columns([2, 1.3])

    with col_conf:
        with st.container(border=True):
            st.markdown("### ⚙️ Inference & Retrieval Parameters")

            # Model selector
            st.session_state.selected_model = st.selectbox(
                "Active Groq LLM Model",
                options=config.AVAILABLE_MODELS,
                index=config.AVAILABLE_MODELS.index(st.session_state.selected_model) if st.session_state.selected_model in config.AVAILABLE_MODELS else 0,
                help="Select which high-speed model generates answers from retrieved documents.",
            )

            # Top-K documents
            st.session_state.top_k = st.slider(
                "Top-K Document Chunks to Retrieve",
                min_value=1,
                max_value=8,
                value=st.session_state.top_k,
                help="Higher values provide more context to the LLM but slightly increase prompt size.",
            )

            # Temperature
            st.session_state.temperature = st.slider(
                "LLM Temperature (Creativity vs Determinism)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.temperature),
                step=0.05,
                help="Lower values (0.0-0.2) ensure factual adherence to college documents.",
            )

            st.write("")
            if st.button("Save & Apply Settings", type="primary", use_container_width=True):
                st.toast("Settings saved successfully!", icon="✅")

    with col_health:
        with st.container(border=True):
            st.markdown("### 🩺 Diagnostic Health Check")
            health_report = check_system_health()

            for key, val in health_report.items():
                if key == "all_healthy":
                    continue
                display_label = key.replace("_", " ").title()
                if isinstance(val, bool):
                    status_emoji = "✅" if val else "❌"
                    st.markdown(f"**{display_label}:** {status_emoji}")
                else:
                    st.markdown(f"**{display_label}:** `{val}`")

            st.write("")
            if st.button("Ping Groq API 📡", use_container_width=True):
                with st.spinner("Testing Groq connection..."):
                    try:
                        from rag_pipeline import get_cached_llm
                        test_llm = get_cached_llm(model_name=st.session_state.selected_model)
                        t0 = time.time()
                        test_resp = test_llm.invoke("Ping")
                        p_time = round((time.time() - t0) * 1000, 1)
                        st.success(f"Groq API Connected! Latency: {p_time}ms")
                    except Exception as ping_err:
                        st.error(f"Ping failed: {ping_err}")

    st.write("")
    with st.container(border=True):
        st.markdown("### 🎓 About MIST College Chatbot")
        st.markdown(f"""
        **CampusConnect AI** is an intelligent retrieval-augmented generation (RAG) assistant created for **{config.COLLEGE_NAME}**.
        - **Vector Index:** ChromaDB persistent store with `{config.EMBEDDING_MODEL}`
        - **Embedding Dimensions:** {config.EMBEDDING_DIMENSIONS} dimensions (normalized cosine similarity)
        - **Frameworks:** LangChain, Streamlit, Groq LPUs, PyPDF
        - **Safety & Verification:** Strict prompt guardrails preventing invented fees, dates, or regulations.
        """)
