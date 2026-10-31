import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
from pypdf import PdfReader

# Reconfigure Windows stdout/stderr to UTF-8
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
)
from ingest import build_vector_database

# =========================================================
# 1. PAGE SETUP & METADATA
# =========================================================
st.set_page_config(
    page_title=f"{config.COLLEGE_SHORT_NAME} Connect AI · Enterprise Campus Assistant",
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
    "pending_query": None,
}

for key, val in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val


# =========================================================
# 3. PRODUCTION-GRADE DESIGN SYSTEM & STYLING (CSS)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-page: #070B16;
    --bg-card: #0D1426;
    --bg-card-hover: #121B33;
    --bg-surface: #101930;
    --border-subtle: rgba(135, 156, 225, 0.12);
    --border-accent: rgba(99, 102, 241, 0.35);
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --brand-primary: #6366F1;
    --brand-secondary: #3B82F6;
    --brand-gradient: linear-gradient(135deg, #6366F1 0%, #3B82F6 100%);
    --accent-emerald: #10B981;
    --accent-amber: #F59E0B;
}

/* Page background with subtle ambient radial gradients */
.stApp {
    background:
        radial-gradient(circle at 82% 0%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
        radial-gradient(circle at 10% 90%, rgba(59, 130, 246, 0.08) 0%, transparent 45%),
        var(--bg-page);
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    max-width: 1320px;
    padding-top: 1rem;
    padding-bottom: 3.5rem;
}

/* Typography */
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.025em;
}

p, span, label, div {
    color: var(--text-primary);
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Streamlit Header / Chrome Cleanup */
#MainMenu, footer, [data-testid="stDeployButton"] {
    visibility: hidden !important;
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
}

/* Custom Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090E1D 0%, #070B16 100%) !important;
    border-right: 1px solid var(--border-subtle);
}

.sidebar-brand-box {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 8px 18px 8px;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 16px;
}

.brand-badge-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: var(--brand-gradient);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
}

.brand-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 19px;
    color: #FFFFFF;
    line-height: 1.2;
}

.brand-subtitle {
    font-size: 9.5px;
    font-weight: 700;
    color: #818CF8;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.nav-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.4px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin: 16px 0 8px 8px;
}

/* Active Nav Button Styling */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--text-secondary) !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 11px !important;
    min-height: 42px !important;
    padding: 0 14px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99, 102, 241, 0.12) !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
    color: #FFFFFF !important;
    transform: translateX(3px);
}

/* Primary Active Button in Sidebar */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.24) 0%, rgba(59, 130, 246, 0.16) 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.45) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
}

/* Hero Section */
.hero-container {
    text-align: center;
    max-width: 920px;
    margin: 15px auto 30px auto;
    padding: 10px 20px;
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 14px;
    border-radius: 100px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.6px;
    color: #A5B4FC;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.hero-header-text {
    font-size: clamp(30px, 4.2vw, 46px);
    font-weight: 700;
    line-height: 1.15;
    margin-bottom: 12px;
    background: linear-gradient(135deg, #FFFFFF 20%, #D8E4FF 65%, #A5B4FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub-text {
    font-size: 14.5px;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 680px;
    margin: 0 auto;
}

/* Topic Starter Cards */
.starter-box {
    background: linear-gradient(145deg, #0D152A 0%, #080D1A 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 15px;
    height: 100%;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.2);
}

.starter-box:hover {
    border-color: rgba(99, 102, 241, 0.45);
    box-shadow: 0 8px 26px rgba(99, 102, 241, 0.2);
    transform: translateY(-2px);
}

.starter-title {
    font-weight: 700;
    font-size: 13.5px;
    color: #FFFFFF;
    margin-bottom: 6px;
}

.starter-desc {
    font-size: 11.5px;
    color: var(--text-secondary);
    line-height: 1.45;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: rgba(13, 20, 38, 0.88) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    padding: 18px 22px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.22);
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(20, 31, 60, 0.88) !important;
    border-color: rgba(99, 102, 241, 0.26) !important;
}

/* Source Citation Accordions & Badges */
.source-pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 8px;
    background: rgba(99, 102, 241, 0.16);
    border: 1px solid rgba(99, 102, 241, 0.32);
    font-size: 11.5px;
    font-weight: 600;
    color: #C7D2FE;
    margin-right: 6px;
    margin-top: 6px;
}

.source-excerpt {
    background: #090E1D;
    border-left: 3px solid var(--brand-primary);
    padding: 10px 14px;
    margin-top: 8px;
    border-radius: 0 8px 8px 0;
    font-size: 12px;
    color: #CBD5E1;
    line-height: 1.55;
}

/* Follow-up Question Chips */
.followup-box {
    margin-top: 14px;
    padding: 10px 14px;
    background: rgba(16, 25, 48, 0.5);
    border: 1px dashed rgba(135, 156, 225, 0.2);
    border-radius: 12px;
}

.followup-label {
    font-size: 10.5px;
    font-weight: 700;
    color: #818CF8;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Metrics Cards */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

[data-testid="stMetricLabel"] {
    font-size: 11.5px !important;
    color: var(--text-secondary) !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    background: #0E162B !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. DISPATCH USER QUESTION
# =========================================================
def set_pending_query(query: str):
    """Sets a query to be executed on next rerun."""
    st.session_state.pending_query = query
    st.session_state.page = "AI Assistant"


# =========================================================
# 5. SIDEBAR NAVIGATION & CONTROL DESK
# =========================================================
with st.sidebar:
    # College Branding Header
    st.markdown(f"""
    <div class="sidebar-brand-box">
      <div class="brand-badge-icon">🎓</div>
      <div>
        <div class="brand-title">{config.COLLEGE_SHORT_NAME} Connect</div>
        <div class="brand-subtitle">Campus AI Intelligence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Workspaces Navigation
    st.markdown('<div class="nav-label">Workspaces</div>', unsafe_allow_html=True)
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
        button_type = "primary" if is_active else "secondary"
        if st.button(label_text, key=f"nav_{tab_name}", type=button_type, use_container_width=True):
            st.session_state.page = tab_name
            st.rerun()

    # Session Controls
    st.markdown('<div class="nav-label">Session Actions</div>', unsafe_allow_html=True)
    col_new, col_exp = st.columns(2)
    with col_new:
        if st.button("＋ New Chat", key="btn_new_chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.session_state.page = "AI Assistant"
            st.rerun()
    with col_exp:
        if st.session_state.messages:
            transcript = f"# MIST Connect AI - Conversation Log\n*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
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

    # Conversation History
    if st.session_state.conversation_history:
        st.markdown('<div class="nav-label">Recent Inquiries</div>', unsafe_allow_html=True)
        for h_idx, h_query in enumerate(st.session_state.conversation_history[:5]):
            short_q = h_query[:30] + ("..." if len(h_query) > 30 else "")
            if st.button(f"◦ {short_q}", key=f"hist_btn_{h_idx}", use_container_width=True):
                set_pending_query(h_query)
                st.rerun()

    # Campus Helpdesk Card
    st.markdown('<div class="nav-label">Official Helpdesk</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: rgba(16,25,48,0.7); border: 1px solid rgba(135,156,225,0.14); border-radius: 12px; padding: 12px; font-size: 11px;">
      <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 8px;">
        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10B981; box-shadow:0 0 8px #10B981;"></span>
        <strong style="color: #FFFFFF;">System Active (Sub-sec)</strong>
      </div>
      <div style="color: #94A3B8; line-height: 1.7;">
        • <strong>Engine:</strong> {st.session_state.selected_model.split('/')[-1]}<br>
        • <strong>Docs:</strong> 6 Official PDFs<br>
        • <strong>Store:</strong> ChromaDB (Cached)
      </div>
      <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(135,156,225,0.12); color: #818CF8;">
        📞 {config.COLLEGE_PHONE}<br>
        ✉️ {config.COLLEGE_EMAIL}<br>
        🌐 <a href="{config.COLLEGE_WEBSITE}" target="_blank" style="color: #A5B4FC; text-decoration: underline;">mist.edu.in</a>
      </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 6. WORKSPACE: AI ASSISTANT (STREAMING & CITATIONS)
# =========================================================
if st.session_state.page == "AI Assistant":
    # Top Banner
    t_left, t_right = st.columns([4, 1.2])
    with t_left:
        st.markdown(f"""
        <div style="margin-bottom: 12px;">
          <div style="font-size: 10.5px; font-weight: 700; color: #818CF8; letter-spacing: 1.5px; text-transform: uppercase;">
            {config.COLLEGE_SHORT_NAME} · ACADEMIC & ADMISSIONS INTELLIGENCE DESK
          </div>
          <h2 style="margin: 3px 0 5px 0; font-size: 26px;">CampusConnect Virtual Advisor</h2>
          <div style="font-size: 13px; color: #94A3B8;">
            Instant answers grounded in official M24/R22 regulations, published fee circulars, and departmental databases.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with t_right:
        st.markdown(f"""
        <div style="text-align: right; margin-top: 8px;">
          <span class="hero-pill">⚡ MODEL: {st.session_state.selected_model.split('/')[-1]}</span>
        </div>
        """, unsafe_allow_html=True)

    # Landing State: Hero Greeting & Interactive Topic Starters
    if not st.session_state.messages and not st.session_state.pending_query:
        st.markdown(f"""
        <div class="hero-container">
          <div class="hero-pill">✦ OFFICIAL COLLEGE KNOWLEDGE PLATFORM</div>
          <div class="hero-header-text">Welcome to MIST Student Assistant 👋</div>
          <p class="hero-sub-text">
            Ask any question regarding admissions eligibility, course fees, attendance rules, autonomous examination grading, or academic departments.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="font-size: 11px; font-weight: 700; color: #8BA1C7; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 12px;">Frequently Explored Topics</div>', unsafe_allow_html=True)

        starter_prompts = [
            ("💳", "Published Fee Structure", "What fee amounts are published for B.Tech, M.Tech, MBA, and Diploma?", "Tuition & program fees"),
            ("🎓", "Admissions & Entry", "How do I apply for B.Tech admission at MIST and what is the entrance rank requirement?", "EAMCET, counselling & eligibility"),
            ("📅", "Attendance Regulations", "What are the attendance regulations, condonation rules, and semester timeline under M24?", "Minimum attendance & condonation"),
            ("📝", "Autonomous Exam Rules", "What examination regulations, CIE, and SEE rules apply to students at MIST?", "Evaluation, hall tickets & results"),
            ("🔬", "Engineering Departments", "What academic departments and specialisations are available at MIST?", "CSE, AI&ML, ECE, EEE, Mech, Civil"),
            ("🏛️", "Campus & Profile", "Where is MIST located and what accreditations does the institution hold?", "NAAC A+, autonomous status & campus"),
        ]

        row_cols = [st.columns(3), st.columns(3)]
        for idx, (emoji, title, query, desc) in enumerate(starter_prompts):
            target_col = row_cols[0 if idx < 3 else 1][idx % 3]
            with target_col:
                st.markdown(f"""
                <div class="starter-box">
                  <div class="starter-title">{emoji} {title}</div>
                  <div class="starter-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Explore Topic →", key=f"start_btn_{idx}", use_container_width=True):
                    set_pending_query(query)
                    st.rerun()

    # Active Conversation History
    for idx, msg in enumerate(st.session_state.messages):
        avatar = "👤" if msg["role"] == "user" else "🎓"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            # Render Verified Citations for Assistant Answers
            if msg["role"] == "assistant" and msg.get("sources"):
                sources = msg["sources"]
                with st.expander(f"📚 Verified Sources & References ({len(sources)} citations)", expanded=False):
                    for s_idx, src in enumerate(sources):
                        st.markdown(
                            f"<span class='source-pill-badge'>📄 <strong>{src['source']}</strong> · Page {src['page']}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"<div class='source-excerpt'>\"{src['snippet']}\"</div>", unsafe_allow_html=True)
                        if src.get("urls"):
                            for u in src["urls"]:
                                st.markdown(f"🔗 [Official Portal Link]({u})")

                # Telemetry & Feedback
                c_meta, c_actions = st.columns([3, 1])
                with c_meta:
                    st.caption(f"⚡ Latency: {msg.get('latency', 0.0)}s · Engine: {msg.get('model', 'Groq LPU')} · {msg.get('timestamp', '')}")
                with c_actions:
                    f_col1, f_col2 = st.columns(2)
                    with f_col1:
                        if st.button("👍", key=f"thumb_up_{idx}", help="Helpful"):
                            st.session_state.feedback_store[idx] = "like"
                            st.toast("Feedback recorded! Thank you.", icon="✨")
                    with f_col2:
                        if st.button("👎", key=f"thumb_down_{idx}", help="Needs improvement"):
                            st.session_state.feedback_store[idx] = "dislike"
                            st.toast("Feedback noted. We will refine document citations.", icon="📝")

                # Suggested Follow-Up Questions
                followups = msg.get("followups", [])
                if followups:
                    st.markdown("""
                    <div class="followup-box">
                      <div class="followup-label">✦ Suggested Follow-up Questions</div>
                    </div>
                    """, unsafe_allow_html=True)
                    f_cols = st.columns(len(followups))
                    for f_idx, f_query in enumerate(followups):
                        with f_cols[f_idx]:
                            if st.button(f_query, key=f"fup_{idx}_{f_idx}", use_container_width=True):
                                set_pending_query(f_query)
                                st.rerun()

    # Process Pending Query with Real-Time Streaming
    if st.session_state.pending_query:
        active_q = st.session_state.pending_query
        st.session_state.pending_query = None

        # Add to conversation history
        if active_q not in st.session_state.conversation_history:
            st.session_state.conversation_history.insert(0, active_q)
            st.session_state.conversation_history = st.session_state.conversation_history[:10]

        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": active_q,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_q)

        # Retrieve context & Stream answer
        start_t = time.perf_counter()
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Retrieving verified college documents..."):
                prompt, sources, context_text = prepare_rag_context(
                    active_q,
                    top_k=st.session_state.top_k,
                )

            # Stream LLM generation
            llm = get_cached_llm(
                model_name=st.session_state.selected_model,
                temperature=st.session_state.temperature,
            )

            def token_generator():
                for chunk in llm.stream(prompt):
                    if chunk.content:
                        yield chunk.content

            response_content = st.write_stream(token_generator())
            elapsed = round(time.perf_counter() - start_t, 2)

            # Render Citations Immediately
            if sources:
                with st.expander(f"📚 Verified Sources & References ({len(sources)} citations)", expanded=False):
                    for src in sources:
                        st.markdown(
                            f"<span class='source-pill-badge'>📄 <strong>{src['source']}</strong> · Page {src['page']}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"<div class='source-excerpt'>\"{src['snippet']}\"</div>", unsafe_allow_html=True)
                        if src.get("urls"):
                            for u in src["urls"]:
                                st.markdown(f"🔗 [Official Portal Link]({u})")

            st.caption(f"⚡ Latency: {elapsed}s · Engine: {st.session_state.selected_model} · {datetime.now().strftime('%H:%M:%S')}")

            # Generate smart follow-up suggestions
            followup_suggestions = get_suggested_followups(active_q, response_content)

            # Store in session state messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_content,
                "sources": sources,
                "latency": elapsed,
                "model": st.session_state.selected_model,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "followups": followup_suggestions,
            })

            # Record telemetry
            st.session_state.query_analytics.append({
                "query": active_q,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "latency": elapsed,
                "sources_count": len(sources),
                "model": st.session_state.selected_model,
                "status": "Success",
            })

            st.rerun()

    # Chat Input Box (Fixed at bottom, Enter-to-send, Auto-clearing)
    chat_input_val = st.chat_input("Ask MIST AI… e.g., 'What is B.Tech fee structure?' or 'What are attendance condonation rules?'")
    if chat_input_val:
        set_pending_query(chat_input_val)
        st.rerun()


# =========================================================
# 7. WORKSPACE: COLLEGE DASHBOARD
# =========================================================
elif st.session_state.page == "College Dashboard":
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
      <div style="font-size: 11px; font-weight: 700; color: #818CF8; letter-spacing: 1.5px; text-transform: uppercase;">
        {config.COLLEGE_SHORT_NAME} · OVERVIEW
      </div>
      <h2 style="margin: 4px 0 6px 0;">College Operations Dashboard</h2>
      <p style="color: #94A3B8; font-size: 14px; margin: 0;">
        Real-time telemetry of indexed knowledge assets, autonomous regulation details, and fast inquiry cards.
      </p>
    </div>
    """, unsafe_allow_html=True)

    health = check_system_health()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Indexed Knowledge Docs", health.get("pdf_document_count", 0), "Official PDF Files")
    m2.metric("Knowledge Vectors", health.get("chunks_indexed", 0), "ChromaDB Chunks")
    m3.metric("Response Engine", "Groq LPU", "Sub-second Inference")
    m4.metric("Accreditation", "NAAC A+", "Autonomous · JNTUH")

    st.write("")

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
                    set_pending_query(prompt)
                    st.rerun()


# =========================================================
# 8. WORKSPACE: DOCUMENT VAULT & INGESTION
# =========================================================
elif st.session_state.page == "Document Vault":
    st.markdown("""
    <div style="margin-bottom: 20px;">
      <div style="font-size: 11px; font-weight: 700; color: #818CF8; letter-spacing: 1.5px; text-transform: uppercase;">
        MIST · KNOWLEDGE VAULT
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
                    st.caption(f"Size: {file_size_kb} KB · Last modified: {modified_time}")
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
# 9. WORKSPACE: SEMANTIC SEARCH LAB
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
                        st.markdown(f"<div class='source-excerpt'>{match['content']}</div>", unsafe_allow_html=True)
                        if match.get("urls"):
                            st.caption(f"Detected links: {', '.join(match['urls'])}")


# =========================================================
# 10. WORKSPACE: USAGE ANALYTICS
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
    a2.metric("Student Inquiries", student_q, "Queries processed")
    a3.metric("AI Responses", ai_resp, "Generated")
    a4.metric("Avg Response Speed", f"{avg_latency}s" if avg_latency else "Sub-sec", "Groq LPU Latency")

    st.write("")

    # Visual chart if we have queries
    if st.session_state.query_analytics:
        st.markdown("### 📊 Query Latency Distribution (Seconds)")
        df = pd.DataFrame(st.session_state.query_analytics)
        st.bar_chart(df, x="query", y="latency", color="#6366F1", height=240)

    st.write("")
    st.markdown("### 📋 Query Audit Log")

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
# 11. WORKSPACE: SETTINGS & SYSTEM HEALTH
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

            st.session_state.selected_model = st.selectbox(
                "Active Groq LLM Model",
                options=config.AVAILABLE_MODELS,
                index=config.AVAILABLE_MODELS.index(st.session_state.selected_model) if st.session_state.selected_model in config.AVAILABLE_MODELS else 0,
                help="Select which high-speed model generates answers from retrieved documents.",
            )

            st.session_state.top_k = st.slider(
                "Top-K Document Chunks to Retrieve",
                min_value=1,
                max_value=8,
                value=st.session_state.top_k,
                help="Higher values provide more context to the LLM but slightly increase prompt size.",
            )

            st.session_state.temperature = st.slider(
                "LLM Temperature (Creativity vs Determinism)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.temperature),
                step=0.05,
                help="Lower values (0.0-0.2) ensure strict adherence to college documents.",
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
