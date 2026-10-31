import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from rag_pipeline import get_answer

# =========================================================
# 1. PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="MIST College Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

# =========================================================
# 2. SESSION STATE
# =========================================================
DEFAULTS = {
    "messages": [],
    "page": "AI Assistant",
    "pending_question": None,
    "chat_input_value": "",
    "conversation_history": [],
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================================================
# 3. PROFESSIONAL DARK WORKSPACE THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --page: #080D1B;
    --panel: #0D1528;
    --panel-2: #111D35;
    --line: rgba(148, 163, 220, .16);
    --muted: #9AA9C8;
    --text: #F4F7FF;
    --accent: #7C72FF;
    --accent-2: #4F8CFF;
}

.stApp {
    background:
      radial-gradient(ellipse at 72% 0%, rgba(55, 75, 165, .18), transparent 34%),
      radial-gradient(ellipse at 0% 100%, rgba(37, 73, 145, .12), transparent 30%),
      var(--page);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    max-width: 1280px;
    padding-top: 1.45rem;
    padding-bottom: 2.8rem;
}
h1, h2, h3 {
    font-family: 'Space Grotesk', 'DM Sans', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -.4px;
}
p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
    color: #D6DEF2;
}
hr {
    border-color: var(--line) !important;
}

/* Hide Streamlit chrome while retaining the app itself. */
#MainMenu, footer, [data-testid="stDeployButton"] {
    visibility: hidden !important;
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090F20 0%, #0D1830 100%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.1rem;
}
[data-testid="stSidebar"] * {
    color: #DCE5FA;
}
.brand-wrap {
    padding: 10px 10px 17px 10px;
}
.brand-mark {
    width: 42px;
    height: 42px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    background: linear-gradient(135deg, #4F8CFF, #7C5CFF);
    box-shadow: 0 8px 24px rgba(86, 100, 255, .28);
    font-size: 21px;
    margin-right: 10px;
    vertical-align: middle;
}
.brand-name {
    display: inline-block;
    vertical-align: middle;
    color: #FFFFFF;
    font: 700 19px 'Space Grotesk', sans-serif;
}
.brand-caption {
    color: #8193B8;
    font-size: 9px;
    letter-spacing: 1.65px;
    line-height: 1.8;
    margin: 10px 0 0 2px;
    text-transform: uppercase;
}
.side-label {
    color: #7184AA;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.6px;
    margin: 17px 0 8px 8px;
    text-transform: uppercase;
}
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: #B9C7E3 !important;
    text-align: left !important;
    border-radius: 10px !important;
    min-height: 42px !important;
    padding: 0 12px !important;
    font-weight: 600 !important;
    transition: .18s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(93, 117, 210, .15) !important;
    border-color: rgba(125, 143, 255, .2) !important;
    color: #FFFFFF !important;
}
.side-note {
    margin: 15px 4px 4px 4px;
    padding: 13px;
    background: linear-gradient(135deg, rgba(57, 86, 174, .20), rgba(100, 76, 206, .12));
    border: 1px solid rgba(122, 139, 255, .20);
    border-radius: 13px;
}
.side-note strong {
    color: #F5F7FF;
    font-size: 12px;
}
.side-note p {
    color: #9EAFD0;
    font-size: 11px;
    line-height: 1.55;
    margin: 5px 0 0 0;
}

/* Top bar and hero */
.topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 14px;
    margin: 2px 0 25px 0;
}
.eyebrow {
    color: #8EA4D1;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
}
.page-heading {
    color: #FFFFFF;
    font: 700 29px 'Space Grotesk', sans-serif;
    margin: 5px 0 4px 0;
}
.page-subtitle {
    color: #9EADCA;
    font-size: 13px;
    margin: 0;
}
.hero {
    max-width: 880px;
    margin: 52px auto 28px auto;
    text-align: center;
    padding: 12px 20px 10px 20px;
}
.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    color: #B7C6FF;
    background: rgba(91, 111, 220, .13);
    border: 1px solid rgba(130, 147, 255, .22);
    border-radius: 100px;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .5px;
}
.hero h1 {
    font-size: clamp(28px, 3.4vw, 43px) !important;
    line-height: 1.16;
    margin: 20px 0 10px 0;
    background: linear-gradient(90deg, #FFFFFF 20%, #C9D5FF 65%, #AFA4FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #9EADCA;
    font-size: 14px;
    margin: 0;
}

/* Composer */
.composer-shell {
    max-width: 850px;
    margin: 0 auto 20px auto;
    padding: 10px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(89, 119, 220, .24), rgba(121, 90, 230, .16));
    border: 1px solid rgba(132, 151, 255, .32);
    box-shadow: 0 14px 40px rgba(0, 0, 0, .20);
}
div[data-testid="stTextInput"] {
    margin: 0 !important;
}
div[data-testid="stTextInput"] input {
    min-height: 48px !important;
    background: #111B31 !important;
    color: #F4F7FF !important;
    -webkit-text-fill-color: #F4F7FF !important;
    border: 1px solid rgba(143, 162, 220, .22) !important;
    border-radius: 11px !important;
    padding: 10px 15px !important;
    font-size: 14px !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #8292B2 !important;
    -webkit-text-fill-color: #8292B2 !important;
    opacity: 1 !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #8276FF !important;
    box-shadow: 0 0 0 3px rgba(124, 114, 255, .12) !important;
}
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(110deg, #4F73F5, #805BEE) !important;
    color: white !important;
    border: 1px solid rgba(177, 185, 255, .24) !important;
    border-radius: 11px !important;
    min-height: 43px !important;
    font-weight: 700 !important;
    transition: .18s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    filter: brightness(1.12);
    border-color: rgba(208, 213, 255, .6) !important;
    transform: translateY(-1px);
}

/* Tool links */
.tools-heading {
    max-width: 850px;
    margin: 25px auto 10px auto;
    color: #8294B8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.tools-frame {
    max-width: 850px;
    margin: 0 auto;
}
.tools {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: clamp(13px, 3vw, 30px);
    flex-wrap: wrap;
    padding: 8px 0 16px 0;
}
.tool {
    width: 74px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 7px;
    text-decoration: none;
    transition: transform .18s ease;
}
.tool:hover { transform: translateY(-4px); }
.tool-icon {
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 15px;
    background: #142344;
    border: 1px solid rgba(142, 164, 235, .24);
    box-shadow: 0 7px 18px rgba(0,0,0,.18);
    overflow: hidden;
}
.tool-icon img {
    width: 27px;
    height: 27px;
    object-fit: contain;
    border-radius: 6px;
}
.tool-name {
    color: #B9C7E3;
    font-size: 10px;
    font-weight: 600;
}
.tool:hover .tool-name { color: #FFFFFF; }

/* Prompt cards */
.prompt-label {
    max-width: 850px;
    margin: 26px auto 10px auto;
    color: #8294B8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
div[data-testid="stHorizontalBlock"]:has(.prompt-card) {
    gap: 12px;
}
.prompt-card {
    min-height: 104px;
    padding: 14px;
    background: linear-gradient(145deg, rgba(21, 34, 62, .95), rgba(14, 24, 44, .95));
    border: 1px solid var(--line);
    border-radius: 13px;
}
.prompt-card b {
    color: #F2F5FF;
    font-size: 12px;
}
.prompt-card p {
    color: #91A2C3;
    font-size: 11px;
    margin: 7px 0 0 0;
    line-height: 1.5;
}

/* Messages and general cards */
[data-testid="stChatMessage"] {
    background: rgba(18, 30, 55, .92) !important;
    border: 1px solid var(--line) !important;
    border-radius: 15px !important;
    padding: 16px !important;
    margin: 12px 0 !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: #F1F5FF !important;
}
[data-testid="stChatMessage"] a { color: #A9B8FF !important; }
[data-testid="stMetric"] {
    background: #111C33;
    border: 1px solid var(--line);
    border-radius: 13px;
    padding: 14px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color: #F4F7FF !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(16, 28, 51, .72);
    border-color: var(--line) !important;
    border-radius: 14px !important;
}
[data-testid="stCaptionContainer"] { color: #8FA0C0 !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. HELPERS
# =========================================================
def start_new_chat():
    if st.session_state.messages:
        first_user_message = next(
            (m["content"] for m in st.session_state.messages if m["role"] == "user"),
            "New conversation",
        )
        st.session_state.conversation_history.insert(0, first_user_message[:55])
        st.session_state.conversation_history = st.session_state.conversation_history[:8]
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.session_state.chat_input_value = ""
    st.session_state.page = "AI Assistant"


def submit_question():
    typed = st.session_state.get("chat_input_value", "").strip()
    if typed:
        st.session_state.pending_question = typed
    st.session_state.chat_input_value = ""


def process_pending_question():
    question = st.session_state.pending_question
    if not question:
        return
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        with st.spinner("Searching MIST college documents..."):
            answer = get_answer(question)
    except Exception as error:
        print("Chatbot error:", error)
        answer = "Sorry, I couldn't process your question right now. Please try again."
    st.session_state.messages.append({"role": "assistant", "content": answer})


def ask_from_card(prompt):
    st.session_state.pending_question = prompt
    st.session_state.page = "AI Assistant"


# =========================================================
# 5. SIDEBAR WORKSPACE
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
      <span class="brand-mark">🎓</span><span class="brand-name">MIST Connect</span>
      <div class="brand-caption">Mother Teresa Institute of Science & Technology</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-label">Workspace</div>', unsafe_allow_html=True)
    nav_items = [
        ("✦", "AI Assistant"),
        ("▦", "Dashboard"),
        ("▤", "Documents"),
        ("◷", "Analytics"),
        ("⬡", "Knowledge Base"),
        ("⚙", "Settings"),
    ]
    for symbol, label in nav_items:
        if st.button(f"{symbol}   {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()

    st.markdown('<div class="side-label">Recent conversations</div>', unsafe_allow_html=True)
    if st.session_state.conversation_history:
        for idx, item in enumerate(st.session_state.conversation_history[:6]):
            st.caption(f"◦  {item}")
    else:
        st.caption("Your recent questions will appear here.")

    st.markdown("""
    <div class="side-note">
      <strong>✦ MIST AI Assistant</strong>
      <p>Ask questions about college services and get answers from the connected knowledge base.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    if st.button("＋   New conversation", key="sidebar_new_chat", use_container_width=True):
        start_new_chat()
        st.rerun()
    st.markdown("""
    <div style="padding:12px 8px 4px 8px;color:#7487AC;font-size:10px;line-height:1.8;">
      <span style="color:#61D6A3;">●</span> SYSTEM READY<br>
      RETRIEVAL AUGMENTED GENERATION<br>
      MIST COLLEGE · VERSION 2.0
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# 6. AI ASSISTANT
# =========================================================
if st.session_state.page == "AI Assistant":
    if st.session_state.messages:
        # Conversation header
        left, right = st.columns([5, 1])
        with left:
            st.markdown("""
            <div class="eyebrow">MIST · STUDENT SUPPORT</div>
            <div class="page-heading">AI College Assistant</div>
            <p class="page-subtitle">Your campus information, one question away.</p>
            """, unsafe_allow_html=True)
        with right:
            if st.button("＋ New Chat", use_container_width=True, key="top_new_chat"):
                start_new_chat()
                st.rerun()

        for message in st.session_state.messages:
            avatar = "🎓" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        st.markdown('<div class="composer-shell">', unsafe_allow_html=True)
        with st.form("mist_chat_form", clear_on_submit=True):
            input_col, send_col = st.columns([8, 1])
            with input_col:
                st.text_input(
                    "Ask MIST AI",
                    placeholder="Message MIST AI… Ask about admissions, fees, attendance, exams",
                    key="chat_input_value",
                    label_visibility="collapsed",
                )
            with send_col:
                submitted = st.form_submit_button("➤", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if submitted:
            typed = st.session_state.chat_input_value.strip()
            if typed:
                st.session_state.pending_question = typed
                process_pending_question()
                st.rerun()
    else:
        # Landing page inspired by a modern AI workspace.
        st.markdown("""
        <div class="topline">
          <div>
            <div class="eyebrow">MOTHER TERESA INSTITUTE OF SCIENCE & TECHNOLOGY</div>
            <div class="page-heading">MIST College Chatbot</div>
            <p class="page-subtitle">Your intelligent campus information assistant.</p>
          </div>
          <div class="hero-kicker">● RAG ASSISTANT ONLINE</div>
        </div>
        <div class="hero">
          <div class="hero-kicker">✦ YOUR MIST AI WORKSPACE</div>
          <h1>Welcome to your campus assistant 👋</h1>
          <p>What would you like to know about your college today?</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="composer-shell">', unsafe_allow_html=True)
        with st.form("mist_first_chat_form", clear_on_submit=True):
            input_col, send_col = st.columns([8, 1])
            with input_col:
                st.text_input(
                    "Ask MIST AI",
                    placeholder="Message MIST AI… Ask about admissions, fees, attendance, exams",
                    key="chat_input_value",
                    label_visibility="collapsed",
                )
            with send_col:
                submitted = st.form_submit_button("➤", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if submitted:
            typed = st.session_state.chat_input_value.strip()
            if typed:
                st.session_state.pending_question = typed
                process_pending_question()
                st.rerun()

        st.markdown('<div class="tools-heading">Explore AI tools for students</div>', unsafe_allow_html=True)
        components.html("""
        <div class="tools-frame">
          <div class="tools">
            <a class="tool" href="https://www.perplexity.ai/" target="_blank" rel="noopener noreferrer">
              <div class="tool-icon"><img src="https://cdn.simpleicons.org/perplexity/FFFFFF" alt="Perplexity"></div>
              <span class="tool-name">Perplexity</span>
            </a>
            <a class="tool" href="https://claude.ai/" target="_blank" rel="noopener noreferrer">
              <div class="tool-icon"><img src="https://cdn.simpleicons.org/claude/FFFFFF" alt="Claude"></div>
              <span class="tool-name">Claude</span>
            </a>
            <a class="tool" href="https://copilot.microsoft.com/" target="_blank" rel="noopener noreferrer">
              <div class="tool-icon"><img src="https://cdn.simpleicons.org/microsoftcopilot/FFFFFF" alt="Copilot"></div>
              <span class="tool-name">Copilot</span>
            </a>
            <a class="tool" href="https://chatgpt.com/" target="_blank" rel="noopener noreferrer">
              <div class="tool-icon"><img src="https://cdn.simpleicons.org/openai/FFFFFF" alt="ChatGPT"></div>
              <span class="tool-name">ChatGPT</span>
            </a>
            <a class="tool" href="https://gemini.google.com/" target="_blank" rel="noopener noreferrer">
              <div class="tool-icon"><img src="https://cdn.simpleicons.org/googlegemini/FFFFFF" alt="Gemini"></div>
              <span class="tool-name">Gemini</span>
            </a>
          </div>
        </div>
        """, height=94, scrolling=False)

        st.markdown('<div class="prompt-label">Or start with a popular question</div>', unsafe_allow_html=True)
        prompts = [
            ("🎓", "Admissions & eligibility", "How do I apply for admission at MIST?"),
            ("💳", "Fees & payments", "What fee information is available for MIST?"),
            ("📅", "Attendance rules", "What are the attendance requirements?"),
            ("📝", "Exams & regulations", "What examination information is available?"),
        ]
        cols = st.columns(4)
        for idx, (emoji, title, prompt) in enumerate(prompts):
            with cols[idx]:
                st.markdown(
                    f'<div class="prompt-card"><b>{emoji} {title}</b>'
                    f'<p>Get information from the college knowledge base.</p></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Explore →", key=f"prompt_{idx}", use_container_width=True):
                    ask_from_card(prompt)
                    process_pending_question()
                    st.rerun()

# =========================================================
# 7. DASHBOARD
# =========================================================
elif st.session_state.page == "Dashboard":
    st.markdown("""
    <div class="eyebrow">MIST · OVERVIEW</div>
    <div class="page-heading">College Dashboard</div>
    <p class="page-subtitle">A quick overview of your college information assistant.</p>
    """, unsafe_allow_html=True)

    pdf_count = len(list(DOCUMENTS_DIR.glob("*.pdf"))) if DOCUMENTS_DIR.exists() else 0
    question_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
    c1, c2, c3 = st.columns(3)
    c1.metric("Knowledge documents", pdf_count, "PDF files")
    c2.metric("Questions this session", question_count, "Student queries")
    c3.metric("Assistant status", "Online", "Ready")

    st.markdown('<div class="prompt-label">Student services</div>', unsafe_allow_html=True)
    services = [
        ("🎓", "Admissions", "Admission process and eligibility.", "Explain the MIST admission process and eligibility."),
        ("💳", "Fee structure", "Tuition and fee information.", "What fee information is available for MIST?"),
        ("📅", "Attendance", "Attendance rules and requirements.", "What are the attendance requirements?"),
        ("📝", "Examinations", "Exam information and regulations.", "What examination information is available?"),
        ("📚", "Departments", "Courses and department information.", "What departments and courses are available at MIST?"),
        ("🏆", "Scholarships", "Scholarship information in the documents.", "What scholarship information is available in the MIST documents?"),
    ]
    cols = st.columns(3)
    for idx, (emoji, title, desc, prompt) in enumerate(services):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"### {emoji} {title}")
                st.caption(desc)
                if st.button("Ask assistant →", key=f"service_{idx}", use_container_width=True):
                    st.session_state.page = "AI Assistant"
                    ask_from_card(prompt)
                    process_pending_question()
                    st.rerun()

# =========================================================
# 8. DOCUMENTS
# =========================================================
elif st.session_state.page == "Documents":
    st.markdown('<div class="eyebrow">MIST · KNOWLEDGE LIBRARY</div><div class="page-heading">College Documents</div>', unsafe_allow_html=True)
    st.caption("PDF files currently present in your local documents folder.")
    if DOCUMENTS_DIR.exists():
        pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
        st.metric("Available PDF documents", len(pdf_files))
        if not pdf_files:
            st.info("No PDF documents were found in the documents folder.")
        for pdf in pdf_files:
            with st.container(border=True):
                file_col, type_col = st.columns([5, 1])
                with file_col:
                    st.markdown(f"**📄 {pdf.name}**")
                    st.caption("Stored in the MIST chatbot document library")
                with type_col:
                    st.caption("PDF")
    else:
        st.warning("The documents folder was not found.")

# =========================================================
# 9. ANALYTICS
# =========================================================
elif st.session_state.page == "Analytics":
    st.markdown('<div class="eyebrow">MIST · USAGE</div><div class="page-heading">Chat Analytics</div>', unsafe_allow_html=True)
    total = len(st.session_state.messages)
    user_total = sum(1 for m in st.session_state.messages if m["role"] == "user")
    assistant_total = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
    a, b, c = st.columns(3)
    a.metric("Total messages", total)
    b.metric("Student questions", user_total)
    c.metric("AI responses", assistant_total)
    st.info("These statistics cover the current browser session only and reset when the session is cleared.")

# =========================================================
# 10. KNOWLEDGE BASE
# =========================================================
elif st.session_state.page == "Knowledge Base":
    st.markdown('<div class="eyebrow">MIST · RETRIEVAL SYSTEM</div><div class="page-heading">RAG Knowledge Base</div>', unsafe_allow_html=True)
    st.write("The chatbot retrieves relevant text from the PDFs indexed in your Chroma vector database.")
    tech = [
        ("Interface", "Streamlit"),
        ("Orchestration", "LangChain"),
        ("Vector database", "ChromaDB"),
        ("Embeddings", "Hugging Face"),
        ("Language model", "Groq (configured in rag_pipeline.py)"),
        ("Source format", "PDF documents"),
    ]
    for label, value in tech:
        with st.container(border=True):
            l, r = st.columns([1, 3])
            l.caption(label)
            r.markdown(f"**{value}**")
    st.success("Connected through your existing rag_pipeline.py. This UI update does not modify your vector database.")

# =========================================================
# 11. SETTINGS
# =========================================================
elif st.session_state.page == "Settings":
    st.markdown('<div class="eyebrow">MIST · APPLICATION</div><div class="page-heading">Settings & About</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### 🎓 MIST College Chatbot")
        st.write("Mother Teresa Institute of Science and Technology")
        st.caption("Version 2.0 · Retrieval Augmented Generation")
        st.write("Answers are generated using the documents indexed in the connected vector database.")
    st.caption("For official college policies, always confirm details with the relevant college office.")
