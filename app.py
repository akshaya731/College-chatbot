
import streamlit as st

from rag_pipeline import get_answer


# Configure Streamlit page
st.set_page_config(
    page_title="College FAQ Chatbot",
    page_icon="🎓",
    layout="centered"
)

# PASTE PURPLE CSS HERE

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4efff, #e5ebff, #e8f4ff);
}

.block-container {
    max-width: 1000px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.header {
    text-align: center;
    padding: 35px 20px;
    margin-bottom: 25px;
    border-radius: 24px;
    background: linear-gradient(120deg, #392078, #6845c5, #3978d8);
    box-shadow: 0 10px 30px rgba(67, 47, 140, 0.2);
}

.header-icon {
    font-size: 48px;
}

.header h1 {
    color: white !important;
    font-size: 40px !important;
    font-weight: 800 !important;
    margin: 8px 0;
}

.header p {
    color: #eeeaff !important;
    font-size: 16px;
}

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.94);
    border: 1px solid #ded4fa;
    border-radius: 20px;
    padding: 18px;
    margin: 14px 0;
    box-shadow: 0 5px 16px rgba(65,48,130,0.08);
}

[data-testid="stChatInput"] {
    background: white;
    border: 2px solid #a78bfa;
    border-radius: 18px;
    box-shadow: 0 5px 20px rgba(75,55,150,0.10);
}

[data-testid="stChatInput"]:focus-within {
    border-color: #6845c5;
}

.stButton button {
    background: linear-gradient(100deg, #6845c5, #3978d8);
    color: white;
    border-radius: 12px;
    border: none;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #30205f, #49358a);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# Application heading
st.title("🎓 College FAQ Chatbot")

st.write(
    "Welcome! Ask me anything about college admissions, "
    "fees, attendance, examinations and departments."
)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous chat messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept student questions
question = st.chat_input("Ask your college-related question...")


if question:

    # Display student question
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Searching college documents..."):

            try:
                answer = get_answer(question)

            except Exception as e:
                answer = (
                    "Sorry, something went wrong. "
                    f"Please check your configuration. Error: {e}"
                )

        st.markdown(answer)

    # Save chatbot response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
