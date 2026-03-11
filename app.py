"""
app.py
======
Streamlit UI for the AI Banking Chatbot.
Run with: streamlit run app.py

Features:
  - Clean chat interface
  - Source document display (shows where answers come from)
  - Chat history
  - New Chat button
"""

import streamlit as st
from agent import BankingChatbotAgent

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🏦 FinSight AI",
    page_icon="🏦",
    layout="centered"
)

# ─────────────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #0052cc;
    }
    .chat-bubble-user {
        background-color: #0052cc;
        color: white;
        border-radius: 15px 15px 0px 15px;
        padding: 12px 16px;
        margin: 6px 0;
        text-align: right;
    }
    .chat-bubble-bot {
        background-color: #ffffff;
        color: #1a1a1a;
        border-radius: 15px 15px 15px 0px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 4px solid #0052cc;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("## 🏦 FinSight AI")
st.markdown("Ask me anything about our banking services, policies, and FAQs.")
st.markdown("---")

# ─────────────────────────────────────────────
# Initialize Agent (once per session)
# ─────────────────────────────────────────────
PDF_PATH = "/Users/adityadharmadhikari/Desktop/banking_chatbot/data/SBI-Info.pdf"
PDF_PATH = "/Users/adityadharmadhikari/Desktop/banking_chatbot/data/SBI-Limits.pdf"
@st.cache_resource(show_spinner="⚙️ Loading AI Banking Assistant... (first load takes ~30 seconds)", ttl=None)
def load_agent():
    """
    Load the RAG pipeline ONCE and cache it for the entire session.
    @st.cache_resource ensures this runs only on first load — not on every message.
    """
    return BankingChatbotAgent(pdf_path=PDF_PATH)

try:
    agent = load_agent()
except Exception as e:
    st.error(f"❌ Failed to load the chatbot: {str(e)}")
    st.info("💡 Make sure your `.env` file has `GROQ_API_KEY=gsk_...` and `HUGGINGFACEHUB_API_TOKEN=hf_...`")
    st.stop()

# ─────────────────────────────────────────────
# Session State: Chat History
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=80)
    st.markdown("### 🏦 FinSight AI")
    st.markdown("Powered by **RAG + HuggingFace**")
    st.markdown("---")

    st.markdown("**📚 Knowledge Base:**")
    st.markdown("- Account types & fees")
    st.markdown("- Loan & mortgage info")
    st.markdown("- Online banking help")
    st.markdown("- Security policies")
    st.markdown("- Card services")

    st.markdown("---")
    st.markdown("**💡 Sample Questions:**")
    sample_questions = [
        "What types of accounts do you offer?",
        "How do I apply for a loan?",
        "What are the ATM withdrawal limits?",
        "How do I reset my online banking password?",
        "What is the interest rate on savings accounts?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.prefill_question = q

    st.markdown("---")
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.messages = []
        agent.reset_history()
        st.rerun()

# ─────────────────────────────────────────────
# Chat History Display
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🏦"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 View Sources"):
                    st.markdown(msg["sources"])

# ─────────────────────────────────────────────
# Chat Input
# ─────────────────────────────────────────────
# Handle prefilled question from sidebar buttons
prefill = st.session_state.pop("prefill_question", None)
user_input = st.chat_input("Ask a banking question...") or prefill

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Get agent response
    with st.chat_message("assistant", avatar="🏦"):
        with st.spinner("🔍 Searching knowledge base..."):
            response = agent.ask(user_input)

        answer = response["answer"]
        sources = response["sources"]

        st.markdown(answer)
        if sources:
            with st.expander("📄 View Sources"):
                st.markdown(sources)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:12px;'>"
    "🏦 AI Banking Chatbot | Powered by LangChain + HuggingFace + ChromaDB"
    "</p>",
    unsafe_allow_html=True
)
