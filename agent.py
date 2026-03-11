"""
agent.py
========
The Banking Chatbot Agent.
Wraps the RAG pipeline and manages conversation logic.

Handles:
  - Loading the pipeline once at startup
  - Routing questions to the RAG chain
  - Fetching source documents for transparency
  - Formatting responses for the UI
  - Keeping a conversation history
"""

from rag_engine import build_rag_pipeline
from utils import format_sources, validate_pdf_path


class BankingChatbotAgent:
    """
    A conversational AI agent for banking queries.
    Uses a RAG pipeline under the hood to answer questions from a PDF knowledge base.
    """

    def __init__(self, pdf_path: str):
        """
        Initialize the agent by building the full RAG pipeline.
        Called ONCE when the app starts.

        Args:
            pdf_path: Path to the banking knowledge base PDF.
        """
        validate_pdf_path(pdf_path)

        print(f"🤖 Initializing BankingChatbotAgent with: {pdf_path}")
        self.pdf_path = pdf_path

        # build_rag_pipeline now returns (chain, retriever)
        self.qa_chain, self.retriever = build_rag_pipeline(pdf_path)
        self.chat_history = []

    def ask(self, question: str) -> dict:
        """
        Ask the agent a banking question.

        Args:
            question: Customer's question as a string.

        Returns:
            {
                "answer": str,    # LLM-generated answer
                "sources": str,   # Formatted source snippets
                "question": str   # Original question
            }
        """
        if not question.strip():
            return {
                "answer": "Please enter a valid question.",
                "sources": "",
                "question": question
            }

        print(f"\n💬 Customer Question: {question}")

        try:
            # Step 1: Get the answer from the LCEL chain
            answer = self.qa_chain.invoke(question)

            # Step 2: Separately fetch source docs for display
            source_docs = self.retriever.invoke(question)
            sources = format_sources(source_docs)

            # Save to history
            self.chat_history.append({"question": question, "answer": answer})

            print(f"🤖 Agent Answer: {answer[:100]}...")

            return {
                "answer": answer,
                "sources": sources,
                "question": question
            }

        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            print(error_msg)
            return {
                "answer": "I'm experiencing technical difficulties. Please try again.",
                "sources": "",
                "question": question
            }

    def reset_history(self):
        """Clear conversation history."""
        self.chat_history = []
        print("🔄 Chat history cleared.")

    def get_history(self) -> list:
        """Return the full conversation history."""
        return self.chat_history
