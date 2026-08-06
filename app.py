import os
import json
import tempfile
import streamlit as st

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Flashcard AI Generator",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ RAG PDF Flashcard Generator")
st.write("Upload your lecture notes/PDFs and extract AI-powered flashcards via Retrieval-Augmented Generation.")

# --- Sidebar: Setup & API Keys ---
with st.sidebar:
    st.header("🔑 Configuration")
    GOOGLE_API_KEY = st.text_input("ENTER GOOGLE API KEY", type="password")
    if not GOOGLE API KEY:
        st.info("Please provide a GOOGLE API KEY to proceed.")
        st.stop()
    
    os.environ["GOOGLE API KEY"] = GOOGLE_API_KEY

# --- Pydantic Schema for Structured Output ---
class Flashcard(BaseModel):
    question: str = Field(description="The front side of the card: clear question or term")
    answer: str = Field(description="The back side of the card: concise explanation or definition")

class FlashcardList(BaseModel):
    cards: list[Flashcard]

# --- Core RAG Processing Functions ---
@st.cache_resource(show_spinner=False)
def process_pdf(pdf_file):
    """Saves uploaded PDF to temporary file, chunks text, and generates FAISS Vector DB."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Chunk Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # Embed & Index
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Clean up temp file
    os.remove(tmp_path)
    return vectorstore

def generate_flashcards(vectorstore, topic, num_cards):
    """Retrieves context via RAG and uses LLM to generate structured flashcards."""
    # Retrieve top relevant context chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(topic if topic else "key concepts core summary")
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Setup LLM & Parser
    parser = JsonOutputParser(pydantic_object=FlashcardList)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    prompt_template = """
    You are an expert study assistant. Use the following context retrieved from the user's notes to create {num_cards} high-quality flashcards.
    Focus on key definitions, core concepts, formulas, and critical distinctions.

    CONTEXT:
    {context}

    TOPIC / INSTRUCTION:
    {topic}

    FORMAT INSTRUCTIONS:
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "topic", "num_cards"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser

    response = chain.invoke({
        "context": context_text,
        "topic": topic if topic else "General summary of important concepts",
        "num_cards": num_cards
    })

    return response.get("cards", [])

# --- Main App Interface ---
uploaded_file = st.file_uploader("Upload Study Notes (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF, building embeddings, and indexing vector database..."):
        try:
            vectorstore = process_pdf(uploaded_file)
            st.success("PDF processed successfully!")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.stop()

    st.markdown("---")
    
    # Input Controls
    col1, col2 = st.columns([2, 1])
    with col1:
        topic_query = st.text_input("Focus Topic (optional)", placeholder="e.g., Photosynthesis, Neural Networks, Chapter 3")
    with col2:
        num_cards = st.slider("Number of Flashcards", min_value=3, max_value=15, value=5)

    if st.button("✨ Generate Flashcards"):
        with st.spinner("Retrieving notes and synthesizing flashcards..."):
            try:
                flashcards = generate_flashcards(vectorstore, topic_query, num_cards)
                st.session_state["flashcards"] = flashcards
                st.session_state["card_index"] = 0
                st.session_state["show_answer"] = False
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

# --- Interactive Flashcard Viewer ---
if "flashcards" in st.session_state and st.session_state["flashcards"]:
    cards = st.session_state["flashcards"]
    idx = st.session_state.get("card_index", 0)
    
    st.markdown("---")
    st.subheader(f"🎴 Card {idx + 1} of {len(cards)}")

    current_card = cards[idx]

    # Flashcard Container Box
    card_container = st.container(border=True)
    with card_container:
        if not st.session_state.get("show_answer", False):
            st.markdown("### **Question / Concept:**")
            st.markdown(f"#### {current_card['question']}")
        else:
            st.markdown("### **Answer / Explanation:**")
            st.info(current_card['answer'])

    # Controls
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("⬅️ Previous") and idx > 0:
            st.session_state["card_index"] -= 1
            st.session_state["show_answer"] = False
            st.rerun()

    with c2:
        if st.button("🔄 Flip Card"):
            st.session_state["show_answer"] = not st.session_state.get("show_answer", False)
            st.rerun()

    with c3:
        if st.button("Next ➡️") and idx < len(cards) - 1:
            st.session_state["card_index"] += 1
            st.session_state["show_answer"] = False
            st.rerun()

    # JSON Export Option
    st.markdown("---")
    st.download_button(
        label="📥 Export Flashcards as JSON",
        data=json.dumps(cards, indent=2),
        file_name="flashcards.json",
        mime="application/json"
    )
