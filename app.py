import os
from typing import List
from pydantic import BaseModel, Field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ==========================================
# 1. Define Data Structure for Flashcards
# ==========================================
class Flashcard(BaseModel):
    front: str = Field(description="The question, concept, or prompt for the front of the flashcard")
    back: str = Field(description="The concise, clear answer or explanation for the back")

class FlashcardSet(BaseModel):
    cards: List[Flashcard] = Field(description="A collection of generated flashcards")

# ==========================================
# 2. Main RAG Flashcard Generator Class
# ==========================================
class GoogleRAGFlashcardGenerator:
    def __init__(self, api_key: str = None, chunk_size: int = 500, chunk_overlap: int = 50):
        # Fetch key from param or environment
        google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY env variable.")

        # Initialize Google Embeddings & Gemini LLM
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=google_api_key
        )
        
        # Text splitter setup
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Output parser for structured JSON output
        self.parser = JsonOutputParser(pydantic_object=FlashcardSet)
        
        # Setup Prompt Template
        self.prompt = PromptTemplate(
            template=(
                "You are an expert tutor creating study materials.\n"
                "Based ONLY on the provided context, generate high-quality flashcards.\n"
                "Focus on key concepts, definitions, and relationships.\n\n"
                "Context:\n{context}\n\n"
                "{format_instructions}\n"
            ),
            input_variables=["context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.vector_store = None

    def ingest_notes(self, text_notes: str):
        """Splits raw text notes into chunks, generates Google embeddings, and stores them in FAISS."""
        # 1. Chunking
        docs = self.text_splitter.create_documents([text_notes])
        
        # 2. Vector Indexing using FAISS with Gemini Embeddings
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        print(f"✅ Successfully indexed {len(docs)} text chunks into FAISS.")

    def generate_flashcards(self, query: str = "Key concepts, definitions, and main ideas", k: int = 4) -> List[dict]:
        """Retrieves relevant chunks using FAISS and generates flashcards with Gemini."""
        if not self.vector_store:
            raise ValueError("Please ingest notes first using ingest_notes().")
            
        # 1. Retrieval (RAG)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(query)
        
        # Combine retrieved context
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # 2. Generation using LCEL (LangChain Expression Language)
        chain = self.prompt | self.llm | self.parser
        response = chain.invoke({"context": context_text})
        
        return response.get("cards", [])

# ==========================================
# 3. Example Execution
# ==========================================
if __name__ == "__main__":
    sample_notes = """
    Photosynthesis is the process used by plants, algae, and certain bacteria to convert light energy into chemical energy.
    This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water.
    
    The overall equation for photosynthesis is: 6CO2 + 6H2O + Light Energy -> C6H12O6 + 6O2.
    
    Photosynthesis occurs in two main stages:
    1. Light-dependent reactions: Occur in the thylakoid membranes of chloroplasts. They require light energy to split water molecules, producing ATP, NADPH, and releasing oxygen as a byproduct.
    2. Calvin Cycle (Light-independent reactions): Occurs in the stroma of chloroplasts. It uses ATP and NADPH produced in the light reactions to fix carbon dioxide into glucose.
    
    Chlorophyll is the primary pigment involved in photosynthesis, absorbing mainly blue and red wavelengths of light while reflecting green light.
    """

    # Initialize with your key (or set GOOGLE_API_KEY in environment)
    generator = GoogleRAGFlashcardGenerator()

    # Step 1: Process and store notes in FAISS using Google Embeddings
    generator.ingest_notes(sample_notes)

    # Step 2: Retrieve context & generate flashcards using Gemini
    cards = generator.generate_flashcards(
        query="Explain the stages of photosynthesis and key pigments",
        k=3
    )

    # Step 3: Print Results
    print("\n--- Generated Flashcards ---\n")
    for idx, card in enumerate(cards, 1):
        print(f"Card {idx}:")
        print(f"  Front: {card['front']}")
        print(f"  Back : {card['back']}\n")
