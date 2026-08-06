import os
from typing import List

from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# =====================================================
# Pydantic Models
# =====================================================

class Flashcard(BaseModel):
    front: str = Field(description="Question shown on the front")
    back: str = Field(description="Answer shown on the back")


class FlashcardSet(BaseModel):
    cards: List[Flashcard]


# =====================================================
# Flashcard Generator
# =====================================================

class GoogleRAGFlashcardGenerator:

    def __init__(
        self,
        api_key=None,
        chunk_size=500,
        chunk_overlap=50,
    ):

        google_api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not google_api_key:
            raise ValueError(
                "Please provide GOOGLE_API_KEY."
            )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_api_key,
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=google_api_key,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.parser = JsonOutputParser(
            pydantic_object=FlashcardSet
        )

        self.prompt = PromptTemplate(
            template="""
You are an expert tutor.

Generate concise, high-quality flashcards ONLY from the context.

Context:
{context}

{format_instructions}
""",
            input_variables=["context"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

        self.vector_store = None

    # -------------------------------------------------

    def ingest_notes(self, notes: str):

        docs = self.text_splitter.create_documents([notes])

        self.vector_store = FAISS.from_documents(
            docs,
            self.embeddings,
        )

        print(f"Indexed {len(docs)} chunks.")

    # -------------------------------------------------

    def generate_flashcards(
        self,
        query="Key concepts",
        k=4,
    ):

        if self.vector_store is None:
            raise ValueError(
                "Run ingest_notes() first."
            )

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )

        docs = retriever.invoke(query)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        chain = (
            self.prompt
            | self.llm
            | self.parser
        )

        result = chain.invoke(
            {"context": context}
        )

        if isinstance(result, dict):
            return result.get("cards", [])

        if hasattr(result, "cards"):
            return result.cards

        return result


# =====================================================
# Example
# =====================================================

if __name__ == "__main__":

    GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"

    notes = """
Photosynthesis is the process by which plants convert light energy into chemical energy.

Overall Equation:
6CO2 + 6H2O + Light -> C6H12O6 + 6O2

Stages:

1. Light-dependent reactions
- Occur in thylakoid membranes
- Produce ATP
- Produce NADPH
- Release oxygen

2. Calvin Cycle
- Occurs in stroma
- Uses ATP and NADPH
- Produces glucose

Chlorophyll absorbs blue and red light and reflects green.
"""

    generator = GoogleRAGFlashcardGenerator(
        api_key=GOOGLE_API_KEY
    )

    generator.ingest_notes(notes)

    cards = generator.generate_flashcards(
        query="Photosynthesis stages",
        k=3,
    )

    print("\nGenerated Flashcards\n")

    for i, card in enumerate(cards, start=1):

        if isinstance(card, dict):
            front = card["front"]
            back = card["back"]
        else:
            front = card.front
            back = card.back

        print(f"Card {i}")
        print("Front:", front)
        print("Back :", back)
        print("-" * 40)
