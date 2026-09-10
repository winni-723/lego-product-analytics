"""RAG assistant over the project's knowledge docs (LangChain + Ollama + Chroma).

This is the retrieval half of the hybrid agent. It answers conceptual questions
("what is a flagship set?", "how does the popularity model work?") by retrieving
the most relevant chunks from docs/knowledge/ and asking a local Llama model to
answer using only that context.

Everything runs locally and free via Ollama. Prerequisites:
    1. Install Ollama (https://ollama.com)
    2. ollama pull llama3.2
    3. ollama pull nomic-embed-text

Usage:
    # build the vector index once (re-run when docs change)
    .\\venv\\Scripts\\python.exe src\\rag_assistant.py build

    # ask a question
    .\\venv\\Scripts\\python.exe src\\rag_assistant.py ask "What is a flagship set?"
"""
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "docs" / "knowledge"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "lego_knowledge"

CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL)


def build_index() -> None:
    """Load the markdown docs, split them into chunks, embed, and store in Chroma."""
    docs = [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
        for p in sorted(KNOWLEDGE_DIR.glob("*.md"))
    ]
    if not docs:
        raise SystemExit(f"No .md files found in {KNOWLEDGE_DIR}")

    # Split into overlapping chunks so retrieval returns focused passages, not
    # whole files. Overlap keeps sentences from being cut mid-thought.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    # Embed each chunk with the local Ollama embedding model and persist to disk.
    Chroma.from_documents(
        chunks,
        embedding=_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION,
    )
    print(f"Indexed {len(chunks)} chunks from {len(docs)} docs -> {CHROMA_DIR}")


def rag_answer(question: str, k: int = 4) -> tuple[str, list[str]]:
    """Retrieve the top-k relevant chunks, have Llama answer, return (answer, sources).

    Reusable by both the CLI and the Streamlit chat UI.
    """
    store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=_embeddings(),
        collection_name=COLLECTION,
    )
    retrieved = store.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(d.page_content for d in retrieved)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a product-analytics assistant for a LEGO dataset project and you are only able to: "
         "(A) answer concept questions according to docs and (B) answer data questions by using SQL. "
         "You cannot run A/B tests, train models, or create dashboards—those are features of the project, not actions you can perform yourself. "
         "Refer to yourself in the FIRST person ('I can answer…', 'I cannot…'). "
         "Refer to the project's features in the third person ('this project includes X'), "
         "and never claim those features as your own actions. "
         "If asked what you can do, describe only the two capabilities mentioned above. "
         "Answer the question using ONLY the context below. Be concise. "
         "If the answer is not in the context, say you don't know."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    answer = (prompt | llm).invoke({"context": context, "question": question}).content

    sources = sorted({d.metadata.get("source", "?") for d in retrieved})
    return answer, sources


def ask(question: str, k: int = 4) -> str:
    """CLI wrapper: answer + a sources footer as one string."""
    answer, sources = rag_answer(question, k)
    return f"{answer}\n\n[sources: {', '.join(sources)}]"


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        build_index()
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        print(ask(" ".join(sys.argv[2:])))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
