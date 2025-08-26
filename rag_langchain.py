import os
import glob
from pathlib import Path

# For loading PDF documents
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# For splitting documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter


# for creating rag
from google.adk.tools.retrieval import LlamaIndexRetrieval
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext



def load_pdf_documents(pdf_dir: str):
    pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    documents = []
    for path in pdf_paths:
        try:
            loader = PyMuPDFLoader(path)
        except Exception:
            loader = UnstructuredPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)
    return documents



def build_vectorstore(documents, persist_dir: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n", "\u200c", " "])
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    return vectordb

def get_autodesk_vectorstore() -> Chroma:
    """
    Load a persisted Chroma DB if it exists, otherwise fail fast.
    """
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db_path = Path(PERSIST_DIRECTORY)
    if db_path.exists() and any(db_path.iterdir()):
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedder,
            # client_settings=settings,   <- drop this line
        )
    else:
        raise FileNotFoundError(
            f"No vectorstore found in {PERSIST_DIRECTORY!r}. "
            "Please run your build step first."
        )


# ------------------ building RAG --------------------
if __name__ == "__main__":
    PERSIST_DIRECTORY = "./persisted_chroma"
    pdf_dir = "./document"  # your folder containing PDFs
    documents = load_pdf_documents(pdf_dir)
    vectordb = build_vectorstore(documents, PERSIST_DIRECTORY)
    vectordb.persist()