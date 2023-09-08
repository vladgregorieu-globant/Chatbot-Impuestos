import os

from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)
INDEX_NAME = "internal-revenue-code-index"

def ingest_docs()->None:
    loader= PDFMinerLoader("tax-code/i1040gi.pdf")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", "", "\xa0"]
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Splitted into {len(documents)} chunks")
    embeddings = HuggingFaceEmbeddings()
    print(f"Going to add {len(documents)} to Pinecone")
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vector store done ***")


if __name__ == "__main__":
    ingest_docs()