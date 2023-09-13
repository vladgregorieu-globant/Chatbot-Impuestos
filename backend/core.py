import os
from typing import Any, Dict, List

from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

INDEX_NAME = "internal-revenue-code-index"


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = HuggingFaceHubEmbeddings()
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")
    repo_id = "bigcode/octocoder"
    chat = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 800}
    )

    qa = RetrievalQA.from_chain_type(
        llm=chat, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    return qa(query)

if __name__ == "__main__":
    print(run_llm(query="file for claiming child tax credit?"))