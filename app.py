import streamlit as st
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "/app/data/BOI.pdf"
MODEL_NAME = "llama3.2:3b"  # Fixed model name
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./app/chroma_db"
OLLAMA_BASE_URL = "http://llama-container:11434"  # Docker service URL



def pull_model_if_needed(model_name, base_url):
    """Pull model if not already available in Docker container."""
    try:
        # Check if model exists
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if model_name not in model_names:
                logging.info(f"Pulling model {model_name}...")
                pull_response = requests.post(f"{base_url}/api/pull", 
                                            json={"name": model_name})
                if pull_response.status_code == 200:
                    logging.info(f"Model {model_name} pulled successfully.")
                else:
                    logging.error(f"Failed to pull model {model_name}")
            else:
                logging.info(f"Model {model_name} already available.")
    except Exception as e:
        logging.warning(f"Could not check/pull model: {e}")


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = PyPDFLoader(file_path=doc_path)  # Change this line
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    pull_model_if_needed(EMBEDDING_MODEL, OLLAMA_BASE_URL)

    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL  # Add base URL for Docker
    )


    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        # Load and process the PDF document
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None

        # Split the documents into chunks
        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    return vector_db


# def create_retriever(vector_db, llm):
#     """Create a multi-query retriever."""
#     QUERY_PROMPT = PromptTemplate(
#         input_variables=["question"],
#         template="""You are an AI language model assistant. Your task is to generate five
# different versions of the given user question to retrieve relevant documents from
# a vector database. By generating multiple perspectives on the user question, your
# goal is to help the user overcome some of the limitations of the distance-based
# similarity search. Provide these alternative questions separated by newlines.
# Original question: {question}""",
#     )

#     retriever = MultiQueryRetriever.from_llm(
#         vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
#     )
#     logging.info("Retriever created.")
#     return retriever


def create_retriever(vector_db, llm):
    """Create a simple retriever (faster than multi-query)."""
    retriever = vector_db.as_retriever(
        search_kwargs={"k": 4}  # Return top 4 most relevant chunks
    )
    return retriever



def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain


def main():
    st.title("Chat with PDF docs...")

    # Debug info
    #st.write(f"Looking for PDF at: {DOC_PATH}")
    #st.write(f"PDF exists: {os.path.exists(DOC_PATH)}")
    #st.write(f"Persist directory exists: {os.path.exists(PERSIST_DIRECTORY)}")

    # Add the reset button here
    # if st.button("Reset Vector Database"):
    #     import shutil
    #     if os.path.exists(PERSIST_DIRECTORY):
    #         shutil.rmtree(PERSIST_DIRECTORY)
    #     st.success("Vector database deleted. Restart the app to recreate.")

    # User input
    user_input = st.text_input("Enter your question:", "")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                # Pull the main model if needed
                pull_model_if_needed(MODEL_NAME, OLLAMA_BASE_URL)
                
                # Initialize the language model with Docker base URL
                llm = ChatOllama(
                    model=MODEL_NAME,
                    base_url=OLLAMA_BASE_URL  # Add base URL for Docker
                )

                # Load the vector database
                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return

                # Create the retriever
                retriever = create_retriever(vector_db, llm)

                # Create the chain
                chain = create_chain(retriever, llm)

                # Get the response
                response = chain.invoke(input=user_input)

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.error(f"Error: {str(e)}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()
