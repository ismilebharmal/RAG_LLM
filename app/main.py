import os
from typing import Tuple, Any

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import  EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter
# Load environment variables
import json
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from langchain_openai import OpenAIEmbeddings
from typing_extensions import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
class State(TypedDict):
    question: str
    retrive_data:dict
    response:str
    error:str

def load_and_split_pdf(pdf_path: str):
    split_tup = os.path.splitext(pdf_path)
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1133, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return chunks

def get_best_page(response):
    max_score = -1
    best_page = None

    for doc in response:
        score = doc.state["query_similarity_score"]
        if score > max_score:
            max_score = score
            best_page = doc

    if best_page:
        highest_page_content = best_page.page_content
        highest_page_metadata = best_page.metadata
        highest_query_similarity_score = best_page.state["query_similarity_score"]
        return highest_query_similarity_score,highest_page_content,highest_page_metadata
    else:
        return None

# -------------------------------
# 1. LLM Management Setup
# -------------------------------

def get_llm(model_name: str, temperature: float = 0):
    """Initialize LLM (OpenAI, Anthropic, etc.) based on model_name and temperature."""
    if model_name == 'openai':
        return ChatOpenAI(model='gpt-40-mini', temperature=temperature)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_vector_stor_for_llm(query: str) -> tuple[Any, Any, Any,Any]:
    # Initialize Chroma vectorstore with the embedding function
    vectordb = Chroma(embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"), persist_directory='some_data/chroma_db_2')
    base_retriver = vectordb.as_retriever()
    retrive= vectordb.similarity_search_with_relevance_scores(query, k=8)
    print("rerive:",retrive)
    splitter = CharacterTextSplitter(chunk_size=650, chunk_overlap=50, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=OpenAIEmbeddings(model="text-embedding-3-small"))
    relevant_filter = EmbeddingsFilter(embeddings=OpenAIEmbeddings(model="text-embedding-3-small"), similarity_threshold=0)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriver)
    retrive_data = compression_retriever.get_relevant_documents(query)
    # Find the best page from the retrieved documents
    highest_query_similarity_score,highest_page_content,highest_page_metadata = get_best_page(retrive_data)

    return  highest_query_similarity_score,highest_page_content,highest_page_metadata,retrive


prompt = PromptTemplate(
    input_variables=["query", "docs"],
    template="""
        You are a helpful AI code assistant with expertise in LLM AI Cybersecurity and Governance."
                    " Use the following docs to produce a concise code solution to the user question.\n\n
                    question: {query}\n\n DOCS: {docs}
        """,
)
    # prompt = PromptTemplate(
#     input_variables=["query", "docs"],
#     template="""
#         You are a helpful assistant that that can answer questions about given docs.
#
#         Answer the following question: {query}
#         By searching the following transcript: {docs}
#
#         Only use the relevent  information from the transcript to answer the question.
#         Note that the asnwer should not be hallucinate , it should be excatly from the transcript only.
#
#         If you feel like you don't have enough information to answer the question, say "I don't know".
#         """,
# )
openLLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def store_chunks_in_vector_store(chunks: list) -> None:
    client = chromadb.Client()
    # Check if the "consent_collection" exists
    collections = client.list_collections()
    # if "consent_collection" not in collections:
    #     client.create_collection("consent_collection")
    # else:
    #     print("Collection already exists")
    if not os.path.exists('some_data/chroma_db_2'):
        vectordb = Chroma.from_documents(
        documents=chunks,
        # embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        # embedding=HuggingFaceEmbeddings(),
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="some_data/chroma_db_2"
        )
        vectordb.persist()

def call_llm(query,match_doc,llm):
    try:
        prompt_to_llm = prompt.format(query=query,docs=match_doc)
        response = llm.invoke(prompt_to_llm)
        return str(response)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}



# generation of the graph values and foundations
def generate_query_node(state: State):
    """Generate the SQL query using an LLM prompt."""
    try:
        if len(state['question'])==1:
            return {"error":"Length of question is too small"}
        else:
            return state
    except Exception as e:
        print("Exception in generation  call:",str(e))
        return {"error": str(e)}

# generation of the graph values and foundations
def retrival_node(state: State):
    """Generate the SQL query using an LLM prompt."""
    try:
        chunks = load_and_split_pdf("app/some_data/igor.pdf")
        store_chunks_in_vector_store(chunks)
        highest_query_similarity_score, highest_page_content, highest_page_metadata,retrive = load_vector_stor_for_llm(state["question"])
        print("highest_query_similarity_score", highest_query_similarity_score)
        print("highest_page_content", highest_page_content)
        print("highest_page_metadata", highest_page_metadata)

        data={
            "highest_query_similarity_score":highest_query_similarity_score,
            "highest_page_content":highest_page_content,
            "highest_page_metadata":highest_page_metadata,
            "meta_response":retrive

        }
        return {'retrive_data':data}
    except Exception as e:
        print("Exception in retrival call  call:",str(e))
        return {"error": str(e)}

# generation of the graph values and foundations
def response_node(state: State):
    """Generate the SQL query using an LLM prompt."""
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )
        print(state['retrive_data'])
        response = call_llm(state["question"], state['retrive_data']['highest_page_content'], llm)
        state['response']=response
        return state
    except Exception as e:
        print("Exception in response  call:",str(e))
        return {"error": str(e)}


# -------------------------------
# 4. Main Workflow Setup
# -------------------------------


def build_workflow():
    """Main function to build and connect the workflow nodes."""
    workflow = StateGraph(State)
    # Add nodes to workflow
    workflow.add_node("generate_query", generate_query_node)
    workflow.add_node("retrieve", retrival_node)
    workflow.add_node("response_output", response_node)
    # Set node edges (flow)
    workflow.set_entry_point("generate_query")
    workflow.add_edge("generate_query", "retrieve")
    workflow.add_edge("retrieve", "response_output")
    workflow.add_edge("response_output", END)
    return workflow.compile()


workflow_app= build_workflow()


