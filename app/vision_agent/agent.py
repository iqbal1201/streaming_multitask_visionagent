from google.adk.agents import Agent
from google.adk.tools import VertexAiSearchTool
from google.adk.tools import google_search  # Import the tool
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag
from dotenv import load_dotenv
from .prompt import return_instructions_root
import os
from pathlib import Path
# Library for local RAG (ChromaDB)
# from google.adk.tools.retrieval import LlamaIndexRetrieval
# from llama_index.core import VectorStoreIndex
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.core import StorageContext
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from llama_index.embeddings.langchain import LangchainEmbedding

load_dotenv()

RAG_CORPUS='projects/aisee-ahlab/locations/us-central1/ragCorpora/2305843009213693952'
DATASTORE_ID = "projects/aisee-ahlab/locations/global/collections/default_collection/dataStores/datastore-poc_1753759382820"


ask_vertex_retrieval = VertexAiRagRetrieval(
    name='retrieve_rag_documentation',
    description=(
        'Use this tool to retrieve documentation and reference materials for the question from the RAG corpus,'
    ),
    rag_resources=[
        rag.RagResource(
            # please fill in your own rag corpus
            # here is a sample rag corpus for testing purpose
            # e.g. projects/123/locations/us-central1/ragCorpora/456
            rag_corpus=RAG_CORPUS
        )
    ],
    similarity_top_k=10,
    vector_distance_threshold=0.6,
)



## ----------------------- Grounding Using Google Search --------------------------------------
# root_agent = Agent(
#    # A unique name for the agent.
#    name="vision_agent",
#    # The Large Language Model (LLM) that agent will use.
#    # Please fill in the latest model id that supports live from
#    # https://google.github.io/adk-docs/get-started/streaming/quickstart-streaming/#supported-models
#    model="gemini-2.0-flash-exp",  # for example: model="gemini-2.0-flash-live-001" or model="gemini-2.0-flash-live-preview-04-09"
#    #model = 'gemini-2.5-flash',
#    # A short description of the agent's purpose.
#    description="Agent to answer questions.",
#    # Instructions to set the agent's behavior.
#    instruction="""You are an AiSee agent that interacts with the user in a conversation.
#     Your aim is to help person with vision disability. So you should be helpful to explain the vision you see to the person who use your service.
#     Always introduce yourself that you are an AiSee vision agent that ready to explain the vision you see. Then make conversation with the user.
#     Please ask if the user still interested in the conversation. If not say thank you and stop speaking until the user start speaking again.""",
#    # Add google_search tool to perform grounding with Google search.
#    # tools=[ask_vertex_retrieval],
#    tools=[VertexAiSearchTool(data_store_id=DATASTORE_ID)]
# )

## ----------------------- Grounding Using RAG Engine --------------------------------------
root_agent = Agent(
    model='gemini-2.0-flash-exp',
    name='vision_agent',
    instruction=return_instructions_root(),
    tools=[
        ask_vertex_retrieval,
    ]
)


# ## ----------------------- Grounding Using Datastore --------------------------------------
# root_agent = Agent(
#    # A unique name for the agent.
#    name="vision_agent",
#    model="gemini-2.0-flash-exp",  # for example: model="gemini-2.0-flash-live-001" or model="gemini-2.0-flash-live-preview-04-09"
#    instruction="Answer questions using Vertex AI Search to find information from internal documents. Always cite sources when available.",
#    description="Enterprise document search assistant with Vertex AI Search capabilities",
#    tools=[VertexAiSearchTool(data_store_id=DATASTORE_ID)]
# )




## ----------------------- Grounding Using LOcal RAG --------------------------------------

# PERSIST_DIRECTORY = "/home/iqbaljanuadi/aisee-adk/persisted_chroma"
# lc_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embed_model = LangchainEmbedding(lc_embed)


# def get_autodesk_vectorstore() -> Chroma:
#     """
#     Load a persisted Chroma DB if it exists, otherwise fail fast.
#     """
#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     db_path = Path(PERSIST_DIRECTORY)
#     if db_path.exists() and any(db_path.iterdir()):
#         return Chroma(
#             persist_directory=PERSIST_DIRECTORY,
#             embedding_function=embedder,
#             # client_settings=settings,   <- drop this line
#         )
#     else:
#         raise FileNotFoundError(
#             f"No vectorstore found in {PERSIST_DIRECTORY!r}. "
#             "Please run your build step first."
#         )

# # 1. Grab the collection
# db = get_autodesk_vectorstore()
# collection = db._collection

# # 2. Wrap it for LlamaIndex
# vector_store = ChromaVectorStore(chroma_collection=collection)

# # 3. Create an index over that store
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_vector_store(vector_store, 
#                                             storage_context=storage_context,
#                                             embed_model=embed_model)

# # 4. Get a semantic retriever
# retriever = index.as_retriever(similarity_top_k=6)

# # 5. Wrap it in the ADK Retrieval tool
# retrieve_docs = LlamaIndexRetrieval(
#     name="local_chroma_docs",
#     description="Enterprise document search assistant with Vertex AI Search capabilities",
#     retriever=retriever,
# )


# root_agent = Agent(
#    # A unique name for the agent.
#    name="vision_agent",
#    model="gemini-2.0-flash-exp",  # for example: model="gemini-2.0-flash-live-001" or model="gemini-2.0-flash-live-preview-04-09"
#    instruction="Answer questions using Vertex AI Search to find information from internal documents. Always cite sources when available.",
#    description="Enterprise document search assistant with Vertex AI Search capabilities",
#    tools=[retrieve_docs]
# )