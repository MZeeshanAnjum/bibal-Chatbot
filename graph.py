'''
title: Eden
description: LangGraph-based Christian Teachings Agent (no external HTTP calls)
required_open_webui_version: 0.4.3
version: 0.1
licence: MIT
'''

import logging
from typing import List, Union, Generator, TypedDict
from pydantic import BaseModel, Field

# Import your LangGraph agent components directly
from langgraph.graph import StateGraph, END, START
from langchain_ollama import OllamaLLM
from langchain.output_parsers import PydanticOutputParser
from transformers import AutoTokenizer, AutoModel
import chromadb
import torch

from utils.keywords import keywords, has_matching_keyword

# ---- Logging Setup ----
logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- Initialize components ----
logger.info("Initializing components...")

# Initialize models and client
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
llm = OllamaLLM(model="deepseek-v2:latest")
chroma_client = chromadb.PersistentClient(path="db/chroma_db")

# ---- Pydantic Models for API ----
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    status: str = "success"

# ---- Reuse your existing functions ----
class State(BaseModel):
    user_query: str
    document_fetch_response: str = ""
    final_response: str = ""

class KeywordDecisionResponse(BaseModel):
    response: bool

parser = PydanticOutputParser(pydantic_object=KeywordDecisionResponse)

def embed_text(text, model, tokenizer):
    logger.info(f"Embedding text: {text[:60]}...")
    try:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = model(**inputs)
        token_embeddings = model_output[0]
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        logger.info("Embedding generated successfully.")
        return pooled[0].numpy()
    except Exception as e:
        logger.error(f"Error during embedding: {e}")
        raise

def retrieve_embeddings(query, collection_name, top_k=5):
    logger.info(f"Retrieving embeddings for query: '{query}' from collection: '{collection_name}'")
    try:
        collection = chroma_client.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' found: {collection is not None}")
        query_embedding = embed_text(query, embedding_model, tokenizer)
        logger.info(f"Querying collection with embedding, expecting top {top_k} results...")
        try:
            results = collection.query(
                query_embeddings=[query_embedding.tolist()], n_results=top_k
            )
            logger.info(f"Raw Chroma results: {results}")
        except Exception as qerr:
            logger.error(f"Chroma query error: {qerr}", exc_info=True)
            return []
        if collection_name == "bible_embeddings":
            docs = results.get("metadatas", [[]])
            return docs
        else:
            docs = results.get("documents", [[]])
        if not docs or not docs[0]:
            logger.warning(f"No documents returned for collection '{collection_name}'.")
            return []
        logger.info(f"Top {top_k} docs retrieved from '{collection_name}': {docs[0]}")
        return docs[0]
    except Exception as e:
        logger.error(f"Error in retrieve_embeddings: {e}", exc_info=True)
        return []

def llm_generate(prompt):
    logger.info(f"Calling LLM with prompt: {prompt[:100]}...")
    try:
        response = llm.invoke(prompt)
        logger.info("LLM response received.")
        return response
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        return "Sorry, there was an error generating the response."

# ---- Node Functions ----
def node_fetch_doc_response(state: State):
    logger.info("node_fetch_doc_response started.")
    user_query = state.user_query
    doc_chunks = retrieve_embeddings(user_query, "pdf_embeddings", top_k=5)
    logger.info(f"Retrieved document chunks: {doc_chunks}")
    context = "\n".join(doc_chunks)
    prompt = (
        f"You are a Christian theology assistant.\n\n"
        f"Use **only** the following Christian documents to answer the question in detail:\n\n{context}\n\n"
        f"⚠️ Strict Instructions:\n"
        f"- **Do NOT include the phrase 'according to Christian teachings'.** Make the response human friendly \n"
        f"- Do NOT use any external knowledge, sources, or interpretations.\n"
        f"- Do NOT add anything that is not explicitly stated in the provided documents.\n"
        f"- Your answer must be fully grounded in and limited to the content above.\n"
        f"- Cite or reference the relevant parts of the documents when forming your answer.\n\n"
        f"- Include the **complete texts** of any Bible verses you reference. You may list them at the end of the answer for clarity.\n\n"
        f"Question: {user_query}\nAnswer:"
        
    )
    doc_response = llm_generate(prompt)
    logger.info("node_fetch_doc_response completed.")
    return {"document_fetch_response": doc_response}

def node_enhance_with_bible_refs(state: State):
    logger.info("node_enhance_with_bible_refs started.")
    user_query = state.user_query
    bible_chunks = retrieve_embeddings(user_query, "bible_embeddings", top_k=5)
    logger.info(f"Retrieved Bible chunks: {bible_chunks}")
    bible_context = bible_chunks
    prev_answer = state.document_fetch_response
    prompt = (
        f"You are a Christian theology assistant.\n\n"
        f"Below is an existing answer based on Christian teachings:\n\n{prev_answer}\n\n"
        f"Now improve this answer using **only** the following Bible passages:\n{bible_context}\n\n"
        f"⚠️ Important Instructions:\n"
        f"- **Do NOT include the phrase 'according to Christian teachings'.** Make the response human friendly \n"
        f"- Do NOT use any external Bible passages or verses not included above.\n"
        f"- Do NOT introduce any ideas, interpretations, or reasoning beyond what is explicitly found in the provided passages.\n"
        f"- Base the entire revised answer strictly and exclusively on the given context.\n"
        f"- Reference the specific Bible passages used.\n"
        f"Your response must remain within the scope of the provided content only."
    )
    final_response = llm_generate(prompt)
    logger.info("node_enhance_with_bible_refs completed.")
    return {"final_response": final_response}

def check_keywords(state:State):
    logger.info("In the keywords check route function")
    user_query = state.user_query
    prompt= (
        f"**You are a strict keyword evaluator.Your job is to check if any of the following keywords or even individual words from the keyword phrases appear in the **user_query**, and whether they relate to **Christian theology, Spirtuality, teachings OR NOT** **\n"
        f"**FORCED INSTRUCTIONS**"
        f"**Understand, reason and Analyze the user_query and the keywords."
        f"If any word from the keywords list appears in the user_query OR the user_query is contextually related to Christianity, Spirituality  or theological discussion, return `true`. Otherwise, for all the other unrelewant questions return `false`.** \n"
        f"Do **not** explain or add anything else.\n"
        f"**Example 1**\n User Query: 'What's the weather today?'\n Response: {{\"response\": false}}\n"
        f"**Example 2**\n User Query: 'How did Jesus respond to suffering?'\n Response: {{\"response\": true}}\n"
        f"**Example 3**\n User Query: 'Tell me about football'\n Response: {{\"response\": false}}\n"
        f"**Example 4**\n User Query: 'Explain the Holy Trinity'\n Response: {{\"response\": true}}\n"
        f"**Example 5**\n User Query: 'Hello'\n Response: {{\"response\": false}}\n\n"
        f" Here is the list of the keywords: {keywords} \n"
        f" ** This is the user_query you have to evaluate \n user_query: {user_query} **\n"
        + parser.get_format_instructions()
    )

    response = llm_generate(prompt)
    logger.info(response)
    # response = parser.parse(raw)
    logger.info(f"Original response: {response} with type {type(response)}")
    parsed = parser.parse(response)
    logger.info(f"Structured response: {parsed} with type {type(parsed)}")
    if parsed.response is True:
        return "doc_fetch"
    else:
        return "simple_llm"

def llm_node(state:State):
    logger.info("In the Simple llm node")
    user_query= state.user_query
    prompt=(f"{user_query}")
    final_response = llm_generate(prompt)
    return {"final_response": final_response}

# ---- Build and cache the graph ----
logger.info("Building LangGraph...")
graph = StateGraph(State)
graph.add_node("doc_fetch", node_fetch_doc_response)
graph.add_node("add_bible_refs", node_enhance_with_bible_refs)
graph.add_node("simple_llm",llm_node)

graph.add_conditional_edges(START, check_keywords)
# graph.add_edge(START, "doc_fetch")
graph.add_edge("doc_fetch", "add_bible_refs")
graph.add_edge("add_bible_refs", END)
compiled_graph = graph.compile()
logger.info("LangGraph built and compiled.")

class Pipeline:
    """
    Eden pipeline: invokes the LangGraph Christian Teachings agent directly.
    """
    def __init__(self):
        self.id = "eden"
        self.name = "eden"
        # self.type = "pipe"

    async def on_startup(self):
        logger.info(f"[Startup] Pipeline: {self.name}")
        pass

    async def on_shutdown(self):
        logger.info(f"[Shutdown] Pipeline: {self.name}")
        pass

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator[str, None, None]]:

        if user_message and user_message is not None:
            result = compiled_graph.invoke({"user_query": user_message})
            answer = result.get("final_response", "")
            print("final_response", answer)
            return answer
 
