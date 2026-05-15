from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# LLM
# ----------------------------

llm = ChatOpenAI(model="gpt-4o-mini")

# ----------------------------
# Sample Docs
# ----------------------------

docs = [
    Document(page_content="Python is used for AI and machine learning."),
    Document(page_content="LangGraph is used to build agent workflows."),
    Document(page_content="FAISS is a vector database for similarity search."),
]

# ----------------------------
# Vector Store
# ----------------------------

embedding_model = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(
    docs,
    embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ----------------------------
# State
# ----------------------------

class RagState(TypedDict):

    question: str
    rewritten_question: str
    documents: list
    relevant: bool
    answer: str

# ----------------------------
# Retrieve Node
# ----------------------------

def retrieve(state: RagState):

    query = state.get("rewritten_question") or state["question"]

    docs = retriever.invoke(query)

    return {
        "documents": docs
    }

# ----------------------------
# Relevance Check
# ----------------------------

def check_relevance(state: RagState):

    docs_text = "\n".join(
        [doc.page_content for doc in state["documents"]]
    )

    prompt = f"""
You are checking retrieval quality.

Question:
{state["question"]}

Retrieved Docs:
{docs_text}

Are these docs relevant enough to answer the question?

Reply only:
YES or NO
"""

    result = llm.invoke(prompt).content.strip().upper()

    return {
        "relevant": result == "YES"
    }

# ----------------------------
# Rewrite Query
# ----------------------------

def rewrite_query(state: RagState):

    prompt = f"""
Rewrite this query to improve retrieval.

Query:
{state["question"]}
"""

    rewritten = llm.invoke(prompt).content

    return {
        "rewritten_question": rewritten
    }

# ----------------------------
# Generate Answer
# ----------------------------

def generate_answer(state: RagState):

    context = "\n".join(
        [doc.page_content for doc in state["documents"]]
    )

    prompt = f"""
Answer using the provided context only.

Context:
{context}

Question:
{state["question"]}
"""

    answer = llm.invoke(prompt).content

    return {
        "answer": answer
    }

# ----------------------------
# Conditional Routing
# ----------------------------

def route_decision(state: RagState):

    if state["relevant"]:
        return "generate"

    return "rewrite"

# ----------------------------
# Build Graph
# ----------------------------

builder = StateGraph(RagState)

builder.add_node("retrieve", retrieve)
builder.add_node("check_relevance", check_relevance)
builder.add_node("rewrite", rewrite_query)
builder.add_node("generate", generate_answer)

builder.add_edge(START, "retrieve")

builder.add_edge("retrieve", "check_relevance")

builder.add_conditional_edges(
    "check_relevance",
    route_decision,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

builder.add_edge("rewrite", "retrieve")

builder.add_edge("generate", END)

graph = builder.compile()

# ----------------------------
# Run
# ----------------------------

result = graph.invoke({
    "question": "What is LangGraph?"
})

print(result["answer"])
