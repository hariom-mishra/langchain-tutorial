from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

load_dotenv()

class ChatbotState(TypedDict):
    #here annotated provides meta data to the state that
    # it will reciever list of messages and when new data comes
    # it should be added not replaced
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatOpenAI()

graph = StateGraph(ChatbotState)

def chat_node(state: ChatbotState) -> ChatbotState:
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


#define node
graph.add_node("chat_node", chat_node)

#define edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile()

initial_state = {"messages": [HumanMessage("What is the capital of France?")]}

output = workflow.invoke(initial_state)

print(output)