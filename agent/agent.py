import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from openai import OpenAI
from agent.tools_used import TOOLS
import time 

#client = OpenAI()
load_dotenv()

def create_graph():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0, max_retries=3)
    llm_with_tools = llm.bind_tools(TOOLS)

    def assistant_node(state: MessagesState):
        
        with open("agent/system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

        messages = state["messages"]

        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages
        time.sleep(2)

        return {"messages": [llm_with_tools.invoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()

