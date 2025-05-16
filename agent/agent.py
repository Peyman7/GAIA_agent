import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from openai import OpenAI
from agent.tools_used import TOOLS


#client = OpenAI()
load_dotenv()

def create_graph():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    llm_with_tools = llm.bind_tools(TOOLS)

    def assistant_node(state: MessagesState):
        
        system_prompt = """You are a helpful general AI assistant for anwering questions using tools provided. 
        I will ask you a question. Always consider the tools provided to you and use them if they are relevant to the question.
        If you think a tool is relevant, use it. 
        If the question has an attachment file as image, document, or video file, you should try to download it. 
        Do not invent or guess task_id; use the exact `task_id` provided in context.
        Report your thoughts step-by-step, and finish your answer with the following template:

        FINAL ANSWER: [YOUR FINAL ANSWER]

        YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
        IMPORTANT: DO NOT include any units such as "USD", "$", "percent", "%", or currency names in your FINAL ANSWER unless the question explicitly asks for it.

        The FINAL ANSWER must always be:
            - A raw number (e.g., 89705, 4)
            - A few words (e.g., big ripe strawberries)
            - Or a comma-separated list (e.g., salt, big ripe strawberries, sugar)

        Your answer must ONLY start with:
        FINAL ANSWER: [your answer with NO additional text or units]
        """

        messages = state["messages"]

        # Only insert the system prompt if not already present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages

        return {"messages": [llm_with_tools.invoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()

