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
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    def assistant_node(state: MessagesState):
        task_id = state.get("task_id", "UNKNOWN_TASK_ID")
        print('assistant_node_task_id: ', task_id)
        system_prompt = """You are a helpful general AI assistant for anwering questions using tools provided. 
        I will ask you a question. Always consider the tools provided to you and use them if they are relevant to the question.
        If you think a tool is relevant, use it. If you think multiple tools are relevant, use them all.
        If the question references an image, document, or video file, you should assume it is already attached and available for download using the {task_id}`.
        Do not invent or guess task_id; use the exact {task_id} provided in context.
        Report your thoughts step-by-step, and finish your answer with the following template:

        FINAL ANSWER: [YOUR FINAL ANSWER]

        YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
        Do not include additional commentary after the final answer line.
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

