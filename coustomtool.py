import os
import psutil
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

@tool
def get_memory_usage() -> str:
    """Returns current RAM usage of the process in MB."""
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_mb = round(mem_bytes / (1024 ** 2), 2)
    return f" RAM usage: {mem_mb} MB"

@tool
def get_ai_education_summary() -> str:
    """Provides a brief summary about AI in education."""
    return (
        " AI in education is transforming learning with personalized tutoring, "
        "automated grading, and adaptive content delivery. Tools like ChatGPT and Gemini "
        "are being integrated into classrooms to help students learn more effectively."
    )
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="AIzaSyBooWQhuxQJOzJWA5JSQceLi9mGczM",
    convert_system_message_to_human=True,
    temperature=0.7
)

tools = [get_ai_education_summary, get_memory_usage]

agent = create_react_agent(llm, tools)

input_message = {
    "role": "user",
    "content": "Give me a summary about AI in education and also show RAM usage."
}

for step in agent.stream({"messages": [input_message]}, stream_mode="values"):
    step["messages"][-1].pretty_print()