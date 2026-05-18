from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

@tool
def triple(num:float) -> float:
    """Returns the triple of a number."""
    return num * 3

tools = [TavilySearch(max_results=1), triple]

llm = ChatOllama(model="qwen3:8b",temperature=0).bind_tools(tools)