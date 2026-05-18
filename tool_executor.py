from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from schemas import ReviseAnswer, AnswerQuestionInput

tavily_tool = TavilySearch(max_results=1)

def run_queries(search_query: str, **kwargs) -> str:
    """Run a list of search queries and return their results."""
    return tavily_tool.run(search_query)


execute_tools = ToolNode([

    StructuredTool.from_function(run_queries, name=AnswerQuestionInput.__name__),
    StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__)
])
