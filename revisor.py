from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, Pydantic
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState

revisor_instructions="""Revise your previous answer to the question, taking into account the following feedback:
1. The question was: {question}
2. Your previous answer was: {answer}
3. you should use the critique to revise your answer and provide a better answer to the question.
4. Critique: {critique}
"""