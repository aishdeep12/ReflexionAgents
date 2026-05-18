import datetime
from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import (JsonOutputKeyToolsParser,PydanticToolsParser)
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

actor_prompt_template = ChatPromptTemplate.from_messages([

    ("system", "You are a helpful assistant that can use tools to answer questions."),])
