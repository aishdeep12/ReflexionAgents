import datetime

from langchain_core.output_parsers.openai_tools import (JsonOutputToolsParser, PydanticToolsParser)
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langgraph.graph import MessagesState
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from schemas import AnswerQuestionInput, ReviseAnswer

class ReflexionAgent:
    def __init__(self):
        self.parser = JsonOutputToolsParser(return_id=True)
        self.parser_pydantic = PydanticToolsParser(tools=[AnswerQuestionInput])
        self.actor_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that can use tools to answer questions.
             "\n1. instruction:{first_instructions}"
             "\n2. Reflect and critique your answer before responding."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        self.revisor_prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are revising your previous answer.

        You MUST respond by calling the aapropriate tools where needed.
        Do not write a normal text response.
        Do not leave the response empty.
        Do not say you cannot comply.

        Use the full conversation history to identify the earlier answer and its weaknesses.
        Then call the ReviseAnswer tool with:
        - revised_answer: a better final answer
        - improvements: a short explanation of what was improved"""
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        self.llm = ChatOllama(model="qwen3:8b", temperature=0)
    

    @tool
    def search_web(self,query: str) -> str:
        """Search the web for the given query and return a concise answer."""
        search_tool = TavilySearch(max_results=1)
        result = search_tool.run(query)
        return result

    def run_agent_reasoning(self, state: MessagesState) -> dict:
        """Run the agent reasoning process."""
        self.first_responder_prompt_template = self.actor_prompt_template.partial(
            first_instructions="Answer the question as best as you can."
        )
        self.first_responder = self.first_responder_prompt_template | self.llm.bind_tools(tools=[AnswerQuestionInput])
        response = self.first_responder.invoke({"messages": state["messages"]})
        return {"messages": [response]}
    
    def run_agent_revision(self, state: MessagesState) -> dict:
        chain = self.revisor_prompt_template | self.llm.bind_tools([self.search_web, ReviseAnswer])
        response = chain.invoke({"messages": state["messages"]})
        return {"messages": [response]}
