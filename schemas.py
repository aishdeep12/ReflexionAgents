from typing import List, Optional
from pydantic import BaseModel, Field

class AnswerQuestionInput(BaseModel):
    answer: str = Field(description="The agent's answer to the question. approx 250 words.")
    search_query: Optional[str] = Field(description="A search query to use if the agent needs to look up information to answer the question.")
    thoughts: str = Field(description="The agent's thoughts on the current state of the conversation.")
    criticism: str = Field(description="The agent's critique of its own answer.")

class ReviseAnswer(BaseModel):
    revised_answer: str = Field(
        description="An improved answer that fixes issues from the critique."
    )
    improvements: str = Field(
        description="Short explanation of what was improved in the revised answer."
    )