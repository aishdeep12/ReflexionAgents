from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from reflexion import ReflexionAgent
from schemas import ReviseAnswer
from tool_executor import execute_tools

load_dotenv()

EXECUTE_TOOLS = "execute_tools"
AGENT_REASON = "agent_reason"
AGENT_REVISE = "agent_revise"
MAX_ITERATIONS = 2
LAST = -1

agent = ReflexionAgent()



# def should_continue(state: MessagesState):
#     count_tool_visits = sum(
#         isinstance(item, ToolMessage) for item in state["messages"]
#     )

#     if count_tool_visits >= MAX_ITERATIONS:
#         return END

#     last_message = state["messages"][-1]
#     if getattr(last_message, "tool_calls", None):
#         return EXECUTE_TOOLS

#     return END
def should_continue(state: MessagesState):
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )

    if count_tool_visits >= MAX_ITERATIONS:
        return END

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if not tool_calls:
        return END

    tool_names = [tc["name"] for tc in tool_calls]

    if "ReviseAnswer" in tool_names:
        return END

    return EXECUTE_TOOLS

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, agent.run_agent_reasoning)
flow.add_node(EXECUTE_TOOLS, execute_tools)
flow.add_node(AGENT_REVISE, agent.run_agent_revision)
flow.add_edge(START, AGENT_REASON)
flow.add_edge(AGENT_REASON, "execute_tools")
flow.add_edge("execute_tools", AGENT_REVISE)
flow.add_conditional_edges(
    AGENT_REVISE,
    should_continue,
    {
        EXECUTE_TOOLS: EXECUTE_TOOLS,
        END: END,
    },
)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="agent_flow.png")
print("Flow graph drawn to agent_flow.png")


def main():
    question = "What is the capital of France?"

    result = app.invoke({
        "messages": [HumanMessage(content=question)]
    })

    for i, msg in enumerate(result["messages"], start=1):
        print(f"\n--- Message {i} ---")
        print(type(msg).__name__)
        print(msg)

        if getattr(msg, "tool_calls", None):
            print("Tool calls:")
            for tc in msg.tool_calls:
                print(tc)


if __name__ == "__main__":
    main()