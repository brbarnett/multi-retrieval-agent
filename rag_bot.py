import os
from typing import Any, Dict, Literal, Sequence
from langchain_openai import AzureChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph


class RagBot:
    model: AzureChatOpenAI
    bot: CompiledStateGraph

    def __init__(self, tools: Sequence[Dict[str, Any]]) -> None:
        load_dotenv()

        self.model = AzureChatOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version="2024-02-01",
            azure_deployment=os.environ["OPENAI_API_DEPLOYMENT"],
            azure_endpoint=os.environ["OPENAI_API_ENDPOINT"],
            streaming=True,
            temperature=0,
            verbose=False,
        ).bind_tools(tools)

        workflow = StateGraph(MessagesState)

        tool_node = ToolNode(tools)

        # define the two nodes we will cycle between
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", tool_node)

        workflow.add_edge("__start__", "agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
        )
        workflow.add_edge("tools", "agent")

        memory = MemorySaver()

        # compile bot
        self.bot = workflow.compile(checkpointer=memory)

    def should_continue(self, state: MessagesState) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    def call_model(self, state: MessagesState):
        messages = state["messages"]
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def invoke(self, query: str, thread_id: str):
        response = self.bot.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}},
        )

        return response
