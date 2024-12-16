"""Module for creating and managing the financial analysis workflow graph."""

from typing import Annotated, TypedDict

# import mlflow
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# mlflow.langchain.autolog()


class State(TypedDict):
    """
    State dictionary for the workflow graph.

    Attributes:
        messages (List[Message]): List of messages in the conversation.
    """

    messages: Annotated[list, add_messages]


class FinancialAnalysisGraph:
    """
    Class for creating and managing the financial analysis workflow graph.
    """

    @classmethod
    def create_graph(
        cls,
        generate_agent,
        reflect_agent,
    ):
        """
        Create a state graph for the financial analysis workflow.

        Args:
            generate_agent: Agent responsible for generating analysis.
            reflect_agent: Agent responsible for critiquing analysis.

        Returns:
            Compiled graph workflow.
        """

        async def generation_node(state: State) -> State:
            """
            Node for generating financial analysis.

            Args:
                state (State): Current workflow state.

            Returns:
                State: Updated workflow state with generated message.
            """
            return {"messages": [await generate_agent.ainvoke(state["messages"])]}

        async def reflection_node(state: State) -> State:
            """
            Node for reflecting on and critiquing the generated analysis.

            Args:
                state (State): Current workflow state.

            Returns:
                State: Updated workflow state with reflection message.
            """
            # Translate messages for reflection
            cls_map = {"ai": HumanMessage, "human": AIMessage}
            translated = [state["messages"][0]] + [
                cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
            ]
            res = await reflect_agent.ainvoke(translated)
            return {"messages": [HumanMessage(content=res.content)]}

        def should_continue(state: State):
            """
            Determine whether to continue the workflow.

            Args:
                state (State): Current workflow state.

            Returns:
                str: Next node or END marker.
            """
            if (len(state["messages"]) > 6) or (
                "ALL_OK" in state["messages"][-2].content
            ):
                return END
            return "reflect"

        # Create and configure graph
        builder = StateGraph(State)
        builder.add_node("generate", generation_node)
        builder.add_node("reflect", reflection_node)
        builder.add_edge(START, "generate")
        builder.add_conditional_edges("generate", should_continue)
        builder.add_edge("reflect", "generate")

        # Compile graph with memory checkpoint
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)
