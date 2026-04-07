from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import Literal

from src.agent.state import AgentState
from src.agent.main_graph_nodes_and_edges import (
    orchestrator_node, researcher_node, quant_node, reviewer_node
)

checkpointer = InMemorySaver()

workflow = StateGraph(AgentState)

workflow.add_node("Orchestrator", orchestrator_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Quant", quant_node)
workflow.add_node("Reviewer", reviewer_node)

workflow.add_edge(START, "Orchestrator")
workflow.add_edge("Orchestrator", "Researcher")
workflow.add_edge("Researcher", "Quant")
workflow.add_edge("Quant", "Reviewer")

def crag_router(state: AgentState) -> Literal["Researcher", "END"]:
    if state.get("iteration_count", 0) >= 3:
        return "END"
    if state.get("needs_rework", False):
        return "Researcher"
    return "END"

workflow.add_conditional_edges("Reviewer", crag_router, {"Researcher": "Researcher", "END": END})

agent_graph = workflow.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    print("Generating LangGraph visualization...")
    with open("main_agent_graph.png", "wb") as f:
        f.write(agent_graph.get_graph().draw_mermaid_png())
    print("Graph saved to main_agent_graph.png")
