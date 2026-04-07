from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

# 1. Import your actual tools
from src.agent.tools import search_financial_tables, search_unstructured_text, get_financial_tools

# 2. Import state models
from src.agent.state import State, AgentState
# 3. Import all graph nodes and edges
from src.agent.main_graph_nodes_and_edges import (
    orchestrator,
    compress_context,
    fallback_response,
    should_compress_context,
    collect_answer,
    route_after_orchestrator_call,
    summarize_history,
    rewrite_query,
    request_clarification,
    aggregate_answers,
    route_after_rewrite
)

checkpointer = InMemorySaver()

# --- Agent Subgraph ---
agent_builder = StateGraph(AgentState)
agent_builder.add_node("orchestrator", orchestrator)
# Fixed: Use the actual tools you built instead of search_child_chunks
agent_builder.add_node("tools", ToolNode(get_financial_tools()))
agent_builder.add_node("compress_context", compress_context)
agent_builder.add_node("fallback_response", fallback_response)
agent_builder.add_node("should_compress_context", should_compress_context)
agent_builder.add_node("collect_answer", collect_answer)

agent_builder.add_edge(START, "orchestrator")
agent_builder.add_conditional_edges(
    "orchestrator", 
    route_after_orchestrator_call, 
    {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"}
)
agent_builder.add_edge("tools", "should_compress_context")
agent_builder.add_edge("compress_context", "orchestrator")
agent_builder.add_edge("fallback_response", "collect_answer")
agent_builder.add_edge("collect_answer", END)
agent_subgraph = agent_builder.compile()

# --- Main Graph ---
graph_builder = StateGraph(State)
graph_builder.add_node("summarize_history", summarize_history)
graph_builder.add_node("rewrite_query", rewrite_query)
graph_builder.add_node("request_clarification", request_clarification)
graph_builder.add_node("agent", agent_subgraph)
graph_builder.add_node("aggregate_answers", aggregate_answers)

graph_builder.add_edge(START, "summarize_history")
graph_builder.add_edge("summarize_history", "rewrite_query")
graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
graph_builder.add_edge("request_clarification", "rewrite_query")
graph_builder.add_edge(["agent"], "aggregate_answers")
graph_builder.add_edge("aggregate_answers", END)

agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])

# --- Visualisation Execution ---
if __name__ == "__main__":
    print("Generating LangGraph visualizations...")
    
    # Save the main graph
    with open("main_agent_graph.png", "wb") as f:
        f.write(agent_graph.get_graph().draw_mermaid_png())
        
    # Save the subgraph
    with open("subgraph.png", "wb") as f:
        f.write(agent_subgraph.get_graph().draw_mermaid_png())
        
    print("Graphs saved successfully to your current directory!")