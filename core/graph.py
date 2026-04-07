from langgraph.graph import StateGraph, END
from core.state import AgentState
from agents.planner import planner_node
from agents.retrieval import retrieval_node
from agents.analysis import analysis_node
from agents.synthesis import synthesis_node
from agents.reflector import reflector_node, should_continue
from agents.memory_agent import inject_memories, extract_memories, summarize_context


def build_graph(adaptive: bool = True, use_memory: bool = True):
    """
    Build the research pipeline graph.

    Args:
        adaptive: If True, includes reflector node that can loop back
                  for more papers when gaps are detected.
        use_memory: If True, includes memory injection and extraction nodes.
    """
    graph = StateGraph(AgentState)

    # Core nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("synthesis", synthesis_node)

    if use_memory:
        # Memory nodes
        graph.add_node("memory_inject", inject_memories)
        graph.add_node("memory_extract", extract_memories)
        graph.add_node("summarize", summarize_context)

        # Start with memory injection
        graph.set_entry_point("memory_inject")
        graph.add_edge("memory_inject", "planner")
    else:
        graph.set_entry_point("planner")

    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "analysis")

    if adaptive:
        # Add reflector node with conditional routing
        graph.add_node("reflector", reflector_node)

        if use_memory:
            # Summarize context before reflection if needed
            graph.add_edge("analysis", "summarize")
            graph.add_edge("summarize", "reflector")
        else:
            graph.add_edge("analysis", "reflector")

        graph.add_conditional_edges(
            "reflector",
            should_continue,
            {
                "retrieval": "retrieval",  # Loop back for more papers
                "synthesis": "synthesis",   # Proceed to final report
            }
        )
    else:
        # Simple linear pipeline
        graph.add_edge("analysis", "synthesis")

    if use_memory:
        # Extract memories after synthesis
        graph.add_edge("synthesis", "memory_extract")
        graph.add_edge("memory_extract", END)
    else:
        graph.add_edge("synthesis", END)

    return graph.compile()