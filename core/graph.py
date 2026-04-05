from langgraph.graph import StateGraph, END
from core.state import AgentState
from agents.planner import planner_node
from agents.retrieval import retrieval_node
from agents.analysis import analysis_node
from agents.synthesis import synthesis_node

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "analysis")
    graph.add_edge("analysis", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()