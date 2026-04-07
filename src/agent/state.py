from langgraph.graph import MessagesState
from typing import Annotated, List

def append_strings(existing: List[str], new: List[str]) -> List[str]:
    return existing + new

class AgentState(MessagesState):
    current_plan: str
    retrieved_context: Annotated[List[str], append_strings]
    calculation_results: Annotated[List[str], append_strings]
    final_answer: str
    needs_rework: bool
    iteration_count: int
    reviewer_feedback: str
