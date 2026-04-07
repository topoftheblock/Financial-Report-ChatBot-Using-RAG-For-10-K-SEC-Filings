from typing import Dict, Any
from src.agent.build_the_langgraph_graphs import agent_graph

class FinancialRAGAgent:
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.graph = agent_graph
        self.config = {"configurable": {"thread_id": "streamlit_ui_session"}}

    def query(self, prompt: str) -> Dict[str, Any]:
        events = []
        
        # Reset the state at the start of every new query
        initial_state = {
            "messages": [("user", prompt)],
            "retrieved_context": [],
            "calculation_results": [],
            "iteration_count": 0,
            "reviewer_feedback": ""
        }
        
        try:
            for event in self.graph.stream(initial_state, config=self.config, stream_mode="updates"):
                events.append(event)
        except Exception as e:
            print(f"Error during graph execution: {e}")
            return {"output": f"Architecture error: {str(e)}", "intermediate_steps": events}
            
        final_answer = "I could not formulate a complete answer. Please try rephrasing."
        
        if events:
            last_event = events[-1]
            if "Reviewer" in last_event:
                if last_event["Reviewer"].get("final_answer"):
                    final_answer = last_event["Reviewer"]["final_answer"]
                elif last_event["Reviewer"].get("needs_rework"):
                    final_answer = "I hit my loop limit while trying to fix calculation errors. Please simplify your query."

        # --- AGENT DEBUG LOG ---
        # Prints the internal thought process of the Reviewer and Quant to the terminal
        print("\n--- AGENT DEBUG LOG ---")
        for event in events:
            if "Reviewer" in event:
                feedback = event["Reviewer"].get("reviewer_feedback", "")
                if feedback:
                    print(f"🕵️‍♂️ REVIEWER COMPLAINT: {feedback}")
            if "Quant" in event:
                calcs_list = event["Quant"].get("calculation_results", [])
                calcs = calcs_list[0] if calcs_list else "No calculations returned."
                print(f"🧮 QUANT OUTPUT: {calcs}")
        print("-----------------------\n")

        return {
            "output": final_answer,
            "intermediate_steps": events
        }