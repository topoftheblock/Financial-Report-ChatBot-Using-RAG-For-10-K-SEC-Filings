import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.agent.state import AgentState
from src.agent.tools import get_researcher_tools, get_quant_tools
from src.agent.prompt import (
    get_orchestrator_prompt, get_researcher_prompt, 
    get_quant_prompt, get_reviewer_prompt
)

llm_orchestrator = ChatOpenAI(model="gpt-4o", temperature=0)
llm_reviewer = ChatOpenAI(model="gpt-4o", temperature=0)

researcher_agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini", temperature=0), get_researcher_tools())
quant_agent = create_react_agent(ChatOpenAI(model="gpt-4o", temperature=0), get_quant_tools())

def orchestrator_node(state: AgentState):
    user_query = state["messages"][0].content
    response = llm_orchestrator.invoke([get_orchestrator_prompt(), HumanMessage(content=user_query)])
    return {"current_plan": response.content, "iteration_count": state.get("iteration_count", 0) + 1}

def researcher_node(state: AgentState):
    user_query = state["messages"][0].content
    plan = state["current_plan"]
    feedback = state.get("reviewer_feedback", "")
    
    # Inject the feedback dynamically into the prompt
    inputs = {"messages": [get_researcher_prompt(plan, feedback), HumanMessage(content=user_query)]}
    result = researcher_agent.invoke(inputs)
    
    return {"retrieved_context": [result["messages"][-1].content]}

def quant_node(state: AgentState):
    user_query = state["messages"][0].content
    context = "\n\n".join(state.get("retrieved_context", []))
    inputs = {"messages": [get_quant_prompt(context), HumanMessage(content=user_query)]}
    result = quant_agent.invoke(inputs)
    return {"calculation_results": [result["messages"][-1].content]}

def reviewer_node(state: AgentState):
    user_query = state["messages"][0].content
    context = "\n\n".join(state.get("retrieved_context", []))
    calcs = "\n\n".join(state.get("calculation_results", []))
    
    response = llm_reviewer.invoke([get_reviewer_prompt(context, calcs), HumanMessage(content=user_query)])
    response_text = response.content
    
    current_iterations = state.get("iteration_count", 0)
    
    # Check if REWORK was triggered and extract the feedback
    if "REWORK" in response_text.upper():
        # Split by the word REWORK and grab whatever the Reviewer wrote after it
        feedback = response_text.split("REWORK", 1)[-1].strip(" :")
        if not feedback:
            feedback = "The previous retrieval was incorrect. Please adjust your parameters."
            
        return {
            "needs_rework": True, 
            "final_answer": "", 
            "iteration_count": current_iterations + 1,
            "reviewer_feedback": feedback
        }
    
    return {
        "needs_rework": False, 
        "final_answer": response_text,
        "iteration_count": current_iterations + 1,
        "reviewer_feedback": "" # Clear the feedback on success
    }
