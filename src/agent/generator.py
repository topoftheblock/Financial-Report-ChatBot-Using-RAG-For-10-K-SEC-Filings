import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from src.agent.tools import get_financial_tools
from src.agent.prompt import get_agent_prompt

class FinancialRAGAgent:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initializes the Financial Agent Executor.
        Temperature is set to 0.0 to ensure deterministic, factual responses.
        """
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.tools = get_financial_tools()
        self.prompt = get_agent_prompt()
        
        # Create the agent that knows how to use OpenAI's tool-calling feature
        self.agent = create_tool_calling_agent(
            llm=self.llm, 
            tools=self.tools, 
            prompt=self.prompt
        )
        
        # The executor runs the ReAct loop (Observe -> Think -> Act)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True, # Set to True to see the thought process in the terminal
            return_intermediate_steps=True, # Useful for UI (showing tool usage)
            max_iterations=5 # Prevent infinite loops
        )

    def query(self, user_question: str) -> dict:
        """
        Takes a user question and executes the agent pipeline.
        
        Returns:
            A dictionary containing the 'output' (final answer) 
            and 'intermediate_steps' (the math and search steps taken).
        """
        try:
            response = self.agent_executor.invoke({"input": user_question})
            return response
        except Exception as e:
            # Phase 8: Here is where you would integrate src.utils.logger
            print(f"Agent Execution Error: {str(e)}")
            return {"output": "I encountered an error processing your financial query."}

# --- Example Usage for app/main.py ---
if __name__ == "__main__":
    # Ensure you have your .env loaded with OPENAI_API_KEY
    agent_app = FinancialRAGAgent()
    
    test_query = "What was AAPL's revenue in 2023, and what was the percentage growth compared to 2022?"
    
    print("User Query:", test_query)
    print("-" * 50)
    
    result = agent_app.query(test_query)
    
    print("-" * 50)
    print("Final Answer:\n", result["output"])