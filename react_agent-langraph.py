from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.tools import Tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from backend import config

# Assuming TOOLS_CONFIG and config are defined elsewhere
TOOLS_CONFIG = {
    "search": {"func": lambda x: "search results", "description": "Search tool", "parameters": None},
    "calc": {"func": lambda x: "calc result", "description": "Calculator tool", "parameters": None},
    "query_insurance_terms": {"func": lambda x: "insurance term", "description": "Insurance terms tool", "parameters": None},
}


class ReactAgent:
    def __init__(self, model: str = config.OPENAI_DEFAULT_MODEL, temperature: float = 0.2):
        self.model = model
        self.temperature = temperature

        # Set up callbacks for streaming and tracing
        stream_handler = StreamingStdOutCallbackHandler()
        tracer = LangChainTracer()
        callbacks = [stream_handler, tracer]

        # Initialize LLM with callbacks
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            callbacks=callbacks,
        )

        # Define tools
        tools = [
            Tool(
                name=name,
                func=info["func"],
                description=info["description"],
                args_schema=info["parameters"],
            )
            for name, info in TOOLS_CONFIG.items()
        ]

        # Create the ReAct agent using LangGraph
        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=tools,
        )

    def run(self, user_input: str) -> str:
        """Run the agent with the given user input."""
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        result = self.agent_executor.invoke(initial_state)
        return result["messages"][-1].content

    async def stream(self, user_input: str):
        """Async generator to stream the agent's output."""
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        async for event in self.agent_executor.astream(initial_state):
            yield event


if __name__ == "__main__":
    agent = ReactAgent()
    answer = agent.run(
        "Define the insurance term 'deductible' for me, and then calculate 20% of a 1500 claim amount and search for what is new in python programming world?"
    )
    print(answer)
