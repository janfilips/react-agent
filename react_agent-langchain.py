import logging
import re
from datetime import datetime
from typing import Any, Dict

from langchain.agents import AgentType, initialize_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.schema import SystemMessage
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI

from backend import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# XXX TODO leaverage chat history to provide context for the agent


# Tool Functions (unchanged)
def search_tool(query: str) -> str:
    return f"Results for '{query}'"


def calc_tool(expression: str) -> str:
    percent_match = re.match(r'^\s*(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)\s*$', expression)
    if percent_match:
        perc = float(percent_match.group(1))
        val = float(percent_match.group(2))
        return str((perc / 100) * val)
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def query_insurance_terms_tool(term: str) -> str:
    return f"Definition of '{term}'"


def query_insurance_products_tool() -> str:
    """Return all products from partner insurance companies."""
    return "List of all insurance products from partner companies"


# Tools Configuration (unchanged)
TOOLS_CONFIG: Dict[str, Dict[str, Any]] = {
    "search": {
        "func": search_tool,
        "description": "Use for looking up things on the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "what to search"}},
            "required": ["query"],
        },
    },
    "calculator": {
        "func": calc_tool,
        "description": "Use for simple math calculations",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "math expression"}},
            "required": ["expression"],
        },
    },
    "query_insurance_terms": {
        "func": query_insurance_terms_tool,
        "description": "Use for looking up definitions of insurance-specific terms",
        "parameters": {
            "type": "object",
            "properties": {"term": {"type": "string", "description": "the insurance term to define"}},
            "required": ["term"]
        }
    },
    "query_insurance_products": {
        "func": query_insurance_products_tool,
        "description": "Use for retrieving all insurance products from partner companies",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
}


class ReactAgent:
    def __init__(self, model: str = config.OPENAI_DEFAULT_MODEL, temperature: float = 0.2, max_calls: int = 10, verbose_mode: bool = True):
        self.model = model
        self.temperature = temperature
        self.max_calls = max_calls
        self.verbose = verbose_mode
        # System instruction
        system_prompt = (
            "You are an expert travel insurance specialist. You possess deep knowledge of travel insurance policies, coverage options, "
            "claim procedures, and regulations worldwide. When interacting with users, provide thorough, accurate guidance on travel insurance matters. "
            "Refrain from discussing subjects outside of the insurance scope, no personal topics or jokes—focus strictly on insurance. "
            "Think step by step, call the available tools as needed to fetch information, or return a final answer. "
            f"Today is {datetime.now().strftime('%Y-%m-%d')}. You do not know what time it is, so do not mention time."
            "Shorter responses are preferred."
            f"Respond in {config.AI_LANGUAGE} language."
        )
        # Maximum number of iterations/tool calls
        self.max_calls = self.max_calls

        # Set up callbacks for streaming and tracing
        stream_handler = StreamingStdOutCallbackHandler()
        tracer = LangChainTracer()
        callbacks = [stream_handler, tracer]

        # Build LangChain tools using StructuredTool
        tools = [
            StructuredTool.from_function(
                func=info["func"],
                name=name,
                description=info["description"],
                args_schema=info["parameters"],
            )
            for name, info in TOOLS_CONFIG.items()
        ]

        # Initialize LLM and AgentExecutor
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            callbacks=callbacks,  # LLM-level callbacks
        )
        self.agent_executor = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=self.verbose,
            max_iterations=self.max_calls,  # Limit iterations
            callbacks=callbacks,  # Enable streaming and tracing
            agent_kwargs={"system_message": SystemMessage(content=system_prompt)},
        )

    def run(self, user_input: str) -> str:
        """Run the agent with a configured trace name for proper grouping in LangSmith."""
        config = {"callbacks": self.agent_executor.callbacks, "run_name": "AgentExecutorRun"}
        result = self.agent_executor.invoke({"input": user_input}, config=config)
        return result["output"]

    async def stream(self, user_input: str):
        """Async generator to stream tool observations and final output, grouped under a single LangSmith run."""
        # Delegate through run() so that invoke() is called with run_name for proper trace grouping
        content = self.run(user_input)
        yield {"content": content}


if __name__ == "__main__":
    agent = ReactAgent()

    answer = agent.run(
        "Search for new insurance laws for this month."
    )
    print(answer)

    answer = agent.run(
        "List all insurance products available from partner companies."
    )
    print(answer)

    answer = agent.run(
        "Define the insurance term 'deductible' for me, and then calculate 20% of a 1500 claim amount."
    )
    print(answer)

    answer = agent.run(
        "Define the insurance term ‘co-payment’, then list all partner insurance products, and calculate what 12% of a 2,500 claim amount would be."
    )
    print(answer)
