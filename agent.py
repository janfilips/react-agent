import json
import logging
from typing import Any, Dict, List, Optional

import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = openai.OpenAI()


def search_tool(query: str) -> str:
    # … call your search API …
    return f"Results for '{query}'"


def calc_tool(expression: str) -> str:
    # … evaluate math expression safely …
    return str(eval(expression))


TOOLS: Dict[str, Dict[str, Any]] = {
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
}


class ReactAgent:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": (
                "You are an agent that reasons step by step. "
                "Think, then call a tool if needed via a function, "
                "or return a final answer."
            )}
        ]

    def should_continue(self, resp: openai.ChatCompletion) -> bool:
        # continue if model asks for a function call
        fc = resp.choices[0].finish_reason == "function_call"
        return fc

    def select_tool(self, resp: openai.ChatCompletion) -> Optional[Dict[str, Any]]:
        choice = resp.choices[0]
        if choice.finish_reason == "function_call":
            name = choice.message.function_call.name
            args = json.loads(choice.message.function_call.arguments)
            tool = TOOLS.get(name)
            return {"tool": tool, "args": args} if tool else None
        return None

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        # protection against infinite loops
        max_calls = 5
        call_count = 0

        while True:
            resp = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=self.messages,
                functions=[
                    {
                        "name": name,
                        "description": info["description"],
                        "parameters": info["parameters"],
                    }
                    for name, info in TOOLS.items()
                ],
                function_call="auto",
            )
            fr = resp.choices[0].finish_reason
            # Debug: log model response finish_reason and potential function call
            logger.info(f"finish_reason: {fr}")
            if fr == "function_call":
                fname = resp.choices[0].message.function_call.name
                fargs = resp.choices[0].message.function_call.arguments
                logger.info(f"Model requested function call: {fname} with args {fargs}")

            if self.should_continue(resp):
                # count and guard function calls
                call_count += 1
                if call_count > max_calls:
                    raise RuntimeError("Maximum function call limit reached.")
                sel = self.select_tool(resp)
                if not sel:
                    raise RuntimeError("Model asked for unknown tool")
                # Debug: about to execute the selected tool
                logger.info(f"Executing tool function for {fname}")
                # call the tool
                result = sel["tool"]["func"](**sel["args"])
                logger.info(f"Tool result: {result}")
                # append the function response for the next turn
                self.messages.append({
                    "role": "function",
                    "name": resp.choices[0].message.function_call.name,
                    "content": result,
                })
                continue

            # no more function calls → final answer
            final = resp.choices[0].message.content
            return final


if __name__ == "__main__":
    agent = ReactAgent()
    answer = agent.run("What’s 42 times 17, and then find me the top news about Python?")
    print(answer)
