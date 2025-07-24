# react-agent

**Simple yet powerful Reason-and-Act (ReAct) agent implemented in Python.**

## Overview

`react-agent` is a lightweight framework for building autonomous agents that follow the ReAct paradigm: interleaving **reasoning** (thinking) with **actions** (tool calls) to solve complex tasks. This agent leverages OpenAI's API to generate step-by-step reasoning and decide when to call external tools (e.g., web search, calculator).

## Features

- **Reasoning and Acting**: Uses ReAct to chain thought and actions.
- **Extensible Tools**: Define custom tools with descriptions and JSON parameters.
- **OpenAI Integration**: Seamlessly integrates with the `openai` Python package.
- **Easy to Use**: Minimal boilerplate to get started.
- **Debug Logging**: Built-in logging to trace thought and function calls.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/react-agent.git
   cd react-agent
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > **Note**: Ensure you have `openai` installed.

3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Basic Example

```python
from agent import ReactAgent

agent = ReactAgent(model="gpt-4o-mini", temperature=0.2)
response = agent.run("What's 42 times 17, and then find me the top news about Python?")
print(response)
```

### Tools

The agent supports custom tools defined in the `TOOLS` dictionary:

- **search**: Performs web search.
  - **Parameters**:
    - `query` (string): What to search.
- **calculator**: Evaluates simple math expressions.
  - **Parameters**:
    - `expression` (string): Math expression to evaluate.

#### Adding Custom Tools

```python
def custom_tool(data: str) -> str:
    # custom logic
    return "result"

TOOLS["custom"] = {
    "func": custom_tool,
    "description": "Describe your tool here",
    "parameters": {
        "type": "object",
        "properties": {
            "data": {"type": "string", "description": "input data"}
        },
        "required": ["data"],
    },
}
```

## Contributing

Contributions welcome! Feel free to open issues or submit pull requests.

## License

MIT License
