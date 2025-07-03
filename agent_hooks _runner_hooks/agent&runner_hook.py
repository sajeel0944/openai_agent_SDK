import asyncio
from typing import Any
from agents import Agent, ModelSettings, RunHooks, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, AgentHooks, RunContextWrapper, Tool, Usage, function_tool, set_tracing_disabled
from dotenv import load_dotenv
import os

#----------------------------------------------------------------------------

set_tracing_disabled(disabled=True)
load_dotenv()

#----------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"  

#----------------------------------------------------------------------------

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=BASE_URL,
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client,
)

run_config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)
 

# ----------------------------------------RunHooks-----------------------------------------------------

class CustomRunnerHooks(RunHooks):
    def __init__(self):
        self.event_counter = 0

    def _usage_to_str(self, usage: Usage) -> str:
        return f"\n{usage.requests} requests, {usage.input_tokens} input tokens, {usage.output_tokens} output tokens, {usage.total_tokens} total tokens"

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Agent {agent.name} started. Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Agent {agent.name} ended with output {output}. Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Tool {tool.name} started. Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Tool {tool.name} ended with result {result}. Usage: {self._usage_to_str(context.usage)}"
        )

    async def on_handoff(
        self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent
    ) -> None:
        self.event_counter += 1
        print(
            f"### {self.event_counter}: Handoff from {from_agent.name} to {to_agent.name}. Usage: {self._usage_to_str(context.usage)}"
        )


# ----------------------------------------------AgentHooks------------------------------------------------

class CustomAgentHooks(AgentHooks):
    def __init__(self, display_name: str):
        self.event_counter = 0
        self.display_name = display_name

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(f"###\n ({self.display_name}) {self.event_counter}: Agent {agent.name} started")

    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended with output {output}"
        )

    async def on_handoff(self, context: RunContextWrapper, agent: Agent, source: Agent) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {source.name} handed off to {agent.name}"
        )

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started tool {tool.name}"
        )

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended tool {tool.name} with result {result}"
        )


runner_hooks = CustomRunnerHooks()

@function_tool
def multiply(num1: int, num2: int):
    """
    this is multiply agent
    num1 : int 
    num2 : int
    """
    return f"multiple {num1*num2}"


@function_tool
def addition(num1: int, num2: int):
    """
    this is addition agent
    num1 : int 
    num2 : int
    """
    return f"addition {num1*num2}"

@function_tool
def minus(num1: int, num2: int):
    """
    this is minus agent
    num1 : int 
    num2 : int
    """
    return f"minus {num1*num2}"

@function_tool
def division(num1: int, num2: int):
    """
    this is division agent
    num1 : int 
    num2 : int
    """
    return f"division {num1*num2}"


calculator_agent = Agent(
    name="calculator Agent",
    instructions="you are calculator agent",
    tools=[multiply, division, minus, addition],
    model_settings=ModelSettings(tool_choice="required"),
    hooks=CustomAgentHooks(display_name="calculate Agent"),
    model=model
)

start_agent = Agent(
    name="Assistant",
    instructions="you are helpful agent",
    handoffs=[calculator_agent],
    hooks=CustomAgentHooks(display_name="Start Agent"),
    model=model
)


async def main() -> None:
    user_input = input("Enter Number: ")
    response = await Runner.run(
        start_agent,
        input=f"{user_input}.",
        hooks=runner_hooks,
    )

    print(response.final_output)


if __name__ == "__main__":
    asyncio.run(main())
