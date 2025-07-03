import asyncio
from typing import Any
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, RunHooks, RunContextWrapper, Tool, Usage, function_tool, set_tracing_disabled
from dotenv import load_dotenv
import os
from pydantic import BaseModel

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

#--------------------------------Example one--------------------------------------------

print("\nExample One\n")

class MytestData(BaseModel): # is ky andar user ka name or age aye gi
    name: str
    age: int


class MyCustomRunnerHook(RunHooks):
    async def on_agent_start(self, context: RunContextWrapper[MytestData], agent: Agent):
        print(f"starting run for agent: {agent.name} \nhook: {agent.hooks}, \ninstructions: {agent.instructions}, \nuser name: {context.context.name}, \nage: {context.context.age}\n")

    async def on_agent_end(self, context: RunContextWrapper[MytestData], agent: Agent, output):
        print(f"Run complete for agent: {agent.name}, output: {output}")


# ✅ Create instance properly
test_data = MytestData(name="sajeel", age=19)

my_agent = Agent(
    name="assistant",
    instructions="you are a helpful assistant",
    model=model
)

# ✅ Run agent with context 
response = Runner.run_sync(
    my_agent,
    input="hello, how are you",
    context=test_data,
    hooks=MyCustomRunnerHook()
)

print("Final Response:", response.final_output)


#--------------------------------Example two--------------------------------------------

print("\nExample Two\n")

class CustomRunnerHooks(RunHooks):
    def __init__(self):
        self.event_counter = 0

    def _usage_to_str(self, usage: Usage) -> str:
        return f"{usage.requests} requests, {usage.input_tokens} input tokens, {usage.output_tokens} output tokens, {usage.total_tokens} total tokens"

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


hooks = CustomRunnerHooks()

###


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
    model=model
)

start_agent = Agent(
    name="assistant",
    instructions="you are helpful agent",
    handoffs=[calculator_agent],
    model=model,
)


async def main() -> None:
    user_input = input("Enter Number: ")
    response = await Runner.run(
        start_agent,
        input=f"{user_input}.",
        hooks=hooks,
    )
    
    print(response.final_output)


if __name__ == "__main__":
    asyncio.run(main())



