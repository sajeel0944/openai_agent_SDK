from dataclasses import dataclass
from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool, set_tracing_disabled, handoff
from agents.run import RunConfig
import os 
from dotenv import load_dotenv
from agents.extensions import handoff_filters

#----------------------------------------------------------------

set_tracing_disabled(disabled=True)
load_dotenv()

#----------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL : str = "gemini-2.5-flash"

#----------------------------------------------------------------

external_client = AsyncOpenAI(
    api_key = GEMINI_API_KEY,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = MODEL,
    openai_client = external_client 
)

config = RunConfig(
    model = model,
    model_provider = external_client ,
    tracing_disabled = True
)

# ----------------------------------------------------------------------------------------------------------

@function_tool
def addition(num1: int | float, num2: int | float) -> str:
    """
    This is an addition tool.

    Args:
        num1: An integer or float.
        num2: An integer or float.

    Returns:
         The result of num1 + num2
    """
    return f"This is calculator agent answer: {num1 + num2}"


@function_tool
def multiplication(num1: int | float, num2: int | float) -> str:
    """
    This is a multiplication tool.

    Args:
        num1: An integer or float.
        num2: An integer or float.

    Returns:
        The result of num1 * num2
    """
    return f"This is calculator agent answer: {num1 * num2}"


@function_tool
def division(num1: int | float, num2: int | float) -> str:
    """
    This is a division tool.

    Args:
        num1: An integer or float.
        num2: An integer or float.

    Returns:
         The result of num1 / num2, or error if division by zero
    """
    if num2 == 0:
        return "Error: Division by zero."
    return f"This is calculator agent answer: {num1 / num2}"


@function_tool
def subtraction(num1: int | float, num2: int | float) -> str:
    """
    This is a subtraction tool.

    Args:
        num1: An integer or float.
        num2: An integer or float.

    Returns:
        The result of num1 - num2
    """
    return f"This is calculator agent answer: {num1 - num2}"


# ----------------------------------------------------------------------------------------------------------

calculator_agent = Agent(
    name="calculator",
    instructions="A simple calculator agent that can perform basic arithmetic operations.",
    model=model,
    tool_use_behavior="stop_on_first_tool",
    tools=[addition, multiplication, division, subtraction],
    handoff_description="Calculator agent can perform basic arithmetic operations like addition, subtraction, multiplication, and division.",
)

# ----------------------------------------------------------------------------------------------------------

print("\nagent_1\n\n")

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. You can answer questions and perform tasks.",
    model=model,
    handoffs=[
        handoff(
            agent=calculator_agent,
            input_filter=handoff_filters.remove_all_tools
        ),
    ]
)

response_1 = Runner.run_sync(agent, "What is 2+2?")
print(response_1.final_output)

# ----------------------------------------------------------------------------------------------------------

print("\n\nagent_2\n\n")

agent_2 = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. You can answer questions and perform tasks.",
    model=model,
    handoffs=[
        handoff(agent=calculator_agent),
    ]
)

response_2 = Runner.run_sync(agent_2, "What is 2+2?")
print(response_2.final_output)
