from agents import Agent, ModelSettings, Runner, set_tracing_disabled, function_tool, FunctionTool
from agents.extensions.models.litellm_model import LitellmModel
import os
from dotenv import load_dotenv
import litellm

#-----------------------------------------------------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)

#------------------------------------------------------------------------------------

# 🔕 output main litellm ki kuch warning arahe thi is sy warning nhi aye gi
litellm.disable_aiohttp_transport=True

#-----------------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini/gemini-2.0-flash"

#----------------------------------------change tool name and describtion-------------------------------------------------
print("\n\t\t\tChange tool name and description\n")

@function_tool(name_override="addition", description_override="this is addition tool")
def add(num1: int, num2: int) -> int:
    """
    add two number
    """
    print("called add tool")
    return num1 + num2

@function_tool
def multiple(num1: int, num2: int) -> int:
    """
    multiple two number
    """
    return num1 * num2

#-----------------agent------------------

agent = Agent(
    name="assiatant",
    instructions="you are helpful agent",
    tools=[add, multiple],
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
)

#--------show functionTool objects list--------

print(agent.tools)





#---------------------------------------------------is_enabled-------------------------------------------------------------
print("\n\n\t\t\tis_enabled\n")
# is_enabled by default True hota hai agar is sy False kar den to tool disable ho jata hai

@function_tool(is_enabled=False) # is_enabled=False sy tool disable ho jaye ga llm ko nazar he nhi aye ga or llm multiple tool call karry ga
def add(num1: int, num2: int) -> int:
    """
    add two number
    """
    print("called add tool")
    return f"Hello Your Answer = {num1 + num2}"

@function_tool(is_enabled=True) # is_enabled=True sy tool llm ko nazar aye ga
def multiple(num1: int, num2: int) -> int:
    """
    multiple two number
    """
    return f"Hello Your Answer = {num1 * num2}"

#-----------------agent------------------

agent_2 = Agent(
    name="assiatant",
    instructions="you are helpful agent",
    tools=[add, multiple],
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    model_settings=ModelSettings(tool_choice="required"), # is sy llm tool zarol use karry ga
    tool_use_behavior="stop_on_first_tool", # llm tool call karry ga lakin us ka output llm ky pass nhi jaye ga wo directly user ko answer dyga
)

print("\n\nTool not celled because is_enabled=False\n")
response = Runner.run_sync(agent_2, "what is 4 plus 4")
print(response.final_output)

print("\n\nTool called because is_enabled=True\n")
response_2 = Runner.run_sync(agent_2, "what is 4 multiple 4")
print(response_2.final_output)

