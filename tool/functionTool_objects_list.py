from agents import Agent, set_tracing_disabled, function_tool
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


#----------------------------------change tool name and describtion-------------------------------------------------

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


#-----------------------------------agent------------------------------------------------

agent = Agent(
    name="assiatant",
    instructions="you are helpful agent",
    tools=[add, multiple],
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
)

#---------------------------------show functionTool objects list--------------------------------------------------

print(agent.tools)

