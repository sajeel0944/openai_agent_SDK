from agents import Agent, Runner, set_tracing_disabled, ModelSettings, function_tool
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

#----------------------------------tool-------------------------------------------------

@function_tool
def say_hello() -> str:
    """
    this is say hello tool
    """
    print("called say_hello tool")
    return ("hello user")


#-----------------------------------reset_tool_choice------------------------------------------------


agent = Agent(
    name="assiatant",
    instructions="you are helpful agent",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    tools=[say_hello],
    model_settings=ModelSettings(tool_choice="required"), # is sy tool zarol use karry ga
    reset_tool_choice=False # is sy tool 10 time chaly ga phir error aye ga
)

#--------------------------------Runner---------------------------------------------------

reponse = Runner.run_sync(agent, "hello")
print(reponse.final_output)

