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


#------------------------------------Forcing tool use----------------------------

#-----------------------------------auto------------------------------------------------

print("\t\t\tauto\n")

auto_agent = Agent(
    name="assiatant",
    instructions="you are helpful agent",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    model_settings=ModelSettings(tool_choice="auto"), # ye auto select karry ga ky tool use kara hai ya nhi
    tools=[say_hello]
)

reponse = Runner.run_sync(auto_agent, "hello")
print(reponse.final_output)

#-----------------------------------required------------------------------------------------

print("\n\t\t\trrequired\n")

required_agent = Agent(
    name="assiatant",
    instructions="you are helpful agent",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    model_settings=ModelSettings(tool_choice="required"), # is sy tool zarol use karry ga
    tools=[say_hello]
)

reponse = Runner.run_sync(required_agent, "hello")
print(reponse.final_output)


#-----------------------------------none------------------------------------------------

print("\n\t\t\tnone\n")

none_agent = Agent(
    name="assiatant",
    instructions="you are helpful agent",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    model_settings=ModelSettings(tool_choice="none"), # is sy tool use karry ga
    tools=[say_hello]
)

reponse = Runner.run_sync(none_agent, "hello")
print(reponse.final_output)