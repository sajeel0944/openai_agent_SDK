from typing import Any
from agents import Agent, FunctionTool, RunContextWrapper, Runner, set_tracing_disabled, ModelSettings, function_tool
from agents.extensions.models.litellm_model import LitellmModel
import os
from dotenv import load_dotenv
import litellm
from pydantic import BaseModel

#-----------------------------------------------------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)

#------------------------------------------------------------------------------------

# 🔕 output main litellm ki kuch warning arahe thi is sy warning nhi aye gi
litellm.disable_aiohttp_transport=True

#-----------------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini/gemini-2.0-flash"

#----------------------------------custom funtion tool-------------------------------------------------

#  Yeh function data ko accept karta hai aur simply print kar ke wapas return karta hai
def do_work_complete(data: str) -> str:
    print("calleld is do_work_complete")
    return(data) # Data return karta hai

# Yeh model define karta hai 2 numbers ko validate karne ke liye
class AddType(BaseModel):
    num1: int
    num2: int

# Yeh async function tool ke through call hota hai
async def run_function(context: RunContextWrapper[Any], args: dict) -> str:
    print(f"called is run_function")
    print(f"this is args {args}") # is main object aye ga us main num1 or num2 hai
    parsed = AddType.model_validate_json(args) # JSON se arguments ko parse karta hai aur AddType model se validate karta hai
    return do_work_complete(data=f"the answer is  {parsed.num1 + parsed.num2} ") # Num1 + Num2 ka addition karke result return karry ga do_some_work wali function ko

add = FunctionTool(
    name="addition",
    description="addition two number",
    params_json_schema=AddType.model_json_schema(),
    on_invoke_tool=run_function,
)

#------------------------------------agent-----------------------------------------------

# Ye tool define karta hai ke addition ka function AI agent use kar sake
agent = Agent(
    name="assiatant",
    instructions="you are helpful agent",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    model_settings=ModelSettings(tool_choice="required"), # # is sy tool zarol use karry ga
    tool_use_behavior="stop_on_first_tool",  # llm add tool call karry ga lakin add tool wrong answer return lakin wo wrong answer llm ky pass nhi jaye ga 
    tools=[add]
)

#------------------------------------show functionTool objects list----------------------

print(agent.tools)
print("\n\n")

#------------------------------------Runner-----------------------------------------------

reponse = Runner.run_sync(agent, "2 + 2")
print(reponse.final_output)
