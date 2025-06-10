from agents import Agent, Runner, set_tracing_disabled, ModelSettings, Usage
from agents.extensions.models.litellm_model import LitellmModel
import os
from dotenv import load_dotenv

#------------------------------------------------------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)

#------------------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini/gemini-2.0-flash"


#-------------------------------------------------------------------------------------------------------------------------
#                                                     max_tokens = 10
#-------------------------------------------------------------------------------------------------------------------------

print("\n\t\t\t\t\t max_tokens = 10 \n")

agent_1 = Agent(
    name = "assistant",
    instructions="you are help full assistant",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    model_settings=ModelSettings(max_tokens=10) # ye ab bas 10 token he use karry ga
)

result = Runner.run_sync(agent_1, "how is imran khan")
print(result.final_output)


#-------------------------------------------------------------------------------------------------------------------------
#                                                     max_tokens = 100 
#-------------------------------------------------------------------------------------------------------------------------

print("\n\t\t\t\t\t max_tokens = 100 \n")

agent_2 = Agent(
    name = "assistant",
    instructions="you are help full assistant",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    model_settings=ModelSettings(max_tokens=100) # ye ab bas 100 token he use karry ga
)

result = Runner.run_sync(agent_2, "how is imran khan")
print(result.final_output)

