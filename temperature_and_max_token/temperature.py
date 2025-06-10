from agents import Agent, Runner, set_tracing_disabled, ModelSettings
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
#                                                     temperature = 0.1  
#-------------------------------------------------------------------------------------------------------------------------

print("\n\t\t\t\t\tTemperature = 0.1\n")

agent_1 = Agent(
    name = "assistant",
    instructions="you are help full assistant",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    model_settings=ModelSettings(temperature=0.1) ## 0.1 is sy ye hoye ga ky LLM ek he answer bar bar dy ga LLM shochy ga nhi 
)

result = Runner.run_sync(starting_agent=agent_1, input="how is imran khan")
print(result.final_output)


#-------------------------------------------------------------------------------------------------------------------------
#                                                     temperature = 1  
#-------------------------------------------------------------------------------------------------------------------------

print("\n\t\t\t\t\tTemperature = 1\n")

agent_2 = Agent(
    name = "assistant",
    instructions="you are help full assistant",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    model_settings=ModelSettings(temperature=1) ## 1 is sy ye hoye ga ky LLM har bar ek he answer nhi dy ga har bar dosara answer dygy shochy ky
)

result = Runner.run_sync(starting_agent=agent_2, input="how is imran khan")
print(result.final_output)
