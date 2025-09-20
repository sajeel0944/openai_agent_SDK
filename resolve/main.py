import os
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI, Runner, set_tracing_disabled
from agents.run import RunConfig
from dotenv import load_dotenv
import rich

#----------------------------------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)

#----------------------------------------------------------------

GEMINI_API_KEY : str = os.getenv("GEMINI_API_KEY")
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

#------------------------------resolve-----------------------------------
# `resolve()` method purani settings ke sath naye settings ko merge karta hai.
# Naye settings agar kisi field mein value dete hain to wo purani value ko override kar dete hain.
# Agar koi nayi setting hai jo pehle exist nahi karti thi, to wo add ho jati hai.

old_ModelSettings = ModelSettings(temperature=0.7)
new_ModelSettings = old_ModelSettings.resolve(ModelSettings(top_p=0.6))

agent = Agent(
    name="Assistant",
    instructions="You are help assistant ",
    model=model,
    model_settings=new_ModelSettings
)

result = Runner.run_sync(agent, "hello")
rich.print(result)


#-----------------------setting override------------------------------------------
print("\n\n\n")

old_ModelSettings_2 = ModelSettings(temperature=0.7)
new_ModelSettings_2 = old_ModelSettings_2.resolve(ModelSettings(top_p=0.6, temperature=0.4))

agent_2 = Agent(
    name="Assistant",
    instructions="You are help assistant ",
    model=model,
    model_settings=new_ModelSettings_2
)

result_2 = Runner.run_sync(agent_2, "hello")
rich.print(result_2)