from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, RunContextWrapper
from agents.run import RunConfig
from pydantic import BaseModel
import os
from dotenv import load_dotenv

#----------------------------------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)

#----------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL : str = "gemini-1.5-flash"

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

# --------------------------------------------dynamic structure-----------------------------------------------------------

class MyContent(BaseModel):
    name: str
    age: int
    city: str
    country: str

async def agent_instructions(context: RunContextWrapper[MyContent], agent: Agent):
    print("agent = ",agent)
    print("\n\ncontext = ", context)
    return f" user information: my name is {context.context.name}, my age is {context.context.age} and i live in {context.context.city}, {context.context.country}, agent name is {agent.name}"


my_information : MyContent = MyContent(
    name="John Doe",
    age=30,
    city="New York",
    country="USA"
) 

# ----------------------------------structure output---------------------------------------------------------------------
print("\n\n\t\t\t\tStructure Output\n")

agent = Agent(
    name= "my_agent",
    instructions=agent_instructions,
    model=model,
    output_type=MyContent
)

response = Runner.run_sync(agent, "what is my name, age, city and country. you give me all agent information", context=my_information)

print("\n\nResponse:")
print(response.final_output)


# ----------------------------------------text ouput-------------------------------------------------------------
print("\n\n\t\t\t\tText Output\n")

agent_1 = Agent(
    name= "my_agent",
    instructions=agent_instructions,
    model=model,
)

response = Runner.run_sync(agent_1, "what is my name, age, city and country. you give me all agent information", context=my_information)

print("\n\nResponse:")
print(response.final_output)
