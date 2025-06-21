from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, AgentHooks, RunContextWrapper, set_tracing_disabled
from dotenv import load_dotenv
import os
from pydantic import BaseModel

#----------------------------------------------------------------------------

set_tracing_disabled(disabled=True)
load_dotenv()

#----------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"  # Make sure this is valid for Gemini

#----------------------------------------------------------------------------

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=BASE_URL,
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client,
)

run_config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

#----------------------------------------------------------------------------

class MytestData(BaseModel):
    name: str
    age: int

class MyCustomAgentHook(AgentHooks):
    async def on_start(self, context: RunContextWrapper[MytestData], agent: Agent):
        print(f"Agent: '{agent.name}' \nhook: {agent.hooks}, \ninstructions: {agent.instructions},  started.\n")

    async def on_end(self, context: RunContextWrapper[MytestData], agent: Agent, output):
        print(f"Agent '{agent.name}' for {context.context.name} and my age is {context.context.age} ended with output: {output}")


# ✅ Create instance properly
test_data = MytestData(name="sajeel", age=19)

my_agent = Agent(
    name="assistant",
    instructions="you are a helpful assistant",
    hooks=MyCustomAgentHook(),
    model=model
)

# ✅ Run agent with context 
response = Runner.run_sync(
    my_agent,
    input="hello, how are you",
    context=test_data,
)

print("Final Response:", response.final_output)
