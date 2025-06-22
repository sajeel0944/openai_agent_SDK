from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, RunHooks, RunContextWrapper, set_tracing_disabled
from dotenv import load_dotenv
import os
from pydantic import BaseModel

#----------------------------------------------------------------------------

set_tracing_disabled(disabled=True)
load_dotenv()

#----------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"  

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

class MytestData(BaseModel): # is ky andar user ka name or age aye gi
    name: str
    age: int


class MyCustomRunnerHook(RunHooks):
    async def on_agent_start(self, context: RunContextWrapper[MytestData], agent: Agent):
        print(f"starting run for agent: {agent.name} \nhook: {agent.hooks}, \ninstructions: {agent.instructions}, \nuser name: {context.context.name}, \nage: {context.context.age}\n")

    async def on_agent_end(self, context: RunContextWrapper[MytestData], agent: Agent, output):
        print(f"Run complete for agent: {agent.name}, output: {output}")


# ✅ Create instance properly
test_data = MytestData(name="sajeel", age=19)

my_agent = Agent(
    name="assistant",
    instructions="you are a helpful assistant",
    model=model
)

# ✅ Run agent with context 
response = Runner.run_sync(
    my_agent,
    input="hello, how are you",
    context=test_data,
    hooks=MyCustomRunnerHook()
)

print("Final Response:", response.final_output)
