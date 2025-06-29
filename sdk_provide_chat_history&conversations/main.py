#-----------------------reply utility-----------------------------------------------------

from agents import  Agent, run_demo_loop, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from agents.run import RunConfig
import asyncio
import os
from dotenv import load_dotenv

#----------------------------------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)

#----------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-1.5-flash"

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


#---------------------------------run_demo_loop---------------------------------------------------

async def main() -> None:
    agent = Agent(name="Assistant", instructions="You are a helpful assistant.", model=model)
    await run_demo_loop(agent) # is ky andar do answer aye gy ek run_streamed wala or dosara run wala a agar run_streamed wala answer nhi chahaye to is ky parameter main stream hai us ko false kardo loop ko stop karny ky liye exit ya quit type

if __name__ == "__main__":
    asyncio.run(main())

# agar kuch samaj nhi aye ko is ky code ko read karro samaj ajaye ga