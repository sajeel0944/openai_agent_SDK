import asyncio
from typing import Literal
from agents import Runner, Agent, TResponseInputItem, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, enable_verbose_stdout_logging, trace, SQLiteSession
from agents.run import RunConfig
from dataclasses import dataclass
from dotenv import load_dotenv
import os

#----------------------------------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)

#----------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
    model_provider = external_client,
    tracing_disabled=True
)

# -------------------------------------------------------------------------------------------------------------------

# is ky andar jab tak coding_agent sahe sy code nhi dyga us time tak ye code generate kar ky rahy ga

# -------------------------------------------------------------------------------------------------------------------


# Create agent
agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
    model=model
)

# Create a session instance with a session ID
# session = SQLiteSession("conversation_123")


async def main():
    arr_list = []

    while True:

        user_input = input("Enter : ")

        reponse = await Runner.run(agent, user_input)
        print(reponse.final_output, "\n\n\n")
        arr_list.append(reponse.to_input_list())
        print(arr_list, "\n\n\n")

asyncio.run(main())