from typing import Any
from agents import  Agent, RunContextWrapper, RunHooks,Runner, OpenAIChatCompletionsModel, AsyncOpenAI, Tool, Usage, set_tracing_disabled
from agents.run import RunConfig
import asyncio
import base64
import os
from dotenv import load_dotenv
from agents import Agent, Runner


#----------------------------------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)

#----------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.0-flash"

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


# -------------------------------------------------------------------------------------------------------


FILEPATH = os.path.join(os.path.dirname(__file__), "image_bison.jpg")


def image_to_base64(image_path): ## is main image ko Base64 ky format main kar raha ho
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# print(image_to_base64(FILEPATH)) 


async def main():
    # Print base64-encoded image
    b64_image = image_to_base64(FILEPATH) ## image Base64 main convert ho ky arahe hai

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model=model
    )

    result = await Runner.run(
        agent,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": f"data:image/jpeg;base64,{b64_image}",
                    }
                ],
            },
            {
                "role": "user",
                "content": "mujy ye bata o ky is image main kia kia hai detail main batao roman urdu main",
            },
        ],
    )
    print(result.final_output)


if __name__  in "__main__":
    asyncio.run(main())
