from agents import Runner, Agent, set_tracing_disabled, ItemHelpers, function_tool
from dotenv import load_dotenv
from agents.extensions.models.litellm_model import LitellmModel
from openai.types.responses import ResponseTextDeltaEvent
import os
import asyncio

set_tracing_disabled(disabled=True)
# .env file load karo
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "gemini/gemini-1.5-flash"


#                                   streaming Text code

async def main():

    agent = Agent(
        name = "agent",
        instructions="",
        model=LitellmModel(model=MODEL, api_key=OPENAI_API_KEY)
    )


    result = Runner.run_streamed(agent, "give me 100 eassy in AI")

    async for event  in result.stream_events():
        print(event, end="", flush=True)

asyncio.run(main())





#                               Stream item code

@function_tool
def news(news_query:str) -> str:
    """
    AI is very powerful and can replace humans.
    """
    return(news_query)

async def main():
    agent = Agent(
        name="news assistant",
        instructions="You are a news agent. You provide the latest AI news.",
        tools=[news],
        model=LitellmModel(model=MODEL, api_key=OPENAI_API_KEY)
    )

    result = Runner.run_streamed(
        agent,
        input="Give me the latest AI news",

    )
    print("=== Run starting ===")
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            continue
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
            continue
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called")
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
            else:
                pass 



asyncio.run(main())

print("=== Run complete ===")