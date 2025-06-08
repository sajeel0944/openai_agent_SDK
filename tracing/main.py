from agents import Agent, Runner, trace, AsyncOpenAI, OpenAIChatCompletionsModel,  function_tool, set_default_openai_api, set_default_openai_client, set_tracing_disabled, set_trace_processors
from agents.run import RunConfig
import os
from dotenv import load_dotenv
import agentops
from openai import AsyncOpenAI
from agents.tracing.processor_interface import TracingProcessor
from pprint import pprint


agentops.init(os.getenv("AGENTOPS_API_KEY"))

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = "gemini-2.0-flash"



async def main():
    agent = Agent(name="Joke generator", instructions="Tell funny jokes.")

    with trace("Joke workflow"):
        first_result = await Runner.run(agent, "Tell me a joke")
        second_result = await Runner.run(agent, f"Rate this joke: {first_result.final_output}")
        print(f"Joke: {first_result.final_output}")
        print(f"Rating: {second_result.final_output}")


if not BASE_URL or not GEMINI_API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )


client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=GEMINI_API_KEY,
)

set_default_openai_client(client=client, use_for_tracing=True)
set_default_openai_api("chat_completions")
# set_tracing_disabled(disabled=True)


# Custom trace processor to collect trace data locally
class LocalTraceProcessor(TracingProcessor):
    def __init__(self):
        self.traces = []
        self.spans = []

    def on_trace_start(self, trace):
        self.traces.append(trace)
        print(f"Trace started: {trace.trace_id}")

    def on_trace_end(self, trace):
        print(f"Trace ended: {trace.export()}")

    def on_span_start(self, span):
        self.spans.append(span)
        print("*"*20)
        print(f"Span started: {span.span_id}")
        print(f"Span details: ")
        pprint(span.export())

    def on_span_end(self, span):
        print(f"Span ended: {span.span_id}")
        print(f"Span details:")
        pprint(span.export())

    def force_flush(self):
        print("Forcing flush of trace data")

    def shutdown(self):
        print("=======Shutting down trace processor========")
        # Print all collected trace and span data
        print("Collected Traces:")
        for trace in self.traces:
            print(trace.export())
        print("Collected Spans:")
        for span in self.spans:
            print(span.export())


if not BASE_URL or not GEMINI_API_KEY or not MODEL_NAME:
    raise ValueError("Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code.")

# Create OpenAI client
client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=GEMINI_API_KEY,
)

# Configure the client
set_default_openai_client(client=client, use_for_tracing=True)
set_default_openai_api("chat_completions")

# Set up the custom trace processor
local_processor = LocalTraceProcessor()
set_trace_processors([local_processor])




async def main():
    agent = Agent(name="Example Agent", instructions="Perform example tasks.", model=MODEL_NAME)

    with trace("Example workflow"):
        first_result = await Runner.run(agent, "Start the task")
        second_result = await Runner.run(agent, f"Rate this result: {first_result.final_output}")
        print(f"Result: {first_result.final_output}")
        print(f"Rating: {second_result.final_output}")

# Run the main function
import asyncio
asyncio.run(main())