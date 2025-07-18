import asyncio
from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, enable_verbose_stdout_logging
from agents.run import RunConfig
import asyncio
from agents.run import AgentRunner, set_default_agent_runner
import os
from dotenv import load_dotenv

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
    model_provider = external_client ,
    tracing_disabled = True
)

# -------------------------------------------------------------------------------------------------------

class CustomAgentRunner(AgentRunner):
    async def run(self, starting_agent: Agent, input, **kwargs):
        # Custom preprocessing
        print(f"\nCustomAgeqntRunner.run()")
        print(f"\n{starting_agent}")
        print(f"\n{input}")
        print(f"\n{kwargs}")
        print(f"\nCustomAgeqntRunner.run()\n\n")
        
        # Call parent with custom logic
        result = await super().run(starting_agent, input, **kwargs)
        return result

set_default_agent_runner(CustomAgentRunner())

set_tracing_disabled(disabled=True)

async def main():
    # This agent will use the custom LLM provider
    agent = Agent(
        name="Assistant",
        instructions="You are helpfull assistant",
        model=model,
    )

    result = await Runner.run(
        agent,
        "Hello",
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
    
