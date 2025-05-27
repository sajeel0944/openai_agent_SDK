from agents import Runner, Agent, function_tool, set_tracing_disabled, enable_verbose_stdout_logging
from agents.extensions.models.litellm_model import LitellmModel
import os
from dotenv import load_dotenv


enable_verbose_stdout_logging()
set_tracing_disabled(disabled=True)


# .env file load karo
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "gemini/gemini-1.5-flash"


web_developer = Agent(
    name="website developer agent",
    instructions="You are an expert in website development. Provide detailed information to the user about becoming a website developer, including which languages (e.g., HTML, CSS, JavaScript) and frameworks (e.g., React, Angular) to learn, along with all relevant details.",
    model=LitellmModel(model=MODEL, api_key=OPENAI_API_KEY),
    handoff_description="This agent provides information about website development."
)

mobile_developer = Agent(
    name="mobile developer agent",
    instructions="You are an expert in mobile development. Provide detailed information to the user about becoming a mobile developer, including which languages (e.g., Swift, Kotlin) and frameworks (e.g., Flutter, React Native) to learn, along with all relevant details.",
    model=LitellmModel(model=MODEL, api_key=OPENAI_API_KEY),
    handoff_description="This agent provides information about mobile development."
)


@function_tool
def developer(information : str) -> str:
    """
    A tool to provide general developer-related information.
    """
    return (f"Developer Info: {information}")

@function_tool
def openai_agent(information: str) -> str:
    """
       A tool to provide information about agentic AI, including calling tools/functions and collaborating with multiple agents via handoff and routing.
    """

    return(f"Agentic AI Info: {information}")


agentic_ai = Agent(
    name="agentic ai developer agent",
    instructions="You are an expert in agentic AI development. Provide detailed information to the user about becoming an agentic AI developer, including which languages (e.g., Python) and frameworks (e.g., OpenAI SDK, LangChain) to learn, along with all relevant details.",
    model=LitellmModel(model=MODEL, api_key=OPENAI_API_KEY),
    handoff_description="This agent provides information about agentic AI development.",
    tools=[developer, openai_agent]
)

panacloud_agent = Agent(
    name="Panavloud assistant",
    instructions=" If the user asks about website development, mobile development, or agentic AI development, directly hand off the query to the appropriate agent (website developer, mobile developer, or agentic AI developer) without asking for confirmation.",
    model=LitellmModel(model=MODEL, api_key=OPENAI_API_KEY),
    handoff_description="This agent routes queries to specialized agents for website development, mobile development, or agentic AI development.",
    handoffs=[web_developer, mobile_developer, agentic_ai]
)


result = Runner.run_sync(panacloud_agent, "Give me information about agentic ai development")
print(result.final_output)