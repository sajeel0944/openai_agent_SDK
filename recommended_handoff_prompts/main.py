from agents import  Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from agents.run import RunConfig
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from dotenv import load_dotenv
import os


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


#----------------------------------------------------------------------------------------


web_developer = Agent(
    name="website developer agent",
    # RECOMMENDED_PROMPT_PREFIX is main opneai agent sdk ka instruction aye gi
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    <You are an expert in website development. Provide detailed information to the user about becoming a 
    website developer, including which languages (e.g., HTML, CSS, JavaScript) and frameworks 
    (e.g., React, Angular) to learn, along with all relevant details.>.""",
    model=model,
    handoff_description="This agent provides information about web development."
)

print(f"\n\nweb_developer instruction : {web_developer.instructions}\n\n\n\n\n\n")


mobile_developer = Agent(
    name="mobile developer agent",
    # RECOMMENDED_PROMPT_PREFIX is main opneai agent sdk ka instruction aye gi
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX} 
    <You are an expert in mobile development. Provide detailed information to the user about becoming a 
    mobile developer, including which languages (e.g., Swift, Kotlin) and frameworks 
    (e.g., Flutter, React Native) to learn, along with all relevant details.>.""",
    model=model,
    handoff_description="This agent provides information about mobile development."
)

print(f"\n\n\n\n\n\n\nmobile_developer instruction : {mobile_developer.instructions}\n\n\n\n\n\n")


panacloud_agent = Agent(
    name="Panavloud assistant",
    handoffs=[web_developer, mobile_developer],
    model=model
    )

result = Runner.run_sync(panacloud_agent, "Give me information about agentic web development")
print(result.final_output)