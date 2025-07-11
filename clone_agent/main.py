from agents import  Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from agents.run import RunConfig
from dotenv import load_dotenv
import os
 
#----------------------------------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)

#----------------------------------------------------------------

MODEL = "gemini-1.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

# ----------------------------------------------------------------------------------------------------------------


# Create the original agent
saif_agent = Agent(
    name="Saif Assistant",
    instructions=(
        "You are an assistant named Saif. "
        "If the user asks 'What is your name?', respond with 'My name is Saif.'"
    ),
    model=model
)

# Clone the agent with a new identity and updated instructions
sajeel_agent = saif_agent.clone(
    name="Sajeel Assistant",
    instructions=(
        "You are an assistant named Sajeel. "
        "If the user asks 'What is your name?', respond with 'My name is Sajeel.'"
    )
)

# Run the cloned agent synchronously with a user query
response = Runner.run_sync(sajeel_agent, "What is your name?")
print(response.final_output)