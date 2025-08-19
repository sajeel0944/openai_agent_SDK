from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, enable_verbose_stdout_logging, trace, SQLiteSession
from agents.run import RunConfig
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
 
# --------------------------------------------------------------

agent = Agent(
    name="Assistant",
    instructions="You are help full assistant",
    model=model
)

# session_id sy memory ki id bun ti hai
session = SQLiteSession(session_id="2323")

# First interaction
result_1 = Runner.run_sync(agent, "my name is sajeel", session=session)
print(result_1.final_output)

# Second interaction (ye memory use karry ga waha sy name lyky aye ga)
result_2 = Runner.run_sync(agent, "my is my name", session=session)
print(result_2.final_output)

# is main Temporary Memory use ho rahe hai ye Temporary Memory kuch time tak he raha gi  