from agents import Runner, Agent, function_tool, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
import os
from dotenv import load_dotenv

#---------------------------------------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)

#---------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"  
MODEL = "gemini-1.5-flash"

#---------------------------------------------------------------------

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=BASE_URL,
)

model = OpenAIChatCompletionsModel(
    model=MODEL,
    openai_client=external_client,
)

run_fig = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

#-------------------------------run_llm_again------------------------------------------

@function_tool
def add(num1: int, num2 : int) -> int:
    """
    add two number
    """
    print("add tool called")
    return num1 + num2 - 2 # wrong output

agent = Agent(
    name="calculator agent",
    instructions="you are claculator assiatant",
    tools=[add],
    tool_use_behavior="run_llm_again",# llm add tool call karry ga lakin add tool wrong answer return karry ga phir wo answer llm ky pass jaye ga llm see karry ga ky function ka output wrong hai to llm add tool ky output user ko nhi dyga llm apny pass sy user ko answer dyga
    model=model,
)

#-------------------------------------Runner-----------------------------------------------------

response = Runner.run_sync(agent, "what is 4 plus 4")
print(response.final_output)