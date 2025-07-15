from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, handoff
from agents.run import RunConfig
import os 
from dotenv import load_dotenv

#----------------------------------------------------------------

set_tracing_disabled(disabled=True)
load_dotenv()

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

# ---------------------------------------------------------------------------------------------------------

calculator_agent = Agent(
    name="claculator",
    instructions="A simple calculator agent that can perform basic arithmetic operations.",
    model=model,
    handoff_description="Calculator agent can perform basic arithmetic operations like addition, subtraction, multiplication, and division.",
)

mobile_developer = Agent(
    name="mobile developer agent",
    instructions="You are an expert in mobile development. Provide detailed information to the user about becoming a mobile developer, including which languages (e.g., Swift, Kotlin) and frameworks (e.g., Flutter, React Native) to learn, along with all relevant details.",
    model=model,
    handoff_description="This agent provides information about mobile development."
)


# -----------------------------------------------------------------------------------------------------------
print("\nagent_1\n\n")

agent = Agent(
    name= "Assistant",
    instructions="You are a helpful assistant. You can answer questions and perform tasks.",
    model=model,
    handoffs=[
        handoff(
            agent=calculator_agent,
            tool_name_override="Calculator",
            tool_description_override="A simple calculator agent that can perform basic arithmetic operations.",
            is_enabled=True  # is_enabled=True karny sy calculator_agent llm ko nazar aye ga to llm is ko use kar sake ga   
        ),
        mobile_developer
    ]
)

reponse_1 = Runner.run_sync(agent, "What is 2+2?")
print(reponse_1.final_output)
print("\n\nlast agent name: ", reponse_1.last_agent.name)
print(f"\n\n{reponse_1.last_agent}\n\n")


# -----------------------------------------------------------------------------------------------------------
print("\n\nagent_2 \n\n")

agent_2 = Agent(
    name= "Assistant",
    instructions="You are a helpful assistant. You can answer questions and perform tasks.",
    model=model,
    handoffs=[
        handoff(
            agent=calculator_agent,
            tool_name_override="Calculator",
            tool_description_override="A simple calculator agent that can perform basic arithmetic operations.",
            is_enabled=False  # is_enabled=False karny sy calculator_agent llm ko nazar nahi aye ga to llm is ko use nahi kar sake ga    
        ),
        mobile_developer
    ]
)

reponse_2 = Runner.run_sync(agent_2, "What is 2+2?")
print(reponse_2.final_output)
print("\n\nlast agent name: ", reponse_2.last_agent.name)
print(f"\n\n{reponse_2.last_agent}\n\n")

