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

urdu = Agent(
    name="Urdu",
    instructions="""
        # You are an Urdu converter.
        # Whatever question the user gives you in any language, your job is to convert it into Urdu.
    """,
    model=model
)

english = Agent(
    name="English",
    instructions="""
        # You are an English converter.
        # Whatever question the user gives you in any language, your job is to convert it into English.
    """,
    model=model
)

french = Agent(
    name="French",
    instructions="""
        # You are an French converter.
        # Whatever question the user gives you in any language, your job is to convert it into French.
    """,
    model=model
)

arabic = Agent(
    name="Arabic",
    instructions="""
        # You are an Arabic converter.
        # Whatever question the user gives you in any language, your job is to convert it into Arabic.
    """,
    model=model
)

chinese = Agent(
    name="Chinese",
    instructions="""
        # You are an Chinese converter.
        # Whatever question the user gives you in any language, your job is to convert it into Chinese.
    """,
    model=model
)

portuguese = Agent(
    name="Portuguese",
    instructions="""
        # You are an Portuguese converter.
        # Whatever question the user gives you in any language, your job is to convert it into Portuguese.
    """,
    model=model
)

# ----------------------------------------------------------------------------------------------------------------

translator_agent = Agent(
    name="Translator Agent",
    instructions="""
    # You are a multi-language translator agent.
    # The user will give you a question or sentence in any language.
    # Your job is to identify the language and convert the input into the target language requested.
    # You must understand the intent and provide an accurate and grammatically correct translation.
    # Make sure to preserve the tone and meaning of the original sentence.
    """,
    handoffs=[urdu, english, portuguese, chinese, arabic, french],
    model=model
)

# ----------------------------------------------------------------------------------------------------------------

result = Runner.run_sync(translator_agent, "mujy kal shoping par jana hai is ko english main kardo")

# ----------------------------------------------------------------------------------------------------------------

print(result.final_output)
print("\n\n Last Agent = ", result.last_agent.name)