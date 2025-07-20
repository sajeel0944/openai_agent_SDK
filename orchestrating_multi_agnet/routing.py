import asyncio
from agents import Runner, Agent, TResponseInputItem, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, enable_verbose_stdout_logging, trace
from agents.run import RunConfig
from dotenv import load_dotenv
import os


#----------------------------------------------------------------

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
    model_provider = external_client,
)

# --------------------------------------------------------------------------------------------------------------------

# is ky andar routing ho rahe hai har agent ky pass handoff agent hai 

# --------------------------------------------------------------------------------------------------------------------

french_agent = Agent(
    name="french_agent",
    instructions="You only speak French",
    model=model
)

spanish_agent = Agent(
    name="spanish_agent",
    instructions="You only speak Spanish",
    model=model

)

english_agent = Agent(
    name="english_agent",
    instructions="You only speak English",
    model=model

)

urdu_agent = Agent(
    name="urdu_agent",
    instructions="You only speak urdu",
    model=model
)

triage_agent = Agent(
    name="triage_agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[french_agent, spanish_agent, english_agent, urdu_agent],
    model=model
)

# is ky andar sary language agent ky andar handoff agent dy raha ho
french_agent.handoffs = [triage_agent, spanish_agent, english_agent, urdu_agent]
spanish_agent.handoffs = [triage_agent, french_agent, english_agent, urdu_agent]
english_agent.handoffs = [triage_agent, french_agent, spanish_agent, urdu_agent]
urdu_agent.handoffs = [triage_agent, french_agent, spanish_agent, english_agent]


async def main() -> None:
    msg : str = input("Hi! We speak French, Spanish, urdu and English. How can I help? ")

    if not msg == "":
        inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}] # is ky andar sary user ky prompt aye  gy

        with trace("routing"):
            while True:
                reponse = await Runner.run(triage_agent, inputs)
                result = reponse.final_output
                print("\n", result, "\n")
                user_msg : str = input("Enter a message: ")
                if not user_msg == "":
                    inputs.append({"content": user_msg, "role": "user"})
                else:
                    print("\n\nGood by")
                    break
    else:
        print("\n\nGood by")

if __name__ in "__main__":
    asyncio.run(main())
