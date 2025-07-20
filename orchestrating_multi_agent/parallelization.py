import asyncio
from agents import ItemHelpers, ModelSettings, Runner, Agent, TResponseInputItem, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, enable_verbose_stdout_logging, trace
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

# -----------------------------------------------------------------------------------------------------------------

# is ky andar english agent Parallel agent ko cell karry ga matarab ky english agent ek saat 3 dafa translate dyga

# -----------------------------------------------------------------------------------------------------------------

english_agent = Agent(
    name="english_agent",
    instructions="You translate the user's message to english",
    model_settings=ModelSettings(temperature=2.0),
    model=model
)

translation_picker = Agent(
    name="translation_picker",
    instructions="You pick the best english translation from the given options and you will return only best english translation",
    model=model
)

# -----------------------------------------------------------------------------------------------------------------

async def main():

    msg = input("Hi! Enter a message, and we'll translate it to english: ")

    with trace("Parallel translation"):
        res_1, res_2, res_3 = await asyncio.gather(
            Runner.run(
                english_agent,
                msg,
            ),
            Runner.run(
                english_agent,
                msg,
            ),
            Runner.run(
                english_agent,
                msg,
            ),
        )

        print(f"\n\nTranslations:\n\n {"res_1: "+ res_1.final_output} \n{" res_2: "+ res_2.final_output} \n {"res_3: "+ res_3.final_output}")

        best_translation = await Runner.run(
            translation_picker,
            f"Input: {msg}\n\nTranslations:\n {"res_1: "+ res_1.final_output} \n {" res_2: "+ res_2.final_output} \n {"res_3: "+ res_3.final_output}",
        )

        print("\n\n-----")

        print(f"Best translation: {best_translation.final_output}")

# -----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())