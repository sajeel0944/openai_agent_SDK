from pydantic import BaseModel
from agents import (
    RunConfig, 
    Agent, 
    Runner, 
    OpenAIChatCompletionsModel, 
    AsyncOpenAI, 
    GuardrailFunctionOutput, 
    RunContextWrapper,
    TResponseInputItem,
    output_guardrail,
    OutputGuardrailTripwireTriggered,
    enable_verbose_stdout_logging

)

from dotenv import load_dotenv 
import os
import asyncio

load_dotenv()
# enable_verbose_stdout_logging()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=model,
    tracing_disabled=True
)

class MessageOutput(BaseModel):
    response: str


## is main guardrail_agent LLM ka response aye ga 
class CodingOutput(BaseModel):
    is_coding: bool  ## agar LLM ny coding ka answer dia to is main True aye ga agar LLM ny koye or answer dia to is main False aye ga
    reasoning: str  ## is main LLM ka respone aye ga



## guardrail agent hai ye check karry ga ky LLM ny kon sa Answer dia hai or us ko CodingOutput class main bhejay  ga 
guardrail_agent = Agent(
    name = "coding check",
    instructions = "Check if the user is asking you to do their coding.",
    output_type = CodingOutput,
    model=model
)


## is ky andar sy main agent ko response jaye ga ky answer dyna hai ya nhi 
@output_guardrail
async def coding_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, output.response, context=ctx.context, run_config = config)

    return GuardrailFunctionOutput(
        output_info=result.final_output,    ## guardrail_agent ka response yahan aayega (CodingOutput format mein)
        ## Agar is_coding False hai to tripwire trigger hoga, warna nahi
        tripwire_triggered =  not result.final_output.is_coding , ## is main guardrail_agent ny jo CodingOutput main is_coding main jo dia hai wo is main aye ga or agar LLm ny coding ka answer dia ko is main True aye ga to wo False hojaye ga or agar LLM ny koye or answer dia ko is main False aye ga ko us True ajaye ga
    )


## ye main agent hai
agent = Agent(
    name = "coding agent",
    instructions="You are a coding agent. You help user with their questions.",
    output_guardrails=[coding_guardrail], ## Yeh guardrail use karega jo CodingOutput return karta hai
    output_type=MessageOutput,
    model=model
)




print("\n\t\t\t\t\t\tpriny except block\n")

async def main():
    ## is main try chaly ga  q ky LLM ka answer  coding ky bary main hoye ga 
    try:
        # Yeh line agent ko run karti hai 
        result = await Runner.run(agent, "how is the founder of pakistan", run_config=config)
        # Agar Output coding se related hai to final output print hoga wana errro generate hoye ag
        print(result.final_output.response)
    except OutputGuardrailTripwireTriggered: # Agar guardrail ne detect kiya ke yeh coding ka answer nahi hai, to yeh exception trigger hoga
        print("coding guardrail tripped")


if __name__ == "__main__":  
    asyncio.run(main())




print("\n\n\t\t\t\t\t\tpriny try block\n\n")

async def main():
    ## is main except chaly ga  q ky LLM ka answer coding ky bary main nhi hoye ga 
    try:
        # Yeh line agent ko run karti hai 
        result = await Runner.run(agent, "write a hello in python", run_config=config)
        # Agar Outpot coding se related hai to final output print hoga wana errro generate hoye ag
        print(result.final_output.response)
    except OutputGuardrailTripwireTriggered: # Agar guardrail ne detect kiya ke yeh coding ka answer nahi hai, to yeh exception trigger hoga
        print("coding guardrail tripped")


if __name__ == "__main__":  
    asyncio.run(main())


# Note
# output_guardrail last agent par chata hai