from pydantic import BaseModel
from agents import (
    RunConfig, 
    Agent, 
    Runner, 
    OpenAIChatCompletionsModel, 
    AsyncOpenAI, 
    GuardrailFunctionOutput, 
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
)

from dotenv import load_dotenv 
import os

load_dotenv()

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


## is main LLM ka response aye ga 
class CodingHelpingOutput(BaseModel):
    is_coding : bool ## agar user ny coding ka question pouch to is main True aye ga agar user ny koye or question pouch to is main False aye ga
    response : str ## is main LLM ka respone aye ga

## guardrail agent hai ye check karry ga ky user ny kon sa question poush ha or us ko CodingHelpingOutput class main bhejay  ga 
guardrail_agent = Agent(
    name = "coding check",
    instructions = "Check if the user is asking you to do their coding.",
    output_type = CodingHelpingOutput
)


## is ky andar sy main agent ko response jaye ga ky answer dyna hai ya nhi 
@input_guardrail
async def coding_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context, run_config = config)

    return GuardrailFunctionOutput(
        output_info=result.final_output,    ## guardrail_agent ka response yahan aayega (CodingHelpingOutput format mein)
        ## Agar is_coding False hai to tripwire trigger hoga, warna nahi
        tripwire_triggered = not result.final_output.is_coding , ## is main guardrail_agent ny jo CodingHelpingOutput main is_coding main jo dia hai wo is main aye ga or agar user ny coding ka question pouch ko is main True aye ga to wo False hojaye ga or agar user ny koye or question pouch ko is main False aye ga ko us True ajaye ga
    )


## ye main agent hai
agent = Agent(
    name = "customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    input_guardrails=[coding_guardrail] ## Yeh guardrail use karega jo CodingHelpingOutput return karta hai
)


print("\n\t\t\t\t\t\tpriny except block\n")

## is main except chaly ga  q ky user ny qusetion coding dary main nhi pouch hai 
try:
    # Yeh line agent ko run karti hai 
    result = Runner.run_sync(agent, "how is the founder of pakistan", run_config=config)
    # Agar input coding se related hai to final output print hoga wana errro generate hoye ag
    print(result.final_output)
except InputGuardrailTripwireTriggered: # Agar guardrail ne detect kiya ke yeh coding ka sawal nahi hai, to yeh exception trigger hoga
    print("coding guardrail tripped")



print("\n\n\t\t\t\t\t\tpriny try block\n\n")


## is main try chaly ga  q ky user ny qusetion coding ky bary main pouch hai 
try:
    # Yeh line agent ko run karti hai 
    result = Runner.run_sync(agent, "write a hello in python", run_config=config)
    # Agar input coding se related hai to final output print hoga wana errro generate hoye ag
    print(result.final_output)
except InputGuardrailTripwireTriggered: # Agar guardrail ne detect kiya ke yeh coding ka sawal nahi hai, to yeh exception trigger hoga
    print("coding guardrail tripped")