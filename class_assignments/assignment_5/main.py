from dataclasses import dataclass
import os
from agents import  AgentOutputSchema, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, RunContextWrapper, Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, TResponseInputItem, set_tracing_disabled, input_guardrail
from agents.run import RunConfig
from openai import BaseModel
from dotenv import load_dotenv

#----------------------------------------------------------------
load_dotenv()
set_tracing_disabled(disabled=True)

#----------------------------------------------------------------

OPENAI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL : str = "gemini-2.5-flash"

#----------------------------------------------------------------

external_client = AsyncOpenAI(
    api_key = OPENAI_API_KEY,
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

# ---------------------------------------------------------

print("\n\n\nExercise 1\n\n\n")

class OutPutSchema(BaseModel):
    is_timing: bool
    response: str

check_agent = Agent(
    name="Check Assistant",
    instructions="""
    You are an assistant that checks if the user wants to change their class timings.
    If the input is about changing class timings, return is_timing: true. Otherwise, return is_timing: false.
    """,
    model=model,
    output_type=AgentOutputSchema(OutPutSchema, strict_json_schema=False)
)

@input_guardrail
async def time_guardrail( ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(check_agent, input, run_config = config)

    return GuardrailFunctionOutput(
        output_info=result.final_output, 
        tripwire_triggered = result.final_output.is_timing, 
    )

agent = Agent(
    name = "Teacher agent",
    instructions="You are a teacher. You help student with their questions.",
    input_guardrails=[time_guardrail]
)

try:
    result = Runner.run_sync(agent, "mujy time change karna hai", run_config=config)
    print(result.final_output)
except InputGuardrailTripwireTriggered: 
    print("coding guardrail tripped")

# --------------------------------------------------------------------------------------------------

print("\n\n\nExercise 2\n\n\n")

class TemperatureCheckOutput(BaseModel):
    is_safe_temperature: bool
    response: str

temperature_check_agent = Agent(
    name="Father Check Agent",
    instructions="""
    You are a father who checks if it's safe for your child to go outside and run.
    If the temperature mentioned in the input is **below 26°C**, respond with:
    {
        "is_safe_temperature": false,
        "response": "No, it's too cold to run outside!"
    }

    If the temperature is **26°C or above**, respond with:
    {
        "is_safe_temperature": true,
        "response": "Yes, you can go run outside."
    }
    """,
    model=model,
    output_type=AgentOutputSchema(TemperatureCheckOutput, strict_json_schema=False)
)

@input_guardrail
async def temperature_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(temperature_check_agent, input, context=ctx.context, run_config=config)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_safe_temperature  
    )

father_agent = Agent(
    name="Father Agent",
    instructions="You are a kind but strict father. You help your child make safe decisions.",
    model=model,
    input_guardrails=[temperature_guardrail]
)

try:
    result = Runner.run_sync(father_agent, "Papa, I want to go run. It's 24C today.", run_config=config)
    print("Final Output:", result.final_output)
except InputGuardrailTripwireTriggered:
    print("⚠️ Father guardrail triggered — it's too cold to run.")


# --------------------------------------------------------------------------------------------------

print("\n\n\nExercise 3\n\n\n")

# Schema for validating school entry
class SchoolCheckOutput(BaseModel):
    is_allowed: bool
    response: str

# Agent that checks the student's school
school_check_agent = Agent(
    name="School Check Agent",
    instructions="""
    You are a gatekeeper who only allows students from 'Allied School' to enter.
    If the input says the student is from Allied School, respond with:
    {
        "is_allowed": true,
        "response": "Welcome to Allied School!"
    }

    If the input mentions any other school, respond with:
    {
        "is_allowed": false,
        "response": "Sorry, only Allied School students are allowed."
    }
    """,
    model=model,
    output_type=AgentOutputSchema(SchoolCheckOutput, strict_json_schema=False)
)

# Guardrail for checking school name
@input_guardrail
async def school_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(school_check_agent, input, context=ctx.context, run_config=config)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_allowed  # Trip if student is not from Allied School
    )

# Gate Keeper Agent using the guardrail
gatekeeper_agent = Agent(
    name="Gate Keeper Agent",
    instructions="You are a strict gatekeeper. Only allow students from Allied School.",
    model=model,
    input_guardrails=[school_guardrail]
)

# Testing the agent with input
try:
    result = Runner.run_sync(gatekeeper_agent, "I am from Beaconhouse and want to enter.", run_config=config)
    print("Agent Response:", result.final_output)
except InputGuardrailTripwireTriggered:
    print("🚫 Access Denied: You are not from Allied School.")
