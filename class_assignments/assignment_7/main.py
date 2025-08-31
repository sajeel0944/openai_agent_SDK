from dataclasses import dataclass
import os
from agents import  ModelSettings, Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool, set_tracing_disabled, RunContextWrapper
from agents.run import RunConfig
from openai import BaseModel
from dotenv import load_dotenv

#----------------------------------------------------------------
load_dotenv()
# set_tracing_disabled(disabled=True)

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
    # tracing_disabled = True
)

#----------------------------------------------------------------

class Information(BaseModel):
    name: str

information = Information(
    name="Airline Seat Preference Agent (Intermediate-Advanced)",
)

#----------------------------------------------------------------

async def dynamic_instructions(ctx: RunContextWrapper[Information], agent: Agent):
    if ctx.context.name == "Medical Consultation Assistant (Intermediate)":
        return"""
            Patient: Use simple, non-technical language. Explain medical terms in everyday words. 
            Be empathetic and reassuring. Medical Student: Use moderate medical terminology with explanations. 
            Include learning opportunities. Doctor: Use full medical terminology, abbreviations, and clinical language. 
            Be concise and professional.
        """
    elif ctx.context.name == "Airline Seat Preference Agent (Intermediate-Advanced)":
        return"""
         Explain window benefits, mention scenic views, reassure about flight experience Middle + 
         Frequent: Acknowledge the compromise, suggest strategies, offer alternatives Any + 
         Premium: Highlight luxury options, upgrades, priority boarding
        """
    elif  ctx.context.name == "Travel Planning Assistant (Intermediate-Advanced)":
        return"""
            Suggest exciting activities, focus on safety tips, recommend social hostels and group tours for meeting people. 
            Cultural + Family: Focus on educational attractions, kid-friendly museums, interactive experiences, 
            family accommodations. Business + Executive: Emphasize efficiency, airport proximity, business centers, 
            reliable wifi, premium lounges. medical_student/doctor
        """


agent = Agent(
    name="Assistant",
    instructions=dynamic_instructions,
    model=model
)

runner = Runner.run_sync(agent, "hello", context=information)
print(runner.final_output)