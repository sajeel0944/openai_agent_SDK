from dataclasses import dataclass
from agents import ModelSettings, Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool, set_tracing_disabled, handoff
from agents.run import RunConfig
import os 
from dotenv import load_dotenv
from agents.extensions import handoff_filters

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

# ----------------------------------------------------------------------------------------------------------

#                               important
# Jab handoff hota hai (yani aik agent se doosray agent ko baat cheet transfer hoti hai), 
# to naya agent puri peechli conversation dekh sakta hai — jaise ke usne shuru se sab suna ho.
# Lekin agar tum chahte ho ke yeh na ho, to tum input_filter use kar sakte ho.
# Input filter aik function hota hai jo peechla input leta hai (via HandoffInputData) aur 
# tumhein aik naya HandoffInputData wapas dena hota hai — jo filter karke diya jaata hai.

# ----------------------------------------------------------------------------------------------------------

calculator_agent = Agent(
    name="calculator",
    instructions="A simple calculator agent that can perform basic arithmetic operations.",
    model=model,
    handoff_description="Calculator agent can perform basic arithmetic operations like addition, subtraction, multiplication, and division.",
)

# -------------------------------------------------remove_all_tools---------------------------------------------------------

print("\nagent_1\n\n")

agent_1 = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. You can answer questions and perform tasks.",
    model=model,
    handoffs=[
        handoff(
            agent=calculator_agent,
            input_filter=handoff_filters.remove_all_tools # Is remove_all_tools ka kaam hai ke jab handoff ho, to main agent ki taraf se koi bhi tool calls (functions waghera) history mein se hata diye jaayen. Calculator agent ko sirf normal conversation milegi, main agent ka tools ka output wagara claculator agent ky pass nhi jaye ga.
        ),
    ]
)

response_1 = Runner.run_sync(agent_1, "What is 2+2?")
print(response_1.final_output)

# -----------------------------------------------_remove_tools_from_items-----------------------------------------------------------

print("\nagent_2\n\n")

agent_2 = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. You can answer questions and perform tasks.",
    model=model,
    handoffs=[
        handoff(
            agent=calculator_agent,
            input_filter=handoff_filters._remove_tools_from_items #  Conversation messages ki list me se tools se related items (jaise tool_use, tool_result, function_call) remove karta hai. Taake jab handoff ho, naye agent ko sirf normal conversation (user aur assistant ke messages) milein — tools ke technical details nahi.
        ),
    ]
)

response_2 = Runner.run_sync(agent_2, "What is 2+2?")
print(response_2.final_output)

# -------------------------------------------_remove_tool_types_from_input---------------------------------------------------------------

print("\nagent_3\n\n")

agent_3 = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. You can answer questions and perform tasks.",
    model=model,
    handoffs=[
        handoff(
            agent=calculator_agent,
            input_filter=handoff_filters._remove_tool_types_from_input # Handoff input se specific tool types (jaise tool_use, tool_result, etc.) remove karta hai. Iska kaam hota hai ke jab conversation history naye agent ko di jaye, to usme tool-related message types na hon — sirf user aur assistant ke normal messages hoon.
        ),
    ]
)

response_3 = Runner.run_sync(agent_3, "What is 2+2?")
print(response_3.final_output)


# Function	                    |                         Kaam
# -----------------------------------------------------------------------------------------------------
# _remove_tools_from_items      |	Messages list me se tool-related items hataata hai.
# _remove_tool_types_from_input	|   Handoff input me se tool-related types hataata hai.
# remove_all_tools              |	Full handoff filter hai jo dono upar wale functions ko use karta hai.
# ----------------------------------------------------------------------------------------------------
