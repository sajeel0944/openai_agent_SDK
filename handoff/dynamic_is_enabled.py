from dataclasses import dataclass
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

# ----------------------------------------------------------------------------------------------------------

#  please give me bank account information. ## is prompt sy LLM bank_account_agent handoff nhi karry ga
# My bank account number is 1234567890, please give me bank account information. ## is prompt sy LLM bank_account_agent handoff karry ga

user_input : str = input("Enter your input: ")

if user_input:

# -----------------------------------------------------------------------------------------------------------

    # agar user ny pana bank account number diya to is_bank_account ky andar chaeck_agent True dyga wana false 
    @dataclass
    class CheckUserInput():
        is_bank_account: bool

    # user ky prompt ko check karry ga ky us ky prompy ky nadar bank account number hai ya nahi agar hai to CheckUserInput is ky andar true aye ga wana flase
    chaeck_agent = Agent(
        name= "CheckUserInputAgent",
        instructions="You are a helpful agent that checks user input for bank account numbers. If the user provide a bank account number then you is_bank_account should be true, otherwise false. If the user input is not a bank account number then you should return false.",
        model=model,
        output_type=CheckUserInput, # llm CheckUserInput is format main answer dyga
    )

    reponse = Runner.run_sync(chaeck_agent, user_input)

    AfterCheckUserInput : CheckUserInput = reponse.final_output # chaeck_agent is ky andar agent ka output aye ga CheckUserInput is format main

    print("\n\nAfter Check User Input:", AfterCheckUserInput)

    if AfterCheckUserInput:
        
# -----------------------------------------------------------------------------------------------------------

        bank_account_agent = Agent(
            name="BankAccountAgent",
            instructions="You are a helpful agent that provides information about bank accounts.",
            model=model,
        )

# -----------------------------------------------------------------------------------------------------------

        agent = Agent(
            name="AssistantAgent",
            instructions="You are a helpful assistant, you have one handoff agent that provides information about bank accounts.",
            model=model,
            handoffs=[
                handoff(
                    agent=bank_account_agent,
                    is_enabled=AfterCheckUserInput.is_bank_account, # agar is main true aya to LLM ko bank_account_agent nazar aye ga wana LLM ko nahi nazar aye ga 
                )
            ]
        )

    final_reponse = Runner.run_sync(agent, user_input)
    print("\n\nFinal Response:", final_reponse.final_output)
    print("\n\nlast agent name:", final_reponse.last_agent.name)
