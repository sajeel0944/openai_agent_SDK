import asyncio
from typing import Literal
from agents import Runner, Agent, TResponseInputItem, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, enable_verbose_stdout_logging, trace
from agents.run import RunConfig
from dataclasses import dataclass
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

# -------------------------------------------------------------------------------------------------------------------

# is ky andar jab tak coding_agent sahe sy code nhi dyga us time tak ye code generate kar ky rahy ga

# -------------------------------------------------------------------------------------------------------------------

# is agent code generate kar raha hai
coding_agent = Agent(
    name="Coding Assistant",
    instructions="""
        You are a code assistant. Your job is to solve the user's coding problems. If the user asks you to generate code, you will generate it for them.
        ## Note
        you only return code not text
    """,
    model=model
)

# -------------------------------------------------------------------------------------------------------------------

@dataclass
class ModifyCode:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]

# ye agent code check karry ga ky code sahe hai ya nhi sahe agar sahe hai to pass aye ga wana needs_improvement ya fail aye ga
check_code_agent = Agent(
    name="Code Review Assistant",
    instructions="""
        You are a code check assistant. Please check if this code is correct and professional, and whether there are any mistakes in it.
        You must review the code carefully. If there is even a small mistake, you will return fail , needs_improvement.
        If the code is very very good, you will return pass.

        In the feedback, you will give only 20 words — not more than that.

    """,
    model=model,
    output_type=ModifyCode
)

# -------------------------------------------------------------------------------------------------------------------

async def main():
    # is ky andar user ka prompt aye ga
    user_input : str = input("Enter Your Code Problem : ")

    if not user_input == "" :
        input_items: list[TResponseInputItem] = [{"content": user_input, "role": "user"}] # is ky andar user ka prompt or check_code_agent ka output aye ga
        with trace("LLM as a judge"):
            while True:
                coding_agent_reponse = await Runner.run(coding_agent, input_items) 
                coding_agent_result = coding_agent_reponse.final_output # is ky andar coding_agent ka reponse aye ga
                if coding_agent_result:
                    check_code_agent_reponse = await Runner.run(check_code_agent, coding_agent_result)
                    check_code_agent_result : ModifyCode = check_code_agent_reponse.final_output # is ky andar check_code_agent ka response aye ga

                    # agar agnent ny sahe code nhi dia to ye chaly ga
                    if check_code_agent_result.score == "needs_improvement" or check_code_agent_result.score == "fail":
                        print(f"\n\n{check_code_agent_result.feedback},\t\t {check_code_agent_result.score}")
                    else: # agar agent ny sahe code dia to ye cahly ga
                        input_items.append({"content": f"feedback : {check_code_agent_result.feedback}, score : {check_code_agent_result.score }", "role": "user"})
                        print("\n\n\n\t\t\t Final Modify code")
                        print(check_code_agent_result.score,  check_code_agent_result.feedback)
                        print(f"\n\n{coding_agent_result}")
                        break
    else:
        print("\n\nGood by")

if __name__ in "__main__":
    asyncio.run(main())


# -------------------------------------------------------------------------------------------------------------------

# is prompt sy jab tak agent sahe code nhi dyga us time tak agent cahta rahy ga =  give me typescripy calculate code short and not professional or not correct
# is prompt sy one time main he agent sahe code dyga =  give me typescripy calculate code short

# -------------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------------

# ye openai agent sdk ka code hai
# https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/llm_as_a_judge.py 

# -------------------------------------------------------------------------------------------------------------------


