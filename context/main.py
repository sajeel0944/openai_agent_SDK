from agents import Agent, function_tool, Runner, set_tracing_disabled, RunContextWrapper, enable_verbose_stdout_logging
from dotenv import load_dotenv
import os
import asyncio
from dataclasses import dataclass
from agents.extensions.models.litellm_model import LitellmModel

# enable_verbose_stdout_logging()
set_tracing_disabled(disabled=True)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gemini/gemini-2.0-flash"

@dataclass
class UserInfo:
    name : str
    age : int
    location : str


name : str = input("Enter Your Name : ")
age : str = int(input("Enter Your Age : "))
location : str = input("Enter Your Location : ")

if name and age and  location:
    print("\nAsk To AI About You\n")
    userinfo : UserInfo = UserInfo(name, age, location)

    user_prompt : str = input("Enter Your Prompt : ")

    if user_prompt:
        @function_tool ## is tool ky andar user ka name or location arahe hai
        def user_location(wrapper: RunContextWrapper[UserInfo]) -> str:
            """Return user location and name"""
            return f"user : {wrapper.context.name} if from {wrapper.context.location}"

        @function_tool ## is tool ky andar user ka age or name arahe hai
        def user_age(wrapper: RunContextWrapper[UserInfo]) -> str:
            """Return user age and name"""
            return f"user {wrapper.context.name} is {wrapper.context.age} year old"

        async def main():
            agent = Agent(
                name= "Assistant",
                tools=[user_age, user_location],
                model = LitellmModel(model=MODEL, api_key=OPENAI_API_KEY)
            )

            result = await Runner.run(
                starting_agent=agent,
                input=user_prompt,
                context=userinfo, ## is ky andar object pass kia hai
            )

            print(result.final_output)

        if __name__ in "__main__":
            asyncio.run(main())

    else:
        print("Please Enter Your Prompt")
       
else:
    print("\nPlease Fill All Field\n")
   