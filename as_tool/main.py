from agents import Agent,  Runner, function_tool, set_tracing_disabled, enable_verbose_stdout_logging
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
import os

enable_verbose_stdout_logging()
set_tracing_disabled(disabled=True)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gemini/gemini-1.5-flash"

sajeel_information = Agent(
    name="sajeel_information_agent",
    instructions="""
    Mein Sajeel ki personal information rakhta hoon:
    - Front-end developer
    - Generative AI seekh rahe hain
    - Second year mein hain
    - Job ki talash kar rahe hain
    
    Jab bhi Sajeel ki personal life ya career ke bare mein puchen to mujhe handoff karo.
    """,
    handoff_description="Sajeel ki personal information ke liye",
    model=LitellmModel(model=MODEL, api_key=OPENAI_API_KEY),
)

sajeel_social_media = Agent(
    name="sajeel_social_media_agent",
    instructions="""
    Sajeel ke social media links:
    1. GitHub: https://github.com/sajeel0944
    2. LinkedIn: https://linkedin.com/in/sajeel-ullah-khan
    3. Twitter: https://x.com/sajeel_khan_
    4. Instagram: https://instagram.com/sajeelullahkhan
    
    Social media links mangne par turant ye links provide karo.
    """,
    handoff_description="Sajeel ke social media links ke liye",
    model=LitellmModel(model=MODEL, api_key=OPENAI_API_KEY),
)

sajeel_location = Agent(
    name="sajeel_location_agent",
    instructions="""
    Sajeel ki location details:
    - Address: Pakistan, Karachi, Buffer Zone, Sector 15A2, R419
    - Google Maps: https://maps.google.com/...sajeel_location
    
    Location, address ya Google Maps link puchen to ye information provide karo.
    """,
    handoff_description="Sajeel ki location ke liye",
    model=LitellmModel(model=MODEL, api_key=OPENAI_API_KEY),
)

sajeel_agent = Agent(
    name="sajeel_assistant",
    instructions="""
    Tum Sajeel ka personal assistant ho. Roman Urdu mein jawab dena.
    
    Tumhare pass ye specialized agents hain:
    1. sajeel_location_agent - Location aur maps ke liye
    2. sajeel_social_agent - Social media links ke liye
    3. sajeel_personal_agent - Personal information ke liye
    
    User ka sawal suno aur uske mutabiq sahi agent ko handoff karo:
    - Agar location, address ya map puchen → sajeel_location_agent
    - Agar social media links puchen → sajeel_social_agent
    - Agar personal info puchen → sajeel_personal_agent
    
    Handoff karte waqt user ka original sawal forward karna.
    """,
    handoff_description="Ye agent Sajeel ke specialized agents ko handoff kar sakta hai",
    tools=[
            sajeel_information.as_tool(
                tool_name="sajeel_information",
                tool_description="Translate the user's message to sajeel_information"
            ),
            sajeel_social_media.as_tool(
                tool_name="sajeel_social_media",
                tool_description="Translate the user's message to sajeel_social_media"
            ),
            sajeel_location.as_tool(
                tool_name="sajeel_location",
                tool_description="Translate the user's message to sajeel_location"
            )
        ],
    model=LitellmModel(model=MODEL, api_key=OPENAI_API_KEY),
)


result = Runner.run_sync(sajeel_agent, "mujy tum sajeel ki social media ki link dydo")
print(result.final_output)

