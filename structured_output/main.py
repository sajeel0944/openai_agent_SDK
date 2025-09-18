import os
from dotenv import load_dotenv
from agents import Agent, AgentOutputSchema, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, enable_verbose_stdout_logging
from pydantic import BaseModel

load_dotenv()
# enable_verbose_stdout_logging()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

external_cilent = AsyncOpenAI(
    api_key = GEMINI_API_KEY,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_cilent
)

config = RunConfig(
    model = model,
    model_provider = external_cilent,
    tracing_disabled = True
)

class WeatherAnswer(BaseModel):
  location: str
  temperature_c: float
  summary: str
  country : str
  date : str
  time : str



agent = Agent(
  name="StructuredWeatherAgent",
  instructions="Use the final_output tool with WeatherAnswer schema. ",
  output_type=AgentOutputSchema(WeatherAnswer, strict_json_schema=False)
)


out = Runner.run_sync(agent, "What's the current temperature in Karachi?", run_config=config)
print(type(out.final_output))
print(out.final_output)
print(f"\nLocation: {out.final_output.location}")
print(f"Temperature_c: {out.final_output.temperature_c}")
print(f"Summary: {out.final_output.summary}")
print(f"Country: {out.final_output.country}")
print(f"Date: {out.final_output.date}")
print(f"Time: {out.final_output.time}")
