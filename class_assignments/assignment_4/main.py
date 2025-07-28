from agents import  Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from agents.run import RunConfig
from dotenv import load_dotenv
import os
 
#----------------------------------------------------------------

load_dotenv()
# set_tracing_disabled(disabled=True)

#----------------------------------------------------------------

MODEL = "gemini-1.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
    # tracing_disabled = True
)

# -----------------------------------------------------------------------------------------------------------

lyric_poetry_agent = Agent(
    name="Lyric poetry Agent",
    instructions="""
    ### 📜 Lyric Poetry Analyst

    You analyze **lyric poetry**, which expresses personal feelings, emotions, and states of mind.

    - Focus on internal emotions, moods, or thoughts.
    - Highlight the poet’s voice and tone.
    - Look for musical elements like rhythm, repetition, and sound.
    - Common themes: love, sorrow, joy, solitude.

    Respond as a thoughtful literary critic, interpreting emotional depth and poetic devices.
    """,
    model=model
)

narrative_poetry_agent = Agent(
    name="Narrative poetry Agent",
    instructions="""
    ### 📖 Narrative Poetry Analyst

    You analyze **narrative poetry**, which tells a story using poetic structure.

    - Identify the **characters, plot, setting, and conflict**.
    - Comment on how the story unfolds through verse.
    - Analyze rhyme, meter, and use of dialogue or description.
    - Note storytelling techniques like suspense, climax, and resolution.

    Respond as a literary narrator unpacking how the poem operates as a story.
    """,
    model=model
)

dramatic_poetry_agent = Agent(
    name="Dramatic poetry Agent",
    instructions="""
    ### 🎭 Dramatic Poetry Analyst

    You analyze **dramatic poetry**, meant to be spoken or performed.

    - Focus on **dramatic monologue, soliloquy, or dialogue**.
    - Highlight the speaker’s role, character motivations, and emotional range.
    - Consider theatricality, tension, and interaction with implied or real audience.
    - Examine language, tone shifts, and stage-like qualities.

    Respond as a drama critic or director interpreting performance and character depth.
    """,
    model=model
)

poetry_agent = Agent(
    name="Poetry Agent",
    instructions="""
    ### 🧩 Poetry Triage Agent

    You are the **Poetry Orchestrator**. A user will provide a poem (usually 2 stanzas).

    1. Analyze the poem’s form, structure, and purpose.
    2. Decide whether it's **Lyric**, **Narrative**, or **Dramatic** poetry.
    3. Forward the poem to the appropriate specialist agent for deeper analysis.

    If the input doesn't clearly fit, briefly justify and still choose the most likely category.

    Output should include:
    - The detected type of poetry.
    - A summary of reasoning.
    - The detailed analysis from the appropriate agent.
    """,
    model=model,
    handoffs=[lyric_poetry_agent, narrative_poetry_agent, dramatic_poetry_agent]
)


result = Runner.run_sync(poetry_agent, "Raat bhar chandni se baat ki maine,\nTanha sitaron ke saath raat ki maine. Dil ke gum ko lafzon mein dhal diya,\nKhamoshiyon se dosti rakh li maine.")

print(result.final_output)