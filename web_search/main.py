from bs4 import BeautifulSoup
import os
from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool, set_tracing_disabled
from agents.run import RunConfig
from dotenv import load_dotenv
import requests
import rich

#----------------------------------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)

#----------------------------------------------------------------

GEMINI_API_KEY : str = os.getenv("GEMINI_API_KEY")
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


#--------------------------------Tool--------------------------------

#                               Stream item code

@function_tool
def web_search_tool(query: str) -> str:
    """
     SerpAPI ka use karke real-time Google search karta hai aur pehla organic result return karta hai.

    Parameters:
        query (str): User ki search query, jaise "latest iPhone news" ya "weather in Lahore".

    Returns:
        str: Pehle search result ka snippet aur link. Agar koi result na mile to 'No results found' return karta hai.
    """
    api_key = os.getenv("STREAM_API_KEY")  # Make sure to set this in your environment
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Simplified result: show top snippet
    try:
        snippet = data['organic_results'][0]['snippet']
        link = data['organic_results'][0]['link']
        return f"{snippet}\nMore info: {link}"
    except (KeyError, IndexError):
        return "No results found."
    

@function_tool
def read_website(url: str) -> str:
    """
    Fetches the content of any public website and extracts its plain text (without HTML).

    Parameters:
        url (str): The website's full URL, for example "https://en.wikipedia.org/wiki/OpenAI".

    Returns:
        str: The extracted plain text content (max 2000 characters). If an error occurs, returns an error message.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"Error fetching the website: {str(e)}"
    
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove scripts and styles
    for script in soup(["script", "style", "noscript"]):
        script.decompose()

    text = soup.get_text(separator=' ', strip=True)
    
    # Optional: Limit output size to avoid flooding the agent
    return text[:2000] + "..." if len(text) > 2000 else text


#------------------------Agent----------------------------------------

def main():
    agent = Agent(
        name="news assistant",
        instructions="""
        You are a helpful assistant. 
        When the user provides a URL or when a URL is found in a search result, 
        you should use the `read_website` tool to fetch and read the content of that webpage. 
        Extract meaningful information and present it clearly to the user.
        """,
        tools=[read_website, web_search_tool],
    )

    user_input = input("inter your question : ")

    result = Runner.run_sync(
        agent,
        input=user_input,
        run_config=config
    )

    rich.print(result.final_output)


main()