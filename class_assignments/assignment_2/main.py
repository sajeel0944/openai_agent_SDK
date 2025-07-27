from agents import  Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool, set_tracing_disabled
from agents.run import RunConfig
from dotenv import load_dotenv
import os
 
#----------------------------------------------------------------

load_dotenv()
set_tracing_disabled(disabled=True)

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
    tracing_disabled = True
)

# ----------------------------------------------------------------------------------------------------------------

@function_tool
def get_crypto_price(coin: str) -> str:
    """
    Returns the current price of a given cryptocurrency.
    Coin should be like 'bitcoin', 'ethereum', etc.
    """
    
    prices = {
        "bitcoin": "Bitcoin is trading at $65,000",
        "ethereum": "Ethereum is trading at $3,200",
        "dogecoin": "Dogecoin is trading at $0.12"
    }
    return prices.get(coin.lower(), "Sorry, I couldn't find the price for that cryptocurrency.")


crypto_currency_agent = Agent(
    name="Crypto Currency Agent",
    instructions="""
        # You are a cryptocurrency expert.
        # You handle questions about Bitcoin, Ethereum, blockchain, wallets, mining, NFTs, and crypto safety.
        # If the user asks for the price of a coin, use the get_crypto_price function.
        # Always be helpful and avoid giving financial or investment advice.
    """,
    tools=[get_crypto_price],
    model=model
)

result = Runner.run_sync(crypto_currency_agent, "Ethereum ki price kya hai?")
print(result.final_output)
