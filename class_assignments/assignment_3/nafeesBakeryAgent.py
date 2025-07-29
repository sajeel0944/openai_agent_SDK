import asyncio
from typing import Any
from agents import GuardrailFunctionOutput, InputGuardrailTripwireTriggered, ModelSettings, OpenAIChatCompletionsModel, RunConfig, RunContextWrapper, RunHooks, TResponseInputItem, Tool, Usage, function_tool, input_guardrail, set_tracing_disabled, Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
import litellm
import warnings
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from tools import read_add_to_card_product, read_all_product, add_to_card, order_information, bakery_information


#-------------------------------------------------------------------------------------

set_tracing_disabled(disabled=True)
load_dotenv()

# enable_verbose_stdout_logging()

# 🔕 output main litellm ki kuch warning arahe thi is sy warning nhi aye gi
litellm.disable_aiohttp_transport=True

# 🔕 output main pydantic ki kuch warning arahe thi is sy warning nhi aye gi
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Pydantic serializer warnings"
)


#-----------------------------------------------------------------------------------------------------------------


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gemini/gemini-2.0-flash" 

external_cilent = litellm.AsyncOpenAI(
    api_key = OPENAI_API_KEY,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.5-flash",
    openai_client = external_cilent
)

config = RunConfig(
    model = model,
    model_provider = external_cilent,
    tracing_disabled = True
)

#------------------------------------------------------------------------------------------------------------------

## is main guardrail_agent LLM ka response aye ga 
class BakeryHelpingOutput(BaseModel):
    is_bakery : bool ## agar user ny bakery ka question pouch to is main True aye ga agar user ny koye or question pouch to is main False aye ga
    response : str ## is main LLM ka respone aye ga

## guardrail agent hai ye check karry ga ky user ny kon sa question poush ha or us ko bakeryHelpingOutput class main bhejay  ga 
guardrail_agent = Agent(
    name = "Bakery Guardrail Agent",
    instructions="""
        You are a guardrail agent. Your job is to check if the user’s question is related to **Nafees Bakery**.

        - If the user is asking about products, shop information, location, timing, contact details, or any services or **complaints** related to Nafees Bakery, 
        then this is considered a valid Nafees Bakery-related query.

        - If the question is unrelated (e.g., about weather, news, or external topics), it's considered **not** bakery-related.

        Return the result using the `BakeryHelpingOutput` format.

        Examples of valid queries:
        - If the user says "Is there a shop in Agra?" ✅
        - If the user says "I want to order this product or item" ✅
        - If the user says "I want to transfer card to card" ✅
        - If the user says "Give me some information" ✅
        - If the user says "How many orders have I placed?" ✅
        - If the user says "What was my last order?" ✅
        - If the user says "Place this order for me" ✅
        - If the user says "Who are you?" ✅
        - If the user just says "Hello" ✅
        - If the user says "I have a complaint" ✅
        - If the user reports a problem with a product or service ✅

        Examples of invalid queries:
        - If the user says "2+2?" ❌
        - If the user says "who is the founder of pakistan" ❌
        - If the user says "please solve this question " ❌
        - If the user just says "what is ai" ❌
    """
,
    output_type = BakeryHelpingOutput,
    model=LitellmModel(api_key=OPENAI_API_KEY, model=MODEL)
)

## is ky andar sy main agent ko response jaye ga ky answer dyna hai ya nhi 
@input_guardrail
async def bakery_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,    ## guardrail_agent ka response yahan aayega (CodingHelpingOutput format mein)
        ## Agar is_bakery False hai to tripwire trigger hoga, warna nahi
        tripwire_triggered = not result.final_output.is_bakery , ## is main guardrail_agent ny jo CodingHelpingOutput main is_bakery main jo dia hai wo is main aye ga or agar user ny coding ka question pouch ko is main True aye ga to wo False hojaye ga or agar user ny koye or question pouch ko is main False aye ga ko us True ajaye ga
    )

#-------------------------------------------------------------------------------------------------------------------

async def main_assistant(messages: list[dict]):
    try:
        agent = Agent(
            name="Nafees Bakery Assistant",
            instructions="""
            # 🍰 Nafees Bakery Assistant — Tool Guidelines

            You control **two function‑tools**:

            | tool name        | purpose                                   |
            |------------------|-------------------------------------------|
            | **read_all_product** | Fetch *every* product from the Sanity CMS and return a list. |
            | **add_to_card**      | Add a specific product to a user’s cart. |

            ---

            ## ▶️ `read_all_product`
            Use this tool when the user asks for:

            * “Show me all products”  
            * “List everything you have”  
            * “What items are available?”

            **Return** Display all available product information clearly and in a well-structured format for the user. 
                    Ensure that any product details received are presented accurately, completely, and in an easily 
                    readable manner..

            ---

            ## ▶️ `add_to_card`
            Call this tool when the user wants to put something in their cart.

            ### Required arguments

            | argument            | Type | Description                                                      |
            |---------------------|------|------------------------------------------------------------------|
            | `email`             | str  | Customer’s e‑mail address                                        |
            | `productQuantity`   | int  | Number of units to add                                           |
            | `productName`       | str  | Human‑readable product name                                      |
            | `productPrice`      | int  | Unit price (used for validation)                                 |
            | `productDescription`| str  | Product description (used for validation)                        |

            **Example queries → tool calls**

            | User says … | You run … |
            |-------------|-----------|
            | “Add product **abc123** to my cart” | `add_to_card( email=<ask>, productQuantity=1, productName=<lookup>, productPrice=<lookup>, productDescription=<lookup>)` |
            | “Put 3 brownies  for **john@x.com**” | `add_to_card( email="john@x.com", productQuantity=3, productName="Brownie", productPrice=399, productDescription="Chocolate brownie")` |

            If the user hasn’t provided email or quantity, politely ask for the missing
            information *before* calling the tool.

            ## Note:

            When the user says to add Blue Band or any other product to the cart, you should call the ___read_all_product___ tool first when you cell this add_to_card.

            ---

            ---
            ## ▶️ `order_information`

            Use this tool when the customer wants to **review past orders**.

            ### Required argument

            | argument | Type | Description                |
            |----------|------|----------------------------|
            | `email`  | str  | Customer’s e‑mail (provided automatically) |

            Example triggers:

            * “Show me my previous orders”
            * “What did I buy last time?”
            * “Display my purchase history”

            ***Return*** Display all available order information clearly and in a well-structured format for the user. 
                    Ensure that any product details received are presented accurately, completely, and in an easily 
                    readable manner..
            ---

            ---

            ## ▶️ `bakery_information`

            Call this tool when the user wants to know about the bakery.

            ### Required argument
            | argument | Type |
            |----------|------|
            | `prompt`  | str |


            **Triggers:**
            * “Tell me about this bakery”
            * “Who owns it?”
            * “Where is it located?”
            * “When is it open?”
            * which is the best item in this website

            **Return** a short summary with bakery name, address, popular items, owner email, phone number, hours, and location map.

            ---

            ---

            ## ▶️ `read_add_to_card_product`

             This function retrieves the products added to cart by a specific user.

             ### Required argument
            | argument | Type |
            |----------|------|
            | `email`  | str |

            ---


            ## 🌐 Language
            * Detect the user’s language and reply in the same language whenever possible.

            ## ✨ Response style
            * Be concise, friendly, and bakery‑themed (“Sure thing! 🍪”).
            * After a successful tool call, confirm with a short human‑readable message
            (e.g. “Got it! 2 Brownies have been added to your cart.”). Do **not** reveal
            raw JSON or internal IDs unless the user explicitly requests them.

            ## Note
            You are provided with the user's email in every request. Do not ask the user to enter or confirm
            their email under any circumstances. If the email is required for processing, use the value you already 
            have. Avoid displaying or referencing the email to the user directly. and you provide amount in PKR
            """,
            tools=[add_to_card, read_all_product, order_information, bakery_information, read_add_to_card_product],
            model_settings=ModelSettings(tool_choice="required"), # is sy tool zarol use karry ga
            input_guardrails=[bakery_guardrail], ## Yeh guardrail use karega jo CodingHelpingOutput return karta hai
            model=model
        )
 
        response = await Runner.run(agent, input=messages)
        return response.final_output
    except InputGuardrailTripwireTriggered as e:
        return "Sorry, I can answer only questions related to Nafees Bakery."
    except Exception as e:
        return "Sorry, I'm currently unavailable. Please try again in a few moments."

