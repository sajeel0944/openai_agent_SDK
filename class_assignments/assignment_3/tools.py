import asyncio
from agents import function_tool
import os
import requests
from cart_manager import ShoppingCartItem
from pymongo import MongoClient
from agents import function_tool, set_tracing_disabled, Runner, Agent
from agents.extensions.models.litellm_model import LitellmModel
import litellm
import warnings
from dotenv import load_dotenv


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
SANITY_API_KEY = os.getenv("SANITY_API_KEY")

#------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------

# is main jo sanity main product hai waha sy sary product arahy hai
@function_tool
def read_all_product() -> list[dict] | str:
    """
    ## 📦 Function `read_all_product`
    
    Fetch **every product** from the Sanity API and return a simplified Python
    list.  
    Each product in the list contains the fields **`_id`**, **`description`**,
    **`name`**, **`price`**, and **`type`**.

    ### How it works
    1. Sends a`GET` request to the Sanity endpoint  
    2. Expects a JSON response with the key `"result"`.
    3. Iterates over every item in `result`, maps each to a dictionary with the
       required keys, and appends it to `all_product_data`.
    4. Returns the final **list of products**.

    ### Notes
    - Uses `requests.get()` and raises for non 200 status codes.  
    - Change the query string if you need more fields from Sanity.  
    - This helper performs no caching; call responsibly.
    - Display all available product information clearly and in a well-structured format for the user. 
    Ensure that any product details received are presented accurately, completely, and in an easily 
    readable manner.
    """
    try:
        response = requests.get(f"{SANITY_API_KEY}") # is main sanity ky product ko fetch kar raha ho
        response.raise_for_status()
        
        convert_json = response.json() # sanity ky dara ko json main kar raha ho

        product_data = convert_json["result"] # result ky andar he sary product hai
        all_product_data : list[dict] = [] # is main product aye gy sanity ky

        for data in product_data:
            # is main sanity ky jo product hai un main sy jo main main chizy hai wo ni kal araha ho
            data_schema : dict = {
                "_id": data["_id"],
                "description": data["description"],
                "name": data["name"],
                "price": data["price"],
                "type": data["type"]
            }
            all_product_data.append(data_schema)

        return(all_product_data)
    except requests.exceptions.RequestException as e:
        return(f"API request failed: {e}")
    except Exception as e:
            return("Something was wrong")


#---------------------------------------------------------------------------------------------------------------

# is main product add to card hoye gy llm is main email, name, prict, discribtion or quality llm dyga
@function_tool
def add_to_card(email: str, productQuantity: int, productName: str, productPrice: int, productDescription: str) -> str:
    """
    ## 🛒 Function `add_to_card`

   Add the specified product to the customer’s shopping‑cart record.

    ---
    ### Parameters
    | Name                | Type | Description                                                      |
    |---------------------|------|------------------------------------------------------------------|
    | `email`             | str  | Customer’s e‑mail address                                        |
    | `productQuantity`   | int  | Number of units to add                                           |
    | `productName`       | str  | Human‑readable product name                                      |
    | `productPrice`      | int  | Unit price (used for validation)                                 |
    | `productDescription`| str  | Product description (used for validation)                        |

    ---
    ### Step‑by‑Step Workflow
    1. **Call `read_all_product` tool**  
      → This retrieves all available products from the database (via Sanity).  
   This step ensures the product being added is valid and exists.
    2. **Query Sanity** for all documents whose `_type == "product"`.  
       Endpoint:<br>
       `GET /v2025-06-22/data/query/production?query=*[_type=="product"]&perspective=drafts`
    3. **Locate the product** whose  
       `(name == productName) AND (description == productDescription) AND (price == productPrice)`.
    4. Build a `ShoppingCartItem` object using the matched product and the user’s e‑mail.
    5. Determine whether the user already has a cart record:
       * **Existing user** → `update_existing_user_cart()`  
         (push the new item).
       * **New user** → `add_new_user_with_cart_item()`  
         (create cart + insert item).
    6. Return a JSON snippet confirming success or an error message.

    ---
    ### Returns
    * **Success** → e.g. `"Product added to existing user's cart successfully"`  
    * **Failure** → network error (`requests.exceptions.RequestException`) or any unexpected
      exception returns a descriptive string.

    ---
    ### Notes
    * A single Sanity query is used for simplicity; consider a parameterised “fetch‑by‑id” query for efficiency.
    * `image` field is stored as the Sanity asset `_ref`.  Convert to CDN URL if needed.
    * Shopping‑cart persistence currently delegates to `ShoppingCartItem`; ensure its
      implementation writes to your database (Mongo, Postgres, etc.) in a production setup.
   
    """
    try:
        response = requests.get(f"{SANITY_API_KEY}") # is main sanity ky product fetch kar raha ho
        response.raise_for_status()
        
        convert_json = response.json() # sanity ky ky data ko json main kar raha ho

        product_data = convert_json["result"] # is ky andar he sary product hai

        # is main jo llm ny price, discribtion or name dia hai us ky zayeye product filter kar raha ho 
        filter_product = list(filter(lambda x: 
            x["name"].strip().lower() == productName.strip().lower() and 
            x["description"].strip().lower() == productDescription.strip().lower() and 
            int(x["price"]) == int(productPrice), 
            product_data))       
        
        get_product = filter_product[0] 
        
        if get_product:
            # is ky zayeye he add to card main data ja raha hai
            card_item : ShoppingCartItem = ShoppingCartItem(email=email, productId=get_product["_id"], category=get_product["category"], description=get_product["description"], image=get_product["image"]["asset"]["_ref"], name=get_product["name"], price=get_product["price"], quantity=productQuantity, type=get_product["type"])

            check_user_email = card_item.is_existing_user() # is main check kar raha ho ky user ka data pahaly sy hai ya nhi 

            if check_user_email:  # agar user ka data pahaly sy howa to ye cahly ga                                 
                add_product = card_item.update_existing_user_cart() # is ky zayeye database main data add ho raha hai
                if add_product: # agar data add hogaya to ye chaly ga
                    return {"Product added to existing user's cart successfull and – Want to add more items? and Ready to place your order? 👉 Go to the Order section → Fill out the form → Click Confirm Order ✅"}
                else:  ## agar user ka data pahaly sy database main nhi howa to ye chaly ga   
                    return {"Product could not be added to existing cart"}
            else:                                  
                add_user_and_product = card_item.add_new_user_with_cart_item() # is ky zayeye database main data add ho raha hai
                if add_user_and_product:
                    return {"New user created and product added successfull"}
                else:
                    return {"New user could not be created"}
        else:
            return "No matching product found"

    except requests.exceptions.RequestException as e:
        return(f"API request failed: {e}")
    except Exception as e:
        return("Something was wrong")


#--------------------------------------------------------------------------------------------------------------- 

# llm is main user ki email dyga or is main user ny jo order hai wo sary order is main aye gy
@function_tool
def order_information(email) -> list[dict]:
    """
    ## 📦 Function `order_information`

    Retrieve **all past orders** that belong to a single customer.

    ---
    ### Parameters
    | Name   | Type | Description               |
    |--------|------|---------------------------|
    | `email`| str  | Customer’s e‑mail address |

    ---
    ### How it works (step‑by‑step)

    1. Read the **MongoDB connection string** from the `MONGODP` environment variable  
       (`os.getenv("MONGODP")`).
    2. Connect to MongoDB:  
       *Database*: `nafeesBakery` &nbsp;|&nbsp; *Collection*: `customerOrder`.
    3. Fetch **all documents** (`collection.find({})`).
    4. Loop through each document.  
       *If* `document["email"] == email`, append every item inside
       `document["allOrder"]` to `add_to_card_data`.
    5. Return the aggregated list **if at least one match is found**,
       otherwise return `{"message": "Not Found"}`.

    ### Notes
    * Wraps all DB operations in a `try / except`; on any exception,
      returns `"Something was wrong"`.
    * Consumes the whole collection in memory (`list(collection.find({}))`);
      for large datasets consider using a direct filter:
      ```python
      collection.find({"email": email})
      ```
    * Ensure the environment variable **`MONGODP`** (typo?) actually contains
      your MongoDB URI, e.g.  
      `MONGODP=mongodb+srv://user:pass@cluster0.mongodb.net`
    """
    try:
        mongodp =  os.getenv("mongodp")

        add_to_card_data : list[dict] = [] # main sary order aye dy 

        with MongoClient(mongodp, tls=True) as client:
            db = client["nafeesBakery"]
            collection = db["customerOrder"]

            all_data = list(collection.find({})) ## is main sary data araha hai
            
            for data in all_data: ## is main email filter kar raha ho
                if email == data["email"]:
                    for addToCardData in data["allOrder"]:
                        add_to_card_data.append(addToCardData)
                    return add_to_card_data
                       
            return {"Not Found"}

    except Exception as e:
        return("Something was wrong")

#---------------------------------------------------------------------------------------------------------------

# is main bakery ki information arahe hai 
@function_tool
async def bakery_information(prompt: str) -> str:
    """
    This function is used to handle user queries related to Nafees Bakery.

    ## Parameters:
    - prompt (str): The user's question or message, typically asking about bakery products,
      location, contact information, opening hours, or other related details.

    ## User Prompt:
    * “Tell me about this bakery”
    * “Who owns it?”
    * “Where is it located?”
    * “When is it open?”
    * which is the best item in this website
    * Hello

    ## Returns:
    - str: A response generated by the Nafees Bakery Assistant based on the user's prompt.
    """
    agent = Agent(
        name="Nafees Bakery Assistant",
        instructions="""
         # 🥐 Nafees Bakery

        **Address**  
        Shop #3, R-561, Sector 15-A/3, Buffer Zone, Karachi  
        📞 0311-2047971 &nbsp;|&nbsp; ✉️ zohaibshamsi442@gmail.com  
        [📍 Google Maps](https://www.google.com/maps?q=24.9611205,67.0686104&z=17)

        | Opening hours | 07 : 00 AM - 01 : 00 AM (daily) |
        |---------------|--------------------------------|

        ## ⭐ Best-selling items
        - Chips Nimco  
        - Fresh & Cake Rusk  
        - Celebration Cakes  
        - Savory Mix  
        - Chilli Chips  
        - Paape  
        - Bread & Bun (Fruit Bun, Maska Bun)

        > **Always fresh — delivered or picked up right out of the oven!**
    ---
    """,
        model=LitellmModel(api_key=OPENAI_API_KEY, model=MODEL)
    )

    response = await Runner.run(agent, prompt)
    return response.final_output


#---------------------------------------------------------------------------------------------------------------------

# is main current user ky add to card email ky zayeye filter hoye ga

@function_tool
def read_add_to_card_product(email: str) -> list[dict]:
    """
    This function retrieves the products added to cart by a specific user. read_add_to_card_product

    ## Parameters:

    ----------

    email : str  
        The email of the user whose cart products need to be fetched.

    ## Returns:

    -------
    list[dict] or dict:
        Returns a list of products added to the cart by the user.
        If the user is not found, returns a message: {"message": "Not Found"}.
        In case of an error, returns the error message.
    """
    mongodp =  os.getenv("mongodp")

    try:
        with MongoClient(mongodp, tls=True) as client:
            db = client["nafeesBakery"]
            collection = db["addToCardData"]
 
            all_data = list(collection.find({})) ## is main sary data araha hai
            
            for data in all_data: ## is main email filter kar raha ho
                if email == data["email"]:
                    return(data["addToCardProduct"])
            
            return {"message": "Not Found"}

    except Exception as e :
        return {"message" : f"error {e}"}
    
