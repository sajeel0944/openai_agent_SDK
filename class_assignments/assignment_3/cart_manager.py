from dataclasses import dataclass
import os
from dotenv import load_dotenv
from pymongo import MongoClient

#-----------------------------------------------------------------------------------------------------------

load_dotenv()
mongodp =  os.getenv("mongodp")
 
#--------------------------------------------------------------------------------------------------------------

@dataclass
class ShoppingCartItem:
    email : str
    productId : str
    category : str
    description : str
    image : str
    name : str
    price : int
    quantity : int
    type : str

#--------------------------------------------------------------------------------------------------------------

    def is_existing_user(self) -> bool: # is main check kar raha ho ky user ka data pahaly sy hai
        """Check if the user already exists in the JSON file."""
        try:
            with MongoClient(mongodp, tls=True) as client:
                db = client["nafeesBakery"]
                collection = db["addToCardData"]

                all_data = list(collection.find({})) ## is main sary data aye ga

                for data in all_data:
                    if self.email == data["email"]: ## agar user ka data pahaly sy mongodp main howa to true aye ga wana false
                        return True  
                return False
        except Exception as e:
            return False

#-----------------------------------------------------------------------------------------------------------

    def update_existing_user_cart(self) -> bool: # is main porany waly data ky andar new product add ho raha hai
        """Add a product to the cart of an existing user."""

        new_product : dict = {
            "productId": self.productId,
            "category": self.category,
            "description": self.description,
            "image": self.image,
            "name": self.name,
            "price": int(self.price),
            "quantity": int(self.quantity),
            "type": self.type
        }

        try:
            with MongoClient(mongodp, tls=True) as client:
                db = client["nafeesBakery"]
                collection = db["addToCardData"]

                # is main addToCardProduct ky andar ek new product add ho raha hai
                result = collection.update_one(
                    {"email": self.email},  # Search condition
                    {"$push": {"addToCardProduct": new_product}}  # Update data
                )

                # Check if update was successful
                if result.modified_count == 1:
                    # Verify the product was added check kar raha ho ky product add ho gaya hai product ki id ky zayeye check kar raha ho
                    updated_user = collection.find_one(
                        {"email": self.email, "addToCardProduct.productId": self.productId},
                        projection={"_id": 1}  # Only return _id for efficiency
                    )
                    return updated_user is not None
                
                return False
               
        except Exception as e:
            return False
 
#-----------------------------------------------------------------------------------------------------------------

    def add_new_user_with_cart_item(self) -> bool: # is main new data add ho araha hai
        """Add a new user in json and insert product in their cart. Returns True if added successfully."""
        new_add_to_card : dict = {
                "email": self.email,
                "addToCardProduct": [
                {
                    "productId": self.productId,
                    "category": self.category,
                    "description": self.description,
                    "image": self.image,
                    "name": self.name,
                    "price": int(self.price),
                    "quantity": int(self.quantity),
                    "type": self.type
                }
            ]
        }
        try :
            with MongoClient(mongodp, tls=True) as client:
                db = client["nafeesBakery"]
                collection = db["addToCardData"]
                                
                # Insert document
                insert_doc = collection.insert_one(new_add_to_card) ## is main pala new data add horaha hai 

                # Verify insertion
                if insert_doc.acknowledged:
                    # is main check kar raha ho ky data mongodp main add ho gaya hai ya nhi
                    found_doc = collection.find_one({"_id": insert_doc.inserted_id})

                    if found_doc:
                        return True
                return False
        except Exception as e:
            return False
