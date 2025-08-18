from dataclasses import dataclass
import os
from agents import  ModelSettings, Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool, set_tracing_disabled, RunContextWrapper
from agents.run import RunConfig
from openai import BaseModel
from dotenv import load_dotenv

#----------------------------------------------------------------
load_dotenv()
set_tracing_disabled(disabled=True)

#----------------------------------------------------------------

OPENAI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL : str = "gemini-2.5-flash"

#----------------------------------------------------------------

external_client = AsyncOpenAI(
    api_key = OPENAI_API_KEY,
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

class BankAccount(BaseModel):
    account_number : str
    customer_name :str
    account_balance :  float
    account_type : str

class StudentProfile(BaseModel):
    student_id: str
    student_name: str
    current_semester: int
    total_courses: int

class LibraryBook(BaseModel):
    book_id: str
    book_title: str
    author_name: str
    is_available: bool

class UserAllDeatil(BaseModel):
    bank_account: BankAccount
    student_profile: StudentProfile
    library_book: LibraryBook


@function_tool
def user_all_deatils(ctx: RunContextWrapper[UserAllDeatil]):
    """Read bank, student, and library book details together."""
   
    return (
        f"account_number : {ctx.context.bank_account.account_number}, account_balance : {ctx.context.bank_account.account_balance}, account_type : {ctx.context.bank_account.account_type}, customer_name : {ctx.context.bank_account.customer_name}"
        f"student_id : {ctx.context.student_profile.student_id}, student_name : {ctx.context.student_profile.student_name}, current_semester : {ctx.context.student_profile.current_semester}, total_courses : {ctx.context.student_profile.total_courses}"
        f"book_id : {ctx.context.library_book.book_id}, book_title : {ctx.context.library_book.book_title}, author_name : {ctx.context.library_book.author_name}, is_available : {ctx.context.library_book.is_available}"
    )

bank_account : BankAccount = BankAccount(
    account_number="ACC-789456",
    customer_name="Fatima Khan",
    account_balance=75500.50,
    account_type="savings"
)

student = StudentProfile(
    student_id="STU-456",
    student_name="Hassan Ahmed",
    current_semester=4,
    total_courses=5
)

library_book = LibraryBook(
    book_id="BOOK-123",
    book_title="Python Programming",
    author_name="John Smith",
    is_available=True
)

user_all_deatil = UserAllDeatil(
    bank_account=bank_account,
    student_profile=student,
    library_book=library_book
)

agent = Agent(
    name="Bank Assistant",
    instructions="You are a helpful assistant. Use user_all_deatils tool to show user info.",
    tools=[user_all_deatils],
    model=model,
    model_settings=ModelSettings(tool_choice="required")
)

response = Runner.run_sync(agent, "please give me my detail", context=user_all_deatil)

print(response.final_output)