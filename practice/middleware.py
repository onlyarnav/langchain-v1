from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

### Summarization Middleware Example

@tool
def search_hotels(city: str) -> str:
    """Search for hotels in a given city."""

    return f"""
Hotels in {city}:

1. Hotel A - 4 stars
2. Hotel B - 3 stars
3. Hotel C - 5 stars
"""


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

# agent = create_agent(
#     model=model,
#     tools=[search_hotels],
#     checkpointer=InMemorySaver(),
#     middleware=[
#         SummarizationMiddleware(
#             model=model,
#             system_message=SystemMessage(
#                 content="You are a helpful assistant that finds hotels."
#             ),
#             trigger=("tokens", 550),
#             keep=("tokens", 200),
#         )
#     ],
# )

# config = {
#     "configurable": {
#         "thread_id": "test-1"
#     }
# }


# def count_tokens(messages):
#     total_words = 0

#     for msg in messages:
#         content = getattr(msg, "content", "")

#         if isinstance(content, str):
#             total_words += len(content.split())

#     return total_words // 4


# cities = [
#     "New York",
#     "Paris",
#     "Tokyo",
#     "Delhi",
#     "Sydney",
# ]

# for city in cities:

#     response = agent.invoke(
#         {
#             "messages": [
#                 HumanMessage(
#                     content=f"Find hotels in {city}"
#                 )
#             ]
#         },
#         config=config,
#     )

#     token_count = count_tokens(
#         response["messages"]
#     )

#     print(
#         f"\nResponse for {city}:\n"
#         f"{response['messages'][-1].content}\n"
#     )

#     print(
#         f"Approx Tokens in Conversation: "
#         f"{token_count}\n"
#     )

### Human in the Loop Middleware Example

@tool
def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}"

agent = create_agent(
    model = model,
    tools = [read_email_tool, send_email_tool],
    checkpointer = InMemorySaver(),
    middleware = {
        HumanInTheLoopMiddleware(
            interrupt_on = {
                "send_email_tool":{
                    "allowed_decision":["approve", "edit", "reject"]
                },
                "read_email_tool":False,
            }
        )
        }
)

config = {
    "configurable": {
        "thread_id": "test-approved"
    }
}

result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"Send email to arnav@example.con with subjetct 'Hello' & body 'How are you'"
                )
            ]
        },
        config=config,
    )