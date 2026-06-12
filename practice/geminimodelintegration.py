from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a helpful AI assistant."
)

while True:
    query = input("\nYou: ")

    if query.lower() in {"exit", "quit"}:
        break

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )

    print("\nAssistant:", response["messages"][-1].content)