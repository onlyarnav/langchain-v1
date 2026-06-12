from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0,
)

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a helpful assistant."
)

while True:
    query = input("\nYou: ")

    if query.lower() in {"quit", "exit"}:
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

    print(
        "\nAssistant:",
        response["messages"][-1].content,
    )