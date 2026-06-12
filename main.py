from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama(
    model="gemma4:31b-cloud"
)

response = llm.invoke("Hello!")

print(response.content)