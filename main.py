from dotenv import load_dotenv

## Ollama Model Integration
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama(
    model="gemma4:31b-cloud"
)

response = llm.invoke("Hello!")

# print(response.content)

## Google Gemini Model Integration
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

response = llm.invoke("Explain AI agents in 2 lines.")

print(response.content)