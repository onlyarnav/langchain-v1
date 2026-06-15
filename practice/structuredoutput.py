from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_methods import TypeDict
from langchain_ollama import ChatOllama

load_dotenv()

class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The release year of the movie")
    director: str = Field(description="The director of the movie")
    genre: str = Field(description="The genre of the movie")

model = ChatOllama(
    model="minimax-m3:cloud",
    system_prompt="You are a helpful assistant that provides details of movie in a structured format."
)

class MovieType(TypeDict):
    title: str
    year: int
    director: str
    genre: str

model_structure_basemodel = model.with_structured_output(Movie)
model_structure_typedict = model.with_structured_output(MovieType)

resp1 = model_structure_basemodel.invoke("tell me about movie Inception")
resp2 = model_structure_typedict.invoke("tell me about movie Inception")
resp1
resp2