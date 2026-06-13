import os
import requests

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.tools import tool

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GEOCODING_API_URL = os.getenv("GEOCODING_API_URL")
WEATHER_API_URL = os.getenv("WEATHER_API_URL")
POLLUTION_API_URL = os.getenv("POLLUTION_API_URL")


@tool
def get_coordinates(location: str) -> dict:
    """
    Get latitude and longitude for a city.
    """

    response = requests.get(
        GEOCODING_API_URL,
        params={
            "q": location,
            "limit": 1,
            "appid": WEATHER_API_KEY,
        },
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        return {
            "error": f"Location '{location}' not found."
        }

    return {
        "city": data[0]["name"],
        "country": data[0]["country"],
        "lat": data[0]["lat"],
        "lon": data[0]["lon"],
    }


@tool
def get_weather(location: str) -> dict:
    """
    Get weather information for a city.
    """

    geo = get_coordinates.invoke({"location": location})

    if "error" in geo:
        return geo

    response = requests.get(
        WEATHER_API_URL,
        params={
            "lat": geo["lat"],
            "lon": geo["lon"],
            "appid": WEATHER_API_KEY,
            "units": "metric",

        },
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    return {
        "city": geo["city"],
        "country": geo["country"],
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity_percent": data["main"]["humidity"],
        "pressure_hpa": data["main"]["pressure"],
        "condition": data["weather"][0]["description"],
        "wind_speed_mps": data["wind"]["speed"],
    }


@tool
def get_air_quality(location: str) -> dict:
    """
    Get air quality information for a city.
    """

    geo = get_coordinates.invoke({"location": location})

    if "error" in geo:
        return geo

    response = requests.get(
        POLLUTION_API_URL,
        params={
            "lat": geo["lat"],
            "lon": geo["lon"],
            "appid": WEATHER_API_KEY,
        },
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("list"):
        return {
            "error": "Air quality data unavailable."
        }

    pollution = data["list"][0]

    return {
        "city": geo["city"],
        "country": geo["country"],
        "aqi": pollution["main"]["aqi"],
        "co": pollution["components"]["co"],
        "no": pollution["components"]["no"],
        "no2": pollution["components"]["no2"],
        "o3": pollution["components"]["o3"],
        "so2": pollution["components"]["so2"],
        "pm2_5": pollution["components"]["pm2_5"],
        "pm10": pollution["components"]["pm10"],
        "nh3": pollution["components"]["nh3"],
    }


model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0,
)

agent = create_agent(
    model=model,
    tools=[
        get_coordinates,
        get_weather,
        get_air_quality,
    ],
    system_prompt="""
    You are a weather and air-quality assistant.

    Always use tools when weather or pollution data is requested.

    Rules:
    - If user asks temperature, use get_weather.
    - If user asks humidity, use get_weather.
    - If user asks AQI or pollution, use get_air_quality.
    - If user asks for complete weather report, use get_weather.
    - If user asks for both weather and pollution, call both tools.
    - Present information in a clean human-readable format.
    - Never make up weather data.
    """,
)

while True:
    query = input("\nYou: ")

    if query.lower() in {"exit", "quit"}:
        break

    try:
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

        print("\nAssistant:"),
        print(response["messages"][-1].content),

    except Exception as e:
        print(f"\nError: {e}")